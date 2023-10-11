//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#include "elem_color.h"

#include <Epetra_MapColoring.h>
#include <Epetra_Util.h>

#include <Tpetra_ComputeGatherMap.hpp>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ArrayRCPDecl.hpp>
#include <Teuchos_VerboseObject.hpp>

#include <Zoltan2_TpetraRowGraphAdapter.hpp>
#include <Zoltan2_ColoringProblem.hpp>

//#include <MatrixMarket_Tpetra.hpp>

std::string getmypidstring(const int mypid, const int numproc);

elem_color::elem_color(const Teuchos::RCP<const Epetra_Comm>& comm, 
		       Mesh *mesh,
		       bool dorestart):  
  comm_(comm),
  mesh_(mesh)
{
  //ts_time_create= Teuchos::TimeMonitor::getNewTimer("Total Elem Create Color Time");
  //ts_time_elemadj= Teuchos::TimeMonitor::getNewTimer("Total Elem Adj Fill Time");
  ts_time_color= Teuchos::TimeMonitor::getNewTimer("Tusas: Total Elem Color Time");
  Teuchos::TimeMonitor ElemcolTimer(*ts_time_color);

  //cn to revert to old functionality uncomment:

  dorestart = false;


  if(dorestart){
    restart();
  } else {
    //mesh_->compute_nodal_patch_overlap();
    //compute_graph();
    create_colorer();
    init_mesh_data();
  }
}

elem_color::~elem_color()
{
  //delete mesh_;
}

void elem_color::compute_graph()
{

  auto comm = Teuchos::DefaultComm<int>::getComm(); 

  int mypid = comm_->MyPID();
  if( 0 == mypid )
    std::cout<<std::endl<<"elem_color::compute_graph() started."<<std::endl<<std::endl;

  //std::cout<<mypid<<" "<<mesh_->get_num_elem()<<" "<<mesh_->get_num_elem_in_blk(0)<<std::endl;
  using Teuchos::rcp;

  std::vector<Mesh::mesh_lint_t> elem_num_map(*(mesh_->get_elem_num_map()));

  //map_->Print(std::cout);
  const global_size_t numGlobalEntries = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  const global_ordinal_type indexBase = 0;
  std::vector<global_ordinal_type> elem_num_map1(elem_num_map.begin(),elem_num_map.end());
  Teuchos::ArrayView<global_ordinal_type> AV(elem_num_map1);
  elem_map_ = rcp(new map_type(numGlobalEntries,
 				 AV,
 				 indexBase,
 				 comm));
  map_ = rcp(new Epetra_Map(-1,
 				 elem_num_map.size(),
 				 &elem_num_map[0],
 				 0,
 				 *comm_));
  //elem_map_ ->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );

  #ifdef ELEM_COLOR_USE_ZOLTAN
  //size_t ni = 27;//hex now; dont know what this would be for tri/tet
  size_t ni = 81;//hex now; dont know what this would be for tri/tet
  elem_graph_ = Teuchos::rcp(new crs_graph_type(elem_map_, ni));
  #else
  graph_ = rcp(new Epetra_CrsGraph(Copy, *map_, 0));
  #endif

  if( 0 == mypid )
    std::cout<<std::endl<<"Mesh::compute_elem_adj() started."<<std::endl<<std::endl;

  mesh_->compute_elem_adj();

  if( 0 == mypid )
    std::cout<<std::endl<<"Mesh::compute_elem_adj() ended."<<std::endl<<std::endl;

  for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){
    
    for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {
      Mesh::mesh_lint_t row = mesh_->get_global_elem_id(ne);
      std::vector<Mesh::mesh_lint_t> col = mesh_->get_elem_connect(ne);//this is appearently global id, not local

#ifdef ELEM_COLOR_USE_ZOLTAN
      std::vector<global_ordinal_type> col1(col.begin(),col.end());
      const Teuchos::ArrayView<global_ordinal_type> CV(col1);
      //for(int k =0;k<col1.size(); k++)std::cout<<ne<<" "<<CV[k]<<std::endl;
      const global_ordinal_type row1 = row;
      elem_graph_->insertGlobalIndices(row1, CV);
#else
      graph_->InsertGlobalIndices(row, (int)(col.size()), &col[0]);
#endif
    }//ne
  }//blk
  //graph_->Print(std::cout);
  //elem_graph_->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );

  //#define COLOR_USE_OFFPROC
#ifdef ELEM_COLOR_USE_OFFPROC 
  insert_off_proc_elems();
#endif

#ifdef ELEM_COLOR_USE_ZOLTAN
  elem_graph_->fillComplete();
  //describe outputs -1 for most column locations; even though insertion appears correct and the coloring
  //is ultimately correct. Similar output for W_graph in the preconditioner.
  //dumping via matrix market produces the right data.

  //elem_graph_->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
//   const std::string graphName="";
//   const std::string graphDescription="";
//   const std::string fname="g.dat";
//   Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<> >::writeSparseGraphFile(fname, *elem_graph_, graphName, graphDescription, false);
//   exit(0);
#else
  //if (graph_->GlobalAssemble() != 0){
  if (graph_->FillComplete() != 0){
    std::cout<<"error graph_->GlobalAssemble()"<<std::endl;
    exit(0);
  } 
  //graph_->Print(std::cout);
#endif
  //exit(0);
  if( 0 == mypid )
    std::cout<<std::endl<<"elem_color::compute_graph() ended."<<std::endl<<std::endl;
}

void elem_color::create_colorer()
{
  using Teuchos::rcp;

  compute_graph();

#ifdef ELEM_COLOR_USE_ZOLTAN
  auto comm = Teuchos::DefaultComm<int>::getComm(); 
  int mypid = comm->getRank();
#else
  int mypid = comm_->MyPID();
#endif

  std::cout<<std::endl<<"elem_color::create_colorer() started on proc "<<mypid<<std::endl<<std::endl;

#ifdef ELEM_COLOR_USE_ZOLTAN

  typedef Tpetra::RowGraph<local_ordinal_type, global_ordinal_type, node_type> row_graph_type;
  typedef Zoltan2::TpetraRowGraphAdapter<row_graph_type> graphAdapter_type;

  Teuchos::RCP<row_graph_type> RowGraph =
    Teuchos::rcp_dynamic_cast<row_graph_type>(elem_graph_);

//   Teuchos::RCP<const row_graph_type> constRowGraph =
//     Teuchos::rcp_const_cast<const row_graph_type>(RowGraph);

  graphAdapter_type adapter(RowGraph);

  Teuchos::ParameterList params;
  std::string colorMethod("FirstFit");
  params.set("color_choice", colorMethod);

  Zoltan2::ColoringProblem<graphAdapter_type> problem(&adapter, &params, comm);

  problem.solve();

  size_t checkLength;
  int *checkColoring;
  Zoltan2::ColoringSolution<graphAdapter_type> *soln = problem.getSolution();
  checkLength = soln->getColorsSize();
  checkColoring = soln->getColors();
  num_color_ = soln->getNumColors ();

  //std::cout<<"checkLength = "<<checkLength<<"  znumcolor = "<<num_color_<<std::endl<<std::endl;

  elem_LIDS_.resize(num_color_);

  for ( int i = 0; i < (int)checkLength; i++ ){
    const int color = checkColoring[i]-1;
    if ( color < 0 ){
      if( 0 == mypid ){
	std::cout<<std::endl<<"elem_color::create_colorer() error.  color < 0."<<std::endl<<std::endl;
	exit(0);
      }
    }
    const int lid = i;
    //std::cout<<comm->getRank()<<"   "<<lid<<"   "<<color<<std::endl;
    elem_LIDS_[color].push_back(lid);
  }

  //RowGraph->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
  elem_graph_ = Teuchos::null;
  elem_map_ = Teuchos::null;

  //exit(0);

#else

  //cn looks like the default is a distance-2 coloring,
  //   we need a distance-1 coloring



  Teuchos::ParameterList paramList;
  paramList.set("DISTANCE","1","");

  //cn this call is very expensive......it seems that it might be mpi-only and not threaded in any way
  Teuchos::RCP<Isorropia::Epetra::Colorer> elem_colorer_;
  {
    //Teuchos::TimeMonitor ElemcreTimer(*ts_time_create);
    elem_colorer_ = rcp(new Isorropia::Epetra::Colorer(  graph_.getConst(), paramList, true));
  }
  Teuchos::RCP< Epetra_MapColoring > map_coloring_ = rcp( 
			new Epetra_MapColoring(
					       *(elem_colorer_->Isorropia::Epetra::Colorer::generateRowMapColoring())
					       )
			);
  
  //map_coloring_->Print(std::cout);


  num_color_ = elem_colorer_->numColors();

  color_list_.assign(map_coloring_->ListOfColors(),map_coloring_->ListOfColors()+num_color_);

  elem_LIDS_.resize(num_color_);

  //colors seem to begin with 1, which is the defaultcolor?
  //int default_color_=map_coloring_->DefaultColor();

  for(int i = 1; i < num_color_+1; i++){
    //int num_elem = map_coloring_->NumElementsWithColor(color_list_[i]);
    int num_elem = elem_colorer_->numElemsWithColor(i);
    elem_LIDS_[i-1].resize(num_elem);
    elem_colorer_->elemsWithColor(i,
		&elem_LIDS_[i-1][0],
		num_elem ) ;	
    //elem_LIDS_[i].assign(map_coloring_->ColorLIDList(color_list_[i]),map_coloring_->ColorLIDList(color_list_[i])+num_elem);
    //elem_LIDS_[i].assign(map_coloring_->ColorLIDList(i),map_coloring_->ColorLIDList(i)+num_elem);

    //std::cout<<color_list_[i]<<" ("<<map_coloring_->NumElementsWithColor(color_list_[i])<<") "; 
  }

  graph_ = Teuchos::null;
  map_ = Teuchos::null;

#endif

//   int sum = 0;
//   for ( int i = 0; i < num_color_; i++){
//     std::cout<<mypid<<"   "<<i<<"   "<<elem_LIDS_[i].size()<<std::endl;
//     sum = sum + elem_LIDS_[i].size();
//   }
//   std::cout<<mypid<<"   sum =  "<<sum<<std::endl;

  std::cout<<std::endl<<"elem_color::create_colorer() ended on proc "<<mypid<<". With num_color_ = "<<num_color_<<std::endl<<std::endl;

  //exit(0);
}
void elem_color::init_mesh_data()
{
  std::string cstring="color";
  mesh_->add_elem_field(cstring);
  return;
}
void elem_color::update_mesh_data()
{
  int num_elem = mesh_->get_elem_num_map()->size();
  std::vector<double> color(num_elem,0.);
  for(int c = 0; c < num_color_; c++){
    std::vector<int> elem_map = get_color(c);
    int num_elem = elem_map.size();
    for (int ne=0; ne < num_elem; ne++) {// Loop Over # of Finite Elements on Processor 
      const int lid = elem_LIDS_[c][ne];
      color[lid] = c;
    }
    
  }

  std::string cstring="color";
  mesh_->update_elem_data(cstring, &color[0]);
  return;
}
void elem_color::insert_off_proc_elems(){

  //see comments below about create_root_map...
  //right now we need an epetra map in order to facilitate this.
  //we probably need to fix this by implementing our own  create_root_map
  //for tpetra::map



  //note this is very similar to how we can create the patch for error estimator...
  //although we would need to communicate the nodes also...


  auto comm = Teuchos::DefaultComm<int>::getComm(); 

  const int mypid = comm_->MyPID();
  if( 0 == mypid )
    std::cout<<std::endl<<"elem_color::insert_off_proc_elems() started."<<std::endl<<std::endl;


  std::vector<Mesh::mesh_lint_t> node_num_map(mesh_->get_node_num_map());
//   Teuchos::RCP<const Epetra_Map> o_map_= Teuchos::rcp(new Epetra_Map(-1,
// 									   node_num_map.size(),
// 									   &node_num_map[0],
// 									   0,
// 									   *comm_));
  //map_->Print(std::cout);
  //o_map_->Print(std::cout);
  const global_size_t numGlobalEntries = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  const global_ordinal_type indexBase = 0;
  std::vector<global_ordinal_type> node_num_map1(node_num_map.begin(),node_num_map.end());
  Teuchos::ArrayView<global_ordinal_type> AV(node_num_map1);
  Teuchos::RCP<const map_type> overlap_map_ = rcp(new map_type(numGlobalEntries,
							       AV,
							       indexBase,
							       comm));
  //overlap_map_ ->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
  //exit(0);


  const int blk = 0;
  const int n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);
  const int num_elem = mesh_->get_num_elem();

  //Teuchos::RCP<const Epetra_Map> n_map_;

  Teuchos::RCP<const map_type> node_map_;
  if( 1 == comm_->NumProc() ){
    //n_map_ = o_map_;
    node_map_ = overlap_map_;
  }else{
// #ifdef MESH_64
//     n_map_ = Teuchos::rcp(new Epetra_Map(Create_OneToOne_Map64(*o_map_)));
// #else
//     n_map_ = Teuchos::rcp(new Epetra_Map(Epetra_Util::Create_OneToOne_Map(*o_map_)));
// #endif
    GreedyTieBreak<local_ordinal_type,global_ordinal_type> greedy_tie_break;
    node_map_ = Teuchos::rcp(new map_type(*(Tpetra::createOneToOne(overlap_map_,greedy_tie_break))));
  }

  //n_map_->Print(std::cout);

  std::vector<Mesh::mesh_lint_t> shared_nodes;

  //for(int i = 0; i < o_map_->NumMyElements (); i++){
#if (TRILINOS_MAJOR_VERSION < 14) 
  for(int i = 0; i < overlap_map_->getNodeNumElements(); i++){
#else
  for(int i = 0; i < overlap_map_->getLocalNumElements(); i++){
#endif
// #ifdef MESH_64
//     Mesh::mesh_lint_t ogid = o_map_->GID64(i);
// #else
//     Mesh::mesh_lint_t ogid = o_map_->GID(i);
// #endif
    
    const Mesh::mesh_lint_t ogid = overlap_map_->getGlobalElement ((local_ordinal_type) i); //global_ordinal_type 
    //std::cout<<comm_->MyPID()<<" "<<ogid<<" "<<n_map_->LID(ogid)<<std::endl;
    //if(n_map_->LID(ogid) < 0 ) shared_nodes.push_back(ogid);
    if((local_ordinal_type)(node_map_->getLocalElement(ogid)) 
       == Teuchos::OrdinalTraits<Tpetra::Details::DefaultTypes::local_ordinal_type>::invalid() ) shared_nodes.push_back(ogid);
  }
  Teuchos::RCP<const Epetra_Map> s_map_= Teuchos::rcp(new Epetra_Map(-1,
								     shared_nodes.size(),
								     &shared_nodes[0],
								     0,
								     *comm_));
  //s_map_->Print(std::cout);
  std::vector<global_ordinal_type> shared_nodes1(shared_nodes.begin(),shared_nodes.end());
  Teuchos::ArrayView<global_ordinal_type> AV1(shared_nodes1);
  Teuchos::RCP<const map_type> shared_node_map_ = rcp(new map_type(numGlobalEntries,
								   AV1,
								   indexBase,
								   comm));


  //shared_node_map_->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
  //exit(0);

#ifdef MESH_64
  Teuchos::RCP<const Epetra_Map> o_shared_node_map_ = 
    Teuchos::rcp(new Epetra_Map(Create_OneToOne_Map64(*s_map_)));
#else
  Teuchos::RCP<const Epetra_Map> o_shared_node_map_ = 
    Teuchos::rcp(new Epetra_Map(Epetra_Util::Create_OneToOne_Map(*s_map_)));
#endif


  //o_shared_node_map_->Print(std::cout);
#ifdef MESH_64
  Teuchos::RCP<const Epetra_Map> r_shared_node_map_ 
    = Teuchos::rcp(new Epetra_Map(Create_Root_Map64( *o_shared_node_map_, -1))); 
#else	
  Teuchos::RCP<const Epetra_Map> r_shared_node_map_ 
    = Teuchos::rcp(new Epetra_Map(Epetra_Util::Create_Root_Map( *o_shared_node_map_, -1))); 
#endif


 //   GreedyTieBreak<local_ordinal_type,global_ordinal_type> greedy_tie_break;
//   Teuchos::RCP<const map_type> onetoone_shared_node_map_ = Teuchos::rcp(new map_type(*(Tpetra::createOneToOne(shared_node_map_,greedy_tie_break))));
  //would like to create a replicated tpetra::map, not clear how to easily do this....
  Teuchos::RCP<Teuchos::FancyOStream>
    out = Teuchos::VerboseObjectBase::getDefaultOStream();
  Teuchos::RCP<const map_type> rep_shared_node_map_ = 
    Tpetra::Details::computeGatherMap (shared_node_map_,
		      out);
  //Teuchos::rcp(new map_type(*(Tpetra::createLocalMapWithNode<local_ordinal_type, global_ordinal_type, node_type>((size_t)(onetoone_shared_node_map_->getGlobalNumElements()), comm))));

  const global_ordinal_type ng = rep_shared_node_map_->getGlobalNumElements();

  Teuchos::RCP<const map_type> rep_map_ = rcp(new map_type(ng,
							   indexBase,
							   comm,
							   Tpetra::LocallyReplicated));

  Tpetra::Import<local_ordinal_type, global_ordinal_type, node_type> rep_importer_(rep_map_, rep_shared_node_map_);

  Tpetra::Vector<global_ordinal_type, local_ordinal_type,
    global_ordinal_type, node_type> rep_shared_vec_(rep_shared_node_map_);

#if (TRILINOS_MAJOR_VERSION < 14) 
  for (int i=0; i<rep_shared_node_map_->getNodeNumElements(); i++) {
#else
  for (int i=0; i<rep_shared_node_map_->getLocalNumElements(); i++) {
#endif
    rep_shared_vec_.replaceLocalValue(i, (long long) rep_shared_node_map_->getGlobalElement(i));
  }

  Tpetra::Vector<global_ordinal_type, local_ordinal_type,
    global_ordinal_type, node_type> rep_vec_(rep_map_);
  rep_vec_.doImport(rep_shared_vec_,rep_importer_,Tpetra::INSERT);
  rep_vec_.describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );

  Teuchos::ArrayRCP<global_ordinal_type> rv = rep_vec_.get1dViewNonConst();
  std::vector<global_ordinal_type> vals(rep_vec_.getLocalLength());
  for(int i = 0; i < rep_vec_.getLocalLength(); i++){
    vals[i] = (global_ordinal_type)rv[i];
  }

  const Teuchos::ArrayView<global_ordinal_type> AV2(vals);
  Teuchos::RCP<const map_type> replicated_map_ = rcp(new map_type( rep_vec_.getLocalLength(),
 								  AV2,
 							   indexBase,
								  comm));

  r_shared_node_map_->Print(std::cout);
  replicated_map_->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
//exit(0);
#if (TRILINOS_MAJOR_VERSION < 14) 
  std::cout<<r_shared_node_map_->NumMyElements ()<<"   "<<rep_shared_node_map_->getNodeNumElements ()<<std::endl;
#else
  std::cout<<r_shared_node_map_->NumMyElements ()<<"   "<<rep_shared_node_map_->getLocalNumElements ()<<std::endl;
#endif









  for(int i = 0; i < r_shared_node_map_->NumMyElements (); i++){
  //for(int i = 0; i < rep_shared_node_map_->getNodeNumElements (); i++){


#ifdef MESH_64
    const Mesh::mesh_lint_t rsgid = r_shared_node_map_->GID64(i);
#else
    const Mesh::mesh_lint_t rsgid = r_shared_node_map_->GID(i);
#endif
    //const int rsgid = r_shared_node_map_->GID(i);
    //const int ogid = o_map_->LID(rsgid);//local
    const int ogid = overlap_map_->getLocalElement(rsgid);//local
    std::vector<int> mypatch;//local
    if(ogid != -1){
      mypatch = mesh_->get_nodal_patch_overlap(ogid);//get local elem_id
    }
    int p_size = mypatch.size();
    int max_size = 0;
    comm_->MaxAll(&p_size,
		  &max_size,
		  (int)1 );
      
    std::vector<Mesh::mesh_lint_t> gidmypatch(max_size,(Mesh::mesh_lint_t)(-99));
    for(int j = 0; j < p_size; j++){
#ifdef ELEM_COLOR_USE_ZOLTAN
      gidmypatch[j] = elem_map_->getGlobalElement(mypatch[j]);
#else
      gidmypatch[j] = map_->GID(mypatch[j]); 
#endif    
      //std::cout<<" "<<rsgid<<" "<<gidmypatch[j]<<" "<<mypatch[j]<<std::endl;
    }//j
      
    int count = comm_->NumProc()*max_size;
    std::vector<Mesh::mesh_lint_t> AllVals(count,-99);
    
    comm_->GatherAll(&gidmypatch[0],
		     &AllVals[0],
		     max_size );
    
    //cn need to fix Allvals here
    
    std::vector<Mesh::mesh_lint_t> g_gids;
    for(int j = 0; j< count ;j++){
      if(-99 < AllVals[j]) {
	//std::cout<<"   "<<comm_->MyPID()<<" "<<i<<" "<<AllVals[j]<<" "<<rsgid<<std::endl;
	g_gids.push_back(AllVals[j]);
      }
    }//j
    
    for(int j = 0; j < g_gids.size(); j++){
      int elid = map_->LID(g_gids[j]);
      if(elid > -1){
	for(int k = 0;k< g_gids.size(); k++){
#ifdef ELEM_COLOR_USE_ZOLTAN
	  local_ordinal_type eelid = elem_map_->getLocalElement (g_gids[k]);//local_ordinal_type
#else
	  int eelid = map_->LID(g_gids[k]);
#endif
	  //if(eelid > -1){
	  //std::cout<<"   "<<comm_->MyPID()<<" "<<g_gids[j]<<" "<<g_gids[k]<<std::endl;//" "<<rsgid<<" "<<elid<<std::endl;

#ifdef ELEM_COLOR_USE_ZOLTAN
	  global_ordinal_type gk = (global_ordinal_type)g_gids[k];
	  elem_graph_->insertGlobalIndices(g_gids[j], (local_ordinal_type)1, &gk);
#else
	  graph_->InsertGlobalIndices(g_gids[j], (int)1, &g_gids[k]);
#endif
	  //}
	}//k
      }//elid
      
    }//j
    
    
  }//i
  //exit(0);

  //graph_->Print(std::cout);
  //exit(0);


  if( 0 == mypid )
    std::cout<<std::endl<<"elem_color::insert_off_proc_elems() ended."<<std::endl<<std::endl;

  return;

}

void elem_color::restart(){
  //we will first read in a vector of size num elem on this proc
  // fill it with color
  //find the min and max values to determine the number of colors
  //allocate first dimension of elem_LIDS_ to num colors
  //fill each color with elem ids via push back
  //verify that this is by LID

  int mypid = comm_->MyPID();
  int numproc = comm_->NumProc();

  std::cout<<std::endl<<"elem_color::restart() started on proc "<<mypid<<std::endl<<std::endl;

  const int num_elem = mesh_->get_num_elem();

  int ex_id_;

  //this code is replicated in a number of places.  Need to have a function for this somewhere
  if( 1 == numproc ){//cn for now
    //if( 0 == mypid ){
    const char *outfilename = "results.e";
    ex_id_ = mesh_->open_exodus(outfilename,Mesh::READ);

    std::cout<<"  Opening file for restart; ex_id_ = "<<ex_id_<<" filename = "<<outfilename<<std::endl;
    
  }
  else{
    std::string decompPath="decomp/";
    //std::string pfile = decompPath+std::to_string(mypid+1)+"/results.e."+std::to_string(numproc)+"."+std::to_string(mypid);
    
    std::string mypidstring(getmypidstring(mypid,numproc));

    std::string pfile = decompPath+"results.e."+std::to_string(numproc)+"."+mypidstring;
    ex_id_ = mesh_->open_exodus(pfile.c_str(),Mesh::READ);
    
    std::cout<<"  Opening file for restart; ex_id_ = "<<ex_id_<<" filename = "<<pfile<<std::endl;

    //cn we want to check the number of procs listed in the nem file as well    
    int nem_proc = -99;
    int error = mesh_->read_num_proc_nemesis(ex_id_, &nem_proc);
    if( 0 > error ) {
      std::cout<<"Error obtaining restart num procs in file"<<std::endl;
      exit(0);
    }
    if( nem_proc != numproc ){
      std::cout<<"Error restart nem_proc = "<<nem_proc<<" does not equal numproc = "<<numproc<<std::endl;
      exit(0);
    }
  }

  int step = -99;
  int error = mesh_->read_last_step_exodus(ex_id_,step);
  if( 0 == mypid )
    std::cout<<"  Reading restart last step = "<<step<<std::endl;
  if( 0 > error ) {
    std::cout<<"Error obtaining restart last step"<<std::endl;
    exit(0);
  }



  //we read a double from exodus; convert to int below
  std::vector<double> colors(num_elem);

  //note that in the future (and in case of nemesis) there may be other elem data in the file, ie error_est or procid
  //we will need to sort through this
  init_mesh_data();
  std::string cstring="color";
  error = mesh_->read_elem_data_exodus(ex_id_,step,cstring,&colors[0]);
  if( 0 > error ) {
    std::cout<<"Error reading color at step "<<step<<std::endl;
    exit(0);
  }

  mesh_->close_exodus(ex_id_);
  
  int max_color = (int)(*max_element(colors.begin(), colors.end()));
  int min_color = (int)(*min_element(colors.begin(), colors.end()));
  num_color_ = max_color - min_color+1;
  elem_LIDS_.resize(num_color_);

  
  for(int i = 0; i < num_elem; i++){
    const int c = (int)colors[i];
    const int lid = i;
    //std::cout<<mypid<<" : "<<c<<"   "<<lid<<std::endl;
    elem_LIDS_[c].push_back(lid);

  }
  //std::cout<<mypid<<" : "<<*max_element(colors.begin(), colors.end())<<"  "<<num_color_<<std::endl;


  std::cout<<std::endl<<"elem_color::restart() ended on proc "<<mypid<<std::endl<<std::endl;
  //exit(0);
}
