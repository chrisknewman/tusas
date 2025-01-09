//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#include "elem_color.h"

#include <Tpetra_ComputeGatherMap.hpp>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ArrayRCPDecl.hpp>
#include <Teuchos_VerboseObject.hpp>

#include <Zoltan2_TpetraRowGraphAdapter.hpp>
#include <Zoltan2_ColoringProblem.hpp>

//#include <MatrixMarket_Tpetra.hpp>

#include "greedy_tie_break.hpp"

std::string getmypidstring(const int mypid, const int numproc);

elem_color::elem_color(Mesh *mesh,
		       bool dorestart,
		       bool writedata): 
  mesh_(mesh),
  writedata_(writedata)
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

  const int mypid = comm->getRank();
  if( 0 == mypid )
    std::cout<<std::endl<<"elem_color::compute_graph() started."<<std::endl<<std::endl;

  //std::cout<<mypid<<" "<<mesh_->get_num_elem()<<" "<<mesh_->get_num_elem_in_blk(0)<<std::endl;
  using Teuchos::rcp;

  const std::vector<Mesh::mesh_lint_t> elem_num_map(*(mesh_->get_elem_num_map()));

  //map_->Print(std::cout);
  const global_size_t numGlobalEntries = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  const global_ordinal_type indexBase = 0;
  std::vector<global_ordinal_type> elem_num_map1(elem_num_map.begin(),elem_num_map.end());
  Teuchos::ArrayView<global_ordinal_type> AV(elem_num_map1);
  Teuchos::RCP<const map_type> elem_map_ = rcp(new map_type(numGlobalEntries,
 				 AV,
 				 indexBase,
 				 comm));
  
  //elem_map_ ->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );

  const size_t ni = TUSAS_MAX_NODE_PER_ROW_PER_EQN_HEX;//hex now; dont know what this would be for tri/tet
  elem_graph_ = Teuchos::rcp(new crs_graph_type(elem_map_, ni));

  if( 0 == mypid )
    std::cout<<std::endl<<"Mesh::compute_elem_adj() started."<<std::endl<<std::endl;

  //cn 7-30-24 compute_elem_adj seems to be agnostic of blk
  mesh_->compute_elem_adj();

  if( 0 == mypid )
    std::cout<<std::endl<<"Mesh::compute_elem_adj() ended."<<std::endl<<std::endl;
  int c = 0;
  for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){
    
    for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {
      //this is global tusas id
      Mesh::mesh_lint_t row = mesh_->get_global_elem_id(ne+c);

      //cn 7-30-24 elem_connect probably needs a blk also for this to work
      //cn note that get_elem_connect is element-element connectivity
      //cn takes local elem id and returns global elem id
      std::vector<Mesh::mesh_lint_t> col = mesh_->get_elem_connect(ne+c);
      //std::cout<<blk<<" "<<ne+c<<" "<<row<<" "<<col.size()<<std::endl;
      std::vector<global_ordinal_type> col1(col.begin(),col.end());
      const Teuchos::ArrayView<global_ordinal_type> CV(col1);
      const global_ordinal_type row1 = row;
      elem_graph_->insertGlobalIndices(row1, CV);
    }//ne
    c = mesh_->get_num_elem_blks();
  }//blk

  //elem_graph_->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
  //exit(0);

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

  //exit(0);
  if( 0 == mypid )
    std::cout<<std::endl<<"elem_color::compute_graph() ended."<<std::endl<<std::endl;
}

void elem_color::create_colorer()
{
  using Teuchos::rcp;

  compute_graph();

  auto comm = Teuchos::DefaultComm<int>::getComm(); 
  const int mypid = comm->getRank();

  std::cout<<std::endl<<"elem_color::create_colorer() started on proc "<<mypid<<std::endl<<std::endl;

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

  //exit(0);

//   int sum = 0;
//   for ( int i = 0; i < num_color_; i++){
//     std::cout<<mypid<<"   "<<i<<"   "<<elem_LIDS_[i].size()<<std::endl;
//     sum = sum + elem_LIDS_[i].size();
//   }
//   std::cout<<mypid<<"   sum =  "<<sum<<std::endl;

  std::cout<<std::endl<<"elem_color::create_colorer() ended on proc "<<mypid<<". With num_color_ = "<<num_color_<<std::endl<<std::endl;

  //exit(0);
}
void elem_color::init_mesh_data() const
{
  if(!writedata_) return;
  const std::string cstring="color";
  mesh_->add_elem_field(cstring);
  return;
}
void elem_color::update_mesh_data() const
{
  if(!writedata_) return;
  const int num_elem = mesh_->get_elem_num_map()->size();
  std::vector<double> color(num_elem,0.);
  for(int c = 0; c < num_color_; c++){
    std::vector<int> elem_map = get_color(c);
    int num_elem = elem_map.size();
    for (int ne=0; ne < num_elem; ne++) {// Loop Over # of Finite Elements on Processor 
      const int lid = elem_LIDS_[c][ne];
      color[lid] = c;
      //std::cout<<lid<<" "<<c<<std::endl;
    }
    
  }

  const std::string cstring="color";
  mesh_->update_elem_data(cstring, &color[0]);
  return;
}

void elem_color::restart(){
  //we will first read in a vector of size num elem on this proc
  // fill it with color
  //find the min and max values to determine the number of colors
  //allocate first dimension of elem_LIDS_ to num colors
  //fill each color with elem ids via push back
  //verify that this is by LID

  auto comm = Teuchos::DefaultComm<int>::getComm(); 
  const int mypid = comm->getRank();
  const int numproc = comm->getSize();

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
    const std::string decompPath="decomp/";
    //std::string pfile = decompPath+std::to_string(mypid+1)+"/results.e."+std::to_string(numproc)+"."+std::to_string(mypid);
    
    const std::string mypidstring(getmypidstring(mypid,numproc));

    const std::string pfile = decompPath+"results.e."+std::to_string(numproc)+"."+mypidstring;
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
  const std::string cstring="color";
  error = mesh_->read_elem_data_exodus(ex_id_,step,cstring,&colors[0]);
  if( 0 > error ) {
    std::cout<<"Error reading color at step "<<step<<std::endl;
    exit(0);
  }

  mesh_->close_exodus(ex_id_);
  
  const int max_color = (int)(*max_element(colors.begin(), colors.end()));
  const int min_color = (int)(*min_element(colors.begin(), colors.end()));
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
