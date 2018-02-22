//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#include "elem_color.h"

#include <Epetra_MapColoring.h>

elem_color::elem_color(const Teuchos::RCP<const Epetra_Comm>& comm, 
				 Mesh *mesh):  
  comm_(comm),
  mesh_(mesh)
{
  //ts_time_elemadj= Teuchos::TimeMonitor::getNewTimer("Total Elem Adj Fill Time");
  compute_graph();
  create_colorer();
  init_mesh_data();
}

elem_color::~elem_color()
{
  delete mesh_;
}

void elem_color::compute_graph()
{

  //will need to get this working in parallel

  int mypid = comm_->MyPID();
  if( 0 == mypid )
    std::cout<<std::endl<<"elem_color::compute_graph() started."<<std::endl<<std::endl;

  //std::cout<<mypid<<" "<<mesh_->get_num_elem()<<" "<<mesh_->get_num_elem_in_blk(0)<<std::endl;
  using Teuchos::rcp;

  elem_map_ = rcp(new Epetra_Map(-1,
 				 mesh_->get_num_elem(),
 				 &(*(mesh_->get_elem_num_map()))[0],
 				 0,
 				 *comm_));

  //elem_map_->Print(std::cout);

  graph_ = rcp(new Epetra_CrsGraph(Copy, *elem_map_, 0));

  if( 0 == mypid )
    std::cout<<std::endl<<"Mesh::compute_elem_adj() started."<<std::endl<<std::endl;
  {
    //Teuchos::TimeMonitor ElemadjTimer(*ts_time_elemadj); 
  mesh_->compute_elem_adj();
  }
  if( 0 == mypid )
    std::cout<<std::endl<<"Mesh::compute_elem_adj() ended."<<std::endl<<std::endl;

  for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){
    
    //int n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);
    for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {
      int row = mesh_->get_global_elem_id(ne);
      std::vector<int> col = mesh_->get_elem_connect(ne);
      graph_->InsertGlobalIndices(row, (int)(col.size()), &col[0]);

    }
  }
  //if (graph_->GlobalAssemble() != 0){
  if (graph_->FillComplete() != 0){
    std::cout<<"error graph_->GlobalAssemble()"<<std::endl;
    exit(0);
  } 
  //graph_->Print(std::cout);
  //exit(0);
  if( 0 == mypid )
    std::cout<<std::endl<<"elem_color::compute_graph() ended."<<std::endl<<std::endl;
}

void elem_color::create_colorer()
{
  using Teuchos::rcp;
  int mypid = comm_->MyPID();
  if( 0 == mypid )
    std::cout<<std::endl<<"elem_color::create_colorer() started."<<std::endl<<std::endl;

  //cn looks like the default is a distance-2 coloring,
  //   we need a distance-1 coloring

  Teuchos::ParameterList paramList;
  paramList.set("DISTANCE","1","");

  elem_colorer_ = rcp(new Isorropia::Epetra::Colorer(  graph_.getConst(), paramList, true));

  map_coloring_ = rcp( 
		      new Epetra_MapColoring(
					     *(elem_colorer_->Isorropia::Epetra::Colorer::generateRowMapColoring())
					     )
		      );
  //map_coloring_->Print(std::cout);

  num_color_ = map_coloring_->NumColors ();

  color_list_.assign(map_coloring_->ListOfColors(),map_coloring_->ListOfColors()+num_color_);

  elem_LIDS_.resize(num_color_);

  //colors seem to begin with 1, which is the defaultcolor?
  int default_color_=map_coloring_->DefaultColor();

  for(int i = 0; i < num_color_; i++){
    int num_elem = map_coloring_->NumElementsWithColor(color_list_[i]);
    //int num_elem = map_coloring_->NumElementsWithColor(i);

    elem_LIDS_[i].assign(map_coloring_->ColorLIDList(color_list_[i]),map_coloring_->ColorLIDList(color_list_[i])+num_elem);
    //elem_LIDS_[i].assign(map_coloring_->ColorLIDList(i),map_coloring_->ColorLIDList(i)+num_elem);

    //std::cout<<color_list_[i]<<" ("<<map_coloring_->NumElementsWithColor(color_list_[i])<<") "; 
  }
  //std::cout<<std::endl;
  //std::cout<<"num_color_"<<num_color_<<std::endl;


//   for(int i = 0; i < num_color_; i++){
//     int num_elem = elem_LIDS_[i].size();

//     std::cout<<mypid<<" "<<i<<" ("<<color_list_[i]<<") : ";
//     for(int n = 0; n < num_elem; n++){
//       std::cout<<elem_LIDS_[i][n]<<" ";
//     }
//     std::cout<<std::endl;
//   }


  if( 0 == mypid )
    std::cout<<std::endl<<"elem_color::create_colorer() ended."<<std::endl<<std::endl;
  //exit(0);
}
void elem_color::init_mesh_data()
{
  std::string cstring="color";
  mesh_->add_elem_field(cstring);
}
void elem_color::update_mesh_data()
{
  int num_elem = mesh_->get_elem_num_map()->size();
  std::vector<double> color(num_elem,0.);
  for(int c = 0; c < num_color_; c++){
    std::vector<int> elem_map = get_color(c);
    int num_elem = elem_map.size();
    for (int ne=0; ne < num_elem; ne++) {// Loop Over # of Finite Elements on Processor 
      int elem = elem_map[ne];
      color[elem] = c;  
    }
    
  }

  std::string cstring="color";
  mesh_->update_elem_data(cstring, &color[0]);
}

