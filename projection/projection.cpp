//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#include "projection.h"

#include <iostream>

projection::projection(const Teuchos::RCP<const Epetra_Comm>& comm) :
  comm_(comm)
{
  using Teuchos::rcp;
  //std::cout<<"projection()"<<std::endl;

  int mypid = comm_->MyPID();
  int numproc = comm_->NumProc();
  sourcemesh_ = new Mesh(mypid,numproc,false);
  std::string meshNameString = "source2d.e";

  if( 1 != numproc ) exit(0);

  sourcemesh_->read_exodus((meshNameString ).c_str());
  sourcemesh_->compute_nodal_adj();

  //cn calling create_sorted_nodelist()
  //sourcemesh_->create_sorted_nodelist();
  //cn creates get_sorted_node_num_map with nodes order by increasing x,y,z
  //cn so we should be able to read Matt's files directly
  //cn since they are in that order

  std::vector<int> node_num_map(sourcemesh_->get_node_num_map());

  //std::vector<int> node_num_map(sourcemesh_->get_sorted_node_num_map());

  source_map_ = rcp(new Epetra_Map(-1,
				      node_num_map.size(),
				      &node_num_map[0],
				      0,
				      *comm_));


  //source_map_->Print(std::cout);
  source_ = Teuchos::rcp(new Epetra_Vector(*source_map_));
  //source_->Print(std::cout);
  read_file();
  update_mesh_data();
  exit(0);
}

projection::~projection()
{
  delete sourcemesh_;
}

void projection::read_file()
{
  std::vector<double> data;
  std::ifstream ifile("test.txt", std::ios::in);
  double num = 0.0;
    //keep storing values from the text file so long as data exists:
  while (ifile >> num) {
    data.push_back(num);
  }
  sourcemesh_->create_sorted_nodelist();
  exit(0);
  std::vector<int> node_num_map(sourcemesh_->get_sorted_node_num_map());

  //std::cout<<data.size()<<" "<<sourcemesh_->get_num_nodes()<<std::endl;

  //if(sourcemesh_->get_num_nodes() != data.size()) exit(0);
  for( int i = 0; i < data.size(); i++){
    std::cout<<i<<" "<<data[i]<<" "<<source_map_->GID(i)<<" "<<node_num_map[i]<<std::endl;
    //(*source_)[node_num_map[i]] = data[i];
    double val = data[i];
    int gid = node_num_map[i];
    //val=sourcemesh_->get_x(ind)+sourcemesh_->get_y(ind);
    source_->ReplaceGlobalValues((int)1,
				 &val,
				 &gid 
				 ) ;	
  }
  source_->Print(std::cout);
  return;
}
void projection::update_mesh_data()
{
  int num_nodes = source_map_->NumMyElements ();
  //num_nodes = 100;
  std::vector<double> data(num_nodes,0.);
  for (int nn=0; nn < num_nodes; nn++) {
    data[nn]=(*source_)[nn];
    //data[nn]=sourcemesh_->get_x(nn)+sourcemesh_->get_y(nn);
      //std::cout<<data[nn]<<std::endl;
  }
  std::string string_="source";
  sourcemesh_->add_nodal_field(string_);
  sourcemesh_->update_nodal_data(string_, &data[0]);

  const char *outfilename = "sourcedata.e";
  int ex_id_ = sourcemesh_->create_exodus(outfilename);
  sourcemesh_->write_exodus(ex_id_);
}

