//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#include "post_process.h"
#include "greedy_tie_break.hpp"

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Tpetra_MultiVector.hpp>

#include <iostream>
#include <fstream>

//template<class Scalar>
post_process::post_process(//const Teuchos::RCP<const Epetra_Comm>& comm, 
			   Mesh *mesh, 
			   const int index,
			   SCALAR_OP s_op,
			   bool restart,
			   const int eqn_id,
			   const std::string basename,
			   const int precision,
               const bool writedata):
  mesh_(mesh),
  index_(index),
  s_op_(s_op),
  restart_(restart),
  eqn_id_(eqn_id),
  basename_(basename),
  precision_(precision),
  writedata_(writedata)
{

  scalar_val_ =  0.;

  const std::vector<Mesh::mesh_lint_t> node_num_map(mesh_->get_node_num_map());

  auto comm_ = Teuchos::DefaultComm<int>::getComm();

  const Tpetra::global_size_t numGlobalEntries = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  const Tpetra::Map<>::global_ordinal_type indexBase = 0;

  std::vector<Tpetra::Map<>::global_ordinal_type> my_global_nodes(node_num_map.size());
  for(int i = 0; i < node_num_map.size(); i++){
    my_global_nodes[i] = node_num_map[i];
  }

  const Teuchos::ArrayView<Tpetra::Map<>::global_ordinal_type> AV(my_global_nodes);
  overlap_map_ = Teuchos::rcp(new map_type(numGlobalEntries,
					     AV,
					     indexBase,
					     comm_));
 
  if( 1 == comm_->getSize() ){
    node_map_ = overlap_map_;
  }else{
    GreedyTieBreak<local_ordinal_type,global_ordinal_type> greedy_tie_break;
    node_map_ = Teuchos::rcp(new map_type(*(Tpetra::createOneToOne(overlap_map_,greedy_tie_break))));
  };
  importer_ = Teuchos::rcp(new import_type(node_map_, overlap_map_));

  ppvar_ = Teuchos::rcp(new vector_type(node_map_));

  const std::string ystring=basename_+std::to_string(index_);
  if(writedata_) mesh_->add_nodal_field(ystring);

  if ( (0 == comm_->getRank()) && (s_op_ != NONE) && (writedata_ == true) ){
    filename_ = ystring+".dat";
    std::ofstream outfile;
    if( restart_ ){
      outfile.open(filename_, std::ios::app );
    }else{
      outfile.open(filename_);
    }
    outfile.close();
  }

  if ( 0 == comm_->getRank())
    std::cout<<"Post process created for variable "<<index_
             <<" with name "<<ystring
             <<". Write to exodus = "<<writedata_<<std::endl<<std::endl;
  //exit(0);
};

post_process::~post_process(){};

void post_process::process(const int i,
			   const double *u, 
			   const double *uold, 
			   const double *uoldold, 
			   const double *gradu, 
			   const double &time, 
			   const double &dt, 
			   const double &dtold)
{
  const global_ordinal_type gid_node = node_map_->getGlobalElement(i);
  const local_ordinal_type lid_overlap_ = overlap_map_->getLocalElement(gid_node);

  std::vector<double> xyz(3);
  xyz[0]=mesh_->get_x(lid_overlap_);
  xyz[1]=mesh_->get_y(lid_overlap_);
  xyz[2]=mesh_->get_z(lid_overlap_);
  {
    auto ppv = ppvar_->get1dViewNonConst();
    ppv[i] = (*postprocfunc_)(u, uold, uoldold, gradu, &xyz[0], time, dt, dtold, eqn_id_);
  }
};

void post_process::update_mesh_data() const {

  if(!writedata_) return;

  const int num_nodes = overlap_map_->getLocalNumElements();

  Teuchos::RCP<vector_type> temp = Teuchos::rcp(new vector_type(overlap_map_));
  temp->doImport(*ppvar_,*importer_,Tpetra::INSERT);
  auto tv = temp->get1dView();

  std::vector<double> ppvar(num_nodes,0.);
  for (int nn=0; nn < num_nodes; nn++) {
      ppvar[nn]=tv[nn];
  }

  const std::string ystring=basename_+std::to_string(index_);
  mesh_->update_nodal_data(ystring, &ppvar[0]);

};
void post_process::update_scalar_data(const double &time){
  
  auto comm_ = Teuchos::DefaultComm<int>::getComm();
  scalar_reduction();//not sure if we need this here
  if ( (0 == comm_->getRank()) && (s_op_ != NONE) && (writedata_ == true) ){
    std::ofstream outfile;
    outfile.open(filename_, std::ios::app );
    outfile << std::setprecision(precision_)
      //<< std::setprecision(std::numeric_limits<double>::digits10 + 1)
	    <<time<<" "<<scalar_val_<<std::endl;
    outfile.close();
  }

};
double post_process::get_scalar_val() const {
  return scalar_val_;
};
void post_process::scalar_reduction(){

  auto comm_ = Teuchos::DefaultComm<int>::getComm();

  scalar_val_ =  0.;
  
  switch(s_op_){

  case NONE:   
    return;

  case NORM1:
    scalar_val_ = ppvar_->norm1();
    break;

  case NORM2:
    scalar_val_ = ppvar_->norm2();
    break;

    //cn normrms is used by adaptive time integration
    //wrms norm is sqrt( 1/N sum_i (x_i/w_i)^2)
    //cn with w = 1...1, we really just want 1/sqrt(n) *norm2
  case NORMRMS:{

    const double norm = ppvar_->norm2();
    const global_ordinal_type n = node_map_->getGlobalNumElements ();
    scalar_val_ = norm/sqrt((double)n);
    break;
  }

  case NORMINF:
    scalar_val_ = ppvar_->normInf();
    break;

  case MAXVALUE:{
    const global_ordinal_type n = node_map_->getGlobalNumElements ();
    auto ppv = ppvar_->get1dViewNonConst();
    double localmax = ppv[0];
    for( auto i = 1; i < n; i++ ) localmax = ((ppv[n] > localmax) ? ppv[n]:localmax);
    double globalmax = 0.;
    Teuchos::reduceAll(*comm_,Teuchos::REDUCE_MAX,1,&localmax,&globalmax);
    scalar_val_ = globalmax;
    break;
  }

  case MINVALUE:{
    const global_ordinal_type n = node_map_->getGlobalNumElements ();
    auto ppv = ppvar_->get1dViewNonConst();
    double localmin = ppv[0];
    for( auto i = 1; i < n; i++ ) localmin = ((ppv[n] < localmin) ? ppv[n]:localmin);
    double globalmin = 0.;
    Teuchos::reduceAll(*comm_,Teuchos::REDUCE_MAX,1,&localmin,&globalmin);
    scalar_val_ = globalmin;
    break;
  }

  case MEANVALUE:
    scalar_val_ = ppvar_->meanValue();
    break;

  default:    
    return;
  }
};
