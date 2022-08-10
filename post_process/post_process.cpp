//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#include "post_process.h"

#include <iostream>
#include <fstream>

//template<class Scalar>
post_process::post_process(const Teuchos::RCP<const Epetra_Comm>& comm, 
			   Mesh *mesh, 
			   const int index,
			   SCALAR_OP s_op,
			   bool restart,
			   const int eqn_id,
			   const std::string basename,
			   double precision):  
  comm_(comm),
  mesh_(mesh),
  index_(index),
  s_op_(s_op),
  restart_(restart),
  eqn_id_(eqn_id),
  basename_(basename),
  precision_(precision)
{
  scalar_val_ =  0.;

  std::vector<Mesh::mesh_lint_t> node_num_map(mesh_->get_node_num_map());

  overlap_map_ = Teuchos::rcp(new Epetra_Map(-1,
					     node_num_map.size(),
					     &node_num_map[0],
					     0,
					     *comm_));
  if( 1 == comm_->NumProc() ){
    node_map_ = overlap_map_;
  }else{
#ifdef MESH_64
    node_map_ = Teuchos::rcp(new Epetra_Map(Create_OneToOne_Map64(*overlap_map_)));
#else
    node_map_ = Teuchos::rcp(new Epetra_Map(Epetra_Util::Create_OneToOne_Map(*overlap_map_)));
#endif
  }
  importer_ = Teuchos::rcp(new Epetra_Import(*overlap_map_, *node_map_));

  ppvar_ = Teuchos::rcp(new Epetra_Vector(*node_map_));
  std::string ystring=basename_+std::to_string(index_);
  mesh_->add_nodal_field(ystring);

  if ( (0 == comm_->MyPID()) && (s_op_ != NONE) ){
    filename_ = ystring+".dat";
    std::ofstream outfile;
    if( restart_ ){
      outfile.open(filename_, std::ios::app );
    }else{
      outfile.open(filename_);
    }
    outfile.close();
  }

  if ( 0 == comm_->MyPID())
    std::cout<<"Post process created for variable "<<index_<<" with name "<<ystring<<std::endl<<std::endl;
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

#ifdef MESH_64
  Mesh::mesh_lint_t gid_node = node_map_->GID64(i);
#else
  Mesh::mesh_lint_t gid_node = node_map_->GID(i);
#endif
  int lid_overlap = overlap_map_->LID(gid_node); 
  std::vector<double> xyz(3);
  xyz[0]=mesh_->get_x(lid_overlap);
  xyz[1]=mesh_->get_y(lid_overlap);
  xyz[2]=mesh_->get_z(lid_overlap);
  (*ppvar_)[i] = (*postprocfunc_)(u, uold, uoldold, gradu, &xyz[0], time, dt, dtold, eqn_id_);
};

void post_process::update_mesh_data(){

  Epetra_Vector *temp = new Epetra_Vector(*overlap_map_);
  temp->Import(*ppvar_, *importer_, Insert);
  int num_nodes = overlap_map_->NumMyElements();
  std::vector<double> ppvar(num_nodes,0.);
#pragma omp parallel for
  for (int nn=0; nn < num_nodes; nn++) {
      ppvar[nn]=(*temp)[nn];
  }
  std::string ystring=basename_+std::to_string(index_);
  mesh_->update_nodal_data(ystring, &ppvar[0]);

};
void post_process::update_scalar_data(double time){
  
  scalar_reduction();//not sure if we need this here
  if ( (0 == comm_->MyPID()) && (s_op_ != NONE) ){
    std::ofstream outfile;
    outfile.open(filename_, std::ios::app );
    outfile << std::setprecision(precision_)
      //<< std::setprecision(std::numeric_limits<double>::digits10 + 1)
	    <<time<<" "<<scalar_val_<<std::endl;
    outfile.close();
  }

};
double post_process::get_scalar_val(){
  return scalar_val_;
};
void post_process::scalar_reduction(){

  scalar_val_ =  0.;
  
  switch(s_op_){

  case NONE:   
    return;

  case NORM1:
    ppvar_->Norm1(&scalar_val_);
    break;

  case NORM2:
    ppvar_->Norm2(&scalar_val_);
    break;

  case NORMRMS:{
    Epetra_Vector *temp = new Epetra_Vector(*ppvar_);
    temp->PutScalar((double)1.);
    ppvar_->NormWeighted(*temp,&scalar_val_);
    break;
  }

  case NORMINF:
    ppvar_->NormInf(&scalar_val_);
    break;

  case MAXVALUE:
    ppvar_->MaxValue(&scalar_val_);
    break;

  case MINVALUE:
    ppvar_->MinValue(&scalar_val_);
    break;

  case MEANVALUE:
    ppvar_->MeanValue(&scalar_val_);
    break;

  default:    
    return;
  }
};
