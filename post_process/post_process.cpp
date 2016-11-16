#include "post_process.h"

#include <iostream>

//template<class Scalar>
post_process::post_process(const Teuchos::RCP<const Epetra_Comm>& comm, 
				 Mesh *mesh, 
				 const int index):  
  comm_(comm),
  mesh_(mesh),
  index_(index)
{
  std::vector<int> node_num_map(mesh_->get_node_num_map());

  Epetra_Map overlap_map(-1,
			 node_num_map.size(),
			 &node_num_map[0],
			 0,
			 *comm_);
  node_map_ = Teuchos::rcp(new Epetra_Map(Epetra_Util::Create_OneToOne_Map(overlap_map)));

  ppvar_ = Teuchos::rcp(new Epetra_Vector(*node_map_));
  std::string ystring="pp"+std::to_string(index_);
  mesh_->add_nodal_field(ystring);

  std::cout<<"Post process created for variable "<<index_<<std::endl;
  //exit(0);
};


post_process::~post_process(){};

void post_process::process(const int i,const double *u, const double *gradu)
{
  (*ppvar_)[i] = (*postprocfunc_)(u, gradu);
};

void post_process::update_mesh_data(){

  //int num_nodes = mesh_->get_node_num_map().size();
  int num_nodes = node_map_->NumMyElements();
  std::vector<double> ppvar(num_nodes);
  //#pragma omp parallel for
  for (int nn=0; nn < num_nodes; nn++) {
      ppvar[nn]=(*ppvar_)[nn];
  }
  std::string ystring="pp"+std::to_string(index_);
  mesh_->update_nodal_data(ystring, &ppvar[0]);

};
