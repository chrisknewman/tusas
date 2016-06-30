#include "error_estimator.h"
#include "basis.hpp"

#include <iostream>

#include "clapack.h"


//template<class Scalar>
error_estimator::error_estimator(const Teuchos::RCP<const Epetra_Comm>& comm, 
				 Mesh *mesh, 
				 const int numeqs, 
				 const int index):  
  comm_(comm),
  mesh_(mesh),
  numeqs_(numeqs),
  index_(index)
{
  int blk = 0;
  std::string elem_type=mesh_->get_blk_elem_type(blk);
  bool quad_type = (0==elem_type.compare("QUAD4")) || (0==elem_type.compare("QUAD")) || (0==elem_type.compare("quad4")) || (0==elem_type.compare("quad"));
  if( !quad_type ){ // linear quad
    std::cout<<"Error estimator only supports bilinear quad element types at this time."<<std::endl;
    std::cout<<elem_type<<" not supported."<<std::endl;
    exit(0);
  }

  mesh_->compute_nodal_patch();

  std::vector<int> node_num_map(mesh_->get_node_num_map());
  node_map_ = Teuchos::rcp(new Epetra_Map(-1,
				      node_num_map.size(),
				      &node_num_map[0],
				      0,
				      *comm_));
  gradx_ = Teuchos::rcp(new Epetra_Vector(*node_map_));
  grady_ = Teuchos::rcp(new Epetra_Vector(*node_map_));
  std::string xstring="grad"+std::to_string(index_)+"x";
  std::string ystring="grad"+std::to_string(index_)+"y";
  mesh_->add_nodal_field(xstring);
  mesh_->add_nodal_field(ystring);

  std::vector<int> elem_num_map(*(mesh_->get_elem_num_map()));
  elem_map_ = Teuchos::rcp(new Epetra_Map(-1,
				      elem_num_map.size(),
				      &elem_num_map[0],
				      0,
				      *comm_));
  elem_error_ = Teuchos::rcp(new Epetra_Vector(*elem_map_));
  std::string estring="error"+std::to_string(index_);
  mesh_->add_elem_field(estring);
  global_error_ = 0.;
  std::cout<<"Error estimator created for variable "<<index_<<std::endl;
  //exit(0);
};


error_estimator::~error_estimator(){};

void error_estimator::estimate_gradient(const Teuchos::RCP<Epetra_Vector>& u){

  //not tested nor working in parallel

  //according to the ainsworth book, for bilinear quads it is better to sample
  //at centroids, rather than guass pts as is done here. This is due to
  //superconvergence at centroids. Guass pts are used for biquadratic quads.
  //There is a mechanism in the basis class where one guass point at the centroid could
  //be used. p would then be [x y 1] rather than [y*x x y 1].

  //Also alot of cleaning up could be done another time.  The changes above
  //could be incorporated then during ass wiping and generalization to other elements.



  int blk = 0;//for now

  const int num_q_pts = 4;

  const int dimp = 4;//dimension of 2d basis

  int nrhs = 2;//num right hand size or dim of gradient = 2 for 2d

  int qpt_for_basis = sqrt(num_q_pts);

  Basis * basis = new BasisLQuad(qpt_for_basis);

  for(int nn = 0; nn < mesh_->get_num_nodes(); nn++ ){

    int num_elem_in_patch = mesh_->get_nodal_patch(nn).size();

    //std::cout<<nn<<" "<<num_elem_in_patch<<std::endl;

    int q = num_q_pts*num_elem_in_patch;//the matrix p will be q rows x dimp cols, 
    //the rows of p will be the basis evaluated at quadrature pts

    std::vector<std::vector<double>> p(q);
    for(int i = 0; i < q; i++) p[i].resize(dimp);

    //the vector b will be q rows and 2 cols, grad u evaluated at the quadrature points

    double * b = new double[2*q];

    int row = 0;
    std::vector<int> n_patch(mesh_->get_nodal_patch(nn));

    for(int ne = 0; ne < num_elem_in_patch; ne++){

      int n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);

      //these are coords at nodes
      double *xx, *yy, *zz, *uu;
      xx = new double[n_nodes_per_elem];
      yy = new double[n_nodes_per_elem];
      zz = new double[n_nodes_per_elem];
      uu = new double[n_nodes_per_elem];
      for(int k = 0; k < n_nodes_per_elem; k++){
	
	//int nodeid = mesh_->get_node_id(blk, ne, k);
	int nodeid = mesh_->get_node_id(blk, n_patch[ne], k);
	xx[k] = mesh_->get_x(nodeid);
	yy[k] = mesh_->get_y(nodeid);
	zz[k] = mesh_->get_z(nodeid);

	uu[k] = (*u)[numeqs_*nodeid+index_]; 

      }
      for(int gp=0; gp < basis->ngp; gp++) {// Loop Over Gauss Points 
	basis->getBasis(gp, xx, yy, zz, uu);
	double x = basis->xx;
	double y = basis->yy;
	//double z = ubasis->zz;
	p[row][0] = 1.;
	p[row][1] = x;
	p[row][2] = y;
	p[row][3] = x*y;
	b[row] = basis->dudx;// du/dx
	b[row+q]  = basis->dudy;// du/dy
// 	std::cout<<row<<" "<<row+q<<" "<<p[row][0]<<" "<<p[row][1]<<" "<<p[row][2]<<" "<<p[row][3]
// 		 <<" : "<<b[row]<<" "<<b[row+q]<<std::endl;
	row++;
      }
      delete xx, yy, zz, uu;
    }
    
    int m = q;
    int n = dimp;
    int lda = m;
    int ldb = m;//b needs to hold the solution on return
    int info, lwork;
    
    double wkopt;
    double* work;
    
    double * a;
    
    //note that we fill a by column
    a = new double[lda*n];
    
    for(int j = 0; j < n; j++){
      for(int i = 0; i < m; i++){
	a[j*lda+i] = p[i][j];
      }
    }
    
    //the first call queries the workspace
    lwork = -1;
    char msg[] = "No transpose";
    dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, &wkopt, &lwork,
	    &info );
    
    lwork = (int)wkopt;
    work = new double[lwork];
    //second call does the solve
    dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
	    &info );
    delete work;
//     for(int i = 0; i < n; i++){
//       std::cout<<b[i]<<" ";
//     }
//     std::cout<<std::endl;
    
    double x = mesh_->get_x(nn);
    double y = mesh_->get_y(nn);
    p[0][0] = 1.;
    p[0][1] = x;
    p[0][2] = y;
    p[0][3] = x*y;
    double gradx = 0.;
    double grady = 0.;
    for(int i = 0; i < n; i++){
      gradx = gradx + p[0][i]*b[i];
      grady = grady + p[0][i]*b[i+q];
    }
    //std::cout<<nn<<" "<<x<<" "<<y<<" "<<gradx<<" "<<grady<<std::endl;
    //std::cout<<x<<"   "<<y<<"            "<<gradx<<std::endl;
    
    gradx_->ReplaceGlobalValues ((int) 1, (int) 0, &gradx, &nn);
    grady_->ReplaceGlobalValues ((int) 1, (int) 0, &grady, &nn);
    
    delete a,b;
  }
  delete basis;
  //gradx_->Print(std::cout);
  //exit(0);
};

void error_estimator::test_lapack(){

  // solve the test problem:
  // >> A = [0.2 0.25; 
  //         0.4 0.5; 
  //         0.4 0.25];
  // >> b = [0.9 1.7 1.2]';
  // and
  // >> b = [0.9 3.4 2.4]';
  // >> x = A \ b
  // x =
  // 1.7000
  // 2.0800
  // and
  // x =
  // 4.3
  // 2.72

  //using dgels_
  //there is an example on how to call dgels_ at
  //https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgels_ex.c.htm

  int m = 3;
  int n = 2;
  int lda = m;
  int ldb = m;
  int nrhs = 2;
  int info, lwork;

  double wkopt;
  double* work;

  double * a;
  double * b;

  double p[m][n] = {
    {0.2, 0.25},
    {0.4, 0.5},
    {0.4, 0.25}
  };

  //note that we fill a by column
  a = new double[lda*n];

  for(int j = 0; j < n; j++){
    for(int i = 0; i < m; i++){
      a[j*lda+i] = p[i][j];
    }
  }


  b = new double[ldb*nrhs]  {
    0.9, 1.7, 1.2,
    0.9, 3.4, 2.4
  };

  //the first call queries the workspace
  lwork = -1;
  char msg[] = "No transpose";
  dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, &wkopt, &lwork,
	 &info );

  lwork = (int)wkopt;
  work = new double[lwork];
  //second call does the solve
  dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
                        &info );
  delete work;

  std::cout<<"test_lapack"<<std::endl;
  std::cout<<" info = "<<info<<std::endl;
  std::cout<<" b[0] = "<<b[0]<<" b[1] = "<<b[1]<<std::endl;
  std::cout<<" b[3] = "<<b[3]<<" b[4] = "<<b[4]<<std::endl;

  delete a,b;

  exit(0);

};

void error_estimator::update_mesh_data(){
  
  int num_nodes = mesh_->get_node_num_map().size();
  std::vector<double> gradx(num_nodes);
  std::vector<double> grady(num_nodes);
#pragma omp parallel for
  for (int nn=0; nn < num_nodes; nn++) {
      gradx[nn]=(*gradx_)[nn];
      grady[nn]=(*grady_)[nn];
  }
  std::string xstring="grad"+std::to_string(index_)+"x";
  std::string ystring="grad"+std::to_string(index_)+"y";
  mesh_->update_nodal_data(xstring, &gradx[0]);
  mesh_->update_nodal_data(ystring, &grady[0]);

  int num_elem = mesh_->get_elem_num_map()->size();
  std::vector<double> error(num_elem);
#pragma omp parallel for
  for (int nn=0; nn < num_elem; nn++) {
      error[nn]=(*elem_error_)[nn];
  }
  std::string estring="error"+std::to_string(index_);
  mesh_->update_elem_data(estring, &error[0]);

};

void error_estimator::estimate_error(const Teuchos::RCP<Epetra_Vector>& u){

  int blk = 0;//for now

  double *xx, *yy, *zz;
  double *uu, *ux, *uy;
  int n_nodes_per_elem;

  Basis *basis;

  for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){
    basis = new BasisLQuad();
    
    n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);
    
    xx = new double[n_nodes_per_elem];
    yy = new double[n_nodes_per_elem];
    zz = new double[n_nodes_per_elem];
    uu = new double[n_nodes_per_elem];
    ux = new double[n_nodes_per_elem];
    uy = new double[n_nodes_per_elem];
    
    // Loop Over # of Finite Elements on Processor
    
    for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {
      double error = 0.;
      for(int k = 0; k < n_nodes_per_elem; k++){
	
	int nodeid = mesh_->get_node_id(blk, ne, k);
	
	xx[k] = mesh_->get_x(nodeid);
	yy[k] = mesh_->get_y(nodeid);
	zz[k] = mesh_->get_z(nodeid);
	uu[k] = (*u)[numeqs_*nodeid+index_]; 
	ux[k] = (*gradx_)[nodeid];
	uy[k] = (*grady_)[nodeid];
      }//k
      for(int gp=0; gp < basis->ngp; gp++) { 
	//ux is uuold, uy is uuoldold
	basis->getBasis(gp, xx, yy, zz, uu, ux, uy);
	double ex = (basis->dudx - basis->uuold);
	double ey = (basis->dudy - basis->uuoldold);
	error += basis->jac * basis->wt *(ex*ex + ey*ey);

	//std::cout<<ne<<"  "<<gp<<"  "<<basis->dudx<<" "<<basis->uuold<<std::endl;
	//std::cout<<ne<<"  "<<gp<<"  "<<ex*ex<<" "<<ey*ey<<std::endl;
      }//gp
      error = sqrt(error);
      elem_error_->ReplaceGlobalValues ((int) 1, (int) 0, &error, &ne);
      //std::cout<<ne<<"  "<<error<<std::endl;
    }//ne
    delete xx, yy, zz, uu, ux, uy;
  }//blk
//   std::cout<<estimate_global_error()<<std::endl;
//   exit(0);
};

double error_estimator::estimate_global_error(){
  elem_error_->Norm2(&global_error_);
  return global_error_;
};

