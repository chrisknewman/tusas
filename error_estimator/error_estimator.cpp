//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#include "error_estimator.h"
#include "basis.hpp"
#include "elem_color.h"

#include <iostream>

#ifdef TUSAS_HAVE_ACML
#include "acml.h"
#elif defined TUSAS_HAVE_MKL
#include "mkl.h"
#else
//cn #include "clapack.h"
#endif

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

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
  bool quad_type = (0==elem_type.compare("QUAD4")) || (0==elem_type.compare("QUAD")) || (0==elem_type.compare("quad4")) || (0==elem_type.compare("quad")) 
    || (0==elem_type.compare("QUAD9")) || (0==elem_type.compare("quad9"));
  bool tri_type= (0==elem_type.compare("TRI3")) || (0==elem_type.compare("TRI")) || (0==elem_type.compare("tri3")) || (0==elem_type.compare("tri")); 
//     || (0==elem_type.compare("TRI6")) || (0==elem_type.compare("tri6"));
 
  if( !(quad_type || tri_type) ){ // linear quad
    if( 0 == comm_->MyPID() )std::cout<<"Error estimator only supports bilinear and quadratic quad and tri element types at this time."<<std::endl
	     <<elem_type<<" not supported."<<std::endl;
    exit(0);
  }

  mesh_->compute_nodal_patch_old();

  std::vector<int> node_num_map(mesh_->get_node_num_map());

  //we want this map to be a one equation version of the x_owned_map in tusas
  //do it this way and hope it is the same

  overlap_map_ = Teuchos::rcp(new Epetra_Map(-1,
					     node_num_map.size(),
					     &node_num_map[0],
					     0,
					     *comm_));
  if( 1 == comm_->NumProc() ){
    node_map_ = overlap_map_;
  }else{
    node_map_ = Teuchos::rcp(new Epetra_Map(Epetra_Util::Create_OneToOne_Map(*overlap_map_)));
  }
  //node_map_->Print(std::cout);
  importer_ = Teuchos::rcp(new Epetra_Import(*overlap_map_, *node_map_));

  gradx_ = Teuchos::rcp(new Epetra_Vector(*node_map_));
  grady_ = Teuchos::rcp(new Epetra_Vector(*node_map_));
  std::string xstring="grad"+std::to_string(index_)+"x";
  std::string ystring="grad"+std::to_string(index_)+"y";
  mesh_->add_nodal_field(xstring);
  mesh_->add_nodal_field(ystring);


  //cn this is the map of elements belonging to this processor
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
   
  if( 0 == comm_->MyPID() )std::cout<<"Error estimator created for variable "<<index_<<std::endl;

  ts_time_grad= Teuchos::TimeMonitor::getNewTimer("Total Gradient Est Time");
  ts_time_error= Teuchos::TimeMonitor::getNewTimer("Total Error Est Time");
  //exit(0);
}


error_estimator::~error_estimator()
{
  delete mesh_;
}

void error_estimator::estimate_gradient(const Teuchos::RCP<Epetra_Vector>& u_in){

  //not tested nor working with mpi

  //according to the ainsworth book, for bilinear quads it is better to sample
  //at centroids, rather than guass pts as is done here. This is due to
  //superconvergence at centroids. Guass pts are used for biquadratic quads.
  //There is a mechanism in the basis class where one guass point at the centroid could
  //be used. p would then be [x y 1] rather than [y*x x y 1].

  //Also alot of cleaning up could be done another time.  The changes above
  //could be incorporated then during ass wiping and generalization to other elements.

  //the vector coming in is dimensioned to numeqs_*num_nodes, hence we need to rethink the import here
  //the maps created above assume output of one variable
  //hack a copy/import for now....



  //also, currently working in 2D means a mesh in the x-y plane.
  //what do we do about a 2D mesh in the general x-y-z plane?
  //need to address this-- probably map x y z into a canonical 2d elem
  //similar to what is done for Neumann BCs
  //it is possible that this is already handled correctly within the basis class


  //also right now, with quadratic tris, it seems the midside nodes are getting garbage,
  //while the vertex nodes are correct
  //it looks like reasonable numbers are provided to lapack, but the solution is way off...
  //this corresponds to a square 6 x 6 matrix, could it be ill conditioned?--according to matlab
  //a significant number of these matrices have full rank
  //seems as though lapack is fucking up?
  //on QuadQ mesh, node 224:
  //a= [1 0.499429 0.458333 0.228905 0.249429 0.210069] b=[0.999963 -1.81203e-07] x=[7.04542  2.45351e+08]
  //   [1 0.454993 0.395833 0.180102 0.207019 0.156684]   [0.99996 -8.17299e-08]    [-20.7964  -8.44006e+08]
  //   [1 0.544074 0.395833 0.215362 0.296016 0.156684]   [0.999965 -1.77881e-07]   [-4.7735  -1.93729e+08]
  //   [1 0.499588 0.291667 0.145713 0.249589 0.0850694]  [0.999962 -1.51674e-07]   [0.0399212  1.62017e+06]
  //   [ 1 0.544114 0.354167 0.192707 0.29606 0.125434]   [0.999965 -1.95127e-08]   [20.8001  8.44152e+08]
  //   [1 0.455033 0.354167 0.161158 0.207055 0.125434]   [0.99996 -2.66636e-07]    [6.33808  2.57226e+08]
  // x(:,1) solves the system while x(:,2) is not even close

  //it could be that we need to call
  //dgesv_(integer *n, integer *nrhs, doublereal *a, integer 
  //	*lda, integer *ipiv, doublereal *b, integer *ldb, integer *info)
  //when n = m ?
  //or possibly use more than 3 guass pts for Qtri?

  //5-22-2017 cn thinks the best approach 
  //would check for gtri && m==n, then use dgesv_ instead of dgels_



  Teuchos::TimeMonitor GradEstTimer(*ts_time_grad);  

  Teuchos::RCP< Epetra_Vector> u1 = Teuchos::rcp(new Epetra_Vector(*node_map_));    
#pragma omp parallel for 
  for(int nn = 0; nn < mesh_->get_num_my_nodes(); nn++ ){
    (*u1)[nn]=(*u_in)[numeqs_*nn+index_]; 
  }
  Teuchos::RCP< Epetra_Vector> u = Teuchos::rcp(new Epetra_Vector(*overlap_map_));
  u->Import(*u1, *importer_, Insert);

  const int blk = 0;//for now

  int num_q_pts = -999;

  int dimp = -999;//dimension of 2d basis

  int nrhs = mesh_->get_num_dim();//num right hand size or dim of gradient = 2 for 2d

  std::string elem_type=mesh_->get_blk_elem_type(blk);
   
#ifdef ERROR_ESTIMATOR_OMP
#pragma omp parallel for
  for(int nn = 0; nn < mesh_->get_num_my_nodes(); nn++ ){
#endif
 
  Basis * basis;

  if( (0==elem_type.compare("QUAD4")) || (0==elem_type.compare("QUAD")) || (0==elem_type.compare("quad4")) || (0==elem_type.compare("quad")) ){ // linear quad   
    dimp = 4;
    num_q_pts = 4;
    int qpt_for_basis = sqrt(num_q_pts);
    basis = new BasisLQuad(qpt_for_basis);
  } 
  else if( (0==elem_type.compare("TRI3")) || (0==elem_type.compare("TRI")) || (0==elem_type.compare("tri3"))  || (0==elem_type.compare("tri"))) { // linear triangle
    dimp = 3;
    num_q_pts = 3;
    basis=new BasisLTri(3);
  }
  else if( (0==elem_type.compare("QUAD9")) || (0==elem_type.compare("quad9")) ){ // quadratic quad 
    dimp = 9;
    num_q_pts = 9;
    basis = new BasisQQuad();
  }
  else if( (0==elem_type.compare("TRI6")) || (0==elem_type.compare("tri6"))) { // quadratic triangle
    dimp = 6;
    num_q_pts = 3;
    basis=new BasisQTri();
  }
  else{
    std::cout<<"Error estimator only supports bilinear and quadratic quad and tri element types at this time."<<std::endl;
    std::cout<<elem_type<<" not supported."<<std::endl;
    exit(0);
  }

#ifdef ERROR_ESTIMATOR_OMP
#else
  for(int nn = 0; nn < mesh_->get_num_my_nodes(); nn++ ){
#endif

    int num_elem_in_patch = mesh_->get_nodal_patch(nn).size();

    //std::cout<<comm_->MyPID()<<" "<<nn<<" "<<num_elem_in_patch<<std::endl;

    int q = num_q_pts*num_elem_in_patch;//the matrix p will be q rows x dimp cols, 
    //the rows of p will be the basis evaluated at quadrature pts
//     if ( q < 3 || q > 18 )
    //std::cout<<"nn = "<<nn<<" q = "<<q<<" num_elem_in_patch = "<<num_elem_in_patch<<std::endl<<std::endl<<std::endl;

    std::vector<std::vector<double>> p(q);//q rows

    for(int i = 0; i < q; i++) p[i].resize(dimp);//dimp cols
    //the vector b will be q rows and 2 cols, grad u evaluated at the quadrature points

    int m = q;
    int n = dimp;
    int lda = q;
    int ldb = q;//b needs to hold the solution on return
    if ( n > m ) {ldb = n;}//cn need to really figure out what ldb is
    double * b = new double[ldb*nrhs];

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
	


	//cn I believe mesh_->get_node_id(blk, n_patch[ne], k); takes a local elem id in the second slot, not a global one
	//cn ie it expects the ith elem on this proc

	//cn right now, the patch has elements (on and off proc) according to the nodes on this proc, but the underlying mesh
	// does not know about off proc elements


	//int nodeid = mesh_->get_node_id(blk, ne, k);

	int lid = elem_map_->LID(n_patch[ne]);//cn this wont work because it is not an overlap map


	//int nodeid = mesh_->get_node_id(blk, n_patch[ne], k);
	int nodeid = mesh_->get_node_id(blk,lid, k);
	//std::cout<<comm_->MyPID()<<" "<<nn<<" "<<mesh_->node_num_map[nn]<<" "<<k<<" "<<n_patch[ne]<<" "<<nodeid<<" "<<lid<<std::endl;
	xx[k] = mesh_->get_x(nodeid);
	yy[k] = mesh_->get_y(nodeid);
	zz[k] = mesh_->get_z(nodeid);

	//uu[k] = (*u)[numeqs_*nodeid+index_]; 
	uu[k] = (*u)[nodeid]; 

      }//k

      //std::cout<<comm_->MyPID()<<" "<<nn<<" "<<num_elem_in_patch<<std::endl;

      //we could loop over dimension of p here to avoid if statements....
      for(int gp=0; gp < basis->ngp; gp++) {// Loop Over Gauss Points 
	basis->getBasis(gp, xx, yy, zz, uu);
	double x = basis->xx;
	double y = basis->yy;
	//double z = basis->zz;
	p[row][0] = 1.;            //tri3 || quad4 || tri6 || quad9
	p[row][1] = x;             //tri3 || quad4 || tri6 || quad9
	p[row][2] = y;             //tri3 || quad4 || tri6 || quad9
	//cn we would skip this for tris. However we compute it, but this column is not copied to a below
	//cn we need something better for quadratic and 3d
	if(3 < dimp) p[row][3] = x*y;      //quad4 || tri6 || quad9
	if(5 < dimp) {
	  p[row][4] = x*x;             //tri6 || quad9
	  p[row][5] = y*y;             //tri6 || quad9
	}
	if(8 < dimp) {
	  p[row][6] = x*x*y;                   //quad9
	  p[row][7] = x*y*y;                   //quad9
	  p[row][8] = x*x*y*y;                 //quad9
	}
	b[row] = basis->dudx;// du/dx
	b[row+ldb]  = basis->dudy;// du/dy
	//b[row+2*q]  = basis->dudz;// du/dz, ie in 3d => nrhs = 3 cols of b
// 	if(224==nn){
// 	  std::cout<<nn<<" "<<row<<" "<<row+ldb<<" : "<<p[row][0]<<" "<<p[row][1]<<" "<<p[row][2]<<" "<<p[row][3]
// 		   <<" "<<p[row][4]<<" "<<p[row][5]
// 		   <<" : "<<b[row]<<" "<<b[row+ldb]<<std::endl;
// 	}


	row++;
      }//gp
      delete xx, yy, zz, uu;
    }//ne
  

    //cn there is definitely something weird at domain corners and ltris with one quass pt,
    //cn this corresponds with ldb = q = 1 and the lapack error,
    //cn probably because there is only one guass pt per tri
    //cn on quads there are 4 guass pts

    //cn when changing  ltris to 3 gauss pts above, it seems fixed

    int info, lwork;
    
    double wkopt;
    double* work;
    
    double * a;
    
    //note that we fill a by column
    a = new double[lda*n];
#pragma omp parallel for collapse(2)
    for(int j = 0; j < n; j++){
      for(int i = 0; i < m; i++){
	a[j*lda+i] = p[i][j];
      }
    }
    
    //the first call queries the workspace
    lwork = -1;
    char msg[] = "No transpose";
    //char msg[] = "N";
#if TUSAS_HAVE_ACML
    dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, &wkopt, &lwork,
	    &info,0 );
#else
    //cn    dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, &wkopt, &lwork,
    //cn    &info );
#endif

    //std::cout<<"info 1 = "<<info<<std::endl<<std::endl;
    lwork = (int)wkopt;
    work = new double[lwork];
    //second call does the solve

//     std::cout<<std::endl;
//     if(224==nn){
//       for(int i = 0; i < ldb*nrhs; i++){
// 	std::cout<<b[i]<<" ";
//       }
//     }
//     std::cout<<std::endl;
#if TUSAS_HAVE_ACML
    dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
	    &info,0 );
#else
    //cn    dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
    //cn	    &info );
#endif
    if( info < 0 ) exit(0);
    //std::cout<<"info 2 = "<<info<<std::endl<<std::endl;
    //std::cout<<"info 2 = "<<info<<" ldb = "<<ldb<<std::endl; cout<<std::endl;
      
    delete work;

//     if(224==nn){
//       for(int i = 0; i < ldb*nrhs; i++){
// 	std::cout<<nn<<" "<<i<<":"<<b[i]<<" ";
//       }
//       std::cout<<std::endl;
//     }

    double x = mesh_->get_x(nn);
    double y = mesh_->get_y(nn);
    p[0][0] = 1.;
    p[0][1] = x;
    p[0][2] = y;
    //cn this would be skipped for tris as well; however it does not get summed in below
    //cn we only dimension p to 3 cols for tris and 4 cols for quad earlier
    if(3 < dimp) p[0][3] = x*y;
    if(5 < dimp) {
      p[0][4] = x*x;
      p[0][5] = y*y;
    }
    if(8 < dimp) {
      p[0][6] = x*x*y;
      p[0][7] = x*y*y;
      p[0][8] = x*x*y*y;
    }
    double gradx = 0.;
    double grady = 0.;
    for(int i = 0; i < n; i++){//cn could we use dimp rather than n here?
      gradx = gradx + p[0][i]*b[i];
      //grady = grady + p[0][i]*b[i+q];
      grady = grady + p[0][i]*b[i+ldb];
    }
    //std::cout<<nn<<" "<<x<<" "<<y<<" "<<gradx<<" "<<grady<<std::endl;
    //std::cout<<x<<"   "<<y<<"            "<<gradx<<std::endl;
    
    int gid = (mesh_->get_node_num_map())[nn];
    gradx_->ReplaceGlobalValues ((int) 1, (int) 0, &gradx, &gid);
    grady_->ReplaceGlobalValues ((int) 1, (int) 0, &grady, &gid);
   
    delete a;
    delete b;
#ifdef ERROR_ESTIMATOR_OMP
    delete basis;
#endif
  }//nn
#ifdef ERROR_ESTIMATOR_OMP
#else
  delete basis;
#endif
  //gradx_->Print(std::cout);
  //exit(0);
}

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

#if TUSAS_HAVE_ACML
  dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, &wkopt, &lwork,
	  &info,0 );
#else
  //cn  dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, &wkopt, &lwork,
  //cn	 &info );
#endif
  lwork = (int)wkopt;
  work = new double[lwork];
  //second call does the solve

#if TUSAS_HAVE_ACML
  dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
	  &info,0 );
#else
  //cn  dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
  //cn                        &info );
#endif
  delete work;

  std::cout<<"test_lapack"<<std::endl;
  std::cout<<" info = "<<info<<std::endl;
  std::cout<<" b[0] = "<<b[0]<<" b[1] = "<<b[1]<<std::endl;
  std::cout<<" b[3] = "<<b[3]<<" b[4] = "<<b[4]<<std::endl;

  delete a,b;

  exit(0);
};

void error_estimator::update_mesh_data(){
  
  Epetra_Vector *tempx,*tempy;
  tempx = new Epetra_Vector(*overlap_map_);
  tempx->Import(*gradx_, *importer_, Insert);
  tempy = new Epetra_Vector(*overlap_map_);
  tempy->Import(*grady_, *importer_, Insert);

  int num_nodes = overlap_map_->NumMyElements ();
  std::vector<double> gradx(num_nodes,0.);
  std::vector<double> grady(num_nodes,0.);
#pragma omp parallel for
  for (int nn=0; nn < num_nodes; nn++) {
      gradx[nn]=(*tempx)[nn];
      grady[nn]=(*tempy)[nn];
      //std::cout<<comm_->MyPID()<<" "<<nn<<" "<<grady[nn]<<" "<<(*grady_)[nn]<<std::endl;
  }
  std::string xstring="grad"+std::to_string(index_)+"x";
  std::string ystring="grad"+std::to_string(index_)+"y";
  mesh_->update_nodal_data(xstring, &gradx[0]);
  mesh_->update_nodal_data(ystring, &grady[0]);

  int num_elem = mesh_->get_elem_num_map()->size();
  std::vector<double> error(num_elem,0.);
#pragma omp parallel for
  for (int nn=0; nn < num_elem; nn++) {
      error[nn]=(*elem_error_)[nn];
      //std::cout<<comm_->MyPID()<<" "<<nn<<" "<<error[nn]<<" "<<(*elem_error_)[nn]<<std::endl;
  }
  std::string estring="error"+std::to_string(index_);
  mesh_->update_elem_data(estring, &error[0]);

  delete tempx, tempy;
}

void error_estimator::estimate_error(const Teuchos::RCP<Epetra_Vector>& u_in){
  
  Teuchos::TimeMonitor ErrorEstTimer(*ts_time_error); 

  Teuchos::RCP< Epetra_Vector> u1 = Teuchos::rcp(new Epetra_Vector(*node_map_));    
#pragma omp parallel for 
  for(int nn = 0; nn < mesh_->get_num_my_nodes(); nn++ ){
    (*u1)[nn]=(*u_in)[numeqs_*nn+index_]; 
  }

  Teuchos::RCP< Epetra_Vector> u = Teuchos::rcp(new Epetra_Vector(*overlap_map_));
  u->Import(*u1, *importer_, Insert);

  Epetra_Vector *tempx,*tempy;
  tempx = new Epetra_Vector(*overlap_map_);
  tempx->Import(*gradx_, *importer_, Insert);
  tempy = new Epetra_Vector(*overlap_map_);
  tempy->Import(*grady_, *importer_, Insert);

  const int blk = 0;//for now

#ifdef ERROR_ESTIMATOR_OMP
#pragma omp parallel for
    for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {
#endif

  double *xx, *yy, *zz;
  double *uu, *ux, *uy;
  int n_nodes_per_elem;

  std::string elem_type=mesh_->get_blk_elem_type(blk);

  Basis *basis;

  if( (0==elem_type.compare("QUAD4")) || (0==elem_type.compare("QUAD")) || (0==elem_type.compare("quad4")) || (0==elem_type.compare("quad")) ){ // linear quad   
    basis = new BasisLQuad();
  } 
  else if( (0==elem_type.compare("TRI3")) || (0==elem_type.compare("TRI")) || (0==elem_type.compare("tri3"))  || (0==elem_type.compare("tri"))) { // linear triangle
    basis = new BasisLTri();
  }
  else if( (0==elem_type.compare("QUAD9")) || (0==elem_type.compare("quad9")) ){ // quadratic quad 
    basis = new BasisQQuad();
  }
  else if( (0==elem_type.compare("TRI6")) || (0==elem_type.compare("tri6"))) { // quadratic triangle
    basis=new BasisQTri();
  }
  else{
    std::cout<<"Error estimator only supports bilinear and quadratic quad and tri element types at this time."<<std::endl;
    std::cout<<elem_type<<" not supported."<<std::endl;
    exit(0);
  }
  
  n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);
  
  xx = new double[n_nodes_per_elem];
  yy = new double[n_nodes_per_elem];
  zz = new double[n_nodes_per_elem];
  uu = new double[n_nodes_per_elem];
  ux = new double[n_nodes_per_elem];
  uy = new double[n_nodes_per_elem];
  
  // Loop Over # of Finite Elements on Processor

#ifdef ERROR_ESTIMATOR_OMP
#else  
  for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {
#endif
    double error = 0.;
    for(int k = 0; k < n_nodes_per_elem; k++){
      
      int nodeid = mesh_->get_node_id(blk, ne, k);
      
      xx[k] = mesh_->get_x(nodeid);
      yy[k] = mesh_->get_y(nodeid);
      zz[k] = mesh_->get_z(nodeid);
      uu[k] = (*u)[nodeid]; 
      ux[k] = (*tempx)[nodeid];
      uy[k] = (*tempy)[nodeid];
    }//k
    for(int gp=0; gp < basis->ngp; gp++) { 
      //ux is uuold, uy is uuoldold
      basis->getBasis(gp, xx, yy, zz, uu, ux, uy);
      double ex = (basis->dudx - basis->uuold);
      double ey = (basis->dudy - basis->uuoldold);

      error += basis->jac * basis->wt *(ex*ex + ey*ey);
      
//       std::cout<<comm_->MyPID()<<" "<<ne<<"  "<<gp<<"  "<<basis->dudx<<" "<<basis->uuold<<std::endl;
//       std::cout<<comm_->MyPID()<<" "<<ne<<"  "<<gp<<"  "<<ex*ex<<" "<<ey*ey<<std::endl;
    }//gp
    error = sqrt(error);
    int gid = (*(mesh_->get_elem_num_map()))[ne];
    elem_error_->ReplaceGlobalValues ((int) 1, (int) 0, &error, &gid);
    //std::cout<<ne<<"  "<<error<<std::endl;
#ifdef ERROR_ESTIMATOR_OMP
     delete xx, yy, zz, uu, ux, uy;
     delete basis;
#endif
  }//ne
#ifdef ERROR_ESTIMATOR_OMP
#else
  delete basis;
  delete xx, yy, zz, uu, ux, uy;
#endif
  delete tempx, tempy;
  //elem_error_->Print(std::cout);
  //std::cout<<estimate_global_error()<<std::endl;
  //   exit(0);
}

double error_estimator::estimate_global_error(){
  elem_error_->Norm2(&global_error_);
  return global_error_;
}

