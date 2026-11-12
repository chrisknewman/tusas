//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
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
#elif defined TUSAS_NO_LAPACK
#elif defined TUSAS_HAVE_CLAPACK
#include "clapack.h"
#endif
#include <Teuchos_LAPACK.hpp>

#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_OrdinalTraits.hpp>

#include "greedy_tie_break.hpp"

//template<class Scalar>
error_estimator::error_estimator(Mesh *mesh, 
				 const int numeqs, 
				 const int index):  
  mesh_(mesh),
  numeqs_(numeqs),
  index_(index)
{
  comm_ = Teuchos::DefaultComm<int>::getComm();

  //In parallel MPI, only nodes are ghosted, elements are not. Each processor accumulates the
  //least squares normal equations of the zz patch recovery from the elements it owns, and these
  //are summed on the owning processor (see estimate_gradient), so the recovered gradient uses the
  //full patch across processor boundaries.

  int blk = 0;
  std::string elem_type=mesh_->get_blk_elem_type(blk);
  bool quad_type = (0==elem_type.compare("QUAD4")) || (0==elem_type.compare("QUAD")) 
    || (0==elem_type.compare("quad4")) || (0==elem_type.compare("quad")) 
    || (0==elem_type.compare("QUAD9")) || (0==elem_type.compare("quad9"));
  bool tri_type= (0==elem_type.compare("TRI3")) || (0==elem_type.compare("TRI")) 
    || (0==elem_type.compare("tri3")) || (0==elem_type.compare("tri"));
  //    || (0==elem_type.compare("TRI6")) || (0==elem_type.compare("tri6"));
  bool hex_type = (0==elem_type.compare("HEX8")) || (0==elem_type.compare("HEX")) 
    || (0==elem_type.compare("hex8")) || (0==elem_type.compare("hex")); 
  bool tet_type= (0==elem_type.compare("TETRA4")) || (0==elem_type.compare("TETRA")) 
    || (0==elem_type.compare("tetra4")) || (0==elem_type.compare("tetra"));
 
  if( !(quad_type || tri_type || hex_type || tet_type) ){ 
    if( 0 == comm_->getRank() ){
      std::cout<<"Error estimator only supports bilinear and quadratic quad and tri element types at this time."<<std::endl
	       <<elem_type<<" not supported."<<std::endl;
    }
    exit(0);
  }
  //this is probably one of the problems
  //we can get the bounday nodes from nemesis::node_mapb, then use this to get boundary elements
  //note that elem_mapb only provides boundary elements with a face, so it does not give elems in
  //a reentrent corner, which is needed for the patch
  //we could then communicate the boundary elems and their nodal connectivity to create the maps
  mesh_->compute_nodal_patch_overlap();

  std::vector<Mesh::mesh_lint_t> node_num_int(mesh_->get_node_num_map());
  std::vector<global_ordinal_type> node_num_map(node_num_int.begin(),node_num_int.end());

  //we want this map to be a one equation version of the x_owned_map in tusas
  //do it this way and hope it is the same (same tie break as ModelEvaluatorTPETRA)

  const Tpetra::global_size_t numGlobalEntries = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  const global_ordinal_type indexBase = 0;

  overlap_map_ = Teuchos::rcp(new map_type(numGlobalEntries,
					   Teuchos::ArrayView<const global_ordinal_type>(node_num_map),
					   indexBase,
					   comm_));
  if( 1 == comm_->getSize() ){
    node_map_ = overlap_map_;
  }else{
    GreedyTieBreak<local_ordinal_type,global_ordinal_type> greedy_tie_break;
    node_map_ = Tpetra::createOneToOne(overlap_map_,greedy_tie_break);
  }
  //node_map_->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
  importer_ = Teuchos::rcp(new import_type(node_map_, overlap_map_));

  //recovery polynomial dimension and sampling points for this element type
  {
    const int nrhs = mesh_->get_num_dim();
    Basis * basis = create_recovery_basis();
    num_q_pts_ = basis->ngp;
    delete basis;
    if( quad_type ){
      const int nnpe = mesh_->get_num_nodes_per_elem_in_blk(blk);
      dimp_ = (9 == nnpe) ? 9 : 4;               //biquadratic : bilinear
    }
    else if( tri_type ) dimp_ = 3;               //linear
    else if( hex_type ) dimp_ = 8;               //trilinear
    else dimp_ = 4;                              //tet4, linear
    if( 3 == nrhs && quad_type ){
      if( 0 == comm_->getRank() ) std::cout<<"Error estimator: 2D elements in a 3D mesh not supported."<<std::endl;
      exit(0);
    }
  }

  //patch length scale for each node: max distance from the node to the nodes of its patch elements.
  //computed locally, reduced with ABSMAX on the owner and imported back so all processors agree.
  {
    vector_type hloc(overlap_map_);
    {
      Teuchos::ArrayRCP<scalar_type> hv = hloc.get1dViewNonConst();
      const int nnpe = mesh_->get_num_nodes_per_elem_in_blk(blk);
      for(int nn = 0; nn < mesh_->get_num_nodes(); nn++ ){
	const double x0 = mesh_->get_x(nn), y0 = mesh_->get_y(nn), z0 = mesh_->get_z(nn);
	double h = 0.;
	std::vector<int> n_patch(mesh_->get_nodal_patch_overlap(nn));
	for(int ne = 0; ne < (int)n_patch.size(); ne++){
	  for(int k = 0; k < nnpe; k++){
	    const int nodeid = mesh_->get_node_id(blk, n_patch[ne], k);
	    const double dx = mesh_->get_x(nodeid)-x0, dy = mesh_->get_y(nodeid)-y0, dz = mesh_->get_z(nodeid)-z0;
	    h = std::max(h, std::sqrt(dx*dx+dy*dy+dz*dz));
	  }
	}
	hv[nn] = h;
      }
    }
    vector_type hown(node_map_);
    hown.putScalar(0.);
    hown.doExport(hloc, *importer_, Tpetra::ABSMAX);
    patch_scale_ = Teuchos::rcp(new vector_type(overlap_map_));
    patch_scale_->doImport(hown, *importer_, Tpetra::INSERT);
  }

  int nrhs = mesh_->get_num_dim();

  gradx_ = Teuchos::rcp(new vector_type(node_map_));
  grady_ = Teuchos::rcp(new vector_type(node_map_));
  gradz_ = Teuchos::rcp(new vector_type(node_map_));

  std::string xstring="grad"+std::to_string(index_)+"x";
  std::string ystring="grad"+std::to_string(index_)+"y";
  mesh_->add_nodal_field(xstring);
  mesh_->add_nodal_field(ystring);
  std::string zstring="grad"+std::to_string(index_)+"z";
  mesh_->add_nodal_field(zstring);

  //cn this is the map of elements belonging to this processor
  std::vector<global_ordinal_type> elem_num_map(mesh_->get_elem_num_map()->begin(),
						mesh_->get_elem_num_map()->end());
  elem_map_ = Teuchos::rcp(new map_type(numGlobalEntries,
					Teuchos::ArrayView<const global_ordinal_type>(elem_num_map),
					indexBase,
					comm_));
  //elem_map_->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
  elem_error_ = Teuchos::rcp(new vector_type(elem_map_));
  std::string estring="error"+std::to_string(index_);
  mesh_->add_elem_field(estring);
  global_error_ = 0.;
   
  if( 0 == comm_->getRank() )std::cout<<"Error estimator created for variable "<<index_<<std::endl;

  ts_time_grad= Teuchos::TimeMonitor::getNewTimer("Total Gradient Est Time");
  ts_time_error= Teuchos::TimeMonitor::getNewTimer("Total Error Est Time");
  //exit(0);
}


error_estimator::~error_estimator()
{
  //delete mesh_;
}

void error_estimator::estimate_gradient(const Teuchos::RCP<Tpetra::Vector<> >& u_in){
  auto uview = u_in->get1dView();
  const int n = (int)(uview.size());
  std::vector<double> uvec(uview.get(),uview.get()+n);
  estimate_gradient(&uvec[0]);
}

Basis * error_estimator::create_recovery_basis() const
{
  //according to the ainsworth book, for bilinear quads it is better to sample
  //at centroids, rather than gauss pts as is done here. This is due to
  //superconvergence at centroids. Gauss pts are used for biquadratic quads.
  const int blk = 0;//for now
  std::string elem_type=mesh_->get_blk_elem_type(blk);
  if( (0==elem_type.compare("QUAD4")) || (0==elem_type.compare("QUAD")) || (0==elem_type.compare("quad4")) || (0==elem_type.compare("quad")) ){ // linear quad   
    return new BasisLQuad(2);
  } 
  else if( (0==elem_type.compare("TRI3")) || (0==elem_type.compare("TRI")) || (0==elem_type.compare("tri3"))  || (0==elem_type.compare("tri"))) { // linear triangle
    return new BasisLTri(3);
  }
  else if( (0==elem_type.compare("QUAD9")) || (0==elem_type.compare("quad9")) ){ // quadratic quad 
    return new BasisQQuad();
  }
  else if((0==elem_type.compare("HEX8")) || (0==elem_type.compare("HEX")) 
	  || (0==elem_type.compare("hex8")) || (0==elem_type.compare("hex"))){// hex8
    return new BasisLHex(2);
  }
  else if((0==elem_type.compare("TETRA4")) || (0==elem_type.compare("TETRA")) 
    || (0==elem_type.compare("tetra4")) || (0==elem_type.compare("tetra"))){// tet4
    return new BasisLTet();
  }
  std::cout<<"Error estimator only supports bilinear and quadratic quad and tri element types at this time."<<std::endl;
  std::cout<<elem_type<<" not supported."<<std::endl;
  exit(0);
  return NULL;
}

void error_estimator::eval_poly(const double x, const double y, const double z, double * p) const
{
  p[0] = 1.;            //tri3 || quad4 || quad9 || hex8 || tet4
  p[1] = x;
  p[2] = y;
  if(3 == dimp_) return;                       //tri3
  if(4 == dimp_ && 3 == mesh_->get_num_dim()){ //tet4
    p[3] = z;
    return;
  }
  p[3] = x*y;
  if(4 == dimp_) return;                       //quad4
  if(8 == dimp_){                              //hex8
    p[4] = z;
    p[5] = x*z;
    p[6] = y*z;
    p[7] = x*y*z;
    return;
  }
  p[4] = x*x;                                  //quad9
  p[5] = y*y;
  p[6] = x*x*y;
  p[7] = x*y*y;
  p[8] = x*x*y*y;
}

void error_estimator::estimate_gradient(const double * uvec ){
  //Zienkiewicz-Zhu superconvergent patch recovery.
  //For node n with patch elements e and Gauss points g, solve for each gradient component d
  //   A a_d = b_d,  A = sum_{e,g} p(xg) p(xg)^T,  b_d = sum_{e,g} p(xg) du_h/dx_d(xg)
  //with p evaluated in the local coordinates xi = (x - x_n)/h_n, then the recovered gradient
  //at the node is p(0)^T a_d = a_d[0].
  //Each processor sums over the elements it owns; the partial sums live on the overlap map
  //and are added on the owning processor with an export, so the patch crosses processor
  //boundaries without ghosting elements.

  //also, currently working in 2D means a mesh in the x-y plane.
  //what do we do about a 2D mesh in the general x-y-z plane?

  Teuchos::TimeMonitor GradEstTimer(*ts_time_grad);  

  Teuchos::RCP<vector_type> u1 = Teuchos::rcp(new vector_type(node_map_));
  {
    Teuchos::ArrayRCP<scalar_type> u1view = u1->get1dViewNonConst();
    //uvec is on the owned solution map (x_owned_map_ in ModelEvaluatorTPETRA), which has the same
    //nodes in the same order as node_map_. Note this can be more than mesh_->get_num_my_nodes(),
    //since shared nodes are assigned to the lowest rank by the tie break.
    const int num_my_nodes = (int)node_map_->getLocalNumElements();
#pragma omp parallel for 
    for(int nn = 0; nn < num_my_nodes; nn++ ){
      u1view[nn]=uvec[numeqs_*nn+index_]; 
    }
  }

  Teuchos::RCP<vector_type> u_ov = Teuchos::rcp(new vector_type(overlap_map_));
  u_ov->doImport(*u1, *importer_, Tpetra::INSERT);

  const int blk = 0;//for now
  const int nrhs = mesh_->get_num_dim();
  const int n = dimp_;
  const int nA = n*n;
  const int ncols = nA + n*nrhs;//column major A (n x n), then b (n x nrhs)

  //partial normal equations on the overlap map
  mv_type sys_ov(overlap_map_, ncols);
  {
    Teuchos::ArrayRCP<const scalar_type> u = u_ov->get1dView();
    Teuchos::ArrayRCP<const scalar_type> hv = patch_scale_->get1dView();
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_type> > sys = sys_ov.get2dViewNonConst();

    Basis * basis = create_recovery_basis();
    const int n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);
    std::vector<double> xx(n_nodes_per_elem), yy(n_nodes_per_elem), zz(n_nodes_per_elem), uu(n_nodes_per_elem);
    std::vector<double> p(n);

    for(int nn = 0; nn < mesh_->get_num_nodes(); nn++ ){
      const double x0 = mesh_->get_x(nn), y0 = mesh_->get_y(nn), z0 = mesh_->get_z(nn);
      const double h = (hv[nn] > 0.) ? hv[nn] : 1.;
      std::vector<int> n_patch(mesh_->get_nodal_patch_overlap(nn));

      for(int ne = 0; ne < (int)n_patch.size(); ne++){
	//n_patch[ne] is the local elem id; the patch only contains elements owned by this proc
	const int lid = n_patch[ne];
	for(int k = 0; k < n_nodes_per_elem; k++){
	  const int nodeid = mesh_->get_node_id(blk, lid, k);
	  xx[k] = mesh_->get_x(nodeid);
	  yy[k] = mesh_->get_y(nodeid);
	  zz[k] = mesh_->get_z(nodeid);
	  uu[k] = u[nodeid]; 
	}//k

	for(int gp=0; gp < basis->ngp; gp++) {
	  basis->getBasis(gp, &xx[0], &yy[0], &zz[0], &uu[0]);
	  eval_poly((basis->xx-x0)/h, (basis->yy-y0)/h, (basis->zz-z0)/h, &p[0]);
	  const double g[3] = {basis->dudx, basis->dudy, basis->dudz};
	  for(int j = 0; j < n; j++){
	    for(int i = 0; i < n; i++) sys[j*n+i][nn] += p[i]*p[j];
	    for(int d = 0; d < nrhs; d++) sys[nA+d*n+j][nn] += p[j]*g[d];
	  }
	}//gp
      }//ne
    }//nn
    delete basis;
  }//scope for host views

  //sum the partial systems on the owning processor
  mv_type sys_own(node_map_, ncols);
  sys_own.doExport(sys_ov, *importer_, Tpetra::ADD);

  //solve on owned nodes
  {
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<const scalar_type> > sys = sys_own.get2dView();
    Teuchos::ArrayRCP<scalar_type> gx = gradx_->get1dViewNonConst();
    Teuchos::ArrayRCP<scalar_type> gy = grady_->get1dViewNonConst();
    Teuchos::ArrayRCP<scalar_type> gz = gradz_->get1dViewNonConst();

    Teuchos::LAPACK<int,double> lapack;
    std::vector<double> a(nA), b(n*nrhs), sv(n);
    //SVD least squares handles rank deficient patches (eg corners with few elements) with the min norm solution
    const double rcond = 1.e-12;
    int rank, info;
    double wkopt;
    lapack.GELSS(n, n, nrhs, &a[0], n, &b[0], n, &sv[0], rcond, &rank, &wkopt, -1, &info);
    std::vector<double> work((int)wkopt);

    const int num_owned = (int)node_map_->getLocalNumElements();
    for(int i = 0; i < num_owned; i++){
      for(int c = 0; c < nA; c++) a[c] = sys[c][i];
      for(int c = 0; c < n*nrhs; c++) b[c] = sys[nA+c][i];
      lapack.GELSS(n, n, nrhs, &a[0], n, &b[0], n, &sv[0], rcond, &rank, &work[0], (int)work.size(), &info);
      if( info != 0 ){
	std::cout<<"Error estimator: GELSS failed with info = "<<info<<std::endl;
	exit(0);
      }
      //recovered gradient at the node is the constant coefficient, since p(0) = e_0
      gx[i] = b[0];
      gy[i] = b[n];
      gz[i] = (3 == nrhs) ? b[2*n] : 0.;
    }
  }
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

//   double p[m][n] = {
//     {0.2, 0.25},
//     {0.4, 0.5},
//     {0.4, 0.25}
//   };
  double ** p;p = new double*[m];p[0] = new double[n];p[1] = new double[n];p[2] = new double[n];
  p[0][0]=.2;p[0][1]=.25;p[1][0]=.4;p[1][1]=.5;p[2][0]=.4;p[2][1]=.25;

  //note that we fill a by column
  a = new double[lda*n];

  for(int j = 0; j < n; j++){
    for(int i = 0; i < m; i++){
      a[j*lda+i] = p[i][j];
    }
  }


//   b = new double[ldb*nrhs]  {
//     0.9, 1.7, 1.2,
//     0.9, 3.4, 2.4
//   };
  b = new double[ldb*nrhs];b[0]=0.9;b[1]= 1.7;b[2]= 1.2;b[3]= 0.9;b[4]= 3.4;b[5]= 2.4;

  //the first call queries the workspace
  lwork = -1;
  char msg[] = "No transpose";

#if TUSAS_HAVE_ACML
  dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, &wkopt, &lwork,
	  &info,0 );
#elif defined TUSAS_NO_LAPACK
  exit(0);
#elif defined TUSAS_HAVE_MKL
  dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, &wkopt, &lwork,
	 &info );
#elif defined TUSAS_HAVE_CLAPACK
  dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, &wkopt, &lwork,
	 &info );
#else
    Teuchos::LAPACK<int,double> lapack;    
    lapack.GELS( 'N', m, n, nrhs, a, lda, b, ldb, &wkopt, lwork,
	 &info );
#endif
  lwork = (int)wkopt;
  work = new double[lwork];
  //second call does the solve

#if TUSAS_HAVE_ACML
  dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
	  &info,0 );
#elif defined TUSAS_NO_LAPACK
#elif defined TUSAS_HAVE_MKL
  dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
                        &info );
#elif defined TUSAS_HAVE_CLAPACK
  dgels_( msg, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork,
                        &info );
#else   
    lapack.GELS( 'N', m, n, nrhs, a, lda, b, ldb, work, lwork,
                        &info );
#endif
  delete work;

  std::cout<<"test_lapack"<<std::endl;
  std::cout<<" info = "<<info<<std::endl;
  std::cout<<" b[0] = "<<b[0]<<" b[1] = "<<b[1]<<std::endl;
  std::cout<<" b[3] = "<<b[3]<<" b[4] = "<<b[4]<<std::endl;

  delete a;
  delete b;
  delete p[0];
  delete p[1];
  delete p[2];
  delete p;

  exit(0);
};

void error_estimator::update_mesh_data(){
  
  int nrhs = mesh_->get_num_dim();

  vector_type tempx(overlap_map_);
  tempx.doImport(*gradx_, *importer_, Tpetra::INSERT);
  vector_type tempy(overlap_map_);
  tempy.doImport(*grady_, *importer_, Tpetra::INSERT);
  vector_type tempz(overlap_map_);
  tempz.doImport(*gradz_, *importer_, Tpetra::INSERT);
  Teuchos::ArrayRCP<const scalar_type> tempxview = tempx.get1dView();
  Teuchos::ArrayRCP<const scalar_type> tempyview = tempy.get1dView();
  Teuchos::ArrayRCP<const scalar_type> tempzview = tempz.get1dView();
  
  const int num_nodes = (int)tempxview.size();
  std::vector<double> gradx(num_nodes,0.);
  std::vector<double> grady(num_nodes,0.);
  std::vector<double> gradz(num_nodes,0.);
#pragma omp parallel for
  for (int nn=0; nn < num_nodes; nn++) {
      gradx[nn]=tempxview[nn];
      grady[nn]=tempyview[nn];
      gradz[nn]=tempzview[nn];
  }
  std::string xstring="grad"+std::to_string(index_)+"x";
  std::string ystring="grad"+std::to_string(index_)+"y";
  mesh_->update_nodal_data(xstring, &gradx[0]);
  mesh_->update_nodal_data(ystring, &grady[0]);
  std::string zstring="grad"+std::to_string(index_)+"z";
  mesh_->update_nodal_data(zstring, &gradz[0]);

  int num_elem = mesh_->get_elem_num_map()->size();
  std::vector<double> error(num_elem,0.);
  {
    Teuchos::ArrayRCP<const scalar_type> errview = elem_error_->get1dView();
#pragma omp parallel for
    for (int nn=0; nn < num_elem; nn++) {
      error[nn]=errview[nn];
    }
  }
  std::string estring="error"+std::to_string(index_);
  mesh_->update_elem_data(estring, &error[0]);
}

void error_estimator::estimate_error(const Teuchos::RCP<Tpetra::Vector<> >& u_in){
  auto uview = u_in->get1dView();
  const int n = (int)(uview.size());
  std::vector<double> uvec(uview.get(),uview.get()+n);
  estimate_error(&uvec[0]);
}

void error_estimator::estimate_error(const double * uvec ){
  
  Teuchos::TimeMonitor ErrorEstTimer(*ts_time_error); 

  int nrhs = mesh_->get_num_dim();

  Teuchos::RCP<vector_type> u1 = Teuchos::rcp(new vector_type(node_map_));
  {
    Teuchos::ArrayRCP<scalar_type> u1view = u1->get1dViewNonConst();
    //uvec is on the owned solution map (x_owned_map_ in ModelEvaluatorTPETRA), which has the same
    //nodes in the same order as node_map_. Note this can be more than mesh_->get_num_my_nodes(),
    //since shared nodes are assigned to the lowest rank by the tie break.
    const int num_my_nodes = (int)node_map_->getLocalNumElements();
#pragma omp parallel for 
    for(int nn = 0; nn < num_my_nodes; nn++ ){
      u1view[nn]=uvec[numeqs_*nn+index_]; 
    }
  }

  Teuchos::RCP<vector_type> u_ov = Teuchos::rcp(new vector_type(overlap_map_));
  u_ov->doImport(*u1, *importer_, Tpetra::INSERT);

  vector_type tempx(overlap_map_);
  tempx.doImport(*gradx_, *importer_, Tpetra::INSERT);
  vector_type tempy(overlap_map_);
  tempy.doImport(*grady_, *importer_, Tpetra::INSERT);
  vector_type tempz(overlap_map_);
  tempz.doImport(*gradz_, *importer_, Tpetra::INSERT);

  Teuchos::ArrayRCP<const scalar_type> u = u_ov->get1dView();
  Teuchos::ArrayRCP<const scalar_type> tempxview = tempx.get1dView();
  Teuchos::ArrayRCP<const scalar_type> tempyview = tempy.get1dView();
  Teuchos::ArrayRCP<const scalar_type> tempzview = tempz.get1dView();
  Teuchos::ArrayRCP<scalar_type> errview = elem_error_->get1dViewNonConst();

  const int blk = 0;//for now

#ifdef ERROR_ESTIMATOR_OMP
#pragma omp parallel for
    for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {
#endif

  double *xx, *yy, *zz;
  double *uu, *ux, *uy, *uz;
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
  else if((0==elem_type.compare("HEX8")) || (0==elem_type.compare("HEX")) 
	  || (0==elem_type.compare("hex8")) || (0==elem_type.compare("hex"))){// hex8
    basis=new BasisLHex();
  }
  else if((0==elem_type.compare("TETRA4")) || (0==elem_type.compare("TETRA")) 
	  || (0==elem_type.compare("tetra4")) || (0==elem_type.compare("tetra"))){//tet4
    basis=new BasisLTet();
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
  uz = new double[n_nodes_per_elem];
  
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
      uu[k] = u[nodeid]; 
      ux[k] = tempxview[nodeid];
      uy[k] = tempyview[nodeid];
      uz[k] = tempzview[nodeid];
    }//k
    for(int gp=0; gp < basis->ngp; gp++) { 
      //ux is uuold, uy is uuoldold
      basis->getBasis(gp, xx, yy, zz, uu, ux, uy);
      double ex = (basis->dudx - basis->uuold);
      double ey = (basis->dudy - basis->uuoldold);

      error += basis->jac * basis->wt *(ex*ex + ey*ey);

      //cn need to do another call to get error in z...
      if(3 == nrhs){
	basis->getBasis(gp, xx, yy, zz, uu, uz);
	ex = (basis->dudz - basis->uuold);
	error += basis->jac * basis->wt *(ex*ex);
      }

    }//gp
    error = sqrt(error);
    //elem_map_ is built from elem_num_map, so local elem ne is local index ne in elem_map_
    errview[ne] = error;
    //std::cout<<ne<<"  "<<error<<std::endl;
#ifdef ERROR_ESTIMATOR_OMP
    delete xx;
    delete yy;
    delete zz;
    delete uu;
    delete ux;
    delete uy;
    delete uz;
    delete basis;
#endif
  }//ne
#ifdef ERROR_ESTIMATOR_OMP
#else
  delete basis;
  delete xx;
  delete yy;
  delete zz;
  delete uu;
  delete ux;
  delete uy;
  delete uz;
#endif
  //elem_error_->Print(std::cout);
  //std::cout<<estimate_global_error()<<std::endl;
  //   exit(0);
}

double error_estimator::estimate_global_error(){
  global_error_ = elem_error_->norm2();
  return global_error_;
}
