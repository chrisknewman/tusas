//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef NOX_THYRA_MODEL_EVALUATOR_PHASE_HEAT_Exp_DEF_HPP
#define NOX_THYRA_MODEL_EVALUATOR_PHASE_HEAT_Exp_DEF_HPP

// Thyra support
#include "Thyra_DefaultSpmdVectorSpace.hpp"
#include "Thyra_DefaultSerialDenseLinearOpWithSolveFactory.hpp"
#include "Thyra_DetachedMultiVectorView.hpp"
#include "Thyra_DetachedVectorView.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Thyra_PreconditionerBase.hpp"
#include "Thyra_MLPreconditionerFactory.hpp"
#include "Thyra_DetachedSpmdVectorView.hpp"

// NOX support
#include "NOX_Thyra_MatrixFreeJacobianOperator.hpp"
#include "NOX_MatrixFree_ModelEvaluatorDecorator.hpp"

// Epetra support
#include "Thyra_EpetraThyraWrappers.hpp"
#include "Thyra_get_Epetra_Operator.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_Import.h"
#include "Epetra_CrsGraph.h"
#include "Epetra_CrsMatrix.h"

//teuchos support
#include <Teuchos_RCP.hpp>

// local support
#include "preconditioner.hpp"
#include "basis.hpp"

#include <iomanip>
// Nonmember constuctors

template<class Scalar>
Teuchos::RCP<ModelEvaluatorPHASE_HEAT_Exp<Scalar> >
modelEvaluatorPHASE_HEAT_Exp(const Teuchos::RCP<const Epetra_Comm>& comm,
            Mesh *mesh,
            const Scalar dt)
{
  return Teuchos::rcp(new ModelEvaluatorPHASE_HEAT_Exp<Scalar>(comm,mesh,dt));
}

// Constructor

template<class Scalar>
ModelEvaluatorPHASE_HEAT_Exp<Scalar>::
ModelEvaluatorPHASE_HEAT_Exp(const Teuchos::RCP<const Epetra_Comm>& comm,
            Mesh *mesh,
            const Scalar dt) :
  comm_(comm),
  dt_(dt),
  mesh_(mesh),
  showGetInvalidArg_(false)
{
  numeqs_ = 2;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using ::Thyra::VectorBase;
  typedef ::Thyra::ModelEvaluatorBase MEB;
  typedef Teuchos::ScalarTraits<Scalar> ST;

  TEUCHOS_ASSERT(nonnull(comm_));
  if (comm_->NumProc() != 1) {
    if (comm_->MyPID() == 0) std::cout<<std::endl<<"Only one processor supported"<<std::endl<<std::endl<<std::endl;
    exit(0);
  }
  mesh_->compute_nodal_adj();
  //const int num_nodes = num_global_elements_ + 1;
  const int num_nodes = mesh_->get_num_nodes();

  // owned space
  x_owned_map_ = rcp(new Epetra_Map(num_nodes*numeqs_,0,*comm_));
  //x_owned_map_->Print(std::cout);
  x_space_ = ::Thyra::create_VectorSpace(x_owned_map_);

  // residual space
  f_owned_map_ = x_owned_map_;
  f_space_ = x_space_;

  x0_ = ::Thyra::createMember(x_space_);
  V_S(x0_.ptr(), ST::zero());

  // Initialize the graph for W CrsMatrix object
  W_graph_ = createGraph();
  P_ = rcp(new Epetra_CrsMatrix(Copy,*W_graph_));
  prec_ = Teuchos::rcp(new preconditioner<Scalar>(P_, comm_));
  u_old_ = rcp(new Epetra_Vector(*f_owned_map_));
  dudt_ = rcp(new Epetra_Vector(*f_owned_map_));

  MEB::InArgsSetup<Scalar> inArgs;
  inArgs.setModelEvalDescription(this->description());
  inArgs.setSupports(MEB::IN_ARG_x);
  prototypeInArgs_ = inArgs;

  MEB::OutArgsSetup<Scalar> outArgs;
  outArgs.setModelEvalDescription(this->description());
  outArgs.setSupports(MEB::OUT_ARG_f);
  //outArgs.setSupports(MEB::OUT_ARG_W_op);
  outArgs.setSupports(MEB::OUT_ARG_W_prec);
//   outArgs.set_W_properties(DerivativeProperties(
//                  DERIV_LINEARITY_NONCONST
//                  ,DERIV_RANK_FULL
//                  ,true // supportsAdjoint
//                  ));
  prototypeOutArgs_ = outArgs;

  nominalValues_ = inArgs;
  nominalValues_.set_x(x0_);
  init_nox();
  time_=0.;

  K_ = 4.;
  T_m_ = 1.55;
  T_inf_ = 1.;
  alpha_ = 191.82;
  eps_ = .05;
  M_= 4.;
#if 0
  double pi = 3.141592653589793;
  std::cout<<"1  0    "<<theta(1.,0.)<<"  "<<0<<std::endl;
  std::cout<<"1  1    "<<theta(1.,1.)<<"  "<<pi/4.<<std::endl;
  std::cout<<"0  1    "<<theta(0.,1.)<<"  "<<pi/2.<<std::endl;
  std::cout<<"-1  1    "<<theta(-1.,1.)<<"  "<<3.*pi/4.<<std::endl;
  std::cout<<"-1  0    "<<theta(-1.,0.)<<"  "<<pi<<std::endl;
  std::cout<<"-1  -1    "<<theta(-1.,-1.)<<"  "<<5.*pi/4.<<std::endl;
  std::cout<<"0  -1    "<<theta(0.,-1.)<<"  "<<3.*pi/2.<<std::endl;
  std::cout<<"1  -1    "<<theta(1.,-1.)<<"  "<<7.*pi/4.<<std::endl;
  exit(0);
#endif
}

// Initializers/Accessors

template<class Scalar>
Teuchos::RCP<Epetra_CrsGraph>
ModelEvaluatorPHASE_HEAT_Exp<Scalar>::createGraph()
{
  Teuchos::RCP<Epetra_CrsGraph> W_graph;

  // Create the shell for the
  W_graph = Teuchos::rcp(new Epetra_CrsGraph(Copy, *x_owned_map_, 5));

  //nodes are numbered consecutively by nodeid
  for (int i=0; i < mesh_->get_num_nodes(); i++) {
    int num_nodes = (mesh_->get_nodal_adj(i)).size();
    std::vector<int> column (num_nodes);

    for (int j = 0; j < num_nodes; j++){
      column[j] = numeqs_*(mesh_->get_nodal_adj(i))[j];
    }
    column.push_back(numeqs_*i);//cn put the diagonal in
    W_graph->InsertGlobalIndices(numeqs_*i, column.size(), &column[0]);

    column.resize(num_nodes);
    for (int j = 0; j < num_nodes; j++){
      column[j] = numeqs_*(mesh_->get_nodal_adj(i))[j]+1;
    }
    column.push_back(numeqs_*i+1);//cn put the diagonal in
    W_graph->InsertGlobalIndices(numeqs_*i+1, column.size(), &column[0]);


  }//i

  W_graph->FillComplete();
  //W_graph->Print(std::cout);
  //exit(0);
  return W_graph;
}

template<class Scalar>
void ModelEvaluatorPHASE_HEAT_Exp<Scalar>::set_x0(const Teuchos::ArrayView<const Scalar> &x0_in)
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_EQUALITY(x_space_->dim(), x0_in.size());
#endif
  Thyra::DetachedVectorView<Scalar> x0(x0_);
  x0.sv().values()().assign(x0_in);
}


template<class Scalar>
void ModelEvaluatorPHASE_HEAT_Exp<Scalar>::setShowGetInvalidArgs(bool showGetInvalidArg)
{
  showGetInvalidArg_ = showGetInvalidArg;
}

template<class Scalar>
void ModelEvaluatorPHASE_HEAT_Exp<Scalar>::
set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory)
{
  W_factory_ = W_factory;
}

// Public functions overridden from ModelEvaulator


template<class Scalar>
Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
ModelEvaluatorPHASE_HEAT_Exp<Scalar>::get_x_space() const
{
  return x_space_;
}


template<class Scalar>
Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
ModelEvaluatorPHASE_HEAT_Exp<Scalar>::get_f_space() const
{
  return f_space_;
}


template<class Scalar>
Thyra::ModelEvaluatorBase::InArgs<Scalar>
ModelEvaluatorPHASE_HEAT_Exp<Scalar>::getNominalValues() const
{
  return nominalValues_;
}


template<class Scalar>
Teuchos::RCP<Thyra::LinearOpBase<Scalar> >
ModelEvaluatorPHASE_HEAT_Exp<Scalar>::create_W_op() const
{
  Teuchos::RCP<Epetra_CrsMatrix> W_epetra =
    Teuchos::rcp(new Epetra_CrsMatrix(::Copy,*W_graph_));

  return Thyra::nonconstEpetraLinearOp(W_epetra);
}

template<class Scalar>
Teuchos::RCP< ::Thyra::PreconditionerBase<Scalar> >
ModelEvaluatorPHASE_HEAT_Exp<Scalar>::create_W_prec() const
{

  const Teuchos::RCP<Thyra::LinearOpBase< Scalar > > P_op = prec_;

  Teuchos::RCP<Thyra::DefaultPreconditioner<Scalar> > prec =
    Teuchos::rcp(new Thyra::DefaultPreconditioner<Scalar>(Teuchos::null,P_op));

  return prec;
//  return Teuchos::null;
}

template<class Scalar>
Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> >
ModelEvaluatorPHASE_HEAT_Exp<Scalar>::get_W_factory() const
{
  return W_factory_;
}


template<class Scalar>
Thyra::ModelEvaluatorBase::InArgs<Scalar>
ModelEvaluatorPHASE_HEAT_Exp<Scalar>::createInArgs() const
{
  return prototypeInArgs_;
}


// Private functions overridden from ModelEvaulatorDefaultBase


template<class Scalar>
Thyra::ModelEvaluatorBase::OutArgs<Scalar>
ModelEvaluatorPHASE_HEAT_Exp<Scalar>::createOutArgsImpl() const
{
  return prototypeOutArgs_;
}


template<class Scalar>
void ModelEvaluatorPHASE_HEAT_Exp<Scalar>::evalModelImpl(
  const Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
  const Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs
  ) const
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcp_dynamic_cast;

  TEUCHOS_ASSERT(nonnull(inArgs.get_x()));

  const RCP<Thyra::VectorBase<Scalar> > f_out = outArgs.get_f();
  //const RCP<Thyra::LinearOpBase<Scalar> > W_out = outArgs.get_W_op();
  const RCP<Thyra::PreconditionerBase<Scalar> > W_prec_out = outArgs.get_W_prec();

  if ( nonnull(f_out) ||  nonnull(W_prec_out) ) {

    // ****************
    // Get the underlying epetra objects
    // ****************

    RCP<Epetra_Vector> f;
    if (nonnull(f_out)) {
      f = Thyra::get_Epetra_Vector(*f_owned_map_,outArgs.get_f());//f_out?
      //f->Print(std::cout);
    }

    if (nonnull(f_out))
      f->PutScalar(0.0);
    if (nonnull(W_prec_out))
      P_->PutScalar(0.0);

    RCP<const Epetra_Vector> u = (Thyra::get_Epetra_Vector(*x_owned_map_,inArgs.get_x()));
    //const Epetra_Vector &u = *(Thyra::get_Epetra_Vector(*x_owned_map_,inArgs.get_x()));

    double jac;
    double *xx, *yy;
    double *uu, *uu_old, *phiphi, *phiphi_old;
    int n_nodes_per_elem;

    Basis *ubasis, *phibasis;

    for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){

      n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);

      switch(n_nodes_per_elem){
	
      case 3 : // linear triangle
	ubasis = new BasisLTri;
	phibasis = new BasisLTri;
	break;
	
      case 4 : // linear quad
	ubasis = new BasisLQuad;
	phibasis = new BasisLQuad;
	break;
	
      }

      xx = new double[n_nodes_per_elem];
      yy = new double[n_nodes_per_elem];
      uu = new double[n_nodes_per_elem];
      uu_old = new double[n_nodes_per_elem];
      phiphi = new double[n_nodes_per_elem];
      phiphi_old = new double[n_nodes_per_elem];

      for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {// Loop Over # of Finite Elements on Processor

	for(int k = 0; k < n_nodes_per_elem; k++){
	  
	  int nodeid = mesh_->get_node_id(blk, ne, k);
	  
	  xx[k] = mesh_->get_x(nodeid);
	  yy[k] = mesh_->get_y(nodeid);
	  uu[k] = (*u)[numeqs_*nodeid]; 
	  uu_old[k] = (*u_old_)[numeqs_*nodeid];
	  phiphi[k] = (*u)[numeqs_*nodeid+1]; 
	  phiphi_old[k] = (*u_old_)[numeqs_*nodeid+1];
	  
	}//k

	double dx = 0.;
	for(int gp=0; gp < ubasis->ngp; gp++) {
	  ubasis->getBasis(gp, xx, yy);
	  dx += ubasis->jac*ubasis->wt;
	}
	if ( dx < 1e-6){
	  std::cout<<"dx = "<<dx<<"  ne = "<<ne<<std::endl<<std::endl<<std::endl;
	  exit(0);
	}
	dx = sqrt(dx);	
	double W_ = dx/.4;
	if ( W_ < 1e-6){
	  std::cout<<"W_ = "<<W_<<std::endl<<std::endl<<std::endl;
	  exit(0);
	}

	for(int gp=0; gp < ubasis->ngp; gp++) {// Loop Over Gauss Points 

	  // Calculate the basis function at the gauss point

	  ubasis->getBasis(gp, xx, yy, NULL, uu, uu_old, NULL);
	  phibasis->getBasis(gp, xx, yy, NULL, phiphi, phiphi_old, NULL);

	  // Loop over Nodes in Element

	  for (int i=0; i< n_nodes_per_elem; i++) {
	    int row = numeqs_*(mesh_->get_node_id(blk, ne, i));
	    double dphidx = ubasis->dphidxi[i]*ubasis->dxidx+ubasis->dphideta[i]*ubasis->detadx;
	    double dphidy = ubasis->dphidxi[i]*ubasis->dxidy+ubasis->dphideta[i]*ubasis->detady;
	    if (nonnull(f_out)) {
	      //double x = ubasis->xx;
	      //double y = ubasis->yy;
	      double delta = dx;	      

	      double divgradu = K_*ubasis->duolddx*dphidx + K_*ubasis->duolddy*dphidy;//(grad u,grad phi)
	      double ut = (ubasis->uu-ubasis->uuold)/dt_*ubasis->phi[i];
	      double phitu = -(phibasis->uu-phibasis->uuold)/dt_*ubasis->phi[i];     
	      double val = ubasis->jac * ubasis->wt * (ut + divgradu + phitu);
	      f->SumIntoGlobalValues ((int) 1, &val, &row);

	      double dphiphidx = phibasis->duolddx;
	      double dphiphidy = phibasis->duolddy;

	      double theta_ = theta(dphiphidx,dphiphidy);
	      double gs2_ = gs2(theta_);

	      double divgradphi = gs2_*phibasis->duolddx*dphidx + gs2_*phibasis->duolddy*dphidy;//(grad u,grad phi)
	      double phit = gs2_*(phibasis->uu-phibasis->uuold)/dt_*phibasis->phi[i];
	      double dg2 = dgs2_2dtheta(theta_);
	      double curlgrad = -dg2*(phibasis->duolddy*dphidx -phibasis->duolddx*dphidy);

	      double phidel2 = phibasis->uuold*(1.-phibasis->uuold)*(1.-2.*phibasis->uuold)/delta/delta*phibasis->phi[i];

	      double phidel = -5.*alpha_*(T_m_ - ubasis->uuold)
		*phibasis->uuold*phibasis->uuold*(1.-phibasis->uuold)*(1.-phibasis->uuold)/delta*phibasis->phi[i];

	      val = phibasis->jac * phibasis->wt * (phit + divgradphi + curlgrad + phidel2 + phidel);
	      int row1 = row+1;
	      f->SumIntoGlobalValues ((int) 1, &val, &row1);
	    }


	    // Loop over Trial Functions
	    if (nonnull(W_prec_out)) {
	      for(int j=0;j < n_nodes_per_elem; j++) {
		int column = numeqs_*(mesh_->get_node_id(blk, ne, j));
		double dtestdx = K_*ubasis->dphidxi[j]*ubasis->dxidx+K_*ubasis->dphideta[j]*ubasis->detadx;
		double dtestdy = K_*ubasis->dphidxi[j]*ubasis->dxidy+K_*ubasis->dphideta[j]*ubasis->detady;
		double divgrad = dtestdx * dphidx + dtestdy * dphidy;
		double phi_t = ubasis->phi[i] * ubasis->phi[j]/dt_;
		//double jac = ubasis->jac*ubasis->wt*(phi_t + divgrad);
		double jac = ubasis->jac*ubasis->wt*phi_t;
		//std::cout<<row<<" "<<column<<" "<<jac<<std::endl;
		P_->SumIntoGlobalValues(row, 1, &jac, &column);

		int row1 = row+1;
		int column1 = column+1;
		double dphiphidx = phibasis->dudx;
		double dphiphidy = phibasis->dudy;
		
		double theta_ = theta(dphiphidx,dphiphidy);
		double gs2_ = gs2(theta_);
		dtestdx = phibasis->dphidxi[j]*phibasis->dxidx+phibasis->dphideta[j]*phibasis->detadx;
		dtestdy = phibasis->dphidxi[j]*phibasis->dxidy+phibasis->dphideta[j]*phibasis->detady;
		divgrad = gs2_*dtestdx * dphidx + gs2_*dtestdy * dphidy;
		phi_t = gs2_*phibasis->phi[i] * phibasis->phi[j]/dt_;
		//jac = phibasis->jac*phibasis->wt*(phi_t + divgrad);
		jac = phibasis->jac*phibasis->wt*phi_t;
		P_->SumIntoGlobalValues(row1, 1, &jac, &column1);
	      }//j
	    }

	  }//i
	}//gp
      }//ne
#if 0
      //cn we should have a bc class that takes care of this
      if (nonnull(f_out)) {//cn double check the use of notnull throughout
	for ( int j = 0; j < mesh_->get_node_set(1).size(); j++ ){
	  
	  int row = numeqs_*(mesh_->get_node_set_entry(1, j));
	  
	  (*f)[row] =
	    (*u)[row] - T_inf_; // Dirichlet BC of zero
	  (*f)[row+1] =
	    (*u)[row+1] - 0.0;
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(2).size(); j++ ){
	  
	  int row = numeqs_*mesh_->get_node_set_entry(2, j);
	  
	  (*f)[row] =
	    (*u)[row] - T_inf_; // Dirichlet BC of zero
	  (*f)[row+1] =
	    (*u)[row+1] - 0.0;
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(3).size(); j++ ){
	  
	  int row = (numeqs_*mesh_->get_node_set_entry(3, j));
	  
	  (*f)[row] =
	    (*u)[row] - T_inf_; // Dirichlet BC of zero
	  (*f)[row+1] =
	    (*u)[row+1] - 0.0;
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(0).size(); j++ ){
	  
	  int row = numeqs_*(mesh_->get_node_set_entry(0, j));
	  
	  (*f)[row] =
	    (*u)[row] - T_inf_; // Dirichlet BC of zero
	  (*f)[row+1] =
	    (*u)[row+1] - 0.0;
	  
	}
	
      }
      
      if (nonnull(W_prec_out)) {
	int ns_id = 0;
	for ( int j = 0; j < mesh_->get_node_set(ns_id).size(); j++ ){
	  
	  int node = mesh_->get_node_set_entry(ns_id, j);
	  int row = numeqs_*node;
	  int num_nodes = (mesh_->get_nodal_adj(node)).size();

	  std::vector<int> column (num_nodes);
	  for (int k = 0; k < num_nodes; k++){
	    column[k] = numeqs_*(mesh_->get_nodal_adj(node))[k];
	  }
	  std::vector<double> vals (column.size(),0.);
	  column.push_back(row);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row, vals.size(), &vals[0],&column[0] );

	  column.resize(num_nodes);
	  vals.resize(num_nodes,0.);
	  int row1 = row +1;
	  for (int k = 0; k < num_nodes; k++){
	    column[k] = numeqs_*(mesh_->get_nodal_adj(node))[k]+1;
	  }
	  column.push_back(row1);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row1, vals.size(), &vals[0],&column[0] );

	  
	}
	ns_id = 1;
	for ( int j = 0; j < mesh_->get_node_set(2).size(); j++ ){
	  	  
	  int node = mesh_->get_node_set_entry(ns_id, j);
	  int row = numeqs_*node;
	  int num_nodes = (mesh_->get_nodal_adj(node)).size();

	  std::vector<int> column (num_nodes);
	  for (int k = 0; k < num_nodes; k++){
	    column[k] = numeqs_*(mesh_->get_nodal_adj(node))[k];
	  }
	  std::vector<double> vals (column.size(),0.);
	  column.push_back(row);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row, vals.size(), &vals[0],&column[0] );

	  column.resize(num_nodes);
	  vals.resize(num_nodes,0.);
	  int row1 = row +1;
	  for (int k = 0; k < num_nodes; k++){
	    column[k] = numeqs_*(mesh_->get_nodal_adj(node))[k]+1;
	  }
	  column.push_back(row1);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row1, vals.size(), &vals[0],&column[0] );

	}

	ns_id = 2;
	for ( int j = 0; j < mesh_->get_node_set(3).size(); j++ ){
	 
 
	  int node = mesh_->get_node_set_entry(ns_id, j);
	  int row = numeqs_*node;
	  int num_nodes = (mesh_->get_nodal_adj(node)).size();

	  std::vector<int> column (num_nodes);
	  for (int k = 0; k < num_nodes; k++){
	    column[k] = numeqs_*(mesh_->get_nodal_adj(node))[k];
	  }
	  std::vector<double> vals (column.size(),0.);
	  column.push_back(row);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row, vals.size(), &vals[0],&column[0] );

	  column.resize(num_nodes);
	  vals.resize(num_nodes,0.);
	  int row1 = row +1;
	  for (int k = 0; k < num_nodes; k++){
	    column[k] = numeqs_*(mesh_->get_nodal_adj(node))[k]+1;
	  }
	  column.push_back(row1);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row1, vals.size(), &vals[0],&column[0] );
	  
	}
	ns_id = 3;
	for ( int j = 0; j < mesh_->get_node_set(0).size(); j++ ){


	  int node = mesh_->get_node_set_entry(ns_id, j);
	  int row = numeqs_*node;
	  int num_nodes = (mesh_->get_nodal_adj(node)).size();

	  std::vector<int> column (num_nodes);
	  for (int k = 0; k < num_nodes; k++){
	    column[k] = numeqs_*(mesh_->get_nodal_adj(node))[k];
	  }
	  std::vector<double> vals (column.size(),0.);
	  column.push_back(row);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row, vals.size(), &vals[0],&column[0] );

	  column.resize(num_nodes);
	  vals.resize(num_nodes,0.);
	  int row1 = row +1;
	  for (int k = 0; k < num_nodes; k++){
	    column[k] = numeqs_*(mesh_->get_nodal_adj(node))[k]+1;
	  }
	  column.push_back(row1);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row1, vals.size(), &vals[0],&column[0] );
	  
	}
      
      }
#endif
    }//blk

    delete xx, yy, uu, phiphi, phiphi_old;
    delete ubasis, phibasis;
    
    if (nonnull(f_out)){
      //f->Print(std::cout);
    }
    if (nonnull(W_prec_out)) {
      P_->FillComplete();
      //P_->Print(std::cout);
      //exit(0);
      //std::cout<<" one norm P_ = "<<P_->NormOne()<<std::endl<<" inf norm P_ = "<<P_->NormInf()<<std::endl<<" fro norm P_ = "<<P_->NormFrobenius()<<std::endl;
      //Epetra_Vector d(*f_owned_map_);P_->ExtractDiagonalCopy(d);d.Print(std::cout);	
      prec_->ReComputePreconditioner();
    }
  }	
}

//====================================================================

template<class Scalar>
ModelEvaluatorPHASE_HEAT_Exp<Scalar>::~ModelEvaluatorPHASE_HEAT_Exp()
{
  //  if(!prec_.is_null()) prec_ = Teuchos::null;
}
template<class Scalar>
void ModelEvaluatorPHASE_HEAT_Exp<Scalar>::init_nox()
{
  ::Stratimikos::DefaultLinearSolverBuilder builder;

  Teuchos::RCP<Teuchos::ParameterList> lsparams =
    Teuchos::rcp(new Teuchos::ParameterList);
  //   lsparams->set("Linear Solver Type", "Belos");
//   lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Pseudo Block GMRES").set("Num Blocks",1);
//   lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Pseudo Block GMRES").set("Maximum Restarts",200);
  //lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Psuedo Block GMRES").set("Output Frequency",1);

  lsparams->set("Linear Solver Type", "AztecOO");
  lsparams->sublist("Linear Solver Types").sublist("AztecOO").sublist("Forward Solve").sublist("AztecOO Settings").set("Output Frequency",1);
  //lsparams->sublist("Linear Solver Types").sublist("AztecOO").sublist("Forward Solve").sublist("AztecOO Settings").sublist("AztecOO Preconditioner", "None");

  lsparams->set("Preconditioner Type", "None");
  builder.setParameterList(lsparams);
  //lsparams->print(cout);
  builder.getParameterList()->print(std::cout);

  Teuchos::RCP< ::Thyra::LinearOpWithSolveFactoryBase<double> >
    lowsFactory = builder.createLinearSolveStrategy("");

  // Setup output stream and the verbosity level
  Teuchos::RCP<Teuchos::FancyOStream>
    out = Teuchos::VerboseObjectBase::getDefaultOStream();
  lowsFactory->setOStream(out);
  lowsFactory->setVerbLevel(Teuchos::VERB_EXTREME);

  this->set_W_factory(lowsFactory);

  // Create the initial guess
  Teuchos::RCP< ::Thyra::VectorBase<double> >
    initial_guess = this->getNominalValues().get_x()->clone_v();

  Thyra::V_S(initial_guess.ptr(),Teuchos::ScalarTraits<double>::one());

  // Create the JFNK operator
  Teuchos::ParameterList printParams;
  Teuchos::RCP<Teuchos::ParameterList> jfnkParams = Teuchos::parameterList();
  jfnkParams->set("Difference Type","Forward");
  //jfnkParams->set("Perturbation Algorithm","KSP NOX 2001");
  jfnkParams->set("lambda",1.0e-4);
  Teuchos::RCP<NOX::Thyra::MatrixFreeJacobianOperator<double> > jfnkOp =
    Teuchos::rcp(new NOX::Thyra::MatrixFreeJacobianOperator<double>(printParams));
  jfnkOp->setParameterList(jfnkParams);
  jfnkParams->print(std::cout);

  Teuchos::RCP< ::Thyra::ModelEvaluator<double> > Model = Teuchos::rcpFromRef(*this);
  // Wrap the model evaluator in a JFNK Model Evaluator
  Teuchos::RCP< ::Thyra::ModelEvaluator<double> > thyraModel =
    Teuchos::rcp(new NOX::MatrixFreeModelEvaluatorDecorator<double>(Model));

  // Wrap the model evaluator in a JFNK Model Evaluator
//   Teuchos::RCP< ::Thyra::ModelEvaluator<double> > thyraModel =
//     Teuchos::rcp(new NOX::MatrixFreeModelEvaluatorDecorator<double>(this));

  Teuchos::RCP< ::Thyra::PreconditionerBase<double> > precOp = thyraModel->create_W_prec();
  // Create the NOX::Thyra::Group




  //bool precon = true;
  bool precon = false;
  Teuchos::RCP<NOX::Thyra::Group> nox_group;
  if(precon){
    nox_group =
      Teuchos::rcp(new NOX::Thyra::Group(*initial_guess, thyraModel, jfnkOp, lowsFactory, precOp, Teuchos::null));
  }
  else {
    nox_group =
      Teuchos::rcp(new NOX::Thyra::Group(*initial_guess, thyraModel, jfnkOp, lowsFactory, Teuchos::null, Teuchos::null));
  }

  nox_group->computeF();

  // VERY IMPORTANT!!!  jfnk object needs base evaluation objects.
  // This creates a circular dependency, so use a weak pointer.
  jfnkOp->setBaseEvaluationToNOXGroup(nox_group.create_weak());

  // Create the NOX status tests and the solver
  // Create the convergence tests
  Teuchos::RCP<NOX::StatusTest::NormF> absresid =
    Teuchos::rcp(new NOX::StatusTest::NormF(1.0e-8));
  Teuchos::RCP<NOX::StatusTest::NormF> relresid = 
    Teuchos::rcp(new NOX::StatusTest::NormF(*nox_group.get(), 1.0e-6));//1.0e-6 for paper
  Teuchos::RCP<NOX::StatusTest::NormWRMS> wrms =
    Teuchos::rcp(new NOX::StatusTest::NormWRMS(1.0e-2, 1.0e-8));
  Teuchos::RCP<NOX::StatusTest::Combo> converged =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::AND));
  //converged->addStatusTest(absresid);
  converged->addStatusTest(relresid);
  //converged->addStatusTest(wrms);
  Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters =
    Teuchos::rcp(new NOX::StatusTest::MaxIters(200));
  Teuchos::RCP<NOX::StatusTest::FiniteValue> fv =
    Teuchos::rcp(new NOX::StatusTest::FiniteValue);
  Teuchos::RCP<NOX::StatusTest::Combo> combo =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
  combo->addStatusTest(fv);
  combo->addStatusTest(converged);
  combo->addStatusTest(maxiters);

  // Create nox parameter list
  Teuchos::RCP<Teuchos::ParameterList> nl_params =
    Teuchos::rcp(new Teuchos::ParameterList);
  nl_params->set("Nonlinear Solver", "Line Search Based");
  //nl_params->sublist("Direction").sublist("Newton").sublist("Linear Solver").set("Tolerance", 1.0e-10);
  Teuchos::ParameterList& nlPrintParams = nl_params->sublist("Printing");
  nlPrintParams.set("Output Information",
		  NOX::Utils::OuterIteration  +
		  //                      NOX::Utils::OuterIterationStatusTest +
		  NOX::Utils::InnerIteration +
		  NOX::Utils::Details +
		  NOX::Utils::LinearSolverDetails);
  nl_params->sublist("Direction").sublist("Newton").set("Forcing Term Method", "Type 2");
  nl_params->sublist("Direction").sublist("Newton").set("Forcing Term Initial Tolerance", 1.0e-1);
  nl_params->sublist("Direction").sublist("Newton").set("Forcing Term Maximum Tolerance", 1.0e-2);
  nl_params->sublist("Direction").sublist("Newton").set("Forcing Term Minimum Tolerance", 1.0e-5);//1.0e-6
  // Create the solver
  solver_ =  NOX::Solver::buildSolver(nox_group, combo, nl_params);
}


template<class Scalar>
void ModelEvaluatorPHASE_HEAT_Exp<Scalar>::advance()
{
  Teuchos::RCP< VectorBase< double > > guess = Thyra::create_Vector(u_old_,x_space_);
  NOX::Thyra::Vector thyraguess(*guess);//by sending the dereferenced pointer, we instigate a copy rather than a view
  solver_->reset(thyraguess);

  NOX::StatusTest::StatusType solvStatus = solver_->solve();
  if( !(NOX::StatusTest::Converged == solvStatus)) {
    std::cout<<" NOX solver failed to converge. Status = "<<solvStatus<<std::endl<<std::endl;
    exit(0);
  }
  
  const Thyra::VectorBase<double> * sol = 
    &(dynamic_cast<const NOX::Thyra::Vector&>(
					      solver_->getSolutionGroup().getX()
					      ).getThyraVector()
      );
  //u_old_ = get_Epetra_Vector (*f_owned_map_,Teuchos::rcp_const_cast< ::Thyra::VectorBase< double > >(Teuchos::rcpFromRef(*sol))	);

  Thyra::ConstDetachedSpmdVectorView<double> x_vec(sol->col(0));

  for (int nn=0; nn < mesh_->get_num_nodes(); nn++) {//cn figure out a better way here...
    //(*dudt_)[numeqs_*nn]=(x_vec[numeqs_*nn] - (*u_old_)[numeqs_*nn])/dt_;
    //(*dudt_)[numeqs_*nn+1]=(x_vec[numeqs_*nn+1] - (*u_old_)[numeqs_*nn+1])/dt_;
    (*u_old_)[numeqs_*nn]=x_vec[numeqs_*nn];
    (*u_old_)[numeqs_*nn+1]=x_vec[numeqs_*nn+1];
  }
  //u_old_->Print(std::cout);
  time_ +=dt_;
}

template<class Scalar>
void ModelEvaluatorPHASE_HEAT_Exp<Scalar>::initialize()
{

  //cn we need an ic class that takes care of this


  //double pi = 3.141592653589793;
  for (int nn=0; nn < mesh_->get_num_nodes(); nn++) {
    double x = mesh_->get_x(nn);
    double y = mesh_->get_y(nn);
    double t = theta(x,y);
    double r = R(t);
    if(x*x+y*y < r*r){
      (*u_old_)[numeqs_*nn]=T_m_;
      (*u_old_)[numeqs_*nn+1]=1.;
    }
    else {
      (*u_old_)[numeqs_*nn]=T_inf_;
      (*u_old_)[numeqs_*nn+1]=0.;
    }
    

    //std::cout<<nn<<" "<<x<<" "<<y<<" "<<r<<"      "<<(*u_old_)[numeqs_*nn]<<"           "<<x*x+y*y<<" "<<r*r<<std::endl;
  }
  //exit(0);
}

template<class Scalar>
void ModelEvaluatorPHASE_HEAT_Exp<Scalar>::finalize()
{
  double outputu[mesh_->get_num_nodes()];
  double outputphi[mesh_->get_num_nodes()];
  double outputdudt[mesh_->get_num_nodes()];
  double outputdphidt[mesh_->get_num_nodes()];
  for (int nn=0; nn < mesh_->get_num_nodes(); nn++) {
    outputu[nn]=(*u_old_)[numeqs_*nn];
    //outputdudt[nn]=(*dudt_)[numeqs_*nn];
    outputphi[nn]=(*u_old_)[numeqs_*nn+1];
    //outputdphidt[nn]=(*dudt_)[numeqs_*nn+1];
    //std::cout<<nn<<" "<<outputu[nn]<<" "<<outputphi[nn]<<std::endl;
  }


  //cout<<"norm = "<<sqrt(norm)<<endl;
  const char *outfilename = "results.e";
  mesh_->add_nodal_data("u", outputu);
  mesh_->add_nodal_data("phi", outputphi);
  //mesh_->add_nodal_data("dudt", outputdudt);
  //mesh_->add_nodal_data("dphidt", outputdphidt);
  mesh_->write_exodus(outfilename);
  //compute_error(&outputu[0]);

}

template<class Scalar>
void ModelEvaluatorPHASE_HEAT_Exp<Scalar>::compute_error( double *u)
{
  double error = 0.;
  double jac;
  double *xx, *yy;
  double *uu;
  int n_nodes_per_elem, nodeid;
  
  Basis *ubasis;
  
  for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){
    
    n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);

    switch(n_nodes_per_elem){
      
    case 3 : // linear triangle
      ubasis = new BasisLTri;
      break;
      
    case 4 : // linear quad
      ubasis = new BasisLQuad;
      break;
      
    }
    xx = new double[n_nodes_per_elem];
    yy = new double[n_nodes_per_elem];
    uu = new double[n_nodes_per_elem];
    
    // Loop Over # of Finite Elements on Processor
    
    for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {
      
      
      for(int k = 0; k < n_nodes_per_elem; k++){
	
	nodeid = mesh_->get_node_id(blk, ne, k);
	
	xx[k] = mesh_->get_x(nodeid);
	yy[k] = mesh_->get_y(nodeid);
	uu[k] = u[nodeid];  // copy initial guess or old solution into local temp
	
	//std::cout<<"u  "<<u[k]<<std::endl;

      }//k
      
      // Loop Over Gauss Points
      for(int gp=0; gp < ubasis->ngp; gp++) { 
	
	// Calculate the basis function at the gauss point
	
	ubasis->getBasis(gp, xx, yy, uu);
	
	// Loop over Nodes in Element
	
	//for (int i=0; i< n_nodes_per_elem; i++) {
	  //nodeid = mesh_->get_node_id(blk, ne, i);
	  //double dphidx = ubasis->dphidxi[i]*ubasis->dxidx+ubasis->dphideta[i]*ubasis->detadx;
	  //double dphidy = ubasis->dphidxi[i]*ubasis->dxidy+ubasis->dphideta[i]*ubasis->detady;
	  double x = ubasis->xx;
	  double y = ubasis->yy;
	  
	  //double divgradu = ubasis->dudx*dphidx + ubasis->dudy*dphidy;//(grad u,grad phi)
	  //double ut = (ubasis->uu)/dt_*ubasis->phi[i];
	  double pi = 3.141592653589793;
	  //double ff = 2.*ubasis->phi[i];
	  double u_ex = sin(pi*x)*sin(pi*y)*exp(-2.*pi*pi*time_);	
	  //double u_ex = -x*(x-1.);	      
	  
	  error  += ubasis->jac * ubasis->wt * ( ((ubasis->uu)- u_ex)*((ubasis->uu)- u_ex));
	  //std::cout<<(ubasis->uu)<<" "<<u_ex<<" "<<(ubasis->uu)- u_ex<<std::endl;
	  
	  
	  //}//i
      }//gp
    }//ne
  }//blk
  std::cout<<"num dofs  "<<mesh_->get_num_nodes()<<"  num elem  "<<mesh_->get_num_elem()<<"  error is  "<<sqrt(error)<<std::endl;
  delete xx, yy, uu, ubasis;
}
template<class Scalar>
const double ModelEvaluatorPHASE_HEAT_Exp<Scalar>::gs( const double &theta)
{
  return 1. + eps_ * (M_*cos(theta));
}
template<class Scalar>
double ModelEvaluatorPHASE_HEAT_Exp<Scalar>::gs2( const double &theta) const
{ 
  //double g = 1. + eps_ * (M_*cos(theta));
  double g = 1. + eps_ * (cos(M_*theta));
  return g*g;
}
template<class Scalar>
double ModelEvaluatorPHASE_HEAT_Exp<Scalar>::dgs2_2dtheta(const double &theta) const
{
  //return -1.*(eps_*M_*(1. + eps_*M_*cos(theta))*sin(theta));
  return -1.*(eps_*M_*(1. + eps_*cos(M_*theta))*sin(M_*theta));
}
template<class Scalar>
const double ModelEvaluatorPHASE_HEAT_Exp<Scalar>::R(const double &theta)
{
  return .3*(1. + eps_ * cos(M_*theta));
}

template<class Scalar>
double ModelEvaluatorPHASE_HEAT_Exp<Scalar>::theta(double &x,double &y) const
{
  double small = 1e-9;
  double pi = 3.141592653589793;
  double t = 0.;
  if(std::abs(x) < small && y > 0. ) t = pi/2.;
  if(std::abs(x) < small && y < 0. ) t = 3.*pi/2.;
  if(x > small && y >= 0.) t= atan(y/x);
  if(x > small && y <0.) t= atan(y/x) + 2.*pi;
  if(x < -small) t= atan(y/x)+ pi;

  return t;
}
#endif
