#ifndef NOX_THYRA_MODEL_EVALUATOR_PHASE_HEAT_DEF_HPP
#define NOX_THYRA_MODEL_EVALUATOR_PHASE_HEAT_DEF_HPP

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
#include "Teuchos_ParameterList.hpp"
#include <Teuchos_TimeMonitor.hpp>

// local support
#include "preconditioner.hpp"
#include "basis.hpp"
#include "ParamNames.h"

#include <iomanip>
#include <iostream>

#include "function_def.hpp"

// Nonmember constuctors

template<class Scalar>
Teuchos::RCP<ModelEvaluatorPHASE_HEAT<Scalar> >
modelEvaluatorPHASE_HEAT(const Teuchos::RCP<const Epetra_Comm>& comm,
			 Mesh *mesh,
			 Teuchos::ParameterList plist
			 )
{
  return Teuchos::rcp(new ModelEvaluatorPHASE_HEAT<Scalar>(comm,mesh,plist));
}

// Constructor

template<class Scalar>
ModelEvaluatorPHASE_HEAT<Scalar>::
ModelEvaluatorPHASE_HEAT(const Teuchos::RCP<const Epetra_Comm>& comm,
			 Mesh *mesh,
			 Teuchos::ParameterList plist 
			 ) :
  comm_(comm),
  paramList(plist),
  mesh_(mesh),
  showGetInvalidArg_(false)
{
  dt_ = paramList.get<double> (TusasdtNameString);
  t_theta_ = paramList.get<double> (TusasthetaNameString);
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
  u_old_old_ = rcp(new Epetra_Vector(*f_owned_map_));
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
  time_=0.;

  K_ = 4.;
  T_m_ = 1.55;
  T_inf_ = 1.;
  alpha_ = 191.82;
  eps_ = .05;
  M_= 4.;
  theta_0_ =0.;

  //function pointers
  hp1_ = &hp1_cummins_;
  w_ = &w_cummins_;
  m_ = &m_cummins_;
  rand_phi_ = &rand_phi_cummins_;
  gp1_ = &gp1_cummins_;
  hp2_ = &hp2_cummins_;

  nnewt_=0;
  init_nox();
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
ModelEvaluatorPHASE_HEAT<Scalar>::createGraph()
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
void ModelEvaluatorPHASE_HEAT<Scalar>::set_x0(const Teuchos::ArrayView<const Scalar> &x0_in)
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_EQUALITY(x_space_->dim(), x0_in.size());
#endif
  Thyra::DetachedVectorView<Scalar> x0(x0_);
  x0.sv().values()().assign(x0_in);
}


template<class Scalar>
void ModelEvaluatorPHASE_HEAT<Scalar>::setShowGetInvalidArgs(bool showGetInvalidArg)
{
  showGetInvalidArg_ = showGetInvalidArg;
}

template<class Scalar>
void ModelEvaluatorPHASE_HEAT<Scalar>::
set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory)
{
  W_factory_ = W_factory;
}

// Public functions overridden from ModelEvaulator


template<class Scalar>
Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
ModelEvaluatorPHASE_HEAT<Scalar>::get_x_space() const
{
  return x_space_;
}


template<class Scalar>
Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
ModelEvaluatorPHASE_HEAT<Scalar>::get_f_space() const
{
  return f_space_;
}


template<class Scalar>
Thyra::ModelEvaluatorBase::InArgs<Scalar>
ModelEvaluatorPHASE_HEAT<Scalar>::getNominalValues() const
{
  return nominalValues_;
}


template<class Scalar>
Teuchos::RCP<Thyra::LinearOpBase<Scalar> >
ModelEvaluatorPHASE_HEAT<Scalar>::create_W_op() const
{
  Teuchos::RCP<Epetra_CrsMatrix> W_epetra =
    Teuchos::rcp(new Epetra_CrsMatrix(::Copy,*W_graph_));

  return Thyra::nonconstEpetraLinearOp(W_epetra);
}

template<class Scalar>
Teuchos::RCP< ::Thyra::PreconditionerBase<Scalar> >
ModelEvaluatorPHASE_HEAT<Scalar>::create_W_prec() const
{

  const Teuchos::RCP<Thyra::LinearOpBase< Scalar > > P_op = prec_;

  Teuchos::RCP<Thyra::DefaultPreconditioner<Scalar> > prec =
    Teuchos::rcp(new Thyra::DefaultPreconditioner<Scalar>(Teuchos::null,P_op));

  return prec;
//  return Teuchos::null;
}

template<class Scalar>
Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> >
ModelEvaluatorPHASE_HEAT<Scalar>::get_W_factory() const
{
  return W_factory_;
}


template<class Scalar>
Thyra::ModelEvaluatorBase::InArgs<Scalar>
ModelEvaluatorPHASE_HEAT<Scalar>::createInArgs() const
{
  return prototypeInArgs_;
}


// Private functions overridden from ModelEvaulatorDefaultBase


template<class Scalar>
Thyra::ModelEvaluatorBase::OutArgs<Scalar>
ModelEvaluatorPHASE_HEAT<Scalar>::createOutArgsImpl() const
{
  return prototypeOutArgs_;
}


template<class Scalar>
void ModelEvaluatorPHASE_HEAT<Scalar>::evalModelImpl(
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
    double *xx, *yy, *zz;
    double *uu, *uu_old, *phiphi, *phiphi_old, *phiphi_old_old;
    int n_nodes_per_elem;

    Basis *ubasis, *phibasis, *phibasis2;

    int dim = mesh_->get_num_dim();

    for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){

      n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);

      switch(n_nodes_per_elem){
	
      case 3 : // linear triangle
	ubasis = new BasisLTri;
	phibasis = new BasisLTri;
	phibasis2 = new BasisLTri;
	break;
	
      case 4 : // linear quad
	ubasis = new BasisLQuad;
	phibasis = new BasisLQuad;
	phibasis2 = new BasisLQuad;
	break;
	
      case 8 : // linear hex
	ubasis = new BasisLHex;
	phibasis = new BasisLHex;
	phibasis2 = new BasisLHex;
	break;
	
	
      }

      xx = new double[n_nodes_per_elem];
      yy = new double[n_nodes_per_elem];
      zz = new double[n_nodes_per_elem];
      uu = new double[n_nodes_per_elem];
      uu_old = new double[n_nodes_per_elem];
      phiphi = new double[n_nodes_per_elem];
      phiphi_old = new double[n_nodes_per_elem];
      phiphi_old_old = new double[n_nodes_per_elem];

      for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {// Loop Over # of Finite Elements on Processor

	for(int k = 0; k < n_nodes_per_elem; k++){
	  
	  int nodeid = mesh_->get_node_id(blk, ne, k);
	  
	  xx[k] = mesh_->get_x(nodeid);
	  yy[k] = mesh_->get_y(nodeid);
	  zz[k] = mesh_->get_z(nodeid);
	  uu[k] = (*u)[numeqs_*nodeid]; 
	  uu_old[k] = (*u_old_)[numeqs_*nodeid];
	  phiphi[k] = (*u)[numeqs_*nodeid+1]; 
	  phiphi_old[k] = (*u_old_)[numeqs_*nodeid+1];
	  phiphi_old_old[k] = (*u_old_old_)[numeqs_*nodeid+1];
	  
	}//k

	double dx = 0.;
	for(int gp=0; gp < ubasis->ngp; gp++) {
 	  if(3 == dim) {
 	    ubasis->getBasis(gp, xx, yy, zz);
 	  }else{
	    ubasis->getBasis(gp, xx, yy);
 	  }
	  //std::cout<<ubasis->jac<<"   "<<ubasis->wt<<std::endl;
	  dx += ubasis->jac*ubasis->wt;
	}
	if ( dx < 1e-6){
	  std::cout<<std::endl<<"Negative element size found"<<std::endl;
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

 	  if(3 == dim) {
 	    ubasis->getBasis(gp, xx, yy, zz, uu, uu_old);
 	    phibasis->getBasis(gp, xx, yy, zz, phiphi, phiphi_old);
 	    phibasis2->getBasis(gp, xx, yy, zz, phiphi_old_old);
 	  }else{
	    ubasis->getBasis(gp, xx, yy, uu, uu_old);
	    phibasis->getBasis(gp, xx, yy, phiphi, phiphi_old);
	    phibasis2->getBasis(gp, xx, yy, phiphi_old_old);
 	  }

	  // Loop over Nodes in Element

	  //srand(123);

	  for (int i=0; i< n_nodes_per_elem; i++) {
	    int row = numeqs_*(mesh_->get_node_id(blk, ne, i));

	    double dphidx = ubasis->dphidxi[i]*ubasis->dxidx
	      +ubasis->dphideta[i]*ubasis->detadx
	      +ubasis->dphidzta[i]*ubasis->dztadx;
	    double dphidy = ubasis->dphidxi[i]*ubasis->dxidy
	      +ubasis->dphideta[i]*ubasis->detady
	      +ubasis->dphidzta[i]*ubasis->dztady;
	    double dphidz = ubasis->dphidxi[i]*ubasis->dxidz
	      +ubasis->dphideta[i]*ubasis->detadz
	      +ubasis->dphidzta[i]*ubasis->dztadz;

	    if (nonnull(f_out)) {
	      //double x = ubasis->xx;
	      //double y = ubasis->yy;
	      double delta = dx;	      

	      double ut = (ubasis->uu-ubasis->uuold)/dt_*ubasis->phi[i];
	      double divgradu = K_*ubasis->dudx*dphidx + K_*ubasis->dudy*dphidy + K_*ubasis->dudz*dphidz;//(grad u,grad phi)
	      double divgradu_old = K_*ubasis->duolddx*dphidx + K_*ubasis->duolddy*dphidy + K_*ubasis->duolddz*dphidz;//(grad u,grad phi)

	      double hp2 = hp2_(1.);	

	      double phitu = -hp2*(phibasis->uu-phibasis->uuold)/dt_*ubasis->phi[i]; 
	      double phitu2 = -hp2*(phibasis->uuold-phibasis2->uu)/dt_*ubasis->phi[i]; 
    
	      double val = ubasis->jac * ubasis->wt * (ut + t_theta_*divgradu + (1.-t_theta_)*divgradu_old + t_theta_*phitu 
						       + (1.-t_theta_)*phitu2);
	      f->SumIntoGlobalValues ((int) 1, &val, &row);

	      double dphiphidx = phibasis->dudx;
	      double dphiphidy = phibasis->dudy;
	      double dphiphidz = phibasis->dudz;
	      double theta_ = theta(dphiphidx,dphiphidy,dphiphidz)-theta_0_;

	      double gs2_ = gs2(theta_);

	      double m = m_(theta_, M_, eps_);
     
	      double phit = m*(phibasis->uu-phibasis->uuold)/dt_*phibasis->phi[i];

	      double divgradphi = gs2_*phibasis->dudx*dphidx + gs2_*phibasis->dudy*dphidy + gs2_*phibasis->dudz*dphidz;//(grad u,grad phi)

	      double dg2 = dgs2_2dtheta(theta_);	

	      double curlgrad = -dg2*(phibasis->dudy*dphidx -phibasis->dudx*dphidy);//cn not sure about 3d yet
	      //curlgrad = -dg2*(phibasis->dudy*dphidx -phibasis->dudx*dphidy -phibasis->dudz*dphidz);

	      double w = w_(delta);
	      //double gp1 = phibasis->uu*(1.-phibasis->uu)*(1.-2.*phibasis->uu);
	      double gp1 = gp1_(phibasis->uu);

	      //double phidel2 = phibasis->uu*(1.-phibasis->uu)*(1.-2.*phibasis->uu)/delta/delta*phibasis->phi[i];
	      double phidel2 = gp1*w*phibasis->phi[i];

// 	      double phidel = -5.*alpha_*(T_m_ - ubasis->uu)
// 		*phibasis->uu*phibasis->uu*(1.-phibasis->uu)*(1.-phibasis->uu)/delta*phibasis->phi[i];

	      double hp1 = hp1_(phibasis->uu,5.*alpha_/delta);

	      double phidel = hp1*(T_m_ - ubasis->uu)*phibasis->phi[i];
	      
	      double rhs = divgradphi + curlgrad + phidel2 + phidel;

	      dphiphidx = phibasis->duolddx;
	      dphiphidy = phibasis->duolddy;
	      dphiphidz = phibasis->duolddz;
	      theta_ = theta(dphiphidx,dphiphidy,dphiphidz)-theta_0_;
	      gs2_ = gs2(theta_);
	      divgradphi = gs2_*phibasis->duolddx*dphidx + gs2_*phibasis->duolddy*dphidy + gs2_*phibasis->duolddz*dphidz;//(grad u,grad phi)
	      dg2 = dgs2_2dtheta(theta_);

	      curlgrad = -dg2*(phibasis->duolddy*dphidx -phibasis->duolddx*dphidy);//cn not sure about 3d yet
	      //curlgrad = -dg2*(phibasis->duolddy*dphidx -phibasis->duolddx*dphidy -phibasis->duolddz*dphidz);

	      gp1 = gp1_(phibasis->uuold);

	      //phidel2 = phibasis->uuold*(1.-phibasis->uuold)*(1.-2.*phibasis->uuold)/delta/delta*phibasis->phi[i];
	      phidel2 = gp1*w*phibasis->phi[i];

	      hp1 = hp1_(phibasis->uuold,5.*alpha_/delta);

// 	      phidel = -5.*alpha_*(T_m_ - ubasis->uuold)
// 		*phibasis->uuold*phibasis->uuold*(1.-phibasis->uuold)*(1.-phibasis->uuold)/delta*phibasis->phi[i];
	      phidel = hp1*(T_m_ - ubasis->uuold)*phibasis->phi[i];

	      double rhs_old = divgradphi + curlgrad + phidel2 + phidel;

	      double rand_phi = rand_phi_(phibasis->uu);
	      double r_phi = rand_phi*phibasis->phi[i];
	
	      val = phibasis->jac * phibasis->wt * (phit + t_theta_*rhs + (1.-t_theta_)*rhs_old + r_phi);
	      int row1 = row+1;
	      f->SumIntoGlobalValues ((int) 1, &val, &row1);
	    }


	    // Loop over Trial Functions


	    //cn add the phitu term here
	    if (nonnull(W_prec_out)) {
	      for(int j=0;j < n_nodes_per_elem; j++) {
		int column = numeqs_*(mesh_->get_node_id(blk, ne, j));
		double dtestdx = K_*ubasis->dphidxi[j]*ubasis->dxidx
		  +K_*ubasis->dphideta[j]*ubasis->detadx
		  +K_*ubasis->dphidzta[j]*ubasis->dztadx;
		double dtestdy = K_*ubasis->dphidxi[j]*ubasis->dxidy
		  +K_*ubasis->dphideta[j]*ubasis->detady
		  +K_*ubasis->dphidzta[j]*ubasis->dztady;
		double dtestdz = K_*ubasis->dphidxi[j]*ubasis->dxidz
		  +K_*ubasis->dphideta[j]*ubasis->detadz
		  +K_*ubasis->dphidzta[j]*ubasis->dztadz;
		double divgrad = dtestdx * dphidx + dtestdy * dphidy + dtestdz * dphidz;
		double phi_t = ubasis->phi[i] * ubasis->phi[j]/dt_;
		double jac = ubasis->jac*ubasis->wt*(phi_t + t_theta_*divgrad);
		//std::cout<<row<<" "<<column<<" "<<jac<<std::endl;
		P_->SumIntoGlobalValues(row, 1, &jac, &column);

		int row1 = row+1;
		int column1 = column+1;
		double dphiphidx = phibasis->dudx;
		double dphiphidy = phibasis->dudy;
		double dphiphidz = phibasis->dudz;
		
		double theta_ = theta(dphiphidx,dphiphidy,dphiphidz) - theta_0_;
		double gs2_ = gs2(theta_);
		dtestdx = phibasis->dphidxi[j]*phibasis->dxidx
		  +phibasis->dphideta[j]*phibasis->detadx
		  +phibasis->dphidzta[j]*phibasis->dztadx;
		dtestdy = phibasis->dphidxi[j]*phibasis->dxidy
		  +phibasis->dphideta[j]*phibasis->detady
		  +phibasis->dphidzta[j]*phibasis->dztady;
		dtestdz = phibasis->dphidxi[j]*phibasis->dxidz
		  +phibasis->dphideta[j]*phibasis->detadz
		  +phibasis->dphidzta[j]*phibasis->dztadz;
		divgrad = gs2_*dtestdx * dphidx + gs2_*dtestdy * dphidy + gs2_*dtestdz * dphidz;

		double m = m_(theta_,M_,eps_);

		phi_t = m*phibasis->phi[i] * phibasis->phi[j]/dt_;
		jac = phibasis->jac*phibasis->wt*(phi_t + t_theta_*divgrad);
		P_->SumIntoGlobalValues(row1, 1, &jac, &column1);
	      }//j
	    }

	  }//i
	}//gp
      }//ne

#if 0
      if (nonnull(f_out)) {//cn double check the use of notnull throughout
	if(paramList.get<std::string> (TusastestNameString)=="pool"){
	  for ( int j = 0; j < mesh_->get_node_set(1).size(); j++ ){
	    
	    int row = numeqs_*(mesh_->get_node_set_entry(0, j));
	    
	    (*f)[row] =
	      (*u)[row] - T_inf_; // Dirichlet BC of zero
	    (*f)[row+1] =
	      (*u)[row+1] - 0.0;
	    
	  }
	}
      }
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

    delete xx, yy, zz, uu, uu_old, phiphi, phiphi_old, phiphi_old_old;
    delete ubasis, phibasis, phibasis2;
    
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
ModelEvaluatorPHASE_HEAT<Scalar>::~ModelEvaluatorPHASE_HEAT()
{
  //  if(!prec_.is_null()) prec_ = Teuchos::null;
}
template<class Scalar>
void ModelEvaluatorPHASE_HEAT<Scalar>::init_nox()
{
  ::Stratimikos::DefaultLinearSolverBuilder builder;

  Teuchos::RCP<Teuchos::ParameterList> lsparams =
    Teuchos::rcp(new Teuchos::ParameterList);
#if 0
  lsparams->set("Linear Solver Type", "Belos");
  lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Pseudo Block GMRES").set("Num Blocks",1);
  lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Pseudo Block GMRES").set("Maximum Restarts",200);
  lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Psuedo Block GMRES").set("Output Frequency",1);
#else
  lsparams->set("Linear Solver Type", "AztecOO");
  lsparams->sublist("Linear Solver Types").sublist("AztecOO").sublist("Forward Solve").sublist("AztecOO Settings").set("Output Frequency",1);
  //lsparams->sublist("Linear Solver Types").sublist("AztecOO").sublist("Forward Solve").sublist("AztecOO Settings").sublist("AztecOO Preconditioner", "None");
#endif
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


  bool precon = paramList.get<bool> (TusaspreconNameString);
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
  Teuchos::ParameterList& searchParams = nl_params->sublist("Line Search");
  //searchParams.set("Method", "Full Step");
  //searchParams.set("Method", "Interval Halving");
  //searchParams.set("Method", "Polynomial");
  //searchParams.set("Method", "Backtrack");
  //searchParams.set("Method", "NonlinearCG");
  //searchParams.set("Method", "Quadratic");
  //searchParams.set("Method", "More'-Thuente");
  
  Teuchos::ParameterList& btParams = nl_params->sublist("Backtrack");
  btParams.set("Default Step",1.0);
  btParams.set("Max Iters",20);
  btParams.set("Minimum Step",1e-6);
  btParams.set("Recovery Step",1e-3);
	    
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
  //nl_params->sublist("Direction").sublist("Newton").sublist("Linear Solver").sublist("Output").set("Total Number of Linear Iterations",0);
  // Create the solver
  solver_ =  NOX::Solver::buildSolver(nox_group, combo, nl_params);
  std::cout<<"init_nox() completed."<<std::endl<<std::endl;
}


template<class Scalar>
void ModelEvaluatorPHASE_HEAT<Scalar>::advance()
{
  Teuchos::RCP< VectorBase< double > > guess = Thyra::create_Vector(u_old_,x_space_);
  NOX::Thyra::Vector thyraguess(*guess);//by sending the dereferenced pointer, we instigate a copy rather than a view
  solver_->reset(thyraguess);

  NOX::StatusTest::StatusType solvStatus = solver_->solve();
  if( !(NOX::StatusTest::Converged == solvStatus)) {
    std::cout<<" NOX solver failed to converge. Status = "<<solvStatus<<std::endl<<std::endl;
    exit(0);
  }

  nnewt_ += solver_->getNumIterations();

  //std::cout<<solver_->getList();
  //std::cout<<std::endl;
  //std::cout<<*(W_factory_->getParameterList ());

  const Thyra::VectorBase<double> * sol = 
    &(dynamic_cast<const NOX::Thyra::Vector&>(
					      solver_->getSolutionGroup().getX()
					      ).getThyraVector()
      );
  //u_old_ = get_Epetra_Vector (*f_owned_map_,Teuchos::rcp_const_cast< ::Thyra::VectorBase< double > >(Teuchos::rcpFromRef(*sol))	);

  Thyra::ConstDetachedSpmdVectorView<double> x_vec(sol->col(0));

  *u_old_old_ = *u_old_;

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
void ModelEvaluatorPHASE_HEAT<Scalar>::initialize()
{
  if(paramList.get<std::string> (TusastestNameString)=="cummins"){
    init(u_old_);
  }else if(paramList.get<std::string> (TusastestNameString)=="multi"){
    multi(u_old_);
  }else if(paramList.get<std::string> (TusastestNameString)=="pool"){
    pool(u_old_);
  }else{
    std::cout<<"Unknown initialization testcase."<<std::endl;
    exit(0);
  }
  *u_old_old_ = *u_old_;
}

template<class Scalar>
void ModelEvaluatorPHASE_HEAT<Scalar>::finalize()
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

  std::cout<<(solver_->getList()).sublist("Direction").sublist("Newton").sublist("Linear Solver")<<std::endl;
  int ngmres = 0;

  if ( (solver_->getList()).sublist("Direction").sublist("Newton").sublist("Linear Solver")
       .getEntryPtr("Total Number of Linear Iterations") != NULL)
    ngmres = ((solver_->getList()).sublist("Direction").sublist("Newton").sublist("Linear Solver")
	      .getEntry("Total Number of Linear Iterations")).getValue(&ngmres);

  std::cout<<std::endl
	   <<"Total number of Newton iterations:     "<<nnewt_<<std::endl
	   <<"Total number of GMRES iterations:      "<<ngmres<<std::endl 
	   <<"Total number of Timesteps:             "<<paramList.get<int> (TusasntNameString)<<std::endl
	   <<"Average number of Newton per Timestep: "<<(float)nnewt_/(float)(paramList.get<int> (TusasntNameString))<<std::endl
	   <<"Average number of GMRES per Newton:    "<<(float)ngmres/(float)nnewt_<<std::endl
	   <<"Average number of GMRES per Timestep:  "<<(float)ngmres/(float)(paramList.get<int> (TusasntNameString))<<std::endl;
  
  std::ofstream outfile;
  outfile.open("jfnk.dat");
  outfile 
	   <<"Total number of Newton iterations:     "<<nnewt_<<std::endl
	   <<"Total number of GMRES iterations:      "<<ngmres<<std::endl 
	   <<"Total number of Timesteps:             "<<paramList.get<int> (TusasntNameString)<<std::endl
	   <<"Average number of Newton per Timestep: "<<(float)nnewt_/(float)(paramList.get<int> (TusasntNameString))<<std::endl
	   <<"Average number of GMRES per Newton:    "<<(float)ngmres/(float)nnewt_<<std::endl
	   <<"Average number of GMRES per Timestep:  "<<(float)ngmres/(float)(paramList.get<int> (TusasntNameString))<<std::endl; 	
  outfile.close();

  std::ofstream timefile;
  timefile.open("time.dat");
  Teuchos::TimeMonitor::summarize(timefile);

  if(!x_space_.is_null()) x_space_=Teuchos::null;
  if(!x_owned_map_.is_null()) x_owned_map_=Teuchos::null;
  if(!f_owned_map_.is_null()) f_owned_map_=Teuchos::null;
  if(!W_graph_.is_null()) W_graph_=Teuchos::null;
  if(!W_factory_.is_null()) W_factory_=Teuchos::null;
  if(!x0_.is_null()) x0_=Teuchos::null;
  if(!P_.is_null())  P_=Teuchos::null;
  if(!prec_.is_null()) prec_=Teuchos::null;
  if(!solver_.is_null()) solver_=Teuchos::null;
  if(!u_old_.is_null()) u_old_=Teuchos::null;
  if(!dudt_.is_null()) dudt_=Teuchos::null;
}

template<class Scalar>
void ModelEvaluatorPHASE_HEAT<Scalar>::compute_error( double *u)
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

    case 8 : // linear hex
      ubasis = new BasisLHex;
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
	
	//for (int i=0; i< n_nodes_per_elem; i++) { = mesh_->get_node_id(blk, ne, i);
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
const double ModelEvaluatorPHASE_HEAT<Scalar>::gs( const double &theta)
{
  return 1. + eps_ * (M_*cos(theta));
}
template<class Scalar>
double ModelEvaluatorPHASE_HEAT<Scalar>::gs2( const double &theta) const
{ 
  //double g = 1. + eps_ * (M_*cos(theta));
  double g = 1. + eps_ * (cos(M_*(theta)));
  return g*g;
}
template<class Scalar>
double ModelEvaluatorPHASE_HEAT<Scalar>::dgs2_2dtheta(const double &theta) const
{
  //return -1.*(eps_*M_*(1. + eps_*M_*cos(theta))*sin(theta));
  return -1.*(eps_*M_*(1. + eps_*cos(M_*(theta)))*sin(M_*(theta)));
}
template<class Scalar>
const double ModelEvaluatorPHASE_HEAT<Scalar>::R(const double &theta)
{
  return .3*(1. + eps_ * cos(M_*(theta)));
}

template<class Scalar>
double ModelEvaluatorPHASE_HEAT<Scalar>::theta(double &x,double &y,double &z) const
{
  double small = 1e-9;
  double pi = 3.141592653589793;
  double t = 0.;
  double sy = 1.;
  if(y < 0.) sy = -1.;
  double n = sy*sqrt(y*y+z*z);
  //double n = y;
  //std::cout<<y<<"   "<<n<<std::endl;
//   if(abs(x) < small && y > 0. ) t = pi/2.;
//   else if(abs(x) < small && y < 0. ) t = 3.*pi/2.;
//   else t= atan(n/x);
  if(abs(x) < small && y > 0. ) t = pi/2.;
  if(abs(x) < small && y < 0. ) t = 3.*pi/2.;
  if(x > small && y >= 0.) t= atan(n/x);
  if(x > small && y <0.) t= atan(n/x) + 2.*pi;
  if(x < -small) t= atan(n/x)+ pi;

  return t;
}

template<class Scalar>
void ModelEvaluatorPHASE_HEAT<Scalar>::init(Teuchos::RCP<Epetra_Vector> u)
{
  for (int nn=0; nn < mesh_->get_num_nodes(); nn++) {
    double x = mesh_->get_x(nn);
    double y = mesh_->get_y(nn);
    double z = mesh_->get_z(nn);
    double t = theta(x,y,z);
    double r = R(t);
    if(x*x+y*y+z*z < r*r){
      (*u)[numeqs_*nn]=T_m_;
      (*u)[numeqs_*nn+1]=1.;
    }
    else {
      (*u)[numeqs_*nn]=T_inf_;
      (*u)[numeqs_*nn+1]=0.;
    }
    

    //std::cout<<nn<<" "<<x<<" "<<y<<" "<<r<<"      "<<(*u_old_)[numeqs_*nn]<<"           "<<x*x+y*y<<" "<<r*r<<std::endl;
  }
}

template<class Scalar>
void ModelEvaluatorPHASE_HEAT<Scalar>::multi(Teuchos::RCP<Epetra_Vector> u)
{
  for (int nn=0; nn < mesh_->get_num_nodes(); nn++) {
    double x = mesh_->get_x(nn);
    double y = mesh_->get_y(nn);
    double pi = 3.141592653589793;
    //double z = mesh_->get_z(nn);
    //double t = theta(x,y,z);
    srand(123);
    double r1 = rand()%15;
    double r2 = rand()%9+1;
    double r3 = rand()%15;
    double r4 = rand()%9+1;
    double r5 = rand()%15;
    double r6 = rand()%9+1;

    double r7 = .5*(sin(r1*y/r2)*cos(r3*y/r4+r2)*sin(r5*y/r6+r3)+1.);
    double r = r7*.3*fabs(sin(15.* pi* y/9.));
    if(x < r){
      (*u)[numeqs_*nn]=T_m_;
      (*u)[numeqs_*nn+1]=1.;
    }
    else {
      (*u)[numeqs_*nn]=T_inf_;
      (*u)[numeqs_*nn+1]=0.;
    }
    

    //std::cout<<nn<<" "<<x<<" "<<y<<" "<<r<<"      "<<(*u_old_)[numeqs_*nn]<<"           "<<x*x+y*y<<" "<<r*r<<std::endl;
  }
}
template<class Scalar>
void ModelEvaluatorPHASE_HEAT<Scalar>::pool(Teuchos::RCP<Epetra_Vector> u)
{
  for (int nn=0; nn < mesh_->get_num_nodes(); nn++) {
    double x = mesh_->get_x(nn);
    double y = mesh_->get_y(nn);
    double pi = 3.141592653589793;
    //double z = mesh_->get_z(nn);
    //double t = theta(x,y,z);
    srand(123);
    double r1 = rand()%15;
    double r2 = rand()%9+1;
    double r3 = rand()%15;
    double r4 = rand()%9+1;
    double r5 = rand()%15;
    double r6 = rand()%9+1;

    double r7 = .5*(sin(r1*y/r2)*cos(r3*y/r4+r2)*sin(r5*y/r6+r3)+1.);

    double rr= 4.5;

    double r = r7*.3*fabs(sin(24.* pi* y/14.)) - sqrt(rr*rr-y*y);
    if(x < r){
      (*u)[numeqs_*nn]=T_m_;
      (*u)[numeqs_*nn+1]=1.;
    }
    else {
      (*u)[numeqs_*nn]=T_inf_;
      (*u)[numeqs_*nn+1]=0.;
    }
    

    //std::cout<<nn<<" "<<x<<" "<<y<<" "<<r<<"      "<<(*u_old_)[numeqs_*nn]<<"           "<<x*x+y*y<<" "<<r*r<<std::endl;
  }
}
#endif
