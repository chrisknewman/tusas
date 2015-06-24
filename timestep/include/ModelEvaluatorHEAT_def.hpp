#ifndef NOX_THYRA_MODEL_EVALUATOR_HEAT_DEF_HPP
#define NOX_THYRA_MODEL_EVALUATOR_HEAT_DEF_HPP

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
#include "ParamNames.h"
// Nonmember constuctors

template<class Scalar>
Teuchos::RCP<ModelEvaluatorHEAT<Scalar> >
modelEvaluatorHEAT(const Teuchos::RCP<const Epetra_Comm>& comm,
            Mesh *mesh,
			 Teuchos::ParameterList plist)
{
  return Teuchos::rcp(new ModelEvaluatorHEAT<Scalar>(comm,mesh,plist));
}

// Constructor

template<class Scalar>
ModelEvaluatorHEAT<Scalar>::
ModelEvaluatorHEAT(const Teuchos::RCP<const Epetra_Comm>& comm,
            Mesh *mesh,
			 Teuchos::ParameterList plist) :
  comm_(comm),
  paramList(plist),
  mesh_(mesh),
  showGetInvalidArg_(false)
{
  dt_ = paramList.get<double> (TusasdtNameString);
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
  x_owned_map_ = rcp(new Epetra_Map(num_nodes,0,*comm_));
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
  prec_ = Teuchos::rcp(new preconditioner<Scalar>(P_, comm_, paramList.sublist("ML")));
  u_old_ = rcp(new Epetra_Vector(*f_owned_map_));

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
}

// Initializers/Accessors

template<class Scalar>
Teuchos::RCP<Epetra_CrsGraph>
ModelEvaluatorHEAT<Scalar>::createGraph()
{
  Teuchos::RCP<Epetra_CrsGraph> W_graph;

  // Create the shell for the
  W_graph = Teuchos::rcp(new Epetra_CrsGraph(Copy, *x_owned_map_, 5));

  //nodes are numbered consecutively by nodeid
  for (int i=0; i < mesh_->get_num_nodes(); i++) {
    std::vector<int> column (mesh_->get_nodal_adj(i));

    column.push_back(i);//cn put the diagonal in
    //cn need something here for more than one pde (see 2dstokes ex)
    W_graph->InsertGlobalIndices(i, column.size(), &column[0]);


  }//i

  W_graph->FillComplete();
  //W_graph->Print(std::cout);
  //exit(0);
  return W_graph;
}

template<class Scalar>
void ModelEvaluatorHEAT<Scalar>::set_x0(const Teuchos::ArrayView<const Scalar> &x0_in)
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_EQUALITY(x_space_->dim(), x0_in.size());
#endif
  Thyra::DetachedVectorView<Scalar> x0(x0_);
  x0.sv().values()().assign(x0_in);
}


template<class Scalar>
void ModelEvaluatorHEAT<Scalar>::setShowGetInvalidArgs(bool showGetInvalidArg)
{
  showGetInvalidArg_ = showGetInvalidArg;
}

template<class Scalar>
void ModelEvaluatorHEAT<Scalar>::
set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory)
{
  W_factory_ = W_factory;
}

// Public functions overridden from ModelEvaulator


template<class Scalar>
Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
ModelEvaluatorHEAT<Scalar>::get_x_space() const
{
  return x_space_;
}


template<class Scalar>
Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
ModelEvaluatorHEAT<Scalar>::get_f_space() const
{
  return f_space_;
}


template<class Scalar>
Thyra::ModelEvaluatorBase::InArgs<Scalar>
ModelEvaluatorHEAT<Scalar>::getNominalValues() const
{
  return nominalValues_;
}


template<class Scalar>
Teuchos::RCP<Thyra::LinearOpBase<Scalar> >
ModelEvaluatorHEAT<Scalar>::create_W_op() const
{
  Teuchos::RCP<Epetra_CrsMatrix> W_epetra =
    Teuchos::rcp(new Epetra_CrsMatrix(::Copy,*W_graph_));

  return Thyra::nonconstEpetraLinearOp(W_epetra);
}

template<class Scalar>
Teuchos::RCP< ::Thyra::PreconditionerBase<Scalar> >
ModelEvaluatorHEAT<Scalar>::create_W_prec() const
{

  const Teuchos::RCP<Thyra::LinearOpBase< Scalar > > P_op = prec_;

  Teuchos::RCP<Thyra::DefaultPreconditioner<Scalar> > prec =
    Teuchos::rcp(new Thyra::DefaultPreconditioner<Scalar>(Teuchos::null,P_op));

  return prec;
//  return Teuchos::null;
}

template<class Scalar>
Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> >
ModelEvaluatorHEAT<Scalar>::get_W_factory() const
{
  return W_factory_;
}


template<class Scalar>
Thyra::ModelEvaluatorBase::InArgs<Scalar>
ModelEvaluatorHEAT<Scalar>::createInArgs() const
{
  return prototypeInArgs_;
}


// Private functions overridden from ModelEvaulatorDefaultBase


template<class Scalar>
Thyra::ModelEvaluatorBase::OutArgs<Scalar>
ModelEvaluatorHEAT<Scalar>::createOutArgsImpl() const
{
  return prototypeOutArgs_;
}


template<class Scalar>
void ModelEvaluatorHEAT<Scalar>::evalModelImpl(
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
    double *uu, *uu_old;
    int n_nodes_per_elem;

    Basis *ubasis;

    int dim = mesh_->get_num_dim();

    for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){

      n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);
      std::string elem_type=mesh_->get_blk_elem_type(blk);

      if( (0==elem_type.compare("QUAD4")) || (0==elem_type.compare("QUAD")) ){ // linear quad
	ubasis = new BasisLQuad;
      }
      else if( (0==elem_type.compare("TRI3")) || (0==elem_type.compare("TRI")) ){ // linear triangle
	ubasis = new BasisLTri;
      }
      else if( (0==elem_type.compare("HEX8")) || (0==elem_type.compare("HEX")) ){ // linear hex
	ubasis = new BasisLHex;
      } 
      else if( (0==elem_type.compare("TETRA4")) || (0==elem_type.compare("TETRA")) ){ // linear tet
 	ubasis = new BasisLTet;
      } 
      else {
	std::cout<<"Unsupported element type"<<std::endl<<std::endl;
	exit(0);
      }

      xx = new double[n_nodes_per_elem];
      yy = new double[n_nodes_per_elem];
      zz = new double[n_nodes_per_elem];
      uu = new double[n_nodes_per_elem];
      uu_old = new double[n_nodes_per_elem];

      for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {// Loop Over # of Finite Elements on Processor

	for(int k = 0; k < n_nodes_per_elem; k++){
	  
	  int nodeid = mesh_->get_node_id(blk, ne, k);
	  
	  xx[k] = mesh_->get_x(nodeid);
	  yy[k] = mesh_->get_y(nodeid);
	  zz[k] = mesh_->get_z(nodeid);
	  //uu[k] = u[nodeid]; 
	  uu[k] = (*u)[nodeid];  // copy initial guess 
	                      //or old solution into local temp
	  uu_old[k] = (*u_old_)[nodeid];
	  
	}//k
	for(int gp=0; gp < ubasis->ngp; gp++) {// Loop Over Gauss Points 

	  // Calculate the basis function at the gauss point

 	  if(3 == dim) {
 	    ubasis->getBasis(gp, xx, yy, zz, uu, uu_old);
	  }else{
	    ubasis->getBasis(gp, xx, yy, uu, uu_old);
	  }
	  // Loop over Nodes in Element

	  for (int i=0; i< n_nodes_per_elem; i++) {
	    int row = mesh_->get_node_id(blk, ne, i);

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
	      double x = ubasis->xx;
	      double y = ubasis->yy;
	      
	      double divgradu = ubasis->dudx*dphidx + ubasis->dudy*dphidy + ubasis->dudz*dphidz;//(grad u,grad phi)
	      double ut = (ubasis->uu-ubasis->uuold)/dt_*ubasis->phi[i];
	      double pi = 3.141592653589793;
	      double ff = 2.*ubasis->phi[i];
	      //double ff = ((1. + 5.*dt_*pi*pi)*sin(pi*x)*sin(2.*pi*y)/dt_)*ubasis->phi[i];	      
	      double val = ubasis->jac * ubasis->wt * (ut + divgradu - ff);	      
	      //double val = ubasis->jac * ubasis->wt * (ut + divgradu);
	      f->SumIntoGlobalValues ((int) 1, &val, &row);
	    }


	    // Loop over Trial Functions
	    if (nonnull(W_prec_out)) {
	      for(int j=0;j < n_nodes_per_elem; j++) {
		int column = mesh_->get_node_id(blk, ne, j);
		double dtestdx = ubasis->dphidxi[j]*ubasis->dxidx
		  +ubasis->dphideta[j]*ubasis->detadx
		  +ubasis->dphidzta[j]*ubasis->dztadx;
		double dtestdy = ubasis->dphidxi[j]*ubasis->dxidy
		  +ubasis->dphideta[j]*ubasis->detady
		  +ubasis->dphidzta[j]*ubasis->dztady;
		double dtestdz = ubasis->dphidxi[j]*ubasis->dxidz
		  +ubasis->dphideta[j]*ubasis->detadz
		  +ubasis->dphidzta[j]*ubasis->dztadz;
		double divgrad = dtestdx * dphidx + dtestdy * dphidy + dtestdz * dphidz;
		double phi_t = ubasis->phi[i] * ubasis->phi[j]/dt_;
		double jac = ubasis->jac*ubasis->wt*(phi_t + divgrad);
		//std::cout<<row<<" "<<column<<" "<<jac<<std::endl;
		P_->SumIntoGlobalValues(row, 1, &jac, &column);
	      }//j
	    }

	  }//i
	}//gp
      }//ne

      if (nonnull(f_out)) {//cn double check the use of notnull throughout
	for ( int j = 0; j < mesh_->get_node_set(1).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(1, j);
	  
	  (*f)[row] =
	    (*u)[row] - 0.0; // Dirichlet BC of zero
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(2).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(2, j);
	  
	  (*f)[row] =
	    (*u)[row] - 0.0; // Dirichlet BC of zero
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(3).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(3, j);
	  
	  (*f)[row] =
	    (*u)[row] - 0.0; // Dirichlet BC of zero
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(0).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(0, j);
	  
	  (*f)[row] =
	    (*u)[row] - 0.0; // Dirichlet BC of zero
	  
	}
	
      }
      if (nonnull(W_prec_out)) {
	for ( int j = 0; j < mesh_->get_node_set(1).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(1, j);
	  //clear row and put 1 on diagonal
	  std::vector<int> column (mesh_->get_nodal_adj(row));
	  std::vector<double> vals (column.size(),0.);
	  column.push_back(row);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row, vals.size(), &vals[0],&column[0] );
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(2).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(2, j);
	  
	  std::vector<int> column (mesh_->get_nodal_adj(row));
	  std::vector<double> vals (column.size(),0.);
	  column.push_back(row);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row, vals.size(), &vals[0],&column[0] );
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(3).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(3, j);
	  
	  std::vector<int> column (mesh_->get_nodal_adj(row));
	  std::vector<double> vals (column.size(),0.);
	  column.push_back(row);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row, vals.size(), &vals[0],&column[0] );
	  
	}
	for ( int j = 0; j < mesh_->get_node_set(0).size(); j++ ){
	  
	  int row = mesh_->get_node_set_entry(0, j);
	  
	  std::vector<int> column (mesh_->get_nodal_adj(row));
	  std::vector<double> vals (column.size(),0.);
	  column.push_back(row);
	  vals.push_back(1.);
	  P_->ReplaceGlobalValues (row, vals.size(), &vals[0],&column[0] );
	  
	}
      
      }

    }//blk

    delete xx, yy, uu;
    delete ubasis;
    
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
ModelEvaluatorHEAT<Scalar>::~ModelEvaluatorHEAT()
{
  //  if(!prec_.is_null()) prec_ = Teuchos::null;
}
template<class Scalar>
void ModelEvaluatorHEAT<Scalar>::init_nox()
{
  ::Stratimikos::DefaultLinearSolverBuilder builder;

  Teuchos::RCP<Teuchos::ParameterList> lsparams =
    Teuchos::rcp(new Teuchos::ParameterList);
#if 0
  lsparams->set("Linear Solver Type", "Belos");
  lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Pseudo Block GMRES").set("Num Blocks",1);
  lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Pseudo Block GMRES").set("Maximum Restarts",200);
  //lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Psuedo Block GMRES").set("Output Frequency",1);
#else
  lsparams->set("Linear Solver Type", "AztecOO");
  lsparams->sublist("Linear Solver Types").sublist("AztecOO").sublist("Forward Solve").sublist("AztecOO Settings").set("Output Frequency",1);
  //lsparams->sublist("Linear Solver Types").sublist("AztecOO").sublist("Forward Solve").sublist("AztecOO Settings").sublist("AztecOO Preconditioner", "None");
#endif
  lsparams->set("Preconditioner Type", "None");
  builder.setParameterList(lsparams);
  lsparams->print(std::cout);
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
    Teuchos::rcp(new NOX::StatusTest::NormF(*nox_group.get(), 1.0e-8));//1.0e-6 for paper
  Teuchos::RCP<NOX::StatusTest::NormWRMS> wrms =
    Teuchos::rcp(new NOX::StatusTest::NormWRMS(1.0e-2, 1.0e-8));
  Teuchos::RCP<NOX::StatusTest::Combo> converged =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::AND));
  //converged->addStatusTest(absresid);
  converged->addStatusTest(relresid);
  //converged->addStatusTest(wrms);
  Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters =
    Teuchos::rcp(new NOX::StatusTest::MaxIters(20));
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
  //nl_params->sublist("Direction").sublist("Newton").sublist("Linear Solver").set("Tolerance", 1.0e-4);
  Teuchos::ParameterList& nlPrintParams = nl_params->sublist("Printing");
  nlPrintParams.set("Output Information",
		  NOX::Utils::OuterIteration  +
		  //                      NOX::Utils::OuterIterationStatusTest +
		  NOX::Utils::InnerIteration +
		  NOX::Utils::Details +
		  NOX::Utils::LinearSolverDetails);
  nl_params->set("Forcing Term Method", "Type 2");
  nl_params->set("Forcing Term Initial Tolerance", 1.0e-1);
  nl_params->set("Forcing Term Maximum Tolerance", 1.0e-2);
  nl_params->set("Forcing Term Minimum Tolerance", 1.0e-5);//1.0e-6
  // Create the solver
  solver_ =  NOX::Solver::buildSolver(nox_group, combo, nl_params);
}


template<class Scalar>
void ModelEvaluatorHEAT<Scalar>::advance()
{
  Teuchos::RCP< VectorBase< double > > guess = Thyra::create_Vector(u_old_,x_space_);
  NOX::Thyra::Vector thyraguess(*guess);//by sending the dereferenced pointer, we instigate a copy rather than a view
  solver_->reset(thyraguess);

  NOX::StatusTest::StatusType solvStatus = solver_->solve();
  if( !(NOX::StatusTest::Converged == solvStatus)) {
    std::cout<<" NOX solver failed to converge. Status = "<<solvStatus<<std::endl<<std::endl;
    exit(0);
  }
  
  std::cout<<solver_->getList();
  const Thyra::VectorBase<double> * sol = 
    &(dynamic_cast<const NOX::Thyra::Vector&>(
					      solver_->getSolutionGroup().getX()
					      ).getThyraVector()
      );
  //u_old_ = get_Epetra_Vector (*f_owned_map_,Teuchos::rcp_const_cast< ::Thyra::VectorBase< double > >(Teuchos::rcpFromRef(*sol))	);

  Thyra::ConstDetachedSpmdVectorView<double> x_vec(sol->col(0));

  for (int nn=0; nn < mesh_->get_num_nodes(); nn++) {//cn figure out a better way here...
    (*u_old_)[nn]=x_vec[nn];
  }
  //u_old_->Print(std::cout);
  time_ +=dt_;
}

template<class Scalar>
void ModelEvaluatorHEAT<Scalar>::initialize()
{
  double pi = 3.141592653589793;
  for (int nn=0; nn < mesh_->get_num_nodes(); nn++) {
    double x = mesh_->get_x(nn);
    double y = mesh_->get_y(nn);
    (*u_old_)[nn]=sin(pi*x)*sin(pi*y);
    //std::cout<<nn<<" "<<x<<" "<<y<<std::endl;
  }
  //exit(0);
}

template<class Scalar>
void ModelEvaluatorHEAT<Scalar>::finalize()
{
  double outputdata[mesh_->get_num_nodes()];
  for (int nn=0; nn < mesh_->get_num_nodes(); nn++) {
    outputdata[nn]=(*u_old_)[nn];
    //std::cout<<nn<<" "<<outputdata[nn]<<" "<<std::endl;
  }


  //cout<<"norm = "<<sqrt(norm)<<endl;
  const char *outfilename = "results.e";
  mesh_->add_nodal_data("u", outputdata);
  mesh_->write_exodus(outfilename);
  compute_error(&outputdata[0]);
}

template<class Scalar>
void ModelEvaluatorHEAT<Scalar>::compute_error( double *u)
{
  double error = 0.;
  double jac;
  double *xx, *yy;
  double *uu;
  int n_nodes_per_elem, nodeid;
  
  Basis *ubasis;
  
  for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){
    
    n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);
    std::string elem_type=mesh_->get_blk_elem_type(blk);

    if( (0==elem_type.compare("QUAD4")) || (0==elem_type.compare("QUAD")) ){ // linear quad
      ubasis = new BasisLQuad;
    }
    else if( (0==elem_type.compare("TRI3")) || (0==elem_type.compare("TRI")) ){ // linear triangle
      ubasis = new BasisLTri;
    }
    else if( (0==elem_type.compare("HEX8")) || (0==elem_type.compare("HEX")) ){ // linear hex
      ubasis = new BasisLHex;
    } 
    else if( (0==elem_type.compare("TETRA4")) || (0==elem_type.compare("TETRA")) ){ // linear tet
      ubasis = new BasisLTet;
    } 
    else {
      std::cout<<"Unsupported element type"<<std::endl<<std::endl;
      exit(0);
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


#endif
