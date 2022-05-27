//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////


#ifndef NOX_THYRA_MODEL_EVALUATOR_NEMESIS_DEF_HPP
#define NOX_THYRA_MODEL_EVALUATOR_NEMESIS_DEF_HPP

// local support
#include "function_def.hpp"
#ifdef TUSAS_HAVE_CUDA
#else

#include "preconditioner.hpp"
#include "ParamNames.h"

#include "elem_color.h"

#include "interpfluxavg.h"
#include "interpflux.h"
#include "projection.h"

// Thyra support
//#include "Thyra_DefaultSpmdVectorSpace.hpp"
#include "Thyra_DefaultSerialDenseLinearOpWithSolveFactory.hpp"
#include "Thyra_DetachedMultiVectorView.hpp"
#include "Thyra_DetachedVectorView.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Thyra_PreconditionerBase.hpp"
#include "Thyra_MLPreconditionerFactory.hpp"
#include "Thyra_DetachedSpmdVectorView.hpp"
#include "Thyra_EpetraThyraWrappers.hpp"

// NOX support
#include "NOX_Thyra_MatrixFreeJacobianOperator.hpp"
#include "NOX_MatrixFree_ModelEvaluatorDecorator.hpp"
#include <NOX_Epetra_Vector.H>

// Epetra support
#include "Thyra_EpetraThyraWrappers.hpp"
#include "Thyra_get_Epetra_Operator.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_FEVector.h"
#include "Epetra_Import.h"
#include "Epetra_Export.h"
#include "Epetra_FECrsGraph.h"
#include "Epetra_FECrsMatrix.h"
#include "EpetraExt_VectorOut.h"

//teuchos support
#include <Teuchos_RCP.hpp>	
#include "Teuchos_ParameterList.hpp"
#include <Teuchos_TimeMonitor.hpp>
#include "Teuchos_Array.hpp"

#include <iomanip>
#include <iostream>
#include <string>

#ifdef TUSAS_COLOR_CPU
#include <omp.h>
#endif

#include <boost/ptr_container/ptr_vector.hpp>
#include <limits>

std::string getmypidstring(const int mypid, const int numproc);

// Nonmember constuctors
Basis* new_clone(Basis const& other){
  return other.clone();
}


template<class Scalar>
Teuchos::RCP<ModelEvaluatorNEMESIS<Scalar> >
modelEvaluatorNEMESIS(const Teuchos::RCP<const Epetra_Comm>& comm,
			 Mesh *mesh,
			 Teuchos::ParameterList plist
			 )
{
  return Teuchos::rcp(new ModelEvaluatorNEMESIS<Scalar>(comm,plist));
}

// Constructor

template<class Scalar>
ModelEvaluatorNEMESIS<Scalar>::
ModelEvaluatorNEMESIS(const Teuchos::RCP<const Epetra_Comm>& comm,
			 Mesh *mesh,
			 Teuchos::ParameterList plist 
			 ) :
  comm_(comm),
  paramList(plist),
  mesh_(mesh),
  showGetInvalidArg_(false)
{
  dt_ = paramList.get<double> (TusasdtNameString);
  dtold_ = dt_;
  t_theta_ = paramList.get<double> (TusasthetaNameString);
  t_theta2_ = 0.;

  set_test_case();

  using Teuchos::RCP;
  using Teuchos::rcp;
  using ::Thyra::VectorBase;
  typedef ::Thyra::ModelEvaluatorBase MEB;
  typedef Teuchos::ScalarTraits<Scalar> ST;

  TEUCHOS_ASSERT(nonnull(comm_));

  int mypid = comm_->MyPID();
  int numproc = comm_->NumProc();

  mesh_->compute_nodal_adj();
  
  std::vector<nemesis_lint_t> node_num_map(mesh_->get_node_num_map());

  //cn for the overlap space
  //cn all procs have all their nodes
  std::vector<nemesis_lint_t> my_global_nodes(numeqs_*node_num_map.size());
  
  for(int i = 0; i < node_num_map.size(); i++){
    
    for(int k = 0; k < numeqs_; k++ ){
      my_global_nodes[numeqs_*i+k] = numeqs_*node_num_map[i]+k;
    }
  }
  x_overlap_map_ = rcp(new Epetra_Map((nemesis_lint_t)-1,
				      my_global_nodes.size(),
				      &my_global_nodes[0],
				      0,
				      *comm_));
  
  //x_overlap_map_->Print(std::cout);

  //cn for the owned space we either copy the map in serial or
  //cn reduce it such that each proc owns unique nodes

  //cn it seems that create_onetoone_map does not have a 64 bit version...it is only implemented for blockmap
  if( 1 ==numproc ){
    x_owned_map_ = x_overlap_map_;
  }else{
#ifdef MESH_64
    x_owned_map_ = rcp(new Epetra_Map(Create_OneToOne_Map64(*x_overlap_map_)));
#else
    x_owned_map_ = rcp(new Epetra_Map(Epetra_Util::Create_OneToOne_Map(*x_overlap_map_)));
#endif
  }

  //x_owned_map_->Print(std::cout); 
  //exit(0);
  num_my_nodes_ = x_owned_map_->NumMyElements ()/numeqs_;
  num_nodes_ = x_overlap_map_->NumMyElements ()/numeqs_;

//   std::cout<< x_owned_map_->NumMyElements ()<<std::endl;
//   exit(0);

  x_space_ = ::Thyra::create_VectorSpace(x_owned_map_);
  
  
  importer_ = rcp(new Epetra_Import(*x_overlap_map_, *x_owned_map_));
  
  // residual space
  f_owned_map_ = x_owned_map_;
  f_space_ = x_space_;

  x0_ = ::Thyra::createMember(x_space_);
  V_S(x0_.ptr(), ST::zero());

  bool precon = paramList.get<bool> (TusaspreconNameString);
  if(precon){
    // Initialize the graph for W CrsMatrix object
    W_graph_ = createGraph();
    //W_graph_->Print(std::cout);
    P_ = rcp(new Epetra_FECrsMatrix(Copy,*W_graph_));
    prec_ = Teuchos::rcp(new preconditioner<Scalar>(P_, comm_, paramList.sublist("ML")));
  }

  u_old_ = rcp(new Epetra_Vector(*f_owned_map_));
  u_new_ = rcp(new Epetra_Vector(*f_owned_map_));
  u_old_old_ = rcp(new Epetra_Vector(*f_owned_map_));
  u_old_old_old_ = rcp(new Epetra_Vector(*f_owned_map_));
  dudt_ = rcp(new Epetra_Vector(*f_owned_map_));

//   if(paramList.get<bool> (TusasestimateTimestepNameString)){
//     Teuchos::ParameterList *atsList;
//     atsList = &paramList.sublist (TusasatslistNameString, false );
//     if(atsList->get<std::string> (TusasatstypeNameString) == "predictor corrector")
      pred_temp_ = rcp(new Epetra_Vector(*f_owned_map_));
//   }

  random_vector_ = rcp(new Epetra_Vector(*f_owned_map_));
  random_vector_->Random();
  random_vector_old_ = rcp(new Epetra_Vector(*f_owned_map_));

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

  ts_time_import= Teuchos::TimeMonitor::getNewTimer("Tusas: Total Import Time");
  ts_time_resfill= Teuchos::TimeMonitor::getNewTimer("Tusas: Total Residual Fill Time");
  ts_time_precfill= Teuchos::TimeMonitor::getNewTimer("Tusas: Total Preconditioner Fill Time");
  ts_time_nsolve= Teuchos::TimeMonitor::getNewTimer("Tusas: Total Nonlinear Solver Time");

#ifdef TUSAS_COLOR_CPU
  Elem_col = rcp(new elem_color(comm_,mesh_));
#endif

  init_nox();

  std::vector<int> indices = (Teuchos::getArrayFromStringParameter<int>(paramList,
								       TusaserrorestimatorNameString)).toVector();
  std::vector<int>::iterator it;
  for(it = indices.begin();it != indices.end(); ++it){
    //std::cout<<*it<<" "<<std::endl;
    int error_index = *it;
    Error_est.push_back(new error_estimator(comm_,mesh_,numeqs_,error_index));
  }

  //exit(0);

}

// Initializers/Accessors

template<class Scalar>
Teuchos::RCP<Epetra_FECrsGraph>
ModelEvaluatorNEMESIS<Scalar>::createGraph()
{
  Teuchos::RCP<Epetra_FECrsGraph> W_graph;

  int numind = 9*numeqs_;//this is an approximation 9 for lquad; 25 for qquad; 9*3 for lhex; 25*3 for qhex; 6 ltris ??, tets ??
                         //this was causing problems with clang
  if(3 == mesh_->get_num_dim() ) numind = 27*numeqs_;

  W_graph = Teuchos::rcp(new Epetra_FECrsGraph(Copy, *x_owned_map_, numind));

  for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){
    
    int n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);
    for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {
      for (int i=0; i< n_nodes_per_elem; i++) {
	nemesis_lint_t row = numeqs_*(
			   mesh_->get_global_node_id(mesh_->get_node_id(blk, ne, i))
			   ); 
	for(int j=0;j < n_nodes_per_elem; j++) {
	  nemesis_lint_t column = numeqs_*(mesh_->get_global_node_id(mesh_->get_node_id(blk, ne, j)));

	  for( int k = 0; k < numeqs_; k++ ){
	    nemesis_lint_t row1 = row + k;
	    nemesis_lint_t column1 = column + k;
	    
	    //W_graph->InsertGlobalIndices((int)1,&row1, (int)1, &column1);
	    W_graph->InsertGlobalIndices(row1, (int)1, &column1);

	  }
	}
      }
    }
  }
  //W_graph->FillComplete();
  if (W_graph->GlobalAssemble() != 0){
    std::cout<<"error W_graph->GlobalAssemble()"<<std::endl;
    exit(0);
  }
  //W_graph->Print(std::cout);
//   exit(0);
  return W_graph;
}

template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::set_x0(const Teuchos::ArrayView<const Scalar> &x0_in)
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_EQUALITY(x_space_->dim(), x0_in.size());
#endif
  Thyra::DetachedVectorView<Scalar> x0(x0_);
  x0.sv().values()().assign(x0_in);
}


template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::setShowGetInvalidArgs(bool showGetInvalidArg)
{
  showGetInvalidArg_ = showGetInvalidArg;
}

template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::
set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory)
{
  W_factory_ = W_factory;
}

// Public functions overridden from ModelEvaulator


template<class Scalar>
Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
ModelEvaluatorNEMESIS<Scalar>::get_x_space() const
{
  return x_space_;
}


template<class Scalar>
Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
ModelEvaluatorNEMESIS<Scalar>::get_f_space() const
{
  return f_space_;
}


template<class Scalar>
Thyra::ModelEvaluatorBase::InArgs<Scalar>
ModelEvaluatorNEMESIS<Scalar>::getNominalValues() const
{
  return nominalValues_;
}


template<class Scalar>
Teuchos::RCP<Thyra::LinearOpBase<Scalar> >
ModelEvaluatorNEMESIS<Scalar>::create_W_op() const
{
  Teuchos::RCP<Epetra_CrsMatrix> W_epetra =
    Teuchos::rcp(new Epetra_CrsMatrix(::Copy,*W_graph_));

  return Thyra::nonconstEpetraLinearOp(W_epetra);
}

template<class Scalar>
Teuchos::RCP< ::Thyra::PreconditionerBase<Scalar> >
ModelEvaluatorNEMESIS<Scalar>::create_W_prec() const
{
  const Teuchos::RCP<Thyra::LinearOpBase< Scalar > > P_op = prec_;

  Teuchos::RCP<Thyra::DefaultPreconditioner<Scalar> > prec =
    Teuchos::rcp(new Thyra::DefaultPreconditioner<Scalar>(Teuchos::null,P_op));

  return prec;
//  return Teuchos::null;
}

template<class Scalar>
Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> >
ModelEvaluatorNEMESIS<Scalar>::get_W_factory() const
{
  return W_factory_;
}


template<class Scalar>
Thyra::ModelEvaluatorBase::InArgs<Scalar>
ModelEvaluatorNEMESIS<Scalar>::createInArgs() const
{
  return prototypeInArgs_;
}


// Private functions overridden from ModelEvaulatorDefaultBase


template<class Scalar>
Thyra::ModelEvaluatorBase::OutArgs<Scalar>
ModelEvaluatorNEMESIS<Scalar>::createOutArgsImpl() const
{
  return prototypeOutArgs_;
}


template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::evalModelImpl(
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
    Epetra_FEVector f_fe(*f_owned_map_);//shared
    f_fe.PutScalar(0.0);
    Epetra_FEVector * f_fe_p = &f_fe;
    if (nonnull(f_out)) {
      f = Thyra::get_Epetra_Vector(*f_owned_map_,outArgs.get_f());//f_out?
      //f->Print(std::cout);
    }

    if (nonnull(f_out)){
      f->PutScalar(0.0);
    }
    if (nonnull(W_prec_out))
      P_->PutScalar(0.0);

    RCP<const Epetra_Vector> u_in = (Thyra::get_Epetra_Vector(*x_owned_map_,inArgs.get_x()));//shared
    RCP< Epetra_Vector> u = rcp(new Epetra_Vector(*x_overlap_map_));//shared
    //cn could probably just make u_old_(*x_overlap_map_) (and u_old_old_) instead of communicating here
    RCP< Epetra_Vector> u_old = rcp(new Epetra_Vector(*x_overlap_map_));//shared
    RCP< Epetra_Vector> u_old_old = rcp(new Epetra_Vector(*x_overlap_map_));//shared
    {
      Teuchos::TimeMonitor ImportTimer(*ts_time_import);
      u->Import(*u_in, *importer_, Insert);
      u_old->Import(*u_old_, *importer_, Insert);
      u_old_old->Import(*u_old_old_, *importer_, Insert);
    }
    int n_nodes_per_elem;//shared

    if (nonnull(f_out)) {
      Teuchos::TimeMonitor ResFillTimer(*ts_time_resfill);  
      for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){//shared
   
	n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);//shared
	std::string elem_type=mesh_->get_blk_elem_type(blk);//shared
	std::string * elem_type_p = &elem_type;
		
	//#ifdef TUSAS_COLOR_CPU
#ifdef TUSAS_COLOR_CPU
	int num_color = Elem_col->get_num_color();
	//std::vector< std::vector< int > > colors = Elem_col->get_colors();
	for(int c = 0; c < num_color; c++){
	  std::vector<int> elem_map = Elem_col->get_color(c);
	  //std::vector<int> elem_map = colors[c];
	  int num_elem = elem_map.size();
	
#pragma omp declare reduction (merge : std::vector<int>, std::vector<double> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
     
	  std::vector<nemesis_lint_t> offrows;
	  std::vector<double> offvals;
#pragma omp parallel for reduction(merge: offrows, offvals)
	  for (int ne=0; ne < num_elem; ne++) {// Loop Over # of Finite Elements on Processor 
	    int elem = elem_map[ne];//private
	    std::vector<double> xx(n_nodes_per_elem);//private
	    std::vector<double> yy(n_nodes_per_elem);//private
	    std::vector<double> zz(n_nodes_per_elem);//private
	    
	    std::vector<std::vector<double>> uu(numeqs_,std::vector<double>(n_nodes_per_elem));//private
	    std::vector<std::vector<double>> uu_old(numeqs_,std::vector<double>(n_nodes_per_elem));//private
	    std::vector<std::vector<double>> uu_old_old(numeqs_,std::vector<double>(n_nodes_per_elem));//private
	    boost::ptr_vector<Basis> basis;//private
	    
	    set_basis(basis,*elem_type_p);//cn really want this out at the block level
	    //std::cout<<c<<" "<<ne<<" "<<omp_get_thread_num()<<std::endl;
#else
#endif		
#ifdef TUSAS_COLOR_CPU		
#else
	int num_color = 1;
	std::vector<double> xx(n_nodes_per_elem);
	std::vector<double> yy(n_nodes_per_elem);
	std::vector<double> zz(n_nodes_per_elem);
	
	std::vector<std::vector<double>> uu(numeqs_,std::vector<double>(n_nodes_per_elem));
	std::vector<std::vector<double>> uu_old(numeqs_,std::vector<double>(n_nodes_per_elem));
	std::vector<std::vector<double>> uu_old_old(numeqs_,std::vector<double>(n_nodes_per_elem));
	boost::ptr_vector<Basis> basis;
	
	set_basis(basis,elem_type);
	
	for(int c = 0; c < num_color; c++){
	  std::vector<nemesis_lint_t> elem_map = *mesh_->get_elem_num_map();
	  int num_elem = elem_map.size();
	  for (int ne=0; ne < num_elem; ne++) {// Loop Over # of Finite Elements on Processor 
	    int elem = ne;
#endif
	    for(int k = 0; k < n_nodes_per_elem; k++){
	      
	      const int nodeid = mesh_->get_node_id(blk, elem, k);//cn appears this is the local id
	      //std::cout<<nodeid<<std::endl;
	      xx[k] = mesh_->get_x(nodeid);
	      yy[k] = mesh_->get_y(nodeid);
	      zz[k] = mesh_->get_z(nodeid);
	      
	      for( int neq = 0; neq < numeqs_; neq++ ){
		uu[neq][k] = (*u)[numeqs_*nodeid+neq]; 
		uu_old[neq][k] = (*u_old)[numeqs_*nodeid+neq];
		uu_old_old[neq][k] = (*u_old_old)[numeqs_*nodeid+neq];
	      }//neq
	    }//k
	    
	    //  double dx = 0.;
	    //  for(int gp=0; gp < basis[0].ngp; gp++) {
	    //  
	    //    basis[0].getBasis(gp, &xx[0], &yy[0], &zz[0]);
	      
	    //    dx += basis[0].jac*basis[0].wt;
	    // }
	    // 	if ( dx < 1e-16){
	    // 	  std::cout<<std::endl<<"Negative element size found"<<std::endl;
	    // 	  std::cout<<"dx = "<<dx<<"  elem = "<<elem<<" jac = "<<basis[0].jac<<" wt = "<<basis[0].wt<<std::endl<<std::endl<<std::endl;
	    // 	  exit(0);
	    // 	}
	    //cn should be cube root in 3d
	    //dx = sqrt(dx);	
	    
	    for(int gp=0; gp < basis[0].ngp; gp++) {// Loop Over Gauss Points 
	      
	      // Calculate the basis function at the gauss point
	      for( int neq = 0; neq < numeqs_; neq++ ){
		basis[neq].getBasis(gp, &xx[0], &yy[0], &zz[0], &uu[neq][0], &uu_old[neq][0], &uu_old_old[neq][0]);
	      }	      	  
    
	      //srand(123);
	      
	      for (int i=0; i< n_nodes_per_elem; i++) {// Loop over Nodes in Element; ie sum over test functions
		
		const nemesis_lint_t row = numeqs_*(
				   mesh_->get_global_node_id(mesh_->get_node_id(blk, elem, i))
				   );				
		for( int k = 0; k < numeqs_; k++ ){
		  const nemesis_lint_t row1 = row + k;
		  const double jacwt = basis[0].jac * basis[0].wt;
		  const double val = jacwt * (*residualfunc_)[k](basis,
								 i,
								 dt_,
								 dtold_,
								 t_theta_,
								 t_theta2_,
								 time_,
								 k);

#ifdef TUSAS_COLOR_CPU
		  if(f_owned_map_->MyGID(row1)){		    
		    f_fe_p->SumIntoGlobalValue (row1,(int) 0, val);//multivector version--may be faster
#else	
		    f_fe_p->SumIntoGlobalValues ((int) 1, &row1, &val);	//fevector version--needed in mpi
#endif
#ifdef TUSAS_COLOR_CPU
		  }else{
		    //std::cout<<comm_->MyPID()<<":"<<row<<std::endl;
		    offrows.push_back(row);
		    offvals.push_back(val);
		  }//if
#endif
		}//k
	      }//i
	    }//gp
	  }//ne	
	  //#ifdef TUSAS_COLOR_CPU
#ifdef TUSAS_COLOR_CPU
	  f_fe_p->SumIntoGlobalValues (offrows.size(), &offrows[0], &offvals[0]);//we need fevector version here	    
#endif
	}//c	
	  //exit(0);	
      }//blk
	  f_fe_p->GlobalAssemble(Epetra_CombineMode::Add,true);
    }//if f

    if (nonnull(W_prec_out)) {
      Teuchos::TimeMonitor PrecFillTimer(*ts_time_precfill);  
      for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){
	
	// #ifdef TUSAS_OMP
	// #pragma omp parallel for
	//       for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {// Loop Over # of Finite Elements on Processor
	// #endif
	
	n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);
	std::string elem_type=mesh_->get_blk_elem_type(blk);
	
	//cn for now we will turn coloring for matrix fill off, until we get a good handle on residual fill
#ifdef TUSAS_COLOR_CPU
	int num_color = Elem_col->get_num_color();
	for(int c = 0; c < num_color; c++){
	  std::vector<int> elem_map = Elem_col->get_color(c);
	  int num_elem = elem_map.size();
	
#pragma omp declare reduction (merge : std::vector<int>, std::vector<double> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
     
	  std::vector<nemesis_lint_t> offrows;
	  std::vector<nemesis_lint_t> offcols;
	  std::vector<double> offvals;
#pragma omp parallel for reduction(merge: offrows, offcols, offvals)
	  for (int ne=0; ne < num_elem; ne++) {// Loop Over # of Finite Elements on Processor 
	    int elem = elem_map[ne];//private
	    std::vector<double> xx(n_nodes_per_elem);//private
	    std::vector<double> yy(n_nodes_per_elem);//private
	    std::vector<double> zz(n_nodes_per_elem);//private
	    
	    std::vector<std::vector<double>> uu(numeqs_,std::vector<double>(n_nodes_per_elem));//private
	    std::vector<std::vector<double>> uu_old(numeqs_,std::vector<double>(n_nodes_per_elem));//private
	    std::vector<std::vector<double>> uu_old_old(numeqs_,std::vector<double>(n_nodes_per_elem));//private
	    boost::ptr_vector<Basis> basis;//private
	    
	    set_basis(basis,elem_type);//cn really want this out at the block level
#else
	int num_color = 1;	
	std::vector<double> xx(n_nodes_per_elem);
	std::vector<double> yy(n_nodes_per_elem);
	std::vector<double> zz(n_nodes_per_elem);
	
	std::vector<std::vector<double>> uu(numeqs_,std::vector<double>(n_nodes_per_elem));
	std::vector<std::vector<double>> uu_old(numeqs_,std::vector<double>(n_nodes_per_elem));
	std::vector<std::vector<double>> uu_old_old(numeqs_,std::vector<double>(n_nodes_per_elem));
	boost::ptr_vector<Basis> basis;
	
	set_basis(basis,elem_type);
	
	//   std::cout<<"DEBUG PROC="<<comm_->MyPID()<<std::endl;



	for(int c = 0; c < num_color; c++){
	  std::vector<nemesis_lint_t> elem_map = *mesh_->get_elem_num_map();
	  int num_elem = elem_map.size();
	  
	  for (int ne=0; ne < num_elem; ne++) {// Loop Over # of Finite Elements on Processor 
	    int elem = ne;
#endif
	    for(int k = 0; k < n_nodes_per_elem; k++){
	      
	      int nodeid = mesh_->get_node_id(blk, elem, k);//cn appears this is the local id
	      
	      xx[k] = mesh_->get_x(nodeid);
	      yy[k] = mesh_->get_y(nodeid);
	      zz[k] = mesh_->get_z(nodeid);
	      
	      for( int neq = 0; neq < numeqs_; neq++ ){
		uu[neq][k] = (*u)[numeqs_*nodeid+neq]; 
		uu_old[neq][k] = (*u_old)[numeqs_*nodeid+neq];
		uu_old_old[neq][k] = (*u_old_old)[numeqs_*nodeid+neq];
	      }//neq
	    }//k
	    	
	    for(int gp=0; gp < basis[0].ngp; gp++) {// Loop Over Gauss Points 
	      
	      // Calculate the basis function at the gauss point
	      for( int neq = 0; neq < numeqs_; neq++ ){
		basis[neq].getBasis(gp, &xx[0], &yy[0], &zz[0], &uu[neq][0], &uu_old[neq][0], &uu_old_old[neq][0]);
	      }
	      
	      //srand(123);//note that if this is activated, we get a different random number in f and prec
	      
	      for (int i=0; i< n_nodes_per_elem; i++) {// Loop over Nodes in Element; ie sum over test functions
		nemesis_lint_t row = numeqs_*(
					      mesh_->get_global_node_id(mesh_->get_node_id(blk, elem, i))
					      );
		
		// Loop over Trial (basis) Functions
		
		for(int j=0;j < n_nodes_per_elem; j++) {
		  //int column = numeqs_*(x_overlap_map_->GID(mesh_->get_node_id(blk, elem, j)));
		  nemesis_lint_t column = numeqs_*(mesh_->get_global_node_id(mesh_->get_node_id(blk, elem, j)));
		  
		  for( int k = 0; k < numeqs_; k++ ){
		    nemesis_lint_t row1 = row + k;
		    nemesis_lint_t column1 = column + k;
		    double jacwt = basis[0].jac * basis[0].wt;
		    double val = jacwt*(*preconfunc_)[k](basis,i,j,dt_,t_theta_,k);
		    
#ifdef TUSAS_COLOR_CPU
		    if(f_owned_map_->MyGID(row1) && f_owned_map_->MyGID(column1)){
#endif
		      P_->SumIntoGlobalValues(row1, 1, &val, &column1);
#ifdef TUSAS_COLOR_CPU
		    }else{
		      offrows.push_back(row1);
		      offcols.push_back(column1);
		      offvals.push_back(val);
		    }//if
#endif
		  }//k		    
		}//j
	      }//i
	    }//gp	    
	  }//ne
#ifdef TUSAS_COLOR_CPU
	  for( int k = 0; k < offrows.size(); k++ ){
	    P_->SumIntoGlobalValues(offrows[k], (int)1, &offvals[k], offcols[k]);
	  }//k	    
#endif
	}//c	
      }//blk      
    }//if prec



    for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){

#ifdef PERIODIC_BC
      if (nonnull(f_out) && 0 != periodic_bc_.size()) {
#else
      if (nonnull(f_out) && NULL != periodicbc_) {
#endif
 	//f_fe_p->GlobalAssemble(Epetra_CombineMode::Add,true);
	std::vector<nemesis_lint_t> node_num_map(mesh_->get_node_num_map());
	//f_fe.Print(std::cout);



	//cn there is alot going on here, this should be implemented into a class
	//cn especially when unstructured meshes are used...and parallel
	//cn see:
	// http://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/0836-11.pdf
	// and similar to
	// https://orbi.ulg.ac.be/bitstream/2268/100283/1/2012_COMMAT_PBC.pdf


#ifdef PERIODIC_BC
	  
	for(boost::ptr_vector<periodic_bc>::const_iterator it = periodic_bc_.begin();it != periodic_bc_.end();++it){
	  for(std::vector<int>::const_iterator it2 = (it->eqn_indices_).begin(); it2 !=(it->eqn_indices_).end();++it2){
	    int k = *it2;

	    //cn it still could be the case that the global assemble will be messed up if procs are in a funny order, 
	    //cn however this is not the case in our current tests.

	    //u->Print(std::cout);

	    //it->import_data(*f_fe_p,u);
	    it->import_data(*f_fe_p,u_in,k);

	    int ns_size = it->u_rep_->Map().NumMyElements ();

	    //f_rep_(ns1); u_rep_(ns2)

	    //first we will add f(ns1) to f(ns2) by looping over nodeset 2
	    //ie f(ns2) = f(ns2) + f_rep_(ns1)

	    //loop over ns2 and add value from f_rep_(ns1)
	    
	    for ( int j = 0; j < ns_size; j++ ){

	      //this is the gid for ns2
#ifdef MESH_64
	      Mesh::mesh_lint_t gid2 = it->u_rep_->Map().GID64(j);
#else
	      Mesh::mesh_lint_t gid2 = it->u_rep_->Map().GID(j);
#endif

	      bool local = x_owned_map_->MyGID(numeqs_*gid2 + k);
	      if(local) {
		int lid2 = (x_owned_map_->LID(numeqs_*gid2 + k) - k)/numeqs_;
		double val1 = (*(it->f_rep_))[j];
		Mesh::mesh_lint_t row = numeqs_*gid2 + k;
		f_fe_p->SumIntoGlobalValues ((int) 1, &row, &val1);
		//if( 1 == comm_->MyPID() ) std::cout<<j<<" "<<gid2<<" "<<lid2<<" "<<val1<<std::endl;
	      }	
	      //f_fe_p.GlobalAssemble(Epetra_CombineMode::Add,true);
	    }//j

	    //exit(0);

	    //loop over ns1
	    // u(ns1) - u_rep_(ns2)
	    for ( int j = 0; j < ns_size; j++ ){

	      //this is the gid for ns1
#ifdef MESH_64
	      Mesh::mesh_lint_t gid1 = it->f_rep_->Map().GID64(j);//this is the mesh gid
#else
	      Mesh::mesh_lint_t gid1 = it->f_rep_->Map().GID(j);//this is the mesh gid
#endif

	      bool local = x_owned_map_->MyGID(numeqs_*gid1 + k);
	      if(local) {
		int lid1 = (x_owned_map_->LID(numeqs_*gid1 + k) - k)/numeqs_;
		double val1 = (*(it->u_rep_))[j];
		Mesh::mesh_lint_t row = numeqs_*gid1 + k;
		int lid = x_owned_map_->LID(row);
		// u(ns1) - u(ns2)
		double val = (*u_in)[numeqs_*lid1 + k] - val1;
		f_fe_p->ReplaceGlobalValue (row, (int)0, val);
		//std::cout<<j<<" "<<gid1<<" "<<std::endl;
		//if( 3 == comm_->MyPID() ) std::cout<<"   "<<j<<" "<<gid1<<" "<<val1<<" "<<val<<std::endl;

	      }
	    }//j
	    //exit(0);
	  }//k
	}//it
#else
        for( int k = 0; k < numeqs_; k++ ){
	  std::vector<std::pair<int,int>>::iterator it;
	  for(it = (*periodicbc_)[k].begin();it != (*periodicbc_)[k].end(); ++it){

	    int ns_id1 = it->first;
	    int ns_id2 = it->second;
	    //cn the ids in sorted_node_set are local
	    //std::vector<int> ns1 = mesh_->get_sorted_node_set(ns_id1);

	    int ns_size1 = mesh_->get_sorted_node_set(ns_id1).size();
	    
	    //cn the first step is add all of the ns1 equations to the ns2 equations
	    //cn then replace all the ns1 eqns with u(ns1)-u(ns2)
	    for ( int j = 0; j < ns_size1; j++ ){
	      //cn the ids in sorted_node_set are local
	      //cn we will want to create the comm maps with global ids
	      //cn we could do this before sending to the periodicbc constructor
	      int lid1 = mesh_->get_sorted_node_set_entry(ns_id1, j);
	      int lid2 = mesh_->get_sorted_node_set_entry(ns_id2, j);
	      Mesh::mesh_lint_t gid1 = node_num_map[lid1];
	      Mesh::mesh_lint_t gid2 = node_num_map[lid2];

	      Mesh::mesh_lint_t row1 = numeqs_*gid1;//global row
	      Mesh::mesh_lint_t row2 = numeqs_*gid2;
	      
	      Mesh::mesh_lint_t row1k = row1 + k;
	      Mesh::mesh_lint_t row2k = row2 + k;
 	      //std::cout<<"k: "<<k<<std::endl;
 	      //std::cout<<"l: "<<lid1<<" "<<lid2<<std::endl;
	      //std::cout<<"g: "<<gid1<<" "<<gid2<<std::endl;
	      double val1 = f_fe[0][numeqs_*lid1 + k];
 	      //std::cout<<"f: "<<row2k<<" "<<val1<<std::endl;
	      f_fe_p->SumIntoGlobalValues ((int) 1, &row2k, &val1);
	      val1 = (*u)[numeqs_*lid2 + k];
	      double val = (*u_in)[numeqs_*lid1 + k]  - val1;
  	      //std::cout<<"u: "<<row1k<<" "<<val1<<std::endl;
	      f_fe_p->ReplaceGlobalValue (row1k, (int)0, val);
	    }//j
	  }//it
	}//k
#endif
	f_fe_p->GlobalAssemble(Epetra_CombineMode::Add,false);
      }//if
      //exit(0);
      if (nonnull(f_out) && NULL != dirichletfunc_) {
	//f_fe_p->GlobalAssemble(Epetra_CombineMode::Add,true);
	
	std::vector<nemesis_lint_t> node_num_map(mesh_->get_node_num_map());
	std::map<int,DBCFUNC>::iterator it;
      
        for( int k = 0; k < numeqs_; k++ ){
	  for(it = (*dirichletfunc_)[k].begin();it != (*dirichletfunc_)[k].end(); ++it){
	    int ns_id = it->first;
	    //std::cout<<it->first<<std::endl;
	    
	    //cn there is a fundamental difference between Epetra_FEVector::ReplaceGlobalValues
	    // and Epetra_FEVector(Epetra_MultiVector)::ReplaceGlobalValue
	    // the first performs an MPI communication, while the second only updates local values.
	    // the first case leads to a clash between MPI and OpenMP.
	    // ie the global id could be accessed simultaneously
	    // on threads in each mpi process. This caused segfault with mpi+omp here.


#pragma omp parallel for
	    for ( int j = 0; j < mesh_->get_node_set(ns_id).size(); j++ ){
	      
	      int lid = mesh_->get_node_set_entry(ns_id, j);

	      //cn not sure why this next line is here.....
	      //cn probably because node_set_entry is on overlap map (includes ghosts)
	      //if(!x_owned_map_->MyLID(lid) ) exit(0);//break;//check that this node lives on this proc, otherwise skip it
	      nemesis_lint_t gid = node_num_map[lid];
	      //std::cout<<ns_id<<" "<<gid<<" "<<mesh_->get_node_set(ns_id).size()<<std::endl;
	      nemesis_lint_t row = numeqs_*gid;//global row
	      //int row = numeqs_*lid;//local row
	      double x = mesh_->get_x(lid);
	      double y = mesh_->get_y(lid);
	      double z = mesh_->get_z(lid);
	      
	      nemesis_lint_t row1 = row + k;
	      double val1 = (it->second)(x,y,z,time_);//the function pointer eval
	      double val = (*u)[numeqs_*lid + k]  - val1;
	      //std::cout<<comm_->MyPID()<<" "<<row1<<" "<<std::endl;
	      //f_fe.ReplaceGlobalValues ((int) 1, &row1, &val);
	      f_fe_p->ReplaceGlobalValue (row1, (int)0, val);
	      
	    }//j
	  }//it
	}//k
	f_fe_p->GlobalAssemble(Epetra_CombineMode::Zero,true);
      }//if

      if (nonnull(f_out) && NULL != neumannfunc_) {
	//f_fe.GlobalAssemble(Epetra_CombineMode::Zero,true);
	Basis * basis;
	//this is the number of nodes per side edge
	//int num_node_per_side = mesh_->get_num_node_per_side(ss_id);
	int num_node_per_side = 2;
	
	std::string elem_type=mesh_->get_blk_elem_type(blk);

	if( (0==elem_type.compare("QUAD4")) 
	    || (0==elem_type.compare("QUAD")) 
	    || (0==elem_type.compare("quad4")) 
	    || (0==elem_type.compare("quad"))  
	    || (0==elem_type.compare("TRI3")) 
	    || (0==elem_type.compare("TRI")) 
	    || (0==elem_type.compare("tri3"))  
	    || (0==elem_type.compare("tri")) ){ // linear 2d element
	  
	  num_node_per_side = 2; 
	  basis = new BasisLBar();
	}
	else if( (0==elem_type.compare("QUAD9")) 
		 || (0==elem_type.compare("quad9")) 
		 || (0==elem_type.compare("TRI6")) 
		 || (0==elem_type.compare("tri6")) ){ // quadratic 2d
	  
	  num_node_per_side = 3;
	  basis = new BasisQBar();
	} 
	else if( (0==elem_type.compare("HEX8")) 
		 || (0==elem_type.compare("HEX")) 
		 || (0==elem_type.compare("hex8")) 
		 || (0==elem_type.compare("hex"))  ){ // linear hex
	  num_node_per_side = 4;
	  basis = new BasisLQuad();
	} 
	else if( (0==elem_type.compare("TETRA4")) 
		 || (0==elem_type.compare("TETRA")) 
		 || (0==elem_type.compare("tetra4")) 
		 || (0==elem_type.compare("tetra")) ){ // linear tet
	  num_node_per_side = 3;
	  basis = new BasisLTri();
	}
	
	std::vector<nemesis_lint_t> node_num_map(mesh_->get_node_num_map());
	std::map<int,NBCFUNC>::iterator it;
      
#ifdef TUSAS_INTERPFLUX
	//cn we need to move this to private data and initialize just once
	boost::ptr_vector<projection> proj;
	proj.push_back(new projection(comm_,"v3/side0.e","v3/flux_0.txt"));
	proj.push_back(new projection(comm_,"v3/side1.e","v3/flux_1.txt"));
	proj.push_back(new projection(comm_,"v3/side2.e","v3/flux_2.txt"));
	proj.push_back(new projection(comm_,"v3/side3.e","v3/flux_3.txt"));
	proj.push_back(new projection(comm_,"v3/side4.e","v3/flux_4.txt"));
	proj.push_back(new projection(comm_,"v3/side5.e","v3/flux_5.txt"));
	interpflux ifa(comm_,"v3/flux_time.txt");
#endif

        for( int k = 0; k < numeqs_; k++ ){
#ifdef TUSAS_INTERPFLUXAVG
	  interpfluxavg ifa(comm_,"v3/flux_thist.txt");
#endif

	  for(it = (*neumannfunc_)[k].begin();it != (*neumannfunc_)[k].end(); ++it){
	    //std::cout<<k<<std::endl;
	    int ss_id = it->first;

	    double *xx, *yy, *zz, *uu;
	    xx = new double[num_node_per_side];
	    yy = new double[num_node_per_side];
	    zz = new double[num_node_per_side];
	    uu = new double[num_node_per_side];

#ifdef TUSAS_INTERPFLUXAVG
	    double avgval = 0.;
	    ifa.get_source_value(time_, ss_id, avgval);
	    //avgval = avgval*100.;
	    //cn we need to check with Matt and see for sure what this valus is.  If it
	    //is the integrated value, we should probably divide by the area here
	    //9-27-17 it is the average data, ie sum of all fluxes divided by 100
	    //so we probably need to multiply by N here?

#endif
#ifdef TUSAS_INTERPFLUX
	    ifa.interp_time(time_);
	    proj[ss_id].fill_time_interp_values(ifa.timeindex_,ifa.theta_);
#endif
	    for ( int j = 0; j < mesh_->get_side_set(ss_id).size(); j++ ){//loop over element faces

	      for ( int ll = 0; ll < num_node_per_side; ll++){//loop over nodes in each face
		int lid = mesh_->get_side_set_node_list(ss_id)[j*num_node_per_side+ll];
		xx[ll] = mesh_->get_x(lid);
		yy[ll] = mesh_->get_y(lid);
		zz[ll] = mesh_->get_z(lid);
		uu[ll] = (*u)[numeqs_*lid+k];
		//std::cout<<ll<<" "<<lid<<" "<<xx[ll]<<" "<<yy[ll]<<" "<<zz[ll]<<" "<<uu[ll]<<" "<<mesh_->get_side_set_node_list(ss_id).size()<<std::endl;
	      }//ll
	      for ( int gp = 0; gp < basis->ngp; gp++){//loop over gauss pts
		basis->getBasis(gp,xx,yy,zz,uu);

		double jacwt = basis->jac * basis->wt;

		for( int i = 0; i < num_node_per_side; i++ ){

		  const int lid = mesh_->get_side_set_node_list(ss_id)[j*num_node_per_side+i];
		  const nemesis_lint_t gid = node_num_map[lid];
		  
		  //std::cout<<i<<" "<<lid<<" "<<gid<<" "<<basis->jac<<" "<<basis->wt<<std::endl;
		  const nemesis_lint_t row = numeqs_*gid;
		  const nemesis_lint_t row1 = row + k;

		  double val = -jacwt*(it->second)(basis,i,dt_,t_theta_,time_);//the function pointer eval

		  //here we have the convention that -k(lap u,phi) = k (grad u, grad phi) - <n.k grad u, phi>
		  //I believe the truchas convention is that flux g = -n.k grad u
		  //hence we have the minus sign below, ie -k(lap u,phi) = k (grad u, grad phi) - <-g, phi>

		  // also to do crank-nicolson here.  We could store a vector of vals
		  // at the previous step, ie it is hard to get a gradient on the surface
#ifdef TUSAS_INTERPFLUXAVG
		  if(0 == k){
		    double test = basis->phi[i];
		    //std::cout<<avgval<<" "<<test<<std::endl;
		    val = -jacwt*test*(-avgval);
	    //cn we need to check with Matt and see for sure what this valus is.  If it
	    //is the integrated value, we should probably divide by the area here
		  }
#endif
#ifdef TUSAS_INTERPFLUX
		  if(0 == k){
		    double test = basis->phi[i];
		    double gval = 0.;
		    proj[ss_id].get_source_value(basis->xx,basis->yy,basis->zz,gval);
		    val = -jacwt*test*(-gval);
		  }
#endif		  

		  f_fe_p->SumIntoGlobalValues ((int) 1, &row1, &val);
		}//i
	      }//gp
	    }//j
	    delete xx;
	    delete yy;
	    delete zz;
	    delete uu;
	  }//it
	}//k
#ifdef TUSAS_INTERPFLUX
	proj.clear();
#endif		  
	delete basis;
 	f_fe_p->GlobalAssemble(Epetra_CombineMode::Add,true);
      }//if (nonnull(f_out) && NULL != neumannfunc_)


      //cn WARNING the residual and precon are not fully tested, especially with numeqs_ > 1 !!!!!!!
      if (nonnull(W_prec_out) && NULL != dirichletfunc_) {
	//P_->GlobalAssemble(true,Epetra_CombineMode::Add,true);
#ifdef PERIODIC_BC
	if(0 != periodic_bc_.size() )exit(0);
#endif
	P_->GlobalAssemble();
	std::vector<nemesis_lint_t> node_num_map(mesh_->get_node_num_map());
	int lenind = 27;//cn 27 in 3d
	std::map<int,DBCFUNC>::iterator it;
        for( int k = 0; k < numeqs_; k++ ){
	  for(it = (*dirichletfunc_)[k].begin();it != (*dirichletfunc_)[k].end(); ++it){
	    int ns_id = it->first;
	    //#pragma omp parallel for
	    for ( int j = 0; j < mesh_->get_node_set(ns_id).size(); j++ ){
	      int lid = mesh_->get_node_set_entry(ns_id, j);

	      //cn we could implement this up front;
	      //cn ie construct a node_num_map that only contains locally owned nodes,
	      //cn then we could turn openmp back on above
	      //cn however mesh_->get_node_set(ns_id).size() would need to be fixed as well...
	 
	      if(!x_owned_map_->MyLID(lid) ) break;//check that this node live on this proc, otherwise skip it
	      nemesis_lint_t gid = node_num_map[lid];
	      //int gid = x_owned_map_->GID(lid);
	      nemesis_lint_t row = numeqs_*gid + k;
	      int num_nodes;
	    
	      std::vector<nemesis_lint_t> column(lenind);
	    
	      int err = W_graph_->ExtractGlobalRowCopy 	( 	row,
								lenind,
								num_nodes,
								&column[0]
								) ;

	      //cn we seem to get a garbage value for num_nodes out of here...
	      //cn must mean this row is not on this proc?
	      //std::cout<<comm_->MyPID()<<" "<<ns_id<<" "<<k<<" "<<num_nodes<<" "<<err<<std::endl;

	      //cn this is an awful hack, this part of the code has been causing problems
	      //cn forever. There has to be a better way.
	      if(err < 0 ) num_nodes = 0;
	      //num_nodes =P_-> NumGlobalEntries(row);

	      //column.resize(num_nodes);

	      std::vector<nemesis_lint_t> column1(num_nodes);
	      for(int ii = 0; ii<num_nodes; ii++) column1[ii]=column[ii];
	    
	      double d = 1.;
	      std::vector<double> vals (num_nodes,0.);
	      //P_->ReplaceGlobalValues (row, num_nodes, &vals[0],&column[0] );
	      P_->ReplaceGlobalValues (row, num_nodes, &vals[0],&column1[0] );
	      P_->ReplaceGlobalValues (row, (int)1, &d ,&row );

	    }//j

	  }//it
	}//k
      }//if
      }//blk

    
      if (nonnull(f_out)){
	//f->Print(std::cout);
// 	if (f_fe_p->GlobalAssemble(Epetra_CombineMode::Zero,true) != 0){
// 	  std::cout<<"error f_fe.GlobalAssemble()"<<std::endl;
// 	  exit(0);
// 	}
	
	f->Update(1,*f_fe(0),0);
	//*f=*f_fe(0);
      }
      if (nonnull(W_prec_out)) {
	//P_->GlobalAssemble(true,Epetra_CombineMode::Add,true);
	P_->GlobalAssemble();
	//P_->Print(std::cout);
	//exit(0);
	//std::cout<<" one norm P_ = "<<P_->NormOne()<<std::endl<<" inf norm P_ = "<<P_->NormInf()<<std::endl<<" fro norm P_ = "<<P_->NormFrobenius()<<std::endl;
	//Epetra_Vector d(*f_owned_map_);P_->ExtractDiagonalCopy(d);d.Print(std::cout);	
	prec_->ReComputePreconditioner(false);
      }
    }	
    return;
      }

//====================================================================

template<class Scalar>
ModelEvaluatorNEMESIS<Scalar>::~ModelEvaluatorNEMESIS()
{
  //  if(!prec_.is_null()) prec_ = Teuchos::null;
}
template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::init_nox()
{
  int mypid = comm_->MyPID();
  if( 0 == mypid )
    std::cout<<std::endl<<"init_nox() started."<<std::endl<<std::endl;
  nnewt_=0;

  ::Stratimikos::DefaultLinearSolverBuilder builder;

  Teuchos::RCP<Teuchos::ParameterList> lsparams =
    Teuchos::rcp(new Teuchos::ParameterList(paramList.sublist(TusaslsNameString)));

  builder.setParameterList(lsparams);

  if( 0 == mypid )
    builder.getParameterList()->print(std::cout);

  Teuchos::RCP< ::Thyra::LinearOpWithSolveFactoryBase<double> >
    lowsFactory = builder.createLinearSolveStrategy("");

  // Setup output stream and the verbosity level
  Teuchos::RCP<Teuchos::FancyOStream>
    out = Teuchos::VerboseObjectBase::getDefaultOStream();
  lowsFactory->setOStream(out);
  lowsFactory->setVerbLevel(Teuchos::VERB_EXTREME);

  this->set_W_factory(lowsFactory);

  //we could probably use an epetra_vector here 
  //that will solve our extraction problems in advance()

  // Create the initial guess
  Teuchos::RCP< ::Thyra::VectorBase<double> >
    initial_guess = this->getNominalValues().get_x()->clone_v();

//   Teuchos::RCP<Epetra_Vector> e_weight = Teuchos::rcp(new Epetra_Vector(*f_owned_map_,1.));
//   Teuchos::RCP<Thyra::VectorBase<double> >
//       weight = Thyra::create_Vector( e_weight, x_space_ );
  //weight = Teuchos::null;

  Thyra::V_S(initial_guess.ptr(),Teuchos::ScalarTraits<double>::one());

  // Create the JFNK operator
  Teuchos::ParameterList printParams;//cn this is empty??? for now
  Teuchos::RCP<NOX::Thyra::MatrixFreeJacobianOperator<double> > jfnkOp =
    Teuchos::rcp(new NOX::Thyra::MatrixFreeJacobianOperator<double>(printParams));

  Teuchos::RCP<Teuchos::ParameterList> jfnkParams = Teuchos::rcp(new Teuchos::ParameterList(paramList.sublist(TusasjfnkNameString)));
  jfnkOp->setParameterList(jfnkParams);
  if( 0 == mypid )
    jfnkParams->print(std::cout);

  Teuchos::RCP< ::Thyra::ModelEvaluator<double> > Model = Teuchos::rcpFromRef(*this);
  // Wrap the model evaluator in a JFNK Model Evaluator
  Teuchos::RCP< ::Thyra::ModelEvaluator<double> > thyraModel =
    Teuchos::rcp(new NOX::MatrixFreeModelEvaluatorDecorator<double>(Model));

  // Wrap the model evaluator in a JFNK Model Evaluator
//   Teuchos::RCP< ::Thyra::ModelEvaluator<double> > thyraModel =
//     Teuchos::rcp(new NOX::MatrixFreeModelEvaluatorDecorator<double>(this));

  //Teuchos::RCP< ::Thyra::PreconditionerBase<double> > precOp = thyraModel->create_W_prec();
  // Create the NOX::Thyra::Group


  bool precon = paramList.get<bool> (TusaspreconNameString);
  Teuchos::RCP<NOX::Thyra::Group> nox_group;
  if(precon){
    Teuchos::RCP< ::Thyra::PreconditionerBase<double> > precOp = thyraModel->create_W_prec();
    nox_group =
      Teuchos::rcp(new NOX::Thyra::Group(*initial_guess, 
					 thyraModel, 
					 jfnkOp, 
					 lowsFactory, 
					 precOp, 
					 Teuchos::null, 
					 Teuchos::null, 
					 Teuchos::null));
  }
  else {
    nox_group =
      Teuchos::rcp(new NOX::Thyra::Group(*initial_guess, 
					 thyraModel, 
					 jfnkOp, 
					 lowsFactory, 
					 Teuchos::null, 
					 Teuchos::null, 
					 Teuchos::null, 
					 Teuchos::null));
   }

  nox_group->computeF();

  // VERY IMPORTANT!!!  jfnk object needs base evaluation objects.
  // This creates a circular dependency, so use a weak pointer.
  jfnkOp->setBaseEvaluationToNOXGroup(nox_group.create_weak());

  // Create the NOX status tests and the solver
  // Create the convergence tests
  Teuchos::RCP<NOX::StatusTest::NormF> absresid =
    Teuchos::rcp(new NOX::StatusTest::NormF(1.0e-8));

  double relrestol = 1.0e-6;
  relrestol = paramList.get<double> (TusasnoxrelresNameString);

  Teuchos::RCP<NOX::StatusTest::NormF> relresid = 
    Teuchos::rcp(new NOX::StatusTest::NormF(*nox_group.get(), relrestol));//1.0e-6 for paper

  Teuchos::RCP<NOX::StatusTest::NormWRMS> wrms =
    Teuchos::rcp(new NOX::StatusTest::NormWRMS(1.0e-2, 1.0e-8));
  Teuchos::RCP<NOX::StatusTest::Combo> converged =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::AND));
  //converged->addStatusTest(absresid);
  converged->addStatusTest(relresid);
  //converged->addStatusTest(wrms);

  int maxit = 200;
  maxit = paramList.get<int> (TusasnoxmaxiterNameString);

  Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters =
    Teuchos::rcp(new NOX::StatusTest::MaxIters(maxit));//200

  Teuchos::RCP<NOX::StatusTest::FiniteValue> fv =
    Teuchos::rcp(new NOX::StatusTest::FiniteValue);
  Teuchos::RCP<NOX::StatusTest::Combo> combo =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
  //combo->addStatusTest(fv);
  combo->addStatusTest(converged);
  combo->addStatusTest(maxiters);

  Teuchos::RCP<Teuchos::ParameterList> nl_params =
    Teuchos::rcp(new Teuchos::ParameterList(paramList.sublist(TusasnlsNameString)));
  Teuchos::ParameterList& nlPrintParams = nl_params->sublist("Printing");
  nlPrintParams.set("Output Information",
		  NOX::Utils::OuterIteration  +
		  //                      NOX::Utils::OuterIterationStatusTest +
		  NOX::Utils::InnerIteration +
		    NOX::Utils::Details //+
		    //NOX::Utils::LinearSolverDetails
		    );
  //nl_params->print(std::cout);
  //combo->print(std::cout);
  // Create the solver
  solver_ =  NOX::Solver::buildSolver(nox_group, combo, nl_params);
  solver_->getList().print(std::cout);


  if(paramList.get<bool> (TusasestimateTimestepNameString)){
    Teuchos::ParameterList *atsList;
    atsList = &paramList.sublist (TusasatslistNameString, false );
    if(atsList->get<std::string> (TusasatstypeNameString) == "predictor corrector"){
      //init_predictor(); 
      
      Teuchos::RCP<NOX::Thyra::MatrixFreeJacobianOperator<double> > jfnkOp1 =
	Teuchos::rcp(new NOX::Thyra::MatrixFreeJacobianOperator<double>(printParams));
      
      jfnkOp1->setParameterList(jfnkParams);
      
      Teuchos::RCP<NOX::Thyra::Group> noxpred_group =
	Teuchos::rcp(new NOX::Thyra::Group(*initial_guess, 
					   thyraModel, 
					   jfnkOp1, 
					   lowsFactory, 
					   Teuchos::null, 
					   Teuchos::null, 
					   Teuchos::null, 
					   Teuchos::null));
      jfnkOp1->setBaseEvaluationToNOXGroup(noxpred_group.create_weak());
      noxpred_group->computeF();
      Teuchos::RCP<NOX::StatusTest::NormF>relresid1 = 
	Teuchos::rcp(new NOX::StatusTest::NormF(*noxpred_group.get(), relrestol));
      Teuchos::RCP<NOX::StatusTest::Combo> converged1 =
	Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::AND));
      converged1->addStatusTest(relresid1);
      //combo->addStatusTest(converged);
      
      Teuchos::RCP<Teuchos::ParameterList> nl_params1 =
	Teuchos::rcp(new Teuchos::ParameterList);
      nl_params1->set("Nonlinear Solver", "Line Search Based");
      nl_params1->sublist("Direction").sublist("Newton").set("Forcing Term Method", "Type 2");
      nl_params1->sublist("Direction").sublist("Newton").set("Forcing Term Initial Tolerance", 1.0e-1);
      nl_params1->sublist("Direction").sublist("Newton").set("Forcing Term Maximum Tolerance", 1.0e-2);
      nl_params1->sublist("Direction").sublist("Newton").set("Forcing Term Minimum Tolerance", 1.0e-5);
      Teuchos::ParameterList& nlPrintParams1 = nl_params1->sublist("Printing");
      nlPrintParams1.set("Output Information",
			 NOX::Utils::OuterIteration  +
			 //                      NOX::Utils::OuterIterationStatusTest +
			 NOX::Utils::InnerIteration +
		    NOX::Utils::Details //+
			 //NOX::Utils::LinearSolverDetails
			 );
      
      predictor_ =  NOX::Solver::buildSolver(noxpred_group, converged1, nl_params1);
    }
  }

 

  if( 0 == mypid )
    std::cout<<std::endl<<"init_nox() completed."<<std::endl<<std::endl;
}

template<class Scalar>
double ModelEvaluatorNEMESIS<Scalar>::advance()
{
  //if( 0 == comm_->MyPID()) std::cout<<std::endl<<"advance started"<<std::endl<<std::endl;

  const int mypid = comm_->MyPID();

  random_number_= ((double)rand()/(RAND_MAX)*2.-1.);
  random_vector_->Random();

  int maxiter = 1;
  bool timeadapt = paramList.get<bool>(TusasadaptiveTimestepNameString);
  Teuchos::ParameterList *atsList;
  atsList = &paramList.sublist (TusasatslistNameString, false );
  if( timeadapt ) maxiter = atsList->get<int>(TusasatsmaxiterNameString);
//   std::cout<<maxiter<<std::endl;
//   exit(0);

  double dtpred = dt_;
  int numit = 0;
  for(int iter = 0; iter<maxiter; iter++){
    //newton solve
    {
      Teuchos::RCP< VectorBase< double > > guess;
      guess = Thyra::create_Vector(u_old_,x_space_);
      Teuchos::TimeMonitor NSolveTimer(*ts_time_nsolve);
      
      if(paramList.get<bool> (TusasestimateTimestepNameString)){
	Teuchos::ParameterList *atsList;
	atsList = &paramList.sublist (TusasatslistNameString, false );
	if(atsList->get<std::string> (TusasatstypeNameString) == "predictor corrector"){
	  predictor();
	  guess = Thyra::create_Vector(pred_temp_,x_space_);
	}//if
      }//if
      //note that solver_->reset() will utilize the last solution as next guess
      NOX::Thyra::Vector thyraguess(*guess);//by sending the dereferenced pointer, we instigate a copy rather than a view
      solver_->reset(thyraguess);

      NOX::StatusTest::StatusType solvStatus = solver_->solve();
      if( !(NOX::StatusTest::Converged == solvStatus)) {
	if( 0 == mypid )
	  std::cout<<" NOX solver failed to converge. Status = "<<solvStatus<<std::endl<<std::endl;
	if(paramList.get<bool> (TusasnoxacceptNameString)){
	  if( 0 == mypid )
	    std::cout<<" Accepting step since "<<TusasnoxacceptNameString<<" is true."<<std::endl<<std::endl;
	}else{
	  exit(0);
	}
      }//if
      numit++;
    }//timer
    nnewt_ += solver_->getNumIterations();
    
    const Thyra::VectorBase<double> * sol = 
      &(dynamic_cast<const NOX::Thyra::Vector&>(solver_->getSolutionGroup().getX()
						).getThyraVector()
	);

    //there seems to be issues with casting between
    //NOX::Abstract::Vector,
    //NOX::Thyra::Vector and
    //NOX::Epetra::Vector
    //it compiles but segfaults with std::bad_cast
//     const Epetra_Vector * sol = 
//       &(dynamic_cast<const NOX::Epetra::Vector&>(solver_->getSolutionGroup().getX()
// 					    ).getEpetraVector()
// 	);

    Thyra::ConstDetachedSpmdVectorView<double> x_vec(sol->col(0));
    
    for (int nn=0; nn < num_my_nodes_; nn++) {//cn figure out a better way here...
      for( int k = 0; k < numeqs_; k++ ){
	//(*(*u_old_)(0))[numeqs_*nn+k]=x_vec[numeqs_*nn+k];
	(*u_new_)[numeqs_*nn+k]=x_vec[numeqs_*nn+k];
      }
    } 
    
    if((paramList.get<bool> (TusasestimateTimestepNameString))
       && !timeadapt){ 
      const double d = estimatetimestep();
    }//if
    if(timeadapt){ 
      dtpred = estimatetimestep();
    }//if
    
    if( timeadapt ){

      if(dtpred < dt_){
	
	dt_ = dtpred;

	if( 0 == comm_->MyPID())std::cout<<"     advance() step NOT ACCEPTED with dt = "<<dt_
					 <<"; new dt = "<<dtpred
					 <<"; and iterations = "<<numit<<std::endl;
	
      }else{
	if( 0 == comm_->MyPID())std::cout<<"     advance() step accepted with dt = "<<dt_
					 <<"; new dt = "<<dtpred
					 <<"; and iterations = "<<numit<<std::endl;

	break;

      }//if
    }//if

  }//iter
    
  *u_old_old_old_ = *u_old_old_;
  *u_old_old_ = *u_old_;
  *u_old_ = *u_new_;
    
  //end of timestep updates here
  //u_old_->Print(std::cout);
  random_number_old_=random_number_;
  random_vector_old_->Scale((double)1.,*random_vector_);
  //update_mesh_data();
  //if((paramList.get<std::string> (TusastestNameString)=="cummins") && ( (TusasmethodNameString)  == "phaseheat")){
  if((paramList.get<std::string> (TusastestNameString)=="cummins") && (1== comm_->NumProc())){
    find_vtip();
  }
  //boost::ptr_vector<error_estimator>::iterator it;
  for(boost::ptr_vector<error_estimator>::iterator it = Error_est.begin();it != Error_est.end();++it){
    //it->test_lapack();
    it->estimate_gradient(u_old_);
    it->estimate_error(u_old_);
  }
  
  dtold_ = dt_;
  time_ += dt_;
  //std::cout<<"time = "<<time_<<std::endl;
  postprocess(post_proc);
  dt_ = dtpred;
  return dtold_;
  //return dt_;
}

template<class Scalar>
  void ModelEvaluatorNEMESIS<Scalar>::initialize()
{
  if( 0 == comm_->MyPID()) std::cout<<std::endl<<"initialize started"<<std::endl<<std::endl;
  bool dorestart = paramList.get<bool> (TusasrestartNameString);
  if (!dorestart){
   
    init(u_old_);
    //u_old_old_->PutScalar(0.0);
    //u_old_old_->Update (1.,*u_old_ , 0.);
    u_old_old_->Scale(1.,*u_old_);
    //u_old_old_old_->PutScalar(0.0);
    u_old_old_old_->Scale(1.,*u_old_);

    Teuchos::ParameterList *atsList;
    atsList = &paramList.sublist (TusasatslistNameString, false );

    //initial solve need by second derivative error estimate
    //and for lagged coupled time derivatives
    //ie get a solution at u_{-1}
    if(((atsList->get<std::string> (TusasatstypeNameString) == "second derivative")
	&&paramList.get<bool> (TusasestimateTimestepNameString))
       ||((atsList->get<std::string> (TusasatstypeNameString) == "predictor corrector")
	&&paramList.get<bool> (TusasestimateTimestepNameString)&&t_theta_ < 1.)
       ||paramList.get<bool> (TusasinitialSolveNameString)){

      initialsolve();
    }
    
    //u_old_old_->Update (1.,*u_old_ , 0.);
    //u_old_old_old_->PutScalar(0.0);

    if(paramList.get<bool> (TusasestimateTimestepNameString)){    
      setadaptivetimestep();
    }

    int mypid = comm_->MyPID();
    int numproc = comm_->NumProc();
    
    if( 1 == numproc ){//cn for now
      //if( 0 == mypid ){
      outfilename = "results.e";
      ex_id_ = mesh_->create_exodus(outfilename.c_str());
      
    }
    else{
      std::string decompPath="decomp/";
      std::string mypidstring;
      if ( numproc > 9 && mypid < 10 ){
	mypidstring = std::to_string(0)+std::to_string(mypid);
      }
      else{
	mypidstring = std::to_string(mypid);
      }
      
      outfilename = decompPath+"/results.e."+std::to_string(numproc)+"."+mypidstring;
      ex_id_ = mesh_->create_exodus(outfilename.c_str());
    }//numproc

    mesh_->close_exodus(ex_id_);
    
    for( int k = 0; k < numeqs_; k++ ){
      mesh_->add_nodal_field((*varnames_)[k]);
    }
    
    output_step_ = 1;
    write_exodus();
    //if((paramList.get<std::string> (TusastestNameString)=="cummins") && ( (TusasmethodNameString)  == "phaseheat")){
    if((paramList.get<std::string> (TusastestNameString)=="cummins") && (1== comm_->NumProc()) ){
      init_vtip();
    }
  }
  else{
    restart(u_old_,u_old_old_);

//     if(1==comm_->MyPID())
//       std::cout<<"Restart unavailable"<<std::endl<<std::endl;
//     exit(0);
    for( int k = 0; k < numeqs_; k++ ){
      mesh_->add_nodal_field((*varnames_)[k]);
    }
  }// !restart

  *u_new_ =  *u_old_;
  postprocess(post_proc);

  if( 0 == comm_->MyPID()) std::cout<<std::endl<<"initialize finished"<<std::endl<<std::endl;
}

template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::finalize()
{

  int mypid = comm_->MyPID();
  int numproc = comm_->NumProc();

  bool dorestart = paramList.get<bool> (TusasrestartNameString);

  //update_mesh_data();
   
  //mesh_->write_exodus(ex_id_,2,time_);

  //****this writes the last timestep into the exodus file a second time
      //**** probably only when outputting at every step, we need to check for this 
      //****removing it at this point would mean reblessing all tests...
  write_exodus();
  
  //cn we should trigger this in xml file
  //write_matlab();

  std::cout<<(solver_->getList()).sublist("Direction").sublist("Newton").sublist("Linear Solver")<<std::endl;
  int ngmres = 0;

  if ( (solver_->getList()).sublist("Direction").sublist("Newton").sublist("Linear Solver")
       .isSublist("Output") == true){
    if ( (solver_->getList()).sublist("Direction").sublist("Newton").sublist("Linear Solver")
	 .sublist("Output").getEntryPtr("Cumulative Iteration Count") != NULL){
      ngmres = ((solver_->getList()).sublist("Direction").sublist("Newton").sublist("Linear Solver")
		.sublist("Output").getEntry("Cumulative Iteration Count")).getValue(&ngmres);
    }
  }

  if( 0 == mypid ){
    int numstep = paramList.get<int> (TusasntNameString) - this->start_step;
    std::cout<<std::endl
	     <<"Total number of Newton iterations:     "<<nnewt_<<std::endl
	     <<"Total number of GMRES iterations:      "<<ngmres<<std::endl 
	     <<"Total number of Timesteps:             "<<numstep<<std::endl
	     <<"Average number of Newton per Timestep: "<<(float)nnewt_/(float)(numstep)<<std::endl
	     <<"Average number of GMRES per Newton:    "<<(float)ngmres/(float)nnewt_<<std::endl
	     <<"Average number of GMRES per Timestep:  "<<(float)ngmres/(float)(numstep)<<std::endl;
    if( dorestart ) std::cout<<"============THIS IS A RESTARTED RUN============"<<std::endl;
    std::ofstream outfile;
    outfile.open("jfnk.dat");
    outfile 
      <<"Total number of Newton iterations:     "<<nnewt_<<std::endl
      <<"Total number of GMRES iterations:      "<<ngmres<<std::endl 
      <<"Total number of Timesteps:             "<<numstep<<std::endl
      <<"Average number of Newton per Timestep: "<<(float)nnewt_/(float)(numstep)<<std::endl
      <<"Average number of GMRES per Newton:    "<<(float)ngmres/(float)nnewt_<<std::endl
      <<"Average number of GMRES per Timestep:  "<<(float)ngmres/(float)(numstep)<<std::endl; 
    if( dorestart ) outfile<<"============THIS IS A RESTARTED RUN============"<<std::endl;	
    outfile.close();
  }
#ifdef TUSAS_COLOR_CPU
  int nt = 0;
#pragma omp parallel reduction(+:nt)
  
  nt += 1;
  //nt = omp_get_num_threads();
#ifdef _OPENMP
  int ompmt = 0; 
  ompmt = omp_get_max_threads();
  std::ofstream outfile;
  outfile.open("openmp.dat");
  outfile 	
    <<"mpirank :    "<<mypid<<" omp_get_num_threads() :    "<<nt
    <<" omp_get_max_threads() :    "<<ompmt<<std::endl;
  outfile.close();
#endif
#endif
  
  //if((paramList.get<std::string> (TusastestNameString)=="cummins") && ( (TusasmethodNameString)  == "phaseheat")){
  if((paramList.get<std::string> (TusastestNameString)=="cummins") && (1== comm_->NumProc())){
    finalize_vtip();
  }

  //finalize exaConstit here...
  if(paramList.get<bool> (TusasexaConstitNameString)) {
    dump_exaconstit();
  }

  if(!x_space_.is_null()) x_space_=Teuchos::null;
  if(!x_owned_map_.is_null()) x_owned_map_=Teuchos::null;
  if(!f_owned_map_.is_null()) f_owned_map_=Teuchos::null;
  if(!W_graph_.is_null()) W_graph_=Teuchos::null;
  if(!W_factory_.is_null()) W_factory_=Teuchos::null;
  if(!x0_.is_null()) x0_=Teuchos::null;
  if(!P_.is_null())  P_=Teuchos::null;
  if(!prec_.is_null()) prec_=Teuchos::null;
  //if(!solver_.is_null()) solver_=Teuchos::null;
  if(!u_old_.is_null()) u_old_=Teuchos::null;
  if(!dudt_.is_null()) dudt_=Teuchos::null;

  delete residualfunc_;
  delete preconfunc_;
  delete initfunc_;
  delete varnames_;
  if( NULL != dirichletfunc_) delete dirichletfunc_;
#ifdef PERIODIC_BC
#else
  if( NULL != periodicbc_) delete periodicbc_;
#endif
  //if( NULL != postprocfunc_) delete postprocfunc_;
}

template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::compute_error( double *u)
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
  delete xx;
  delete yy;
  delete uu;
  delete ubasis;
}
template<class Scalar>
const double ModelEvaluatorNEMESIS<Scalar>::gs( const double &theta)
{
  return 1. + eps_ * (M_*cos(theta));
}
// template<class Scalar>
// double ModelEvaluatorNEMESIS<Scalar>::gs2_( const double &theta, const double &eps) const
// { 
//   //double g = 1. + eps_ * (M_*cos(theta));
//   double g = eps_0_*(1. + eps * (cos(M_*theta)));
//   return g*g;
// }
template<class Scalar>
const double ModelEvaluatorNEMESIS<Scalar>::R(const double &theta)
{
  return R_0_*(1. + eps_ * cos(M_*(theta)));
}
template<class Scalar>
const double ModelEvaluatorNEMESIS<Scalar>::R(const double &theta,const double &psi)
{

  double g = cummins::gs_cummins_(theta,M_,eps_,psi);
  return R_0_*g;
}


template<class Scalar>
double ModelEvaluatorNEMESIS<Scalar>::theta(double &x,double &y) const
{
  double small = 1e-9;
  double pi = 3.141592653589793;
  double t = 0.;
  double sy = 1.;
  if(y < 0.) sy = -1.;
  double n = sy*sqrt(y*y);
  //double n = y;
  //std::cout<<y<<"   "<<n<<std::endl;
//   if(abs(x) < small && y > 0. ) t = pi/2.;
//   else if(abs(x) < small && y < 0. ) t = 3.*pi/2.;
//   else t= atan(n/x);
  if(std::abs(x) < small && y > 0. ) t = pi/2.;
  if(std::abs(x) < small && y < 0. ) t = 3.*pi/2.;
  if(x > small && y >= 0.) t= atan(n/x);
  if(x > small && y <0.) t= atan(n/x) + 2.*pi;
  if(x < -small) t= atan(n/x)+ pi;

  return t;
}

template<class Scalar>
double ModelEvaluatorNEMESIS<Scalar>::psi(double &x,double &y,double &z) const
{
  //cn only first upper quadrant now
  double small = 1e-9;
  double pi = 3.141592653589793;
  double t = 0.;
  double sz = 1.;
  if(z < 0.) sz = -1.;
  double n = sz*sqrt(z*z);
  //double n = y;
  //std::cout<<y<<"   "<<n<<std::endl;
//   if(abs(x) < small && y > 0. ) t = pi/2.;
//   else if(abs(x) < small && y < 0. ) t = 3.*pi/2.;
//   else t= atan(n/x);
  if(std::abs(x) < small && y > 0. ) t = pi/2.;
  if(std::abs(x) < small && y < 0. ) t = 3.*pi/2.;
  if(x > small && y >= 0.) t= atan(n/x);
  if(x > small && y <0.) t= atan(n/x) + 2.*pi;
  if(x < -small) t= atan(n/x)+ pi;

  return t;

}

template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::init(Teuchos::RCP<Epetra_Vector> u)
{
  srand(123);

#ifdef TUSAS_PROJECTION
  projection proj(comm_,"v3/target3d.e","v3/temperature.txt");
  proj.fill_initial_values();
  //projection proj(comm_,"target2d.e","test_v2.txt");
#endif
  for( int k = 0; k < numeqs_; k++ ){
#ifdef TUSAS_PROJECTION
#else
#pragma omp parallel for
#endif
    for (int nn=0; nn < num_my_nodes_; nn++) {

#ifdef MESH_64
      //const nemesis_lint_t gid_node = epetra_map_gid(x_owned_map_,nn*numeqs_);
      const nemesis_lint_t gid_node = x_owned_map_->GID64(nn*numeqs_);
#else
      const int gid_node = x_owned_map_->GID(nn*numeqs_);
#endif
      const int lid_overlap = (x_overlap_map_->LID(gid_node))/numeqs_; 

      const double x = mesh_->get_x(lid_overlap);
      const double y = mesh_->get_y(lid_overlap);
      const double z = mesh_->get_z(lid_overlap);
      
#ifdef TUSAS_PROJECTION
      if(0 == k){
	double val = 0.;
	//      std::cout<<x<<" "<<y<<" "<<z<<std::endl;
	proj.get_source_value(x,y,z,val);
	(*u)[numeqs_*nn+k] = val;
      }
      else{
	(*u)[numeqs_*nn+k] = (*initfunc_)[k](x,y,z,k);
      }
#else
      (*u)[numeqs_*nn+k] = (*initfunc_)[k](x,y,z,k);
#endif
    }
  }
  //u->Print(std::cout);
}

template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::init_karma(Teuchos::RCP<Epetra_Vector> u)
{
  for (int nn=0; nn < num_my_nodes_; nn++) {
    double x = mesh_->get_x(nn);
    double y = mesh_->get_y(nn);
    double z = mesh_->get_z(nn);
    double t = theta(x,y);
    double r = R(t);
    if(x*x+y*y+z*z < r*r){
      (*u)[numeqs_*nn]= T_m_;
      //(*u)[numeqs_*nn]=T_inf_;
      (*u)[numeqs_*nn+1]=1.;
    }
    else {
      (*u)[numeqs_*nn]=T_inf_;
      (*u)[numeqs_*nn+1]=-1.;
    }
    

    //std::cout<<nn<<" "<<x<<" "<<y<<" "<<r<<"      "<<(*u_old_)[numeqs_*nn]<<"           "<<x*x+y*y<<" "<<r*r<<std::endl;
  }
}

template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::init_square(Teuchos::RCP<Epetra_Vector> u)
{
  for (int nn=0; nn < num_my_nodes_; nn++) {
    double x = mesh_->get_x(nn);
    double y = mesh_->get_y(nn);
    double z = mesh_->get_z(nn);
    if((std::abs(x) < R_0_) && (std::abs(y) < R_0_) && (std::abs(z) < R_0_)){
      //(*u)[numeqs_*nn]=T_m_;
      (*u)[numeqs_*nn]=T_inf_;
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
void ModelEvaluatorNEMESIS<Scalar>::multi(Teuchos::RCP<Epetra_Vector> u)
{
  for (int nn=0; nn < num_my_nodes_; nn++) {
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
void ModelEvaluatorNEMESIS<Scalar>::pool(Teuchos::RCP<Epetra_Vector> u)
{
  for (int nn=0; nn < num_my_nodes_; nn++) {
    double x = mesh_->get_x(nn);
    double y = mesh_->get_y(nn);
    double pi = 3.141592653589793;
    //double z = mesh_->get_z(nn);
    //double t = theta(x,y);
    srand(123);
    double r1 = rand()%15;
    double r2 = rand()%9+1;
    double r3 = rand()%15;
    double r4 = rand()%9+1;
    double r5 = rand()%15;
    double r6 = rand()%9+1;

    double r7 = .5*(sin(r1*y/r2)*cos(r3*y/r4+r2)*sin(r5*y/r6+r3)+1.);

    //double rr= 4.5;
    double rr= 9.;

    double r = r7*.3*fabs(sin(24.* pi* y/14.)) - sqrt(rr*rr-y*y);
    if(x < r){
      (*u)[numeqs_*nn]=T_m_;
      (*u)[numeqs_*nn+1]=1.;
    }
    else {
      (*u)[numeqs_*nn]=T_inf_;
      (*u)[numeqs_*nn+1]=0.;
    }
    

    //std::cout<<nn<<" "<<x<<" "<<y<<" "<<r<<"      "<<(*u)[numeqs_*nn]<<"           "<<x*x+y*y<<" "<<r*r<<std::endl;
  }
}
template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::init_vtip()
{
  //cn hack
  std::cout<<"init_vtip() started"<<std::endl;
  if(0 != mesh_->get_node_set(0).size() ){
    // std::cout<<"init_vtip() mesh_->get_node_set(0).size() == 0"<<std::endl;
    //exit(0);
  
    for ( int j = 0; j < mesh_->get_node_set(0).size(); j++ ){
      //int nodeid = mesh_->get_node_id(blk, ne, j);
      int nodeid = mesh_->get_node_set_entry(0, j);
      
      //std::cout<<j<<" "<<nodeid<<" "<<mesh_->get_x(nodeid)<<" "<<mesh_->get_y(nodeid)<<std::endl;	    
      x_node.insert(std::pair<double,int>(mesh_->get_x(nodeid),nodeid) ); 
    }
    
    vtip_x_ = 0.;
    find_vtip_x();
    vtip_x_old_ = vtip_x_;  
    std::ofstream outfile;
    outfile.open("vtip.dat");
    outfile.close();
    std::cout<<"init_vtip() ended"<<std::endl;
  }
}
template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::find_vtip()
{
  vtip_x_old_ = vtip_x_;
  find_vtip_x();
  if( vtip_x_ > 0. && vtip_x_ < 4.5 ){
    std::cout<<"vtip_x_     = "<<vtip_x_<<std::endl;
    std::cout<<"vtip_x_old_ = "<<vtip_x_old_<<std::endl;
    std::cout<<"vtip        = "<<(vtip_x_-vtip_x_old_)/dt_<<std::endl<<std::endl;
    std::ofstream outfile;
    
    std::cout<<"Writing vtip data start proc: "<<comm_->MyPID()<<std::endl;
    outfile.open("vtip.dat", std::ios::app );
    outfile << std::setprecision(16)
	    <<time_<<" "<<(vtip_x_-vtip_x_old_)/dt_<<" "<<vtip_x_<<std::endl;
    outfile.close();
    std::cout<<"Writing vtip data end proc: "<<comm_->MyPID()<<std::endl;
  }
  //exit(0);
}
template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::finalize_vtip()
{
}
template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::find_vtip_x()
{
  vtip_x_ = -9999.;
  double phi_avg = .5*(phi_sol_ + phi_liq_);
  std::map<double,int>::iterator it;
  for (it=x_node.begin(); it!=x_node.end(); ++it){
    int nodeid = it->second;
    double x2 = it->first;
    double phi2 = (*u_old_)[numeqs_*nodeid+1];
    //std::cout << it->first << " => " << it->second << " => " <<(*u_old_)[numeqs_*nodeid+1] << std::endl;
    if ((phi2 < phi_avg) && (phi2 > phi_liq_+1e-6)){
//       std::cout<<x2<<" "<<nodeid<<" "<<phi2<<std::endl;
      --it;
      double x1 = it->first;
      double phi1 = (*u_old_)[numeqs_*(it->second)+1];
      double m = (phi2-phi1)/(x2-x1);
      vtip_x_ = (m*x2-phi2+phi_avg)/m;
      std::cout<<"x1: "<<x1<<" "<<phi1<<" "<<std::endl;
      std::cout<<"x2: "<<x2<<" "<<phi2<<" "<<std::endl;
      std::cout<<"x:  "<<vtip_x_<<" "<<phi_avg<<" "<<m<<" proc: "<<comm_->MyPID()<<std::endl;
      std::cout<<"nodeid:  "<<nodeid<<std::endl;
      break;
    }
  }
  //exit(0);
}
template<class Scalar>
int ModelEvaluatorNEMESIS<Scalar>:: update_mesh_data()
{
  //std::vector<int> node_num_map(mesh_->get_node_num_map());

  Epetra_Vector *temp;
  if( 1 == comm_->NumProc() ){
    temp = new Epetra_Vector(*u_old_);
  }
  else{
    temp = new Epetra_Vector(*x_overlap_map_);//cn might be better to have u_old_ live on overlap map
    temp->Import(*u_old_, *importer_, Insert);

  }

  //int num_nodes = mesh_->get_node_num_map().size();

  std::vector<std::vector<double>> output(numeqs_, std::vector<double>(num_nodes_));

  for( int k = 0; k < numeqs_; k++ ){
#pragma omp parallel for
    for (int nn=0; nn < num_nodes_; nn++) {
      output[k][nn]=(*temp)[numeqs_*nn+k];
    }
//     outputu[nn]=(*u_old_)[numeqs_*nn];
//     outputphi[nn]=(*u_old_)[numeqs_*nn+1];
    //std::cout<<comm_->MyPID()<<" "<<nn<<" "<<outputu[nn]<<" "<<outputphi[nn]<<std::endl;
  }
  int err = 0;
  for( int k = 0; k < numeqs_; k++ ){
    mesh_->update_nodal_data((*varnames_)[k], &output[k][0]);
  }

  boost::ptr_vector<error_estimator>::iterator it;
  for(it = Error_est.begin();it != Error_est.end();++it){
    it->update_mesh_data();
  }

  boost::ptr_vector<post_process>::iterator itp;
  for(itp = post_proc.begin();itp != post_proc.end();++itp){
    itp->scalar_reduction();
    itp->update_mesh_data();
    itp->update_scalar_data(time_);
  }

  for(itp = temporal_est.begin();itp != temporal_est.end();++itp){
    itp->scalar_reduction();
    itp->update_mesh_data();
    itp->update_scalar_data(time_);
  }

  for(itp = temporal_norm.begin();itp != temporal_norm.end();++itp){
    itp->scalar_reduction();
    itp->update_mesh_data();
    itp->update_scalar_data(time_);
  }

#ifdef TUSAS_COLOR_CPU
  Elem_col->update_mesh_data();
#endif

  delete temp;
  return err;

}

//cn this will eventually be called from tusas
template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::write_exodus()
//void ModelEvaluatorNEMESIS<Scalar>::write_exodus(const int output_step)
{
  //std::cout<<"write_exodus(): output_step_ = "<<output_step_<<" time_ = "<<time_<<std::endl;
  update_mesh_data();
  mesh_->open_exodus(outfilename.c_str(),Mesh::WRITE);
  mesh_->write_exodus(ex_id_,output_step_,time_);
  mesh_->close_exodus(ex_id_);
  output_step_++;
}
template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::write_matlab()
{
  Epetra_Vector *temp;
  if( 1 == comm_->NumProc() ){
    temp = new Epetra_Vector(*u_old_);
  }
  else{
    temp = new Epetra_Vector(*x_overlap_map_);//cn might be better to have u_old_ live on overlap map
    temp->Import(*u_old_, *importer_, Insert);

  }

  double theta=paramList.get<double> (TusasthetaNameString);
  std::string method;
  if( theta < .4 ) {
    method = "ee";
  }
  else if ( theta > .6 ){
    method = "ie";
  }
  else {
    method = "cn";
  }
  int numstep = paramList.get<int> (TusasntNameString);
  std::string filename="results-"+method+"-"+std::to_string(numstep)+"-"+std::to_string(dt_)+".dat";

  std::cout<<filename<<std::endl;

//   std::ofstream outfile;
//   outfile.open(filename);
//   outfile << std::setprecision(16);
//   temp->Print(outfile);
//   comm_->Barrier();

  EpetraExt::VectorToMatlabFile (filename.c_str(), *temp);

  //cn in matlab
  /*
  cn8=load(['/Users/cnewman/work/tusas/trunk/temporal/'...
          'results-cn-128-0.000008.dat']);	
  */
}
template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::restart(Teuchos::RCP<Epetra_Vector> u,Teuchos::RCP<Epetra_Vector> u_old)
{
  //cn we need to get u_old_ and u_old_old_
  //and start_time and start_step and modify time_
  int mypid = comm_->MyPID();
  int numproc = comm_->NumProc();
  if( 0 == mypid )
    std::cout<<std::endl<<"Entering restart: PID "<<mypid<<" NumProcs "<<numproc<<std::endl<<std::endl;
  
  if( 1 == numproc ){//cn for now
    //if( 0 == mypid ){
    outfilename = "results.e";
    ex_id_ = mesh_->open_exodus(outfilename.c_str(),Mesh::READ);

    std::cout<<"  Opening file for restart; ex_id_ = "<<ex_id_<<" filename = "<<outfilename<<std::endl;
    
  }
  else{
    std::string decompPath="decomp/";
    std::string mypidstring(getmypidstring(mypid,numproc));

    outfilename = decompPath+"results.e."+std::to_string(numproc)+"."+mypidstring;
    ex_id_ = mesh_->open_exodus(outfilename.c_str(),Mesh::READ);
    
    std::cout<<"  Opening file for restart; ex_id_ = "<<ex_id_<<" filename = "<<outfilename<<std::endl;

    //cn we want to check the number of procs listed in the nem file as well    
    int nem_proc = -99;
    int error = mesh_->read_num_proc_nemesis(ex_id_, &nem_proc);
    if( 0 > error ) {
      std::cout<<"Error obtaining restart num procs in file"<<std::endl;
      exit(0);
    }
    if( nem_proc != numproc ){
      std::cout<<"Error restart nem_proc = "<<nem_proc<<" does not equal numproc = "<<numproc<<std::endl;
      exit(0);
    }
  }
  int step = -99;
  int error = mesh_->read_last_step_exodus(ex_id_,step);
  if( 0 == mypid )
    std::cout<<"  Reading restart last step = "<<step<<std::endl;
  if( 0 > error ) {
    std::cout<<"Error obtaining restart last step"<<std::endl;
    exit(0);
  }

  double time = -99.99;
  error = mesh_->read_time_exodus(ex_id_, step, time);
  if( 0 == mypid )
    std::cout<<"  Reading restart last time = "<<time<<std::endl;
  if( 0 > error ) {
    std::cout<<"Error obtaining restart last time"<<std::endl;
    exit(0);
  }

  double min_time = 1.e12;
  comm_->MinAll(&time, &min_time, 1);
  double max_time = 1.e-12;
  comm_->MaxAll(&time, &max_time, 1);
  if(fabs(max_time-min_time)>dt_/4.){
    if( 0 == mypid ){
      std::cout<<"  Error reading restart min and max time differ"<<std::endl;
      std::cout<<"    Reading restart min time = "<<min_time<<std::endl;
      std::cout<<"    Reading restart max time = "<<max_time<<std::endl;
      std::cout<<"    Reading restart difference = "<<fabs(max_time-min_time)<<std::endl;
    }
    exit(0);
  }

#if 0
  int step_old = step - 1;
  double time_old = -99.99;
  error = mesh_->read_time_exodus(ex_id_, step_old, time_old);
  if( 0 > error ) {
    std::cout<<"Error obtaining restart previous time"<<std::endl;
    exit(0);
  }
  if( 0 == mypid ){
    std::cout<<std::endl<<"  Restart last step found = "<<step<<"  time = "<<time<<std::endl;
    std::cout<<"      previous step found = "<<step_old<<"  time = "<<time_old<<std::endl<<std::endl;
  }

  if( dt_ != (time - time_old) ){
    std::cout<<"Error dt_ = "<<dt_<<"; time - time_old = "<<time - time_old<<std::endl;
    //exit(0);
  }
#endif

  std::vector<std::vector<double>> inputu(numeqs_,std::vector<double>(num_nodes_));

  for( int k = 0; k < numeqs_; k++ ){

    error = mesh_->read_nodal_data_exodus(ex_id_,step,(*varnames_)[k],&inputu[k][0]);
    if( 0 > error ) {
      std::cout<<"Error reading u at step "<<step<<std::endl;
      exit(0);
    }
  }

  mesh_->close_exodus(ex_id_);

  //cn for now just put current values into old values, 
  //cn ie just start with an initial condition

  //cn lets not worry about two different time steps for normal simulations
#if 0
  for (int nn=0; nn < num_nodes_; nn++) {
    (*u)[numeqs_*nn] = inputu[nn];
    (*u)[numeqs_*nn+1] = inputphi[nn];
  }

  error = mesh_->read_nodal_data_exodus(ex_id_,step_old,1,inputu);
  if( 0 > error ) {
    std::cout<<"Error reading u at step "<<step_old<<std::endl;
    exit(0);
  }
  error = mesh_->read_nodal_data_exodus(ex_id_,step_old,2,inputphi);
  if( 0 > error ) {
    std::cout<<"Error reading phi at step "<<step_old<<std::endl;
    exit(0);
  }
#endif

  Teuchos::RCP< Epetra_Vector> u_temp = Teuchos::rcp(new Epetra_Vector(*x_overlap_map_));
  //Teuchos::RCP< Epetra_Vector> u_old_temp = Teuchos::rcp(new Epetra_Vector(*x_overlap_map_));

  for( int k = 0; k < numeqs_; k++ ){
    for (int nn=0; nn < num_nodes_; nn++) {
      (*u_temp)[numeqs_*nn+k] = inputu[k][nn];
      //(*u_old_temp)[numeqs_*nn+k] = inputu[k][nn];
    }
  }

  Teuchos::RCP< Epetra_Export > exporter = Teuchos::rcp(new Epetra_Export(*x_overlap_map_, *x_owned_map_));

  u->Export(*u_temp,*exporter, Insert);
  u_old->Export(*u_temp,*exporter, Insert);
  this->start_time = time;
  int ntstep = (int)(time/dt_);
  //this->start_step = step-1;//this corresponds to the output frequency, not the actual timestep
  this->start_step = ntstep+1;
  time_=time;
  output_step_ = step+1;
  //   u->Print(std::cout);
  //   exit(0);
  if( 0 == mypid ){
    std::cout<<"Restarting at time = "<<time<<" and step = "<<step<<std::endl<<std::endl;
    std::cout<<"Exiting restart"<<std::endl<<std::endl;
  }
}


template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::set_test_case()
{
  numeqs_ = 2;
  random_number_ =((double)rand()/(RAND_MAX)*2.-1.);
  random_number_old_ = 0.;

  paramfunc_ = NULL;
#ifdef PERIODIC_BC
#else
  periodicbc_ = NULL;
#endif

  phi_sol_ = 1.;
  phi_liq_ = 0.;

  dgs2_2dpsi_ = &cummins::dgs2_2dpsi_cummins_;

  if("furtado" == paramList.get<std::string> (TusastestNameString)){
    exit(0);
    D_ = 1.55e-5;
    T_m_ = 1728.;
    T_inf_ = 300.;
    alpha_ = 191.82;
    eps_ = .025;
    eps_0_ = 2.01e-4;
    M_= 6.;
    theta_0_ =-1.5707963267949/2.;
    R_0_ =1.1e-6;
    
    //function pointers
    hp1_ = &hp1_furtado_;
    hpp1_ = &hpp1_furtado_;
    w_ = &w_furtado_;
    m_ = &m_furtado_;
    rand_phi_ = &rand_phi_furtado_;
    //rand_phi_ = &rand_phi_cummins_;
    gp1_ = &gp1_furtado_;
    gpp1_ = &gpp1_furtado_;
    hp2_ = &hp2_furtado_;

  }else if("karma" == paramList.get<std::string> (TusastestNameString)){
    exit(0);

    phi_liq_ = -1.;
    
    D_ = 1.;
    T_m_ = 1.-.55;
    T_inf_ = -.55;
    alpha_ = 191.82;
    eps_ = .02;
    eps_0_ = 1.;
    M_= 4.;
    theta_0_ =0.;
    R_0_ =.30;
    
    //function pointers
    hp1_ = &hp1_karma_;
    hpp1_ = &hpp1_karma_;
    w_ = &w_karma_;
    m_ = &m_karma_;
    rand_phi_ = &rand_phi_zero_;
    gp1_ = &gp1_karma_;
    gpp1_ = &gpp1_karma_;
    hp2_ = &hp2_karma_;

    gs2_ = &gs2_karma_;
    dgs2_2dtheta_ = &dgs2_2dtheta_karma_;
    //sort_nodeset();

  }else if("pool" == paramList.get<std::string> (TusastestNameString)){
    exit(0);
    D_ = 4.;
    T_m_ = 1.55;
    T_inf_ = 1.;
    alpha_ = 191.82;
    eps_ = .04;
    //eps_ = .2;
    eps_0_ = 1.;
    M_= 4.;
    theta_0_ =0.;
    R_0_ =.3;
    //R_0_ =.1;
    
    //function pointers
    hp1_ = &cummins::hp1_cummins_;
    hpp1_ = &cummins::hpp1_cummins_;
    w_ = &cummins::w_cummins_;
    m_ = &cummins::m_cummins_;
    //m_ = &m_furtado_;
    rand_phi_ = &rand_phi_furtado_;
    //rand_phi_ = &rand_phi_zero_;
    gp1_ = &cummins::gp1_cummins_;
    gpp1_ = &cummins::gpp1_cummins_;
    //hp2_ = &hp2_cummins_;
    hp2_ = &hp2_furtado_;

    gs2_ = &cummins::gs2_cummins_;
    dgs2_2dtheta_ = &cummins::dgs2_2dtheta_cummins_;
  

  }else if("branch" == paramList.get<std::string> (TusastestNameString)){
    exit(0);
    D_ = 4.;
    T_m_ = 1.55;
    T_inf_ = 1.25;
    alpha_ = 191.82;
    eps_ = .04;
    //eps_ = .2;
    eps_0_ = 1.;
    M_= 4.;
    theta_0_ =3.14159/4.;
    R_0_ =.3;
    //R_0_ =.1;
    
    //function pointers
    hp1_ = &cummins::hp1_cummins_;
    hpp1_ = &cummins::hpp1_cummins_;
    w_ = &cummins::w_cummins_;
    m_ = &cummins::m_cummins_;
    //m_ = &m_furtado_;
    rand_phi_ = &rand_phi_furtado_;
    //rand_phi_ = &rand_phi_zero_;
    gp1_ = &cummins::gp1_cummins_;
    gpp1_ = &cummins::gpp1_cummins_;
    //hp2_ = &hp2_cummins_;
    hp2_ = &hp2_furtado_;

    gs2_ = &cummins::gs2_cummins_;
    dgs2_2dtheta_ = &cummins::dgs2_2dtheta_cummins_;
  

  }else if("cummins" == paramList.get<std::string> (TusastestNameString)){
    
    numeqs_ = 2;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &cummins::residual_heat_;
    (*residualfunc_)[1] = &cummins::residual_phase_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &cummins::prec_heat_;
    (*preconfunc_)[1] = &cummins::prec_phase_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &cummins::init_heat_;
    //(*initfunc_)[0] = &cummins::init_heat_const_;
    (*initfunc_)[1] = &cummins::init_phase_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    (*varnames_)[1] = "phi";

    dirichletfunc_ = NULL;

    neumannfunc_ = NULL;

    paramfunc_ = cummins::param_;

  }else if("heat" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 1;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &heat::residual_heat_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &heat::prec_heat_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &heat::init_heat_test_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);

//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][nodeset id]
//  [variable index][nodeset index]
    (*dirichletfunc_)[0][0] = &dbc_zero_;							 
    (*dirichletfunc_)[0][1] = &dbc_zero_;						 
    (*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[0][3] = &dbc_zero_;

    neumannfunc_ = NULL;

  }else if("timeadapt" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 1;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &heat::residual_heat_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &heat::prec_heat_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &heat::init_heat_test_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);

//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][nodeset id]
//  [variable index][nodeset index]
    (*dirichletfunc_)[0][0] = &dbc_zero_; 
    (*dirichletfunc_)[0][1] = &dbc_zero_;						 
    (*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[0][3] = &dbc_zero_;

    neumannfunc_ = NULL;

    post_proc.push_back(new post_process(comm_,
					 mesh_,
					 (int)0,
					 post_process::NORMRMS,
					 0,
					 "pp",
					 16));
    post_proc[0].postprocfunc_ = &timeadapt::postproc1_;

    post_proc.push_back(new post_process(comm_,
					 mesh_,
					 (int)1,
					 post_process::MAXVALUE,
					 0,
					 "pp",
					 16));
    post_proc[1].postprocfunc_ = &timeadapt::normu_;

    post_proc.push_back(new post_process(comm_,
					 mesh_,
					 (int)2,
					 post_process::MAXVALUE,
					 0,
					 "pp",
					 16));
    post_proc[2].postprocfunc_ = &timeadapt::postproc2_;


  }else if("timeonly" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 1;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &timeonly::residual_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &heat::prec_heat_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &timeonly::init_test_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = NULL;

    neumannfunc_ = NULL;

    post_proc.push_back(new post_process(comm_,
					 mesh_,
					 (int)0,
					 post_process::MAXVALUE,
					 0,
					 "pp",
					 16));
    post_proc[0].postprocfunc_ = &timeadapt::postproc1_;

//     post_proc.push_back(new post_process(comm_,
// 					 mesh_,
// 					 (int)1,
// 					 post_process::MAXVALUE,
// 					 0,
// 					 "pp",
// 					 16));
//     post_proc[1].postprocfunc_ = &timeadapt::postproc2_;

//     post_proc.push_back(new post_process(comm_,
// 					 mesh_,
// 					 (int)2,
// 					 post_process::MAXVALUE,
// 					 0,
// 					 "pp",
// 					 16));
//     post_proc[2].postprocfunc_ = &timeonly::postproc3_;




  }else if("autocatalytic4" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 4;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &autocatalytic4::residual_a_;
    (*residualfunc_)[1] = &autocatalytic4::residual_b_;
    (*residualfunc_)[2] = &autocatalytic4::residual_ab_;
    (*residualfunc_)[3] = &autocatalytic4::residual_c_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &heat::prec_heat_test_;
    (*preconfunc_)[1] = &heat::prec_heat_test_;
    (*preconfunc_)[2] = &heat::prec_heat_test_;
    (*preconfunc_)[3] = &heat::prec_heat_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &autocatalytic4::init_a_;
    (*initfunc_)[1] = &autocatalytic4::init_b_;
    (*initfunc_)[2] = &autocatalytic4::init_ab_;
    (*initfunc_)[3] = &autocatalytic4::init_c_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "a";
    (*varnames_)[1] = "b";
    (*varnames_)[2] = "ab";
    (*varnames_)[3] = "c";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = NULL;

    neumannfunc_ = NULL;

  }else if("chem" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 3;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &chem::residual_a_;
    (*residualfunc_)[1] = &chem::residual_b_;
    (*residualfunc_)[2] = &chem::residual_c_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &heat::prec_heat_test_;
    (*preconfunc_)[1] = &heat::prec_heat_test_;
    (*preconfunc_)[2] = &heat::prec_heat_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &chem::init_a_;
    (*initfunc_)[1] = &chem::init_b_;
    (*initfunc_)[2] = &chem::init_c_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "a";
    (*varnames_)[1] = "b";
    (*varnames_)[2] = "c";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = NULL;

    neumannfunc_ = NULL;

  }else if("omp" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 1;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &heat::residual_heat_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &heat::prec_heat_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &heat::init_heat_test_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";

    // numeqs_ number of variables(equations)
    dirichletfunc_ = NULL;

    neumannfunc_ = NULL;


  }else if("neumann" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 1;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &heat::residual_heat_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &heat::prec_heat_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    //(*initfunc_)[0] = &init_neumann_test_;
    (*initfunc_)[0] = &init_zero_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);

//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][nodeset id]
//  [variable index][nodeset index]
    //(*dirichletfunc_)[0][0] = &dbc_zero_;							 
    //(*dirichletfunc_)[0][1] = &dbc_zero_;						 
    //(*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[0][3] = &dbc_zero_;

    // numeqs_ number of variables(equations) 
    neumannfunc_ = new std::vector<std::map<int,NBCFUNC>>(numeqs_);
    //neumannfunc_ = NULL;
    //(*neumannfunc_)[0][0] = &nbc_one_;							 
    (*neumannfunc_)[0][1] = &nbc_one_;						 
    //(*neumannfunc_)[0][2] = &nbc_zero_;						 
    //(*neumannfunc_)[0][3] = &nbc_zero_;

  }else if("robin_steadystate" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 1;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &robin_steadystate::residual_robin_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &robin_steadystate::prec_robin_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    //(*initfunc_)[0] = &init_neumann_test_;
    (*initfunc_)[0] = &init_zero_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);

//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][nodeset id]
//  [variable index][nodeset index]
    //(*dirichletfunc_)[0][0] = &dbc_zero_;							 
    //(*dirichletfunc_)[0][1] = &dbc_zero_;						 
    //(*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[0][3] = &dbc_zero_;

    // numeqs_ number of variables(equations) 
    neumannfunc_ = new std::vector<std::map<int,NBCFUNC>>(numeqs_);
    //neumannfunc_ = NULL;
    //(*neumannfunc_)[0][0] = &nbc_one_;							 
    (*neumannfunc_)[0][1] = &robin_steadystate::nbc_robin_test_;						 
    //(*neumannfunc_)[0][2] = &nbc_zero_;						 
    //(*neumannfunc_)[0][3] = &nbc_zero_;

  }else if("robin" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 1;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &robin::residual_robin_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &robin::prec_robin_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    //(*initfunc_)[0] = &init_neumann_test_;
    (*initfunc_)[0] = &robin::init_robin_test_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);

//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][nodeset id]
//  [variable index][nodeset index]
    //(*dirichletfunc_)[0][0] = &dbc_zero_;							 
    //(*dirichletfunc_)[0][1] = &dbc_zero_;						 
    //(*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[0][3] = &dbc_zero_;

    // numeqs_ number of variables(equations) 
    neumannfunc_ = new std::vector<std::map<int,NBCFUNC>>(numeqs_);
    //neumannfunc_ = NULL;
    //(*neumannfunc_)[0][0] = &nbc_one_;							 
    (*neumannfunc_)[0][1] = &robin::nbc_robin_test_;						 
    //(*neumannfunc_)[0][1] = &nbc_zero_;						 
    //(*neumannfunc_)[0][2] = &nbc_zero_;						 
    //(*neumannfunc_)[0][3] = &nbc_zero_;

  }else if("liniso" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 3;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &liniso::residual_liniso_x_test_;
    (*residualfunc_)[1] = &liniso::residual_liniso_y_test_;
    (*residualfunc_)[2] = &liniso::residual_liniso_z_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &liniso::prec_liniso_x_test_;
    (*preconfunc_)[1] = &liniso::prec_liniso_y_test_;
    (*preconfunc_)[2] = &liniso::prec_liniso_z_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    //(*initfunc_)[0] = &init_neumann_test_;
    (*initfunc_)[0] = &init_zero_;
    (*initfunc_)[1] = &init_zero_;
    (*initfunc_)[2] = &init_zero_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "x_disp";
    (*varnames_)[1] = "y_disp";
    (*varnames_)[2] = "z_disp";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);

//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][nodeset id]
//  [variable index][nodeset index]
    //(*dirichletfunc_)[0][0] = &dbc_zero_;							 
    (*dirichletfunc_)[0][3] = &dbc_zero_;						 
    (*dirichletfunc_)[1][2] = &dbc_zero_;						 
    //(*dirichletfunc_)[1][4] = &dbc_mone_;

    // numeqs_ number of variables(equations) 
    neumannfunc_ = new std::vector<std::map<int,NBCFUNC>>(numeqs_);
    //neumannfunc_ = NULL;
    //(*neumannfunc_)[0][0] = &nbc_one_;							 
    //(*neumannfunc_)[0][1] = &nbc_robin_test_;						 
    //(*neumannfunc_)[0][2] = &nbc_zero_;						 
    //(*neumannfunc_)[0][3] = &nbc_zero_;
    (*neumannfunc_)[1][4] = &nbc_mone_;


    //std::cout<<"liniso"<<std::endl;
    //exit(0);

  }else if("linisobodyforce" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 3;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &liniso::residual_liniso_x_test_;
    (*residualfunc_)[1] = &liniso::residual_linisobodyforce_y_test_;
    (*residualfunc_)[2] = &liniso::residual_liniso_z_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &liniso::prec_liniso_x_test_;
    (*preconfunc_)[1] = &liniso::prec_liniso_y_test_;
    (*preconfunc_)[2] = &liniso::prec_liniso_z_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    //(*initfunc_)[0] = &init_neumann_test_;
    (*initfunc_)[0] = &init_zero_;
    (*initfunc_)[1] = &init_zero_;
    (*initfunc_)[2] = &init_zero_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "x_disp";
    (*varnames_)[1] = "y_disp";
    (*varnames_)[2] = "z_disp";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);

//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][nodeset id]
//  [variable index][nodeset index]
    //(*dirichletfunc_)[0][0] = &dbc_zero_;							 
    (*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[1][2] = &dbc_zero_;						 
    (*dirichletfunc_)[2][2] = &dbc_zero_;						 
    //(*dirichletfunc_)[1][4] = &dbc_mone_;

    // numeqs_ number of variables(equations) 
    neumannfunc_ = new std::vector<std::map<int,NBCFUNC>>(numeqs_);
    //neumannfunc_ = NULL;
    //(*neumannfunc_)[0][0] = &nbc_one_;							 
    //(*neumannfunc_)[0][1] = &nbc_robin_test_;						 
    //(*neumannfunc_)[0][2] = &nbc_zero_;						 
    //(*neumannfunc_)[0][3] = &nbc_zero_;
    //(*neumannfunc_)[1][4] = &nbc_mone_;


    //std::cout<<"liniso"<<std::endl;
    //exit(0);


  }else if("linisoheat" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 4;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &liniso::residual_linisoheat_x_test_;
    (*residualfunc_)[1] = &liniso::residual_linisoheat_y_test_;
    (*residualfunc_)[2] = &liniso::residual_linisoheat_z_test_;
    (*residualfunc_)[3] = &liniso::residual_divgrad_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &liniso::prec_liniso_x_test_;
    (*preconfunc_)[1] = &liniso::prec_liniso_y_test_;
    (*preconfunc_)[2] = &liniso::prec_liniso_z_test_;
    (*preconfunc_)[3] = &heat::prec_heat_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    //(*initfunc_)[0] = &init_neumann_test_;
    (*initfunc_)[0] = &init_zero_;
    (*initfunc_)[1] = &init_zero_;
    (*initfunc_)[2] = &init_zero_;
    (*initfunc_)[3] = &init_zero_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "x_disp";
    (*varnames_)[1] = "y_disp";
    (*varnames_)[2] = "z_disp";
    (*varnames_)[3] = "u";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);

//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][nodeset id]
//  [variable index][nodeset index]
    //(*dirichletfunc_)[0][0] = &dbc_zero_;							 
    (*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[1][2] = &dbc_zero_;						 
    (*dirichletfunc_)[2][2] = &dbc_zero_;						 
    (*dirichletfunc_)[3][2] = &dbc_ten_;						 
    (*dirichletfunc_)[3][4] = &dbc_zero_;

    // numeqs_ number of variables(equations) 
    neumannfunc_ = new std::vector<std::map<int,NBCFUNC>>(numeqs_);
    //neumannfunc_ = NULL;
    //(*neumannfunc_)[0][0] = &nbc_one_;							 
    //(*neumannfunc_)[0][1] = &nbc_robin_test_;						 
    //(*neumannfunc_)[0][2] = &nbc_zero_;						 
    //(*neumannfunc_)[0][3] = &nbc_zero_;
    //(*neumannfunc_)[1][4] = &nbc_mone_;


    //std::cout<<"liniso"<<std::endl;
    //exit(0);



  }else if("uehara" == paramList.get<std::string> (TusastestNameString)){

    bool stress = false;
    //bool stress = true;

    numeqs_ = 4;
    if(stress) numeqs_ = 7;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &uehara::residual_phase_;
    (*residualfunc_)[1] = &uehara::residual_heat_;
    (*residualfunc_)[2] = &uehara::residual_liniso_x_test_;
    (*residualfunc_)[3] = &uehara::residual_liniso_y_test_;
    if(stress)(*residualfunc_)[4] = &uehara::residual_stress_x_test_;
    if(stress)(*residualfunc_)[5] = &uehara::residual_stress_y_test_;
    if(stress)(*residualfunc_)[6] = &uehara::residual_stress_xy_test_;
    
    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &uehara::prec_phase_;
    (*preconfunc_)[1] = &uehara::prec_heat_;
    (*preconfunc_)[2] = &uehara::prec_liniso_x_test_;
    (*preconfunc_)[3] = &uehara::prec_liniso_y_test_;
    if(stress)(*preconfunc_)[4] = &uehara::prec_stress_test_;
    if(stress)(*preconfunc_)[5] = &uehara::prec_stress_test_;
    if(stress)(*preconfunc_)[6] = &uehara::prec_stress_test_;
    
    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    //(*initfunc_)[0] = &uehara::init_phase_;
    (*initfunc_)[0] = &uehara::init_phase_c_;
    (*initfunc_)[1] = &uehara::init_heat_;
    //(*initfunc_)[1] = &uehara::init_heat_seed_c_;
    (*initfunc_)[2] = &init_zero_;
    (*initfunc_)[3] = &init_zero_;
    if(stress)(*initfunc_)[4] = &init_zero_;
    if(stress)(*initfunc_)[5] = &init_zero_;
    if(stress)(*initfunc_)[6] = &init_zero_;
    
    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "phi";
    (*varnames_)[1] = "u";
    (*varnames_)[2] = "dispx";
    (*varnames_)[3] = "dispy";
    if(stress)(*varnames_)[4] = "x_stress";
    if(stress)(*varnames_)[5] = "y_stress";
    if(stress)(*varnames_)[6] = "xy_stress";
    
    // numeqs_ number of variables(equations) 
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
    
    //dirichletfunc_ = NULL;
    //  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
    //               [numeq][nodeset id]
    //  [variable index][nodeset index]						 
//     (*dirichletfunc_)[1][1] = &uehara::dbc_;						 
//     (*dirichletfunc_)[1][2] = &uehara::dbc_;

    (*dirichletfunc_)[2][3] = &dbc_zero_;
    //cn failing in parallel for some wierd reason
    (*dirichletfunc_)[3][0] = &dbc_zero_;


    //(*dirichletfunc_)[4][1] = &dbc_zero_;
    
    // numeqs_ number of variables(equations) 
    neumannfunc_ = new std::vector<std::map<int,NBCFUNC>>(numeqs_);
    //neumannfunc_ = NULL;
//     (*neumannfunc_)[0][0] = &nbc_zero_;
//     (*neumannfunc_)[0][1] = &nbc_zero_;
//     (*neumannfunc_)[0][2] = &nbc_zero_;
//     (*neumannfunc_)[0][3] = &nbc_zero_;
    (*neumannfunc_)[1][1] = &uehara::conv_bc_;
    (*neumannfunc_)[1][2] = &uehara::conv_bc_;
    
    post_proc.push_back(new post_process(comm_,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &uehara::postproc_stress_x_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)1));
//     post_proc[1].postprocfunc_ = &uehara::postproc_stress_y_;
    post_proc[1].postprocfunc_ = &uehara::postproc_stress_xd_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)2));
//     post_proc[2].postprocfunc_ = &uehara::postproc_stress_xy_;
    post_proc[2].postprocfunc_ = &uehara::postproc_stress_eq_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)3));
    post_proc[3].postprocfunc_ = &uehara::postproc_stress_eqd_;
//     post_proc.push_back(new post_process(comm_,mesh_,(int)4));
//     post_proc[4].postprocfunc_ = &uehara::postproc_strain_;

    //std::cout<<"uehara"<<std::endl;
    //exit(0);

}else if("uehara2" == paramList.get<std::string> (TusastestNameString)){

    bool stress = false;
    //bool stress = true;

    numeqs_ = 4;
    if(stress) numeqs_ = 7;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &uehara::residual_phase_;
    (*residualfunc_)[1] = &uehara2::residual_heat_;
    (*residualfunc_)[2] = &uehara::residual_liniso_x_test_;
    (*residualfunc_)[3] = &uehara::residual_liniso_y_test_;
    if(stress)(*residualfunc_)[4] = &uehara::residual_stress_x_test_;
    if(stress)(*residualfunc_)[5] = &uehara::residual_stress_y_test_;
    if(stress)(*residualfunc_)[6] = &uehara::residual_stress_xy_test_;
    
    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &uehara::prec_phase_;
    (*preconfunc_)[1] = &uehara::prec_heat_;
    (*preconfunc_)[2] = &uehara::prec_liniso_x_test_;
    (*preconfunc_)[3] = &uehara::prec_liniso_y_test_;
    if(stress)(*preconfunc_)[4] = &uehara::prec_stress_test_;
    if(stress)(*preconfunc_)[5] = &uehara::prec_stress_test_;
    if(stress)(*preconfunc_)[6] = &uehara::prec_stress_test_;
    
    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    //(*initfunc_)[0] = &uehara::init_phase_;
    (*initfunc_)[0] = &uehara2::init_phase_c_;
    (*initfunc_)[1] = &uehara2::init_heat_;
    //(*initfunc_)[1] = &uehara::init_heat_seed_c_;
    (*initfunc_)[2] = &init_zero_;
    (*initfunc_)[3] = &init_zero_;
    if(stress)(*initfunc_)[4] = &init_zero_;
    if(stress)(*initfunc_)[5] = &init_zero_;
    if(stress)(*initfunc_)[6] = &init_zero_;
    
    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "phi";
    (*varnames_)[1] = "u";
    (*varnames_)[2] = "dispx";
    (*varnames_)[3] = "dispy";
    if(stress)(*varnames_)[4] = "x_stress";
    if(stress)(*varnames_)[5] = "y_stress";
    if(stress)(*varnames_)[6] = "xy_stress";
    
    // numeqs_ number of variables(equations) 
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
    
    //dirichletfunc_ = NULL;
    //  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
    //               [numeq][nodeset id]
    //  [variable index][nodeset index]						 
//     (*dirichletfunc_)[1][1] = &uehara::dbc_;						 
//     (*dirichletfunc_)[1][2] = &uehara::dbc_;
    (*dirichletfunc_)[2][1] = &dbc_zero_;
    (*dirichletfunc_)[2][3] = &dbc_zero_;
    (*dirichletfunc_)[3][0] = &dbc_zero_;
    (*dirichletfunc_)[3][2] = &dbc_zero_;
    //(*dirichletfunc_)[4][1] = &dbc_zero_;
    
    // numeqs_ number of variables(equations) 
    neumannfunc_ = new std::vector<std::map<int,NBCFUNC>>(numeqs_);
    //neumannfunc_ = NULL;
//     (*neumannfunc_)[0][0] = &nbc_zero_;
//     (*neumannfunc_)[0][1] = &nbc_zero_;
//     (*neumannfunc_)[0][2] = &nbc_zero_;
//     (*neumannfunc_)[0][3] = &nbc_zero_;
    (*neumannfunc_)[1][0] = &uehara::conv_bc_;
    (*neumannfunc_)[1][1] = &uehara::conv_bc_;
    (*neumannfunc_)[1][2] = &uehara::conv_bc_;
    (*neumannfunc_)[1][3] = &uehara::conv_bc_;
    
    post_proc.push_back(new post_process(comm_,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &uehara::postproc_stress_x_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)1));
//     post_proc[1].postprocfunc_ = &uehara::postproc_stress_y_;
    post_proc[1].postprocfunc_ = &uehara::postproc_stress_xd_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)2));
//     post_proc[2].postprocfunc_ = &uehara::postproc_stress_xy_;
    post_proc[2].postprocfunc_ = &uehara::postproc_stress_eq_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)3));
    post_proc[3].postprocfunc_ = &uehara::postproc_stress_eqd_;
//     post_proc.push_back(new post_process(comm_,mesh_,(int)4));
//     post_proc[4].postprocfunc_ = &uehara::postproc_strain_;

    //std::cout<<"uehara"<<std::endl;
    //exit(0);


  }else if("laplace" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 1;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &laplace::residual_heat_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &heat::prec_heat_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &init_zero_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);

//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][nodeset id]
//  [variable index][nodeset index]
    //(*dirichletfunc_)[0][0] = &dbc_zero_;							 
    (*dirichletfunc_)[0][1] = &dbc_zero_;						 
    //(*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[0][3] = &dbc_zero_;

    neumannfunc_ = NULL;



  }else if("coupledstress" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 5;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &coupledstress::residual_liniso_x_test_;
    (*residualfunc_)[1] = &coupledstress::residual_liniso_y_test_;
    (*residualfunc_)[2] = &coupledstress::residual_stress_x_test_;
    (*residualfunc_)[3] = &coupledstress::residual_stress_y_test_;
    (*residualfunc_)[4] = &coupledstress::residual_stress_xy_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &coupledstress::prec_liniso_x_test_;
    (*preconfunc_)[1] = &coupledstress::prec_liniso_y_test_;
    (*preconfunc_)[2] = &coupledstress::prec_stress_test_;
    (*preconfunc_)[3] = &coupledstress::prec_stress_test_;
    (*preconfunc_)[4] = &coupledstress::prec_stress_test_;

//     initfunc_ = new  std::vector<double (*)(const double &x,
// 					    const double &y,
// 					    const double &z)>(numeqs_);
    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &init_zero_;
    (*initfunc_)[1] = &init_zero_;
    (*initfunc_)[2] = &init_zero_;
    (*initfunc_)[3] = &init_zero_;
    (*initfunc_)[4] = &init_zero_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "x_disp";
    (*varnames_)[1] = "y_disp";
    (*varnames_)[2] = "x_stress";
    (*varnames_)[3] = "y_stress";
    (*varnames_)[4] = "xy_stress";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);

//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][nodeset id]
//  [variable index][nodeset index]
    (*dirichletfunc_)[0][0] = &dbc_zero_;							 
    //(*dirichletfunc_)[0][3] = &dbc_zero_;						 
    (*dirichletfunc_)[1][0] = &dbc_zero_;						 
    //(*dirichletfunc_)[1][4] = &dbc_mone_;

    // numeqs_ number of variables(equations) 
    neumannfunc_ = new std::vector<std::map<int,NBCFUNC>>(numeqs_);
    //neumannfunc_ = NULL;
    //(*neumannfunc_)[0][0] = &nbc_one_;							 
    //(*neumannfunc_)[0][1] = &nbc_robin_test_;						 
    //(*neumannfunc_)[0][2] = &nbc_zero_;						 
    //(*neumannfunc_)[0][3] = &nbc_zero_;
    (*neumannfunc_)[1][2] = &nbc_mone_;


    //cn this needs to be better...
    post_proc.push_back(new post_process(comm_,mesh_,(int)0,post_process::SCALAR_OP::NORM1));
    post_proc[0].postprocfunc_ = &coupledstress::postproc_stress_x_;
    post_proc.push_back(new post_process(comm_,mesh_,(int)1,post_process::SCALAR_OP::MEANVALUE,(int)5));
    post_proc[1].postprocfunc_ = &coupledstress::postproc_stress_y_;
    post_proc.push_back(new post_process(comm_,mesh_,(int)2));
    post_proc[2].postprocfunc_ = &coupledstress::postproc_stress_xy_;


    //std::cout<<"coupledstress"<<std::endl;
    //exit(0);


  }else if("farzadi" == paramList.get<std::string> (TusastestNameString)){
    //farzadi test

    numeqs_ = 2;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &farzadi::init_conc_farzadi_;
    (*initfunc_)[1] = &farzadi::init_phase_farzadi_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &farzadi::residual_conc_farzadi_;
    (*residualfunc_)[1] = &farzadi::residual_phase_farzadi_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &farzadi::prec_conc_farzadi_;
    (*preconfunc_)[1] = &farzadi::prec_phase_farzadi_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    (*varnames_)[1] = "phi";

    dirichletfunc_ = NULL;

//     dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
    //(*dirichletfunc_)[0][1] = &dbc_mone_;	
    //(*dirichletfunc_)[0][3] = &dbc_zero_;
    //(*dirichletfunc_)[1][1] = &dbc_mone_;
    //(*dirichletfunc_)[1][3] = &dbc_one_;

    neumannfunc_ = NULL;

    post_proc.push_back(new post_process(comm_,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &farzadi::postproc_c_;
    post_proc.push_back(new post_process(comm_,mesh_,(int)1));
    post_proc[1].postprocfunc_ = &farzadi::postproc_t_;

    paramfunc_ = farzadi::param_;
						 
    //exit(0);
  }else if("cahnhilliard" == paramList.get<std::string> (TusastestNameString)){
    //std::cout<<"cahnhilliard"<<std::endl;

    numeqs_ = 2;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &cahnhilliard::init_c_;
    (*initfunc_)[1] = &cahnhilliard::init_mu_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &cahnhilliard::residual_c_;
    (*residualfunc_)[1] = &cahnhilliard::residual_mu_;

    preconfunc_ = NULL;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "c";
    (*varnames_)[1] = "mu";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][nodeset id]
//  [variable index][nodeset index]
    (*dirichletfunc_)[0][1] = &dbc_zero_;
    (*dirichletfunc_)[0][3] = &dbc_zero_;
    (*dirichletfunc_)[1][1] = &dbc_zero_;
    (*dirichletfunc_)[1][3] = &dbc_zero_;

    neumannfunc_ = NULL;
    paramfunc_ = cahnhilliard::param_;

    //exit(0);
  }else if("grain" == paramList.get<std::string> (TusastestNameString)){


    Teuchos::ParameterList *problemList;
    problemList = &paramList.sublist ( "ProblemParams", false );

    numeqs_ = problemList->get<int>("numgrain");

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    for( int k = 0; k < numeqs_; k++ )(*initfunc_)[k] = &grain::init_;
//     (*initfunc_)[0] = &grain::init_;
//     (*initfunc_)[1] = &grain::init_;
//     (*initfunc_)[2] = &grain::init_;
//     (*initfunc_)[3] = &grain::init_;
//     (*initfunc_)[4] = &grain::init_;
//     (*initfunc_)[5] = &grain::init_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    for( int k = 0; k < numeqs_; k++ )(*residualfunc_)[k] = &grain::residual_;
//     (*residualfunc_)[0] = &grain::residual_;
//     (*residualfunc_)[1] = &grain::residual_;
//     (*residualfunc_)[2] = &grain::residual_;
//     (*residualfunc_)[3] = &grain::residual_;
//     (*residualfunc_)[4] = &grain::residual_;
//     (*residualfunc_)[5] = &grain::residual_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    for( int k = 0; k < numeqs_; k++ )(*preconfunc_)[k] = &grain::prec_;

    varnames_ = new std::vector<std::string>(numeqs_);
    for( int k = 0; k < numeqs_; k++ ) (*varnames_)[k] = "n"+std::to_string(k);


    dirichletfunc_ = NULL;
    neumannfunc_ = NULL;

    post_proc.push_back(new post_process(comm_,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &grain::postproc_;

    paramfunc_ = grain::param_;


  }else if("grainp" == paramList.get<std::string> (TusastestNameString)){


    Teuchos::ParameterList *problemList;
    problemList = &paramList.sublist ( "ProblemParams", false );

    numeqs_ = problemList->get<int>("numgrain");

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    for( int k = 0; k < numeqs_; k++ )(*initfunc_)[k] = &grain::init_;
//     (*initfunc_)[0] = &grain::init_;
//     (*initfunc_)[1] = &grain::init_;
//     (*initfunc_)[2] = &grain::init_;
//     (*initfunc_)[3] = &grain::init_;
//     (*initfunc_)[4] = &grain::init_;
//     (*initfunc_)[5] = &grain::init_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    for( int k = 0; k < numeqs_; k++ )(*residualfunc_)[k] = &grain::residual_;
//     (*residualfunc_)[0] = &grain::residual_;
//     (*residualfunc_)[1] = &grain::residual_;
//     (*residualfunc_)[2] = &grain::residual_;
//     (*residualfunc_)[3] = &grain::residual_;
//     (*residualfunc_)[4] = &grain::residual_;
//     (*residualfunc_)[5] = &grain::residual_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    for( int k = 0; k < numeqs_; k++ )(*preconfunc_)[k] = &grain::prec_;

    varnames_ = new std::vector<std::string>(numeqs_);
    for( int k = 0; k < numeqs_; k++ ) (*varnames_)[k] = "n"+std::to_string(k);


    dirichletfunc_ = NULL;
    neumannfunc_ = NULL;

    post_proc.push_back(new post_process(comm_,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &grain::postproc_;

    paramfunc_ = grain::param_;

#ifdef PERIODIC_BC
    periodic_bc_.push_back(new periodic_bc(0,2,numeqs_,mesh_,comm_));
    periodic_bc_.push_back(new periodic_bc(1,3,numeqs_,mesh_,comm_));
    if(3 == mesh_->get_num_dim() ) periodic_bc_.push_back(new periodic_bc(4,5,numeqs_,mesh_,comm_));
    for( int k = 0; k < numeqs_; k++ ){
      periodic_bc_[0].add_eqn_index(k);
      periodic_bc_[1].add_eqn_index(k);
      if(3 == mesh_->get_num_dim() ) periodic_bc_[2].add_eqn_index(k);
    }
#else
    periodicbc_ = new std::vector<std::vector<std::pair<int,int>>>(numeqs_);
//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][bc number][nodeset id 1][nodeset id 2]
    for( int k = 0; k < numeqs_; k++ ){
      (*periodicbc_)[k].push_back(std::make_pair(0,2));
      (*periodicbc_)[k].push_back(std::make_pair(1,3));
      if(3 == mesh_->get_num_dim() ) (*periodicbc_)[k].push_back(std::make_pair(4,5));
    }
#endif




  }else if("periodic" == paramList.get<std::string> (TusastestNameString)){
    //std::cout<<"periodic"<<std::endl;
    
    numeqs_ = 1;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &init_zero_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &periodic::residual_;

    preconfunc_ = NULL;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";

    dirichletfunc_ = NULL;

    neumannfunc_ = NULL;

#ifdef PERIODIC_BC
    periodic_bc_.push_back(new periodic_bc(1,3,numeqs_,mesh_,comm_));
    periodic_bc_[0].add_eqn_index(0);

#else
    periodicbc_ = new std::vector<std::vector<std::pair<int,int>>>(numeqs_);
//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][bc number][nodeset id 1][nodeset id 2]
    (*periodicbc_)[0].push_back(std::make_pair(1,3));
#endif
    //exit(0);





  }else if("periodicdbg" == paramList.get<std::string> (TusastestNameString)){
    //std::cout<<"periodic"<<std::endl;
    
    Teuchos::ParameterList *problemList;
    problemList = &paramList.sublist ( "ProblemParams", false );

    numeqs_ = problemList->get<int>("numgrain");

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &init_zero_;
    if(2==numeqs_) (*initfunc_)[1] = &init_zero_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &periodic::residual_;
    //if(2==numeqs_) (*residualfunc_)[1] = &residual_heat_test_;
    if(2==numeqs_) (*residualfunc_)[1] = &periodic::residual_;

    preconfunc_ = NULL;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    if(2==numeqs_) (*varnames_)[1] = "v";

    dirichletfunc_ = NULL;

    neumannfunc_ = NULL;

#ifdef PERIODIC_BC
    periodic_bc_.push_back(new periodic_bc(0,2,numeqs_,mesh_,comm_));
    periodic_bc_.push_back(new periodic_bc(1,3,numeqs_,mesh_,comm_));
    if(3 == mesh_->get_num_dim() ) periodic_bc_.push_back(new periodic_bc(4,5,numeqs_,mesh_,comm_));
    for( int k = 0; k < numeqs_; k++ ){
      periodic_bc_[0].add_eqn_index(k);
      periodic_bc_[1].add_eqn_index(k);
      if(3 == mesh_->get_num_dim() ) periodic_bc_[2].add_eqn_index(k);
    }

#else
    periodicbc_ = new std::vector<std::vector<std::pair<int,int>>>(numeqs_);
//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][bc number][nodeset id 1][nodeset id 2]
    (*periodicbc_)[0].push_back(std::make_pair(1,3));
    if(2==numeqs_) (*periodicbc_)[1].push_back(std::make_pair(1,3));
#endif
    //exit(0);




  }else if("kundin" == paramList.get<std::string> (TusastestNameString)){
    //std::cout<<"kundin"<<std::endl;
    //Teuchos::ParameterList *problemList;
    //problemList = &paramList.sublist ( "ProblemParams", false );

    numeqs_ = 7;
  
    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &kundin::cinit_;
    (*initfunc_)[1] = &kundin::cinit_;
    (*initfunc_)[2] = &kundin::cinit_;
    (*initfunc_)[3] = &kundin::cinit_;
    (*initfunc_)[4] = &kundin::cinit_;
    (*initfunc_)[5] = &kundin::cinit_;
    (*initfunc_)[6] = &kundin::phiinit_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &kundin::cresidual_;
    (*residualfunc_)[1] = &kundin::cresidual_;
    (*residualfunc_)[2] = &kundin::cresidual_;
    (*residualfunc_)[3] = &kundin::cresidual_;
    (*residualfunc_)[4] = &kundin::cresidual_;
    (*residualfunc_)[5] = &kundin::cresidual_;
    (*residualfunc_)[6] = &kundin::phiresidual_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &kundin::cprec_;
    (*preconfunc_)[1] = &kundin::cprec_;
    (*preconfunc_)[2] = &kundin::cprec_;
    (*preconfunc_)[3] = &kundin::cprec_;
    (*preconfunc_)[4] = &kundin::cprec_;
    (*preconfunc_)[5] = &kundin::cprec_;
    (*preconfunc_)[6] = &kundin::phiprec_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "cr";
    (*varnames_)[1] = "fe";
    (*varnames_)[2] = "mo";
    (*varnames_)[3] = "nb";
    (*varnames_)[4] = "ti";
    (*varnames_)[5] = "al";
    (*varnames_)[6] = "phi";

    dirichletfunc_ = NULL;
    //dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][nodeset id]
//  [variable index][nodeset index]
//     (*dirichletfunc_)[0][2] = &kundin::dbc0_;
//     (*dirichletfunc_)[1][2] = &kundin::dbc1_;
//     (*dirichletfunc_)[2][2] = &kundin::dbc2_;
//     (*dirichletfunc_)[3][2] = &kundin::dbc3_;
//     (*dirichletfunc_)[4][2] = &kundin::dbc4_;
//     (*dirichletfunc_)[5][2] = &kundin::dbc5_;
//     (*dirichletfunc_)[6][2] = &dbc_zero_;
//     (*dirichletfunc_)[6][0] = &dbc_one_;

    neumannfunc_ = NULL;

#ifdef PERIODIC_BC
//     periodic_bc_.resize(6);
//     periodic_bc_[0].push_back(new periodic_bc(1,3,0,numeqs_,mesh_,comm_));
//     periodic_bc_[1].push_back(new periodic_bc(1,3,1,numeqs_,mesh_,comm_));
//     periodic_bc_[2].push_back(new periodic_bc(1,3,2,numeqs_,mesh_,comm_));
//     periodic_bc_[3].push_back(new periodic_bc(1,3,3,numeqs_,mesh_,comm_));
//     periodic_bc_[4].push_back(new periodic_bc(1,3,4,numeqs_,mesh_,comm_));
//     periodic_bc_[5].push_back(new periodic_bc(1,3,5,numeqs_,mesh_,comm_));
#else
#endif
     post_proc.push_back(new post_process(comm_,mesh_,(int)0));
     post_proc[0].postprocfunc_ = &kundin::postproc_;


    //exit(0);


  }else if("kundinphi" == paramList.get<std::string> (TusastestNameString)){
    //std::cout<<"kundin"<<std::endl;
    //Teuchos::ParameterList *problemList;
    //problemList = &paramList.sublist ( "ProblemParams", false );

    numeqs_ = 1;
  
    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &kundin::phiinit_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &kundin::phiresidual_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &kundin::phiprec_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "phi";

    dirichletfunc_ = NULL;
    //dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][nodeset id]
//  [variable index][nodeset index]
    //(*dirichletfunc_)[6][2] = &dbc_zero_;

    neumannfunc_ = NULL;


    //exit(0);



  }else if("truchas" == paramList.get<std::string> (TusastestNameString)){

    //std::cout<<"truchas"<<std::endl;

    numeqs_ = 1;
    //numeqs_ = 2;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &truchas::residual_heat_;
    if(2 == numeqs_) (*residualfunc_)[1] = &truchas::residual_phase_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &truchas::prec_heat_;
    if(2 == numeqs_) (*preconfunc_)[1] = &truchas::prec_phase_;

    //dummy for now we will overide this
    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &init_zero_;
    if(2 == numeqs_) (*initfunc_)[1] = &truchas::init_phase_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    if(2 == numeqs_) (*varnames_)[1] = "phi";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = NULL; 
    //dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);

//  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
//               [numeq][nodeset id]
//  [variable index][nodeset index]
    //(*dirichletfunc_)[0][0] = &dbc_zero_;							 
//     (*dirichletfunc_)[0][1] = &dbc_zero_;
    
    //(*dirichletfunc_)[0][2] = &dbc_zero_;						 
//     (*dirichletfunc_)[0][3] = &dbc_zero_;

    //dummy for now we will overide this
    //neumannfunc_ = NULL;
    // numeqs_ number of variables(equations) 
    neumannfunc_ = new std::vector<std::map<int,NBCFUNC>>(numeqs_);
    (*neumannfunc_)[0][0] = &nbc_zero_;							 
    (*neumannfunc_)[0][1] = &nbc_zero_;						 
    (*neumannfunc_)[0][2] = &nbc_zero_;						 
    (*neumannfunc_)[0][3] = &nbc_zero_;					 
    (*neumannfunc_)[0][4] = &nbc_zero_;						 
    (*neumannfunc_)[0][5] = &nbc_zero_;
    if(2 == numeqs_){
      (*neumannfunc_)[1][0] = &nbc_zero_;							 
      (*neumannfunc_)[1][1] = &nbc_zero_;						 
      (*neumannfunc_)[1][2] = &nbc_zero_;						 
      (*neumannfunc_)[1][3] = &nbc_zero_;					 
      (*neumannfunc_)[1][4] = &nbc_zero_;						 
      (*neumannfunc_)[1][5] = &nbc_zero_;
    }

  }else if("takaki" == paramList.get<std::string> (TusastestNameString)){
    //farzadi test

    numeqs_ = 2;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &takaki::init_conc_;
    (*initfunc_)[1] = &takaki::init_phase_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &takaki::residual_conc_;
    (*residualfunc_)[1] = &takaki::residual_phase_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &farzadi::prec_conc_farzadi_;
    (*preconfunc_)[1] = &farzadi::prec_phase_farzadi_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    (*varnames_)[1] = "phi";

    dirichletfunc_ = NULL;

//     dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
    //(*dirichletfunc_)[0][1] = &dbc_mone_;	
    //(*dirichletfunc_)[0][3] = &dbc_zero_;
    //(*dirichletfunc_)[1][1] = &dbc_mone_;
    //(*dirichletfunc_)[1][3] = &dbc_one_;

    neumannfunc_ = NULL;

    //post_proc.push_back(new post_process(comm_,mesh_,(int)0));
    //post_proc[0].postprocfunc_ = &farzadi::postproc_c_;
    //post_proc.push_back(new post_process(comm_,mesh_,(int)1));
    //post_proc[1].postprocfunc_ = &farzadi::postproc_t_;

    //paramfunc_ = farzadi::param_;
						 
    //exit(0);

  }else if("allencahn" == paramList.get<std::string> (TusastestNameString)){
    //farzadi test

    numeqs_ = 1;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &allencahn::init_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &allencahn::residual_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &allencahn::prec_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";

    //dirichletfunc_ = NULL;

    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
    (*dirichletfunc_)[0][0] = &dbc_one_;	
    (*dirichletfunc_)[0][2] = &dbc_zero_;

    neumannfunc_ = NULL;
#ifdef PERIODIC_BC
    periodic_bc_.push_back(new periodic_bc(1,3,numeqs_,mesh_,comm_));
    periodic_bc_[0].add_eqn_index(0);

#else
    exit(0);
#endif

    post_proc.push_back(new post_process(comm_,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &allencahn::postproc_;
    post_proc.push_back(new post_process(comm_,mesh_,(int)1,post_process::SCALAR_OP::NORM2));
    post_proc[1].postprocfunc_ = &allencahn::postproc_error;

    //paramfunc_ = farzadi::param_;
						 
    //exit(0);

  }else if("pfhub2" == paramList.get<std::string> (TusastestNameString)){

    //this is pfhub2 as described on website. WBM model with k_c=0, no kks

    Teuchos::ParameterList *problemList;
    problemList = &paramList.sublist ( "ProblemParams", false );

    int numeta = problemList->get<int>("N");

    numeqs_ = numeta+1;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &pfhub2::residual_c_;
    (*residualfunc_)[1] = &pfhub2::residual_eta_;
    if( 4 == numeta){
      (*residualfunc_)[2] = &pfhub2::residual_eta_;
      (*residualfunc_)[3] = &pfhub2::residual_eta_;
      (*residualfunc_)[4] = &pfhub2::residual_eta_;
    }

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &pfhub2::prec_c_;
    (*preconfunc_)[1] = &pfhub2::prec_eta_;
    if( 4 == numeta){
      (*preconfunc_)[2] = &pfhub2::prec_eta_;
      (*preconfunc_)[3] = &pfhub2::prec_eta_;
      (*preconfunc_)[4] = &pfhub2::prec_eta_;
    }

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &pfhub2::init_c_;
    (*initfunc_)[1] = &pfhub2::init_eta_;
    if( 4 == numeta){
      (*initfunc_)[2] = &pfhub2::init_eta_;
      (*initfunc_)[3] = &pfhub2::init_eta_;
      (*initfunc_)[4] = &pfhub2::init_eta_;
    }

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "c";
    (*varnames_)[1] = "eta0";
    if( 4 == numeta){
      (*varnames_)[2] = "eta1";
      (*varnames_)[3] = "eta2";
      (*varnames_)[4] = "eta3";
    }

    // numeqs_ number of variables(equations) 
    //dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_); 
    dirichletfunc_ = NULL;


    neumannfunc_ = NULL;

    paramfunc_ = pfhub2::param_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &pfhub2::postproc_c_a_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)1));
    post_proc[1].postprocfunc_ = &pfhub2::postproc_c_b_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)2));
    post_proc[2].postprocfunc_ = &pfhub2::postproc_c_;

  }else if("pfhub2wbmkks_g" == paramList.get<std::string> (TusastestNameString)){

    //same WBM model as above with c_alpha and c_beta globally coupled via the global kks equations

    Teuchos::ParameterList *problemList;
    problemList = &paramList.sublist ( "ProblemParams", false );

    int numeta = problemList->get<int>("N");

    numeqs_ = numeta+3;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &pfhub2::residual_c_;
    (*residualfunc_)[1] = &pfhub2::residual_c_alpha_g_;
    (*residualfunc_)[2] = &pfhub2::residual_c_beta_g_;
    (*residualfunc_)[3] = &pfhub2::residual_eta_;
    if( 4 == numeta){
      (*residualfunc_)[4] = &pfhub2::residual_eta_;
      (*residualfunc_)[5] = &pfhub2::residual_eta_;
      (*residualfunc_)[6] = &pfhub2::residual_eta_;
    }

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &pfhub2::prec_c_;
    (*preconfunc_)[1] = &pfhub2::prec_c_;
    (*preconfunc_)[2] = &pfhub2::prec_c_;
    (*preconfunc_)[3] = &pfhub2::prec_eta_;
    if( 4 == numeta){
      (*preconfunc_)[4] = &pfhub2::prec_eta_;
      (*preconfunc_)[5] = &pfhub2::prec_eta_;
      (*preconfunc_)[6] = &pfhub2::prec_eta_;
    }

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &pfhub2::init_c_;
    (*initfunc_)[1] = &pfhub2::init_c_alpha_;
    (*initfunc_)[2] = &pfhub2::init_c_beta_;
    (*initfunc_)[3] = &pfhub2::init_eta_;
    if( 4 == numeta){
      (*initfunc_)[4] = &pfhub2::init_eta_;
      (*initfunc_)[5] = &pfhub2::init_eta_;
      (*initfunc_)[6] = &pfhub2::init_eta_;
    }

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "c";
    (*varnames_)[1] = "c_a";
    (*varnames_)[2] = "c_b";
    (*varnames_)[3] = "eta0";
    if( 4 == numeta){
      (*varnames_)[4] = "eta1";
      (*varnames_)[5] = "eta2";
      (*varnames_)[6] = "eta3";
    }

    // numeqs_ number of variables(equations) 
    //dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_); 
    dirichletfunc_ = NULL;

    neumannfunc_ = NULL;

    paramfunc_ = pfhub2::param_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &pfhub2::postproc_c_a_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)1));
    post_proc[1].postprocfunc_ = &pfhub2::postproc_c_b_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)2));
    post_proc[2].postprocfunc_ = &pfhub2::postproc_c_;

}else if("pfhub2wbmkks_l" == paramList.get<std::string> (TusastestNameString)){

    //same WBM model as above with c_alpha and c_beta globally coupled via local kks equations

    Teuchos::ParameterList *problemList;
    problemList = &paramList.sublist ( "ProblemParams", false );

    int numeta = problemList->get<int>("N");

    numeqs_ = numeta+3;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &pfhub2::residual_c_;
    (*residualfunc_)[1] = &pfhub2::residual_c_alpha_l_;
    (*residualfunc_)[2] = &pfhub2::residual_c_beta_l_;
    (*residualfunc_)[3] = &pfhub2::residual_eta_;
    if( 4 == numeta){
      (*residualfunc_)[4] = &pfhub2::residual_eta_;
      (*residualfunc_)[5] = &pfhub2::residual_eta_;
      (*residualfunc_)[6] = &pfhub2::residual_eta_;
    }

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &pfhub2::prec_c_;
    (*preconfunc_)[1] = &pfhub2::prec_c_alpha_;
    (*preconfunc_)[2] = &pfhub2::prec_c_beta_;
    (*preconfunc_)[3] = &pfhub2::prec_eta_;
    if( 4 == numeta){
      (*preconfunc_)[4] = &pfhub2::prec_eta_;
      (*preconfunc_)[5] = &pfhub2::prec_eta_;
      (*preconfunc_)[6] = &pfhub2::prec_eta_;
    }

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &pfhub2::init_c_;
    (*initfunc_)[1] = &pfhub2::init_c_alpha_;
    (*initfunc_)[2] = &pfhub2::init_c_beta_;
    (*initfunc_)[3] = &pfhub2::init_eta_;
    if( 4 == numeta){
      (*initfunc_)[4] = &pfhub2::init_eta_;
      (*initfunc_)[5] = &pfhub2::init_eta_;
      (*initfunc_)[6] = &pfhub2::init_eta_;
    }

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "c";
    (*varnames_)[1] = "ca";
    (*varnames_)[2] = "cb";
    (*varnames_)[3] = "eta0";
    if( 4 == numeta){
      (*varnames_)[4] = "eta1";
      (*varnames_)[5] = "eta2";
      (*varnames_)[6] = "eta3";
    }

    // numeqs_ number of variables(equations) 
    //dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_); 
    dirichletfunc_ = NULL;


    neumannfunc_ = NULL;

    paramfunc_ = pfhub2::param_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &pfhub2::postproc_c_a_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)1));
    post_proc[1].postprocfunc_ = &pfhub2::postproc_c_b_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)2));
    post_proc[2].postprocfunc_ = &pfhub2::postproc_c_;

  }else if("pfhub2kks" == paramList.get<std::string> (TusastestNameString)){

    //this is pfhub2 brute force kks with k_c=0

    Teuchos::ParameterList *problemList;
    problemList = &paramList.sublist ( "ProblemParams", false );

    int numeta = problemList->get<int>("N");

    numeqs_ = numeta+1;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &pfhub2::residual_c_kks_;
    (*residualfunc_)[1] = &pfhub2::residual_eta_kks_;
    if( 4 == numeta){
      (*residualfunc_)[2] = &pfhub2::residual_eta_;
      (*residualfunc_)[3] = &pfhub2::residual_eta_;
      (*residualfunc_)[4] = &pfhub2::residual_eta_;
    }

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &pfhub2::prec_c_;
    (*preconfunc_)[1] = &pfhub2::prec_eta_;
    if( 4 == numeta){
      (*preconfunc_)[2] = &pfhub2::prec_eta_;
      (*preconfunc_)[3] = &pfhub2::prec_eta_;
      (*preconfunc_)[4] = &pfhub2::prec_eta_;
    }

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &pfhub2::init_c_;
    (*initfunc_)[1] = &pfhub2::init_eta_;
    if( 4 == numeta){
      (*initfunc_)[2] = &pfhub2::init_eta_;
      (*initfunc_)[3] = &pfhub2::init_eta_;
      (*initfunc_)[4] = &pfhub2::init_eta_;
    }

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "c";
    (*varnames_)[1] = "eta0";
    if( 4 == numeta){
      (*varnames_)[2] = "eta1";
      (*varnames_)[3] = "eta2";
      (*varnames_)[4] = "eta3";
    }

    // numeqs_ number of variables(equations) 
    //dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_); 
    dirichletfunc_ = NULL;


    neumannfunc_ = NULL;

    paramfunc_ = pfhub2::param_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &pfhub2::postproc_c_a_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)1));
    post_proc[1].postprocfunc_ = &pfhub2::postproc_c_b_;

    post_proc.push_back(new post_process(comm_,mesh_,(int)2));
    post_proc[2].postprocfunc_ = &pfhub2::postproc_c_;

  }else if("pfhub3" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 2;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::pfhub3::init_heat_pfhub3_;
    (*initfunc_)[1] = &tpetra::pfhub3::init_phase_pfhub3_;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = tpetra::pfhub3::residual_heat_pfhub3_n_;
    (*residualfunc_)[1] = tpetra::pfhub3::residual_phase_pfhub3_n_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::pfhub3::prec_heat_pfhub3_n_;
    (*preconfunc_)[1] = tpetra::pfhub3::prec_phase_pfhub3_n_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    (*varnames_)[1] = "phi";

    dirichletfunc_ = NULL;

    paramfunc_ = tpetra::pfhub3::param_;

    neumannfunc_ = NULL;


  }else if("heat2" == paramList.get<std::string> (TusastestNameString)){
    
    numeqs_ = 2;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);

    (*residualfunc_)[0] = &heat::residual_heat_test_;
    (*residualfunc_)[1] = &heat::residual_heat_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &heat::prec_heat_test_;
    (*preconfunc_)[1] = &heat::prec_heat_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);

    (*initfunc_)[0] = &heat::init_heat_test_;
    (*initfunc_)[1] = &heat::init_heat_test_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    (*varnames_)[1] = "phi";

    //dirichletfunc_ = NULL;
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
    
    //  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
    //               [numeq][nodeset id]
    //  [variable index][nodeset index]
    (*dirichletfunc_)[0][0] = &dbc_zero_;							 
    (*dirichletfunc_)[0][1] = &dbc_zero_;						 
    (*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[0][3] = &dbc_zero_;
    (*dirichletfunc_)[1][0] = &dbc_zero_;							 
    (*dirichletfunc_)[1][1] = &dbc_zero_;						 
    (*dirichletfunc_)[1][2] = &dbc_zero_;						 
    (*dirichletfunc_)[1][3] = &dbc_zero_;

    neumannfunc_ = NULL;
    //std::cout<<"heat2 ended"<<std::endl;

  }else {

    D_ = 4.;
    T_m_ = 1.55;
    T_inf_ = 1.;
    alpha_ = 191.82;
    eps_ = .05;
    eps_0_ = 1.;
    M_= 4.;
    theta_0_ =0.;
    R_0_ =.3;
    
    //function pointers
    hp1_ = &cummins::hp1_cummins_;
    hpp1_ = &cummins::hpp1_cummins_;
    w_ = &cummins::w_cummins_;
    m_ = &cummins::m_cummins_;
    rand_phi_ = &rand_phi_zero_;
    gp1_ = &cummins::gp1_cummins_;
    gpp1_ = &cummins::gpp1_cummins_;
    hp2_ = &cummins::hp2_cummins_;

    gs2_ = &cummins::gs2_cummins_;
    dgs2_2dtheta_ = &cummins::dgs2_2dtheta_cummins_;
  }

  int mypid = comm_->MyPID();
  if(0 == mypid) {
    std::cout<<"set_test_case started"<<std::endl;
    if(0 <  numeqs_){
      std::cout<<"  numeqs_ = "<<numeqs_<<std::endl;
    }else{
      std::cout<<"  numeqs < 0. Exiting."<<std::endl;
      exit(0);
    }

    if(NULL != residualfunc_){
      std::cout<<"  residualfunc_ with size "<<residualfunc_->size()<<" found."<<std::endl;
    }else{
      std::cout<<"  residualfunc_ not found. Exiting."<<std::endl;
      exit(0);
    }

    if(NULL != preconfunc_){
      std::cout<<"  preconfunc_ with size "<<preconfunc_->size()<<" found."<<std::endl;
    }

    if(NULL != initfunc_){
      std::cout<<"  initfunc_ with size "<<initfunc_->size()<<" found."<<std::endl;
    }else{
      std::cout<<"  initfunc_ not found. Exiting."<<std::endl;
      exit(0);
    }

    if(NULL != varnames_){
      std::cout<<"  varnames_ with size "<<varnames_->size()<<" found."<<std::endl;
    }else{
      std::cout<<"  varnames_ not found. Exiting."<<std::endl;
      exit(0);
    }

    if(NULL != dirichletfunc_){
      std::cout<<"  dirichletfunc_ with size "<<dirichletfunc_->size()<<" found."<<std::endl;

      std::map<int,DBCFUNC>::iterator it;
      
      for( int k = 0; k < numeqs_; k++ ){
	for(it = (*dirichletfunc_)[k].begin();it != (*dirichletfunc_)[k].end(); ++it){
	  int ns_id = it->first;
	  std::cout<<"    Equation: "<<k<<" nodeset: "<<ns_id<<std::endl;
	}
      }
    }

    if(NULL != neumannfunc_){
      std::cout<<"  neumannfunc_ with size "<<neumannfunc_->size()<<" found."<<std::endl;

      std::map<int,NBCFUNC>::iterator it;
      
      for( int k = 0; k < numeqs_; k++ ){
	for(it = (*neumannfunc_)[k].begin();it != (*neumannfunc_)[k].end(); ++it){
	  int ns_id = it->first;
	  std::cout<<"    Equation: "<<k<<" sideset: "<<ns_id<<std::endl;
	}
      }
    }

    if(post_proc.size() > 0 ){
      std::cout<<"  post_proc with size "<<post_proc.size()<<" found."<<std::endl;
    }
#ifdef PERIODIC_BC
#else
    if(NULL != periodicbc_){
      if(1 != comm_->NumProc()){
	std::cout<<"Periodic bc only implemented in serial at this time."<<std::endl;
	exit(0);
      }

      mesh_->create_sorted_nodesetlists();

      std::cout<<"  periodicbc_ with size "<<periodicbc_->size()<<" found."<<std::endl;
      
      std::vector<std::pair<int,int>>::iterator it;
      
      for( int k = 0; k < numeqs_; k++ ){
 	for(it = (*periodicbc_)[k].begin();it != (*periodicbc_)[k].end(); ++it){
 	  int ns_id1 = it->first;
 	  int ns_id2 = it->second;
 	  //int ns_id1 = (*periodicbc_)[k][0].first;
 	  //int ns_id2 = (*periodicbc_)[k][0].second;
	  int ns_size1 = mesh_->get_sorted_node_set(ns_id1).size();
	  int ns_size2 = mesh_->get_sorted_node_set(ns_id2).size();
 	  std::cout<<"    Equation: "<<k<<" nodeset 1: "<<ns_id1<<" size: "<<ns_size1<<std::endl
		   <<"                nodeset 2: "<<ns_id2<<" size: "<<ns_size2<<std::endl;
	  if(ns_size1 != ns_size2 ){
	    std::cout<<"Incompatible nodeset sizes found."<<std::endl;
	    exit(0);
	  }//if
 	}//it
      }//k

    }//if
#endif
    std::cout<<"set_test_case ended"<<std::endl;
  }
    
  //set the params in the test case now...
  Teuchos::ParameterList *problemList;
  problemList = &paramList.sublist ( "ProblemParams", false );
  
  if ( NULL != paramfunc_ ){
    
    paramfunc_(problemList);
  }
  

}
template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::postprocess()
{
  if(0 == post_proc.size() ) return;

  int numee = Error_est.size();
  //ordering is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz

  const int dim = 3;

  std::vector<double> uu(numeqs_);
  std::vector<double> uuold(numeqs_);
  std::vector<double> uuoldold(numeqs_);
  std::vector<double> ug(dim*numee);

  //#pragma omp parallel for
  for (int nn=0; nn < num_my_nodes_; nn++) {
    for( int k = 0; k < numeqs_; k++ ){
      uu[k] = (*u_old_)[numeqs_*nn+k];
      uuold[k] = (*u_old_old_)[numeqs_*nn+k];
      uuoldold[k] = (*u_old_old_old_)[numeqs_*nn+k];
    }

    for( int k = 0; k < numee; k++ ){
      ug[k*dim] = (*(Error_est[k].gradx_))[nn];
      ug[k*dim+1] = (*(Error_est[k].grady_))[nn];
      ug[k*dim+2] = (*(Error_est[k].gradz_))[nn];
    }

    boost::ptr_vector<post_process>::iterator itp;
    for(itp = post_proc.begin();itp != post_proc.end();++itp){
      itp->process(nn,&uu[0],&uuold[0],&uuoldold[0],&ug[0],time_,dt_,dtold_);
      //std::cout<<nn<<" "<<mesh_->get_local_id((x_owned_map_->GID(nn))/numeqs_)<<" "<<xyz[0]<<std::endl;
    }
    for(itp = temporal_est.begin();itp != temporal_est.end();++itp){
      itp->process(nn,&uu[0],&uuold[0],&uuoldold[0],&ug[0],time_,dt_,dtold_);
      //std::cout<<nn<<" "<<mesh_->get_local_id((x_owned_map_->GID(nn))/numeqs_)<<" "<<xyz[0]<<std::endl;
    }

  }//nn

}

template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::postprocess(boost::ptr_vector<post_process> pp)
{
  if(0 == pp.size() ) return;

  //there is currently an issue where the post process variables or files are not written correctly
  //at t=0.  The values are 0 at t=0, and should be based on the initial condition.

  //also the last timestep is being written to data files twice


  int numee = Error_est.size();
  //ordering is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz

  const int dim = 3;

  std::vector<double> uu(numeqs_);
  std::vector<double> uuold(numeqs_);
  std::vector<double> uuoldold(numeqs_);
  std::vector<double> ug(dim*numee);

  //#pragma omp parallel for
  for (int nn=0; nn < num_my_nodes_; nn++) {
    for( int k = 0; k < numeqs_; k++ ){
      uu[k] = (*u_old_)[numeqs_*nn+k];
      uuold[k] = (*u_old_old_)[numeqs_*nn+k];
      uuoldold[k] = (*u_old_old_old_)[numeqs_*nn+k];
    }

    for( int k = 0; k < numee; k++ ){
      ug[k*dim] = (*(Error_est[k].gradx_))[nn];
      ug[k*dim+1] = (*(Error_est[k].grady_))[nn];
      ug[k*dim+2] = (*(Error_est[k].gradz_))[nn];
    }

    boost::ptr_vector<post_process>::iterator itp;
    for(itp = pp.begin();itp != pp.end();++itp){
      itp->process(nn,&uu[0],&uuold[0],&uuoldold[0],&ug[0],time_,dt_,dtold_);
      //std::cout<<nn<<" "<<mesh_->get_local_id((x_owned_map_->GID(nn))/numeqs_)<<" "<<xyz[0]<<std::endl;
    }

  }//nn

}

template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::temporalpostprocess(boost::ptr_vector<post_process> pp)
{
  if(0 == pp.size() ) return;

  int numee = Error_est.size();
  //ordering is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz

  const int dim = 3;

  std::vector<double> uu(numeqs_);
  std::vector<double> uuold(numeqs_);
  std::vector<double> uuoldold(numeqs_);
  std::vector<double> ug(numeqs_);

  for (int nn=0; nn < num_my_nodes_; nn++) {
    for( int k = 0; k < numeqs_; k++ ){
      uu[k] = (*u_new_)[numeqs_*nn+k];
      uuold[k] = (*u_old_)[numeqs_*nn+k];
      //uuoldold[k] = (*u_old_old_)[numeqs_*nn+k];
      uuoldold[k] = (*pred_temp_)[numeqs_*nn+k];
      ug[k] = (*pred_temp_)[numeqs_*nn+k];
    }

    boost::ptr_vector<post_process>::iterator itp;
    for(itp = pp.begin();itp != pp.end();++itp){
      itp->process(nn,&uu[0],&uuold[0],&uuoldold[0],&ug[0],time_,dt_,dtold_);
      //std::cout<<nn<<" "<<mesh_->get_local_id((x_owned_map_->GID(nn))/numeqs_)<<" "<<xyz[0]<<std::endl;
    }

  }//nn

}

//cn seems this should live in the basis class.....
template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::set_basis( boost::ptr_vector<Basis> &basis, const std::string elem_type) const
{
      basis.resize(0);

      int LTP_quadrature_order = paramList.get<int> (TusasltpquadordNameString);
      int QTP_quadrature_order = paramList.get<int> (TusasqtpquadordNameString);
      int LTri_quadrature_order = paramList.get<int> (TusasltriquadordNameString);
      int QTri_quadrature_order = paramList.get<int> (TusasqtriquadordNameString);

      if( (0==elem_type.compare("QUAD4")) || (0==elem_type.compare("QUAD")) || (0==elem_type.compare("quad4")) || (0==elem_type.compare("quad")) ){ // linear quad
	for ( int nb = 0; nb < numeqs_; nb++ )
	  basis.push_back(new BasisLQuad(LTP_quadrature_order));
      }
      else if( (0==elem_type.compare("TRI3")) || (0==elem_type.compare("TRI")) || (0==elem_type.compare("tri3"))  || (0==elem_type.compare("tri"))){ // linear triangle
	for ( int nb = 0; nb < numeqs_; nb++ )
	  basis.push_back(new BasisLTri(LTri_quadrature_order));
      }
      else if( (0==elem_type.compare("HEX8")) || (0==elem_type.compare("HEX")) || (0==elem_type.compare("hex8")) || (0==elem_type.compare("hex"))  ){ // linear hex
	for ( int nb = 0; nb < numeqs_; nb++ )
	  basis.push_back(new BasisLHex(LTP_quadrature_order));
      } 
      else if( (0==elem_type.compare("TETRA4")) || (0==elem_type.compare("TETRA")) || (0==elem_type.compare("tetra4")) || (0==elem_type.compare("tetra")) ){ // linear tet
	for ( int nb = 0; nb < numeqs_; nb++ )
	  basis.push_back(new BasisLTet());
      } 
      else if( (0==elem_type.compare("QUAD9")) || (0==elem_type.compare("quad9")) ){ // quadratic quad
	for ( int nb = 0; nb < numeqs_; nb++ )
	  basis.push_back(new BasisQQuad(QTP_quadrature_order));
      }
      else if( (0==elem_type.compare("TRI6")) || (0==elem_type.compare("tri6")) ){ // quadratic triangle
	for ( int nb = 0; nb < numeqs_; nb++ )
	  basis.push_back(new BasisQTri(QTri_quadrature_order));
      } 
      else if( (0==elem_type.compare("HEX27")) || (0==elem_type.compare("hex27")) ){ // quadratic hex
	for ( int nb = 0; nb < numeqs_; nb++ )
	  {//basis.push_back(new BasisQHex(QTP_quadrature_order));
	  }
	std::cout<<"Unsupported element type : "<<elem_type<<std::endl<<std::endl;
	exit(0);
      }
      else if( (0==elem_type.compare("TETRA10")) || (0==elem_type.compare("tetra10")) ){ // quadratic tet
	for ( int nb = 0; nb < numeqs_; nb++ )
	  {//basis.push_back(new BasisQTet());
	  }
	std::cout<<"Unsupported element type : "<<elem_type<<std::endl<<std::endl;
	exit(0);
      }
      else {
	std::cout<<"Unsupported element type : "<<elem_type<<std::endl<<std::endl;
	exit(0);
      }
//       if( basis.size() != numeqs_ ){
// 	std::cout<<" basis.size() != numeqs_ "<<std::endl;
// 	exit(0);
//       }

}  

template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::dump_exaconstit(){

  std::vector<nemesis_lint_t> node_num_map(mesh_->get_node_num_map());
  Teuchos::RCP<const Epetra_Map> overlap_map_ = Teuchos::rcp(new Epetra_Map(-1,
					     node_num_map.size(),
					     &node_num_map[0],
					     0,
					     *comm_));

  Teuchos::RCP<const Epetra_Map> node_map_;

  if( 1 == comm_->NumProc() ){
    node_map_ = overlap_map_;
  }else{
    node_map_ = Teuchos::rcp(new Epetra_Map(Epetra_Util::Create_OneToOne_Map(*overlap_map_)));
  }

  Teuchos::RCP<const Epetra_Map> root_node_map_ = Teuchos::rcp( new Epetra_Map( Epetra_Util::Create_Root_Map( *node_map_)));

  //node_map_->Print(std::cout);
  //root_node_map_->Print(std::cout);
  Teuchos::RCP<Epetra_Vector> x_vec = Teuchos::rcp(new Epetra_Vector(*node_map_));
  Teuchos::RCP<Epetra_Vector> y_vec = Teuchos::rcp(new Epetra_Vector(*node_map_));
  Teuchos::RCP<Epetra_Vector> z_vec = Teuchos::rcp(new Epetra_Vector(*node_map_));

  //the mesh_->get_x() mthods are indexed by overlap map

  for (int nn=0; nn < num_my_nodes_; nn++) {
    int gid = node_map_->GID(nn);
    int lid = overlap_map_->LID(gid); 
    double val = mesh_->get_x(lid);
    x_vec->ReplaceGlobalValues ((int) 1, &val, &gid);
    val = mesh_->get_y(lid);
    y_vec->ReplaceGlobalValues ((int) 1, &val, &gid);
    val = mesh_->get_z(lid);
    z_vec->ReplaceGlobalValues ((int) 1, &val, &gid);
  }
  //x_vec->Print(std::cout);

  Teuchos::RCP<const Epetra_Import> importer1_ = Teuchos::rcp(new Epetra_Import(*root_node_map_, *node_map_));

  Teuchos::RCP<Epetra_Vector> root_x_vec = Teuchos::rcp(new Epetra_Vector(*root_node_map_));
  Teuchos::RCP<Epetra_Vector> root_y_vec = Teuchos::rcp(new Epetra_Vector(*root_node_map_));
  Teuchos::RCP<Epetra_Vector> root_z_vec = Teuchos::rcp(new Epetra_Vector(*root_node_map_));

  root_x_vec->Import(*x_vec, *importer1_, Insert);
  root_y_vec->Import(*y_vec, *importer1_, Insert);
  root_z_vec->Import(*z_vec, *importer1_, Insert);
  //root_x_vec->Print(std::cout);


  Teuchos::RCP<const Epetra_Map> root_f_owned_map_ = Teuchos::rcp( new Epetra_Map( Epetra_Util::Create_Root_Map( *f_owned_map_)));
  Teuchos::RCP<const Epetra_Import> importer2_ = Teuchos::rcp(new Epetra_Import(*root_f_owned_map_, *f_owned_map_));
  Teuchos::RCP<Epetra_Vector> root_u_vec = Teuchos::rcp(new Epetra_Vector(*root_f_owned_map_));
  root_u_vec->Import(*u_old_, *importer2_, Insert);

  std::cout.precision(std::numeric_limits<double>::digits10 + 2);
  if( 0 == comm_->MyPID() ){

    std::ofstream coordfile;
    coordfile.open("ec_coord.txt");
    coordfile.precision(std::numeric_limits<double>::digits10 + 2);

    std::ofstream varfile;
    varfile.open("ec_var.txt");
    varfile.precision(std::numeric_limits<double>::digits10 + 2);

    for ( int nn = 0; nn < root_node_map_->NumGlobalElements(); nn++ ){
      int gid = root_node_map_->GID(nn);
      //std::cout<<gid<<std::scientific<<" "<<(*root_x_vec)[nn]<<" "<<(*root_y_vec)[nn]<<" "<<(*root_z_vec)[nn]<<"\n";
      coordfile<<gid<<std::scientific<<" "<<(*root_x_vec)[nn]<<" "<<(*root_y_vec)[nn]<<" "<<(*root_z_vec)[nn]<<"\n";
      int ogid = gid*numeqs_;

      //std::cout<<gid;
      varfile<<gid;
      for(int k = 0; k < numeqs_; k++ ){
	double val = (*root_u_vec)[numeqs_*nn+k];
	//std::cout<<std::scientific<<" "<<val;
	varfile<<std::scientific<<" "<<val*val;
      }
      //std::cout<<"\n";
      varfile<<"\n";
    }
   
    std::ofstream orientfile;
    orientfile.open("ec_orient.txt");
    orientfile.precision(std::numeric_limits<double>::digits10 + 2);

    for(int k = 0; k < numeqs_; k++ ){
      double pi = 3.141592653589793;
      double phi = 2.*pi*(rand()%101)/100.;// [0, 2 pi]
      double theta = pi*(rand()%101)/100.;// [0, pi]
      double omega = 2.*pi*(rand()%101)/100.;// [0, 2 pi]
      //std::cout<<k<<std::scientific<<" "<<phi<<" "<<theta<<" "<<omega<<"\n";
      orientfile<<k<<std::scientific<<" "<<phi<<" "<<theta<<" "<<omega<<"\n";
    }
  }
}

template<class Scalar>
double ModelEvaluatorNEMESIS<Scalar>::estimatetimestep()
{
  double dtpred = 0.;
  //std::cout<<std::setprecision(std::numeric_limits<double>::digits10 + 1);
  Teuchos::ParameterList *atsList;
  atsList = &paramList.sublist (TusasatslistNameString, false );

  const double atol = atsList->get<double>(TusasatsatolNameString);
  const double rtol = atsList->get<double>(TusasatsrtolNameString);
  const double sf = atsList->get<double>(TusasatssfNameString);
  const double rmax = atsList->get<double>(TusasatsrmaxNameString);
  const double rmin = atsList->get<double>(TusasatsrminNameString);
  const double eps = atsList->get<double>(TusasatsepsNameString);
  const double dtmax = atsList->get<double>(TusasatsmaxdtNameString);

  temporalpostprocess(temporal_est); 
  boost::ptr_vector<post_process>::iterator itp;
  for(itp = temporal_est.begin();itp != temporal_est.end();++itp){
    itp->scalar_reduction();
  }

  //norm for rtol
  temporalpostprocess(temporal_norm); 
  for(itp = temporal_norm.begin();itp != temporal_norm.end();++itp){
    itp->scalar_reduction();
  }
  
  std::vector<double> maxdt(numeqs_);
  std::vector<double> mindt(numeqs_);
  std::vector<double> newdt(numeqs_);
  std::vector<double> error(numeqs_);
  std::vector<double> norm(numeqs_,0.);
  
  if( 0 == comm_->MyPID()){
    std::cout<<std::endl<<"     Estimating timestep size:"<<std::endl;
    std::cout<<"     using "<<atsList->get<std::string> (TusasatstypeNameString)
	     <<" and theta = "<<t_theta_<<std::endl;
    std::cout<<"     with atol = "<<atol
	     <<"; rtol = "<<rtol
	     <<"; sf = "<<sf<<"; rmax = "<<rmax<<"; rmin = "<<rmin<<"; current dt = "<<dt_<<std::endl;
  }

  double err_coef = 1.;
  if(atsList->get<std::string> (TusasatstypeNameString) == "second derivative")
    err_coef = 1.;
  if(atsList->get<std::string> (TusasatstypeNameString) == "predictor corrector")
    if(t_theta_ > .51){
      err_coef = .5;
    }
    else{
      err_coef = 1./3./(1.+dtold_/dt_);
    }

  for( int k = 0; k < numeqs_; k++ ){
    error[k] = err_coef*(temporal_est[k].get_scalar_val());
    norm[k] = temporal_norm[k].get_scalar_val();
    const double abserr = std::max(error[k],eps);
    const double tol = atol + rtol*norm[k];
    double rr;
    if(t_theta_ > .51){
      rr = std::sqrt(tol/abserr);
    }
    else{
      rr = std::cbrt(tol/abserr);
      //rr = std::pow(tol/abserr, 1./3.);
    }
    const double h1 = sf*dt_*rr;
    maxdt[k] = std::max(h1,dt_*rmin);
    mindt[k] = std::min(h1,dt_*rmax);
    if( 0 == comm_->MyPID()){
      std::cout<<std::endl<<"     Variable: "<<(*varnames_)[k]<<std::endl;
      //std::cout<<"                              tol = "<<tol<<std::endl;
      std::cout<<"                            error = "<<error[k]<<std::endl;
      std::cout<<"                           max dt = "<<dtmax<<std::endl;
      std::cout<<"                   max(error,eps) = "<<abserr<<std::endl;
      std::cout<<"                    (tol/err)^1/p = "<<rr<<std::endl;
      std::cout<<"          h = sf*dt*(tol/err)^1/p = "<<h1<<std::endl;
      std::cout<<"                   max(h,dt*rmin) = "<<maxdt[k]<<std::endl;
      std::cout<<"                   min(h,dt*rmax) = "<<mindt[k]<<std::endl<<std::endl;
    }
    if( h1 < dt_ ){
      newdt[k] = maxdt[k];
    }else{
      newdt[k] = mindt[k];
    }
  }//k

  dtpred = *min_element(newdt.begin(), newdt.end());

  if( 0 == comm_->MyPID()){
    std::cout<<std::endl<<"     Estimated timestep size : "<<dtpred<<std::endl;	
  }
  dtpred = std::min(dtpred,dtmax);
  if( 0 == comm_->MyPID()){
    std::cout<<std::endl<<"           min(dtpred,dtmax) : "<<dtpred<<std::endl<<std::endl;	
  }
  
  return dtpred;
}

template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::predictor()
{
  //right now theta2=0 corresponds to FE, BE and TR
  //theta2=1 corresponds to AB

  if( 0 == comm_->MyPID()){
    std::cout<<std::endl<<std::endl<<std::endl<<"     Predictor step started"<<std::endl;	
  }

  const double t_theta_temp = t_theta_;

  t_theta2_ = 0.;
  if(t_theta_ > 0.45 && t_theta_ <.55) t_theta2_ = 1.;//ab predictor tr corrector
  //fe predictor    be corrector
  t_theta_ = 0.;

  Teuchos::RCP< VectorBase< double > > guess = Thyra::create_Vector(u_old_,x_space_);
  NOX::Thyra::Vector thyraguess(*guess);//by sending the dereferenced pointer, we instigate a copy rather than a view

  predictor_->reset(thyraguess);

  NOX::StatusTest::StatusType solvStatus = predictor_->solve();
  
  const Thyra::VectorBase<double> * sol = 
    &(dynamic_cast<const NOX::Thyra::Vector&>(
					      predictor_->getSolutionGroup().getX()
					      ).getThyraVector()
      );    
  Thyra::ConstDetachedSpmdVectorView<double> x_vec(sol->col(0));

  for (int nn=0; nn < num_my_nodes_; nn++) {//cn figure out a better way here...
    for( int k = 0; k < numeqs_; k++ ){
      (*pred_temp_)[numeqs_*nn+k]=x_vec[numeqs_*nn+k];
    }
  } 

  if( 0 == comm_->MyPID()){
    std::cout<<std::endl<<"     Predictor step ended"<<std::endl<<std::endl<<std::endl;
  }

  t_theta_ = t_theta_temp;
  t_theta2_ = 0.;
 }

template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::initialsolve()
 {      
   //right now, for TR it doesn't really matter in turns of performance if we set theta to 1
   //here or leave it at .5
   const double t_theta_temp = t_theta_;
   t_theta_ = 1.;
   
   t_theta2_ = 0.;
   
   if( 0 == comm_->MyPID()) 
     std::cout<<std::endl<<"Performing initial NOX solve"<<std::endl<<std::endl;
   
   Teuchos::RCP< VectorBase< double > > guess = Thyra::create_Vector(u_old_,x_space_);
   NOX::Thyra::Vector thyraguess(*guess);//by sending the dereferenced pointer, we instigate a copy rather than a view
   solver_->reset(thyraguess);
   
   Teuchos::TimeMonitor NSolveTimer(*ts_time_nsolve);
   NOX::StatusTest::StatusType solvStatus = solver_->solve();
   if( !(NOX::StatusTest::Converged == solvStatus)) {
     std::cout<<" NOX solver failed to converge. Status = "<<solvStatus<<std::endl<<std::endl;
     exit(0);
   }
   
   if( 0 == comm_->MyPID()) 
     std::cout<<std::endl<<"Initial NOX solve completed"<<std::endl<<std::endl;
   const Thyra::VectorBase<double> * sol = 
     &(dynamic_cast<const NOX::Thyra::Vector&>(
					       solver_->getSolutionGroup().getX()
					       ).getThyraVector());
   
   Thyra::ConstDetachedSpmdVectorView<double> x_vec(sol->col(0));
   
   //now,
   //dudt(t=0) = (x_vec-u_old_)/dt_
   //
   //so, also
   //dudt(t=0) = (u_old_ - u_old_old_)/dt_
   //u_old_old_ = u_old_ - dt_*dudt(t=0)
   //           = 2*u_old_ - x_vec
   for (int nn=0; nn < num_my_nodes_; nn++) {//cn figure out a better way here...
     for( int k = 0; k < numeqs_; k++ ){
       
       (*u_old_old_)[numeqs_*nn+k] = 2.*(*u_old_)[numeqs_*nn+k] - x_vec[numeqs_*nn+k];
       
       //  	  std::cout<<(*u_old_old_)[numeqs_*nn+k]<<"  "<<(*u_old_)[numeqs_*nn+k]<<"  "<<x_vec[numeqs_*nn+k]<<"  "
       // 		   <<-dt_*(x_vec[numeqs_*nn+k]-(*u_old_)[numeqs_*nn+k])+(*u_old_)[numeqs_*nn+k]<<std::endl;
     }
   }
   
   t_theta_ = t_theta_temp;
 }

template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::setadaptivetimestep()
  {
      //cn this is not going to work with multiple k
      //what do we do with temporal_est[0].pos....
      //is index_ correct here??
      Teuchos::ParameterList *atsList;
      atsList = &paramList.sublist (TusasatslistNameString, false );
      for( int k = 0; k < numeqs_; k++ ){
	temporal_est.push_back(new post_process(comm_,
						mesh_,
						k, 
						post_process::NORMRMS, 
						k, 
						"temperror",
						16));
	//be with an error estimate based on second derivative
	if(atsList->get<std::string> (TusasatstypeNameString) == "second derivative")
	   temporal_est[k].postprocfunc_ = &timeadapt::d2udt2_;
	//be with error estimate based on fe predictor: fe-be
	if(atsList->get<std::string> (TusasatstypeNameString) == "predictor corrector")
	  temporal_est[k].postprocfunc_ = &timeadapt::predictor_fe_;

	//we will have tr with adams-bashforth predictor: ab-tr
	//would require a small first step to get ab going
	//or get u_n-1 via initial solve above with maybe a better update
	//see gresho for a wierd ab way for this--can use regular ab

	//and possibly tr with an explicit midpoint rule (huen) predictor: huen-tr
	//(we could use initial solve above 
	//with better update to get
	//(up_n+1 - u_n-1)/(2 dt) = f(t_n)
	//with error = - (u_n+1 - up_n+1)/5
	//see https://www.cs.princeton.edu/courses/archive/fall11/cos323/notes/cos323_f11_lecture17_ode2.pdf
	//we would utilize t_theta2_ similarly

	temporal_norm.push_back(new post_process(comm_,
						 mesh_,
						 k, 
						 post_process::NORMRMS, 
						 k, 
						 "tempnorm",
						 16));
 	temporal_norm[k].postprocfunc_ = &timeadapt::normu_;

      }
  }
template<class Scalar>
void ModelEvaluatorNEMESIS<Scalar>::init_predictor()
  { 
    //currently there is some issue with the predictor_ solver not being created correctly
    //right now we dont call this code
    //this is handled in init_nox()
#if 0
    double relrestol = 1.0e-6;
    relrestol = paramList.get<double> (TusasnoxrelresNameString);
    
    Teuchos::ParameterList printParams;    
    Teuchos::RCP<Teuchos::ParameterList> jfnkParams = Teuchos::rcp(new Teuchos::ParameterList(paramList.sublist(TusasjfnkNameString)));
    

    Teuchos::RCP< ::Thyra::VectorBase<double> >
      initial_guess = this->getNominalValues().get_x()->clone_v();

    Teuchos::RCP< ::Thyra::ModelEvaluator<double> > Model = Teuchos::rcpFromRef(*this);
    // Wrap the model evaluator in a JFNK Model Evaluator
    Teuchos::RCP< ::Thyra::ModelEvaluator<double> > thyraModel =
      Teuchos::rcp(new NOX::MatrixFreeModelEvaluatorDecorator<double>(Model));
    
    ::Stratimikos::DefaultLinearSolverBuilder builder;

    Teuchos::RCP<Teuchos::ParameterList> lsparams =
      Teuchos::rcp(new Teuchos::ParameterList(paramList.sublist(TusaslsNameString)));
    
    builder.setParameterList(lsparams);

    Teuchos::RCP< ::Thyra::LinearOpWithSolveFactoryBase<double> >
      lowsFactory = builder.createLinearSolveStrategy("");
    
    // Setup output stream and the verbosity level
    Teuchos::RCP<Teuchos::FancyOStream>
      out = Teuchos::VerboseObjectBase::getDefaultOStream();
    lowsFactory->setOStream(out);
    lowsFactory->setVerbLevel(Teuchos::VERB_EXTREME);
    
    //this->set_W_factory(lowsFactory);
    
    Teuchos::RCP<NOX::Thyra::MatrixFreeJacobianOperator<double> > jfnkOp1 =
      Teuchos::rcp(new NOX::Thyra::MatrixFreeJacobianOperator<double>(printParams));
    
    jfnkOp1->setParameterList(jfnkParams);
    
    Teuchos::RCP<NOX::Thyra::Group> noxpred_group =
      Teuchos::rcp(new NOX::Thyra::Group(*initial_guess, 
					 thyraModel, 
					 jfnkOp1, 
					 lowsFactory, 
					 Teuchos::null, 
					 Teuchos::null, 
					 Teuchos::null, 
					 Teuchos::null));
    jfnkOp1->setBaseEvaluationToNOXGroup(noxpred_group.create_weak());
    noxpred_group->computeF();
    Teuchos::RCP<NOX::StatusTest::NormF>relresid1 = 
      Teuchos::rcp(new NOX::StatusTest::NormF(*noxpred_group.get(), relrestol));//1.0e-6 for paper
    Teuchos::RCP<NOX::StatusTest::Combo> converged1 =
      Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::AND));
    converged1->addStatusTest(relresid1);
    //combo->addStatusTest(converged);
    Teuchos::RCP<Teuchos::ParameterList> nl_params1 =
      Teuchos::rcp(new Teuchos::ParameterList);
   
    nl_params1->set("Nonlinear Solver", "Line Search Based");
    nl_params1->sublist("Direction").sublist("Newton").set("Forcing Term Method", "Type 2");
    nl_params1->sublist("Direction").sublist("Newton").set("Forcing Term Initial Tolerance", 1.0e-1);
    nl_params1->sublist("Direction").sublist("Newton").set("Forcing Term Maximum Tolerance", 1.0e-2);
    nl_params1->sublist("Direction").sublist("Newton").set("Forcing Term Minimum Tolerance", 1.0e-5);
    Teuchos::ParameterList& nlPrintParams = nl_params1->sublist("Printing");
    nlPrintParams.set("Output Information",
		      NOX::Utils::OuterIteration  +
		      //                      NOX::Utils::OuterIterationStatusTest +
		      NOX::Utils::InnerIteration +
		      NOX::Utils::Details //+
		      //NOX::Utils::LinearSolverDetails
		    );
    
    predictor_ =  NOX::Solver::buildSolver(noxpred_group, converged1, nl_params1);
    if( 0 == comm_->MyPID() ){
      std::cout<<std::endl<<std::endl<<std::endl<<std::endl;
      std::cout<<std::endl<<std::endl<<std::endl<<std::endl;
      std::cout<<std::endl<<std::endl<<std::endl<<std::endl;
      std::cout<<std::endl<<std::endl<<std::endl<<std::endl;
      std::cout<<std::endl<<std::endl<<std::endl<<std::endl;
      //nl_params1->print(std::cout);
    }
    predictor_->getList().print(std::cout);
    //exit(0);
#endif
  }
#endif //TUSAS_HAVE_CUDA
#endif
