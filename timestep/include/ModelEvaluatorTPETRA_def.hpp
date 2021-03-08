//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
/////////////////////////////////////////////////////////////////////////////


#ifndef NOX_THYRA_MODEL_EVALUATOR_TPETRA_DEF_HPP
#define NOX_THYRA_MODEL_EVALUATOR_TPETRA_DEF_HPP

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>
#include "Teuchos_AbstractFactoryStd.hpp"
//#include <Kokkos_View.hpp> 	
#include <Kokkos_Vector.hpp>
#include <Kokkos_Core.hpp>
//#include <Kokkos_CrsMatrix.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_Import.hpp>

#include <Thyra_TpetraThyraWrappers.hpp>
#include <Thyra_VectorBase.hpp>
#include "Thyra_PreconditionerFactoryBase.hpp"
#include <Thyra_TpetraLinearOp_decl.hpp>
#include "Thyra_DetachedSpmdVectorView.hpp"

//#include <MueLu_ParameterListInterpreter_decl.hpp>
//#include <MueLu_MLParameterListInterpreter_decl.hpp>
//#include <MueLu_ML2MueLuParameterTranslator.hpp>
//#include <MueLu_HierarchyManager.hpp>
//#include <MueLu_Hierarchy_decl.hpp> 
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include "Thyra_MueLuPreconditionerFactory.hpp"

#include <Stratimikos_MueLuHelpers.hpp>

#include "NOX_Thyra_MatrixFreeJacobianOperator.hpp"
#include "NOX_MatrixFree_ModelEvaluatorDecorator.hpp"

#include <algorithm>

//#include <string>

#include "function_def.hpp"
#include "ParamNames.h"

#define TUSAS_RUN_ON_CPU

#define TUSAS_MAX_NUMEQS 2

// IMPORTANT!!! this macro should be set to TUSAS_MAX_NUMEQS * BASIS_NODES_PER_ELEM
#define TUSAS_MAX_NUMEQS_X_BASIS_NODES_PER_ELEM 16

std::string getmypidstring(const int mypid, const int numproc);

template<class Scalar>
Teuchos::RCP<ModelEvaluatorTPETRA<Scalar> >
modelEvaluatorTPETRA( const Teuchos::RCP<const Epetra_Comm>& comm,
			Mesh *mesh,
			 Teuchos::ParameterList plist
			 )
{
  return Teuchos::rcp(new ModelEvaluatorTPETRA<Scalar>(mesh,plist));
}

// Constructor

template<class Scalar>
ModelEvaluatorTPETRA<Scalar>::
ModelEvaluatorTPETRA( const Teuchos::RCP<const Epetra_Comm>& comm,
			Mesh *mesh,
			 Teuchos::ParameterList plist 
		     ) :
  paramList(plist),
  mesh_(mesh),
  Comm(comm)
{
  dt_ = paramList.get<double> (TusasdtNameString);
  t_theta_ = paramList.get<double> (TusasthetaNameString);

  set_test_case();

  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
  //comm_->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
  
  if( 0 == comm_->getRank()) {
    if (sizeof(Mesh::mesh_lint_t) != sizeof(global_ordinal_type) ){
      std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
      std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
      std::cout<<" WARNING::  sizeof(Mesh::mesh_lint_t) != sizeof(global_ordinal_type)"<<std::endl;
      std::cout<<"sizeof(Mesh::mesh_lint_t) = "<<sizeof(Mesh::mesh_lint_t)<<std::endl;
      std::cout<<"sizeof(long long) =  "<<sizeof(long long)<<std::endl;
      std::cout<<"<sizeof(global_ordinal_type) =  "<<sizeof(global_ordinal_type)<<std::endl<<std::endl;
      std::cout<<"This is due to incompatablility with global_ordinal_type in Trilinos"<<std::endl;
      std::cout<<"Mesh::mesh_lint_t in Tusas. Can be addressed via -DNO_MESH_64."<<std::endl;
      std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
      std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    }
  }
  //mesh_ = Teuchos::rcp(new Mesh(*mesh));
  mesh_->compute_nodal_adj(); 

  std::vector<Mesh::mesh_lint_t> node_num_int(mesh_->get_node_num_map());
  std::vector<Mesh::mesh_lint_t>::iterator max;
  std::vector<Mesh::mesh_lint_t>::iterator min;
  max = std::max_element(node_num_int.begin(), node_num_int.end());
  min = std::min_element(node_num_int.begin(), node_num_int.end());
  if(*min < (Mesh::mesh_lint_t) 0){
    if( 0 == comm_->getRank()){
      std::cout<<"node_num_int:   bad min value"<<std::endl<<std::endl;
    }
    //std::cout<<*min<<"   "<<*max<<"   "<<LLONG_MAX<<"   "<<node_num_int.max_size()<<std::endl;
    exit(0);
  }
  if( 0 == comm_->getRank()) std::cout<<"node_num_int constructed"<<std::endl;

  //std::vector<global_ordinal_type> node_num_map(node_num_int.begin(),node_num_int.end());
  std::vector<global_ordinal_type> node_num_map(node_num_int.size());
  for( int i = 0; i < node_num_int.size(); i++ ) node_num_map[i] = (global_ordinal_type)node_num_int[i];
  std::vector<global_ordinal_type>::iterator maxg;
  std::vector<global_ordinal_type>::iterator ming;
  maxg = std::max_element(node_num_map.begin(), node_num_map.end());
  ming = std::min_element(node_num_map.begin(), node_num_map.end());
  if(*ming < (global_ordinal_type) 0){
    if( 0 == comm_->getRank()){
      std::cout<<"node_num_map:   bad min value"<<std::endl<<std::endl;
    }
    //std::cout<<*min<<"   "<<*max<<"   "<<LLONG_MAX<<"   "<<node_num_int.max_size()<<std::endl;
    exit(0);
  }
  if( 0 == comm_->getRank()) std::cout<<"node_num_map constructed"<<std::endl;
  
  std::vector<global_ordinal_type> my_global_nodes(numeqs_*node_num_map.size());
  for(int i = 0; i < node_num_map.size(); i++){    
    global_ordinal_type ngid = node_num_map[i];
    for( int k = 0; k < numeqs_; k++ ){
      my_global_nodes[numeqs_*i+k] = numeqs_*ngid+k;
      //my_global_nodes[numeqs_*i+k] = (global_ordinal_type)numeqs_*(global_ordinal_type)node_num_int[i]+(global_ordinal_type)k;
    }
  }

  const global_size_t numGlobalEntries = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  const global_ordinal_type indexBase = 0;

  const Teuchos::ArrayView<global_ordinal_type> AV(my_global_nodes);

  x_overlap_map_ = Teuchos::rcp(new map_type(numGlobalEntries,
					     AV,
					     indexBase,
					     comm_
					     ));

  //x_overlap_map_ ->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
  GreedyTieBreak<local_ordinal_type,global_ordinal_type> greedy_tie_break;
  x_owned_map_ = Teuchos::rcp(new map_type(*(Tpetra::createOneToOne(x_overlap_map_,greedy_tie_break))));
  //x_owned_map_ ->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
  
  importer_ = Teuchos::rcp(new import_type(x_owned_map_, x_overlap_map_));
  //exporter_ = Teuchos::rcp(new export_type(x_owned_map_, x_overlap_map_));
  exporter_ = Teuchos::rcp(new export_type(x_overlap_map_, x_owned_map_));
  
  num_owned_nodes_ = x_owned_map_->getNodeNumElements()/numeqs_;
  num_overlap_nodes_ = x_overlap_map_->getNodeNumElements()/numeqs_;

  Teuchos::ArrayView<global_ordinal_type> NV(node_num_map);
  node_overlap_map_ = Teuchos::rcp(new map_type(numGlobalEntries,
						NV,
						indexBase,
						comm_
						));

  //cn we could store previous time values in a multivector
  u_old_ = Teuchos::rcp(new vector_type(x_owned_map_));
  u_old_->putScalar(Teuchos::ScalarTraits<scalar_type>::zero());

  x_ = Teuchos::rcp(new vector_type(node_overlap_map_));
  y_ = Teuchos::rcp(new vector_type(node_overlap_map_));
  z_ = Teuchos::rcp(new vector_type(node_overlap_map_));
  Teuchos::ArrayRCP<scalar_type> xv = x_->get1dViewNonConst();
  Teuchos::ArrayRCP<scalar_type> yv = y_->get1dViewNonConst();
  Teuchos::ArrayRCP<scalar_type> zv = z_->get1dViewNonConst();
  const size_t localLength = node_overlap_map_->getNodeNumElements();
  for (size_t nn=0; nn < localLength; nn++) {
    xv[nn] = mesh_->get_x(nn);
    yv[nn] = mesh_->get_y(nn);
    zv[nn] = mesh_->get_z(nn);
  }
 
  x_space_ = Thyra::createVectorSpace<scalar_type>(x_owned_map_);
  f_space_ = x_space_;
  //x0_ = Thyra::createMember(x_space_);
  x0_ = Teuchos::rcp(new vector_type(x_owned_map_));
  x0_->putScalar(Teuchos::ScalarTraits<scalar_type>::zero());

  bool precon = paramList.get<bool> (TusaspreconNameString);
  if(precon){
    // Initialize the graph for W CrsMatrix object
    W_graph_ = createGraph();
    W_overlap_graph_ = createOverlapGraph();
    P = rcp(new matrix_type(W_overlap_graph_));
    P_ = rcp(new matrix_type(W_graph_));
    P->setAllToScalar((scalar_type)1.0); 
    P->fillComplete();
    //cn we need to fill the matrix for muelu
    init_P_();
    //P_->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
    
    Teuchos::ParameterList mueluParamList;

    std::string optionsFile = "mueluOptions.xml";  
    Teuchos::updateParametersFromXmlFileAndBroadcast(optionsFile,Teuchos::Ptr<Teuchos::ParameterList>(&mueluParamList), *P_->getComm());
    if( 0 == comm_->getRank() ){
      std::cout << "\nReading MueLu parameter list from the XML file \""<<optionsFile<<"\" ...\n";
      mueluParamList.print(std::cout, 2, true, true );
    }

    prec_ = MueLu::CreateTpetraPreconditioner<scalar_type,local_ordinal_type, global_ordinal_type, node_type>(P_, mueluParamList);
#if 0
    prec_ = MueLu::CreateTpetraPreconditioner<scalar_type,local_ordinal_type, global_ordinal_type, node_type>(P_, mueluParamList, mueluParamList);
#endif
    //prec_ = MueLu::CreateTpetraPreconditioner<scalar_type,local_ordinal_type, global_ordinal_type, node_type>(P_,optionsFile  );
    //exit(0);
    if( 0 == comm_->getRank() ){
      std::cout <<" MueLu preconditioner created"<<std::endl<<std::endl;
    }
  }


  Thyra::ModelEvaluatorBase::InArgsSetup<scalar_type> inArgs;
  inArgs.setModelEvalDescription(this->description());
  inArgs.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x);
  prototypeInArgs_ = inArgs;

  Thyra::ModelEvaluatorBase::OutArgsSetup<scalar_type> outArgs;
  outArgs.setModelEvalDescription(this->description());
  outArgs.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_f);
  outArgs.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W_prec);
  prototypeOutArgs_ = outArgs;
  nominalValues_ = inArgs;
  //nominalValues_.set_x(x0_);
  nominalValues_.set_x(Thyra::createVector(x0_, x_space_));
  time_=0.;
  
  ts_time_import= Teuchos::TimeMonitor::getNewTimer("Tusas: Total Import Time");
  ts_time_resfill= Teuchos::TimeMonitor::getNewTimer("Tusas: Total Residual Fill Time");
  ts_time_precfill= Teuchos::TimeMonitor::getNewTimer("Tusas: Total Preconditioner Fill Time");
  ts_time_nsolve= Teuchos::TimeMonitor::getNewTimer("Tusas: Total Nonlinear Solver Time");
  ts_time_view= Teuchos::TimeMonitor::getNewTimer("Tusas: Total View Time");
  ts_time_iowrite= Teuchos::TimeMonitor::getNewTimer("Tusas: Total IO Write Time");
  //ts_time_ioread= Teuchos::TimeMonitor::getNewTimer("Tusas: Total IO Read Time");

  //HACK
  //cn 8-28-18 currently elem_color takes an epetra_mpi_comm....
  //there are some epetra_maps and a routine that does mpi calls for off proc comm const 
  //Comm = Teuchos::rcp(new Epetra_MpiComm( MPI_COMM_WORLD ));
  bool dorestart = paramList.get<bool> (TusasrestartNameString);
  Elem_col = Teuchos::rcp(new elem_color(Comm,mesh,dorestart));

  init_nox();

  std::vector<int> indices = (Teuchos::getArrayFromStringParameter<int>(paramList,
								       TusaserrorestimatorNameString)).toVector();
  std::vector<int>::iterator it;
  for(it = indices.begin();it != indices.end(); ++it){
    //std::cout<<*it<<" "<<std::endl;
    int error_index = *it;
    Error_est.push_back(new error_estimator(Comm,mesh_,numeqs_,error_index));
  }


}

template<class Scalar>
Teuchos::RCP<Tpetra::CrsMatrix<>::crs_graph_type> ModelEvaluatorTPETRA<Scalar>::createGraph()
{
  Teuchos::RCP<crs_graph_type> W_graph;

  int numind = 9*numeqs_;//this is an approximation 9 for lquad; 25 for qquad; 9*3 for lhex; 25*3 for qhex; 6 ltris ??, tets ??
                         //this was causing problems with clang
  if(3 == mesh_->get_num_dim() ) numind = 27*numeqs_;

  size_t ni = numind;

  W_graph = Teuchos::rcp(new crs_graph_type(x_owned_map_, ni));

  for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){
    
    int n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);
    for (int ne=0; ne < mesh_->get_num_elem_in_blk(blk); ne++) {
      for (int i=0; i< n_nodes_per_elem; i++) {
	int row = numeqs_*(
			   mesh_->get_global_node_id(mesh_->get_node_id(blk, ne, i))
			   ); 
	for(int j=0;j < n_nodes_per_elem; j++) {
	  int column = numeqs_*(mesh_->get_global_node_id(mesh_->get_node_id(blk, ne, j)));

	  for( int k = 0; k < numeqs_; k++ ){
	    global_ordinal_type row1 = row + k;
	    global_ordinal_type column1 = column + k;
	    Teuchos::ArrayView<global_ordinal_type> CV(&column1,1);

	    //W_graph->InsertGlobalIndices((int)1,&row1, (int)1, &column1);
	    //W_graph->insertGlobalIndices(row1, (local_ordinal_type)1, column1);
	    W_graph->insertGlobalIndices(row1, CV);

	  }
	}
      }
    }
  }

  W_graph->fillComplete();

  //W_graph->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
//   exit(0);

  return W_graph;
}

template<class Scalar>
Teuchos::RCP<Tpetra::CrsMatrix<>::crs_graph_type> ModelEvaluatorTPETRA<Scalar>::createOverlapGraph()
{
  Teuchos::RCP<crs_graph_type> W_graph;

  int numind = 9*numeqs_;//this is an approximation 9 for lquad; 25 for qquad; 9*3 for lhex; 25*3 for qhex; 6 ltris ??, tets ??
                         //this was causing problems with clang
  if(3 == mesh_->get_num_dim() ) numind = 27*numeqs_;

  size_t ni = numind;

  W_graph = Teuchos::rcp(new crs_graph_type(x_overlap_map_, ni));

  for(int blk = 0; blk < mesh_->get_num_elem_blks(); blk++){
    
    const int n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);

    const int num_elem = (*mesh_->get_elem_num_map()).size();

    for (int ne=0; ne < num_elem; ne++) {
      for (int i=0; i< n_nodes_per_elem; i++) {
	const global_ordinal_type row = numeqs_*(
			   mesh_->get_global_node_id(mesh_->get_node_id(blk, ne, i))
			   ); 
	for(int j=0;j < n_nodes_per_elem; j++) {
	  const global_ordinal_type column = numeqs_*(mesh_->get_global_node_id(mesh_->get_node_id(blk, ne, j)));

	  for( int k = 0; k < numeqs_; k++ ){
	    const global_ordinal_type row1 = row + k;
	    global_ordinal_type column1 = column + k;
	    Teuchos::ArrayView<global_ordinal_type> CV(&column1,1);

	    //W_graph->InsertGlobalIndices((int)1,&row1, (int)1, &column1);
	    //W_graph->insertGlobalIndices(row1, (local_ordinal_type)1, column1);
	    W_graph->insertGlobalIndices(row1, CV);

	  }
	}
      }
    }
  }

  W_graph->fillComplete();

  //W_graph->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );;
  //exit(0);

  return W_graph;
}

template<class Scalar>
void ModelEvaluatorTPETRA<Scalar>::set_x0(const Teuchos::ArrayView<const Scalar> &x0_in)
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_EQUALITY(x_space_->dim(), x0_in.size());
#endif
  x0_->get1dViewNonConst()().assign(x0_in);
}

template<class Scalar>
void ModelEvaluatorTPETRA<Scalar>::evalModelImpl(
  const Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
  const Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs
  ) const
{  

  //cn the easiest way probably to do the sum into off proc nodes is to load a 
  //vector(overlap_map) the export with summation to the f_vec(owned_map)
  //after summing into. ie import is uniquely-owned to multiply-owned
  //export is multiply-owned to uniquely-owned

  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 

  typedef Thyra::TpetraOperatorVectorExtraction<Scalar,int> ConverterT;

  const Teuchos::RCP<const vector_type > x_vec =
    ConverterT::getConstTpetraVector(inArgs.get_x());

  Teuchos::RCP<vector_type > u = Teuchos::rcp(new vector_type(x_overlap_map_));
  Teuchos::RCP<vector_type > uold = Teuchos::rcp(new vector_type(x_overlap_map_));
  {
    Teuchos::TimeMonitor ImportTimer(*ts_time_import);
    u->doImport(*x_vec,*importer_,Tpetra::INSERT);
    uold->doImport(*u_old_,*importer_,Tpetra::INSERT);
  }

  auto x_view = x_->getLocalView<Kokkos::DefaultExecutionSpace>();
  auto y_view = y_->getLocalView<Kokkos::DefaultExecutionSpace>();
  auto z_view = z_->getLocalView<Kokkos::DefaultExecutionSpace>();
  //using RandomAccess should give better memory performance on better than tesla gpus (guido is tesla and does not show performance increase)
  //this will utilize texture memory not available on tesla or earlier gpus
  Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> x_1dra = Kokkos::subview (x_view, Kokkos::ALL (), 0);
  Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> y_1dra = Kokkos::subview (y_view, Kokkos::ALL (), 0);
  Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> z_1dra = Kokkos::subview (z_view, Kokkos::ALL (), 0);

  auto u_view = u->getLocalView<Kokkos::DefaultExecutionSpace>();
  Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> 
    u_1dra = Kokkos::subview (u_view, Kokkos::ALL (), 0);

  const int blk = 0;
  const int n_nodes_per_elem = mesh_->get_num_nodes_per_elem_in_blk(blk);//shared
  const int num_color = Elem_col->get_num_color();

  Kokkos::View<int*,Kokkos::DefaultExecutionSpace> meshc_1d("meshc_1d",((mesh_->connect)[0]).size());


  Kokkos::View<int**,Kokkos::DefaultExecutionSpace> meshc_2d("meshc_2d",n_nodes_per_elem,(*mesh_->get_elem_num_map()).size());

  //std::cout<<n_nodes_per_elem*(*mesh_->get_elem_num_map()).size()<<"  "<<((mesh_->connect)[0]).size()<<std::endl;

  //Kokkos::vector<int> meshc(((mesh_->connect)[0]).size());
  for(int i = 0; i<((mesh_->connect)[0]).size(); i++) {
    meshc_1d(i)=(mesh_->connect)[0][i];
  }
  Kokkos::View<const int*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> meshc_1dra(meshc_1d);

  const double dt = dt_; //cuda 8 lambdas dont capture private data
  const double t_theta = t_theta_; //cuda 8 lambdas dont capture private data
  const double time = time_; //cuda 8 lambdas dont capture private data
  const int numeqs = numeqs_; //cuda 8 lambdas dont capture private data
  const int LTP_quadrature_order = paramList.get<int> (TusasltpquadordNameString);
  if (4 <  LTP_quadrature_order ){
      if( 0 == comm_->getRank() ){
	std::cout<<std::endl<<std::endl<<"4 <  LTP_quadrature_order" <<std::endl<<std::endl<<std::endl;
      }
      exit(0);
  }
  
  if (nonnull(outArgs.get_f())){

    const Teuchos::RCP<vector_type> f_vec =
      ConverterT::getTpetraVector(outArgs.get_f());

    Teuchos::RCP<vector_type> f_overlap = Teuchos::rcp(new vector_type(x_overlap_map_));
    f_vec->scale(0.);
    f_overlap->scale(0.);
    Teuchos::TimeMonitor ResFillTimer(*ts_time_resfill);  

//     std::string elem_type=mesh_->get_blk_elem_type(blk);
//     std::string * elem_type_p = &elem_type;

    auto uold_view = uold->getLocalView<Kokkos::DefaultExecutionSpace>();
    
    auto f_view = f_overlap->getLocalView<Kokkos::DefaultExecutionSpace>();
        
    auto f_1d = Kokkos::subview (f_view, Kokkos::ALL (), 0);
    //Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> f_1d = Kokkos::subview (f_view, Kokkos::ALL (), 0);

    //using RandomAccess should give better memory performance on better than tesla gpus (guido is tesla and does not show performance increase)
    //this will utilize texture memory not available on tesla or earlier gpus
    Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> 
      uold_1dra = Kokkos::subview (uold_view, Kokkos::ALL (), 0);

    RESFUNC * h_rf;
    h_rf = (RESFUNC*)malloc(numeqs_*sizeof(RESFUNC));

#ifdef TUSAS_HAVE_CUDA
    RESFUNC * d_rf;
    cudaMalloc((double**)&d_rf,numeqs_*sizeof(RESFUNC));

    if("heat" == paramList.get<std::string> (TusastestNameString)){
      //cn this will need to be done for each equation
      cudaMemcpyFromSymbol( &h_rf[0], tpetra::residual_heat_test_dp_, sizeof(RESFUNC));
    }else if("NLheatIMR" == paramList.get<std::string> (TusastestNameString)){
      //cn this will need to be done for each equation
      //kodiak cudaMemcpyFromSymbol( &h_rf[0], tpetra::residual_nlheatimr_test_dp_, sizeof(RESFUNC));
    }else if("NLheatCN" == paramList.get<std::string> (TusastestNameString)){
      //cn this will need to be done for each equation
      //kodiak cudaMemcpyFromSymbol( &h_rf[0], tpetra::residual_nlheatcn_test_dp_, sizeof(RESFUNC));
    }else if("heat2" == paramList.get<std::string> (TusastestNameString)){
      cudaMemcpyFromSymbol( &h_rf[0], tpetra::residual_heat_test_dp_, sizeof(RESFUNC));
      cudaMemcpyFromSymbol( &h_rf[1], tpetra::residual_heat_test_dp_, sizeof(RESFUNC));
    }else if("farzadi" == paramList.get<std::string> (TusastestNameString)){
      //kodiak cudaMemcpyFromSymbol( &h_rf[0], tpetra::farzadi3d::residual_conc_farzadi_dp_, sizeof(RESFUNC));
      //kodiak cudaMemcpyFromSymbol( &h_rf[1], tpetra::farzadi3d::residual_phase_farzadi_dp_, sizeof(RESFUNC));
    }else if("farzadi_test" == paramList.get<std::string> (TusastestNameString)){
      //kodiak cudaMemcpyFromSymbol( &h_rf[0], tpetra::farzadi3d::residual_conc_farzadi_dp_, sizeof(RESFUNC));
      //kodiak cudaMemcpyFromSymbol( &h_rf[1], tpetra::farzadi3d::residual_phase_farzadi_dp_, sizeof(RESFUNC));
    }else if("pfhub3" == paramList.get<std::string> (TusastestNameString)){
      //kodiak cudaMemcpyFromSymbol( &h_rf[0], tpetra::pfhub3::residual_heat_pfhub3_dp_, sizeof(RESFUNC));
      //kodiak cudaMemcpyFromSymbol( &h_rf[1], tpetra::pfhub3::residual_phase_pfhub3_dp_, sizeof(RESFUNC));
    }else if("pfhub2kks" == paramList.get<std::string> (TusastestNameString)){
      //kodiak cudaMemcpyFromSymbol( &h_rf[0], tpetra::pfhub2::residual_c_kks_dp_, sizeof(RESFUNC));
      //kodiak cudaMemcpyFromSymbol( &h_rf[1], tpetra::pfhub2::residual_eta_kks_dp_, sizeof(RESFUNC));

    } else {
      if( 0 == comm_->getRank() ){
	std::cout<<std::endl<<std::endl<<"Test case: "<<paramList.get<std::string> (TusastestNameString)
		 <<" residual function not found. (void ModelEvaluatorTPETRA<Scalar>::evalModelImpl(...))" <<std::endl<<std::endl<<std::endl;
      }
      exit(0);
    }

    cudaMemcpy(d_rf,h_rf,numeqs_*sizeof(RESFUNC),cudaMemcpyHostToDevice);

#else
    //it seems that evaluating the function via pointer ie h_rf[0] is way faster that evaluation via (*residualfunc_)[0]
    h_rf = &(*residualfunc_)[0];
#endif

    //cn want this stuff in the ifdef but also outside of re and prec fills
//     GPUBasis* f_1 = (GPUBasis*)Kokkos::kokkos_malloc<Kokkos::CudaUVMSpace>(sizeof(GPUBasis));
//     f_1 = (GPUBasis*)Kokkos::kokkos_malloc<Kokkos::CudaSpace>(sizeof(GPUBasis));

//     Kokkos::fence();     
//     auto f_1 = Kokkos::View<GPUBasis*, Kokkos::CudaUVMSpace>
//       (Kokkos::ViewAllocateWithoutInitializing ("view_group"), 1);
//     //printf("CreateObjects1\n");

//     Kokkos::fence();

   


    for(int c = 0; c < num_color; c++){
      //std::vector<int> elem_map = colors[c];
      const std::vector<int> elem_map = Elem_col->get_color(c);//local

      const int num_elem = elem_map.size();

      Kokkos::View<int*,Kokkos::DefaultExecutionSpace> elem_map_1d("elem_map_1d",num_elem);
      //Kokkos::vector<int> elem_map_k(num_elem);
      for(int i = 0; i<num_elem; i++) {
	//elem_map_k[i] = elem_map[i]; 
	elem_map_1d(i) = elem_map[i]; 
      }
      
      //Kokkos::fence();     
      //auto f_1 = Kokkos::View<GPUBasis*, Kokkos::CudaSpace>(Kokkos::ViewAllocateWithoutInitializing ("view_group"), 1);
      //printf("CreateObjects1\n");   
      //Kokkos::View<GPUBasis*,Kokkos::Cuda> f_1(Kokkos::ViewAllocateWithoutInitializing ("view_group"), 1);
      //Kokkos::View<GPUBasis*, Kokkos::Device<Kokkos::Cuda,Kokkos::CudaSpace>> 
      //	f_1(Kokkos::ViewAllocateWithoutInitializing ("view_group"), 1);
      //GPUBasis* f_1 = (GPUBasis*)Kokkos::kokkos_malloc<Kokkos::Cuda>(sizeof(GPUBasisLQuad));

      
      //Kokkos::fence();

   




      //auto elem_map_2d = Kokkos::subview(elem_map_1d, Kokkos::ALL (), Kokkos::ALL (), 0);
      //std::cout<<elem_map_2d.extent(0)<<"   "<<elem_map_2d.extent(1)<<std::endl;
      //for (int ne=0; ne < num_elem; ne++) { 
      //#define USE_TEAM
#ifdef USE_TEAM
#ifdef TUSAS_HAVE_CUDA
      int team_size = 512;//this is teamsize (#of threads in team) < 1024; preferably 256
#else
      int team_size = 1;//openmp
#endif
      int num_teams = (num_elem/team_size)+1;//this is # of thread teams (also league size); unlimited
      Kokkos::View<const int*,Kokkos::DefaultExecutionSpace> elem_map_1dConst(elem_map_1d);

//       int strides[1]; // any integer type works in stride()
//       elem_map_1dConst.stride (strides);
//       std::cout<<strides[0]<<std::endl;


      typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type member_type;

      //TeamPolicy <ExecutionSpace >( numberOfTeams , teamSize)
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy (num_teams, team_size );
      //std::cout<<policy.league_size()<<"    "<<policy.team_size()<<std::endl;
      Kokkos::parallel_for (policy, KOKKOS_LAMBDA (member_type team_member) {
        // Calculate a global thread id
        int ne = team_member.league_rank () * team_member.team_size () +
                team_member.team_rank ();
	if(ne < num_elem) {
	  const int elem = elem_map_1dConst(ne);
#else




    Kokkos::fence();
      Kokkos::parallel_for(num_elem,KOKKOS_LAMBDA(const size_t ne){//this loop is fine for openmp re access to elem_map
			     //for(int ne =0; ne<num_elem; ne++){
	const int elem = elem_map_1d(ne);
#endif


	//const int ngp = BGPU[0]->ngp;

	double xx[BASIS_NODES_PER_ELEM];
	double yy[BASIS_NODES_PER_ELEM];
	double zz[BASIS_NODES_PER_ELEM];

	double uu[TUSAS_MAX_NUMEQS_X_BASIS_NODES_PER_ELEM];
	double uu_old[TUSAS_MAX_NUMEQS_X_BASIS_NODES_PER_ELEM];

	const int elemrow = elem*n_nodes_per_elem;

	for(int k = 0; k < n_nodes_per_elem; k++){
	  
	  //meshc is a kokkos::vector, not sure how efficient this is
	  //maybe a view is better?
	  //const int nodeid = meshc[elemrow+k];//cn this is the local id
	  const int nodeid = meshc_1dra(elemrow+k);//cn this is the local id
	  
	  xx[k] = x_1dra(nodeid);
	  yy[k] = y_1dra(nodeid);
	  zz[k] = z_1dra(nodeid);

	  //std::cout<<k<<"   "<<xx[k]<<"   "<<yy[k]<<"   "<<zz[k]<<"   "<<nodeid<<std::endl;
	  for( int neq = 0; neq < numeqs; neq++ ){
	    //std::cout<<numeqs*k+neq<<"           "<<n_nodes_per_elem*neq+k <<"      "<<nodeid<<"    "<<numeqs_*nodeid+neq<<std::endl;

	    uu[n_nodes_per_elem*neq+k] = u_1dra(numeqs*nodeid+neq); 
	    uu_old[n_nodes_per_elem*neq+k] = uold_1dra(numeqs*nodeid+neq);
	  }//neq
	}//k
	//printf("computeElemData step1\n");

        //new(&f_1(0)) GPUBasisLQuad(2);  
	//new((GPUBasisLQuad*)f_1) GPUBasisLQuad(2);

	GPUBasisLQuadNew B;

	//printf("%d\n",B.getNgp());

	for( int neq = 0; neq < numeqs; neq++ ){
	  //BGPU[neq]->computeElemData(&xx[0], &yy[0], &zz[0]);
	  //f_1(neq).computeElemData(&xx[0], &yy[0], &zz[0]);//we could interp x y z here to save time?
	  //f_1->computeElemData(&xx[0], &yy[0], &zz[0]);//we could interp x y z here to save time?
	  B.computeElemData(&xx[0], &yy[0], &zz[0]);//we could interp x y z here to save time?
	}//neq
	//printf("computeElemData step2\n");

	//const int ngp = f_1(0).getNgp();
	const int ngp = B.getNgp();

	for(int gp=0; gp < ngp; gp++) {//gp
	  //double jb =0.;
	  double jf =0.;
	  for( int neq = 0; neq < numeqs; neq++ ){
	    //we need a basis object that stores all equations here..
	    //jb=BGPU[neq]->getBasis(gp, &xx[0], &yy[0], &zz[0], &uu[neq*n_nodes_per_elem], &uu_old[neq*n_nodes_per_elem],NULL);
	    //jf=f_1(neq).getBasis(gp, &xx[0], &yy[0], &zz[0], &uu[neq*n_nodes_per_elem], &uu_old[neq*n_nodes_per_elem]);
	    //jf=f_1->getBasis(gp, &xx[0], &yy[0], &zz[0], &uu[neq*n_nodes_per_elem], &uu_old[neq*n_nodes_per_elem]);

	    jf=B.getBasis(gp, &xx[0], &yy[0], &zz[0], &uu[neq*n_nodes_per_elem], &uu_old[neq*n_nodes_per_elem]);
	  }//neq

	  const double jacwt = jf;
	  for (int i=0; i< n_nodes_per_elem; i++) {//i

	    //const int lrow = numeqs*meshc[elemrow+i];
	    const int lrow = numeqs*meshc_1dra(elemrow+i);

	    for( int neq = 0; neq < numeqs; neq++ ){
#ifdef TUSAS_HAVE_CUDA
	      //const double val = jacwt*((d_rf[neq])(BGPU,i,dt,t_theta,time,neq));
	      //const double val = jacwt*((d_rf[neq])(f_1(neq),i,dt,t_theta,time,neq));
	      double val =0;
	      //if(1==c) val = jacwt*((d_rf[neq])(*f_1,i,dt,t_theta,time,neq));
	      val = jacwt*((d_rf[neq])(B,i,dt,t_theta,time,neq));
	      //printf("BGPU %d %le %le\n",neq,jacwt*((d_rf[neq])(*BGPU,i,dt,t_theta,time,neq)),val);
#else
	      //const double val = BGPU->jac*BGPU->wt*(*residualfunc_)[0](BGPU,i,dt,1.,0.,0);
	      const double val = jacwt*(h_rf[neq](*BGPU,i,dt,t_theta,time,neq));
#endif
	      //cn this works because we are filling an overlap map and exporting to a node map below...
	      const int lid = lrow+neq;
	      //std::cout<<lid<<"   "<<jacwt*(h_rf[neq](*BGPU,i,dt,t_theta,time,neq))<<std::endl;
	      f_1d[lid] += val;
	      //if(0==c) printf("%d %le %le %d\n",lid,jacwt*((d_rf[neq])(*f_1,i,dt,t_theta,time,neq)),f_1d[lid],c);
	    }//neq
	  }//i
	}//gp
#ifdef USE_TEAM
			       }//if ne
#else
#endif
			   //f_1(0).~GPUBasis();
			   //f_1->~GPUBasis();
      });//parallel_for
		     //};//ne
	//Kokkos::kokkos_free<Kokkos::CudaSpace>(f_1);
	//Kokkos::fence();
    }//c 

#ifdef TUSAS_HAVE_CUDA
  cudaFree(d_rf);
  free(h_rf);
#endif


    {
      Teuchos::TimeMonitor ImportTimer(*ts_time_import);  
      f_vec->doExport(*f_overlap, *exporter_, Tpetra::ADD);
    }
//    f_overlap->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
//     f_vec->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
//     auto f_view = f_overlap->getLocalView<Kokkos::HostSpace>();
//     auto f_1d = Kokkos::subview (f_view, Kokkos::ALL (), 0);
//     for(int i = 0; i<15; i++)std::cout<<comm_->getRank()<<" "<<i<<" "<<f_1d[i]<<" "<<x_overlap_map_->getGlobalElement(i)<<std::endl;
    //exit(0);

  }//get_f

  if (nonnull(outArgs.get_f()) && NULL != dirichletfunc_){
    const Teuchos::RCP<vector_type> f_vec =
      ConverterT::getTpetraVector(outArgs.get_f());
    std::vector<Mesh::mesh_lint_t> node_num_map(mesh_->get_node_num_map());
    std::map<int,DBCFUNC>::iterator it;
    
    Teuchos::RCP<vector_type> f_overlap = Teuchos::rcp(new vector_type(x_overlap_map_));
    {
      Teuchos::TimeMonitor ImportTimer(*ts_time_import);
      f_overlap->doImport(*f_vec,*importer_,Tpetra::INSERT);
    }

    //u is already imported to overlap_map here
    auto u_view = u->getLocalView<Kokkos::DefaultExecutionSpace>();
    auto f_view = f_overlap->getLocalView<Kokkos::DefaultExecutionSpace>();
    
    
    //auto u_1d = Kokkos::subview (u_view, Kokkos::ALL (), 0);
    Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> u_1dra = Kokkos::subview (u_view, Kokkos::ALL (), 0);
    auto f_1d = Kokkos::subview (f_view, Kokkos::ALL (), 0);
	

    for( int k = 0; k < numeqs_; k++ ){
      for(it = (*dirichletfunc_)[k].begin();it != (*dirichletfunc_)[k].end(); ++it){
	const int ns_id = it->first;
	const int num_node_ns = mesh_->get_node_set(ns_id).size();


	size_t ns_size = (mesh_->get_node_set(ns_id)).size();
	Kokkos::View <int*> node_set_view("nsv",ns_size);
	for (size_t i = 0; i < ns_size; ++i) {
	  node_set_view(i) = (mesh_->get_node_set(ns_id))[i];
        }

#ifdef TUSAS_RUN_ON_CPU	
 	for ( int j = 0; j < num_node_ns; j++ ){
#else
	Kokkos::parallel_for(num_node_ns,KOKKOS_LAMBDA (const size_t& j){
#endif
			       const int lid = node_set_view(j);//could use Kokkos::vector here...

#ifdef TUSAS_RUN_ON_CPU	
			       const double val1 = (it->second)(0.,0.,0.,time);
#else
			       const double val1 = tusastpetra::dbc_zero_(0.,0.,0.,time);
#endif
			       const double val = u_1dra(numeqs_*lid + k)  - val1;
			       f_1d(numeqs_*lid + k) = val;
#ifdef TUSAS_RUN_ON_CPU	
			     }//j
#else
			     });//parallel_for
#endif
      }//it
    }//k
    {
      Teuchos::TimeMonitor ImportTimer(*ts_time_import);  
      f_vec->doExport(*f_overlap, *exporter_, Tpetra::REPLACE);//REPLACE ???
    }
  }//get_f
      
  if( nonnull(outArgs.get_W_prec() )){

    Teuchos::TimeMonitor PrecFillTimer(*ts_time_precfill);

    P_->resumeFill();
    P_->setAllToScalar((scalar_type)0.0); 

    P->resumeFill();
    P->setAllToScalar((scalar_type)0.0); 

    auto PV = P->getLocalMatrix();//this is a KokkosSparse::CrsMatrix<scalar_type,local_ordinal_type, node_type> PV = P->getLocalMatrix();


    PREFUNC * h_pf;
    h_pf = (PREFUNC*)malloc(numeqs_*sizeof(PREFUNC));

#ifdef TUSAS_HAVE_CUDA
    PREFUNC * d_pf;
    cudaMalloc((double**)&d_pf,numeqs_*sizeof(PREFUNC));

    if("heat" == paramList.get<std::string> (TusastestNameString)){
      cudaMemcpyFromSymbol( &h_pf[0], tpetra::prec_heat_test_dp_, sizeof(PREFUNC));
    }else if("heat2" == paramList.get<std::string> (TusastestNameString)){
      //kodiak cudaMemcpyFromSymbol( &h_pf[0], tpetra::prec_heat_test_dp_, sizeof(PREFUNC));
      //kodiak cudaMemcpyFromSymbol( &h_pf[1], tpetra::prec_heat_test_dp_, sizeof(PREFUNC));
    }else if("farzadi" == paramList.get<std::string> (TusastestNameString)){
      //kodiak cudaMemcpyFromSymbol( &h_pf[0], tpetra::farzadi3d::prec_conc_farzadi_dp_, sizeof(PREFUNC));
      //kodiak cudaMemcpyFromSymbol( &h_pf[1], tpetra::farzadi3d::prec_phase_farzadi_dp_, sizeof(PREFUNC));
    }else if("farzadi_test" == paramList.get<std::string> (TusastestNameString)){
      //kodiak cudaMemcpyFromSymbol( &h_pf[0], tpetra::farzadi3d::prec_conc_farzadi_dp_, sizeof(PREFUNC));
      //kodiak cudaMemcpyFromSymbol( &h_pf[1], tpetra::farzadi3d::prec_phase_farzadi_dp_, sizeof(PREFUNC));
    }else if("pfhub3" == paramList.get<std::string> (TusastestNameString)){
      //kodiak cudaMemcpyFromSymbol( &h_pf[0], tpetra::pfhub3::prec_heat_pfhub3_dp_, sizeof(PREFUNC));
      //kodiak cudaMemcpyFromSymbol( &h_pf[1], tpetra::pfhub3::prec_phase_pfhub3_dp_, sizeof(PREFUNC));
    }else if("NLheatCN" == paramList.get<std::string> (TusastestNameString)){
      //kodiak cudaMemcpyFromSymbol( &h_pf[0], tpetra::prec_nlheatcn_test_dp_, sizeof(PREFUNC));
    }else if("NLheatIMR" == paramList.get<std::string> (TusastestNameString)){
      //kodiak cudaMemcpyFromSymbol( &h_pf[0], tpetra::prec_nlheatcn_test_dp_, sizeof(PREFUNC));
    } else {
      if( 0 == comm_->getRank() ){
	std::cout<<std::endl<<std::endl<<"Test case: "<<paramList.get<std::string> (TusastestNameString)
		 <<" precon function not found. (void ModelEvaluatorTPETRA<Scalar>::evalModelImpl(...))" <<std::endl<<std::endl<<std::endl;
      }
      exit(0);
    }

    cudaMemcpy(d_pf,h_pf,numeqs_*sizeof(PREFUNC),cudaMemcpyHostToDevice);

#else
    h_pf = &(*preconfunc_)[0];
#endif


    for(int c = 0; c < num_color; c++){
      //std::vector<int> elem_map = colors[c];
      std::vector<int> elem_map = Elem_col->get_color(c);

      const int num_elem = elem_map.size();


      Kokkos::View<int*,Kokkos::DefaultExecutionSpace> elem_map_1d("elem_map_1d",num_elem);
      //Kokkos::vector<int> elem_map_k(num_elem);
      for(int i = 0; i<num_elem; i++) {
	//elem_map_k[i] = elem_map[i];
	elem_map_1d(i) = elem_map[i]; 
	//std::cout<<comm_->getRank()<<" "<<c<<" "<<i<<" "<<elem_map_k[i]<<std::endl;
      }
      //exit(0);	

      Kokkos::parallel_for(num_elem,KOKKOS_LAMBDA(const size_t ne){


	GPUBasis * BGPU[TUSAS_MAX_NUMEQS];
	
	GPUBasisLQuad Bq[TUSAS_MAX_NUMEQS];
	GPUBasisLHex Bh[TUSAS_MAX_NUMEQS];
	if(4 == n_nodes_per_elem)  {
	  for( int neq = 0; neq < numeqs; neq++ )
	    BGPU[neq] = &Bq[neq];
	}else{
	  for( int neq = 0; neq < numeqs; neq++ )
	    BGPU[neq] = &Bh[neq];
	}
	
	const int ngp = BGPU[0]->getNgp();

	//const int elem = elem_map_k[ne];
	const int elem = elem_map_1d(ne);

	double xx[BASIS_NODES_PER_ELEM];
	double yy[BASIS_NODES_PER_ELEM];
	double zz[BASIS_NODES_PER_ELEM];
	double uu[TUSAS_MAX_NUMEQS_X_BASIS_NODES_PER_ELEM];

	const int elemrow = elem*n_nodes_per_elem;
	for(int k = 0; k < n_nodes_per_elem; k++){
	  
	  //const int nodeid = meshc[elemrow+k];
	  const int nodeid = meshc_1d(elemrow+k);
	  
	  xx[k] = x_1dra(nodeid);
	  yy[k] = y_1dra(nodeid);
	  zz[k] = z_1dra(nodeid);

	  for( int neq = 0; neq < numeqs; neq++ ){
	    uu[n_nodes_per_elem*neq+k] = u_1dra(numeqs*nodeid+neq); 
	  }//neq
	}//k

	for( int neq = 0; neq < numeqs; neq++ ){
	  BGPU[neq]->computeElemData(&xx[0], &yy[0], &zz[0]);
	}//neq

	for(int gp=0; gp < ngp; gp++) {//gp
	  for( int neq = 0; neq < numeqs; neq++ ){
	    BGPU[neq]->getBasis(gp, &xx[0], &yy[0], &zz[0], &uu[neq*n_nodes_per_elem], NULL);
	  }//neq
	  const double jacwt = BGPU[0]->getJac()*BGPU[0]->getWt();
	  for (int i=0; i< n_nodes_per_elem; i++) {//i
	    //const local_ordinal_type lrow = numeqs*meshc[elemrow+i];
	    const local_ordinal_type lrow = numeqs*meshc_1d(elemrow+i);
	    for(int j=0;j < n_nodes_per_elem; j++) {
	      //local_ordinal_type lcol[1] = {numeqs*meshc[elemrow+j]};
	      local_ordinal_type lcol[1] = {numeqs*meshc_1d(elemrow+j)};
	      
	      for( int neq = 0; neq < numeqs; neq++ ){
#ifdef TUSAS_HAVE_CUDA
		scalar_type val[1] = {jacwt*d_pf[neq](*BGPU,i,j,dt,t_theta,neq)};
#else
		scalar_type val[1] = {jacwt*h_pf[neq](*BGPU,i,j,dt,t_theta,neq)};
#endif
		
		//cn probably better to fill a view for val and lcol for each column
		const local_ordinal_type row = lrow +neq; 
		local_ordinal_type col[1] = {lcol[0] + neq};
		
		//cn this call seems to be what is crashing the cuda version
		
	      //P->sumIntoLocalValues(lrow,(local_ordinal_type)1,val,lcol,false);
		PV.sumIntoValues (row, col,(local_ordinal_type)1,val);
		
	      }//neq
		
	    }//j
	    
	  }//i

	}//gp

      });//parallel_for

    }//c

#ifdef TUSAS_HAVE_CUDA
  cudaFree(d_pf);
  free(h_pf);
#endif

    //cn we need to do a similar comm here...
    P->fillComplete();

    //P->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
  
    {
      Teuchos::TimeMonitor ImportTimer(*ts_time_import);  
      P_->doExport(*P, *exporter_, Tpetra::ADD);
    }

    P_->fillComplete();

    //P_->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
    //exit(0);

  }//outArgs.get_W_prec() 



  if(nonnull(outArgs.get_W_prec() ) && NULL != dirichletfunc_){
    Teuchos::TimeMonitor PrecFillTimer(*ts_time_precfill);
    P->resumeFill();//this is overlap, P_ is owned

    // local nodeset ids are on overlap
   
#ifdef TUSAS_RUN_ON_CPU
    auto PV = P->getLocalMatrix();
    std::vector<Mesh::mesh_lint_t> node_num_map(mesh_->get_node_num_map());
    std::map<int,DBCFUNC>::iterator it;
    for( int k = 0; k < numeqs_; k++ ){
      for(it = (*dirichletfunc_)[k].begin();it != (*dirichletfunc_)[k].end(); ++it){
	const int ns_id = it->first;
	const int num_node_ns = mesh_->get_node_set(ns_id).size();
  
	size_t ns_size = (mesh_->get_node_set(ns_id)).size();
	Kokkos::View <int*> node_set_view("nsv",ns_size);
	for (size_t i = 0; i < ns_size; ++i) {
	  node_set_view(i) = (mesh_->get_node_set(ns_id))[i];
        }
	
 	//for ( int j = 0; j < num_node_ns; j++ ){
	Kokkos::parallel_for(num_node_ns,KOKKOS_LAMBDA(const size_t j){

	  const int lid_overlap = node_set_view(j);
	  //const global_ordinal_type gid_overlap = x_overlap_map_->getGlobalElement(lid_overlap);
	  //const local_ordinal_type lrow = x_owned_map_->getLocalElement(gid_overlap);
	  const local_ordinal_type lrow = (local_ordinal_type)lid_overlap;
	 
	  size_t ncol = 0;
	  const local_ordinal_type row = numeqs*lrow + k;
	  
	  auto RV = PV.row(row);
	  //const Kokkos::SparseRowView<Kokkos::CrsMatrix> RV = PV.row(row);
	  ncol = RV.length;
	  
	  scalar_type * vals = new scalar_type[ncol];
	  local_ordinal_type * inds = new local_ordinal_type[ncol];
	  
	  for(int i = 0; i<(int)ncol; i++){
	    inds[i] = RV.colidx(i);
	    vals[i] = 0.0;
	    ( inds[i] == row ) ? ( vals[i] = 1.0 ) : ( vals[i] = 0.0 );
	    //std::cout<<row<<"   "<<inds[i]<<"  "<<vals[i]<<std::endl;
	    RV.value(i) = vals[i];
	  }
	  
	  //P_->replaceLocalValues(row, ncol, vals, inds );	    
	  //PV.replaceValues(row, inds, ncol, vals );
	  
	  delete[] vals;
	  delete[] inds;
	
	  });//parallel_for
	  //}//j

      }//it
    }//k
#else
#endif
    P->fillComplete();
    //P->fillComplete();
    //P->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
//     exit(0);


    P_->resumeFill();
    {
      Teuchos::TimeMonitor ImportTimer(*ts_time_import);  
      P_->doExport(*P, *exporter_, Tpetra::REPLACE);
    }

    P_->fillComplete();
    //P_->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );


  }//outArgs.get_W_prec() && dirichletfunc_

  if( nonnull(outArgs.get_W_prec() )){

    MueLu::ReuseTpetraPreconditioner( P_, *prec_  );

  }//outArgs.get_W_prec() 

  return;
}

//====================================================================

template<class scalar_type>
void ModelEvaluatorTPETRA<scalar_type>::init_nox()
{
  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
  int mypid = comm_->getRank() ;
  if( 0 == mypid )
    std::cout<<std::endl<<"init_nox() started."<<std::endl<<std::endl;

  nnewt_=0;

  ::Stratimikos::DefaultLinearSolverBuilder builder;
  Teuchos::RCP<Teuchos::ParameterList> lsparams =
    Teuchos::rcp(new Teuchos::ParameterList(paramList.sublist(TusaslsNameString)));
   
  //::Stratimikos::enableMueLu<local_ordinal_type,global_ordinal_type, node_type>(builder); 
  using Base = Thyra::PreconditionerFactoryBase<scalar_type>;
  using Impl = Thyra::MueLuPreconditionerFactory<scalar_type,local_ordinal_type,global_ordinal_type, node_type>;
  builder.setPreconditioningStrategyFactory(Teuchos::abstractFactoryStd<Base,Impl>(), "MueLu");

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

  // Create the initial guess
  Teuchos::RCP< ::Thyra::VectorBase<double> >
    initial_guess = this->getNominalValues().get_x()->clone_v();

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
      Teuchos::rcp(new NOX::Thyra::Group(*initial_guess, thyraModel, jfnkOp, lowsFactory, precOp, Teuchos::null, Teuchos::null, Teuchos::null));
  }
  else {
    nox_group =
      Teuchos::rcp(new NOX::Thyra::Group(*initial_guess, thyraModel, jfnkOp, lowsFactory, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null));
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
    Teuchos::rcp(new NOX::StatusTest::NormF(*nox_group.get(), relrestol));

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
  if( 0 == mypid )
    nl_params->print(std::cout);
  Teuchos::ParameterList& nlPrintParams = nl_params->sublist("Printing");
  nlPrintParams.set("Output Information",
		  NOX::Utils::OuterIteration  +
		  //                      NOX::Utils::OuterIterationStatusTest +
		  NOX::Utils::InnerIteration +
		    NOX::Utils::Details //+
		    //NOX::Utils::LinearSolverDetails
		    );
  // Create the solver
  solver_ =  NOX::Solver::buildSolver(nox_group, combo, nl_params);

  if( 0 == mypid )
    std::cout<<std::endl<<"init_nox() completed."<<std::endl<<std::endl;
}

template<class Scalar>
void ModelEvaluatorTPETRA<Scalar>::
set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory)
{
  W_factory_ = W_factory;
}

template<class Scalar>
Teuchos::RCP< ::Thyra::PreconditionerBase<Scalar> >
ModelEvaluatorTPETRA<Scalar>::create_W_prec() const
{

  //cn prec_ is MueLu::TpetraOperator
  //cn which inherits from Tpetra::Operator
  //cn need to cast prec_ to a Tpetra::Operator

  Teuchos::RCP<Tpetra::Operator<scalar_type,local_ordinal_type, global_ordinal_type, node_type> > Tprec =
    Teuchos::rcp_dynamic_cast<Tpetra::Operator<scalar_type,local_ordinal_type, global_ordinal_type, node_type> >(prec_,true);

  const Teuchos::RCP<Thyra::LinearOpBase< scalar_type > > P_op = 
    Thyra::tpetraLinearOp<scalar_type,local_ordinal_type, global_ordinal_type, node_type>(f_space_,x_space_,Tprec);

  Teuchos::RCP<Thyra::DefaultPreconditioner<Scalar> > prec =
    Teuchos::rcp(new Thyra::DefaultPreconditioner<Scalar>(Teuchos::null,P_op));

  //prec->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
  //exit(0);

  return prec;
}

template<class scalar_type>
Thyra::ModelEvaluatorBase::OutArgs<scalar_type>
ModelEvaluatorTPETRA<scalar_type>::createOutArgsImpl() const
{
  return prototypeOutArgs_;
}

template<class scalar_type>
void ModelEvaluatorTPETRA<scalar_type>::advance()
{
  Teuchos::RCP< Thyra::VectorBase< double > > guess = Thyra::createVector(u_old_,x_space_);
  NOX::Thyra::Vector thyraguess(*guess);//by sending the dereferenced pointer, we instigate a copy rather than a view
  solver_->reset(thyraguess);

  {
    Teuchos::TimeMonitor NSolveTimer(*ts_time_nsolve);

    NOX::StatusTest::StatusType solvStatus = solver_->solve();
    if( !(NOX::StatusTest::Converged == solvStatus)) {
      std::cout<<" NOX solver failed to converge. Status = "<<solvStatus<<std::endl<<std::endl;
      if(200 == paramList.get<int> (TusasnoxmaxiterNameString)) exit(0);
    }
  }
  nnewt_ += solver_->getNumIterations();

  const Thyra::VectorBase<double> * sol = 
    &(dynamic_cast<const NOX::Thyra::Vector&>(
					      solver_->getSolutionGroup().getX()
					      ).getThyraVector()
      );
  Thyra::ConstDetachedSpmdVectorView<double> x_vec(sol->col(0));

  Teuchos::ArrayRCP<scalar_type> uv = u_old_->get1dViewNonConst();
  const size_t localLength = num_owned_nodes_;


  for (int nn=0; nn < localLength; nn++) {//cn figure out a better way here...

    for( int k = 0; k < numeqs_; k++ ){
      uv[numeqs_*nn+k]=x_vec[numeqs_*nn+k];
    }
  }

  time_ +=dt_;

  for(boost::ptr_vector<error_estimator>::iterator it = Error_est.begin();it != Error_est.end();++it){
    //it->test_lapack();

    it->estimate_gradient(u_old_);
    it->estimate_error(u_old_);

  }
   
  postprocess();

}

template<class scalar_type>
  void ModelEvaluatorTPETRA<scalar_type>::initialize()
{
  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
  if( 0 == comm_->getRank()) std::cout<<std::endl<<"inititialize started"<<std::endl<<std::endl;
  bool dorestart = paramList.get<bool> (TusasrestartNameString);
  if (!dorestart){ 
    init(u_old_); 
#if 0 
    *u_old_old_ = *u_old_;
#endif  

    int mypid = comm_->getRank();
    int numproc = comm_->getSize();
    
    if( 1 == numproc ){//cn for now
      //if( 0 == mypid ){
      const char *outfilename = "results.e";
      ex_id_ = mesh_->create_exodus(outfilename);
      
    }
    else{
      //std::string decompPath="decomp/";
      std::string decompPath=paramList.get<std::string> (TusasoutputpathNameString);
      //std::string pfile = decompPath+std::to_string(mypid+1)+"/results.e."+std::to_string(numproc)+"."+std::to_string(mypid);
      
      std::string mypidstring(getmypidstring(mypid,numproc));

      std::string pfile = decompPath+"/results.e."+std::to_string(numproc)+"."+mypidstring;
      ex_id_ = mesh_->create_exodus(pfile.c_str());
    }  
    for( int k = 0; k < numeqs_; k++ ){
      mesh_->add_nodal_field((*varnames_)[k]);
    }
    
    output_step_ = 1;
    write_exodus();
  }


  else{
    restart(u_old_);//,u_old_old_);
//     if(1==comm_->MyPID())
//       std::cout<<"Restart unavailable"<<std::endl<<std::endl;
//     exit(0);
    for( int k = 0; k < numeqs_; k++ ){
      mesh_->add_nodal_field((*varnames_)[k]);
    }
  }
   
  if( 0 == comm_->getRank()) std::cout<<std::endl<<"initialize finished"<<std::endl<<std::endl;
}
template<class scalar_type>
void ModelEvaluatorTPETRA<scalar_type>::init(Teuchos::RCP<vector_type> u)
{
  //ArrayRCP<scalar_type> uv = u->get1dViewNonConst();

  auto u_view = u->getLocalView<Kokkos::DefaultExecutionSpace>();
  auto u_1d = Kokkos::subview (u_view, Kokkos::ALL (), 0);
  
  const size_t localLength = num_owned_nodes_;
  for( int k = 0; k < numeqs_; k++ ){
    //#pragma omp parallel for
    for (size_t nn=0; nn < localLength; nn++) {

      const global_ordinal_type gid_node = x_owned_map_->getGlobalElement(nn*numeqs_); 
      const local_ordinal_type lid_overlap = (x_overlap_map_->getLocalElement(gid_node))/numeqs_;

      const double x = mesh_->get_x(lid_overlap);
      const double y = mesh_->get_y(lid_overlap);
      const double z = mesh_->get_z(lid_overlap);
#ifdef TUSAS_RUN_ON_CPU
      u_1d[numeqs_*nn+k] = (*initfunc_)[k](x,y,z,k);
#else
      u_1d[numeqs_*nn+k] = tusastpetra::init_heat_test_(x,y,z,k);
#endif
    }


  }
  //exit(0);
}

template<class scalar_type>
void ModelEvaluatorTPETRA<scalar_type>::set_test_case()
{
  paramfunc_ = NULL;

  if("heat" == paramList.get<std::string> (TusastestNameString)){
    // numeqs_ number of variables(equations) 
    numeqs_ = 1;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    //(*residualfunc_)[0] = &tusastpetra::residual_heat_test_;
    (*residualfunc_)[0] = tpetra::residual_heat_test_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::prec_heat_test_dp_;
    
    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    
    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::init_heat_test_;
    
    
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
    
    //  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
    //               [numeq][nodeset id]
    //  [variable index][nodeset index]
    (*dirichletfunc_)[0][0] = &dbc_zero_;							 
    (*dirichletfunc_)[0][1] = &dbc_zero_;						 
    (*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[0][3] = &dbc_zero_;

    paramfunc_ = tpetra::param_;

    post_proc.push_back(new post_process(Comm,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &tpetra::postproc_;

  }else if("NLheatIMR" == paramList.get<std::string> (TusastestNameString)){
    // numeqs_ number of variables(equations) 
    numeqs_ = 1;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    //(*residualfunc_)[0] = &tusastpetra::residual_heat_test_;
    //kodiak (*residualfunc_)[0] = tpetra::residual_nlheatimr_test_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::prec_heat_test_dp_;
    
    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    
    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::init_heat_test_;
    
    
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
    
    //  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
    //               [numeq][nodeset id]
    //  [variable index][nodeset index]
    (*dirichletfunc_)[0][0] = &dbc_zero_;							 
    (*dirichletfunc_)[0][1] = &dbc_zero_;						 
    (*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[0][3] = &dbc_zero_;

    paramfunc_ = tpetra::param_;

    post_proc.push_back(new post_process(Comm,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &tpetra::postproc_;

  }else if("NLheatCN" == paramList.get<std::string> (TusastestNameString)){
    // numeqs_ number of variables(equations) 
    numeqs_ = 1;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    //(*residualfunc_)[0] = &tusastpetra::residual_heat_test_;
    //kodiak (*residualfunc_)[0] = tpetra::residual_nlheatcn_test_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::prec_nlheatcn_test_dp_;
    
    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    
    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::init_heat_test_;
    
    
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
    
    //  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
    //               [numeq][nodeset id]
    //  [variable index][nodeset index]
    (*dirichletfunc_)[0][0] = &dbc_zero_;							 
    (*dirichletfunc_)[0][1] = &dbc_zero_;						 
    (*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[0][3] = &dbc_zero_;

    paramfunc_ = tpetra::param_;

    post_proc.push_back(new post_process(Comm,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &tpetra::postproc_;


  }else if("heat2" == paramList.get<std::string> (TusastestNameString)){
    
    numeqs_ = 2;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);

    //kodiak (*residualfunc_)[0] = tpetra::residual_heat_test_dp_;
    //kodiak (*residualfunc_)[1] = tpetra::residual_heat_test_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::prec_heat_test_dp_;
    (*preconfunc_)[1] = tpetra::prec_heat_test_dp_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);

    (*initfunc_)[0] = &tpetra::init_heat_test_;
    (*initfunc_)[1] = &tpetra::init_heat_test_;

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

//     neumannfunc_ = NULL;

  }else if("cummins" == paramList.get<std::string> (TusastestNameString)){
    
    numeqs_ = 2;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
//     (*residualfunc_)[0] = &cummins::residual_heat_;
//     (*residualfunc_)[1] = &cummins::residual_phase_;
    //kodiak (*residualfunc_)[0] = tpetra::residual_heat_test_dp_;
    //kodiak (*residualfunc_)[1] = tpetra::residual_heat_test_dp_;

//     preconfunc_ = new std::vector<PREFUNC>(numeqs_);
//     (*preconfunc_)[0] = &cummins::prec_heat_;
//     (*preconfunc_)[1] = &cummins::prec_phase_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    //(*initfunc_)[0] = &cummins::init_heat_;
    //(*initfunc_)[1] = &cummins::init_phase_;
    (*initfunc_)[0] = &tpetra::init_heat_test_;
    (*initfunc_)[1] = &tpetra::init_heat_test_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    (*varnames_)[1] = "phi";

    dirichletfunc_ = NULL;

//     neumannfunc_ = NULL;

    paramfunc_ = cummins::param_;

  }else if("farzadi" == paramList.get<std::string> (TusastestNameString)){
    //farzadi test

    numeqs_ = 2;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::farzadi3d::init_conc_farzadi_;
    (*initfunc_)[1] = &tpetra::farzadi3d::init_phase_farzadi_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    //kodiak (*residualfunc_)[0] = tpetra::farzadi3d::residual_conc_farzadi_dp_;
    //kodiak (*residualfunc_)[1] = tpetra::farzadi3d::residual_phase_farzadi_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::farzadi3d::prec_conc_farzadi_dp_;
    (*preconfunc_)[1] = tpetra::farzadi3d::prec_phase_farzadi_dp_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    (*varnames_)[1] = "phi";

    dirichletfunc_ = NULL;

    paramfunc_ = tpetra::farzadi3d::param_;
    //paramfunc_ = farzadi::param_;

  }else if("farzadi_test" == paramList.get<std::string> (TusastestNameString)){
    //farzadi test

    numeqs_ = 2;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::farzadi3d::init_conc_farzadi_;
    (*initfunc_)[1] = &tpetra::farzadi3d::init_phase_farzadi_test_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    //kodiak (*residualfunc_)[0] = tpetra::farzadi3d::residual_conc_farzadi_dp_;
    //kodiak (*residualfunc_)[1] = tpetra::farzadi3d::residual_phase_farzadi_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::farzadi3d::prec_conc_farzadi_dp_;
    (*preconfunc_)[1] = tpetra::farzadi3d::prec_phase_farzadi_dp_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    (*varnames_)[1] = "phi";

    dirichletfunc_ = NULL;

    post_proc.push_back(new post_process(Comm,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &tpetra::farzadi3d::postproc_c_;
    post_proc.push_back(new post_process(Comm,mesh_,(int)1));
    post_proc[1].postprocfunc_ = &tpetra::farzadi3d::postproc_t_;

    paramfunc_ = tpetra::farzadi3d::param_;

  }else if("pfhub3" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 2;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::pfhub3::init_heat_pfhub3_;
    (*initfunc_)[1] = &tpetra::pfhub3::init_phase_pfhub3_;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    //kodiak (*residualfunc_)[0] = tpetra::pfhub3::residual_heat_pfhub3_dp_;
    //kodiak (*residualfunc_)[1] = tpetra::pfhub3::residual_phase_pfhub3_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::pfhub3::prec_heat_pfhub3_dp_;
    (*preconfunc_)[1] = tpetra::pfhub3::prec_phase_pfhub3_dp_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    (*varnames_)[1] = "phi";

    dirichletfunc_ = NULL;

    //paramfunc_ = tpetra::pfhub3::param_;

  }else if("pfhub2kks" == paramList.get<std::string> (TusastestNameString)){

    Teuchos::ParameterList *problemList;
    problemList = &paramList.sublist ( "ProblemParams", false );

    int numeta = problemList->get<int>("N");

    numeqs_ = numeta+1;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    //kodiak (*residualfunc_)[0] = tpetra::pfhub2::residual_c_kks_dp_;
    //kodiak (*residualfunc_)[1] = tpetra::pfhub2::residual_eta_kks_dp_;
#if 0
    if( 4 == numeta){
      (*residualfunc_)[2] = &pfhub2::residual_eta_kks_;
      (*residualfunc_)[3] = &pfhub2::residual_eta_kks_;
      (*residualfunc_)[4] = &pfhub2::residual_eta_kks_;
    }
#endif
    preconfunc_ = NULL;
#if 0
    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &pfhub2::prec_c_;
    (*preconfunc_)[1] = &pfhub2::prec_eta_;
    if( 4 == numeta){
      (*preconfunc_)[2] = &pfhub2::prec_eta_;
      (*preconfunc_)[3] = &pfhub2::prec_eta_;
      (*preconfunc_)[4] = &pfhub2::prec_eta_;
    }
#endif

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

    //neumannfunc_ = NULL;

    paramfunc_ = tpetra::pfhub2::param_;

    post_proc.push_back(new post_process(Comm,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &pfhub2::postproc_c_b_;

    post_proc.push_back(new post_process(Comm,mesh_,(int)1));
    post_proc[1].postprocfunc_ = &tpetra::pfhub2::postproc_c_a_;

  } else {
    auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
    if( 0 == comm_->getRank() ){
      std::cout<<std::endl<<std::endl<<"Test case: "<<paramList.get<std::string> (TusastestNameString)
	       <<" not found. (void ModelEvaluatorTPETRA<scalar_type>::set_test_case())" <<std::endl<<std::endl<<std::endl;
    }
    exit(0);
  }

  if(numeqs_ > TUSAS_MAX_NUMEQS){
    auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
    if( 0 == comm_->getRank() ){
      std::cout<<std::endl<<std::endl<<"numeqs_ > TUSAS_MAX_NUMEQS; 1. increase TUSAS_MAX_NUMEQS to "
	       <<numeqs_<<" ; 2. adjust TUSAS_MAX_NUMEQS_X_BASIS_NODES_PER_ELEM appropriately and recompile." <<std::endl<<std::endl<<std::endl;
    }
    exit(0);
  } 

  //set the params in the test case now...
  Teuchos::ParameterList *problemList;
  problemList = &paramList.sublist ( "ProblemParams", false );
  
  if ( NULL != paramfunc_ ){
    
    paramfunc_(problemList);
  }
  

}

template<class scalar_type>
void ModelEvaluatorTPETRA<scalar_type>::write_exodus()
//void ModelEvaluatorNEMESIS<scalar_type>::write_exodus(const int output_step)
{
  update_mesh_data();

  //not sre what the bug is here...
  Teuchos::TimeMonitor IOWriteTimer(*ts_time_iowrite);
  mesh_->write_exodus(ex_id_,output_step_,time_);
  output_step_++;
}

template<class scalar_type>
int ModelEvaluatorTPETRA<scalar_type>:: update_mesh_data()
{
  //std::cout<<"update_mesh_data()"<<std::endl;

  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
  //std::vector<int> node_num_map(mesh_->get_node_num_map());

  //cn 8-28-18 we need an overlap map with mpi, since shared nodes live
  // on the decomposed mesh----fix this later
  Teuchos::RCP<vector_type> temp;

  if( 1 == comm_->getSize() ){
    temp = Teuchos::rcp(new vector_type(*u_old_));
  }
  else{
    temp = Teuchos::rcp(new vector_type(x_overlap_map_));//cn might be better to have u_old_ live on overlap map
    temp->doImport(*u_old_, *importer_, Tpetra::INSERT);
  }


  //cn 8-28-18 we need an overlap map with mpi, since shared nodes live
  // on the decomposed mesh----fix this later
  int num_nodes = num_overlap_nodes_;
  std::vector<std::vector<double>> output(numeqs_, std::vector<double>(num_nodes));

  const Teuchos::ArrayRCP<scalar_type> uv = temp->get1dViewNonConst();
  //const size_t localLength = num_owned_nodes_;

  for( int k = 0; k < numeqs_; k++ ){
    //#pragma omp parallel for
    for (int nn=0; nn < num_nodes; nn++) {
      //output[k][nn]=(*temp)[numeqs_*nn+k];
      output[k][nn]=uv[numeqs_*nn+k];
      //std::cout<<uv[numeqs_*nn+k]<<std::endl;
    }
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
    itp->update_mesh_data();
    itp->update_scalar_data(time_);
  }

  Elem_col->update_mesh_data();

  return err;
}
template<class scalar_type>
void ModelEvaluatorTPETRA<scalar_type>::finalize()
{
  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
  int mypid = comm_->getRank();
  int numproc = comm_->getSize();

  bool dorestart = paramList.get<bool> (TusasrestartNameString);

  write_exodus();

  //std::cout<<(solver_->getList()).sublist("Direction").sublist("Newton").sublist("Linear Solver")<<std::endl;
  int ngmres = 0;

  if ( (solver_->getList()).sublist("Direction").sublist("Newton").sublist("Linear Solver")
       .sublist("Output").getEntryPtr("Cumulative Iteration Count") != NULL)
    ngmres = ((solver_->getList()).sublist("Direction").sublist("Newton").sublist("Linear Solver")
	      .sublist("Output").getEntry("Cumulative Iteration Count")).getValue(&ngmres);

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

#if 0
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
#endif

  delete residualfunc_;
  delete preconfunc_;
  delete initfunc_;
  delete varnames_;
  if( NULL != dirichletfunc_) delete dirichletfunc_;
#if 0
#ifdef PERIODIC_BC
#else
  if( NULL != periodicbc_) delete periodicbc_;
#endif
  //if( NULL != postprocfunc_) delete postprocfunc_;
#endif
}


template<class scalar_type>
  void ModelEvaluatorTPETRA<scalar_type>::restart(Teuchos::RCP<vector_type> u)//,Teuchos::RCP<vector_type> u_old)
{
  //cn we need to get u_old_ and u_old_old_
  //and start_time and start_step and modify time_
  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
  int mypid = comm_->getRank();
  int numproc = comm_->getSize();
  if( 0 == mypid )
    std::cout<<std::endl<<"Entering restart: PID "<<mypid<<" NumProcs "<<numproc<<std::endl<<std::endl;
  
  if( 1 == numproc ){//cn for now
    //if( 0 == mypid ){
    const char *outfilename = "results.e";
    ex_id_ = mesh_->open_exodus(outfilename);

    std::cout<<"  Opening file for restart; ex_id_ = "<<ex_id_<<" filename = "<<outfilename<<std::endl;
    
  }
  else{
    std::string decompPath="decomp/";
    //std::string pfile = decompPath+std::to_string(mypid+1)+"/results.e."+std::to_string(numproc)+"."+std::to_string(mypid);
    
    std::string mypidstring(getmypidstring(mypid,numproc));

    std::string pfile = decompPath+"results.e."+std::to_string(numproc)+"."+mypidstring;
    ex_id_ = mesh_->open_exodus(pfile.c_str());
    
    std::cout<<"  Opening file for restart; ex_id_ = "<<ex_id_<<" filename = "<<pfile<<std::endl;

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

  const double dt = paramList.get<double> (TusasdtNameString);
  const int numSteps = paramList.get<int> (TusasntNameString);

  if( step > numSteps || time >numSteps*dt ){
    if( 0 == mypid ){
      std::cout<<"  Error reading restart last time = "<<time<<std::endl;
      std::cout<<"    is greater than    "<<numSteps*dt<<std::endl<<std::endl<<std::endl;
      exit(0);
    }
  }

  std::vector<std::vector<double>> inputu(numeqs_,std::vector<double>(num_overlap_nodes_));

  for( int k = 0; k < numeqs_; k++ ){
    error = mesh_->read_nodal_data_exodus(ex_id_,step,(*varnames_)[k],&inputu[k][0]);

    if( 0 > error ) {
      std::cout<<"Error reading u at step "<<step<<std::endl;
      exit(0);
    }
  }

  //cn for now just put current values into old values, 
  //cn ie just start with an initial condition

  //cn lets not worry about two different time steps for normal simulations

  Teuchos::RCP< vector_type> u_temp = Teuchos::rcp(new vector_type(x_overlap_map_));
  //Teuchos::RCP< Epetra_Vector> u_old_temp = Teuchos::rcp(new Epetra_Vector(*x_overlap_map_));

  auto u_view = u_temp->getLocalView<Kokkos::DefaultExecutionSpace>();
  auto u_1d = Kokkos::subview (u_view, Kokkos::ALL (), 0);
  for( int k = 0; k < numeqs_; k++ ){
    for (int nn=0; nn < num_overlap_nodes_; nn++) {
      u_1d[numeqs_*nn+k] = inputu[k][nn];
      //(*u_old_temp)[numeqs_*nn+k] = inputu[k][nn];
      //std::cout<<u_1d[numeqs_*nn+k]<<"   "<<inputu[k][nn]<<"  "<<k<<"  "<<nn<<std::endl;
    }
  }

  u->doExport(*u_temp,*exporter_, Tpetra::INSERT);
  //u_old->doExport(*u_temp,*exporter_, Tpetra::INSERT);

  this->start_time = time;
  int ntstep = (int)(time/dt_);
  //this->start_step = step-1;//this corresponds to the output frequency, not the actual timestep
  this->start_step = ntstep;
  time_=time;
  output_step_ = step+1;
  //   u->Print(std::cout);
  //   exit(0);
  if( 0 == mypid ){
    std::cout<<"Restarting at time = "<<time<<" and step = "<<step<<std::endl<<std::endl;
    std::cout<<"Exiting restart"<<std::endl<<std::endl;
  }
  //exit(0);
}

template<class Scalar>
void ModelEvaluatorTPETRA<Scalar>::postprocess()
{
  if(0 == post_proc.size() ) return;

  int numee = Error_est.size();
  //ordering is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz

  const int dim = 3;

  std::vector<double> uu(numeqs_);
  std::vector<double> ug(dim*numee);

  auto uview = u_old_->get1dView();

  for (int nn=0; nn < num_owned_nodes_; nn++) {

    for( int k = 0; k < numeqs_; k++ ){
      uu[k] = uview[numeqs_*nn+k];
    }

    for( int k = 0; k < numee; k++ ){
      ug[k*dim] = (*(Error_est[k].gradx_))[nn];
      ug[k*dim+1] = (*(Error_est[k].grady_))[nn];
      ug[k*dim+2] = (*(Error_est[k].gradz_))[nn];
    }

    boost::ptr_vector<post_process>::iterator itp;
    for(itp = post_proc.begin();itp != post_proc.end();++itp){
      itp->process(nn,&uu[0],&ug[0],time_);
    }

  }//nn
}

template<class Scalar>
void ModelEvaluatorTPETRA<Scalar>::init_P_()
{
    P_->setAllToScalar((scalar_type)-1.0); 

    auto PV = P_->getLocalMatrix();

    const int numrows = PV.numRows();
    for ( int i; i < numrows; i++ ){
      local_ordinal_type row = i;
      auto RV = PV.row(row);
      RV.value(row) = 27.;
    }

    P_->fillComplete();
}
#endif
