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
#include <Tpetra_replaceDiagonalCrsMatrix_decl.hpp>

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

//#define TUSASMUELU
//#ifndef TUSASMUELU
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include "Thyra_MueLuPreconditionerFactory.hpp"
//#endif

//#include <Stratimikos_MueLuHelpers.hpp>

#include "NOX_Thyra_MatrixFreeJacobianOperator.hpp"
#include "NOX_MatrixFree_ModelEvaluatorDecorator.hpp"

#include <algorithm>

#include <string>

#include "function_def.hpp"
#include "ParamNames.h"

#ifdef TUSAS_HAVE_CUDA
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif
#if 0  
typedef double (*RESFUNC)(GPUBasisLHex * basis, 
			  const int &i, 
			  const double &dt_, 
			  const double &dtold_, 
			  const double &t_theta_, 
			  const double &t_theta2_, 
			  const double &time,
			  const int &eqn_id);
//TUSAS_DEVICE
__device__
RESFUNC *flist;
#endif

#define TUSAS_RUN_ON_CPU

// IMPORTANT!!! this macro should be set to TUSAS_MAX_NUMEQS * BASIS_NODES_PER_ELEM
#define TUSAS_MAX_NUMEQS_X_BASIS_NODES_PER_ELEM 32

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
  dtold_ = dt_;
  t_theta_ = paramList.get<double> (TusasthetaNameString);
  t_theta2_ = 0.;
  numsteps_ = 0;
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
  u_old_old_ = Teuchos::rcp(new vector_type(x_owned_map_));
  u_old_old_->putScalar(Teuchos::ScalarTraits<scalar_type>::zero());
  u_new_ = Teuchos::rcp(new vector_type(x_owned_map_));
  u_new_->putScalar(Teuchos::ScalarTraits<scalar_type>::zero());
  pred_temp_ = Teuchos::rcp(new vector_type(x_owned_map_));
  pred_temp_->putScalar(Teuchos::ScalarTraits<scalar_type>::zero());

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

  //#ifndef TUSASMUELU
  bool precon = paramList.get<bool> (TusaspreconNameString);
  if(precon){
    // Initialize the graph for W CrsMatrix object
    W_graph_ = createGraph();
    W_overlap_graph_ = createOverlapGraph();
    P = rcp(new matrix_type(W_overlap_graph_));
    P_ = rcp(new matrix_type(W_graph_));
    P->setAllToScalar((scalar_type)1.0); 
    P->fillComplete();
    //P->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
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

    if( 0 == comm_->getRank() ){
      std::cout <<" MueLu preconditioner created"<<std::endl<<std::endl;
    }
  }
  //#endif

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
  ts_time_temperr= Teuchos::TimeMonitor::getNewTimer("Tusas: Total Temporal Error Est Time");
  ts_time_predsolve= Teuchos::TimeMonitor::getNewTimer("Tusas: Total Predictor Solve Time");
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
//   auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
//   int mypid = comm_->getRank() ;
//   if( 0 == mypid )
//     std::cout<<std::endl<<"createOverlapGraph() started."<<std::endl<<std::endl;

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

//   if( 0 == mypid )
//     std::cout<<std::endl<<"createOverlapGraph() ended."<<std::endl<<std::endl;
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

//   Teuchos::ParameterList plist(paramList);
//   Teuchos::ParameterList *problemList = &plist.sublist ("ProblemParams", false);
  
//   for( int k = 0; k <  paramfunc_.size(); k++ ){
//     paramfunc_[k](problemList);
//   }

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
  Teuchos::RCP<vector_type > uoldold = Teuchos::rcp(new vector_type(x_overlap_map_));
  {
    Teuchos::TimeMonitor ImportTimer(*ts_time_import);
    u->doImport(*x_vec,*importer_,Tpetra::INSERT);
    uold->doImport(*u_old_,*importer_,Tpetra::INSERT);
    uoldold->doImport(*u_old_old_,*importer_,Tpetra::INSERT);
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
#ifdef TUSAS3D	
  if(n_nodes_per_elem != 8) exit(0);
#else
  if(n_nodes_per_elem != 4) exit(0);
#endif
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
  const double dtold = dtold_; //cuda 8 lambdas dont capture private data
  const double t_theta = t_theta_; //cuda 8 lambdas dont capture private data
  const double t_theta2 = t_theta2_; //cuda 8 lambdas dont capture private data
  const double time = time_; //cuda 8 lambdas dont capture private data
  const int numeqs = numeqs_; //cuda 8 lambdas dont capture private data
  const int LTP_quadrature_order = paramList.get<int> (TusasltpquadordNameString);
  if (4 <  LTP_quadrature_order ){
      if( 0 == comm_->getRank() ){
	std::cout<<std::endl<<std::endl<<"4 <  LTP_quadrature_order" <<std::endl<<std::endl<<std::endl;
      }
      exit(0);
  }

  const std::string testname = paramList.get<std::string> (TusastestNameString);
  
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
    auto uoldold_view = uoldold->getLocalView<Kokkos::DefaultExecutionSpace>();
    
    auto f_view = f_overlap->getLocalView<Kokkos::DefaultExecutionSpace>();
        
    auto f_1d = Kokkos::subview (f_view, Kokkos::ALL (), 0);
    //Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> f_1d = Kokkos::subview (f_view, Kokkos::ALL (), 0);

    //using RandomAccess should give better memory performance on better than tesla gpus (guido is tesla and does not show performance increase)
    //this will utilize texture memory not available on tesla or earlier gpus
    Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> 
      uold_1dra = Kokkos::subview (uold_view, Kokkos::ALL (), 0);
    Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> 
      uoldold_1dra = Kokkos::subview (uoldold_view, Kokkos::ALL (), 0);

    RESFUNC * h_rf;
    h_rf = (RESFUNC*)malloc(numeqs_*sizeof(RESFUNC));

    int testcase = -99;

#ifdef TUSAS_HAVE_CUDA
    //RESFUNC * d_rf;
    //cudaMalloc((double**)&d_rf,numeqs_*sizeof(RESFUNC));


    //better to use testname.compare("heat") > 0
    if("heat" == testname){
      //cn this will need to be done for each equation
      //gpuErrchk(cudaMemcpyFromSymbol( &h_rf[0], tpetra::heat::residual_heat_test_dp_, sizeof(RESFUNC)));
      testcase = 0;
    }else if("NLheatIMR" == testname){
      //cudaMemcpyFromSymbol( &h_rf[0], tpetra::residual_nlheatimr_test_dp_, sizeof(RESFUNC));
      testcase = 1;
    }else if("NLheatCN" == testname){
      //cudaMemcpyFromSymbol( &h_rf[0], tpetra::residual_nlheatcn_test_dp_, sizeof(RESFUNC));
      testcase = 2;
    }else if("heat2" == testname){
      //cudaMemcpyFromSymbol( &h_rf[0], tpetra::heat::residual_heat_test_dp_, sizeof(RESFUNC));
      //cudaMemcpyFromSymbol( &h_rf[1], tpetra::heat::residual_heat_test_dp_, sizeof(RESFUNC));
      testcase = 3;
    }else if("farzadi" == testname){
      //cudaMemcpyFromSymbol( &h_rf[0], tpetra::farzadi3d::residual_conc_farzadi_dp_, sizeof(RESFUNC));
      //cudaMemcpyFromSymbol( &h_rf[1], tpetra::farzadi3d::residual_phase_farzadi_dp_, sizeof(RESFUNC));
      testcase = 4;
    }else if("farzadi_test" == testname){
      //cudaMemcpyFromSymbol( &h_rf[0], tpetra::farzadi3d::residual_conc_farzadi_dp_, sizeof(RESFUNC));
      //cudaMemcpyFromSymbol( &h_rf[1], tpetra::farzadi3d::residual_phase_farzadi_dp_, sizeof(RESFUNC));
      testcase = 5;
    }else if("pfhub3" == testname){
      //cudaMemcpyFromSymbol( &h_rf[0], tpetra::pfhub3::residual_heat_pfhub3_dp_, sizeof(RESFUNC));
      //cudaMemcpyFromSymbol( &h_rf[1], tpetra::pfhub3::residual_phase_pfhub3_dp_, sizeof(RESFUNC));
      testcase = 6;
    }else if("pfhub2kks" == testname){
      //cudaMemcpyFromSymbol( &h_rf[0], tpetra::pfhub2::residual_c_kks_dp_, sizeof(RESFUNC));
      //cudaMemcpyFromSymbol( &h_rf[1], tpetra::pfhub2::residual_eta_kks_dp_, sizeof(RESFUNC));
      testcase = 7;
    }else if("goldak" == testname){
      //cudaMemcpyFromSymbol( &h_rf[0], tpetra::goldak::residual_test_dp_, sizeof(RESFUNC));
      testcase = 8;

    } else {
      if( 0 == comm_->getRank() ){
	std::cout<<std::endl<<std::endl<<"Test case: "<<testname
		 <<" residual function not found. (void ModelEvaluatorTPETRA<Scalar>::evalModelImpl(...))" <<std::endl<<std::endl<<std::endl;
      }
      exit(0);
    }

    //gpuErrchk(cudaMemcpy(d_rf,h_rf,numeqs_*sizeof(RESFUNC),cudaMemcpyHostToDevice));

#else
    //it seems that evaluating the function via pointer ie h_rf[0] is way faster that evaluation via (*residualfunc_)[0]
    h_rf = &(*residualfunc_)[0];
#endif

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
//       Kokkos::parallel_for(numeqs,KOKKOS_LAMBDA(const int& neq){
// 			     flist = tpetra::heat::residual_heat_test_;
// 			   });

      Kokkos::parallel_for(num_elem,KOKKOS_LAMBDA(const int& ne){//this loop is fine for openmp re access to elem_map
			     //for(int ne =0; ne<num_elem; ne++){


        //new((RESFUNC*)rf) &tpetra::heat::residual_heat_test_dp_;
	RESFUNC rf[TUSAS_MAX_NUMEQS];
	if (testcase == 0){
	  rf[0] = tpetra::heat::residual_heat_test_;
	}else if (testcase == 4){
	  rf[0] = tpetra::farzadi3d::residual_conc_farzadi_;
	  rf[1] = tpetra::farzadi3d::residual_phase_farzadi_;
	}else if (testcase == 8){
	  rf[0] = tpetra::goldak::residual_test_;
	  //rf[1] = tpetra::goldak::residual_test_;
	}else{
	  exit(0);
	}

	const int elem = elem_map_1d(ne);
#ifdef TUSAS3D	
	GPUBasisLHex B[TUSAS_MAX_NUMEQS] = {GPUBasisLHex(LTP_quadrature_order), GPUBasisLHex(LTP_quadrature_order), GPUBasisLHex(LTP_quadrature_order), GPUBasisLHex(LTP_quadrature_order)};
	//GPUBasisLHex B[TUSAS_MAX_NUMEQS];// = {GPUBasisLHex(), GPUBasisLHex(), GPUBasisLHex(), GPUBasisLHex()};
#else
 	GPUBasisLQuad B[TUSAS_MAX_NUMEQS] = {GPUBasisLQuad(LTP_quadrature_order), GPUBasisLQuad(LTP_quadrature_order), GPUBasisLQuad(LTP_quadrature_order), GPUBasisLQuad(LTP_quadrature_order)};
#endif
			     //const int ngp = B[0].ngp;
	const int ngp = B[0].ngp();
	//printf("%d\n",ngp);

	double xx[BASIS_NODES_PER_ELEM];
	double yy[BASIS_NODES_PER_ELEM];
	double zz[BASIS_NODES_PER_ELEM];

	double uu[TUSAS_MAX_NUMEQS_X_BASIS_NODES_PER_ELEM];
	double uu_old[TUSAS_MAX_NUMEQS_X_BASIS_NODES_PER_ELEM];
	double uu_oldold[TUSAS_MAX_NUMEQS_X_BASIS_NODES_PER_ELEM];

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
	    uu_oldold[n_nodes_per_elem*neq+k] = uoldold_1dra(numeqs*nodeid+neq);
	  }//neq
	}//k

	for( int neq = 0; neq < numeqs; neq++ ){
	  //BGPU[neq]->computeElemData(&xx[0], &yy[0], &zz[0]);
	  B[neq].computeElemData(&xx[0], &yy[0], &zz[0]);
	}//neq
	
	for(int gp=0; gp < ngp; gp++) {//gp
	  double jacwt = 0.;
	  for( int neq = 0; neq < numeqs; neq++ ){
	    //we need a basis object that stores all equations here..
	    //BGPU[neq]->getBasis(gp, &xx[0], &yy[0], &zz[0], &uu[neq*n_nodes_per_elem], &uu_old[neq*n_nodes_per_elem],&uu_oldold[neq*n_nodes_per_elem]);
	    jacwt = B[neq].getBasis(gp, &xx[0], &yy[0], &zz[0], &uu[neq*n_nodes_per_elem], &uu_old[neq*n_nodes_per_elem],&uu_oldold[neq*n_nodes_per_elem]);
	  }//neq
	     
	  //printf("%f\n",jacwt);
	//const double jacwt = BGPU[0]->jac*BGPU[0]->wt;
			     //const double jacwt = B[0].jac*B[0].wt;

	  for (int i=0; i< n_nodes_per_elem; i++) {//i

	    //const int lrow = numeqs*meshc[elemrow+i];
	    const int lrow = numeqs*meshc_1dra(elemrow+i);

	    for( int neq = 0; neq < numeqs; neq++ ){
#ifdef TUSAS_HAVE_CUDA
	      //printf("%le\n",B[0].phi[0]);
	      //const double val = jacwt*((d_rf[neq])((&B[0]),i,dt,dtold,t_theta,t_theta2,time,neq));
	      const double val = jacwt*(rf[neq])(&B[0],i,dt,dtold,t_theta,t_theta2,time,neq);
#else
	      //const double val = BGPU->jac*BGPU->wt*(*residualfunc_)[0](BGPU,i,dt,1.,0.,0);
	      //const double val = BGPU->jac*BGPU->wt*(tpetra::heat::residual_heat_test_(BGPU,i,dt,1.,0.,0));//cn call directly
	      double val = jacwt*(h_rf[neq]((&B[0]),i,dt,dtold,t_theta,t_theta2,time,neq));
#endif
	      //cn this works because we are filling an overlap map and exporting to a node map below...
	      const int lid = lrow+neq;
	      f_1d[lid] += val;
	      //printf("%d %le %le %d\n",lid,jacwt*((h_rf[neq])(BGPU,i,dt,t_theta,time,neq)),f_1d[lid],c);
	    }//neq
	  }//i
	}//gp
      });//parallel_for
		     //};//ne

    }//c 

#ifdef TUSAS_HAVE_CUDA
    //cudaFree(d_rf);
  free(h_rf);
#endif
  //exit(0);
    {
      Teuchos::TimeMonitor ImportTimer(*ts_time_import);  
      f_vec->doExport(*f_overlap, *exporter_, Tpetra::ADD);
    }
  }//get_f

  if (nonnull(outArgs.get_f())) {
    if (NULL != neumannfunc_) {

      //we would need to utilize coloring and implement a view for:
      //mesh_->get_side_set_node_list(ss_id)
      //in order to get this working with kokkos on openmp, gpu
      //with a kokkos function for the nbc function
      
      //we also need to push the nbcfunc to gpus....


      
      
      //     if( (0==elem_type.compare("HEX8")) 
      // 	|| (0==elem_type.compare("HEX")) 
      // 	|| (0==elem_type.compare("hex8")) 
      // 	|| (0==elem_type.compare("hex"))  ){ // linear hex
      //     } 
      if(8 == n_nodes_per_elem) { // linear hex
      }
      else {
	exit(0);
      }   
      
      const Teuchos::RCP<vector_type> f_vec =
	ConverterT::getTpetraVector(outArgs.get_f());
      Teuchos::RCP<vector_type> f_overlap = Teuchos::rcp(new vector_type(x_overlap_map_));
      //we zero nodes on the nonowning proc here, so that we do not add in the values twice
      //this also does no communication
      {
	Teuchos::TimeMonitor ImportTimer(*ts_time_import);
	f_overlap->doImport(*f_vec,*importer_,Tpetra::ZERO);
      }
      
      auto uold_view = uold->getLocalView<Kokkos::DefaultExecutionSpace>();
      auto uoldold_view = uoldold->getLocalView<Kokkos::DefaultExecutionSpace>();
      auto f_view = f_overlap->getLocalView<Kokkos::DefaultExecutionSpace>();
      auto f_1d = Kokkos::subview (f_view, Kokkos::ALL (), 0);
      Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> 
	uold_1dra = Kokkos::subview (uold_view, Kokkos::ALL (), 0);
      Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> 
	uoldold_1dra = Kokkos::subview (uoldold_view, Kokkos::ALL (), 0);

      GPUBasisLQuad Bq = GPUBasisLQuad(LTP_quadrature_order);
      //GPUBasis * BGPU = &Bq;
      const int num_node_per_side = 4;
      
      //const int ngp = BGPU->ngp;
      const int ngp = Bq.ngp;
      
      std::map<int,NBCFUNC>::iterator it;
      
      for( int k = 0; k < numeqs_; k++ ){
	for(it = (*neumannfunc_)[k].begin();it != (*neumannfunc_)[k].end(); ++it){
	  const int ss_id = it->first;
	  
	  //loop over element faces--this will be the parallel loop eventually
	  //we would need toto know coloring on the sideset or switch to scattered mesh
	  for ( int j = 0; j < mesh_->get_side_set(ss_id).size(); j++ ){//loop over element faces--this will be the parallel loop
	    double xx[BASIS_NODES_PER_ELEM];
	    double yy[BASIS_NODES_PER_ELEM];
	    double zz[BASIS_NODES_PER_ELEM];
	    double uu[BASIS_NODES_PER_ELEM];
	    double uu_old[BASIS_NODES_PER_ELEM];
	    double uu_oldold[BASIS_NODES_PER_ELEM];
	    for ( int ll = 0; ll < num_node_per_side; ll++){//loop over nodes in each face
	      const int lid = mesh_->get_side_set_node_list(ss_id)[j*num_node_per_side+ll];
	      xx[ll] = x_1dra(lid);
	      yy[ll] = y_1dra(lid);
	      zz[ll] = z_1dra(lid);
	      uu[ll] = u_1dra(numeqs_*lid+k);
	      uu_old[ll] = uold_1dra(numeqs_*lid+k);
	      uu_oldold[ll] = uoldold_1dra(numeqs_*lid+k);
	    }//ll
	    
	    //BGPU->computeElemData(&xx[0], &yy[0], &zz[0]);
	    Bq.computeElemData(&xx[0], &yy[0], &zz[0]);
	    for ( int gp = 0; gp < ngp; gp++){
	      //BGPU->getBasis(gp, &xx[0], &yy[0], &zz[0], &uu[0], &uu_old[0], &uu_oldold[0]);//we can add uu_old, uu_oldold
	      Bq.getBasis(gp, &xx[0], &yy[0], &zz[0], &uu[0], &uu_old[0], &uu_oldold[0]);//we can add uu_old, uu_oldold
	      //const double jacwt = BGPU->jac*BGPU->wt;
	      const double jacwt = Bq.jac*Bq.wt;
	      for( int i = 0; i < num_node_per_side; i++ ){  
		
		const int lid = mesh_->get_side_set_node_list(ss_id)[j*num_node_per_side+i];
		const int row = numeqs_*lid + k;
		
		//const double val = -jacwt*(it->second)(BGPU,i,dt,dtold,t_theta,t_theta2,time);
		const double val = -jacwt*(it->second)(&Bq,i,dt,dtold,t_theta,t_theta2,time);
		
		f_1d[row] += val;
		
	      }//i
	    }//gp
	  }//j
	}//it
      }//k
      {
	Teuchos::TimeMonitor ImportTimer(*ts_time_import);  
	f_vec->doExport(*f_overlap, *exporter_, Tpetra::ADD);
      }
    }//neumann
  }//get_f

  if (nonnull(outArgs.get_f()) && NULL != dirichletfunc_){
    const Teuchos::RCP<vector_type> f_vec =
      ConverterT::getTpetraVector(outArgs.get_f());
    //std::vector<Mesh::mesh_lint_t> node_num_map(mesh_->get_node_num_map());
    std::map<int,DBCFUNC>::iterator it;
    
    Teuchos::RCP<vector_type> f_overlap = Teuchos::rcp(new vector_type(x_overlap_map_));
    {
      Teuchos::TimeMonitor ImportTimer(*ts_time_import);
      f_overlap->doImport(*f_vec,*importer_,Tpetra::INSERT);
    }

    //u is already imported to overlap_map here
    //auto u_view = u->getLocalView<Kokkos::DefaultExecutionSpace>();
    auto f_view = f_overlap->getLocalView<Kokkos::DefaultExecutionSpace>();
    
    
    //auto u_1d = Kokkos::subview (u_view, Kokkos::ALL (), 0);
    //Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> u_1dra = Kokkos::subview (u_view, Kokkos::ALL (), 0);
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
			       const double xx = x_1dra(lid);
			       const double yy = y_1dra(lid);
			       const double zz = z_1dra(lid);	
			       const double val1 = (it->second)(xx,yy,zz,time);
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

    int testcase = -99;
#ifdef TUSAS_HAVE_CUDA
    //PREFUNC * d_pf;
    //cudaMalloc((double**)&d_pf,numeqs_*sizeof(PREFUNC));


    if("heat" == paramList.get<std::string> (TusastestNameString)){
      //cn this will need to be done for each equation
      //cudaMemcpyFromSymbol( &h_pf[0], tpetra::heat::prec_heat_test_dp_, sizeof(PREFUNC));
      testcase = 0;
    }else if("NLheatCN" == paramList.get<std::string> (TusastestNameString)){
      //cudaMemcpyFromSymbol( &h_pf[0], tpetra::prec_nlheatcn_test_dp_, sizeof(PREFUNC));
      testcase = 1;
    }else if("NLheatIMR" == paramList.get<std::string> (TusastestNameString)){
      //cudaMemcpyFromSymbol( &h_pf[0], tpetra::prec_nlheatcn_test_dp_, sizeof(PREFUNC));
      testcase = 2;
    }else if("heat2" == paramList.get<std::string> (TusastestNameString)){
      //cudaMemcpyFromSymbol( &h_pf[0], tpetra::heat::prec_heat_test_dp_, sizeof(PREFUNC));
      //cudaMemcpyFromSymbol( &h_pf[1], tpetra::heat::prec_heat_test_dp_, sizeof(PREFUNC));
      testcase = 3;
    }else if("farzadi" == paramList.get<std::string> (TusastestNameString)){
      //cudaMemcpyFromSymbol( &h_pf[0], tpetra::farzadi3d::prec_conc_farzadi_dp_, sizeof(PREFUNC));
      //cudaMemcpyFromSymbol( &h_pf[1], tpetra::farzadi3d::prec_phase_farzadi_dp_, sizeof(PREFUNC));
      testcase = 4;
    }else if("farzadi_test" == paramList.get<std::string> (TusastestNameString)){
      //cudaMemcpyFromSymbol( &h_pf[0], tpetra::farzadi3d::prec_conc_farzadi_dp_, sizeof(PREFUNC));
      //cudaMemcpyFromSymbol( &h_pf[1], tpetra::farzadi3d::prec_phase_farzadi_dp_, sizeof(PREFUNC));
      testcase = 5;
    }else if("pfhub3" == paramList.get<std::string> (TusastestNameString)){
      //cudaMemcpyFromSymbol( &h_pf[0], tpetra::pfhub3::prec_heat_pfhub3_dp_, sizeof(PREFUNC));
      //cudaMemcpyFromSymbol( &h_pf[1], tpetra::pfhub3::prec_phase_pfhub3_dp_, sizeof(PREFUNC));
      testcase = 6;
    }else if("goldak" == paramList.get<std::string> (TusastestNameString)){
      //cudaMemcpyFromSymbol( &h_pf[0], tpetra::heat::prec_heat_test_dp_, sizeof(PREFUNC));
      testcase = 8;

    } else {
      if( 0 == comm_->getRank() ){
	std::cout<<std::endl<<std::endl<<"Test case: "<<paramList.get<std::string> (TusastestNameString)
		 <<" precon function not found. (void ModelEvaluatorTPETRA<Scalar>::evalModelImpl(...))" <<std::endl<<std::endl<<std::endl;
      }
      exit(0);
    }

    //cudaMemcpy(d_pf,h_pf,numeqs_*sizeof(PREFUNC),cudaMemcpyHostToDevice);

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

      //printf("%d\n",c);
      Kokkos::parallel_for(num_elem,KOKKOS_LAMBDA(const int& ne){

			     //printf("%d\n",ne);
	PREFUNC pf[TUSAS_MAX_NUMEQS];
	if (testcase == 0){
	  pf[0] = tpetra::heat::prec_heat_test_;
	}else if (testcase == 4){
	  pf[0] = tpetra::farzadi3d::prec_conc_farzadi_;
	  pf[1] = tpetra::farzadi3d::prec_phase_farzadi_;
	}else if (testcase == 8){
	  pf[0] = tpetra::goldak::prec_test_;
	}else{
	  exit(0);
	}
#ifdef TUSAS3D	
			     //GPUBasisLHex B[TUSAS_MAX_NUMEQS] = {GPUBasisLHex(LTP_quadrature_order), GPUBasisLHex(LTP_quadrature_order), GPUBasisLHex(LTP_quadrature_order), GPUBasisLHex(LTP_quadrature_order)};
        GPUBasisLHex B[TUSAS_MAX_NUMEQS] = {GPUBasisLHex(), GPUBasisLHex(), GPUBasisLHex(), GPUBasisLHex()};
#else
 	GPUBasisLQuad B[TUSAS_MAX_NUMEQS] = {GPUBasisLQuad(LTP_quadrature_order), GPUBasisLQuad(LTP_quadrature_order), GPUBasisLQuad(LTP_quadrature_order), GPUBasisLQuad(LTP_quadrature_order)};
#endif
	const int ngp = B[0].ngp();
	//printf("%d\n",ngp);

	//const int elem = elem_map_k[ne];
	const int elem = elem_map_1d(ne);

	double xx[BASIS_NODES_PER_ELEM];
	double yy[BASIS_NODES_PER_ELEM];
	double zz[BASIS_NODES_PER_ELEM];
	double uu[TUSAS_MAX_NUMEQS_X_BASIS_NODES_PER_ELEM];

	const int elemrow = elem*n_nodes_per_elem;
	for(int k = 0; k < n_nodes_per_elem; k++){
	  
	  const int nodeid = meshc_1d(elemrow+k);
	  
	  xx[k] = x_1dra(nodeid);
	  yy[k] = y_1dra(nodeid);
	  zz[k] = z_1dra(nodeid);

	  for( int neq = 0; neq < numeqs; neq++ ){
	    uu[n_nodes_per_elem*neq+k] = u_1dra(numeqs*nodeid+neq); //we can add uu_old, uu_oldold
	  }//neq
	}//k

	for( int neq = 0; neq < numeqs; neq++ ){
	  B[neq].computeElemData(&xx[0], &yy[0], &zz[0]);
	}//neq

	for(int gp=0; gp < ngp; gp++) {//gp
	  double jacwt = -99.;
	  //printf("%f\n",jacwt);
	  for( int neq = 0; neq < numeqs; neq++ ){
	    //jacwt = B[neq].getBasis(gp, &xx[0], &yy[0], &zz[0], &uu[neq*n_nodes_per_elem], NULL,NULL);//we can add uu_old, uu_oldold
	    jacwt = B[neq].getBasis(gp, &xx[0], &yy[0], &zz[0], &uu[neq*n_nodes_per_elem], &uu[neq*n_nodes_per_elem],&uu[neq*n_nodes_per_elem]);//we can add uu_old, uu_oldold
	  }
	  for (int i=0; i< n_nodes_per_elem; i++) {//i
	    const local_ordinal_type lrow = numeqs*meshc_1d(elemrow+i);
	    for(int j=0;j < n_nodes_per_elem; j++) {
	      local_ordinal_type lcol[1] = {numeqs*meshc_1d(elemrow+j)};
	      
	      for( int neq = 0; neq < numeqs; neq++ ){
#ifdef TUSAS_HAVE_CUDA
		scalar_type val[1] = {jacwt*(pf[neq])(&B[0],i,j,dt,t_theta,neq)};
#else
		scalar_type val[1] = {jacwt*h_pf[neq](&B[0],i,j,dt,t_theta,neq)};
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
    //cudaFree(d_pf);
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
  //#ifndef TUSASMUELU
  if( nonnull(outArgs.get_W_prec() )){

    MueLu::ReuseTpetraPreconditioner( P_, *prec_  );

  }//outArgs.get_W_prec() 
  //#endif
  return;
}

//====================================================================

template<class scalar_type>
void ModelEvaluatorTPETRA<scalar_type>::init_nox()
{
  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
  int mypid = comm_->getRank();
  if( 0 == mypid )
    std::cout<<std::endl<<"init_nox() started."<<std::endl<<std::endl;

  nnewt_=0;

  ::Stratimikos::DefaultLinearSolverBuilder builder;
  Teuchos::RCP<Teuchos::ParameterList> lsparams =
    Teuchos::rcp(new Teuchos::ParameterList(paramList.sublist(TusaslsNameString)));
  //#ifndef TUSASMUELU   
  //::Stratimikos::enableMueLu<local_ordinal_type,global_ordinal_type, node_type>(builder); 
  using Base = Thyra::PreconditionerFactoryBase<scalar_type>;
  using Impl = Thyra::MueLuPreconditionerFactory<scalar_type,local_ordinal_type,global_ordinal_type, node_type>;
  builder.setPreconditioningStrategyFactory(Teuchos::abstractFactoryStd<Base,Impl>(), "MueLu");
  //#endif
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
      atsList = &paramList.sublist (TusasatslistNameString, false );

      double relrestolp = 1.e-6;
      relrestolp = atsList->get<double>(TusaspredrelresNameString,1.e-6);
      int predmaxit = 20;
      predmaxit = paramList.get<int> (TusaspredmaxiterNameString,20);
      Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters1 =
	Teuchos::rcp(new NOX::StatusTest::MaxIters(predmaxit));
      Teuchos::RCP<NOX::StatusTest::NormF>relresid1 = 
	Teuchos::rcp(new NOX::StatusTest::NormF(*noxpred_group.get(), relrestolp));//1.0e-6 for paper
      Teuchos::RCP<NOX::StatusTest::Combo> converged1 =
	Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
      converged1->addStatusTest(relresid1);
      converged1->addStatusTest(maxiters1);
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
void ModelEvaluatorTPETRA<Scalar>::
set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory)
{
  W_factory_ = W_factory;
}

template<class Scalar>
Teuchos::RCP< ::Thyra::PreconditionerBase<Scalar> >
ModelEvaluatorTPETRA<Scalar>::create_W_prec() const
{

  //#ifndef TUSASMUELU
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
  //#else
  //return NULL;
  //#endif
}

template<class scalar_type>
Thyra::ModelEvaluatorBase::OutArgs<scalar_type>
ModelEvaluatorTPETRA<scalar_type>::createOutArgsImpl() const
{
  return prototypeOutArgs_;
}

template<class scalar_type>
double ModelEvaluatorTPETRA<scalar_type>::advance()
{
  auto comm_ = Teuchos::DefaultComm<int>::getComm();
  const int mypid = comm_->getRank();

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
    {
      Teuchos::RCP< Thyra::VectorBase< double > > guess;
      guess = Thyra::createVector(u_old_,x_space_);
      Teuchos::TimeMonitor NSolveTimer(*ts_time_nsolve);
      
      if(paramList.get<bool> (TusasestimateTimestepNameString)){
	Teuchos::ParameterList *atsList;
	atsList = &paramList.sublist (TusasatslistNameString, false );
	if(atsList->get<std::string> (TusasatstypeNameString) == "predictor corrector"){	  
	  predictor();
	  guess = Thyra::createVector(pred_temp_,x_space_);	
	}//if
      }//if

      NOX::Thyra::Vector thyraguess(*guess);
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
      &(dynamic_cast<const NOX::Thyra::Vector&>(
						solver_->getSolutionGroup().getX()
						).getThyraVector()
	);
    Thyra::ConstDetachedSpmdVectorView<double> x_vec(sol->col(0));
    
    Teuchos::ArrayRCP<scalar_type> unewv = u_new_->get1dViewNonConst();
    const size_t localLength = num_owned_nodes_;
    
    for (int nn=0; nn < localLength; nn++) {//cn figure out a better way here...
      for( int k = 0; k < numeqs_; k++ ){
	//uoldoldv[numeqs_*nn+k]=uoldv[numeqs_*nn+k];
	unewv[numeqs_*nn+k]=x_vec[numeqs_*nn+k];
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

	if( 0 == mypid)std::cout<<"     advance() step NOT ACCEPTED with dt = "<<dt_
				<<"; new dt = "<<dtpred
				<<"; and iterations = "<<numit<<std::endl;
	
      }else{
	if( 0 == mypid)std::cout<<"     advance() step accepted with dt = "<<dt_
				<<"; new dt = "<<dtpred
				<<"; and iterations = "<<numit<<std::endl;

	break;

      }//if
    }//if

  }//iter

  dtold_ = dt_;
  time_ += dt_;
  postprocess();
  //*u_old_old_ = *u_old_;
  u_old_old_->update(1.,*u_old_,0.);
  //*u_old_ = *u_new_;
  u_old_->update(1.,*u_new_,0.);

  for(boost::ptr_vector<error_estimator>::iterator it = Error_est.begin();it != Error_est.end();++it){
    //it->test_lapack();
    it->estimate_gradient(u_old_);
    it->estimate_error(u_old_);
  }
  ++numsteps_;   

  dt_ = dtpred;
  return dtold_;
}

template<class scalar_type>
  void ModelEvaluatorTPETRA<scalar_type>::initialize()
{
  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
  if( 0 == comm_->getRank()) std::cout<<std::endl<<"inititialize started"<<std::endl<<std::endl;
  bool dorestart = paramList.get<bool> (TusasrestartNameString);
  if (!dorestart){ 
    init(u_old_); 

    //*u_old_old_ = *u_old_;
    u_old_old_->scale(1.,*u_old_);
  
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
    }//if

    int mypid = comm_->getRank();
    int numproc = comm_->getSize();
    
    if( 1 == numproc ){//cn for now
      //if( 0 == mypid ){
      outfilename = "results.e";
      ex_id_ = mesh_->create_exodus(outfilename.c_str());//this calls ex_open
      
    }
    else{
      //std::string decompPath="decomp/";
      std::string decompPath=paramList.get<std::string> (TusasoutputpathNameString);

      std::string mypidstring(getmypidstring(mypid,numproc));

      outfilename = decompPath+"/results.e."+std::to_string(numproc)+"."+mypidstring;
      ex_id_ = mesh_->create_exodus(outfilename.c_str());
    }//if numproc

    mesh_->close_exodus(ex_id_);

    for( int k = 0; k < numeqs_; k++ ){
      mesh_->add_nodal_field((*varnames_)[k]);
    }

    if(paramList.get<bool> (TusasestimateTimestepNameString)){    
      setadaptivetimestep();
    }

    output_step_ = 1;
    write_exodus();
  }//if !dorestart
  else{
    restart(u_old_);//,u_old_old_);
//     if(1==comm_->MyPID())
//       std::cout<<"Restart unavailable"<<std::endl<<std::endl;
//     exit(0);
    for( int k = 0; k < numeqs_; k++ ){
      mesh_->add_nodal_field((*varnames_)[k]);
    }
  }//if !dorestart
   
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
  paramfunc_.resize(0);

  if("heat" == paramList.get<std::string> (TusastestNameString)){
    // numeqs_ number of variables(equations) 
    numeqs_ = 1;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    //(*residualfunc_)[0] = &tusastpetra::residual_heat_test_;
    (*residualfunc_)[0] = tpetra::heat::residual_heat_test_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::heat::prec_heat_test_dp_;
    
    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    
    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::heat::init_heat_test_;
    
    
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
    
    //  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
    //               [numeq][nodeset id]
    //  [variable index][nodeset index]
    (*dirichletfunc_)[0][0] = &dbc_zero_;							 
    (*dirichletfunc_)[0][1] = &dbc_zero_;						 
    (*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[0][3] = &dbc_zero_;

    paramfunc_.resize(1);
    paramfunc_[0] = &tpetra::heat::param_;

    neumannfunc_ = NULL;

    post_proc.push_back(new post_process(Comm,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &tpetra::heat::postproc_;

  }else if("radconvbc" == paramList.get<std::string> (TusastestNameString)){
    // numeqs_ number of variables(equations) 
    numeqs_ = 1;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    //(*residualfunc_)[0] = &tusastpetra::residual_heat_test_;
    (*residualfunc_)[0] = tpetra::heat::residual_heat_test_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::heat::prec_heat_test_dp_;
    
    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    
    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::radconvbc::init_heat_;
    
    
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
    
    //  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
    //               [numeq][nodeset id]
    //  [variable index][nodeset index]
    //(*dirichletfunc_)[0][0] = &dbc_zero_;							 
    (*dirichletfunc_)[0][1] = &tpetra::radconvbc::dbc_;						 
    //(*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[0][3] = &tpetra::radconvbc::dbc_;

    paramfunc_.resize(2);
    paramfunc_[0] = &tpetra::heat::param_;//heat
    paramfunc_[1] = &tpetra::radconvbc::param_;

    // numeqs_ number of variables(equations) 
    neumannfunc_ = new std::vector<std::map<int,NBCFUNC>>(numeqs_);
    //neumannfunc_ = NULL;
    //(*neumannfunc_)[0][0] = &tpetra::robin::nbc_robin_test_;								 
    //(*neumannfunc_)[0][1] = &nbc_zero_;						 
    (*neumannfunc_)[0][2] = &tpetra::radconvbc::nbc_;						 
    //(*neumannfunc_)[0][3] = &nbc_zero_;

    //post_proc.push_back(new post_process(Comm,mesh_,(int)0));
    //post_proc[0].postprocfunc_ = &tpetra::postproc_;

  }else if("NLheatIMR" == paramList.get<std::string> (TusastestNameString)){
    // numeqs_ number of variables(equations) 
    numeqs_ = 1;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    //(*residualfunc_)[0] = &tusastpetra::residual_heat_test_;
    (*residualfunc_)[0] = tpetra::residual_nlheatimr_test_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::heat::prec_heat_test_dp_;
    
    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    
    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::heat::init_heat_test_;
    
    
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
    
    //  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
    //               [numeq][nodeset id]
    //  [variable index][nodeset index]
    (*dirichletfunc_)[0][0] = &dbc_zero_;							 
    (*dirichletfunc_)[0][1] = &dbc_zero_;						 
    (*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[0][3] = &dbc_zero_;

    paramfunc_.resize(1);
    paramfunc_[0] = &tpetra::heat::param_;

    neumannfunc_ = NULL;

    post_proc.push_back(new post_process(Comm,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &tpetra::heat::postproc_;

  }else if("NLheatCN" == paramList.get<std::string> (TusastestNameString)){
    // numeqs_ number of variables(equations) 
    numeqs_ = 1;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    //(*residualfunc_)[0] = &tusastpetra::residual_heat_test_;
    (*residualfunc_)[0] = tpetra::residual_nlheatcn_test_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::prec_nlheatcn_test_dp_;
    
    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    
    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::heat::init_heat_test_;
    
    
    dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
    
    //  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
    //               [numeq][nodeset id]
    //  [variable index][nodeset index]
    (*dirichletfunc_)[0][0] = &dbc_zero_;							 
    (*dirichletfunc_)[0][1] = &dbc_zero_;						 
    (*dirichletfunc_)[0][2] = &dbc_zero_;						 
    (*dirichletfunc_)[0][3] = &dbc_zero_;

    paramfunc_.resize(1);
    paramfunc_[0] = &tpetra::heat::param_;

    neumannfunc_ = NULL;

    post_proc.push_back(new post_process(Comm,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &tpetra::heat::postproc_;


  }else if("heat2" == paramList.get<std::string> (TusastestNameString)){
    
    numeqs_ = 2;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);

    (*residualfunc_)[0] = tpetra::heat::residual_heat_test_dp_;
    (*residualfunc_)[1] = tpetra::heat::residual_heat_test_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::heat::prec_heat_test_dp_;
    (*preconfunc_)[1] = tpetra::heat::prec_heat_test_dp_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);

    (*initfunc_)[0] = &tpetra::heat::init_heat_test_;
    (*initfunc_)[1] = &tpetra::heat::init_heat_test_;

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

  }else if("cummins" == paramList.get<std::string> (TusastestNameString)){
    
    numeqs_ = 2;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
//     (*residualfunc_)[0] = &cummins::residual_heat_;
//     (*residualfunc_)[1] = &cummins::residual_phase_;
    (*residualfunc_)[0] = tpetra::heat::residual_heat_test_dp_;
    (*residualfunc_)[1] = tpetra::heat::residual_heat_test_dp_;

//     preconfunc_ = new std::vector<PREFUNC>(numeqs_);
//     (*preconfunc_)[0] = &cummins::prec_heat_;
//     (*preconfunc_)[1] = &cummins::prec_phase_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    //(*initfunc_)[0] = &cummins::init_heat_;
    //(*initfunc_)[1] = &cummins::init_phase_;
    (*initfunc_)[0] = &tpetra::heat::init_heat_test_;
    (*initfunc_)[1] = &tpetra::heat::init_heat_test_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    (*varnames_)[1] = "phi";

    dirichletfunc_ = NULL;

    neumannfunc_ = NULL;

    paramfunc_.resize(1);
    paramfunc_[0] = &cummins::param_;

  }else if("farzadi" == paramList.get<std::string> (TusastestNameString)){
    //farzadi test

    if(paramList.get<double> (TusasthetaNameString) < .49) exit(0);

    numeqs_ = 2;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::farzadi3d::init_conc_farzadi_;
    (*initfunc_)[1] = &tpetra::farzadi3d::init_phase_farzadi_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = tpetra::farzadi3d::residual_conc_farzadi_dp_;
    (*residualfunc_)[1] = tpetra::farzadi3d::residual_phase_farzadi_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::farzadi3d::prec_conc_farzadi_dp_;
    (*preconfunc_)[1] = tpetra::farzadi3d::prec_phase_farzadi_dp_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    (*varnames_)[1] = "phi";

    dirichletfunc_ = NULL;

    paramfunc_.resize(1);
    paramfunc_[0] = &tpetra::farzadi3d::param_;
    //paramfunc_ = &farzadi::param_;

    neumannfunc_ = NULL;

  }else if("farzadiexp" == paramList.get<std::string> (TusastestNameString)){
    //farzadi test

    if(paramList.get<double> (TusasthetaNameString) < .49) exit(0);

    numeqs_ = 2;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::farzadi3d::init_conc_farzadi_;
    (*initfunc_)[1] = &tpetra::farzadi3d::init_phase_farzadi_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = tpetra::farzadi3d::residual_conc_farzadi_dp_;
    (*residualfunc_)[1] = tpetra::farzadi3d::residual_phase_farzadi_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::farzadi3d::prec_conc_farzadi_dp_;
    (*preconfunc_)[1] = tpetra::farzadi3d::prec_phase_farzadi_dp_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    (*varnames_)[1] = "phi";

    dirichletfunc_ = NULL;

    paramfunc_.resize(1);
    paramfunc_[0] = &tpetra::farzadi3d::param_;
    //paramfunc_ = &farzadi::param_;

    neumannfunc_ = NULL;

  }else if("farzadi_test" == paramList.get<std::string> (TusastestNameString)){
    //farzadi test

    numeqs_ = 2;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::farzadi3d::init_conc_farzadi_;
    (*initfunc_)[1] = &tpetra::farzadi3d::init_phase_farzadi_test_;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = tpetra::farzadi3d::residual_conc_farzadi_dp_;
    (*residualfunc_)[1] = tpetra::farzadi3d::residual_phase_farzadi_dp_;

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

    paramfunc_.resize(1);
    paramfunc_[0] = &tpetra::farzadi3d::param_;

    neumannfunc_ = NULL;

  }else if("pfhub3" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 2;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::pfhub3::init_heat_pfhub3_;
    (*initfunc_)[1] = &tpetra::pfhub3::init_phase_pfhub3_;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = tpetra::pfhub3::residual_heat_pfhub3_dp_;
    (*residualfunc_)[1] = tpetra::pfhub3::residual_phase_pfhub3_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::pfhub3::prec_heat_pfhub3_dp_;
    (*preconfunc_)[1] = tpetra::pfhub3::prec_phase_pfhub3_dp_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    (*varnames_)[1] = "phi";

    dirichletfunc_ = NULL;

    paramfunc_.resize(1);
    paramfunc_[0] = &tpetra::pfhub3::param_;

    neumannfunc_ = NULL;

  }else if("pfhub2kks" == paramList.get<std::string> (TusastestNameString)){

    Teuchos::ParameterList *problemList;
    problemList = &paramList.sublist ( "ProblemParams", false );

    int numeta = problemList->get<int>("N");

    numeqs_ = numeta+1;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = tpetra::pfhub2::residual_c_kks_dp_;
    (*residualfunc_)[1] = tpetra::pfhub2::residual_eta_kks_dp_;
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

    neumannfunc_ = NULL;

    paramfunc_.resize(1);
    paramfunc_[0] = &tpetra::pfhub2::param_;

    post_proc.push_back(new post_process(Comm,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &pfhub2::postproc_c_b_;

    post_proc.push_back(new post_process(Comm,mesh_,(int)1));
    post_proc[1].postprocfunc_ = &tpetra::pfhub2::postproc_c_a_;

  }else if("robin" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 1;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &tpetra::robin::residual_robin_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &tpetra::robin::prec_robin_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    //(*initfunc_)[0] = &init_neumann_test_;
    (*initfunc_)[0] = &tpetra::robin::init_robin_test_;

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
    (*neumannfunc_)[0][1] = &tpetra::robin::nbc_robin_test_;						 
    //(*neumannfunc_)[0][1] = &nbc_zero_;						 
    //(*neumannfunc_)[0][2] = &nbc_zero_;						 
    //(*neumannfunc_)[0][3] = &nbc_zero_;

    post_proc.push_back(new post_process(Comm,mesh_,(int)0));
    post_proc[0].postprocfunc_ = &tpetra::robin::postproc_robin_;

  }else if("timeonly" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 1;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &tpetra::timeonly::residual_test_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &tpetra::heat::prec_heat_test_;

    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &timeonly::init_test_;

    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";

    // numeqs_ number of variables(equations) 
    dirichletfunc_ = NULL;

    neumannfunc_ = NULL;

  }else if("autocatalytic4" == paramList.get<std::string> (TusastestNameString)){

    numeqs_ = 4;

    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    (*residualfunc_)[0] = &tpetra::autocatalytic4::residual_a_;
    (*residualfunc_)[1] = &tpetra::autocatalytic4::residual_b_;
    (*residualfunc_)[2] = &tpetra::autocatalytic4::residual_ab_;
    (*residualfunc_)[3] = &tpetra::autocatalytic4::residual_c_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = &tpetra::heat::prec_heat_test_;
    (*preconfunc_)[1] = &tpetra::heat::prec_heat_test_;
    (*preconfunc_)[2] = &tpetra::heat::prec_heat_test_;
    (*preconfunc_)[3] = &tpetra::heat::prec_heat_test_;

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

  }else if("goldak" == paramList.get<std::string> (TusastestNameString)){
    // numeqs_ number of variables(equations) 
    numeqs_ = 1;
    
    residualfunc_ = new std::vector<RESFUNC>(numeqs_);
    //(*residualfunc_)[0] = &tusastpetra::residual_heat_test_;
    (*residualfunc_)[0] = tpetra::goldak::residual_test_dp_;

    preconfunc_ = new std::vector<PREFUNC>(numeqs_);
    (*preconfunc_)[0] = tpetra::heat::prec_heat_test_dp_;
    
    varnames_ = new std::vector<std::string>(numeqs_);
    (*varnames_)[0] = "u";
    
    initfunc_ = new  std::vector<INITFUNC>(numeqs_);
    (*initfunc_)[0] = &tpetra::goldak::init_heat_;
    
    
    dirichletfunc_ = NULL;
    //dirichletfunc_ = new std::vector<std::map<int,DBCFUNC>>(numeqs_);
    
    //  cubit nodesets start at 1; exodus nodesets start at 0, hence off by one here
    //               [numeq][nodeset id]
    //  [variable index][nodeset index]
    //(*dirichletfunc_)[0][0] = &dbc_zero_;							 
    //(*dirichletfunc_)[0][1] = &tpetra::radconvbc::dbc_;						 
    //(*dirichletfunc_)[0][2] = &dbc_zero_;						 
    //(*dirichletfunc_)[0][3] = &tpetra::radconvbc::dbc_;

    paramfunc_.resize(3);//maybe push_back() is better here?
    paramfunc_[0] = &tpetra::heat::param_;
    paramfunc_[1] = &tpetra::radconvbc::param_;
    paramfunc_[2] = &tpetra::goldak::param_;

    // numeqs_ number of variables(equations) 
//     neumannfunc_ = NULL;
    
    neumannfunc_ = new std::vector<std::map<int,NBCFUNC>>(numeqs_);	 
    (*neumannfunc_)[0][4] = &tpetra::radconvbc::nbc_;

    post_proc.push_back(new post_process(Comm,mesh_,(int)0, post_process::MAXVALUE));
    post_proc[0].postprocfunc_ = &tpetra::goldak::postproc_qdot_;
    post_proc.push_back(new post_process(Comm,mesh_,(int)1, post_process::MAXVALUE));
    post_proc[1].postprocfunc_ = &tpetra::goldak::postproc_u_;


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
  
  for( int k = 0; k <  paramfunc_.size(); k++ ){
    auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
    if( 0 == comm_->getRank() ){
      std::cout<<"Initial paranfunc evaluated for k = "<<k<<std::endl;
    }
    paramfunc_[k](problemList);
  }
  
}

template<class scalar_type>
void ModelEvaluatorTPETRA<scalar_type>::write_exodus()
//void ModelEvaluatorNEMESIS<scalar_type>::write_exodus(const int output_step)
{
  update_mesh_data();

  //not sre what the bug is here...
  Teuchos::TimeMonitor IOWriteTimer(*ts_time_iowrite);
  mesh_->open_exodus(outfilename.c_str(),Mesh::WRITE);
  mesh_->write_exodus(ex_id_,output_step_,time_);
  mesh_->close_exodus(ex_id_);
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
  comm_->barrier();
  if ( (solver_->getList()).sublist("Direction").sublist("Newton").sublist("Linear Solver")
       .isSublist("Output") == true){
    if ( (solver_->getList()).sublist("Direction").sublist("Newton").sublist("Linear Solver")
	 .sublist("Output").getEntryPtr("Cumulative Iteration Count") != NULL){
      ngmres = ((solver_->getList()).sublist("Direction").sublist("Newton").sublist("Linear Solver")
		.sublist("Output").getEntry("Cumulative Iteration Count")).getValue(&ngmres);
    }
  }

  if( 0 == mypid ){
    //int numstep = paramList.get<int> (TusasntNameString) - this->start_step;
    int numstep = numsteps_;
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
  const int mypid = comm_->getRank();
  const int numproc = comm_->getSize();
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
  Teuchos::reduceAll<int,double>(*comm_,Teuchos::REDUCE_MIN,
				 1,
				 &time,
				 &min_time);
  double max_time = 1.e-12;
  Teuchos::reduceAll<int,double>(*comm_,Teuchos::REDUCE_MAX,
				 1,
				 &time,
				 &max_time);

  //this is probably fixed by setting time = min_time and reading that timestep.
  //care may need to be taken when overwriting the max_time step, may need a clobber/noclobber

  //dt_/4 is arbitrary
  if(fabs(max_time-min_time)>dt_/4.){
    if( 0 == mypid ){
      std::cout<<"  Error reading restart min and max time differ"<<std::endl;
      std::cout<<"    Reading restart min time = "<<min_time<<std::endl;
      std::cout<<"    Reading restart max time = "<<max_time<<std::endl;
      std::cout<<"    Reading restart difference = "<<fabs(max_time-min_time)<<std::endl;
    }
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

  mesh_->close_exodus(ex_id_);

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

  step = step - 1;
  this->start_time = time;
  int ntstep = (int)(time/dt_);
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
  std::vector<double> uuold(numeqs_);
  std::vector<double> uuoldold(numeqs_);
  std::vector<double> ug(dim*numee);

  auto uview = u_new_->get1dView();
  auto uoldview = u_old_->get1dView();
  auto uoldoldview = u_old_old_->get1dView();

  for (int nn=0; nn < num_owned_nodes_; nn++) {

    for( int k = 0; k < numeqs_; k++ ){
      uu[k] = uview[numeqs_*nn+k];
      uuold[k] = uoldview[numeqs_*nn+k];
      uuoldold[k] = uoldoldview[numeqs_*nn+k];
    }

    for( int k = 0; k < numee; k++ ){
      ug[k*dim] = (*(Error_est[k].gradx_))[nn];
      ug[k*dim+1] = (*(Error_est[k].grady_))[nn];
      ug[k*dim+2] = (*(Error_est[k].gradz_))[nn];
    }

    boost::ptr_vector<post_process>::iterator itp;
    for(itp = post_proc.begin();itp != post_proc.end();++itp){
      itp->process(nn,&uu[0],&uuold[0],&uuoldold[0],&ug[0],time_,dt_,dt_);
    }

  }//nn
}

template<class Scalar>
void ModelEvaluatorTPETRA<Scalar>::temporalpostprocess(boost::ptr_vector<post_process> pp)
{
  if(0 == pp.size() ) return;

  Teuchos::TimeMonitor ImportTimer(*ts_time_temperr);

  int numee = Error_est.size();
  //ordering is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz

  const int dim = 3;
  const int numeqs = numeqs_; //cuda 8 lambdas dont capture private data

  std::vector<double> uu(numeqs);
  std::vector<double> uuold(numeqs);
  std::vector<double> uuoldold(numeqs);
  std::vector<double> ug(numeqs);
  
  //right now for cuda, do this on cpu since the function pointer is buried in another class
  //and it takes very little percentage of time
  //another approach would be to use tpetra vectors here
  Teuchos::ArrayRCP<const scalar_type> unewview = u_new_->get1dView();
  Teuchos::ArrayRCP<const scalar_type> uoldview = u_old_->get1dView();
  Teuchos::ArrayRCP<const scalar_type> predtempview = pred_temp_->get1dView();
  for (int nn=0; nn < num_owned_nodes_; nn++) {
    for( int k = 0; k < numeqs; k++ ){
      uu[k] = unewview[numeqs*nn+k];
      uuold[k] = uoldview[numeqs*nn+k];
      //uuoldold[k] = (*u_old_old_)[numeqs_*nn+k];
      uuoldold[k] = predtempview[numeqs*nn+k];
      ug[k] = predtempview[numeqs*nn+k];
    }

    boost::ptr_vector<post_process>::iterator itp;
    for(itp = pp.begin();itp != pp.end();++itp){
      itp->process(nn,&uu[0],&uuold[0],&uuoldold[0],&ug[0],time_,dt_,dtold_);
      //std::cout<<nn<<" "<<mesh_->get_local_id((x_owned_map_->GID(nn))/numeqs_)<<" "<<xyz[0]<<std::endl;
    }
  }//nn
}

template<class Scalar>
void ModelEvaluatorTPETRA<Scalar>::init_P_()
{

  //this could be done with Tpetra::replaceDiagonalCrsMatrix( 	::Tpetra::CrsMatrix< SC, LO, GO, NT > &  	matrix,
  //		const ::Tpetra::Vector< SC, LO, GO, NT > &  	newDiag 
  //	) 	

    P_->setAllToScalar((scalar_type)-1.0); 

    Teuchos::RCP<vector_type > d = Teuchos::rcp(new vector_type(x_owned_map_));
    d->putScalar((Scalar)27.);
    Tpetra::replaceDiagonalCrsMatrix(*P_,*d);

    //P_->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
    P_->fillComplete();
}

template<class Scalar>
double ModelEvaluatorTPETRA<Scalar>::estimatetimestep()
{
  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
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
  
  if( 0 == comm_->getRank()){
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
    if( 0 == comm_->getRank()){
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

  if( 0 == comm_->getRank()){
    std::cout<<std::endl<<"     Estimated timestep size : "<<dtpred<<std::endl;	
  }
  dtpred = std::min(dtpred,dtmax);
  if( 0 == comm_->getRank()){
    std::cout<<std::endl<<"           min(dtpred,dtmax) : "<<dtpred<<std::endl<<std::endl;	
  }  
  return dtpred;
}

template<class Scalar>
void ModelEvaluatorTPETRA<Scalar>::predictor()
{
  //right now theta2=0 corresponds to FE, BE and TR
  //theta2=1 corresponds to AB

  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
  if( 0 == comm_->getRank()){
    std::cout<<std::endl<<std::endl<<std::endl<<"     Predictor step started"<<std::endl;	
  }

  //we might turn off the forcing term temporaily here

  const double t_theta_temp = t_theta_;

  t_theta2_ = 0.;
  if(t_theta_ > 0.45 && t_theta_ <.55) t_theta2_ = 1.;//ab predictor tr corrector
  //fe predictor    be corrector
  t_theta_ = 0.;

  Teuchos::RCP< ::Thyra::VectorBase< double > > guess = Thyra::createVector(u_old_,x_space_);
  NOX::Thyra::Vector thyraguess(*guess);//by sending the dereferenced pointer, we instigate a copy rather than a view
  predictor_->reset(thyraguess);

  {
    Teuchos::TimeMonitor PredTimer(*ts_time_predsolve);
    NOX::StatusTest::StatusType solvStatus = predictor_->solve();
  }

  const Thyra::VectorBase<double> * sol = 
    &(dynamic_cast<const NOX::Thyra::Vector&>(predictor_->getSolutionGroup().getX()
					      ).getThyraVector()
      );    
  Thyra::ConstDetachedSpmdVectorView<double> x_vec(sol->col(0));

  Teuchos::ArrayRCP<scalar_type> predtempview = pred_temp_->get1dViewNonConst();

  const int localLength = num_owned_nodes_;

  for (int nn=0; nn < localLength; nn++) {//cn figure out a better way here...
    for( int k = 0; k < numeqs_; k++ ){
      predtempview[numeqs_*nn+k]=x_vec[numeqs_*nn+k];
    }
  } 

  if( 0 == comm_->getRank()){
    std::cout<<std::endl<<"     Predictor step ended"<<std::endl<<std::endl<<std::endl;
  }

  t_theta_ = t_theta_temp;
  t_theta2_ = 0.;
 }

template<class Scalar>
void ModelEvaluatorTPETRA<Scalar>::initialsolve()
 {     
   auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
   //right now, for TR it doesn't really matter in turns of performance if we set theta to 1
   //here or leave it at .5
   const double t_theta_temp = t_theta_;
   t_theta_ = 1.;
   
   t_theta2_ = 0.;
   
   if( 0 == comm_->getRank()) 
     std::cout<<std::endl<<"Performing initial NOX solve"<<std::endl<<std::endl;
   
   Teuchos::RCP< ::Thyra::VectorBase< double > > guess = Thyra::createVector(u_old_,x_space_);
   NOX::Thyra::Vector thyraguess(*guess);//by sending the dereferenced pointer, we instigate a copy rather than a view
   solver_->reset(thyraguess);
   
   Teuchos::TimeMonitor NSolveTimer(*ts_time_nsolve);
   NOX::StatusTest::StatusType solvStatus = solver_->solve();
   if( !(NOX::StatusTest::Converged == solvStatus)) {
     std::cout<<" NOX solver failed to converge. Status = "<<solvStatus<<std::endl<<std::endl;
     exit(0);
   }

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
  Teuchos::ArrayRCP<const scalar_type> uoldview = u_old_->get1dView();
  Teuchos::ArrayRCP<scalar_type> uoldoldview = u_old_old_->get1dViewNonConst();

   for (int nn=0; nn <  num_owned_nodes_; nn++) {//cn figure out a better way here...
     for( int k = 0; k < numeqs_; k++ ){
       
       uoldoldview[numeqs_*nn+k] = 2.*uoldview[numeqs_*nn+k] - x_vec[numeqs_*nn+k];
   
     }
   }
   
   t_theta_ = t_theta_temp;
    
   if( 0 == comm_->getRank()) 
     std::cout<<std::endl<<"Initial NOX solve completed"<<std::endl<<std::endl;
 }

template<class Scalar>
void ModelEvaluatorTPETRA<Scalar>::setadaptivetimestep()
  {
      //cn this is not gong to work with multiple k
      //what do we do with temporal_est[0].pos....
      //is index_ correct here??
      Teuchos::ParameterList *atsList;
      atsList = &paramList.sublist (TusasatslistNameString, false );
      for( int k = 0; k < numeqs_; k++ ){
	temporal_est.push_back(new post_process(Comm,
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

	temporal_norm.push_back(new post_process(Comm,
						 mesh_,
						 k, 
						 post_process::NORMRMS, 
						 k, 
						 "tempnorm",
						 16));
 	temporal_norm[k].postprocfunc_ = &timeadapt::normu_;

      }
  }
#endif
