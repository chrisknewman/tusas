//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef NOX_THYRA_MODEL_EVALUATOR_TPETRA_DECL_HPP
#define NOX_THYRA_MODEL_EVALUATOR_TPETRA_DECL_HPP

#include <Teuchos_Comm.hpp>

#include <Tpetra_Vector.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Export.hpp>
#include <Tpetra_CrsGraph_decl.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>

//teuchos support
#include <Teuchos_RCP.hpp>

//#include <MueLu_HierarchyManager.hpp>
#include <MueLu_TpetraOperator.hpp>

#include "Thyra_StateFuncModelEvaluatorBase.hpp"

#include "Mesh.h"
#include "error_estimator.h"
#include "elem_color.h"
#include "timestep.hpp"
#include "post_process.h"
#include "random_distribution.h"

#include <boost/ptr_container/ptr_vector.hpp>

template<class Scalar> class ModelEvaluatorTPETRA;

template<class Scalar>
Teuchos::RCP<ModelEvaluatorTPETRA<Scalar> >
modelEvaluatorTPETRA( const Teuchos::RCP<const Epetra_Comm>& comm,
			Mesh &mesh,
			 Teuchos::ParameterList plist 
			 );
/// Implentation of timestep with MPI and OpenMP support.
template<class Scalar>
class ModelEvaluatorTPETRA
  : public ::timestep<Scalar>, public ::Thyra::StateFuncModelEvaluatorBase<Scalar>
{
public:
  /// Constructor
  ModelEvaluatorTPETRA( const Teuchos::RCP<const Epetra_Comm>& comm,
			Mesh *mesh,
			   Teuchos::ParameterList plist 
		       );
  /// Destructor
  ~ModelEvaluatorTPETRA(){};

  typedef Tpetra::Vector<>::scalar_type scalar_type;

  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > get_x_space() const{return x_space_;};
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > get_f_space() const{return f_space_;};
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> createInArgs() const{return prototypeInArgs_;};

  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> getNominalValues() const{return nominalValues_;};
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  Teuchos::RCP< ::Thyra::PreconditionerBase< Scalar > > create_W_prec() const;
  Teuchos::RCP< ::Thyra::LinearOpBase< Scalar > > create_W_op() const;

  void initialize();
  void finalize();
  double advance();
  void write_exodus();

  void evalModelImpl(
		     const ::Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
		     const ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs
		     ) const;

private:

  typedef Tpetra::Vector<>::global_ordinal_type global_ordinal_type;
  typedef Tpetra::Vector<>::local_ordinal_type local_ordinal_type;
  typedef Tpetra::Vector<>::node_type node_type;
  typedef Tpetra::CrsMatrix<scalar_type,local_ordinal_type, global_ordinal_type,
                         node_type>::crs_graph_type crs_graph_type;

  typedef Tpetra::global_size_t global_size_t;
  typedef Tpetra::Vector<scalar_type, local_ordinal_type,
			 global_ordinal_type, node_type> vector_type;
  typedef Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type> map_type;
  typedef Tpetra::Import<local_ordinal_type, global_ordinal_type,
                         node_type> import_type;
  typedef Tpetra::Export<local_ordinal_type, global_ordinal_type,
                         node_type> export_type;
  typedef Tpetra::CrsMatrix<scalar_type,local_ordinal_type, global_ordinal_type,
                         node_type> matrix_type;
  typedef Tpetra::MultiVector<scalar_type, local_ordinal_type,
			      global_ordinal_type, node_type> mv_type;

  ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> createOutArgsImpl() const;

  /// Allocates and returns the Jacobian matrix graph.
  virtual Teuchos::RCP<crs_graph_type> createGraph(); 

  Teuchos::RCP<crs_graph_type> createOverlapGraph(); 

  Mesh* mesh_;

  int update_mesh_data();

  void restart(Teuchos::RCP<vector_type> u);//,Teuchos::RCP<vector_type> u_old);

  void set_test_case();

  double time_;

  int ex_id_;

  int output_step_;
  int numeqs_;
  int num_owned_nodes_;
  int num_overlap_nodes_;
  int numsteps_;

  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > x_space_;
  Teuchos::RCP<const map_type > x_overlap_map_;
  Teuchos::RCP<const map_type > x_owned_map_;
  Teuchos::RCP<const map_type > node_overlap_map_;
  Teuchos::RCP<const map_type > node_owned_map_;

  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > f_space_;

  Teuchos::RCP<const import_type > importer_;
  Teuchos::RCP<const export_type > exporter_;
  Teuchos::RCP<NOX::Solver::Generic> solver_;
  Teuchos::RCP<NOX::Solver::Generic> predictor_;

  Teuchos::RCP<vector_type> u_old_;
  Teuchos::RCP<vector_type> u_old_old_;
  Teuchos::RCP<vector_type> u_new_;
  Teuchos::RCP<vector_type> pred_temp_;

  Teuchos::RCP<vector_type> x_;
  Teuchos::RCP<vector_type> y_;
  Teuchos::RCP<vector_type> z_;

  Teuchos::RCP<mv_type> muelucoords_;

  Teuchos::RCP<crs_graph_type>  W_graph_;
  Teuchos::RCP<crs_graph_type>  W_overlap_graph_;
  Teuchos::RCP<matrix_type> P_;
  Teuchos::RCP<matrix_type> P;
//Teuchos::RCP<MueLu::HierarchyManager<scalar_type,local_ordinal_type, global_ordinal_type, node_type>> mueluFactory_;
  Teuchos::RCP<MueLu::TpetraOperator<scalar_type,local_ordinal_type, global_ordinal_type, node_type> > prec_;
  
  int nnewt_;
  double dt_;
  double dtold_;
  double t_theta_;
  double t_theta2_;
  Teuchos::ParameterList paramList;

  Thyra::ModelEvaluatorBase::InArgs<Scalar> nominalValues_;
  //Teuchos::RCP< ::Thyra::VectorBase<scalar_type> > x0_;
  Teuchos::RCP<vector_type > x0_;
  Thyra::ModelEvaluatorBase::InArgs<Scalar> prototypeInArgs_;
  Thyra::ModelEvaluatorBase::OutArgs<Scalar> prototypeOutArgs_;
  Teuchos::RCP<::Thyra::VectorBase< Scalar > > scaling_;

  void update_left_scaling();

  void print_norms();

  /// Initialize and create the NOX and linear solvers.
  void init_nox();
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  void set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory);
 
  Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> > W_factory_;

  void init(Teuchos::RCP<vector_type> u);

  std::vector<std::string> *varnames_;

  //do we want to move these typedefs to function_def.hpp? would need to do it for nemesis class as well
  //typedef double (*RESFUNC)(const GPUBasis * const * basis, 
  typedef const double (*RESFUNC)(GPUBasis * basis[], 
				  const int &i, 
				  const double &dt_, 
				  const double &dtold_, 
				  const double &t_theta_, 
				  const double &t_theta2_, 
				  const double &time,
				  const int &eqn_id, 
				  const double &vol, 
				  const double &rand);

  std::vector<RESFUNC> *residualfunc_;


  typedef const double (*PREFUNC)(GPUBasis *basis[], 
				  const int &i,
				  const int &j, 
				  const double &dt_, 
				  const double &t_theta_, 
				  const int &eqn_id);

  std::vector<PREFUNC> *preconfunc_;


  typedef const double (*DBCFUNC)(const double &x,
				  const double &y,
				  const double &z,
				  const double &t);

  std::vector<std::map<int,DBCFUNC>> *dirichletfunc_;

  typedef const double (*INITFUNC)(const double &x,
				   const double &y,
				   const double &z,
				   const int &eqn_id,
				   const int &lid);

  typedef const double (*NBCFUNC)(const GPUBasis *basis,
				  const int &i, 
				  const double &dt_, 
				  const double &dtold_, 
				  const double &t_theta_,
				  const double &t_theta2_,
				  const double &time);

  std::vector<std::map<int,NBCFUNC>> *neumannfunc_;


  std::vector<INITFUNC> *initfunc_;

  typedef void (*PARAMFUNC)(Teuchos::ParameterList *plist);

  std::vector<PARAMFUNC> paramfunc_;
  //PARAMFUNC paramfunc_;

  Teuchos::RCP<Teuchos::Time> ts_time_import;
  Teuchos::RCP<Teuchos::Time> ts_time_resimport;
  Teuchos::RCP<Teuchos::Time> ts_time_precimport;
  Teuchos::RCP<Teuchos::Time> ts_time_resfill;
  Teuchos::RCP<Teuchos::Time> ts_time_resdirichlet;
  Teuchos::RCP<Teuchos::Time> ts_time_precfill;
  Teuchos::RCP<Teuchos::Time> ts_time_precdirichlet;
  Teuchos::RCP<Teuchos::Time> ts_time_nsolve;
  Teuchos::RCP<Teuchos::Time> ts_time_view;
  Teuchos::RCP<Teuchos::Time> ts_time_iowrite;
  Teuchos::RCP<Teuchos::Time> ts_time_temperr;
  Teuchos::RCP<Teuchos::Time> ts_time_predsolve;
  //RCP<Teuchos::Time> ts_time_ioread;

  //hacked stuff for elem_color
  Teuchos::RCP<elem_color> Elem_col;
  Teuchos::RCP<const Epetra_Comm>  Comm;
  //Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> x_1dra;
  //Kokkos::View<const double*> x_1dra; 
  
  //TUSAS_CUDA_CALLABLE_MEMBER void set_basis( GPUBasis &basis, const std::string elem_type) const;

  boost::ptr_vector<error_estimator> Error_est;
  boost::ptr_vector<post_process> post_proc;
  void postprocess();

  void predictor();
  void initialsolve();

  double estimatetimestep();
  void temporalpostprocess(boost::ptr_vector<post_process>pp);
  boost::ptr_vector<post_process> temporal_est;
  boost::ptr_vector<post_process> temporal_norm;
  boost::ptr_vector<post_process> temporal_dyn;
  void setadaptivetimestep();

  void init_P_();
  std::string outfilename;

  std::vector<int> localprojectionindices_;

  Teuchos::RCP<random_distribution> randomdistribution;

  bool predictor_step;
  bool corrector_step;

  std::string get_basis_name(const std::string elem_type) const;
};

//==================================================================
#include "ModelEvaluatorTPETRA_def.hpp"
//==================================================================

#endif
