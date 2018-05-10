//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef NOX_THYRA_MODEL_EVALUATOR_NEMESIS_DECL_HPP
#define NOX_THYRA_MODEL_EVALUATOR_NEMESIS_DECL_HPP

#include "Thyra_StateFuncModelEvaluatorBase.hpp"	
#include "Teuchos_ParameterList.hpp"
#include <Teuchos_TimeMonitor.hpp>
#include "Epetra_FECrsGraph.h"

#include "Mesh.h"
#include "preconditioner.hpp"
#include "timestep.hpp"
#include "error_estimator.h"
#include "elem_color.h"
#include "post_process.h"
#include "periodic_bc.h"

#include <boost/ptr_container/ptr_vector.hpp>

template<class Scalar> class ModelEvaluatorNEMESIS;

template<class Scalar>
Teuchos::RCP<ModelEvaluatorNEMESIS<Scalar> >
modelEvaluatorNEMESIS(const Teuchos::RCP<const Epetra_Comm>& comm,
			 Mesh &mesh,
			 Teuchos::ParameterList plist 
			 );
/// Implentation of timestep with MPI and OpenMP support.
template<class Scalar>
class ModelEvaluatorNEMESIS
  : public ::timestep<Scalar>, public ::Thyra::StateFuncModelEvaluatorBase<Scalar>
{
public:
  /// Constructor
  ModelEvaluatorNEMESIS(const Teuchos::RCP<const Epetra_Comm>& comm,
			   Mesh *mesh,
			   Teuchos::ParameterList plist 
			   );
  /// Destructor
  ~ModelEvaluatorNEMESIS();
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  void set_x0(const Teuchos::ArrayView<const Scalar> &x0);
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  void setShowGetInvalidArgs(bool showGetInvalidArg);
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  void set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory);
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > get_x_space() const;
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > get_f_space() const;
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> getNominalValues() const;
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  Teuchos::RCP< ::Thyra::LinearOpBase<Scalar> > create_W_op() const;
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> > get_W_factory() const;
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> createInArgs() const;
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  Teuchos::RCP< ::Thyra::PreconditionerBase< Scalar > > create_W_prec() const;

  /// Initialize and create the NOX and linear solvers.
  void init_nox();
  void initialize();
  void finalize();
  void advance();
  /// Compute a global L^2 error based on an analytic solution.
  void compute_error( double *u);
  //void write_exodus(const int output_step);
  void write_exodus();
  /// Write solution to a matlab readable file.
  void write_matlab();
  /// Fill u and u_old with restart values.
  void restart(Teuchos::RCP<Epetra_Vector> u,Teuchos::RCP<Epetra_Vector> u_old);

private:

  /// Allocates and returns the Jacobian matrix graph.
  virtual Teuchos::RCP<Epetra_FECrsGraph> createGraph();


  ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> createOutArgsImpl() const;

  void evalModelImpl(
    const ::Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
    const ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs
    ) const;

  const Teuchos::RCP<const Epetra_Comm>  comm_;

  Mesh *mesh_;

  //const Scalar  dt_;
  double dt_;

  Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > x_space_;
  Teuchos::RCP<const Epetra_Map>   x_owned_map_;
  Teuchos::RCP<const Epetra_Map>   x_overlap_map_;

  Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > f_space_;
  Teuchos::RCP<const Epetra_Map>   f_owned_map_;

  Teuchos::RCP<const Epetra_Import> importer_;

  Teuchos::RCP<Epetra_FECrsGraph>  W_graph_;

  Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> > W_factory_;

  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> nominalValues_;
  Teuchos::RCP< ::Thyra::VectorBase<Scalar> > x0_;
  bool showGetInvalidArg_;
  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> prototypeInArgs_;
  ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> prototypeOutArgs_;
  Teuchos::RCP<Epetra_FECrsMatrix> P_;
  Teuchos::RCP<preconditioner <Scalar> > prec_;
  Teuchos::RCP<NOX::Solver::Generic> solver_;
  Teuchos::RCP<Epetra_Vector> u_old_;
  Teuchos::RCP<Epetra_Vector> u_old_old_;
  Teuchos::RCP<Epetra_Vector> dudt_;

  void set_test_case();

  double time_;

  int ex_id_;

  int output_step_;
  //cn here now to have output control in this class
  //cn a version of this lives in tusas now
  //int curr_step

  int numeqs_;

  int num_my_nodes_;
  int num_nodes_;

  //cn these are parameters for cummins
  double D_, T_m_, T_inf_, alpha_, M_, eps_,eps_0_, R_0_;
  double phi_sol_, phi_liq_;

  double random_number_, random_number_old_;
  Teuchos::RCP<Epetra_Vector> random_vector_;
  Teuchos::RCP<Epetra_Vector> random_vector_old_;

  const double gs(const double &theta);
  const double R(const double &theta);
  const double R(const double &theta,const double &psi);
  double theta(double &x,double &y) const;
  double psi(double &x,double &y,double &z) const;

  void init(Teuchos::RCP<Epetra_Vector> u);
  void init_square(Teuchos::RCP<Epetra_Vector> u);
  void init_karma(Teuchos::RCP<Epetra_Vector> u);
  void multi(Teuchos::RCP<Epetra_Vector> u);
  void pool(Teuchos::RCP<Epetra_Vector> u);

  int update_mesh_data();

  double t_theta_;

  double theta_0_;

  Teuchos::ParameterList paramList;

  int nnewt_;

  //cn function pointers
  double (*hp1_)(const double &phi,const double &delta);
  double (*hpp1_)(const double &phi,const double &delta);
  double (*w_)(const double &delta);
  double (*m_)(const double &theta,const double &M,const double &eps);
  double (*rand_phi_)(const double &phi, const double &random_number);
  double (*gp1_)(const double &phi);
  double (*gpp1_)(const double &phi);
  double (*hp2_)(const double &phi);

  double (*gs2_)(const double &theta,const double &M, const double &eps, const double &psi);
  double (*dgs2_2dtheta_)(const double &theta,const double &M, const double &eps, const double &psi);
  double (*dgs2_2dpsi_)(const double &theta,const double &M, const double &eps, const double &psi);

  typedef double (*RESFUNC)(const boost::ptr_vector<Basis> &basis, 
			    const int &i, 
			    const double &dt_, 
			    const double &t_theta_, 
			    const double &time,
			    const int &eqn_id);

  std::vector<RESFUNC> *residualfunc_;

  typedef double (*PREFUNC)(const boost::ptr_vector<Basis> &basis, 
			    const int &i,  
			    const int &j,
			    const double &dt_, 
			    const double &t_theta_,
			    const int &eqn_id);

  std::vector<PREFUNC> *preconfunc_;

  typedef double (*INITFUNC)(const double &x,
			     const double &y,
			     const double &z,
			     const int &eqn_id);

  std::vector<INITFUNC> *initfunc_;

  std::vector<std::string> *varnames_;

  typedef double (*DBCFUNC)(const double &x,
			    const double &y,
			    const double &z,
			    const double &t);

  std::vector<std::map<int,DBCFUNC>> *dirichletfunc_;

  typedef double (*NBCFUNC)(const Basis *basis,
			    const int &i, 
			    const double &dt_, 
			    const double &t_theta_,
			    const double &time);

  std::vector<std::map<int,NBCFUNC>> *neumannfunc_;

  typedef void (*PARAMFUNC)(Teuchos::ParameterList *plist);

  PARAMFUNC paramfunc_;

#ifdef PERIODIC_BC
  boost::ptr_vector<boost::ptr_vector<periodic_bc>> periodic_bc_;
  //std::vector<periodic_bc> periodic_bc_;
#else
  //cn std::vector<std::map<int,std::pair<int,int>>> *periodicbc_; is probably better here
  std::vector<std::vector<std::pair<int,int>>> *periodicbc_;
#endif
  
  //post process stuff
  //cn need this to be a function of all variables eventually
//   std::vector<double (*)(const double *u, const double *gradu)> *postprocfunc_;
  void postprocess();
//   int numpostprocvar_;
//   std::vector<std::string> *postprocvarnames_;
//   Teuchos::RCP<Epetra_Vector> u_postproc_;

  //tip velocity stuff
  std::map<double,int> x_node;
  void init_vtip();
  void find_vtip();
  void find_vtip_x();
  void finalize_vtip();
  double vtip_x_,vtip_x_old_;

  RCP<Teuchos::Time> ts_time_import;
  RCP<Teuchos::Time> ts_time_resfill;
  RCP<Teuchos::Time> ts_time_precfill;
  RCP<Teuchos::Time> ts_time_nsolve;
  boost::ptr_vector<error_estimator> Error_est;
  boost::ptr_vector<post_process> post_proc;
  Teuchos::RCP<elem_color> Elem_col;

  void set_basis( boost::ptr_vector<Basis> &basis, const std::string elem_type) const;

  double *testptr;

  void write_openmp();

  void copy_mesh_gpu();
  void delete_mesh_gpu();
  //void copy_uold_gpu(RCP< Epetra_Vector> uold, RCP< Epetra_Vector> uoldold);
  void copy_uold_gpu(RCP< Epetra_Vector> uold);

  int * meshc;
  int * meshn;
  double * meshx;
  double * meshy;
  //double * ua;
  double * u_olda;
  int clen, nlen, xlen;
  int alen;

};


//==================================================================
#include "ModelEvaluatorNEMESIS_def.hpp"
//==================================================================


#endif
