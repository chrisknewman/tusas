#ifndef NOX_THYRA_MODEL_EVALUATOR_NEMESIS_DECL_HPP
#define NOX_THYRA_MODEL_EVALUATOR_NEMESIS_DECL_HPP

#include "Thyra_StateFuncModelEvaluatorBase.hpp"	
#include "Teuchos_ParameterList.hpp"
#include "Epetra_FECrsGraph.h"

#include "Mesh.h"
#include "preconditioner.hpp"
#include "timestep.hpp"

#include <boost/ptr_container/ptr_vector.hpp>
template<class Scalar> class ModelEvaluatorNEMESIS;

template<class Scalar>
Teuchos::RCP<ModelEvaluatorNEMESIS<Scalar> >
modelEvaluatorNEMESIS(const Teuchos::RCP<const Epetra_Comm>& comm,
			 Mesh &mesh,
			 Teuchos::ParameterList plist 
			 );

template<class Scalar>
class ModelEvaluatorNEMESIS
  : public ::timestep<Scalar>, public ::Thyra::StateFuncModelEvaluatorBase<Scalar>
{
public:

  ModelEvaluatorNEMESIS(const Teuchos::RCP<const Epetra_Comm>& comm,
			   Mesh *mesh,
			   Teuchos::ParameterList plist 
			   );
  ~ModelEvaluatorNEMESIS();

  /** \name Initializers/Accessors */
  //@{

  /** \brief . */
  void set_x0(const Teuchos::ArrayView<const Scalar> &x0);

  /** \brief . */
  void setShowGetInvalidArgs(bool showGetInvalidArg);

  void set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory);

  //@}

  /** \name Public functions overridden from ModelEvaulator. */
  //@{

  /** \brief . */
  Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > get_x_space() const;
  /** \brief . */
  Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > get_f_space() const;
  /** \brief . */
  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> getNominalValues() const;
  /** \brief . */
  Teuchos::RCP< ::Thyra::LinearOpBase<Scalar> > create_W_op() const;
  /** \brief . */
  Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> > get_W_factory() const;
  /** \brief . */
  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> createInArgs() const;
  /** \brief . */
  Teuchos::RCP< ::Thyra::PreconditionerBase< Scalar > > create_W_prec() const;
  //@}

  void init_nox();
  void initialize();
  void finalize();
  void advance();
  void compute_error( double *u);
  //void write_exodus(const int output_step);
  void write_exodus();

  void write_matlab();

  void restart(Teuchos::RCP<Epetra_Vector> u,Teuchos::RCP<Epetra_Vector> u_old);

private:

  /** Allocates and returns the Jacobian matrix graph */
  virtual Teuchos::RCP<Epetra_FECrsGraph> createGraph();

  /** \name Private functions overridden from ModelEvaulatorDefaultBase. */
  //@{

  /** \brief . */
  ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> createOutArgsImpl() const;
  /** \brief . */
  void evalModelImpl(
    const ::Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
    const ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs
    ) const;

  //@}

private: // data members

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


  std::vector<double (*)(const boost::ptr_vector<Basis> &basis, 
			 const int &i, 
			 const double &dt_, 
			 const double &t_theta_, 
			 const double &delta, 
			 const double &time)> *residualfunc_;


  std::vector<double (*)(const boost::ptr_vector<Basis> &basis, 
			 const int &i,  
			 const int &j,
			 const double &dt_, 
			 const double &t_theta_, 
			 const double &delta)> *preconfunc_;

  std::vector<double (*)(const double &x,
			 const double &y,
			 const double &z)> *initfunc_;

  std::vector<std::string> *varnames_;

  std::vector<std::map<int,double (*)(const double &x,
				      const double &y,
				      const double &z)>> *dirichletfunc_;

  std::map<double,int> x_node;
  void init_vtip();
  void find_vtip();
  void find_vtip_x();
  void finalize_vtip();
  double vtip_x_,vtip_x_old_;

};



//==================================================================
#include "ModelEvaluatorNEMESIS_def.hpp"
//==================================================================


#endif
