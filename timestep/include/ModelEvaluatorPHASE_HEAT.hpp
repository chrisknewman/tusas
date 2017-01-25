//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef NOX_THYRA_MODEL_EVALUATOR_PHASE_HEAT_DECL_HPP
#define NOX_THYRA_MODEL_EVALUATOR_PHASE_HEAT_DECL_HPP

#include "Thyra_StateFuncModelEvaluatorBase.hpp"	
#include "Teuchos_ParameterList.hpp"

#include "Mesh.h"
#include "preconditioner.hpp"
#include "timestep.hpp"
template<class Scalar> class ModelEvaluatorPHASE_HEAT;

template<class Scalar>
Teuchos::RCP<ModelEvaluatorPHASE_HEAT<Scalar> >
modelEvaluatorPHASE_HEAT(const Teuchos::RCP<const Epetra_Comm>& comm,
			 Mesh &mesh,
			 Teuchos::ParameterList plist 
			 );

/// DEPRECATED
template<class Scalar>
class ModelEvaluatorPHASE_HEAT
  : public ::timestep<Scalar>, public ::Thyra::StateFuncModelEvaluatorBase<Scalar>
{
public:

  ModelEvaluatorPHASE_HEAT(const Teuchos::RCP<const Epetra_Comm>& comm,
			   Mesh *mesh,
			   Teuchos::ParameterList plist 
			   );
  ~ModelEvaluatorPHASE_HEAT();

  void set_x0(const Teuchos::ArrayView<const Scalar> &x0);

  void setShowGetInvalidArgs(bool showGetInvalidArg);

  void set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory);

  Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > get_x_space() const;

  Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > get_f_space() const;

  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> getNominalValues() const;

  Teuchos::RCP< ::Thyra::LinearOpBase<Scalar> > create_W_op() const;

  Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> > get_W_factory() const;

  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> createInArgs() const;

  Teuchos::RCP< ::Thyra::PreconditionerBase< Scalar > > create_W_prec() const;


  void init_nox();
  void initialize();
  void finalize();
  void advance();
  void compute_error( double *u);
  //static double gs2(const double &theta);
  void write_exodus(){};


private:

  virtual Teuchos::RCP<Epetra_CrsGraph> createGraph();

  ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> createOutArgsImpl() const;

  void evalModelImpl(
    const ::Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
    const ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs
    ) const;

private: // data members

  const Teuchos::RCP<const Epetra_Comm>  comm_;

  Mesh *mesh_;

  //const Scalar  dt_;
  double dt_;

  Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > x_space_;
  Teuchos::RCP<const Epetra_Map>   x_owned_map_;

  Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > f_space_;
  Teuchos::RCP<const Epetra_Map>   f_owned_map_;

  Teuchos::RCP<Epetra_CrsGraph>  W_graph_;

  Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> > W_factory_;

  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> nominalValues_;
  Teuchos::RCP< ::Thyra::VectorBase<Scalar> > x0_;
  bool showGetInvalidArg_;
  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> prototypeInArgs_;
  ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> prototypeOutArgs_;
  Teuchos::RCP<Epetra_CrsMatrix> P_;
  Teuchos::RCP<preconditioner <Scalar> > prec_;
  Teuchos::RCP<NOX::Solver::Generic> solver_;
  Teuchos::RCP<Epetra_Vector> u_old_;
  Teuchos::RCP<Epetra_Vector> u_old_old_;
  Teuchos::RCP<Epetra_Vector> dudt_;

  double time_;

  int ex_id_;

  int numeqs_;

  //cn these are parameters for cummins
  double D_, T_m_, T_inf_, alpha_, M_, eps_,eps_0_, R_0_;
  double phi_sol_, phi_liq_;

  double random_number_, random_number_old_;
  Teuchos::RCP<Epetra_Vector> random_vector_;
  Teuchos::RCP<Epetra_Vector> random_vector_old_;

  const double gs(const double &theta);
  const double R(const double &theta);
  double theta(double &x,double &y,double &z = 0) const;

  void init(Teuchos::RCP<Epetra_Vector> u);
  void init_square(Teuchos::RCP<Epetra_Vector> u);
  void init_karma(Teuchos::RCP<Epetra_Vector> u);
  void multi(Teuchos::RCP<Epetra_Vector> u);
  void pool(Teuchos::RCP<Epetra_Vector> u);

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

  std::map<double,int> x_node;
  void init_vtip();
  void find_vtip();
  void find_vtip_x();
  void finalize_vtip();
  double vtip_x_,vtip_x_old_;

};



//==================================================================
#include "ModelEvaluatorPHASE_HEAT_def.hpp"
//==================================================================


#endif
