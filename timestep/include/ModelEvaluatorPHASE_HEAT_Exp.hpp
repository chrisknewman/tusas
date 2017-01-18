//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef NOX_THYRA_MODEL_EVALUATOR_PHASE_HEAT_Exp_DECL_HPP
#define NOX_THYRA_MODEL_EVALUATOR_PHASE_HEAT_Exp_DECL_HPP

#include "Thyra_StateFuncModelEvaluatorBase.hpp"

#include "Mesh.h"
#include "preconditioner.hpp"
#include "timestep.hpp"
template<class Scalar> class ModelEvaluatorPHASE_HEAT_Exp;

template<class Scalar>
Teuchos::RCP<ModelEvaluatorPHASE_HEAT_Exp<Scalar> >
modelEvaluatorPHASE_HEAT_Exp(const Teuchos::RCP<const Epetra_Comm>& comm,
            Mesh &mesh,
            const Scalar dt);

/// DEPRECATED
template<class Scalar>
class ModelEvaluatorPHASE_HEAT_Exp
  : public ::timestep<Scalar>, public ::Thyra::StateFuncModelEvaluatorBase<Scalar>
{
public:

  ModelEvaluatorPHASE_HEAT_Exp(const Teuchos::RCP<const Epetra_Comm>& comm,
              Mesh *mesh,
              const Scalar dt);
  ~ModelEvaluatorPHASE_HEAT_Exp();

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

private:

  /** Allocates and returns the Jacobian matrix graph */
  virtual Teuchos::RCP<Epetra_CrsGraph> createGraph();

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

  const Scalar  dt_;

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
  Teuchos::RCP<Epetra_Vector> dudt_;

  double time_;

  int numeqs_;

  double K_, T_m_, T_inf_, alpha_, M_, eps_;

  const double gs(const double &theta);
  double gs2(const double &theta) const;
  double dgs2_2dtheta(const double &theta) const;
  const double R(const double &theta);
  double theta(double &x,double &y) const;



};



//==================================================================
#include "ModelEvaluatorPHASE_HEAT_Exp_def.hpp"
//==================================================================

#endif
