//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef NOX_THYRA_MODEL_EVALUATOR_HEAT_DECL_HPP
#define NOX_THYRA_MODEL_EVALUATOR_HEAT_DECL_HPP

#include "Thyra_StateFuncModelEvaluatorBase.hpp"

#include "Mesh.h"
#include "preconditioner.hpp"
#include "timestep.hpp"
template<class Scalar> class ModelEvaluatorHEAT;

template<class Scalar>
Teuchos::RCP<ModelEvaluatorHEAT<Scalar> >
modelEvaluatorHEAT(const Teuchos::RCP<const Epetra_Comm>& comm,
            Mesh &mesh,
		   Teuchos::ParameterList plist);

/// DEPRECATED
template<class Scalar>
class ModelEvaluatorHEAT
  : public ::timestep<Scalar>, public ::Thyra::StateFuncModelEvaluatorBase<Scalar>
{
public:

  ModelEvaluatorHEAT(const Teuchos::RCP<const Epetra_Comm>& comm,
              Mesh *mesh,
		   Teuchos::ParameterList plist);
  ~ModelEvaluatorHEAT();

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

  Teuchos::ParameterList paramList;
  double  dt_;

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

  double time_;

};



//==================================================================
#include "ModelEvaluatorHEAT_def.hpp"
//==================================================================

#endif
