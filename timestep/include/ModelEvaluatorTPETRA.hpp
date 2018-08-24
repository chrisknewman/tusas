//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef NOX_THYRA_MODEL_EVALUATOR_TPETRA_DECL_HPP
#define NOX_THYRA_MODEL_EVALUATOR_TPETRA_DECL_HPP

#include <Teuchos_Comm.hpp>

#include <Tpetra_Vector.hpp>
#include "Thyra_StateFuncModelEvaluatorBase.hpp"

#include "Mesh.h"

template<class Scalar> class ModelEvaluatorTPETRA;

template<class Scalar>
Teuchos::RCP<ModelEvaluatorTPETRA<Scalar> >
modelEvaluatorTPETRA( Mesh &mesh,
			 Teuchos::ParameterList plist 
			 );
/// Implentation of timestep with MPI and OpenMP support.
template<class Scalar>
class ModelEvaluatorTPETRA
  : public ::timestep<Scalar>, public ::Thyra::StateFuncModelEvaluatorBase<Scalar>
{
public:
  /// Constructor
  ModelEvaluatorTPETRA( Mesh *mesh,
			   Teuchos::ParameterList plist 
		       );
  /// Destructor
  ~ModelEvaluatorTPETRA(){};

  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > get_x_space() const{return x_space_;};
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > get_f_space() const{return f_space_;};
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> createInArgs() const{return prototypeInArgs_;};
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  void set_x0(const Teuchos::ArrayView<const Scalar> &x0);
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  ::Thyra::ModelEvaluatorBase::InArgs<Scalar> getNominalValues() const{return nominalValues_;};

  void initialize(){};
  void finalize(){};
  void advance(){};
  void write_exodus(){};

private:

  typedef Tpetra::Vector<>::global_ordinal_type global_ordinal_type;
  typedef Tpetra::Vector<>::local_ordinal_type local_ordinal_type;
  typedef Tpetra::global_size_t global_size_t;
  typedef Tpetra::Vector<>::node_type node_type;

  ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> createOutArgsImpl() const{};

  void evalModelImpl(
    const ::Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
    const ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs
    ) const;

  //const Teuchos::RCP<const Teuchos::Comm<int> > comm_;
  Teuchos::RCP<Mesh> mesh_;
  int numeqs_;
  int num_owned_nodes_;
  int num_overlap_nodes_;

  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > x_space_;
  Teuchos::RCP<const Tpetra::Map<> > x_overlap_map_;
  Teuchos::RCP<const Tpetra::Map<> > x_owned_map_;

  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > f_space_;

  Teuchos::RCP<const Tpetra::Import<> > importer_;

  double dt_;
  double t_theta_;
  Teuchos::ParameterList paramList;

  Thyra::ModelEvaluatorBase::InArgs<Scalar> nominalValues_;
  //Teuchos::RCP< ::Thyra::VectorBase<Scalar> > x0_;
  Teuchos::RCP<Tpetra::Vector<Scalar,int> > x0_;
  Thyra::ModelEvaluatorBase::InArgs<Scalar> prototypeInArgs_;
  Thyra::ModelEvaluatorBase::OutArgs<Scalar> prototypeOutArgs_;
};

//==================================================================
#include "ModelEvaluatorTPETRA_def.hpp"
//==================================================================

#endif
