//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef PRECONDITIONER_HPP
#define PRECONDITIONER_HPP

#include <Thyra_LinearOpBase_decl.hpp>
#include <Thyra_LinearOpBase_def.hpp>
#include <Thyra_LinearOpBase.hpp>
#include "Thyra_OperatorVectorTypes.hpp"

#include "Teuchos_RCP.hpp"

#include <ml_MultiLevelPreconditioner.h>

#include <Epetra_Comm.h>

//using namespace Teuchos;

using namespace Thyra;

//using Teuchos::RCP;

/// Preconditioner class.
/** Implements an ML preconditioning object based on a given matrix and parameter list. */
template<class Scalar>
class preconditioner : virtual public Thyra::LinearOpBase< Scalar >
{
public:

  //cn this will be a base class for an epetra preconditioner,
  //cn that gives an epetra interface
  //cn subclasses will fill in the details and fill in an apply method
  //cn domain, range and map correspond to the nox space
   



  /// Constructor
  /** Create the preconditioning object given RCP<Epetra_CrsMatrix>& W and Teuchos::ParameterList MLList. */
  preconditioner(const RCP<Epetra_CrsMatrix>& W, ///< preconditioning matrix
		 const Teuchos::RCP<const Epetra_Comm>&  comm,  ///< MPI communicator
		 Teuchos::ParameterList MLList   ///< Parameter list
		 );
  /// Destructor
  ~preconditioner();
  /// Required for Thyra::LinearOpBase< Scalar >
  RCP< const VectorSpaceBase<Scalar> > range() const{//cn could be ModelEvalaluator::get_f_space()
    //std::cout<<"range()"<<std::endl;
    return range_;};
  /// Required for Thyra::LinearOpBase< Scalar >
  RCP< const VectorSpaceBase<Scalar> > domain() const{//cn could be ModelEvalaluator::get_x_space()
    //std::cout<<"domain()"<<std::endl;
    return domain_;
  };
  /// Required for Thyra::LinearOpBase< Scalar >
  RCP< const LinearOpBase<Scalar> > clone() const{std::cout<<"clone()"<<std::endl;
  return Teuchos::null;};

  /// Required for Thyra::LinearOpBase< Scalar >
  bool opSupportedImpl(EOpTransp M_trans) const{std::cout<<"opSupportedImpl"<<std::endl; return false;};

  /// Required for Thyra::LinearOpBase< Scalar >
  /** Wrapper for Apply (const Epetra_MultiVector &X, Epetra_MultiVector &Y). */
  void applyImpl(
		 const EOpTransp M_trans, ///< not used
    const MultiVectorBase<Scalar> &X, ///< input vector
    const Ptr<MultiVectorBase<Scalar> > &Y, ///< output vector
    const Scalar alpha, ///< not used
    const Scalar beta ///< not used
    ) const ;
  /// Required for Thyra::LinearOpBase< Scalar >
  /** This the function that applies the preconditioner. X = M^-1 Y*/
  int Apply (const Epetra_MultiVector &X, ///< input vector
	     Epetra_MultiVector &Y ///< output vector
	     ) const;//cn this will be virtual 
  /// Recompute the ML hierarchy.
  int ReComputePreconditioner () const;
  /// Initially compute  the ML hierarchy
  int ComputePreconditioner () const;
private:
  /// The preconditioning matrix object.
  RCP< Epetra_CrsMatrix> W_;
  /// Required for Thyra::LinearOpBase< Scalar >
  RCP< const VectorSpaceBase<Scalar> > range_;
  /// Required for Thyra::LinearOpBase< Scalar >
  RCP< const VectorSpaceBase<Scalar> > domain_;
  /// The ML object.
  ML_Epetra::MultiLevelPreconditioner *MLPrec_;
  /// MPI comm object.
  Teuchos::RCP< const Epetra_Comm > comm_;
  /// Epetra_Map object.
  Teuchos::RCP< const Epetra_Map > map_;
};

#include "preconditioner_def.hpp"

#endif
