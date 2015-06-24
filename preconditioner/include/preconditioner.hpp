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

template<class Scalar>
class preconditioner : virtual public Thyra::LinearOpBase< Scalar >
{
public:

  //cn this will be a base class for an epetra preconditioner,
  //cn that gives an epetra interface
  //cn subclasses will fill in the details and fill in an apply method
  //cn domain, range and map correspond to the nox space
   




  preconditioner(const RCP<Epetra_CrsMatrix>& W,const Teuchos::RCP<const Epetra_Comm>&  comm,
			 Teuchos::ParameterList MLList );
  ~preconditioner();

  RCP< const VectorSpaceBase<Scalar> > range() const{//cn could be ModelEvalaluator::get_f_space()
    //std::cout<<"range()"<<std::endl;
    return range_;};
  RCP< const VectorSpaceBase<Scalar> > domain() const{//cn could be ModelEvalaluator::get_x_space()
    //std::cout<<"domain()"<<std::endl;
    return domain_;
  };
  RCP< const LinearOpBase<Scalar> > clone() const{std::cout<<"clone()"<<std::endl;
  return Teuchos::null;};

  bool opSupportedImpl(EOpTransp M_trans) const{std::cout<<"opSupportedImpl"<<std::endl; return false;};

  void applyImpl(
    const EOpTransp M_trans,
    const MultiVectorBase<Scalar> &X,
    const Ptr<MultiVectorBase<Scalar> > &Y,
    const Scalar alpha,
    const Scalar beta
    ) const ;
  int Apply (const Epetra_MultiVector &X, Epetra_MultiVector &Y) const;//cn this will be virtual 
  int ReComputePreconditioner () const;
  int ComputePreconditioner () const;
private:
  RCP< Epetra_CrsMatrix> W_;
  RCP< const VectorSpaceBase<Scalar> > range_, domain_;
  ML_Epetra::MultiLevelPreconditioner *MLPrec_;
  Teuchos::RCP< const Epetra_Comm > comm_;
  Teuchos::RCP< const Epetra_Map > map_;
};

#include "preconditioner_def.hpp"

#endif
