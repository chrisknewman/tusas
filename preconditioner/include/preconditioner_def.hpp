

template<class Scalar>
preconditioner<Scalar>::preconditioner(const RCP<Epetra_CrsMatrix>& W,const Teuchos::RCP<const Epetra_Comm>&  comm ){
    W_=W;
    comm_ = comm;
    const Teuchos::RCP<Thyra::LinearOpBase< Scalar > > W_op =
      Thyra::nonconstEpetraLinearOp(W_);
    //std::cout<<"preconditioner()"<<std::endl;;
    range_ = W_op->range();
    domain_ = W_op->domain(); 
    map_ =  Teuchos::rcp(new Epetra_Map(*get_Epetra_Map(*domain_, comm_)));


    Teuchos::ParameterList MLList;
    ML_Epetra::SetDefaults("SA",MLList);
//     MLList.set("smoother: sweeps",(int)5);
     MLList.set("cycle applications",(int)2);
//     MLList.set("prec type","full-MGV");
//     //MLList.set("smoother: type","Chebyshev");

//    MLList.set("coarse: type","Chebyshev");
     MLList.set("coarse: type","Jacobi"); 
     MLList.set("coarse: sweeps",2);  
     MLList.set("coarse: damping factor", 1.0);
    
//     MLList.set("cycle applications",(int)1); 
//     MLList.set("smoother: sweeps",(int)2); 
//     MLList.set("smoother: damping factor", 1.0);
//    MLList.set("ML output",10);


    MLPrec_ =  new ML_Epetra::MultiLevelPreconditioner(*W_, MLList,true);
  };

template<class Scalar>
  void preconditioner<Scalar>::applyImpl(
    const EOpTransp M_trans,
    const MultiVectorBase<Scalar> &X,
    const Ptr<MultiVectorBase<Scalar> > &Y,
    const Scalar alpha,
    const Scalar beta
    ) const
  {
    //std::cout<<"applyImpl"<<std::endl;
    //assign(Y,X);
    //W_->Print(std::cout);

    Teuchos::RCP< Epetra_MultiVector > Xe = Teuchos::rcp(new Epetra_MultiVector(*get_Epetra_MultiVector (*map_, X)));
    Teuchos::RCP< Epetra_MultiVector > Ye = Teuchos::rcp(new Epetra_MultiVector(*map_,(int)1));
    //MLPrec_->ApplyInverse(*Xe, *Ye);
    Apply(*Xe, *Ye);
    //Ye->Print(std::cout);
    assign(Y,*create_MultiVector (Ye, range_));

  } ;
template<class Scalar>
int preconditioner<Scalar>::Apply (const Epetra_MultiVector &X, Epetra_MultiVector &Y) const{
  MLPrec_->ApplyInverse(X, Y);
};
template<class Scalar>
int preconditioner<Scalar>::ReComputePreconditioner () const
{
  MLPrec_->ReComputePreconditioner();
};
