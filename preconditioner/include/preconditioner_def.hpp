

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
    //MLList.set("coarse: max size",(int)128);
    MLList.set("cycle applications",(int)2);
//     MLList.set("prec type","full-MGV");
//     MLList.set("smoother: type","Chebyshev");
    MLList.set("smoother: type","Jacobi");
    MLList.set("smoother: sweeps",(int)2); 
//     MLList.set("smoother: damping factor", 1.0);

//    MLList.set("coarse: type","Chebyshev");
//    MLList.set("coarse: type","Jacobi"); 
//    MLList.set("coarse: sweeps",2);  
//     MLList.set("coarse: damping factor", 1.0);
    
//     MLList.set("ML output",10);

     //W_->Print(std::cout);
    MLPrec_ =  new ML_Epetra::MultiLevelPreconditioner(*W_, MLList,false);
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
    //Xe->Print(std::cout);
    //MLPrec_->ApplyInverse(*Xe, *Ye);
    Apply(*Xe, *Ye);
    //Ye->Print(std::cout);
    //Ye->Print(std::cout);
    assign(Y,*create_MultiVector (Ye, range_));

  } ;
template<class Scalar>
int preconditioner<Scalar>::Apply (const Epetra_MultiVector &X, Epetra_MultiVector &Y) const{
  int err = MLPrec_->ApplyInverse(X, Y);
  //std::cout<<"MLPrec_->ApplyInverse(X, Y) = "<<err<<std::endl;
  return err;
};
template<class Scalar>
int preconditioner<Scalar>::ReComputePreconditioner () const
{
  if (0 == MLPrec_->IsPreconditionerComputed () ) MLPrec_->ComputePreconditioner();
  int err = MLPrec_->ReComputePreconditioner();

  //std::cout<<"MLPrec_->ReComputePreconditioner() = "<<err<<std::endl;
  //std::cout<<" one norm W_ = "<<W_->NormOne()<<std::endl<<" inf norm W_ = "<<W_->NormInf()<<std::endl<<" fro norm W_ = "<<W_->NormFrobenius()<<std::endl;
  //MLPrec_->AnalyzeHierarchy (true, 1,1,1);
  //MLPrec_->TestSmoothers();
  //exit(0);

  return err;
};
template<class Scalar>
int preconditioner<Scalar>::ComputePreconditioner () const
{
  int err = MLPrec_->ComputePreconditioner();
  return err;
};
