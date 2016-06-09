

template<class Scalar>
preconditioner<Scalar>::preconditioner(const RCP<Epetra_CrsMatrix>& W,const Teuchos::RCP<const Epetra_Comm>&  comm,
			 Teuchos::ParameterList MLList ){
    W_=W;
    comm_ = comm;
    const Teuchos::RCP<Thyra::LinearOpBase< Scalar > > W_op =
      Thyra::nonconstEpetraLinearOp(W_);
    //std::cout<<"preconditioner()"<<std::endl;;
    range_ = W_op->range();
    domain_ = W_op->domain(); 
    map_ =  Teuchos::rcp(new Epetra_Map(*get_Epetra_Map(*domain_, comm_)));
    //map_ =  Teuchos::rcp(new Epetra_Map(W_->DomainMap () ));


    //cn
    //cn  I could see block jacobi working in the following way:
    //cn we would need to have an Teuchos::RCP< Epetra_MultiVector > that points
    //cn thatpoints to previous iterate
    //cn gets zeroed out in ReComputePreconditioner ()
    //cn would also need to have access to evaluation on off diagonal blocks
    //cn (in a vector way not neccessarily in)




    MLPrec_ =  new ML_Epetra::MultiLevelPreconditioner(*W_, MLList,false);
    if( 0 == comm->MyPID() ){
      std::cout<<std::endl<<"Creating ML preconditioner with:"<<std::endl;
      std::cout<<MLList<<std::endl<<std::endl;
    }
    //exit(0);
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
template<class Scalar>
preconditioner<Scalar>::~preconditioner ()
{
  delete MLPrec_;
};
