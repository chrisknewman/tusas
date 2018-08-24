//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////


#ifndef NOX_THYRA_MODEL_EVALUATOR_TPETRA_DEF_HPP
#define NOX_THYRA_MODEL_EVALUATOR_TPETRA_DEF_HPP

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_ArrayViewDecl.hpp>

//#include <Kokkos_View.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Import.hpp>

#include <Thyra_TpetraThyraWrappers.hpp>

template<class Scalar>
Teuchos::RCP<ModelEvaluatorTPETRA<Scalar> >
modelEvaluatorTPETRA( Mesh *mesh,
			 Teuchos::ParameterList plist
			 )
{
  return Teuchos::rcp(new ModelEvaluatorTPETRA<Scalar>(mesh,plist));
}

// Constructor

template<class Scalar>
ModelEvaluatorTPETRA<Scalar>::
ModelEvaluatorTPETRA( Mesh *mesh,
			 Teuchos::ParameterList plist 
		     ) :
  paramList(plist)
{
  dt_ = paramList.get<double> (TusasdtNameString);
  t_theta_ = paramList.get<double> (TusasthetaNameString);

  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
  //comm_->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
  
  mesh_ = Teuchos::rcp(new Mesh(*mesh));
  mesh_->compute_nodal_adj(); 
  std::vector<int> node_num_map(mesh_->get_node_num_map());
  numeqs_ = 1;
  std::vector<int> my_global_nodes(numeqs_*node_num_map.size());
  for(int i = 0; i < node_num_map.size(); i++){    
    for( int k = 0; k < numeqs_; k++ ){
      my_global_nodes[numeqs_*i+k] = numeqs_*node_num_map[i]+k;
    }
  }

  const global_size_t numGlobalEntries = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  const global_ordinal_type indexBase = 0;

  Teuchos::ArrayView<int> AV(my_global_nodes);

  x_overlap_map_ = Teuchos::rcp(new Tpetra::Map<>(numGlobalEntries,
						  AV,
						  indexBase,
						  comm_
						  ));

  x_overlap_map_ ->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_DEFAULT );

  x_owned_map_ = Teuchos::rcp(new Tpetra::Map<>(*(createOneToOne(x_overlap_map_))));
  x_owned_map_ ->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_DEFAULT );
  
  importer_ = Teuchos::rcp(new Tpetra::Import<>(x_overlap_map_, x_owned_map_));
  
  num_owned_nodes_ = x_owned_map_->getNodeNumElements()/numeqs_;
  num_overlap_nodes_ = x_overlap_map_->getNodeNumElements()/numeqs_;

  x_space_ = Thyra::createVectorSpace<Scalar>(x_owned_map_);
  f_space_ = x_space_;
  //x0_ = Thyra::createMember(x_space_);
  x0_ = Teuchos::rcp(new Tpetra::Vector<Scalar, int>(x_owned_map_));
  x0_->putScalar(Teuchos::ScalarTraits<Scalar>::zero());
  //V_S(x0_.ptr(), Teuchos::ScalarTraits<Scalar>::zero());  //cn not sure what this is

  Thyra::ModelEvaluatorBase::InArgsSetup<Scalar> inArgs;
  inArgs.setModelEvalDescription(this->description());
  inArgs.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x);
  prototypeInArgs_ = inArgs;

  Thyra::ModelEvaluatorBase::OutArgsSetup<Scalar> outArgs;
  outArgs.setModelEvalDescription(this->description());
  outArgs.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_f);
  outArgs.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_W_prec);
  prototypeOutArgs_ = outArgs;
  nominalValues_ = inArgs;
  //nominalValues_.set_x(x0_);;
  nominalValues_.set_x(Thyra::createVector(x0_, x_space_));

}

template<class Scalar>
void ModelEvaluatorTPETRA<Scalar>::set_x0(const Teuchos::ArrayView<const Scalar> &x0_in)
{
#if 0
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_EQUALITY(x_space_->dim(), x0_in.size());
#endif
  Thyra::DetachedVectorView<Scalar> x0(x0_);
  x0.sv().values()().assign(x0_in);
#endif
#ifdef TEUCHOS_DEBUG
  TEUCHOS_ASSERT_EQUALITY(x_space_->dim(), x0_in.size());
#endif
  x0_->get1dViewNonConst()().assign(x0_in);
}

template<class Scalar>
void ModelEvaluatorTPETRA<Scalar>::evalModelImpl(
  const Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
  const Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs
  ) const
{  
  typedef Thyra::TpetraOperatorVectorExtraction<Scalar,int> ConverterT;

  const RCP<const Tpetra::Vector<Scalar, int> > x_vec =
    ConverterT::getConstTpetraVector(inArgs.get_x());
  const ArrayRCP<const Scalar> x = x_vec->get1dView();

  const RCP<Tpetra::Vector<Scalar, int> > f_vec =
    ConverterT::getTpetraVector(outArgs.get_f());
#if 0
  const RCP<Tpetra::CrsMatrix<Scalar, int> > W =
    rcp_dynamic_cast<Tpetra::CrsMatrix<Scalar,int> >(
      ConverterT::getTpetraOperator(outArgs.get_W_op()),
      true
      );
#endif

  if (nonnull(f_vec)) {
    const ArrayRCP<Scalar> f = f_vec->get1dViewNonConst();
//     f[0] = x[0] + x[1]*x[1] - p_[0];
//     f[1] = d_ * (x[0]*x[0] -x[1] - p_[1]);
  }
#if 0
  if (nonnull(W)) {
    W->setAllToScalar(ST::zero());
    W->sumIntoGlobalValues(0, tuple<int>(0, 1), tuple<Scalar>(1.0, 2.0*x[1]));
    W->sumIntoGlobalValues(1, tuple<int>(0, 1), tuple<Scalar>(2.0*d_*x[0], -d_));
  }
#endif

}



#endif
