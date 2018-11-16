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
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Export.hpp>

#include "Thyra_StateFuncModelEvaluatorBase.hpp"

#include "Mesh.h"
#include "elem_color.h"

template<class Scalar> class ModelEvaluatorTPETRA;

template<class Scalar>
Teuchos::RCP<ModelEvaluatorTPETRA<Scalar> >
modelEvaluatorTPETRA( const Teuchos::RCP<const Epetra_Comm>& comm,
			Mesh &mesh,
			 Teuchos::ParameterList plist 
			 );
/// Implentation of timestep with MPI and OpenMP support.
template<class Scalar>
class ModelEvaluatorTPETRA
  : public ::timestep<Scalar>, public ::Thyra::StateFuncModelEvaluatorBase<Scalar>
{
public:
  /// Constructor
  ModelEvaluatorTPETRA( const Teuchos::RCP<const Epetra_Comm>& comm,
			Mesh *mesh,
			   Teuchos::ParameterList plist 
		       );
  /// Destructor
  ~ModelEvaluatorTPETRA(){};

  typedef Tpetra::Vector<>::scalar_type scalar_type;

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

  void initialize();
  void finalize();
  void advance();
  void write_exodus();

  void evalModelImpl(
		     const ::Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
		     const ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs
		     ) const;

private:

  typedef Tpetra::Vector<>::global_ordinal_type global_ordinal_type;
  typedef Tpetra::Vector<>::local_ordinal_type local_ordinal_type;
  typedef Tpetra::global_size_t global_size_t;
  typedef Tpetra::Vector<>::node_type node_type;
  typedef Tpetra::Vector<scalar_type, local_ordinal_type,
			 global_ordinal_type, node_type> vector_type;
  typedef Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type> map_type;
  typedef Tpetra::Import<local_ordinal_type, global_ordinal_type,
                         node_type> import_type;
  typedef Tpetra::Export<local_ordinal_type, global_ordinal_type,
                         node_type> export_type;

  ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> createOutArgsImpl() const;


//Teuchos::RCP<Mesh> mesh_;
  Mesh* mesh_;

  int update_mesh_data();

  void set_test_case();

  double time_;

  int ex_id_;

  int output_step_;
  int numeqs_;
  int num_owned_nodes_;
  int num_overlap_nodes_;

  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > x_space_;
  Teuchos::RCP<const map_type > x_overlap_map_;
  Teuchos::RCP<const map_type > x_owned_map_;
  Teuchos::RCP<const map_type > node_overlap_map_;

  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > f_space_;

  Teuchos::RCP<const import_type > importer_;
  Teuchos::RCP<const export_type > exporter_;
  Teuchos::RCP<NOX::Solver::Generic> solver_;

  Teuchos::RCP<vector_type> u_old_;

  Teuchos::RCP<vector_type> x_;
  Teuchos::RCP<vector_type> y_;
  Teuchos::RCP<vector_type> z_;

  int nnewt_;
  double dt_;
  double t_theta_;
  Teuchos::ParameterList paramList;

  Thyra::ModelEvaluatorBase::InArgs<Scalar> nominalValues_;
  //Teuchos::RCP< ::Thyra::VectorBase<scalar_type> > x0_;
  Teuchos::RCP<vector_type > x0_;
  Thyra::ModelEvaluatorBase::InArgs<Scalar> prototypeInArgs_;
  Thyra::ModelEvaluatorBase::OutArgs<Scalar> prototypeOutArgs_;

  /// Initialize and create the NOX and linear solvers.
  void init_nox();
  /// Satisfy Thyra::StateFuncModelEvaluatorBase interface
  void set_W_factory(const Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory);
 
  Teuchos::RCP<const ::Thyra::LinearOpWithSolveFactoryBase<Scalar> > W_factory_;

  void init(Teuchos::RCP<vector_type> u);

  std::vector<std::string> *varnames_;

  typedef double (*RESFUNC)(const GPUBasis *basis, 
			    const int &i, 
			    const double &dt_, 
			    const double &t_theta_, 
			    const double &time,
			    const int &eqn_id);

  typedef double (*DBCFUNC)(const double &x,
			    const double &y,
			    const double &z,
			    const double &t);

  std::vector<std::map<int,DBCFUNC>> *dirichletfunc_;

  typedef double (*INITFUNC)(const double &x,
			     const double &y,
			     const double &z,
			     const int &eqn_id);

  std::vector<INITFUNC> *initfunc_;

  RCP<Teuchos::Time> ts_time_import;
  RCP<Teuchos::Time> ts_time_resfill;
  //RCP<Teuchos::Time> ts_time_precfill;
  RCP<Teuchos::Time> ts_time_nsolve;
  RCP<Teuchos::Time> ts_time_view;

  //hacked stuff for elem_color
  Teuchos::RCP<elem_color> Elem_col;
  Teuchos::RCP<const Epetra_Comm>  Comm;
  //Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> x_1dra;
  //Kokkos::View<const double*> x_1dra; 
  
  //TUSAS_CUDA_CALLABLE_MEMBER void set_basis( GPUBasis &basis, const std::string elem_type) const;

};

//==================================================================
#include "ModelEvaluatorTPETRA_def.hpp"
//==================================================================

#endif
