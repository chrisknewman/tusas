//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef ERROR_ESTIMATOR_H
#define ERROR_ESTIMATOR_H

#include "Mesh.h"

//teuchos support
#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_TimeMonitor.hpp>

// Tpetra support
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Import.hpp>

class Basis;

//#define ERROR_ESTIMATOR_OMP

//template<class Scalar>
/// Gradient recovery / error estimation.
/** Compute a Zienkiewicz-Zhu (superconvergent patch recovery) gradient and H^1 error estimator.
    Supports linear tri/quad/hex/tet and quadratic quad elements.

    For each node n the recovered gradient component is the polynomial g = p(x)^T a fit in the
    least squares sense to grad u_h at the Gauss points of all elements in the nodal patch, ie
    (sum_gp p p^T) a = sum_gp p (grad u_h). Each processor accumulates the contributions of the
    elements it owns; the contributions are summed on the owning processor with a Tpetra export,
    so the patch spans processor boundaries and the result is independent of the decomposition.
    p is evaluated in coordinates shifted to the node and scaled by the patch size. */
class error_estimator
{
public:
  /// Constructor.
  /** Input total number of PDEs in the system numeqs, and the index index of the variable to create error estimator for. */
  error_estimator(Mesh *mesh,  ///< mesh object
		  const int numeqs,  ///< the total number of pdes
		  const int index ///< the index of the variable 
		  );
  /// Destructor.
  ~error_estimator();
  /// Estimate the gradient at each node.
  void estimate_gradient(const Teuchos::RCP<Tpetra::Vector<> >& ///< solution vector (input)
			 );
  /// Estimate the error on each element.
  void estimate_error(const Teuchos::RCP<Tpetra::Vector<> >& ///< solution vector (input)
			 );
  /// A helper function to test the Lapack implementation.
  void test_lapack();
  /// Output the nodal gradient and the elemental error contribution to the exodus file.
  void update_mesh_data();
  /// Estimate the global H^1 error.
  double estimate_global_error();
  typedef Tpetra::Vector<>::scalar_type scalar_type;
  typedef Tpetra::Vector<>::local_ordinal_type local_ordinal_type;
  typedef Tpetra::Vector<>::global_ordinal_type global_ordinal_type;
  typedef Tpetra::Vector<>::node_type node_type;
  typedef Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type> map_type;
  typedef Tpetra::Vector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> vector_type;
  typedef Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> mv_type;
  typedef Tpetra::Import<local_ordinal_type, global_ordinal_type, node_type> import_type;

  /// Estimated nodal derivative wrt to x (on the owned node map).
  Teuchos::RCP<vector_type> gradx_;
  /// Estimated nodal derivative wrt to y (on the owned node map).
  Teuchos::RCP<vector_type> grady_;
  /// Estimated nodal derivative wrt to z (on the owned node map).
  Teuchos::RCP<vector_type> gradz_;

private:
  ///Mesh object
  Mesh *mesh_;
  /// Total number of PDEs.
  int numeqs_;
  /// Variable index.
  int index_;
  /// MPI comm object.
  Teuchos::RCP<const Teuchos::Comm<int> > comm_;
  /// Node map object.
  Teuchos::RCP<const map_type>   node_map_;
  /// Node overlap map object.
  Teuchos::RCP<const map_type>   overlap_map_;
  /// Element map object.
  Teuchos::RCP<const map_type>   elem_map_;
  /// Error contribution on each element.
  Teuchos::RCP<vector_type> elem_error_;
  /// Import object (node_map_ -> overlap_map_).
  Teuchos::RCP<const import_type> importer_;
  /// Patch length scale for each node on the overlap map (consistent across processors).
  Teuchos::RCP<vector_type> patch_scale_;
  /// Number of terms in the recovery polynomial.
  int dimp_;
  /// Number of Gauss points per element used to sample the gradient.
  int num_q_pts_;
  /// Global H^1 error estimate.
  double global_error_;
  /// Timing object.
  Teuchos::RCP<Teuchos::Time> ts_time_grad;
  /// Timing object.
  Teuchos::RCP<Teuchos::Time> ts_time_error;

  /// Estimate the gradient at each node.
  void estimate_gradient(const double * uvec ///< solution vector (input)
			 );

  /// Estimate the error on each element.
  void estimate_error(const double * uvec ///< solution vector (input)
			 );

  /// Create a basis object for the element type of block 0, sampled at the recovery Gauss points.
  Basis * create_recovery_basis() const;

  /// Evaluate the recovery polynomial terms at the scaled local coordinates (x,y,z); p has length dimp_.
  void eval_poly(const double x, const double y, const double z, double * p) const;

};

#endif
