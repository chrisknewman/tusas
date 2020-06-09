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

// Epetra support
#include "Epetra_Vector.h"
#include "Epetra_Map.h"
#include "Epetra_Import.h"
#include <Teuchos_TimeMonitor.hpp>

#include <Tpetra_Vector.hpp>




//needed for create_onetoone hack below
#include "Epetra_Comm.h"
#include "Epetra_Directory.h"





//#define ERROR_ESTIMATOR_OMP

//template<class Scalar>
/// Gradient recovery / error estimation.
/** Compute a Zienkiewicz-Zhu gradient recovery and H^1 error estimator. Supports only bilinear quad and tri elements in serial at this time.*/
class error_estimator
{
public:
  /// Constructor.
  /** Input total number of PDEs in the system numeqs, and the index index of the variable to create error estimator for. */
  error_estimator(const Teuchos::RCP<const Epetra_Comm>& comm,  ///< MPI communicator 
		  Mesh *mesh,  ///< mesh object
		  const int numeqs,  ///< the total number of pdes
		  const int index ///< the index of the variable 
		  );
  /// Destructor.
  ~error_estimator();
  /// Estimate the gradient at each node.
  void estimate_gradient(const Teuchos::RCP<Epetra_Vector>& ///< solution vector (input)
			 );
  /// Estimate the gradient at each node.
  void estimate_gradient(const Teuchos::RCP<Tpetra::Vector<> >& ///< solution vector (input)
			 );
  /// Estimate the error on each element.
  void estimate_error(const Teuchos::RCP<Epetra_Vector>& ///< solution vector (input)
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
  /// Estimated nodal derivative wrt to x.
  Teuchos::RCP<Epetra_Vector> gradx_;
  /// Estimated nodal derivative wrt to y.
  Teuchos::RCP<Epetra_Vector> grady_;
  /// Estimated nodal derivative wrt to y.
  Teuchos::RCP<Epetra_Vector> gradz_;

private:
  ///Mesh object
  Mesh *mesh_;
  /// Total number of PDEs.
  int numeqs_;
  /// Variable index.
  int index_;
  /// MPI comm object.
  const Teuchos::RCP<const Epetra_Comm>  comm_;
  /// Node map object.
  Teuchos::RCP<const Epetra_Map>   node_map_;
  /// Node overlap map object.
  Teuchos::RCP<const Epetra_Map>   overlap_map_;
  /// Element map object.
  Teuchos::RCP<const Epetra_Map>   elem_map_;
  /// Error contribution on each element.
  Teuchos::RCP<Epetra_Vector> elem_error_;
  /// Import object.
  Teuchos::RCP<const Epetra_Import> importer_;
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

  //we need this in some public utility class...
  const Mesh::mesh_lint_t epetra_map_gid(Teuchos::RCP<const Epetra_Map> map, const int lid)
  {
#ifdef MESH_64
      const Mesh::mesh_lint_t gid = map->GID64(lid);
#else
      const Mesh::mesh_lint_t gid = map->GID(lid);
#endif
      return gid;
  };



  //we need these in some static public utility class...
Epetra_Map Create_OneToOne_Map64(const Epetra_Map& usermap,
         bool high_rank_proc_owns_shared=false)
{
  //if usermap is already 1-to-1 then we'll just return a copy of it.
  if (usermap.IsOneToOne()) {
    Epetra_Map newmap(usermap);
    return(newmap);
  }

  int myPID = usermap.Comm().MyPID();
  Epetra_Directory* directory = usermap.Comm().CreateDirectory(usermap);

  int numMyElems = usermap.NumMyElements();
  const long long* myElems = usermap.MyGlobalElements64();

  int* owner_procs = new int[numMyElems];

  directory->GetDirectoryEntries(usermap, numMyElems, myElems, owner_procs,
         0, 0, high_rank_proc_owns_shared);

  //we'll fill a list of map-elements which belong on this processor

  long long* myOwnedElems = new long long[numMyElems];
  int numMyOwnedElems = 0;

  for(int i=0; i<numMyElems; ++i) {
    long long GID = myElems[i];
    int owner = owner_procs[i];

    if (myPID == owner) {
      myOwnedElems[numMyOwnedElems++] = GID;
    }
  }

  Epetra_Map one_to_one_map((long long)-1, numMyOwnedElems, myOwnedElems,
       usermap.IndexBase(), usermap.Comm()); // CJ TODO FIXME long long

  delete [] myOwnedElems;
  delete [] owner_procs;
  delete directory;

  return(one_to_one_map);
};


};

#endif
