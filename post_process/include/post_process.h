//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef POST_PROCESS_H
#define POST_PROCESS_H

#include "Mesh.h"

//teuchos support
#include <Teuchos_RCP.hpp>

// Epetra support
#include "Epetra_Vector.h"
#include "Epetra_Map.h"
#include "Epetra_Import.h"
#include <Epetra_Comm.h>

#include <Teuchos_TimeMonitor.hpp>




//needed for create_onetoone hack below
#include "Epetra_Comm.h"
#include "Epetra_Directory.h"






//template<class Scalar>
/// Creates a nodal post process variable as a function of the solution and gradient. 
class post_process
{
public:
  /// Scalar operation enumeration
  /** If \p NONE, no scalar operation will be performed; otherwise the nodal variable will
      be reduced via \p SCALAR_OP and written to text file at each output. */
  enum SCALAR_OP {NONE, ///< No scalar operation
		  NORM1, ///< 1-norm.
		  NORM2, ///< 2-norm.
		  NORMRMS, ///< RMS norm
		  NORMINF, ///< Inf-norm
		  MINVALUE, ///< Minimum value
		  MAXVALUE, ///< Maximum value
		  MEANVALUE ///< Mean value
  };
  /// Constructor
  /** Creates a nodal post process variable with name <CODE>"pp"+index_</CODE>.
      Optionally a scalar operation performed on the variable and written to the text
      file <CODE>"pp"+index_+".dat"</CODE> at each timestep with precision \p precision.*/
  post_process(const Teuchos::RCP<const Epetra_Comm>& comm,  ///< MPI communicator
	       Mesh *mesh, ///< mesh object
	       const int index, ///< index of this post process variable
	       SCALAR_OP s_op = NONE, ///< scalar operation to perform
	       const int eqn_id = 0, ///< associate this post process variable with an equation
	       const std::string basename = "pp", ///< basename this post process variable
	       const double precision = 6 ///< precision for output file
	       );
  /// Destructor
  ~post_process();
  /// Write the post process variable to exodus.
  void update_mesh_data();
  /// Write the scalar op value to a data file.
  /// This should be preceded by a call to scalar_reduction()
  void update_scalar_data(double time///< time to be written 
			  );
  /// Compute the post process variable at node index \p i
  void process(const int i,///< index of vector entry
	       const double *u, ///< solution array
	       const double *uold, ///< solution array at previous timestep
	       const double *uoldold, ///< solution array at previous timestep
	       const double *gradu, ///< solution derivative array
	       const double &time, ///< current time
	       const double &dt, ///< current timestep size 
	       const double &dtold ///< previous timestep size
	       );
  /// typedef for post process function pointer
  typedef double (*PPFUNC)(const double *u, ///< solution array
			   const double *uold, ///< previous solution array
			   const double *uoldold, ///< previous solution array
			   const double *gradu, ///< solution derivative array
			   const double *xyz, ///< node xyz array
			   const double &time, ///< current time
			   const double &dt, ///< current timestep size
			   const double &dtold, ///< last timestep size
			   const int &eqn_id ///< equation this postprocess is associated with
			   );
  /// Return scalar reduction value
  double get_scalar_val();
  /// Perform scalar reduction.
  void scalar_reduction();
  /// Pointer to the post process function.
  PPFUNC postprocfunc_;
  //double (*postprocfunc_)(const double *u, const double *gradu);

private:
  /// Mesh object.
  Mesh *mesh_;
  /// This post process variable index.
  int index_;
  /// MPI comm object.
  const Teuchos::RCP<const Epetra_Comm>  comm_;
  /// Node map object.
  Teuchos::RCP<const Epetra_Map>   node_map_;
  /// Node overlap map object.
  Teuchos::RCP<const Epetra_Map>   overlap_map_;
  /// Import object.
  Teuchos::RCP<const Epetra_Import> importer_;
  /// Vector of the nodal values.
  Teuchos::RCP<Epetra_Vector> ppvar_;
  /// Scalar operator
  SCALAR_OP s_op_;
  /// Output precision 
  double precision_;
  /// Output filename
  std::string filename_;
  /// Scalar reduction value
  double scalar_val_;
  /// Equation this post process variable is associated with
  int eqn_id_;
  /// Variable and file base name
  std::string basename_;




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
