//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
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
#include <Epetra_Comm.h>

//template<class Scalar>
/// Creates a nodal post process variable as a function of the solution and gradient. 
class post_process
{
public:
  /// Scalar operation enumeration
  /** If \p NONE, no scalar operation will be performed; otherwise the nodal variable will
      be reduced via \p SCALAR_OP and written to text file at each timestep. */
  enum SCALAR_OP {NONE, ///< No scalar operation
		  NORM1, ///< 1-norm.
		  NORM2, ///< 2-norm.
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
	       const double precision = 6 ///< precision for output file
	       );
  /// Destructor
  ~post_process();
  /// Write the post process variable to exodus.
  void update_mesh_data();
  /// Write the scalar op value to a data file.
  void update_scalar_data(double time///< time to be written 
			  );
  /// Compute the post process variable at node index \p i
  void process(const int i,///< index of vector entry
	       const double *u, ///< solution array
	       const double *gradu ///< solution derivative array
	       );
  /// typedef for post process function pointer
  typedef double (*PPFUNC)(const double *u, //< solution array
			   const double *gradu ///< solution derivative array
			   );
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
  /// Vector of the nodal values.
  Teuchos::RCP<Epetra_Vector> ppvar_;
  /// Scalar operator
  SCALAR_OP s_op_;
  /// Output precision 
  double precision_;
  /// Output filename
  std::string filename_;

};

#endif
