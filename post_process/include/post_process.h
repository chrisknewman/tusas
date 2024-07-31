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
#include <Tpetra_Vector.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_Import.hpp>

#include <Teuchos_TimeMonitor.hpp>

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
  post_process(//const Teuchos::RCP<const Epetra_Comm>& comm,  ///< MPI communicator
	       Mesh *mesh, ///< mesh object
	       const int index, ///< index of this post process variable
	       SCALAR_OP s_op = NONE, ///< scalar operation to perform
	       bool restart = false, ///< restart bool
	       const int eqn_id = 0, ///< associate this post process variable with an equation
	       const std::string basename = "pp", ///< basename this post process variable
	       const int precision = 6 ///< precision for output file
	       );
  /// Destructor
  ~post_process();
  /// Write the post process variable to exodus.
  void update_mesh_data() const;
  /// Write the scalar op value to a data file.
  /// This should be preceded by a call to scalar_reduction()
  void update_scalar_data(const double &time///< time to be written 
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
  double get_scalar_val() const;
  /// Perform scalar reduction.
  void scalar_reduction();
  /// Pointer to the post process function.
  PPFUNC postprocfunc_;
  //double (*postprocfunc_)(const double *u, const double *gradu);

private:

  typedef Tpetra::Map<>::global_ordinal_type global_ordinal_type;
  typedef Tpetra::Map<>::local_ordinal_type local_ordinal_type;
  typedef Tpetra::Map<>::node_type node_type;
  typedef Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type> map_type;
  typedef Tpetra::Import<local_ordinal_type, global_ordinal_type,
                         node_type> import_type;
  typedef Tpetra::Vector<>::scalar_type scalar_type;
  typedef Tpetra::Vector<scalar_type, local_ordinal_type,
			 global_ordinal_type, node_type> vector_type;
  /// Mesh object.
  Mesh *mesh_;
  /// This post process variable index.
  int index_;
  /// Node map object.
  Teuchos::RCP<const map_type > node_map_;
  /// Node overlap map object.
  Teuchos::RCP<const map_type > overlap_map_;
  /// Import object.
  Teuchos::RCP<const import_type > importer_;
  /// Vector of the nodal values.
  Teuchos::RCP<vector_type> ppvar_;
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
  /// restart boolean
  bool restart_;
  /// write files boolean
  //bool write_files_;

};

#endif
