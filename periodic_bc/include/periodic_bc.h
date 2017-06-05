//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef PERIODIC_BC_H
#define PERIODIC_BC_H

#include "Mesh.h"

//teuchos support
#include <Teuchos_RCP.hpp>

// Epetra support
#include <Epetra_Comm.h>
#include "Epetra_Map.h"
#include "Epetra_Import.h"

/// Creates a periodic bc pair 
class periodic_bc
{
public:

  /// Constructor
  /** Creates a periodic bc pair  */
  periodic_bc(const int ns_id1, ///< Nodeset index 1
	      const int ns_id2, ///< Nodeset index 2
	      const int index, ///< the index of the variable
	      const int numeqs, ///< the number of PDEs  
	      Mesh *mesh, ///< mesh object
	      const Teuchos::RCP<const Epetra_Comm>& comm ///< MPI communicator
	      );

  /// Destructor.
  ~periodic_bc();

private:
  ///Mesh object
  Mesh *mesh_;
  /// MPI comm object.
  const Teuchos::RCP<const Epetra_Comm>  comm_;
  /// Node map object.
  Teuchos::RCP<const Epetra_Map>   ns1_map_;
  /// Node map object.
  Teuchos::RCP<const Epetra_Map>   ns2_map_;
  /// Nodeset 1 index
  int ns_id1_;
  /// Nodeset 2 index
  int ns_id2_;
  /// Variable index.
  int index_;
  /// Number of pdes.
  int numeqs_;
  /// Import object.
  Teuchos::RCP<const Epetra_Import> importer1_;
  /// Import object.
  Teuchos::RCP<const Epetra_Import> importer2_;
  /// Node overlap map object.
  Teuchos::RCP<const Epetra_Map>   overlap_map_;

  Teuchos::RCP<const Epetra_Map> get_replicated_map(const int id);


};


#endif
