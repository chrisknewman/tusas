//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef PROJECTION_H
#define PROJECTION_H

// Epetra support
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"

//teuchos support
#include <Teuchos_RCP.hpp>

#include "Mesh.h"

#define TUSAS_PROJECTION

class projection
{
public:
  /// Constructor.
  projection(const Teuchos::RCP<const Epetra_Comm>& comm);

  /// Destructor.
  ~projection();

private:
  ///Source mesh object
  Mesh *sourcemesh_;

  const Teuchos::RCP<const Epetra_Comm>  comm_;
  Teuchos::RCP<const Epetra_Map>   source_map_;
  Teuchos::RCP<Epetra_Vector> source_;

  void read_file();
  void update_mesh_data();

};
#endif
