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

//#define TUSAS_PROJECTION

class projection
{
public:
  /// Constructor.
  projection(const Teuchos::RCP<const Epetra_Comm>& comm, 
	     const std::string meshNameString, 
	     const std::string dataNameString );

  /// Destructor.
  ~projection();

  bool get_source_value(const double x, const double y, const double z, double &val);
  void fill_initial_values();
  void fill_time_interp_values(const int timeindex, const double theta);

private:
  ///Source mesh object
  Mesh *sourcemesh_;

  const Teuchos::RCP<const Epetra_Comm>  comm_;
  Teuchos::RCP<const Epetra_Map>   source_node_map_;
  Teuchos::RCP<Epetra_Vector> source_node_;
  Teuchos::RCP<const Epetra_Map>   source_elem_map_;
  Teuchos::RCP<Epetra_Vector> source_elem_;

  void read_file();
  void read_file_time_interp(const int timeindex, const double theta);
  void update_mesh_data();
  void elem_to_node_avg();

  std::string meshNameString_;
  std::string dataNameString_;

  const int stride_;

};
#endif
