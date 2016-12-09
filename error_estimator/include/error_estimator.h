//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
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

#define ERROR_ESTIMATOR_OMP

//template<class Scalar>
class error_estimator
{
public:
  error_estimator(const Teuchos::RCP<const Epetra_Comm>& comm, Mesh *mesh, const int numeqs, const int index);
  ~error_estimator();

  void estimate_gradient(const Teuchos::RCP<Epetra_Vector>&);

  void estimate_error(const Teuchos::RCP<Epetra_Vector>&);

  void test_lapack();

  void update_mesh_data();

  double estimate_global_error();

  Teuchos::RCP<Epetra_Vector> gradx_;
  Teuchos::RCP<Epetra_Vector> grady_;

private:

  Mesh *mesh_;

  int numeqs_;
  int index_;
  const Teuchos::RCP<const Epetra_Comm>  comm_;
  Teuchos::RCP<const Epetra_Map>   node_map_;
  Teuchos::RCP<const Epetra_Map>   overlap_map_;
  Teuchos::RCP<const Epetra_Map>   elem_map_;
  Teuchos::RCP<Epetra_Vector> elem_error_;

  Teuchos::RCP<const Epetra_Import> importer_;

  double global_error_;

  Teuchos::RCP<Teuchos::Time> ts_time_grad;
  Teuchos::RCP<Teuchos::Time> ts_time_error;

};

#endif
