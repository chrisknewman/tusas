#ifndef ERROR_ESTIMATOR_H
#define ERROR_ESTIMATOR_H

#include "Mesh.h"

//teuchos support
#include <Teuchos_RCP.hpp>

// Epetra support
#include "Epetra_Vector.h"
#include "Epetra_Map.h"

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

private:

  Mesh *mesh_;

  int numeqs_;
  int index_;
  const Teuchos::RCP<const Epetra_Comm>  comm_;
  Teuchos::RCP<const Epetra_Map>   node_map_;
  Teuchos::RCP<const Epetra_Map>   elem_map_;
  Teuchos::RCP<Epetra_Vector> gradx_;
  Teuchos::RCP<Epetra_Vector> grady_;
  Teuchos::RCP<Epetra_Vector> elem_error_;

  double global_error_;

};

#endif
