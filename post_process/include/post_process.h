#ifndef POST_PROCESS_H
#define POST_PROCESS_H

#include "Mesh.h"

//teuchos support
#include <Teuchos_RCP.hpp>

// Epetra support
#include "Epetra_Vector.h"
#include "Epetra_Map.h"

//template<class Scalar>
class post_process
{
public:
  post_process(const Teuchos::RCP<const Epetra_Comm>& comm, Mesh *mesh, const int index);
  ~post_process();

  void update_mesh_data();

  void process(const int i,const double *u, const double *gradu);

  double (*postprocfunc_)(const double *u, const double *gradu);

private:

  Mesh *mesh_;

  int index_;
  const Teuchos::RCP<const Epetra_Comm>  comm_;
  Teuchos::RCP<const Epetra_Map>   node_map_;
  Teuchos::RCP<Epetra_Vector> ppvar_;

};

#endif
