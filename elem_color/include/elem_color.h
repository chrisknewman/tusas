#ifndef ELEM_COLOR_H
#define ELEM_COLOR_H

#include "Mesh.h"

//teuchos support
#include <Teuchos_RCP.hpp>
#include "Epetra_CrsGraph.h"

#include <Isorropia_EpetraColorer.hpp>


#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif


class elem_color
{
public:
  elem_color(const Teuchos::RCP<const Epetra_Comm>& comm, 
				 Mesh *mesh);
  ~elem_color();

private:

  Mesh *mesh_;
  const Teuchos::RCP<const Epetra_Comm>  comm_;

  void compute_graph();
  void create_colorer();

  Teuchos::RCP<const Epetra_Map>  elem_map_;
  Teuchos::RCP<Epetra_CrsGraph>  graph_;

  Teuchos::RCP<Isorropia::Epetra::Colorer> elem_colorer_;
  Teuchos::RCP< Epetra_MapColoring > map_coloring_;

  std::vector<int> color_list_;
  std::vector< std::vector< int > > elem_LIDS_;

};
#endif
