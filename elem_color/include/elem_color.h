//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



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

//#define TUSAS_COLOR


class elem_color
{
public:
  elem_color(const Teuchos::RCP<const Epetra_Comm>& comm, 
				 Mesh *mesh);
  ~elem_color();

  //we could point to the underlying isorropia data instead, in the future
  std::vector<int> get_color(int i){return elem_LIDS_[i];}
  int get_num_color(){return num_color_;}
  void update_mesh_data();

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
  int num_color_;

  void init_mesh_data();

};
#endif
