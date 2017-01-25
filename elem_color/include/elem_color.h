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

/// Element coloring for residual and preconditioner fill with OpenMP.
/** To enable <code>\#define TUSAS_COLOR</code>. */
class elem_color
{
public:
  /// Constructor
  elem_color(const Teuchos::RCP<const Epetra_Comm>& comm, 
				 Mesh *mesh);
  ///Destructor
  ~elem_color();

  //we could point to the underlying isorropia data instead, in the future
  /// Return a std::vector of elements in the i-th color.
  std::vector<int> get_color(int i){return elem_LIDS_[i];}
  /// Return the number of colors.
  int get_num_color(){return num_color_;}
  /// Output element color to exodus file.
  void update_mesh_data();

private:

  /// Pointer to mesh.
  Mesh *mesh_;
  /// Pointer to mpi comm.
  const Teuchos::RCP<const Epetra_Comm>  comm_;
  /// Compute the element graph.
  void compute_graph();
  /// Compute element graph coloring.
  void create_colorer();
  /// Element map.
  Teuchos::RCP<const Epetra_Map>  elem_map_;
  /// Element graph.
  Teuchos::RCP<Epetra_CrsGraph>  graph_;
  /// Isorropia color object.
  Teuchos::RCP<Isorropia::Epetra::Colorer> elem_colorer_;
  /// Epetra MapColoring object.
  Teuchos::RCP< Epetra_MapColoring > map_coloring_;
  /// List of color ids.
  std::vector<int> color_list_;
  /// List of local element ids.
  std::vector< std::vector< int > > elem_LIDS_;
  /// Number of colors.
  int num_color_;
  /// Initializes element color variable in mesh.
  void init_mesh_data();

};
#endif
