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
//#include <Teuchos_TimeMonitor.hpp>

#include <Isorropia_EpetraColorer.hpp>


#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

/// Element coloring for residual and preconditioner fill with OpenMP.
/** To enable <code>\#define TUSAS_COLOR_CPU</code>. */
class elem_color
{
public:
  /// Constructor
  elem_color(const Teuchos::RCP<const Epetra_Comm>& comm,   ///< MPI communicator
	     Mesh *mesh ///< mesh object
	     );
  ///Destructor
  ~elem_color();

  //we could point to the underlying isorropia data instead, in the future
  /// Return a std::vector of elements in the i-th color.
  std::vector<int> get_color(int i ///<color index
			     ){return elem_LIDS_[i];}
  std::vector< std::vector< int > > get_colors(){return elem_LIDS_;}
  std::vector< int > get_colors_flat(){return elem_LIDS_flat_;}
  /// Return the number of colors.
  int get_num_color(){return num_color_;}
  /// Return the number of elements with color i.
  const int get_num_elem_with_color(int i) {return map_coloring_->NumElementsWithColor(i);}
  /// Return vector with number of elements with color
  std::vector< int > get_num_elem_with_color(){return num_elem_with_color;}
  /// Output element color to exodus file.
  void update_mesh_data();

  /// List of number of elements for each color.
  std::vector<int> num_elem_with_color;
  /// List of local element ids.
  std::vector<int> elem_LIDS_flat_;

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
  /// List of local element ids.
  std::vector<int> num_elem_;
  /// Number of colors.
  int num_color_;
  /// Initializes element color variable in mesh.
  void init_mesh_data();

  void insert_off_proc_elems();

  //Teuchos::RCP<Teuchos::Time> ts_time_elemadj;

};
#endif
