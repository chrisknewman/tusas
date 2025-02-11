//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////


#ifndef ELEM_COLOR_H
#define ELEM_COLOR_H

#include "Mesh.h"

//teuchos support
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <Tpetra_Vector.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Export.hpp>
#include <Tpetra_CrsGraph_decl.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>



#define TUSAS_MAX_NODE_PER_ROW_PER_EQN_HEX 81
// #define TUSAS_MAX_NODE_PER_ROW_PER_EQN_QUAD 27

class elem_color
{
public:
  /// Constructor
  elem_color(Mesh *mesh, ///< mesh object
	     bool dorestart = false, ///< do restart
	     bool writedata = false
	     );
  ///Destructor
  ~elem_color();

  /// Return a std::vector of elements in the i-th color.
  std::vector<int> get_color(const int i ///<color index
			     ) const {return elem_LIDS_[i];}
  std::vector< std::vector< int > > get_colors() const {return elem_LIDS_;}
  /// Return the number of colors.
  const int get_num_color() const {return num_color_;}
  /// Output element color to exodus file.
  void update_mesh_data() const;

private:


  typedef Tpetra::global_size_t global_size_t;
  typedef Tpetra::Vector<>::global_ordinal_type global_ordinal_type;
  typedef Tpetra::Vector<>::local_ordinal_type local_ordinal_type;
  typedef Tpetra::Vector<>::node_type node_type;
  typedef Tpetra::CrsGraph<local_ordinal_type, global_ordinal_type,
                         node_type> crs_graph_type;
  typedef Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type> map_type;

  /// Element graph.
  Teuchos::RCP<crs_graph_type>  elem_graph_;

  /// Pointer to mesh.
  Mesh *mesh_;
  /// Compute the element graph.
  void compute_graph();
  /// Compute element graph coloring.
  void create_colorer();
  /// List of local element ids.
  std::vector< std::vector< int > > elem_LIDS_;
  /// Number of colors.
  int num_color_;
  /// Initializes element color variable in mesh.
  void init_mesh_data() const;
  /// Populate elem_LIDS_
  void restart();

  //Teuchos::RCP<Teuchos::Time> ts_time_elemadj;
  Teuchos::RCP<Teuchos::Time> ts_time_color;
  //Teuchos::RCP<Teuchos::Time> ts_time_create;
  bool writedata_;

};
#endif
