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
#include "Epetra_CrsGraph.h"
#include <Teuchos_TimeMonitor.hpp>

#include <Isorropia_EpetraColorer.hpp>


#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif




#include <Tpetra_Vector.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Export.hpp>
#include <Tpetra_CrsGraph_decl.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>




//needed for create_onetoone hack below
#include "Epetra_Comm.h"
#include "Epetra_Directory.h"


//needed for create_root hack below
#include "Epetra_LocalMap.h"
#include "Epetra_GIDTypeVector.h"
#include "Epetra_Import.h"

//we need to clean up here in a major way....

//revert to issoropia by commenting next line
#define ELEM_COLOR_USE_ZOLTAN

#define TUSAS_MAX_NODE_PER_ROW_PER_EQN_HEX 81
// #define TUSAS_MAX_NODE_PER_ROW_PER_EQN_QUAD 27

//#define TUSAS_COLOR_CPU
//#define TUSAS_COLOR_GPU

/// Element coloring for residual and preconditioner fill with OpenMP.
/** To enable <code>\#define TUSAS_COLOR_CPU</code>. */
class elem_color
{
public:
  /// Constructor
  elem_color(const Teuchos::RCP<const Epetra_Comm>& comm,   ///< MPI communicator
	     Mesh *mesh, ///< mesh object
	     bool dorestart = false ///< do restart
	     );
  ///Destructor
  ~elem_color();

  //we could point to the underlying isorropia data instead, in the future
  /// Return a std::vector of elements in the i-th color.
  std::vector<int> get_color(const int i ///<color index
			     ) const {return elem_LIDS_[i];}
  std::vector< std::vector< int > > get_colors(){return elem_LIDS_;}
  /// Return the number of colors.
  int get_num_color(){return num_color_;}
  /// Output element color to exodus file.
  void update_mesh_data();

private:


  typedef Tpetra::global_size_t global_size_t;
  typedef Tpetra::Vector<>::global_ordinal_type global_ordinal_type;
  typedef Tpetra::Vector<>::local_ordinal_type local_ordinal_type;
  typedef Tpetra::Vector<>::node_type node_type;
  typedef Tpetra::CrsGraph<local_ordinal_type, global_ordinal_type,
                         node_type> crs_graph_type;
  typedef Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type> map_type;

//#ifdef ELEM_COLOR_USE_ZOLTAN
  /// Element map.
  Teuchos::RCP<const map_type>  elem_map_;
  /// Element graph.
  Teuchos::RCP<crs_graph_type>  elem_graph_;

//#else
  /// Element map.
  Teuchos::RCP<const Epetra_Map>  map_;
  /// Element graph.
  Teuchos::RCP<Epetra_CrsGraph>  graph_;
//#endif


  /// Pointer to mesh.
  Mesh *mesh_;
  /// Pointer to mpi comm.
  const Teuchos::RCP<const Epetra_Comm>  comm_;
  /// Compute the element graph.
  void compute_graph();
  /// Compute element graph coloring.
  void create_colorer();
  /// List of local element ids.
  std::vector< std::vector< int > > elem_LIDS_;
  /// Number of colors.
  int num_color_;
  /// Initializes element color variable in mesh.
  void init_mesh_data();
  /// Inserts off processor elements into the graph.
  void insert_off_proc_elems();
  /// List of color ids.
  std::vector<int> color_list_;
  /// Populate elem_LIDS_
  void restart();

  //Teuchos::RCP<Teuchos::Time> ts_time_elemadj;
  Teuchos::RCP<Teuchos::Time> ts_time_color;
  //Teuchos::RCP<Teuchos::Time> ts_time_create;






  //we need these in some static public utility class...
Epetra_Map Create_OneToOne_Map64(const Epetra_Map& usermap,
         bool high_rank_proc_owns_shared=false)
{
  //if usermap is already 1-to-1 then we'll just return a copy of it.
  if (usermap.IsOneToOne()) {
    Epetra_Map newmap(usermap);
    return(newmap);
  }

  int myPID = usermap.Comm().MyPID();
  Epetra_Directory* directory = usermap.Comm().CreateDirectory(usermap);

  int numMyElems = usermap.NumMyElements();
  const long long* myElems = usermap.MyGlobalElements64();

  int* owner_procs = new int[numMyElems];

  directory->GetDirectoryEntries(usermap, numMyElems, myElems, owner_procs,
         0, 0, high_rank_proc_owns_shared);

  //we'll fill a list of map-elements which belong on this processor

  long long* myOwnedElems = new long long[numMyElems];
  int numMyOwnedElems = 0;

  for(int i=0; i<numMyElems; ++i) {
    long long GID = myElems[i];
    int owner = owner_procs[i];

    if (myPID == owner) {
      myOwnedElems[numMyOwnedElems++] = GID;
    }
  }

  Epetra_Map one_to_one_map((long long)-1, numMyOwnedElems, myOwnedElems,
       usermap.IndexBase(), usermap.Comm()); // CJ TODO FIXME long long

  delete [] myOwnedElems;
  delete [] owner_procs;
  delete directory;

  return(one_to_one_map);
};



static Epetra_Map Create_Root_Map64(const Epetra_Map& usermap,
         int root)
{
  int numProc = usermap.Comm().NumProc();
  if (numProc==1) {
    Epetra_Map newmap(usermap);
    return(newmap);
  }

  const Epetra_Comm & comm = usermap.Comm();
  bool isRoot = usermap.Comm().MyPID()==root;

  //if usermap is already completely owned by root then we'll just return a copy of it.
  int quickreturn = 0;
  int globalquickreturn = 0;

  if (isRoot) {
    if (usermap.NumMyElements()==usermap.NumGlobalElements64()) quickreturn = 1;
  }
  else {
    if (usermap.NumMyElements()==0) quickreturn = 1;
  }
  usermap.Comm().MinAll(&quickreturn, &globalquickreturn, 1);

  if (globalquickreturn==1) {
    Epetra_Map newmap(usermap);
    return(newmap);
  }

  // Linear map: Simple case, just put all GIDs linearly on root processor
  if (usermap.LinearMap() && root!=-1) {
    int numMyElements = 0;
    if(usermap.MaxAllGID64()+1 > std::numeric_limits<int>::max())
      throw "Epetra_Util::Create_Root_Map: cannot fit all gids in int";
    if (isRoot) numMyElements = (int)(usermap.MaxAllGID64()+1);
    Epetra_Map newmap((long long) -1, numMyElements, (long long)usermap.IndexBase64(), comm);
    return(newmap);
  }

  if (!usermap.UniqueGIDs())
    throw usermap.ReportError("usermap must have unique GIDs",-1);

  // General map

  // Build IntVector of the GIDs, then ship them to root processor
  int numMyElements = usermap.NumMyElements();
  Epetra_Map allGidsMap((long long) -1, (long long) numMyElements, (long long) 0, comm);
  typename Epetra_GIDTypeVector<long long>::impl allGids(allGidsMap);
  for (int i=0; i<numMyElements; i++) allGids[i] = (long long) usermap.GID64(i);

  if(usermap.MaxAllGID64() > std::numeric_limits<int>::max())
    throw "Epetra_Util::Create_Root_Map: cannot fit all gids in int";
  long long numGlobalElements = (long long) usermap.NumGlobalElements64();
  if (root!=-1) {
    long long n1 = 0; if (isRoot) n1 = numGlobalElements;
    Epetra_Map allGidsOnRootMap((long long) -1, n1, (long long) 0, comm);
    Epetra_Import importer(allGidsOnRootMap, allGidsMap);
    typename Epetra_GIDTypeVector<long long>::impl allGidsOnRoot(allGidsOnRootMap);
    allGidsOnRoot.Import(allGids, importer, Insert);

    Epetra_Map rootMap((long long)-1, allGidsOnRoot.MyLength(), allGidsOnRoot.Values(), (long long)usermap.IndexBase64(), comm);
    return(rootMap);
  }
  else {
    long long n1 = numGlobalElements;
    Epetra_LocalMap allGidsOnRootMap((long long) n1, (long long) 0, comm);
    Epetra_Import importer(allGidsOnRootMap, allGidsMap);
    //importer.Print(std::cout);
    typename Epetra_GIDTypeVector<long long>::impl allGidsOnRoot(allGidsOnRootMap);
    allGidsOnRoot.Import(allGids, importer, Insert);

    Epetra_Map rootMap((long long) -1, allGidsOnRoot.MyLength(), allGidsOnRoot.Values(), (long long)usermap.IndexBase64(), comm);

    return(rootMap);
  }
};



};
#endif
