//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef PERIODIC_BC_H
#define PERIODIC_BC_H

#include "Mesh.h"

//teuchos support
#include <Teuchos_RCP.hpp>

// Epetra support
#include <Epetra_Comm.h>
#include "Epetra_Map.h"
#include "Epetra_Import.h"
#include "Epetra_Vector.h"
#include "Epetra_FEVector.h"



//needed for create_onetoone hack below
#include "Epetra_Comm.h"
#include "Epetra_Directory.h"



#define PERIODIC_BC

/// Creates a periodic bc and executes communication 
class periodic_bc
{
public:

  /// Constructor
  /** Creates a periodic bc pair  */
  periodic_bc(const int ns_id1, ///< Nodeset index 1
	      const int ns_id2, ///< Nodeset index 2
	      const int numeqs, ///< the number of PDEs  
	      Mesh *mesh, ///< mesh object
	      const Teuchos::RCP<const Epetra_Comm>& comm ///< MPI communicator
	      );

  /// Destructor.
  ~periodic_bc();
  /// Import data.
  void import_data(const Epetra_FEVector &f_full, ///< f vector (input)
		   const Teuchos::RCP<const Epetra_Vector> u_full, ///< u vector (input)
		   const int eqn_index ///< eqn_index (input)
		   ) const;

  /// Replicated vector.
  Teuchos::RCP<Epetra_Vector> f_rep_;
  /// Replicated vector.
  Teuchos::RCP<Epetra_Vector> u_rep_;

  /// Vector containing the equation indices associated with this periodic_bc
  std::vector<int> eqn_indices_;

  /// Add an equation index i to this periodic_bc
  void add_eqn_index(const int i){eqn_indices_.push_back(i);};

private:
  ///Mesh object
  Mesh *mesh_;
  /// MPI comm object.
  const Teuchos::RCP<const Epetra_Comm>  comm_;
  /// Node map object.
  Teuchos::RCP<const Epetra_Map>   ns1_map_;
  /// Node map object.
  Teuchos::RCP<const Epetra_Map>   ns2_map_;
  /// Nodeset 1 index
  int ns_id1_;
  /// Nodeset 2 index
  int ns_id2_;
  /// Number of pdes.
  int numeqs_;
  /// Import object.
  Teuchos::RCP<const Epetra_Import> importer1_;
  /// Import object.
  Teuchos::RCP<const Epetra_Import> importer2_;
  /// Node map object.
  Teuchos::RCP<const Epetra_Map>   node_map_;
  /// Sets up the communication maps for nodeset id.
  Teuchos::RCP<const Epetra_Map> get_replicated_map(const int id);

  /// Vector utility function
  void uniquifyWithOrder_set_remove_if(const std::vector<int>& input, std::vector<int>& output);

  //need a common version of this in a utility class
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
  
};


#endif
