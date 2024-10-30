//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#include "periodic_bc.h"
#include "greedy_tie_break.hpp"

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_ArrayViewDecl.hpp>

#include <Tpetra_Map_decl.hpp>

#include <unordered_set>
#include <set>

periodic_bc::periodic_bc(const int ns_id1, 
			 const int ns_id2,
			 const int numeqs, 
			 Mesh *mesh):
  ns_id1_(ns_id1),
  ns_id2_(ns_id2),
  numeqs_(numeqs),
  mesh_(mesh)
{

  auto comm_ = Teuchos::DefaultComm<int>::getComm();
  //cn the first step is add all of the ns1 equations to the ns2 equations
  //cn then replace all the ns1 eqns with u(ns1)-u(ns2)

	//cn see:
	// [1]  http://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/0836-11.pdf
	// and similar to
	// [2] https://orbi.ulg.ac.be/bitstream/2268/100283/1/2012_COMMAT_PBC.pdf

        // we implement eq 14 in [1]

  if( 0 == comm_->getRank() ){
    std::cout<<"Entering periodic_bc::periodic_bc for:"<<std::endl
	     <<"    ns_id1_ = "<<ns_id1_<<std::endl
	     <<"    ns_id2_ = "<<ns_id2_<<std::endl<<std::endl;
  }
  //this sorts all nodelists, we don't want to do this everytime so we need
  //to pass it the nodeset ids eventually
  mesh_->create_sorted_nodesetlists();

  //cn we will eventually do an import of global f and global u into replicated vectors here,
  //cn hence we may need to change the map routine from global mesh id to global u and f id;
  //cn ie numeqs_*gid+index_ this could be done at the if(-99... line below.

  std::vector<Mesh::mesh_lint_t> node_num_map(mesh_->get_node_num_map());

  //we want this map to be a one equation version of the x_owned_map in tusas
  //do it this way and hope it is the same

  //cn we can make overlap local

  const Tpetra::global_size_t numGlobalEntries =
     Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  const Tpetra::Map<>::global_ordinal_type indexBase = 0;

  std::vector<Tpetra::Map<>::global_ordinal_type> my_global_nodes(node_num_map.size());
  for(int i = 0; i < node_num_map.size(); i++){
    my_global_nodes[i] = node_num_map[i];
  }

  const Teuchos::ArrayView<Tpetra::Map<>::global_ordinal_type> AV(my_global_nodes);
  overlap_map_ = Teuchos::rcp(new map_type(numGlobalEntries,
                                              AV,
                                              indexBase,
                                              comm_));

  if( 1 == comm_->getSize() ){
    node_map_ = overlap_map_;
  }else{
    GreedyTieBreak<local_ordinal_type,global_ordinal_type> greedy_tie_break;
    node_map_ = Teuchos::rcp(new map_type(
        *(Tpetra::createOneToOne(overlap_map_,greedy_tie_break))));
  };

  ns1_map_ = get_replicated_map(ns_id1_);

  ns2_map_ = get_replicated_map(ns_id2_);
  //here we check for same size maps

  if( ns1_map_->getGlobalNumElements() != ns2_map_->getGlobalNumElements() ) {
    if( 0 == comm_->getRank() ) {   
      std::cout<<"Incompatible global node set sizes found for ("<<ns_id1_<<","<<ns_id2_<<")."<<std::endl;
    }
    exit(0);
  }
  if( ns1_map_->getLocalNumElements() != ns2_map_->getLocalNumElements() ) {
    if( 0 == comm_->getRank() ) {
      std::cout<<"Incompatible local node set sizes found for ("<<ns_id1_<<","<<ns_id2_<<")."<<std::endl;
    }
    exit(0);
  }



  // source, target
  importer1_ = Teuchos::rcp(new import_type(node_map_, ns1_map_));

  comm_->barrier();

  if(ns2_map_.is_null()) std::cout << "NULL" << std::endl;

  importer2_ = Teuchos::rcp(new import_type(node_map_, ns2_map_));

  f_rep_ = Teuchos::rcp(new vector_type(ns1_map_));
  u_rep_ = Teuchos::rcp(new vector_type(ns2_map_));

  if( 0 == comm_->getRank() ) {
    std::cout<<"periodic_bc::periodic_bc completed for:"<<std::endl
	     <<"    ns_id1_ = "<<ns_id1_<<std::endl
	     <<"    ns_id2_ = "<<ns_id2_<<std::endl<<std::endl;
  }

}

void periodic_bc::import_data(const Epetra_FEVector &f_full,
			      const Teuchos::RCP<const Epetra_Vector> u_full,
			      const int eqn_index ) const
{
  Teuchos::RCP<vector_type> u1 = Teuchos::RCP(new vector_type(node_map_,
                                              true));
  auto uView = u1->get1dView();
  for(int nn = 0; nn < node_map_->getLocalNumElements(); nn++) {
    u1->replaceLocalValue(nn, (*u_full)[numeqs_*nn+eqn_index]);
  }
  u_rep_->doImport(*u1, *importer2_, Tpetra::INSERT);

  Teuchos::RCP<vector_type> f1 = Teuchos::RCP(new vector_type(node_map_,
                                              true));
  for(int nn = 0; nn < node_map_->getLocalNumElements(); nn++) {
    f1->replaceLocalValue(nn, f_full[0][numeqs_*nn+eqn_index]);
  }
  f_rep_->doImport(*f1, *importer1_, Tpetra::INSERT);

}

Teuchos::RCP<const periodic_bc::map_type> periodic_bc::get_replicated_map(const int id){
  //cn I dont think there is any guarantee that the nodes will stay sorted after the
  //cn gather below.  It is enough that the two maps are in order wrt each other.
  //cn I don't believe that is guaranteed either.

  //cn allgather is in order of mpi rank
  //cn so it is possible that the order is not consistent wrt each other
  //cn if procid is not in the order of increasing coords
  //cn (more likely case in 3d rather than 2d)

  //cn also turns out that 
  //cn static Epetra_Map Epetra_Util::Create_Root_Map(const Epetra_Map &usermap,int root = 0 )
  //cn with root = -1 creates replicated map; we might look into this in future 	

  auto comm_ = Teuchos::DefaultComm<int>::getComm();

  Teuchos::RCP<const map_type> ns_map_;

  std::vector<Mesh::mesh_lint_t> node_num_map(mesh_->get_node_num_map());
  //cn the ids in sorted_node_set are local
  //int ns_size = mesh_->get_sorted_node_set(id).size(); 

  std::vector<int> ownedmap;

  for ( int j = 0; j < mesh_->get_sorted_node_set(id).size(); j++ ){
    int lid = mesh_->get_sorted_node_set_entry(id, j);
    Mesh::mesh_lint_t gid = node_num_map[lid];
    if ( node_map_->isNodeGlobalElement(gid) ) ownedmap.push_back(gid);
  }

  //int ns_size = static_cast<int>(ownedmap.size());
  int ns_size = ownedmap.size();
  int max_size = INT_MIN;
  Teuchos::reduceAll<int, int>(*comm_,
                     Teuchos::REDUCE_MAX,
                     1,
                     &ns_size,
                     &max_size);	

//  std::cout<<"max = "<<max_size<<" "<<newcomm_->getRank()<<std::endl;

  std::vector<Mesh::mesh_lint_t> gids(max_size,-99);

  for ( int j = 0; j < ns_size; j++ ){
    //cn this is probably lid on overlap map
    //int lid = mesh_->get_sorted_node_set_entry(id, j);
    //int gid = node_num_map[lid];//cn this is overlap
    Mesh::mesh_lint_t gid = ownedmap[j];
    //cn we could check here if the id is in the node_map_
    //cn this would eliminate duplicates
    //cn or remove the ghosts above
    gids[j] = gid;
  }//j

  int count = comm_->getSize() * max_size;
  std::vector<Mesh::mesh_lint_t> AllVals(count,-99);

  Teuchos::gatherAll(*comm_,
                     max_size,
                     &gids[0],
                     count,
                     &AllVals[0]);

  std::vector<global_ordinal_type> g_gids;

  for ( int j = 0; j < count; j++ ){
    if(-99 < AllVals[j]) {
      g_gids.push_back(static_cast<global_ordinal_type>(AllVals[j]));
    }
  }

  std::vector<global_ordinal_type> result(g_gids);
  Tpetra::global_size_t numEntries = result.size();
  const global_ordinal_type indexBase = 0;

  const Teuchos::ArrayView<global_ordinal_type> AV(result);
  ns_map_ = Teuchos::rcp(new map_type(numEntries,
                                      AV,
                                      indexBase,
                                      comm_));

  if(ns_map_->getGlobalNumElements() != ns_map_->getLocalNumElements() ){
    std::cout<<"periodic_bc::get_replicated_map ns_map_->getGlobalNumElements() != ns_map_->getLocalNumElements()"<<std::endl;
    exit(0);
  }
  if( 0 == comm_->getRank() )
    std::cout<<"periodic_bc::get_replicated_map completed for id: "<<id<<std::endl;
  //ns_map_->Print(std::cout);
  //exit(0);
  

  return ns_map_;
}


periodic_bc::~periodic_bc()
{
  //delete mesh_;
}

void periodic_bc::uniquifyWithOrder_set_remove_if(const std::vector<int>& input, std::vector<int>& output)
{
    std::set<int> seen;

    auto newEnd = remove_if(output.begin(), output.end(), [&seen](const int& value)
    {
        if (seen.find(value) != end(seen))
            return true;

        seen.insert(value);
        return false;
    });

    output.erase(newEnd, output.end());
}
