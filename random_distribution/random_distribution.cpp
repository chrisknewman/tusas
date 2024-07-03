//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////


#include "random_distribution.h"

#include <Tpetra_Map_decl.hpp>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_RCP.hpp>

#include <random>
#include <iostream>
#include <fstream>

random_distribution::random_distribution(Mesh *mesh,
					 const int ltpquadorder
					 )
{
  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
  const int blk = 0;
  std::string elem_type = mesh->get_blk_elem_type(blk);
  bool quad_type = (0==elem_type.compare("QUAD4")) || (0==elem_type.compare("QUAD")) 
    || (0==elem_type.compare("quad4")) || (0==elem_type.compare("quad")) 
    || (0==elem_type.compare("QUAD9")) || (0==elem_type.compare("quad9"));
  bool tri_type= (0==elem_type.compare("TRI3")) || (0==elem_type.compare("TRI")) 
    || (0==elem_type.compare("tri3")) || (0==elem_type.compare("tri"));
  //    || (0==elem_type.compare("TRI6")) || (0==elem_type.compare("tri6"));
  bool hex_type = (0==elem_type.compare("HEX8")) || (0==elem_type.compare("HEX")) 
    || (0==elem_type.compare("hex8")) || (0==elem_type.compare("hex")); 
  bool tet_type= (0==elem_type.compare("TETRA4")) || (0==elem_type.compare("TETRA")) 
    || (0==elem_type.compare("tetra4")) || (0==elem_type.compare("tetra"));

  if( quad_type ){
    ngp = ltpquadorder*ltpquadorder;
  }
  else if( hex_type ){
    ngp = ltpquadorder*ltpquadorder*ltpquadorder;;
  }
  else{
    if( 0 == comm_->getRank() )std::cout<<"random distribution only supports bilinear and quadratic quad and hex element types at this time."<<std::endl
	     <<elem_type<<" not supported."<<std::endl;
    exit(0);
  } 

  //cn this is the map of elements belonging to this processor
  std::vector<Mesh::mesh_lint_t> elem_num_map(*(mesh->get_elem_num_map()));

  const Tpetra::global_size_t numGlobalEntries = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  const Tpetra::Map<>::global_ordinal_type indexBase = 0;

  std::vector<Tpetra::Map<>::global_ordinal_type> my_global_elems(elem_num_map.size());
  for(int i = 0; i < elem_num_map.size(); i++){
    my_global_elems[i] = elem_num_map[i];
  }

  const Teuchos::ArrayView<Tpetra::Map<>::global_ordinal_type> AV(my_global_elems);
  
  Teuchos::RCP<Tpetra::Map<> > elem_map_;
  elem_map_ = Teuchos::rcp(new Tpetra::Map<>(numGlobalEntries,
					     AV,
					     indexBase,
					     comm_));
  num_elem = elem_map_->getLocalNumElements();

  gauss_val.resize(num_elem, std::vector<double>(ngp,0));
  compute_random(0);
  //std::cout<<comm_->MyPID()<<"   "<<elem_num_map.size()<<std::endl;
  //print();
}

random_distribution::~random_distribution()
{
}

void random_distribution::compute_random(const int nt)
{
  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
  const int mypid = comm_->getRank();
  int initial_seed = 12345*(nt+1)*(mypid+1);
  //int initial_seed = 12345*(nt+1);
  std::mt19937 gen(initial_seed); 
  std::uniform_int_distribution<> udist(1,10000); 
  //std::cout<<mypid<<"   "<<udist(gen)<<std::endl;

  for( int i=0; i<num_elem; i++){
    //const int gid = elem_map_->GID64(i)+1;
    //std::cout<<mypid<<"  "<<gid<<std::endl;
    int elem_seed = udist(gen);
    //int elem_seed = udist(gen)*gid;
    std::mt19937 mt(elem_seed);
    std::normal_distribution<> normal_dist(0,1);
    for(int ig=0;ig<ngp;ig++){
      gauss_val[i][ig]=normal_dist(mt);
    }
  }
//   print();
//   exit(0); 
  
  return;
}

void random_distribution::print() const
{
  auto comm_ = Teuchos::DefaultComm<int>::getComm(); 
  const int mypid = comm_->getRank();
  comm_->barrier();

  std::ofstream outfile;
  outfile.open("rand.txt", std::ios::out );

  for( int i=0; i<num_elem; i++){
    for(int ig=0;ig<ngp;ig++){
      std::cout<<mypid<<" "<<" "<<i<<" "<<ig<<" "<<gauss_val[i][ig]<<std::endl;
      outfile<<mypid<<" "<<" "<<i<<" "<<ig<<" "<<gauss_val[i][ig]<<std::endl;
    }
  }
  outfile.close();

  comm_->barrier();   
  return;
}

void random_distribution::compute_correlation() const 
{
   // verify suite of numbers in each element are uncorrelated
  double max_dot=0.;
  for(int i=0;i<num_elem;i++)
    {
      for(int j=0;j<i;j++)
        {
	  double dot=0.;
	  for(int ig=0;ig<ngp;ig++)
            {
	      dot+=gauss_val[i][ig]*gauss_val[j][ig];
            }
	  dot/=(double)ngp;
	  std::cout<<"Element "<<i<<","<<j<<": dot="<<dot<<std::endl;
	  max_dot = dot>max_dot ? dot : max_dot;
        }
    }    
  std::cout<<"Max. dot product: "<<max_dot<<std::endl;    // compute self-correlation for comparison
  double ref_dot=0.;
  for(int ig=0;ig<ngp;ig++)
    {
      ref_dot+=gauss_val[0][ig]*gauss_val[0][ig];
    }
  ref_dot/=(double)ngp;
  std::cout<<"Reference dot="<<ref_dot<<std::endl;    
  return;
}
