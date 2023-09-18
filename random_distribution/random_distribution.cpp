//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////


#include "random_distribution.h"

#include <random>

random_distribution::random_distribution(const Teuchos::RCP<const Epetra_Comm>& comm,
					 Mesh *mesh,
					 const int ltpquadorder
					 ):  
  comm_(comm)
{
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
    if( 0 == comm_->MyPID() )std::cout<<"random distribution only supports bilinear and quadratic quad and hex element types at this time."<<std::endl
	     <<elem_type<<" not supported."<<std::endl;
    exit(0);
  }

  //cn this is the map of elements belonging to this processor
  std::vector<Mesh::mesh_lint_t> elem_num_map(*(mesh->get_elem_num_map()));
  elem_map_ = Teuchos::rcp(new Epetra_Map(-1,
				      elem_num_map.size(),
				      &elem_num_map[0],
				      0,
				      *comm_));
  const int num_elem = elem_map_->NumMyElements();
  gauss_val.resize(num_elem, std::vector<double>(ngp,0));
  compute_random(0);
  //print();
}

random_distribution::~random_distribution()
{
}

void random_distribution::compute_random(const int nt)
{
  const int mypid = comm_->MyPID();
  int initial_seed = 12345*(nt+1)*(mypid+1);
  std::mt19937 gen(initial_seed); 
  std::uniform_int_distribution<> udist(1,10000); 
  const int num_elem = elem_map_->NumMyElements();

  for( int i=0; i<num_elem; i++){
    int elem_seed = udist(gen);
    std::mt19937 mt(elem_seed);
    std::normal_distribution<> normal_dist(0,1);
    for(int ig=0;ig<ngp;ig++){
      gauss_val[i][ig]=normal_dist(mt);
    }
  }   
  return;
}

void random_distribution::print() const
{
  const int mypid = comm_->MyPID();
  comm_->Barrier();

  const int num_elem = elem_map_->NumMyElements();
  for( int i=0; i<num_elem; i++){
    for(int ig=0;ig<ngp;ig++){
      std::cout<<mypid<<" "<<" "<<i<<" "<<ig<<" "<<gauss_val[i][ig]<<std::endl;
    }
  }
  comm_->Barrier();   
  return;
}

void random_distribution::compute_correlation() const 
{
   // verify suite of numbers in each element are uncorrelated
  const int num_elem = elem_map_->NumMyElements();
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
