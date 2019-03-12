//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#include "projection.h"
#include "basis.hpp"

#include <iostream>

projection::projection(const Teuchos::RCP<const Epetra_Comm>& comm, 
		       const std::string meshNameString, 
		       const std::string dataNameString ) :
  comm_(comm),
  meshNameString_(meshNameString),
  dataNameString_(dataNameString),
  stride_(100)
{

  using Teuchos::rcp;

  int mypid = comm_->MyPID();
  int numproc = comm_->NumProc();
  //sourcemesh_ = new Mesh(mypid,numproc,false);
  sourcemesh_ = new Mesh((int)0,int(1),false);

  sourcemesh_->read_exodus((meshNameString ).c_str());

  int blk = 0;
  std::string elem_type=sourcemesh_->get_blk_elem_type(blk);
  bool quad_type = (0==elem_type.compare("QUAD4")) || (0==elem_type.compare("QUAD")) || (0==elem_type.compare("quad4")) || (0==elem_type.compare("quad"));
  bool hex_type= (0==elem_type.compare("HEX")) || (0==elem_type.compare("HEX8")) || (0==elem_type.compare("hex")) || (0==elem_type.compare("hex8")); 
  if( !(quad_type || hex_type) ){ // linear quad
    if( 0 == comm_->MyPID() )std::cout<<"Projection only supports bilinear quad and hex element types at this time."<<std::endl
	     <<elem_type<<" not supported."<<std::endl;
    exit(0);
  }

  sourcemesh_->compute_nodal_adj();

  //cn calling create_sorted_nodelist()
  //sourcemesh_->create_sorted_nodelist();
  //cn creates get_sorted_node_num_map with nodes order by increasing x,y,z
  //cn so we should be able to read Matt's files directly
  //cn since they are in that order

  std::vector<int> node_num_map(sourcemesh_->get_node_num_map());

  //std::vector<int> node_num_map(sourcemesh_->get_sorted_node_num_map());

  source_node_map_ = rcp(new Epetra_Map( node_num_map.size(),
					      node_num_map.size(),
				      &node_num_map[0],
				      0,
				      *comm_));
  //source_node_map_->Print(std::cout);
  source_node_ = Teuchos::rcp(new Epetra_Vector(*source_node_map_));
  //source_node_->Print(std::cout);

  //cn we might use epetra_localmap here???
  std::vector<int> elem_num_map(*(sourcemesh_->get_elem_num_map()));
  source_elem_map_ = Teuchos::rcp(new Epetra_Map( elem_num_map.size(), 
						       elem_num_map.size(),
						       &elem_num_map[0],
				      0,
				      *comm_));
  //source_elem_map_->Print(std::cout);
  source_elem_ = Teuchos::rcp(new Epetra_Vector(*source_elem_map_));
  //source_elem_->Print(std::cout);
  
  //source_node_->Print(std::cout);
  //exit(0);
}

void projection::fill_time_interp_values(const int timeindex, const double theta)
{
  int mypid = comm_->MyPID();
  if(mypid == 0) {
    read_file_time_interp(timeindex, theta);
    sourcemesh_->compute_nodal_patch_old();
    elem_to_node_avg();
    update_mesh_data();
  }
  source_node_->Reduce();
  //exit(0);
}
void projection::fill_initial_values()
{
  int mypid = comm_->MyPID();
  if(mypid == 0) {
    read_file();
    sourcemesh_->compute_nodal_patch_old();
    elem_to_node_avg();
    update_mesh_data();
  }
  source_node_->Reduce();
}
projection::~projection()
{
  delete sourcemesh_;
}

void projection::read_file_time_interp(const int timeindex, const double theta)
{

  //cn right now there is some kind of conflict with std::getline and nvcc
  //   I am #if ing it out for now, since this is used with truchas coupling
  exit(0);
#if 0

  int mypid = comm_->MyPID();

  if(mypid == 0) {
    std::vector<double> data;
    std::vector<double> data1;
    std::ifstream ifile(dataNameString_, std::ios::in);
    
    int i = 0;
    
    //cn this ignores comments
    std::string line;
    while (std::getline(ifile, line))
      {
	if (line[0] != '#' )
	  {
	    std::istringstream iss(line);
	    float num; // The number in the line
	    
	    //while the iss is a number 
	    while ((iss >> num))
	      {
		//std::cout<<i<<" "<<num<<" "<<theta<<std::endl;
		//look at the number
		if( (i >= timeindex*stride_) && (i < (timeindex+1)*stride_))
		  data.push_back(num);
		if( (i >= (timeindex+1)*stride_) && (i < (timeindex+2)*stride_))
		  data1.push_back(num);
		i = i+1;
		if(i > (timeindex+2)*stride_) break;
	      }
	  }
      }
    //std::cout<<data1.size()<<std::endl;
    if(data1.size() != data.size() ){
      std::cout<<"projection::read_file_time_interp: data1.size() != data.size()"<<std::endl<<std::endl<<std::endl;
      exit(0);
    }

    sourcemesh_->create_sorted_elemlist_yxz();
    std::vector<int> elem_num_map(sourcemesh_->get_sorted_elem_num_map());
    
    for( int i = 0; i < data.size(); i++){
      double val = (1.+theta)*data1[i]+theta*data[i];
      int gid = elem_num_map[i];
      source_elem_->ReplaceGlobalValues((int)1,
					&val,
					&gid 
					) ;	
    }
  }
#endif
  return;
}
void projection::read_file()
{
  //cn right now there is some kind of conflict with std::getline and nvcc
  //   I am #if ing it out for now, since this is used with truchas coupling
  exit(0);
#if 0
  int mypid = comm_->MyPID();

  if(mypid == 0) {
    std::vector<double> data;
    std::ifstream ifile(dataNameString_, std::ios::in);
    
    
    //cn this ignores comments
    std::string line;
    while (std::getline(ifile, line))
      {
	if (line[0] != '#' )
	  {
	    std::istringstream iss(line);
	    float num; // The number in the line
	    
	    //while the iss is a number 
	    while ((iss >> num))
	      {
		//look at the number
		data.push_back(num);
	      }
	  }
      }

    //cn the file is y across, x down then z    
    //cn not x across, y down


    sourcemesh_->create_sorted_elemlist_yxz();
    std::vector<int> elem_num_map(sourcemesh_->get_sorted_elem_num_map());
    
    for( int i = 0; i < data.size(); i++){
      double val = data[i];
      int gid = elem_num_map[i];
      source_elem_->ReplaceGlobalValues((int)1,
					&val,
					&gid 
					) ;	
    }
  }
  //source_elem_->Print(std::cout);
#endif
  return;
}
void projection::update_mesh_data()
{
  int mypid = comm_->MyPID();

  if(mypid == 0) {
    int num_nodes = source_node_map_->NumMyElements ();
    std::vector<double> ndata(num_nodes,0.);
    for (int nn=0; nn < num_nodes; nn++) {
      ndata[nn]=(*source_node_)[nn];
      //data[nn]=sourcemesh_->get_x(nn)+sourcemesh_->get_y(nn);
      //std::cout<<data[nn]<<std::endl;
    }
    std::string nstring_="nsource";
    sourcemesh_->add_nodal_field(nstring_);
    sourcemesh_->update_nodal_data(nstring_, &ndata[0]);
    
    //int num_elem = source_elem_map_->NumMyElements ();
    int num_elem = sourcemesh_->get_elem_num_map()->size();
    std::vector<double> edata(num_elem,0.);
    for (int nn=0; nn < num_elem; nn++) {
      edata[nn]=(*source_elem_)[nn];
    }
    std::string estring_="esource";
    sourcemesh_->add_elem_field(estring_);
    sourcemesh_->update_elem_data(estring_, &edata[0]);
#if 0    
    const char *outfilename = "sourcedata.e";
    int ex_id_ = sourcemesh_->create_exodus(outfilename);
    sourcemesh_->write_exodus(ex_id_);
    sourcemesh_->close_exodus(ex_id_);
#endif
  }
}

void projection::elem_to_node_avg()
{

  // Note that this currently invokes an arithmatic average;
  // for unstructured meshes we would need area or volume weighted average.


  int mypid = comm_->MyPID();

  //if(mypid == 0) {

    const int blk = 0;//for now
    
    for(int nn = 0; nn < sourcemesh_->get_num_my_nodes(); nn++ ){
      
      int num_elem_in_patch = sourcemesh_->get_nodal_patch(nn).size();
      
      double e_avg = 0;
      
      for(int ne = 0; ne < num_elem_in_patch; ne++){
	
	int elemlid = source_elem_map_->LID((sourcemesh_->get_nodal_patch(nn))[ne]);
	double val = (*source_elem_)[elemlid];
	
	e_avg += val;
	
      }//ne
      
      e_avg = e_avg/num_elem_in_patch;
      
      int nodegid = (sourcemesh_->get_node_num_map())[nn];
      source_node_->ReplaceGlobalValues ((int) 1, (int) 0, &e_avg, &nodegid);
    }//nn
    //}
  //exit(0);
}

bool projection::get_source_value(const double x, const double y, const double z, double &val)
{
  const int blk = 0;//for now
  std::string elem_type=sourcemesh_->get_blk_elem_type(blk);

  val = -99999999999999.;

  Basis * basis;

  if( (0==elem_type.compare("QUAD4")) || (0==elem_type.compare("QUAD")) || (0==elem_type.compare("quad4")) || (0==elem_type.compare("quad")) ){ // linear quad 
    basis = new BasisLQuad();
  }
  else if( (0==elem_type.compare("HEX8")) || (0==elem_type.compare("HEX")) || (0==elem_type.compare("hex8")) || (0==elem_type.compare("hex")) ){ // linear quad   
    basis = new BasisLHex();
  } 
  else {
    if( 0 == comm_->MyPID() )std::cout<<"Projection only supports bilinear and quadratic quad element types at this time."<<std::endl
	     <<elem_type<<" not supported."<<std::endl;
    exit(0);
  }
 
  //std::cout<<sourcemesh_->get_num_nodes_per_elem_in_blk(blk)<<" "<<sourcemesh_->get_num_elem_in_blk(blk)<<std::endl;
  //source_node_->Print(std::cout);
  //exit(0);


  //cn we need to save the results of the search here so we don't do it everytime
  //we could feed in the compute mesh elemid here and create an std::map with
  // compute mesh elemid and source mesh elemid 



  int n_nodes_per_elem = sourcemesh_->get_num_nodes_per_elem_in_blk(blk);
  for (int ne = 0; ne < sourcemesh_->get_num_elem_in_blk(blk); ne++){
    std::vector<double> xx(n_nodes_per_elem);
    std::vector<double> yy(n_nodes_per_elem);
    std::vector<double> zz(n_nodes_per_elem);
    std::vector<double> uu(n_nodes_per_elem);
    for(int k = 0; k < n_nodes_per_elem; k++){
      
      int nodeid = sourcemesh_->get_node_id(blk, ne, k);//cn appears this is the local id
      
      xx[k] =  sourcemesh_->get_x(nodeid);
      yy[k] =  sourcemesh_->get_y(nodeid);
      zz[k] =  sourcemesh_->get_z(nodeid);
      
      uu[k] = (*source_node_)[nodeid]; 
    }//k

    bool found = basis->evalBasis(&xx[0], &yy[0], &zz[0], &uu[0], x, y, z, val);
    //std::cout<<val<<std::endl;
    if (found) return true;

  }//ne
  std::cout<<"get_source_value:  not found"<<std::endl<<std::endl<<std::endl<<std::endl;
  return false;
}
