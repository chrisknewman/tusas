//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#include "Mesh.h"
#include <iostream>
#include <algorithm>
#include "string.h"

#include "exodusII.h"

//#ifdef NEMESIS
#include "ne_nemesisI.h"
//#endif

/*
Public interface to the Mesh class
*/

//#ifdef NEMESIS
Mesh::Mesh( const int pid, const int np, const bool v ):
  proc_id(pid), nprocs(np), verbose(v) {}
//#else
Mesh::Mesh( const int pid, const bool v ):
  verbose(v) {}
//#endif

Mesh::~Mesh(){};

int Mesh::read_exodus(const char * filename){

  int comp_ws = sizeof(double);//cn send this to exodus to tell it we are using doubles
  int io_ws = 0;
  std::vector<int>::iterator a;

  num_nodal_fields = 0;
  num_elem_fields = 0;

  int ex_id = ex_open(filename,//cn open file
		      EX_READ,
		      &comp_ws,
		      &io_ws,
		      &exodus_version);

  if(ex_id < 0){

  	std::cerr << "Error: cannot open file " << filename << std::endl;
	exit(1);

  }

  char _title[MAX_LINE_LENGTH];

  int ex_err = 0;

  ex_err = ex_get_init(ex_id,//cn read header
		       _title,
		       &num_dim,
		       &num_nodes,
		       &num_elem,
		       &num_elem_blk,
		       &num_node_sets,
		       &num_side_sets);

  check_exodus_error(ex_err,"Mesh::read_exodus ex_get_init");

  title = _title;

  if(verbose)

    std::cout<<"=== ExodusII Read Info ==="<<std::endl
	     <<" ExodusII "<<exodus_version<<std::endl
	     <<" File "<<filename<<std::endl
	     <<" Exodus ID "<<ex_id<<std::endl
	     <<" comp_ws "<<comp_ws<<std::endl
	     <<" io_ws "<<io_ws<<std::endl              
	     <<" Title "<<title<<std::endl
	     <<" num_dim "<<num_dim<<std::endl
	     <<" num_nodes "<<num_nodes<<std::endl
	     <<" num_elem "<<num_elem<<std::endl
	     <<" num_elem_blk "<<num_elem_blk<<std::endl
	     <<" num_node_sets "<<num_node_sets<<std::endl
	     <<" num_side_sets "<<num_side_sets<<std::endl<<std::endl;

  //cn if nemesis, do we need different num_nodes for x,y,z?

  x.resize(num_nodes);
  y.resize(num_nodes);  
  z.resize(num_nodes);

  if( 1 < nprocs ){
    //#ifdef NEMESIS

    if ( num_dim == 2 ){
      
      ne_get_n_coord(ex_id, 1, num_nodes,
		     &x[0],
		     &y[0],
		     NULL);
      
      std::fill(z.begin(), z.end(), 0);
    }
    else {
      
      ne_get_n_coord(ex_id, 1, num_nodes,
		     &x[0],
		     &y[0],
		     &z[0]);
    }
    
    int num_proc;
    
    
    ne_get_init_info(ex_id, &num_proc, &nprocs_infile, &filetype);
    
    if(num_proc != nprocs){
      
      fprintf(stderr, "ERROR in file read: number of processors does not match number of input files\n");
      exit(-10);
      
    }
    
  }
  else {
    //#else
    
    if ( num_dim == 2 ){
      
      ex_err = ex_get_coord(ex_id,
			    &x[0],
			    &y[0],
			    0);
      
      std::fill(z.begin(), z.end(), 0);
    }
    else {
      
      ex_err = ex_get_coord(ex_id,
			    &x[0],
			    &y[0],
			    &z[0]); 
    }
    
    check_exodus_error(ex_err,"Mesh::read_exodus ex_get_coord");
  }
  //#endif

  blk_ids.resize(num_elem_blk);
  num_elem_in_blk.resize(num_elem_blk);
  num_node_per_elem_in_blk.resize(num_elem_blk);
  connect.resize(num_elem_blk);

  ex_err = ex_get_elem_blk_ids(ex_id,//cn read block ids
			       &blk_ids[0]);

  check_exodus_error(ex_err,"Mesh::read_exodus ex_get_elem_blk_ids");

  for (int i = 0; i < num_elem_blk; i++){//cn loop over blocks   we should have block ids start at 0

    char elem_type[MAX_STR_LENGTH];
    int num_attr = 0;

    ex_err = ex_get_elem_block(ex_id,//cn read elems for this blk[i]
			       blk_ids[i],
			       elem_type,
			       &num_elem_in_blk[i],
			       &num_node_per_elem_in_blk[i],
			       &num_attr);

    check_exodus_error(ex_err,"Mesh::read_exodus ex_get_elem_block");

    blk_elem_type.push_back(elem_type);

    if(verbose)

      std::cout<<" +++ Element Block Info +++"<<std::endl
  	       <<"  i "<<i<<std::endl
	       <<"  block_ids[i] "<<blk_ids[i]<<std::endl
	       <<"  elem_type "<<blk_elem_type[i]<<std::endl
	       <<"  num_elem_this_blk "<<num_elem_in_blk[i]<<std::endl
	       <<"  num_node_per_elem "<<num_node_per_elem_in_blk[i]<<std::endl
	       <<"  num_attr "<<num_attr<<std::endl<<std::endl;

    //cn get connectivity info here

    connect[i].resize(num_elem_in_blk[i]*num_node_per_elem_in_blk[i]);

    if( 1 < nprocs ){
      //#ifdef NEMESIS
      
      ne_get_n_elem_conn(ex_id, blk_ids[i], 1, num_elem_in_blk[i], &connect[i][0]);

      //#else
    }
    else {
      ex_err = ex_get_elem_conn(ex_id,
				blk_ids[i],
				&connect[i][0]);
      
      check_exodus_error(ex_err,"Mesh::read_exodus ex_get_elem_conn");
    }
    //#endif

    for(a = connect[i].begin(); a != connect[i].end(); a++)  // fix FORTRAN indexing

          (*a)--;

  }

  if(num_side_sets > 0){

    ss_ids.resize(num_side_sets);
    num_sides_per_ss.resize(num_side_sets);
    num_df_per_ss.resize(num_side_sets);

    ss_ctr_list.resize(num_side_sets);
    ss_node_list.resize(num_side_sets);
    ss_elem_list.resize(num_side_sets);
    ss_side_list.resize(num_side_sets);

    side_set_node_map.resize(num_nodes);

    std::fill(side_set_node_map.begin(), side_set_node_map.end(), -1); // initialize it

    ex_err = ex_get_side_set_ids(ex_id,//cn side set ids
				 &ss_ids[0]);

    check_exodus_error(ex_err,"Mesh::read_exodus ex_get_side_set_ids");
  
    for (int i = 0; i < num_side_sets; i++){//cn loop over sidesets
  
      ex_err = ex_get_side_set_param(ex_id,
  				   ss_ids[i],
  				   &num_sides_per_ss[i],
  				   &num_df_per_ss[i]);

      check_exodus_error(ex_err,"Mesh::read_exodus ex_get_side_set_param");

      ss_ctr_list[i].resize(num_sides_per_ss[i]);
      ss_node_list[i].resize(num_df_per_ss[i]);

      ss_elem_list[i].resize(num_sides_per_ss[i]);
      ss_side_list[i].resize(num_sides_per_ss[i]);

      if(verbose)

        std::cout<<" +++ Sideset Info +++"<<std::endl
  	           <<"  i "<<i<<std::endl
  	           <<"  ss_ids[i] "<<ss_ids[i]<<std::endl
  	           <<"  num_sides_per_set[i] "<<num_sides_per_ss[i]<<std::endl
  	           <<"  num_df_per_sideset[i] "<<num_df_per_ss[i]<<std::endl<<std::endl;

        ex_err = ex_get_side_set(ex_id, ss_ids[i], &ss_elem_list[i][0], &ss_side_list[i][0]);
  
        ex_err = ex_get_side_set_node_list(ex_id, ss_ids[i], &ss_ctr_list[i][0], &ss_node_list[i][0]);

        for(a = ss_node_list[i].begin(); a != ss_node_list[i].end(); a++){  // fix FORTRAN indexing

            (*a)--;

            side_set_node_map[*a] = ss_ids[i];

	}
  
    } // end loop over side sets

  
  } // end if sidesets > 0

  if(num_node_sets > 0){

    ns_ids.resize(num_node_sets);
    num_nodes_per_ns.resize(num_node_sets);
    num_df_per_ns.resize(num_node_sets);
    ns_node_list.resize(num_node_sets);

    node_set_map.resize(num_nodes);

    std::fill(node_set_map.begin(), node_set_map.end(), -1); // initialize it

    ex_err = ex_get_node_set_ids(ex_id,
  				 &ns_ids[0]);

    check_exodus_error(ex_err,"Mesh::read_exodus ex_get_node_set_ids");
  
    for (int i = 0; i < num_node_sets; i++){//cn loop over sidesets
  
      ex_err = ex_get_node_set_param(ex_id,
  				   ns_ids[i],
  				   &num_nodes_per_ns[i],
  				   &num_df_per_ns[i]);

      check_exodus_error(ex_err,"Mesh::read_exodus ex_get_node_set_param");

      ns_node_list[i].resize(num_nodes_per_ns[i]);

      if(verbose)

        std::cout<<" +++ Nodeset Info +++"<<std::endl
  	           <<"  i "<<i<<std::endl
  	           <<"  ns_ids[i] "<<ns_ids[i]<<std::endl
  	           <<"  num_nodes_per_set[i] "<<num_nodes_per_ns[i]<<std::endl
  	           <<"  num_df_per_sideset[i] "<<num_df_per_ns[i]<<std::endl<<std::endl;
  
  
//#ifdef NEMESIS

//        ex_err = ne_get_n_node_set(ex_id, ns_ids[i], 1, num_nodes_per_ns[i], &ns_node_list[i][0]);

//#else

        ex_err = ex_get_node_set(ex_id, ns_ids[i], &ns_node_list[i][0]);

//#endif

        for(a = ns_node_list[i].begin(); a != ns_node_list[i].end(); a++){

           // fix FORTRAN indexing

            (*a)--;

           // fill the hash table

          node_set_map[*a] = ns_ids[i];

        }
  
    } // end loop over node sets
  
  } // end if nodesets > 0

  node_num_map.resize(num_nodes);
  elem_num_map.resize(num_elem);

  if( 1 < nprocs ){
    //#ifdef NEMESIS
    
    ne_get_init_global(ex_id, &ne_num_global_nodes, &ne_num_global_elems, &ne_num_global_elem_blks,
		       &ne_num_global_node_sets, &ne_num_global_side_sets);
    
    ne_get_loadbal_param(ex_id, &num_internal_nodes, &num_border_nodes, &num_external_nodes,
			 &num_internal_elems, &num_border_elems, &num_node_cmaps, &num_elem_cmaps, proc_id);
    
    ne_get_n_node_num_map(ex_id, 1, num_nodes, &node_num_map[0]);
    for(a = node_num_map.begin(); a != node_num_map.end(); a++) (*a)--;
    ne_get_n_elem_num_map(ex_id, 1, num_elem, &elem_num_map[0]);
    for(a = elem_num_map.begin(); a != elem_num_map.end(); a++) (*a)--;
    
    elem_mapi.resize(num_internal_elems);
    elem_mapb.resize(num_border_elems);
    
    ne_get_elem_map(ex_id, &elem_mapi[0], &elem_mapb[0], proc_id);
    for(a = elem_mapi.begin(); a != elem_mapi.end(); a++) (*a)--;
    for(a = elem_mapb.begin(); a != elem_mapb.end(); a++) (*a)--;
    
    node_mapi.resize(num_internal_nodes);
    node_mapb.resize(num_border_nodes);
    node_mape.resize(num_external_nodes);
    
    ne_get_node_map(ex_id, &node_mapi[0], &node_mapb[0], &node_mape[0], proc_id);
    for(a = node_mapi.begin(); a != node_mapi.end(); a++) (*a)--;
    for(a = node_mapb.begin(); a != node_mapb.end(); a++) (*a)--;
    for(a = node_mape.begin(); a != node_mape.end(); a++) (*a)--;

    my_node_num_map = node_mapi;  // nodes this proc is responsible for

    if(proc_id == 0){
      my_node_num_map.insert(my_node_num_map.end(), node_mapb.begin(), node_mapb.end());
    }

    if(ne_num_global_node_sets > 0){
      
      global_ns_ids.resize(ne_num_global_node_sets);
      num_global_node_counts.resize(ne_num_global_node_sets);
      num_global_node_df_counts.resize(ne_num_global_node_sets);
      
      ne_get_ns_param_global(ex_id, &global_ns_ids[0], &num_global_node_counts[0],
			     &num_global_node_df_counts[0]);
      
    }
    
    if(ne_num_global_side_sets > 0){
      
      global_ss_ids.resize(ne_num_global_side_sets);
      num_global_side_counts.resize(ne_num_global_side_sets);
      num_global_side_df_counts.resize(ne_num_global_side_sets);
      
      ne_get_ss_param_global(ex_id, &global_ss_ids[0], &num_global_side_counts[0],
			     &num_global_side_df_counts[0]);
      
    }
    
    global_elem_blk_ids.resize(ne_num_global_elem_blks);
    global_elem_blk_cnts.resize(ne_num_global_elem_blks);
    
    ne_get_eb_info_global(ex_id, &global_elem_blk_ids[0], &global_elem_blk_cnts[0]);
    
    node_cmap_ids.resize(num_node_cmaps);
    node_cmap_node_cnts.resize(num_node_cmaps);
    
    elem_cmap_ids.resize(num_elem_cmaps);
    elem_cmap_elem_cnts.resize(num_elem_cmaps);
    
    ne_get_cmap_params(ex_id, &node_cmap_ids[0], &node_cmap_node_cnts[0],
		       &elem_cmap_ids[0], &elem_cmap_elem_cnts[0],
		       proc_id);
    
    node_ids_in_cmap.resize(num_node_cmaps);
    n_proc_ids_in_cmap.resize(num_node_cmaps);
    
    for(int i = 0; i < num_node_cmaps; i++){
      
      node_ids_in_cmap[i].resize(node_cmap_node_cnts[i]);
      n_proc_ids_in_cmap[i].resize(node_cmap_node_cnts[i]);
      
      ne_get_node_cmap(ex_id, node_cmap_ids[i], &node_ids_in_cmap[i][0], &n_proc_ids_in_cmap[i][0], proc_id);

    }

    elem_ids_in_cmap.resize(num_elem_cmaps);
    e_side_ids_in_cmap.resize(num_elem_cmaps);
    e_proc_ids_in_cmap.resize(num_elem_cmaps);
    
    for(int i = 0; i < num_elem_cmaps; i++){
      
      elem_ids_in_cmap[i].resize(elem_cmap_elem_cnts[i]);
      e_side_ids_in_cmap[i].resize(elem_cmap_elem_cnts[i]);
      e_proc_ids_in_cmap[i].resize(elem_cmap_elem_cnts[i]);
      
      ne_get_elem_cmap(ex_id, elem_cmap_ids[i], &elem_ids_in_cmap[i][0], 
		       &e_side_ids_in_cmap[i][0], &e_proc_ids_in_cmap[i][0], proc_id);
      
    }
    
    #if 0
    
    my_node_num_map = node_mapi;  // start with the nodes internal to this processor
    
  for(int i = 0; i < num_node_cmaps; i++){

	std::vector<int> cmap_node_ids(node_cmap_node_cnts[i]);
	std::vector<int> cmap_node_procids(node_cmap_node_cnts[i]);
	ne_get_node_cmap(ex_id, node_cmap_ids[i], &cmap_node_ids[0], &cmap_node_procids[0], proc_id);

	// build the node array for the nodes local to this processor
	//
	// HACK ALERT!!!!
	//
	// Put nodes on this processor that it shares with a lower numbered processor
	
	for(int j = 0; j < node_cmap_node_cnts[i]; j++){

		if(cmap_node_procids[j] > proc_id) // add the node to my_node_map

			my_node_num_map.push_back(cmap_node_ids[j]-1);
	}

  }
//    std::sort(my_node_num_map.begin(), my_node_num_map.end());
//    my_node_num_map.erase(std::unique(my_node_num_map.begin(), my_node_num_map.end()), my_node_num_map.end());
  #endif

  }
  else {
    //#else
    
    ex_err = ex_get_node_num_map(ex_id, &node_num_map[0]);
    for(a = node_num_map.begin(); a != node_num_map.end(); a++) (*a)--;
    ex_err = ex_get_map(ex_id, &elem_num_map[0]);
    for(a = elem_num_map.begin(); a != elem_num_map.end(); a++) (*a)--;
    
    my_node_num_map = node_num_map;  // same in serial
  }
  //#endif
    
  ex_err = close_exodus(ex_id);//cn close file
  
  if(verbose){
    
    std::cout << "There are " << num_elem << " elements in this mesh." << std::endl;
    std::cout<<"=== End ExodusII Read Info ==="<<std::endl<<std::endl;
    
  }
  
  
  return 0;
  
}

/*
Compute nodal adjacency for a standard serial matrix graph
note that we do not store the diagonal--maybe we should
*/

void Mesh::compute_nodal_adj(){

  std::vector<int> nodal_adj_idx;
  std::vector<int> nodal_adj_array;

  nodal_adj.resize(num_nodes, std::vector<int>(0));
  nodal_adj_idx.resize(num_nodes + 1);
  nodal_adj_idx[0] = 0;   //probably wont work in parallel, or need to start somewhere else

  if(verbose)

    std::cout<<"=== void mesh::compute_nodal_adj ==="<<std::endl<<" nodal_adj"<<std::endl<<"  ";

  for (int blk = 0; blk < num_elem_blk; blk++){

    std::vector<int> temp(num_node_per_elem_in_blk[blk]);

    for(int i = 0; i < num_elem_in_blk[blk]; i++){

      for (int j = 0; j < num_node_per_elem_in_blk[blk]; j++){

        temp[j] = connect[blk][i * num_node_per_elem_in_blk[blk] + j]; //load up nodes on each element
        //std::cout<<temp[j]<<std::endl;

      }

      for (int j = 0; j < num_node_per_elem_in_blk[blk]; j++){

        for(int k = 0; k < num_node_per_elem_in_blk[blk]; k++){

          if(temp[j] != temp[k] ){ //cn skip the diagonal and load up nodes

	    nodal_adj[temp[j]].push_back(temp[k]);

	    //std::cout<<temp[j]<<","<<temp[k]<<std::endl;
  	  }

        }
      }
    }
  }

  for(int i = 0; i < num_nodes; i++) {

    //cn sort and remove duplicates

    std::sort(nodal_adj[i].begin(), nodal_adj[i].end());

    std::vector<int>::iterator unique_end =

      std::unique(nodal_adj[i].begin(), nodal_adj[i].end());

    nodal_adj[i].erase(unique_end, nodal_adj[i].end());
    
    nodal_adj_idx[i + 1] = nodal_adj_idx[i] + nodal_adj[i].size();

    for( int j = nodal_adj_idx[i]; j < nodal_adj_idx[i + 1]; j++)

      nodal_adj_array.push_back(nodal_adj[i][j - nodal_adj_idx[i]]);

    if(verbose){

      for( int j = 0;  j < nodal_adj[i].size(); j++)

	std::cout<<nodal_adj[i][j]<<" ";

      std::cout<<std::endl<<"  ";

    }
  }

  if(verbose){

    std::cout<<std::endl<<" nodal_adj_idx"<<std::endl;

    for( int i = 0;i < num_nodes + 1; i++)

      std::cout<<"  "<<nodal_adj_idx[i]<<std::endl;

    std::cout<<std::endl<<" nodal_adj_array"<<std::endl;

    for( int i = 0; i < nodal_adj_idx[num_nodes]; i++)

      std::cout<<"  "<<nodal_adj_array[i]<<std::endl;

    std::cout<<"=== End void mesh::compute_nodal_adj ==="<<std::endl<<std::endl;
  }

};

int Mesh::get_boundary_status(int blk, int elem){

	int status;

	for(int i = 0; i < num_node_per_elem_in_blk[blk]; i++)

		if(status = node_set_map[connect[blk][elem * num_node_per_elem_in_blk[blk] + i]] >= 0)

			return status;

	return -1;

}

int Mesh::get_node_boundary_status(int nodeid){

	int status;

	if(status = node_set_map[nodeid] >= 0)

		return status;

	return -1;

}

/*
Private interface to Mesh class
*/

int Mesh::close_exodus(int ex_id){

  int ex_err = ex_update (ex_id);

  check_exodus_error(ex_err,"Mesh::close_exodus ex_close");

  ex_err = ex_close(ex_id);

  check_exodus_error(ex_err,"Mesh::close_exodus ex_close");

  if(verbose)

    std::cout<<"=== ExodusII Close Info ==="<<std::endl
	     <<" Exodus ID "<<ex_id<<std::endl
	     <<"=== End ExodusII Close Info ==="<<std::endl<<std::endl;

  return ex_err;

};

void Mesh::check_exodus_error(const int ex_err, const std::string msg){

  if (ex_err < 0)

    std::cout<<"ExodusII error:  "<<msg<<"  ex_err = "<<ex_err<<std::endl<<std::endl;

  return;

};

int Mesh::write_exodus(const char * filename){

   int ex_id = create_exodus(filename);

   write_nodal_coordinates_exodus(ex_id);
   write_element_blocks_exodus(ex_id);
   write_nodal_data_exodus(ex_id);

return 0;

}

int Mesh::write_exodus(const int ex_id){

  //int ex_id = create_exodus(filename);

   write_nodal_coordinates_exodus(ex_id);
   write_element_blocks_exodus(ex_id);
   write_nodal_data_exodus(ex_id);

   return 0;

}

int Mesh::write_exodus(const int ex_id, const int counter, const double time){
  int error = 0;
  error = write_nodal_coordinates_exodus(ex_id);
  //std::cout<<error<<std::endl;
  error = write_element_blocks_exodus(ex_id);
  //std::cout<<error<<std::endl;
  error = write_nodal_data_exodus(ex_id,counter);
  error = write_elem_data_exodus(ex_id,counter);
  //std::cout<<error<<std::endl;
  error = ex_put_time(ex_id,counter,&time);
  //std::cout<<error<<std::endl;

  return error;

}
int Mesh::read_last_step_exodus(const int ex_id, int &timestep){
  float ret_float = 0.0;
  char *ret_char = '\0';
  int error = ex_inquire (ex_id, EX_INQ_TIME, &timestep, &ret_float,ret_char);
  return error;
}

int Mesh::read_time_exodus(const int ex_id, const int counter, double &time){

   int error = ex_get_time(ex_id,counter,&time);

   return error;
}

int Mesh::write_nodal_coordinates_exodus(int ex_id)
{
  
  char ** var_names;
  int ex_err;
  std::vector<int> tmpvec, tmpvec1, tmpvec2;
  std::vector<int>::iterator a;

  if( 1 < nprocs ){
    //#ifdef NEMESIS

    ne_put_init_global(ex_id, ne_num_global_nodes, ne_num_global_elems, ne_num_global_elem_blks,
		       ne_num_global_node_sets, ne_num_global_side_sets);
    
    ne_put_init_info(ex_id, nprocs, nprocs_infile, &filetype);
    
    ne_put_loadbal_param(ex_id, num_internal_nodes, num_border_nodes, num_external_nodes,
			 num_internal_elems, num_border_elems, num_node_cmaps, num_elem_cmaps, proc_id);
    
    tmpvec = node_num_map;
    for(a = tmpvec.begin(); a != tmpvec.end(); a++) (*a)++;
    ne_put_n_node_num_map(ex_id, 1, num_nodes, &tmpvec[0]);
    
    tmpvec = elem_num_map;
    for(a = tmpvec.begin(); a != tmpvec.end(); a++) (*a)++;
    ne_put_n_elem_num_map(ex_id, 1, num_elem, &tmpvec[0]);
    
    tmpvec = elem_mapi;
    for(a = tmpvec.begin(); a != tmpvec.end(); a++) (*a)++;
    tmpvec1 = elem_mapb;
    for(a = tmpvec1.begin(); a != tmpvec1.end(); a++) (*a)++;
    ne_put_elem_map(ex_id, &tmpvec[0], &tmpvec1[0], proc_id);
    
    tmpvec = node_mapi;
    for(a = tmpvec.begin(); a != tmpvec.end(); a++) (*a)++;
    tmpvec1 = node_mapb;
    for(a = tmpvec1.begin(); a != tmpvec1.end(); a++) (*a)++;
    tmpvec2 = node_mape;
    for(a = tmpvec2.begin(); a != tmpvec2.end(); a++) (*a)++;
    ne_put_node_map(ex_id, &tmpvec[0], &tmpvec1[0], &tmpvec2[0], proc_id);
    
    ne_put_n_coord(ex_id, 1, num_nodes, &x[0], &y[0], &z[0]);
    
    if(ne_num_global_node_sets > 0)
      
      ne_put_ns_param_global(ex_id, &global_ns_ids[0], &num_global_node_counts[0],
			     &num_global_node_df_counts[0]);
    
    if(ne_num_global_side_sets > 0)
      
      ne_put_ss_param_global(ex_id, &global_ss_ids[0], &num_global_side_counts[0],
			     &num_global_side_df_counts[0]);
    
    ne_put_eb_info_global(ex_id, &global_elem_blk_ids[0], &global_elem_blk_cnts[0]);
    
    ne_put_cmap_params(ex_id, &node_cmap_ids[0], &node_cmap_node_cnts[0],
		       &elem_cmap_ids[0], &elem_cmap_elem_cnts[0],
		       proc_id);
    
    for(int i = 0; i < num_node_cmaps; i++){
      
      ne_put_node_cmap(ex_id, node_cmap_ids[i], &node_ids_in_cmap[i][0], &n_proc_ids_in_cmap[i][0], proc_id);
      
    }
    
    for(int i = 0; i < num_elem_cmaps; i++){
      
      ne_put_elem_cmap(ex_id, elem_cmap_ids[i], &elem_ids_in_cmap[i][0], 
		       &e_side_ids_in_cmap[i][0], &e_proc_ids_in_cmap[i][0], proc_id);
      
    }
    
  }
  else {
    //#else
    
    ex_err = ex_put_coord(ex_id, &x[0], &y[0], &z[0]);
  }
  //#endif
    
  if(num_node_sets > 0){

//    ex_err = ex_put_node_set_ids(ex_id,
//  				 &ns_ids[0]);

    for (int i = 0; i < num_node_sets; i++){//cn loop over sidesets
  
        ex_err = ex_put_node_set_param(ex_id,
  				   ns_ids[i],
  				   num_nodes_per_ns[i],
  				   num_df_per_ns[i]);

	tmpvec = ns_node_list[i];
        for(a = tmpvec.begin(); a != tmpvec.end(); a++) (*a)++;

	if( 1 < nprocs ){
	  //#ifdef NEMESIS

        ex_err = ne_put_n_node_set(ex_id, ns_ids[i], 1, num_nodes_per_ns[i], &tmpvec[0]);
	}
	else {
	  //#else

        ex_err = ex_put_node_set(ex_id, ns_ids[i], &tmpvec[0]);
	}
	//#endif

        }
  
  } // end if nodesets > 0

  if(num_side_sets > 0){

//    ex_err = ex_get_side_set_ids(ex_id,//cn side set ids
//				 &ss_ids[0]);

  
    for (int i = 0; i < num_side_sets; i++){//cn loop over sidesets
  
        ex_err = ex_put_side_set_param(ex_id,
  				   ss_ids[i],
  				   num_sides_per_ss[i],
  				   num_df_per_ss[i]);

        ex_err = ex_put_side_set(ex_id, ss_ids[i], &ss_elem_list[i][0], &ss_side_list[i][0]);

//	tmpvec = ss_node_list[i];
//        for(a = tmpvec.begin(); a != tmpvec.end(); a++) (*a)++;

//        ex_err = ex_put_side_set_node_list(ex_id, ss_ids[i], &ss_ctr_list[i][0], &tmpvec[0]);

    } // end loop over side sets
  
  } // end if sidesets > 0

  var_names = new char*[3];
  var_names[0] = new char[4];
  var_names[1] = new char[4];
  var_names[2] = new char[4];

  strcpy(var_names[0], "\"x\"");
  strcpy(var_names[1], "\"y\"");
  strcpy(var_names[2], "\"z\"");

  ex_err = ex_put_coord_names(ex_id, var_names);

  delete [] var_names[0];
  delete [] var_names[1];
  delete [] var_names[2];
  delete [] var_names;


  return ex_err;

}

int Mesh::write_element_blocks_exodus(int ex_id){

  int ex_err;

  for(int i = 0; i < num_elem_blk; i++){

    ex_err = ex_put_elem_block(ex_id, blk_ids[i], &blk_elem_type[i][0], num_elem_in_blk[i], 
             num_node_per_elem_in_blk[i], 0);

    std::vector<int> connect_tmp(num_node_per_elem_in_blk[i] * num_elem_in_blk[i]);

    for ( int j = 0; j < num_node_per_elem_in_blk[i] * num_elem_in_blk[i]; j++ )

      connect_tmp[j] = connect[i][j] + 1;


  if( 1 < nprocs ){
    //#ifdef NEMESIS

    ne_put_n_elem_conn(ex_id, blk_ids[i], 1, num_elem_in_blk[i], &connect_tmp[0]);
  }
  else {
    //#else

    ex_err = ex_put_elem_conn(ex_id, blk_ids[i], &connect_tmp[0]);
  }
  //#endif
  }


  return ex_err;

}

int Mesh::write_nodal_data_exodus(int ex_id){

  int ex_err;
  char **var_names;


  if(verbose)

    std::cout<<"=== Write Nodal Data Exodus ==="<<std::endl
	     <<" num_nodal_fields "<<num_nodal_fields<<std::endl;

  if(num_nodal_fields == 0) return 0;

  ex_err = ex_put_var_param (ex_id, "N", num_nodal_fields);

  var_names = new char*[num_nodal_fields];

  for(int i = 0; i < num_nodal_fields; i++){

    var_names[i] = (char *)&nodal_field_names[i][0];

    if(verbose)

      std::cout<<" name  "<<var_names[i]<<std::endl<<std::endl;

  }

  ex_err = ex_put_var_names (ex_id, "N", num_nodal_fields, var_names);

  for(int i = 0; i < num_nodal_fields; i++)

    ex_err = ex_put_nodal_var (ex_id, 1, i + 1, num_nodes, &nodal_fields[i][0]);


  delete [] var_names;

  return ex_err;

}

int Mesh::write_nodal_data_exodus(int ex_id, int counter){

  int ex_err;
  char **var_names;


  if(verbose)

    std::cout<<"=== Write Nodal Data Exodus ==="<<std::endl
	     <<" num_nodal_fields "<<num_nodal_fields<<std::endl;

  if(num_nodal_fields == 0) return 0;

  ex_err = ex_put_var_param (ex_id, "N", num_nodal_fields);

  var_names = new char*[num_nodal_fields];

  for(int i = 0; i < num_nodal_fields; i++){

    var_names[i] = (char *)&nodal_field_names[i][0];

    if(verbose)

      std::cout<<" name  "<<var_names[i]<<std::endl<<std::endl;

  }

  ex_err = ex_put_var_names (ex_id, "N", num_nodal_fields, var_names);

  for(int i = 0; i < num_nodal_fields; i++){

    ex_err = ex_put_nodal_var (ex_id, counter, i + 1, num_nodes, &nodal_fields[i][0]);
    //for(int j = 0; j<(nodal_fields[i]).size();j++ ) std::cout<<nodal_fields[i][j]<<std::endl;
  
  }

  delete [] var_names;

  return ex_err;

}

int Mesh::write_elem_data_exodus(int ex_id, int counter){
  int ex_err;
  char **var_names;


  if(verbose)

    std::cout<<"=== Write Nodal Data Exodus ==="<<std::endl
	     <<" num_nodal_fields "<<num_nodal_fields<<std::endl;

  if(num_elem_fields == 0) return 0;

  ex_err = ex_put_var_param (ex_id, "E", num_elem_fields);

  var_names = new char*[num_elem_fields];

  for(int i = 0; i < num_elem_fields; i++){

    var_names[i] = (char *)&elem_field_names[i][0];

    if(verbose)

      std::cout<<" name  "<<var_names[i]<<std::endl<<std::endl;

  }

  int blk = 1; //hack
  ex_err = ex_put_var_names (ex_id, "E", num_elem_fields, var_names);

  for(int i = 0; i < num_elem_fields; i++){

    ex_err = ex_put_elem_var (ex_id, counter, i + 1, blk,num_elem, &elem_fields[i][0]);

    //(exoid,time_step,elem_var_index,elem_blk_id,num_elem_this_blk,elem_var_vals)
    //for(int j = 0; j<(nodal_fields[i]).size();j++ ) std::cout<<nodal_fields[i][j]<<std::endl;
  
  }

  delete [] var_names;

  return ex_err;

}


int Mesh::read_num_proc_nemesis(int ex_id, int *nproc){
  int num_proc_in_file;
  char ftype;
  int ex_err = ne_get_init_info(ex_id,nproc,&num_proc_in_file,&ftype);
  return ex_err;
}

int Mesh::read_nodal_data_exodus(const int ex_id, const int timestep, const int index, double *data){
  int ex_err = ex_get_nodal_var (ex_id, timestep, index, num_nodes, &data[0]);
  return ex_err;
}

int Mesh::read_nodal_data_exodus(const int ex_id, const int timestep, std::string name, double *data){
  int index = get_nodal_field_index(name) + 1;//exodus starts at 1
  int ex_err = ex_get_nodal_var (ex_id, timestep, index, num_nodes, &data[0]);
  return ex_err;
}

int Mesh::get_nodal_field_index(std::string name){
  int index = -1;
  for (int i = 0; i < num_nodal_fields; i++){
    if(name == nodal_field_names[i]){
      index = i;
    }
  }
  if(0 > index){
    std::cout<<name<<" not found"<<std::endl<<std::endl;
    exit(0);
  }
  return index;
}


/*
int Mesh::add_nodal_data(std::string &name, std::vector<double> &data){

   if(num_nodes != data.size()){

      std::cout<<"ERROR in add_node_data: node data field differs in length from the mesh"<<std::endl;
      exit(0);

   }

   num_nodal_fields++;

//   nodal_field_names.push_back(name);
//   nodal_fields.push_back(data);

   return 1;

}
*/
//int Mesh::add_nodal_data(std::basic_string<char, std::char_traits<char> > name, double *data){return 1;}
int Mesh::add_nodal_data(std::string name, std::vector<double> &data){


   num_nodal_fields++;

   nodal_field_names.push_back(name);
   nodal_fields.push_back(data);

   if(verbose)

    std::cout<<"=== Add nodal fields ==="<<std::endl
	     <<" num_nodal_fields "<<num_nodal_fields<<std::endl
	     <<" sizeof nodal_field_names "<<nodal_field_names.size()<<std::endl
	     <<" sizeof nodal_fields "<<nodal_fields.size()<<std::endl<<std::endl;

   return 1;

}

int Mesh::add_nodal_field(const std::string name){
  //if( 0 == proc_id ){
    num_nodal_fields++;
    
    nodal_field_names.push_back(name);
    nodal_fields.resize(num_nodal_fields);
    if(verbose)
      
      std::cout<<"=== add nodal field ==="<<std::endl
	       <<" num_nodal_fields "<<num_nodal_fields<<std::endl
	       <<" sizeof nodal_field_names "<<nodal_field_names.size()<<std::endl;
    //}
  return 1;

}

int Mesh::add_elem_field(const std::string name){
    num_elem_fields++;
   
    elem_field_names.push_back(name);
    elem_fields.resize(num_elem_fields);
    if(verbose)
      
      std::cout<<"=== add elem field ==="<<std::endl
	       <<" num_elem_fields "<<num_elem_fields<<std::endl
	       <<" sizeof elem_field_names "<<elem_field_names.size()<<std::endl;
  return 1;

}


int Mesh::update_nodal_data(const std::string name, const double *data){

  for (int i = 0; i < num_nodal_fields; i++){
    if(name == nodal_field_names[i]){
      //std::cout<<"found"<<std::endl;
      std::vector<double> a(data, data + num_nodes);
      nodal_fields[i]=a;
      //for(int j = 0; j<(nodal_fields[i]).size();j++ ) std::cout<<proc_id<<" "<<(nodal_fields[i]).size()<<" "<<num_nodes<<" "<<j<<" "<<nodal_fields[i][j]<<std::endl;
      return 1;
    }
  }

   if(verbose)

     std::cout<<"=== Update nodal data ==="<<std::endl
	      <<" num_nodal_fields "<<num_nodal_fields<<std::endl
	      <<" sizeof nodal_field_names "<<nodal_field_names.size()<<std::endl
	      <<" sizeof nodal_fields "<<nodal_fields.size()<<std::endl<<std::endl;

   std::cout<<name<<" not found"<<std::endl<<std::endl;
   //exit(0);
   return 0;

}

int Mesh::update_elem_data(const std::string name, const double *data){

  for (int i = 0; i < num_elem_fields; i++){
    if(name == elem_field_names[i]){
      //std::cout<<"found"<<std::endl;
      std::vector<double> a(data, data + num_elem);
      elem_fields[i]=a;
      //for(int j = 0; j<(nodal_fields[i]).size();j++ ) std::cout<<proc_id<<" "<<(nodal_fields[i]).size()<<" "<<num_nodes<<" "<<j<<" "<<nodal_fields[i][j]<<std::endl;
      return 1;
    }
  }

   if(verbose)

     std::cout<<"=== Update elem data ==="<<std::endl
	      <<" num_elem_fields "<<num_elem_fields<<std::endl
	      <<" sizeof elem_field_names "<<elem_field_names.size()<<std::endl
	      <<" sizeof elem_fields "<<elem_fields.size()<<std::endl<<std::endl;

   std::cout<<name<<" not found"<<std::endl<<std::endl;
   //exit(0);

   return 0;

}
int Mesh::add_nodal_data(std::string name, double *data){


   num_nodal_fields++;

   std::vector<double> a(data, data + num_nodes);;

   nodal_field_names.push_back(name);
   nodal_fields.push_back(a);

   if(verbose)

    std::cout<<"=== Add nodal fields ==="<<std::endl
	     <<" num_nodal_fields "<<num_nodal_fields<<std::endl
	     <<" sizeof nodal_field_names "<<nodal_field_names.size()<<std::endl
	     <<" sizeof nodal_fields "<<nodal_fields.size()<<std::endl<<std::endl;

   return 1;

}

int Mesh::open_exodus(const char * filename){
  int comp_ws = sizeof(double);// = 8
  int io_ws = sizeof(double);// = 8
  float version;

  int ex_id = ex_open(filename, EX_WRITE, &comp_ws, &io_ws, &version);
}

int Mesh::create_exodus(const char * filename){

  //Store things as doubles
  int comp_ws = sizeof(double);// = 8
  int io_ws = sizeof(double);// = 8

  int ex_id = ex_create(filename, EX_CLOBBER, &comp_ws, &io_ws);
  
  ex_id = ex_open(filename,
			  EX_WRITE,
			  &comp_ws,
			  &io_ws,
			  &exodus_version);

  if(verbose)

    std::cout<<"=== ExodusII Create Info ==="<<std::endl
	     <<" ExodusII "<<exodus_version<<std::endl
	     <<" File "<<filename<<std::endl
	     <<" Exodus ID "<<ex_id<<std::endl
	     <<" comp_ws "<<comp_ws<<std::endl
	     <<" io_ws "<<io_ws<<std::endl;              

  char * title = new char[16];

  strcpy(title, "\"Exodus output\"");

  int ex_err = ex_put_init(ex_id, title, num_dim, 
			   num_nodes, num_elem, num_elem_blk, 
			   num_node_sets, num_side_sets);

  check_exodus_error(ex_err,"Mesh::create_exodus ex_put_init");

  if(verbose)

    std::cout<<" Title "<<title<<std::endl
	     <<" num_dim "<<num_dim<<std::endl
	     <<" num_nodes "<<num_nodes<<std::endl
	     <<" num_elem "<<num_elem<<std::endl
	     <<" num_elem_blk "<<num_elem_blk<<std::endl
	     <<" num_node_sets "<<num_node_sets<<std::endl
	     <<" num_side_sets "<<num_side_sets<<std::endl;
    
  if(verbose)

    std::cout<<"=== End ExodusII Create Info ==="<<std::endl<<std::endl;

  delete [] title;

  return ex_id;

}

void Mesh::set_vertex_map(){

  num_vertices = 0;

  for (int blk = 0; blk < num_elem_blk; blk++){

    std::vector<int> temp(num_node_per_elem_in_blk[blk]);
    int num_vertices_in_elem = 3;
    if(4 == num_elem_in_blk[blk] || 9 == num_elem_in_blk[blk])
      num_vertices_in_elem = 4;

    for(int i = 0; i < num_elem_in_blk[blk]; i++){
      for(int j = 0; j < num_vertices_in_elem; j++){
	int nodeid =  get_node_id(blk, i, j);
	if( vertex_map.find( nodeid ) == vertex_map.end() ){
	  vertex_map.insert(std::pair<int, int>(nodeid, num_vertices));
	  //std::cout<<get_node_id(blk, i, j)<<"   "<<num_vertices<<std::endl;
	  num_vertices++;
	}
      }
    }
  }
  return;

}
int  Mesh::get_num_global_nodes(){
  if( 1 < nprocs ){
    return ne_num_global_nodes;
  }else{
    return num_nodes;
  }
}


void Mesh::compute_nodal_patch(){


  //cn not confirmed in parallel yet, there may be some issues

  //cn if we can loop over all elements in the non partitioned file here, rather than the
  //cn elements on this block, this might work in parallel
  
  //cn however, we need a version of get_node_id(blk, ne, k) that works globally as well
  
  //cn one hack option is to pass the filename in here and create a one off of get_node_id(blk, ne, k)
  
  //cn another hack option is to create a new global mesh right here, get this info
  //cn and delete the mesh
  
  
  
  Mesh * global_mesh = new Mesh((int)0,(int)1,false);
  global_mesh->read_exodus(global_file_name.c_str());


  //we really want to search by global id
  num_my_nodes = my_node_num_map.size();

  //std::cout<<"compute_nodal_patch() started on proc_id: "<<proc_id<<" with num_my_nodes "<<num_my_nodes<<std::endl;

  nodal_patch.resize(num_my_nodes);

  //std::cout<<"compute_nodal_patch() "<<nodal_patch.size()<<" "<<num_nodes<<" "<<my_node_num_map.size()<<std::endl<<std::endl;
  for(int blk = 0; blk < get_num_elem_blks(); blk++){
    int n_nodes_per_elem = global_mesh->get_num_nodes_per_elem_in_blk(blk);


    //cn node_num_map is the list of global ids
    //cn it is allocated to the # of internal nodes + ghosts
    //cn my_node_num_map is a mapping: global id = my_node_num_map[local id]
    //cn it is allocated to the # of internal nodes




    for (int ne=0; ne < global_mesh->get_num_elem_in_blk(blk); ne++){
      for(int k = 0; k < n_nodes_per_elem; k++){
	
	int nodeid = global_mesh->get_node_id(blk, ne, k);
	//std::cout<<proc_id<<" "<<global_mesh->get_global_elem_id(ne)<<" "<<nodeid<<" "<<my_node_num_map[nodeid]<<" "<<is_global_node_local(nodeid)<<std::endl;
	//we check here if the node lives on this proc
	if(is_global_node_local(nodeid)){


	  //cn we load global elem id here
	  int elemid = global_mesh->get_global_elem_id(ne);

	  //cn might be better to load the local elem id instead
	  //cn which would mean look up the global elem id from global_mesh in in_mesh

	  //cn or we need some kind of overlap map for elements
	  //cn that we could load up here, we would need
	  // elem, nodes and x, y,z
	  // we could get by with just getting the off processor info for elem, node, coord


	  //int elemid = ne;

	  //cn into a local node id map
	  nodal_patch[nodeid].push_back(elemid);
	}
      }      
    }
  }

//   for(int i=0; i<num_my_nodes; i++){
//     std::cout<<proc_id<<" "<<i<<":: "<<node_num_map[i]<<"::  ";
//     for(int j=0; j< nodal_patch[i].size(); j++){
//       std::cout<<nodal_patch[i][j]<<" ";
//     }
//     std::cout<<std::endl;
//   }

  //std::cout<<"compute_nodal_patch() finished on proc_id: "<<proc_id<<std::endl;
  //exit(0);

  delete global_mesh;
}

bool Mesh::is_global_node_local(int i){
  bool found = false;

  found = (std::find(my_node_num_map.begin(), my_node_num_map.end(), i) != my_node_num_map.end());

  return found;
}

bool Mesh::is_global_elem_local(int i){
  bool found = false;

  found = (std::find(elem_num_map.begin(), elem_num_map.end(), i) != elem_num_map.end());

  return found;
}


void Mesh::compute_nodal_patch_old(){


  //cn not working in parallel yet, there are some issues


  //my_node_num_map is local ids
  //we really want to search by global id
  num_my_nodes = my_node_num_map.size();

  //std::cout<<"compute_nodal_patch() started on proc_id: "<<proc_id<<" with num_my_nodes "<<num_my_nodes<<std::endl;

  nodal_patch.resize(num_my_nodes);

  //std::cout<<"compute_nodal_patch() "<<nodal_patch.size()<<" "<<num_nodes<<" "<<my_node_num_map.size()<<std::endl<<std::endl;
  for(int blk = 0; blk < get_num_elem_blks(); blk++){
    int n_nodes_per_elem = get_num_nodes_per_elem_in_blk(blk);

    for (int ne=0; ne < get_num_elem_in_blk(blk); ne++){
      for(int k = 0; k < n_nodes_per_elem; k++){
	
	int nodeid = get_node_id(blk, ne, k);
	//std::cout<<proc_id<<" "<<get_global_elem_id(ne)<<" "<<nodeid<<" "<<node_num_map[nodeid]<<" "<<num_my_nodes<<std::endl;
	//we check here if the node lives on this proc
	if(nodeid < num_my_nodes){
	  //int elemid = get_global_elem_id(ne);
	  int elemid = ne;
	  nodal_patch[nodeid].push_back(elemid);
	}
      }      
    }
  }

//   for(int i=0; i<num_my_nodes; i++){
//     std::cout<<proc_id<<" "<<i<<":: "<<node_num_map[i]<<"::  ";
//     for(int j=0; j< nodal_patch[i].size(); j++){
//       std::cout<<nodal_patch[i][j]<<" ";
//     }
//     std::cout<<std::endl;
//   }

  //std::cout<<"compute_nodal_patch() finished on proc_id: "<<proc_id<<std::endl;
  //exit(0);

}

void Mesh::compute_elem_adj(){

  //at the end we have a
  //std::vector<std::vector<int>> elem_connect indexed by local elemid
  //where elem_connect[ne] is a vector of global elemids icluding and surrounding ne

  //we have also made blk = 0 assumption



  elem_connect.resize(num_elem);

  for(int blk = 0; blk < get_num_elem_blks(); blk++){

    std::string elem_type=get_blk_elem_type(blk);
 
    int num_vertices_in_elem = 3;

    if( (0==elem_type.compare("QUAD4")) || 
	(0==elem_type.compare("QUAD")) || 
	(0==elem_type.compare("quad4")) || 
	(0==elem_type.compare("quad")) || 
	(0==elem_type.compare("quad9")) || 
	(0==elem_type.compare("QUAD9")) || 
	(0==elem_type.compare("TETRA4")) || 
	(0==elem_type.compare("TETRA")) || 
	(0==elem_type.compare("tetra4")) || 
	(0==elem_type.compare("tetra")) ||
	(0==elem_type.compare("TETRA10")) || 
	(0==elem_type.compare("tetra10")) ){ 
      num_vertices_in_elem = 4;
    }
    else if( (0==elem_type.compare("HEX8")) || 
	     (0==elem_type.compare("HEX")) || 
	     (0==elem_type.compare("hex8")) || 
	     (0==elem_type.compare("hex"))  ||
	     (0==elem_type.compare("HEX27")) || 
	     (0==elem_type.compare("hex27")) ){ 
      num_vertices_in_elem = 8;
    }

//     for (int ne=0; ne < get_num_elem_in_blk(blk); ne++){
//       int elemid = get_global_elem_id(ne);
//       std::cout<<proc_id<<" "<<ne<<" "<<elemid<<" "<<elem_num_map[ne]<<std::endl;
//     }

    for (int ne=0; ne < get_num_elem_in_blk(blk); ne++){
      int elemid = get_global_elem_id(ne);
      for(int k = 0; k < num_vertices_in_elem; k++){

	int nodeid = get_node_id(blk, ne, k);//local node id
	int gnodeid = node_num_map[nodeid];
	//std::cout<<proc_id<<" "<<ne<<" "<<elemid<<" "<<nodeid<<" "<<gnodeid<<std::endl;

	for(int ne2=0; ne2 < get_num_elem_in_blk(blk); ne2++){
	  for(int k2 = 0; k2 < num_vertices_in_elem; k2++){
	    //std::cout<<ne<<" "<<ne2<<std::endl;
	    int nodeid2 = get_node_id(blk, ne2, k2);//local node id
	    //if(nodeid == nodeid2) elem_connect[elemid].push_back(get_global_elem_id(ne2));
	    if(nodeid == nodeid2) elem_connect[ne].push_back(get_global_elem_id(ne2));
	  }
	}

      }
    }
  }


  if(verbose)
    std::cout<<"=== Compute elem adjacencies ==="<<std::endl;

  int blk = 0;
  for (int ne=0; ne < get_num_elem_in_blk(blk); ne++){
    int elemid = get_global_elem_id(ne);

    sort( elem_connect[ne].begin(), elem_connect[ne].end() );
    elem_connect[ne].erase( unique( elem_connect[ne].begin(), elem_connect[ne].end() ), elem_connect[ne].end() );


    if(verbose){

      int n=elem_connect[ne].size();
      std::cout<<elemid<<"::"<<std::endl;
      for (int k =0; k<n; k++){
	std::cout<<"  "<<elem_connect[ne][k];
      }
      std::cout<<std::endl;
    }
  }

}

int Mesh::get_local_id(int gid)
{
  int lid = -999999999;
  std::vector<int>::iterator it;
  it = find (node_num_map.begin(), node_num_map.end(), gid);
  lid = (int)(*it);
  if (lid < 0) exit(0);
  return lid;
}
