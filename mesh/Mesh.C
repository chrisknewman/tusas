//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#include "Mesh.h"
#include <iostream>
#include <algorithm>
#include "string.h"
#include <cmath>
#include <iomanip>

#include "exodusII.h"

//#ifdef NEMESIS
#include "ne_nemesisI.h"
//#endif

//MAX_LINE_LENGTH is defined to 80 in exodusII.h
#define TUSAS_MAX_LINE_LENGTH 120

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

Mesh::~Mesh(){
  //close_exodus(exid);
  //ex_close(exid);
}

int Mesh::read_exodus(const char * filename){

  int comp_ws = sizeof(double);//cn send this to exodus to tell it we are using doubles
  int io_ws = 0;
  std::vector<int>::iterator a;
  std::vector<mesh_lint_t>::iterator aa;

  is_nodesets_sorted = false;
  is_compute_nodal_patch_overlap = false;

  num_nodal_fields = 0;
  num_elem_fields = 0;

  int ex_id = ex_open(filename,//cn open file
		      EX_READ,
		      &comp_ws,
		      &io_ws,
		      &exodus_version);
  exid = ex_id;

  if(ex_id < 0){

  	std::cerr << "Error: cannot open file " << filename << std::endl;
	exit(1);

  }

  int max_name_length = ex_inquire_int(ex_id, EX_INQ_DB_MAX_USED_NAME_LENGTH);

//   std::cout<<"max_name_length = "<<max_name_length<<"      "<<sizeof(mesh_lint_t)<<std::endl;
//   exit(1);

  //char _title[MAX_LINE_LENGTH];
  char _title[TUSAS_MAX_LINE_LENGTH];

  int ex_err = 0;


  bool ex_maps_int64_db = (EX_MAPS_INT64_DB & ex_int64_status(ex_id));
  bool ex_ids_int64_db  = (EX_IDS_INT64_DB & ex_int64_status(ex_id));
  bool ex_bulk_int64_db = (EX_BULK_INT64_DB & ex_int64_status(ex_id));
  bool ex_all_int64_db  = (EX_ALL_INT64_DB & ex_int64_status(ex_id));

  bool ex_maps_int64_api = (EX_MAPS_INT64_API & ex_int64_status(ex_id));
  bool ex_ids_int64_api  = (EX_IDS_INT64_API & ex_int64_status(ex_id));
  bool ex_bulk_int64_api = (EX_BULK_INT64_API & ex_int64_status(ex_id));
  bool ex_inq_int64_api  = (EX_INQ_INT64_API & ex_int64_status(ex_id));
  bool ex_all_int64_api  = (EX_ALL_INT64_API & ex_int64_status(ex_id));
  if( ex_maps_int64_db | ex_all_int64_api ) {
    if( proc_id == 0){
      std::cout<<"ex_ids_int64_db = "<<ex_ids_int64_db<<std::endl;
      std::cout<<"ex_bulk_int64_db = "<<ex_bulk_int64_db<<std::endl;
      std::cout<<"ex_all_int64_db = "<<ex_all_int64_db<<std::endl;
      std::cout<<"ex_maps_int64_api = "<<ex_maps_int64_api<<std::endl;
      std::cout<<"ex_ids_int64_api = "<<ex_ids_int64_api<<std::endl;
      std::cout<<"ex_bulk_int64_api = "<<ex_bulk_int64_api<<std::endl;
      //std::cout<<"ex_all_int64_api = "<<ex_all_int64_api<<std::endl;
      std::cout<<std::endl<<std::endl;
    }
    ex_err = ex_set_int64_status(ex_id,EX_MAPS_INT64_DB);
  }

#ifdef MESH_64
  ex_err = ex_set_int64_status(ex_id,EX_MAPS_INT64_API);
#endif

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
  

      //cn seems arguments here are 64 bit???

    
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

    for(a = connect[i].begin(); a != connect[i].end(); a++) (*a)--;  // fix FORTRAN indexing

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
    //#if MESH_REFACTOR
    ne_get_loadbal_param(ex_id, &num_internal_nodes, &num_border_nodes, &num_external_nodes,
			 &num_internal_elems, &num_border_elems, &num_node_cmaps, &num_elem_cmaps, proc_id);
    //#endif    
    ne_get_n_node_num_map(ex_id, 1, num_nodes, &node_num_map[0]);
    for(aa = node_num_map.begin(); aa != node_num_map.end(); aa++) (*aa)--;
    ne_get_n_elem_num_map(ex_id, 1, num_elem, &elem_num_map[0]);
    for(aa = elem_num_map.begin(); aa != elem_num_map.end(); aa++) (*aa)--;
    
    elem_mapi.resize(num_internal_elems);
    elem_mapb.resize(num_border_elems);
    
    ne_get_elem_map(ex_id, &elem_mapi[0], &elem_mapb[0], proc_id);
    for(aa = elem_mapi.begin(); aa != elem_mapi.end(); aa++) (*aa)--;
    for(aa = elem_mapb.begin(); aa != elem_mapb.end(); aa++) (*aa)--;
    
    node_mapi.resize(num_internal_nodes);
    node_mapb.resize(num_border_nodes);
    node_mape.resize(num_external_nodes);
    
    ne_get_node_map(ex_id, &node_mapi[0], &node_mapb[0], &node_mape[0], proc_id);
    for(aa = node_mapi.begin(); aa != node_mapi.end(); aa++) (*aa)--;
    for(aa = node_mapb.begin(); aa != node_mapb.end(); aa++) (*aa)--;
    for(aa = node_mape.begin(); aa != node_mape.end(); aa++) (*aa)--;

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

    //#if MESH_REFACTOR    
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
    //#endif    
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
    for(aa = node_num_map.begin(); aa != node_num_map.end(); aa++) (*aa)--;
    ex_err = ex_get_map(ex_id, &elem_num_map[0]);
    for(aa = elem_num_map.begin(); aa != elem_num_map.end(); aa++) (*aa)--;
    
    my_node_num_map = node_num_map;  // same in serial
  }
  //#endif
    
  ex_err = close_exodus(ex_id);//cn close file
  
  if(verbose){
    
    std::cout << "There are " << num_elem << " elements in this mesh." << std::endl;
    std::cout<<"=== End ExodusII Read Info ==="<<std::endl<<std::endl;
    
  }
  //create_sorted_nodelist();
  return ex_err;
  
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

	  if((status = node_set_map[connect[blk][elem * num_node_per_elem_in_blk[blk] + i]]) >= 0)

			return status;

	return -1;

}

int Mesh::get_node_boundary_status(int nodeid){

	int status;

	if((status = node_set_map[nodeid]) >= 0)

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
   write_elem_data_exodus(ex_id);
   //ex_put_node_num_map(ex_id,&node_num_map[0]);
   //ex_put_elem_num_map(ex_id,&elem_num_map[0]);
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

int Mesh::write_exodus_no_elem(const int ex_id, const int counter, const double time){
  int error = 0;
  error = write_nodal_coordinates_exodus(ex_id);
  //std::cout<<error<<std::endl;
  error = write_element_blocks_exodus(ex_id);
  //std::cout<<error<<std::endl;
  error = write_nodal_data_exodus(ex_id,counter);
  //error = write_elem_data_exodus(ex_id,counter);
  //std::cout<<error<<std::endl;
  error = ex_put_time(ex_id,counter,&time);
  //std::cout<<error<<std::endl;

  return error;

}
int Mesh::read_last_step_exodus(const int ex_id, int &timestep){
  float ret_float = 0.0;
  char ret_char = '\0';
  int error = ex_inquire (ex_id, EX_INQ_TIME, &timestep, &ret_float,&ret_char);
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
  std::vector<mesh_lint_t> tmpvec, tmpvec3,tmpvec4;
  std::vector<int> tmpvec1;
  std::vector<int>::iterator a;
  std::vector<mesh_lint_t>::iterator aa;

  if( 1 < nprocs ){
    //#ifdef NEMESIS

    ne_put_init_global(ex_id, ne_num_global_nodes, ne_num_global_elems, ne_num_global_elem_blks,
		       ne_num_global_node_sets, ne_num_global_side_sets);
    
    ne_put_init_info(ex_id, nprocs, nprocs_infile, &filetype);
    //#if MESH_REFACTOR    
    ne_put_loadbal_param(ex_id, num_internal_nodes, num_border_nodes, num_external_nodes,
			 num_internal_elems, num_border_elems, num_node_cmaps, num_elem_cmaps, proc_id);
    //#endif    
    tmpvec = node_num_map;
    for(aa = tmpvec.begin(); aa != tmpvec.end(); aa++) (*aa)++;
    ne_put_n_node_num_map(ex_id, 1, num_nodes, &tmpvec[0]);
    
    tmpvec = elem_num_map;
    for(aa = tmpvec.begin(); aa != tmpvec.end(); aa++) (*aa)++;
    ne_put_n_elem_num_map(ex_id, 1, num_elem, &tmpvec[0]);
   
    tmpvec = elem_mapi;
    for(aa = tmpvec.begin(); aa != tmpvec.end(); aa++) (*aa)++;

    tmpvec3 = elem_mapb;
    for(aa = tmpvec3.begin(); aa != tmpvec3.end(); aa++) (*aa)++;
    ne_put_elem_map(ex_id, &tmpvec[0], &tmpvec3[0], proc_id);
 
    tmpvec = node_mapi;
    for(aa = tmpvec.begin(); aa != tmpvec.end(); aa++) (*aa)++;

    tmpvec3 = node_mapb;
    for(aa = tmpvec3.begin(); aa != tmpvec3.end(); aa++) (*aa)++;
    tmpvec4 = node_mape;
    for(aa = tmpvec4.begin(); aa != tmpvec4.end(); aa++) (*aa)++;
    ne_put_node_map(ex_id, &tmpvec[0], &tmpvec3[0], &tmpvec4[0], proc_id);
    
    ne_put_n_coord(ex_id, 1, num_nodes, &x[0], &y[0], &z[0]);
    
    if(ne_num_global_node_sets > 0)
      
      ne_put_ns_param_global(ex_id, &global_ns_ids[0], &num_global_node_counts[0],
			     &num_global_node_df_counts[0]);
    
    if(ne_num_global_side_sets > 0)
      
      ne_put_ss_param_global(ex_id, &global_ss_ids[0], &num_global_side_counts[0],
			     &num_global_side_df_counts[0]);
    
    ne_put_eb_info_global(ex_id, &global_elem_blk_ids[0], &global_elem_blk_cnts[0]);
    //#if MESH_REFACTOR    
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
    //#endif    
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

	tmpvec1 = ns_node_list[i];
        for(a = tmpvec1.begin(); a != tmpvec1.end(); a++) (*a)++;

	if( 1 < nprocs ){
	  //#ifdef NEMESIS

        ex_err = ne_put_n_node_set(ex_id, ns_ids[i], 1, num_nodes_per_ns[i], &tmpvec1[0]);
	}
	else {
	  //#else

        ex_err = ex_put_node_set(ex_id, ns_ids[i], &tmpvec1[0]);
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

    if(verbose)
      std::cout<<"   i  "<<i<<"  "<<nodal_fields[i].size()<<std::endl<<std::endl;
    ex_err = ex_put_nodal_var (ex_id, counter, i + 1, num_nodes, &nodal_fields[i][0]);
    //for(int j = 0; j<(nodal_fields[i]).size();j++ ) std::cout<<nodal_fields[i][j]<<std::endl;
  
  }

  delete [] var_names;

  return ex_err;

}

int Mesh::write_elem_data_exodus(int ex_id){
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

    ex_err = ex_put_elem_var (ex_id, 1, i + 1, blk,num_elem, &elem_fields[i][0]);

    //(exoid,time_step,elem_var_index,elem_blk_id,num_elem_this_blk,elem_var_vals)
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
  char ftype[TUSAS_MAX_LINE_LENGTH];
  int ex_err = ne_get_init_info(ex_id,nproc,&num_proc_in_file,ftype);
  return ex_err;
}

int Mesh::read_nodal_data_exodus(const int ex_id, const int timestep, const int index, double *data){
  int ex_err = ex_get_nodal_var (ex_id, timestep, index, num_nodes, &data[0]);
  return ex_err;
}

int Mesh::read_nodal_data_exodus(const int ex_id, const int timestep, std::string name, double *data){
  //int index = get_nodal_field_index(name) + 1;//exodus starts at 1
  int index = read_nodal_field_index(ex_id, name) + 1;//exodus starts at 1
  int ex_err = ex_get_nodal_var (ex_id, timestep, index, num_nodes, &data[0]);
  return ex_err;
}

int Mesh::read_nodal_field_index(const int ex_id, std::string name){
  //cn this should be the index in the exodus file, since we have not populated these names yet

  int index = -1;

  int num_node_vars;
  int error = ex_get_var_param (ex_id, "n" , &num_node_vars);

  //std::cout<<num_node_vars<<std::endl;

  char ** var_names;
  var_names = new char*[num_node_vars];
  for (int i = 0; i < num_node_vars; i++) var_names[i] = new char[TUSAS_MAX_LINE_LENGTH];


  error = ex_get_var_names (ex_id, "n", num_node_vars, var_names);
  for (int i = 0; i < num_node_vars; i++){
    //std::cout<<std::string(var_names[i])<<std::endl;
    if( name == std::string(var_names[i])) index = i;
  }

  for (int i = 0; i < num_node_vars; i++) delete [] var_names[i];
  delete [] var_names;

  if(0 > index){
    std::cout<<name<<" not found"<<std::endl<<std::endl;
    exit(0);
  }
  //std::cout<<index<<std::endl;

  //exit(0);
  return index;
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


int Mesh::read_elem_data_exodus(const int ex_id, const int timestep, const int index, double *data){
  const int block = 1;//hack exodus blocks start at 1
  int ex_err = ex_get_elem_var (ex_id, timestep, index, block, num_elem, &data[0]);
  return ex_err;
}

int Mesh::read_elem_data_exodus(const int ex_id, const int timestep, std::string name, double *data){
  int index = get_elem_field_index(name) + 1;//exodus starts at 1
  const int block = 1;//hack exodus blocks start at 1
  int ex_err = ex_get_elem_var (ex_id, timestep, index, block, num_elem, &data[0]);
  return ex_err;
}

int Mesh::get_elem_field_index(std::string name){
  int index = -1;
  for (int i = 0; i < num_elem_fields; i++){
    if(name == elem_field_names[i]){
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
	       <<" sizeof nodal_field_names "<<nodal_field_names.size()<<std::endl
	       <<" nodal_field_names "<<name<<std::endl;
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
int Mesh::open_exodus(const char * filename, WR wr){
  int comp_ws = sizeof(double);// = 8
  int io_ws = sizeof(double);// = 8
  float version;

  int ex_err;
  int ex_id;

  switch (wr) {
  case WRITE:
    ex_id = ex_open(filename, EX_WRITE, &comp_ws, &io_ws, &version);
#ifdef MESH_64
    ex_err = ex_set_int64_status(ex_id,EX_MAPS_INT64_API);
#endif
    break;
  case READ:
    ex_id = ex_open(filename, EX_READ, &comp_ws, &io_ws, &version);
#ifdef MESH_64
    ex_err = ex_set_int64_status(ex_id,EX_MAPS_INT64_API);
#endif
    break;
  }
  return ex_id;
}

int Mesh::create_exodus(const char * filename, const bool use64output){

  //Store things as doubles
  int comp_ws = sizeof(double);// = 8
  int io_ws = sizeof(double);// = 8

  int ex_id = -99;
  ex_id = ex_create(filename, EX_CLOBBER, &comp_ws, &io_ws);
  //ex_id = ex_create(filename,EX_CLOBBER|EX_MAPS_INT64_DB|EX_MAPS_INT64_API, &comp_ws, &io_ws);


  //cn 8-20-20 seems there is a bug with ex_open, where it returns the wrong ex_id and exodus version?
  //currently 8.07 version exdus. i would rather not hack it like this

#if EXODUS_VERSION_MAJOR < 8
  ex_id = 
    ex_open(filename,
			  EX_WRITE,
			  &comp_ws,
			  &io_ws,
			  &exodus_version);
#else
  //ex_id = 
    ex_open(filename,
			  EX_WRITE,
			  &comp_ws,
			  &io_ws,
			  &exodus_version);
#endif

  int ex_err;
#ifdef MESH_64
  ex_err = ex_set_int64_status(ex_id,EX_MAPS_INT64_API);
#endif

  if(verbose)

    std::cout<<"=== ExodusII Create Info ==="<<std::endl
	     <<" ExodusII "<<exodus_version<<std::endl
	     <<" File "<<filename<<std::endl
	     <<" Exodus ID "<<ex_id<<std::endl
	     <<" comp_ws "<<comp_ws<<std::endl
	     <<" io_ws "<<io_ws<<std::endl
	     <<" ex_err "<<ex_err<<std::endl
	     <<" exodus_version "<<exodus_version<<std::endl;              

  char * title = new char[16];

  strcpy(title, "\"Exodus output\"");

  ex_err = ex_put_init(ex_id, title, num_dim, 
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

  if( num_my_nodes == nodal_patch.size() ) return;

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

  return;
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

void Mesh::compute_nodal_patch_overlap(){

  //cn 5-23-18
  //cn this version will be called by the error estimator
  //cn not working in parallel yet, there are some issues

  //cn in parallel, I think we need nodal patches for the overlap map,
  //cn not the node map

  //cn then we will average the shared nodes in the estimator

  if(is_compute_nodal_patch_overlap) return;

  //my_node_num_map is local ids
  //we really want to search by global id
  num_my_nodes = my_node_num_map.size();
  //std::cout<<num_my_nodes<<" "<<num_nodes<<std::endl;

  //std::cout<<"compute_nodal_patch() started on proc_id: "<<proc_id<<" with num_my_nodes "<<num_my_nodes<<std::endl;

  //if( num_my_nodes == nodal_patch.size() ) return;
  //exit(0);

  nodal_patch_overlap.resize(num_nodes);

  //std::cout<<"compute_nodal_patch() "<<nodal_patch.size()<<" "<<num_nodes<<" "<<my_node_num_map.size()<<std::endl<<std::endl;
  for(int blk = 0; blk < get_num_elem_blks(); blk++){
    int n_nodes_per_elem = get_num_nodes_per_elem_in_blk(blk);

    for (int ne=0; ne < get_num_elem_in_blk(blk); ne++){
      for(int k = 0; k < n_nodes_per_elem; k++){
	
	int nodeid = get_node_id(blk, ne, k);
	//std::cout<<proc_id<<" "<<get_global_elem_id(ne)<<" "<<nodeid<<" "<<node_num_map[nodeid]<<" "<<num_my_nodes<<std::endl;
	//we check here if the node lives on this proc
	if(nodeid < num_nodes){
	  //int elemid = get_global_elem_id(ne);
	  int elemid = ne;
	  nodal_patch_overlap[nodeid].push_back(elemid);
	}
      }      
    }
  }
  
  is_compute_nodal_patch_overlap = true;


//   for(int i=0; i<num_nodes; i++){
//     std::cout<<proc_id<<" "<<i<<":: "<<node_num_map[i]<<"::  ";
//     //std::cout<<nodal_patch_overlap[i].size();
//     for(int j=0; j< nodal_patch_overlap[i].size(); j++){
//       std::cout<<nodal_patch_overlap[i][j]<<" ";
//     }
//     std::cout<<std::endl;
//   }
  //exit(0);
  //std::cout<<"compute_nodal_patch() finished on proc_id: "<<proc_id<<std::endl;
  //exit(0);
  return;
}

void Mesh::compute_nodal_patch_old(){

  //cn this is called below by compute_elem_adj() that is used in computing the
  //cn element graph for cpu/gpu computations

  //cn not working in parallel yet, there are some issues

  //cn in parallel, I think we need nodal patches for the overlap map,
  //cn not the node map


  //my_node_num_map is local ids
  //we really want to search by global id
  num_my_nodes = my_node_num_map.size();

  //std::cout<<"compute_nodal_patch() started on proc_id: "<<proc_id<<" with num_my_nodes "<<num_my_nodes<<std::endl;

  //if( num_my_nodes == nodal_patch.size() ) return;
  //exit(0);

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
  //where elem_connect[ne] is a vector of global elemids including and surrounding ne

  //we have also made blk = 0 assumption

  //this has been cleaned up on 2-22-18, it seems that the adjacency is not correct
  //in parallel with mpi; hence the hack below.  
  
  //I think it was working in parallel before the attempt to speed it up with 1-11-18
  //commit afdc0e7747ee6396055af96b2734a84dce7c9c3c

  //It really needs to be fixed as in the 
  //else loop for parallel.
  //The new approach uses nodal_patch.
  //The problem is that the nodal_patch reaches over prov boundaries;
  //see comments in error_estimator.

  //cn changed to the following on 6-27-18
  //seems to make it a little better
  //compute_nodal_patch_old();
  compute_nodal_patch_overlap();
  

  elem_connect.resize(num_elem);

  for(int blk = 0; blk < get_num_elem_blks(); blk++){

    std::string elem_type=get_blk_elem_type(blk);
 
    int num_vertices_in_elem = 3;

    int num_elem_in_patch = 4;

    if( (0==elem_type.compare("QUAD4")) || 
	(0==elem_type.compare("QUAD")) || 
	(0==elem_type.compare("quad4")) || 
	(0==elem_type.compare("quad")) || 
	(0==elem_type.compare("quad9")) || 
	(0==elem_type.compare("QUAD9")) 
	//|| 
	//(0==elem_type.compare("TETRA4")) || 
	//(0==elem_type.compare("TETRA")) || 
	//(0==elem_type.compare("tetra4")) || 
	//(0==elem_type.compare("tetra")) ||
	//(0==elem_type.compare("TETRA10")) || 
	//(0==elem_type.compare("tetra10")) 
	){ 
      num_vertices_in_elem = 4;
      //num_elem_in_patch = 4;
    }
    else if( (0==elem_type.compare("HEX8")) || 
	     (0==elem_type.compare("HEX")) || 
	     (0==elem_type.compare("hex8")) || 
	     (0==elem_type.compare("hex"))  ||
	     (0==elem_type.compare("HEX27")) || 
	     (0==elem_type.compare("hex27")) 
	     ){ 
      num_vertices_in_elem = 8;
      //num_elem_in_patch = 8;
    }
    else{
      std::cout<<"Mesh::compute_elem_adj() unsupported element at this time"<<std::endl<<std::endl<<std::endl;
      exit(0);
    }//if 

//     for (int ne=0; ne < get_num_elem_in_blk(blk); ne++){
//       int elemid = get_global_elem_id(ne);
//       std::cout<<proc_id<<" "<<ne<<" "<<elemid<<" "<<elem_num_map[ne]<<std::endl;
//     }

    for (int ne=0; ne < get_num_elem_in_blk(blk); ne++){
#if 0
      if (nprocs > 1){
	int cnt = 0;
	for(int k = 0; k < num_vertices_in_elem; k++){
	  
	  int nodeid = get_node_id(blk, ne, k);//local node id
	  //int gnodeid = node_num_map[nodeid];
	  //std::cout<<proc_id<<" "<<ne<<" "<<" "<<nodeid<<" "<<std::endl;
	  
	  for(int ne2=0; ne2 < get_num_elem_in_blk(blk); ne2++){
	    for(int k2 = 0; k2 < num_vertices_in_elem; k2++){
	      //std::cout<<ne<<" "<<ne2<<std::endl;
	      int nodeid2 = get_node_id(blk, ne2, k2);//local node id
	      //if(nodeid == nodeid2) elem_connect[elemid].push_back(get_global_elem_id(ne2));
	      if(nodeid == nodeid2) {
		elem_connect[ne].push_back(get_global_elem_id(ne2));
		cnt++;
		break;
	      }
	    }//k2
	    //std::cout<<ne<<" "<<elem_connect[ne].size()<<" "<<cnt<<std::endl;
	    if( cnt > num_elem_in_patch - 1) break;
	  }//ne2
	  
	}//k
      }
      else{
#endif
	for(int k = 0; k < num_vertices_in_elem; k++){
	  int nodeid = get_node_id(blk, ne, k);//local node id
  	  //int s = nodal_patch[nodeid].size();
	  int s = nodal_patch_overlap[nodeid].size();
	  for(int np = 0; np < s; np++){
	    //elem_connect[ne].push_back(get_global_elem_id(nodal_patch[nodeid][np]));
	    elem_connect[ne].push_back(get_global_elem_id(nodal_patch_overlap[nodeid][np]));
	  }//np
	}//k
#if 0
      }//if
      sort( elem_connect[ne].begin(), elem_connect[ne].end() );
      elem_connect[ne].erase( unique( elem_connect[ne].begin(), elem_connect[ne].end() ), elem_connect[ne].end() );
#endif
    }//ne
#if 0
    for (int ne=0; ne < get_num_elem_in_blk(blk); ne++){
      int elemid = get_global_elem_id(ne);
      int s = elem_connect[ne].size();
      std::cout<<nprocs<<"ec  : "<<elemid<<" : ";
      for(int k = 0; k < s; k++){
	std::cout<<elem_connect[ne][k]<<" ";
      }
      std::cout<<std::endl;
    }//ne
#endif

  }//blk

  //exit(0);

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

#if MESH_REFACTOR
//needs t be...
// int Mesh::get_local_id(mesh_lint_t gid)
int Mesh::get_local_id(int gid)
{
  int lid = -999999999;
  std::vector<mesh_lint_t>::iterator it;
  it = find (node_num_map.begin(), node_num_map.end(), gid);
  lid = (int)(*it);
  if (lid < 0) exit(0);
  return lid;
}
#endif
bool const essEqual(const double a, const double b, const double epsilon)
{
    return std::fabs(a - b) <= ( (std::fabs(a) > std::fabs(b) ? std::fabs(b) : std::fabs(a)) * epsilon);
}

bool defGreaterThan(const double a, const double b, const double epsilon)
{
    return (a - b) > ( (std::fabs(a) < std::fabs(b) ? std::fabs(b) : std::fabs(a)) * epsilon);
}

bool defLessThan(const double a, const double b, const double epsilon)
{
    return (b - a) > ( (std::fabs(a) < std::fabs(b) ? std::fabs(b) : std::fabs(a)) * epsilon);
}
bool const approxEqual(const double a, const double b, const double epsilon)
{
  return std::fabs(a - b) <= ( (std::fabs(a) < std::fabs(b) ? std::fabs(b) : std::fabs(a)) * epsilon);
}


void Mesh::create_sorted_nodesetlists()
{
//   std::cout<<"Mesh::create_sorted_nodelists()"<<std::endl;
//   std::cout<<"int num_node_sets "<<num_node_sets<<std::endl;
//   std::cout<<"num_nodes_per_ns[0] "<<num_nodes_per_ns[0]<<std::endl;

  if(is_nodesets_sorted) return;

  sorted_ns_node_list.resize(num_node_sets);
  
  typedef std::tuple<int, double, double, double> tuple_t;

  for ( int i = 0; i < num_node_sets; i++ ){
    sorted_ns_node_list[i].resize(num_nodes_per_ns[i]);
    
    std::vector<tuple_t> sns(num_nodes_per_ns[i]);

    for (int n = 0; n < num_nodes_per_ns[i]; n++){
      int lid = ns_node_list[i][n];
      double x = get_x(lid);
      double y = get_y(lid);
      double z = get_z(lid);
//       std::cout<<n<<" "<<lid<<" :"<<x<<" "<<y<<" "<<z<<std::endl;
      sns[n] = std::make_tuple(lid,x,y,z);
    }

    std::stable_sort(begin(sns), end(sns), 
		     [](tuple_t const &t1, tuple_t const &t2) {
		       
		       if(approxEqual(std::get<3>(t1),std::get<3>(t2),1e-10)) {
			 
			 if(approxEqual(std::get<2>(t1),std::get<2>(t2),1e-10)) {
			   return (std::get<1>(t1) < std::get<1>(t2));
			 }
			 
			 
			 return (std::get<2>(t1) < std::get<2>(t2));
		       }	
		       

		       if(std::get<3>(t1)<std::get<3>(t2)) {
			 return true; 
		       } 
		       
		       
		       return false;
		     }
		     );
    
    
    //     std::cout<<"++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    for (int n = 0; n < num_nodes_per_ns[i]; n++){
      //       std::cout<<std::get<0>(sns[n])<<" :"<<std::get<1>(sns[n])<<" "<<std::get<2>(sns[n])<<" "<<std::get<3>(sns[n])<<std::endl;
      sorted_ns_node_list[i][n] = std::get<0>(sns[n]);
    }//n
  }//i

  is_nodesets_sorted = true;

  //   exit(0);
}

void Mesh::create_sorted_nodelist()
{
  sorted_node_num_map.resize(num_nodes);

  typedef std::tuple<int, double, double, double> tuple_t;
  //typedef std::tuple<double, double> tuple_t;

  std::vector<tuple_t> sns(num_nodes);
  
  for (int n = 0; n < num_nodes; n++){
    int gid = node_num_map[n];
    //int lid = n;
    int lid = gid;
    double x = get_x(n);
    double y = get_y(n);
    double z = get_z(n);
    //std::cout<<n<<" "<<lid<<" :"<<x<<" "<<y<<" "<<z<<std::endl;
    sns[n] = std::make_tuple(lid,x,y,z);
  }
  
  //cn the following works for 2d.....
//   std::stable_sort(begin(sns), end(sns), 
// 		   [](tuple_t const &t1, tuple_t const &t2) {

// 		     if(approxEqual(std::get<2>(t1),std::get<2>(t2),1e-10)) {
// 		       return (std::get<1>(t1) < std::get<1>(t2));
// 		     }	
 
// 		     if(std::get<2>(t1)<std::get<2>(t2)) {
// 		       return true;     		  
// 		     }   
// 		     return false;
// 		   }
//  		   );

  std::stable_sort(begin(sns), end(sns), 
		   [](tuple_t const &t1, tuple_t const &t2) {

		     if(approxEqual(std::get<3>(t1),std::get<3>(t2),1e-10)) {

		       if(approxEqual(std::get<2>(t1),std::get<2>(t2),1e-10)) {
			 return (std::get<1>(t1) < std::get<1>(t2));
		       }
		       

		       return (std::get<2>(t1) < std::get<2>(t2));
		     }	


		     if(std::get<3>(t1)<std::get<3>(t2)) {
		       return true; 
		     } 
		     
 		     
		     return false;
		   }
 		   );



  for (int n = 0; n < num_nodes; n++){
    sorted_node_num_map[n] = std::get<0>(sns[n]);
  }//n
}
void Mesh::create_sorted_elemlist()
{

  //this function sorts in increasing x, y, z

  sorted_elem_num_map.resize(num_elem);

  typedef std::tuple<int, double, double, double> tuple_t;

  std::vector<tuple_t> sns(num_elem);

  int blk = 0;//cn for now
  int n_nodes_per_elem = get_num_nodes_per_elem_in_blk(blk);
  for (int ne = 0; ne < get_num_elem_in_blk(blk); ne++){
    
    double x_avg = 0;
    double y_avg = 0;
    double z_avg = 0;
    
    for(int k = 0; k < n_nodes_per_elem; k++){
      int nodeid = get_node_id(blk, ne, k);
      
      double x = get_x(nodeid);
      double y = get_y(nodeid);
      double z = get_z(nodeid);
      
      x_avg += x;
      y_avg += y;
      z_avg += z;
      
    }//k
    int elemid = get_global_elem_id(ne);
    x_avg = x_avg/n_nodes_per_elem;
    y_avg = y_avg/n_nodes_per_elem;
    z_avg = z_avg/n_nodes_per_elem;
    sns[ne] = std::make_tuple(elemid,x_avg,y_avg,z_avg);

    //std::cout<<elemid<<" "<<x_avg<<std::endl;
  }//ne
  std::stable_sort(begin(sns), end(sns), 
		   [](tuple_t const &t1, tuple_t const &t2) {

		     if(approxEqual(std::get<3>(t1),std::get<3>(t2),1e-10)) {

		       if(approxEqual(std::get<2>(t1),std::get<2>(t2),1e-10)) {
			 return (std::get<1>(t1) < std::get<1>(t2));
		       }
		       

		       return (std::get<2>(t1) < std::get<2>(t2));
		     }	


		     if(std::get<3>(t1)<std::get<3>(t2)) {
		       return true; 
		     } 
		     
 		     
		     return false;
		   }
 		   );

  for (int ne = 0; ne < get_num_elem_in_blk(blk); ne++){
    sorted_elem_num_map[ne] = std::get<0>(sns[ne]);
  }
  //exit(0);
}

void Mesh::create_sorted_elemlist_yxz()
{

  //this function sorts in increasing y, x, z

  sorted_elem_num_map.resize(num_elem);

  typedef std::tuple<int, double, double, double> tuple_t;

  std::vector<tuple_t> sns(num_elem);

  int blk = 0;//cn for now
  int n_nodes_per_elem = get_num_nodes_per_elem_in_blk(blk);
  for (int ne = 0; ne < get_num_elem_in_blk(blk); ne++){
    
    double x_avg = 0;
    double y_avg = 0;
    double z_avg = 0;
    
    for(int k = 0; k < n_nodes_per_elem; k++){
      int nodeid = get_node_id(blk, ne, k);
      
      double x = get_x(nodeid);
      double y = get_y(nodeid);
      double z = get_z(nodeid);
      
      x_avg += x;
      y_avg += y;
      z_avg += z;
      
    }//k
    int elemid = get_global_elem_id(ne);
    x_avg = x_avg/n_nodes_per_elem;
    y_avg = y_avg/n_nodes_per_elem;
    z_avg = z_avg/n_nodes_per_elem;
    sns[ne] = std::make_tuple(elemid,y_avg,x_avg,z_avg);

    //std::cout<<elemid<<" "<<x_avg<<std::endl;
  }//ne
  std::stable_sort(begin(sns), end(sns), 
		   [](tuple_t const &t1, tuple_t const &t2) {

		     if(approxEqual(std::get<3>(t1),std::get<3>(t2),1e-10)) {

		       if(approxEqual(std::get<2>(t1),std::get<2>(t2),1e-10)) {
			 return (std::get<1>(t1) < std::get<1>(t2));
		       }
		       

		       return (std::get<2>(t1) < std::get<2>(t2));
		     }	


		     if(std::get<3>(t1)<std::get<3>(t2)) {
		       return true; 
		     } 
		     
 		     
		     return false;
		   }
 		   );

  for (int ne = 0; ne < get_num_elem_in_blk(blk); ne++){
    sorted_elem_num_map[ne] = std::get<0>(sns[ne]);
  }
  //exit(0);
}

bool Mesh::side_set_found(const int ss) const {
  int id = 0;
  return side_set_found(ss,id);
}

bool Mesh::side_set_found(const int ss, int &id) const {
  //it appears the ss_ids and ns_ids are indexed starting at 1
  id = 0;
  for (auto i : ss_ids){
    //std::cout<<i<<std::endl;
    if(ss == i-1){
      //std::cout<<(i-1)<<std::endl;
      return true;
    }
    id++;
  }
  id = -99;
  return false;
}

bool Mesh::node_set_found(const int ns) const {
  int id = 0;
  return node_set_found(ns,id);
}

bool Mesh::node_set_found(const int ns, int &id) const {
  //it appears the ss_ids and ns_ids are indexed starting at 1
  id = 0;
  for (auto i : ns_ids){
    //std::cout<<i<<std::endl;
    if(ns == i-1){
      return true;
    }
    id++;
  }
  id = -99;
  return false;
}
