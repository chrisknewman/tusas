//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////


#ifdef NO_MESH_64
#else
#define MESH_64
#endif

#define MESH_REFACTOR 0


#ifndef MESH_H
#define MESH_H

#include "stdint.h"

#include <vector>
#include <string>
#include <map>


/// Manages all mesh data.
class Mesh
{
 public:

  enum WR{WRITE = 0, READ};

  //#ifdef NEMESIS
  /// Constructor
  /** Parallel, MPI decomposed; where proc_id = current processor id, nprocs = total number of processors, v = verbosity. */
  Mesh(const int proc_id, const int nprocs, const bool v = false);
  /// Constructor
  /** Not valid */
  Mesh(){fprintf(stderr, "Must specify processor ID and number of processors"); exit(-10);}
  //#else
  /// Constructor
  /** Serial */
  Mesh(const int proc_id = 0, const bool v = false);
  //#endif
  /// Destructor
  ~Mesh();


  //the convention will be that global ids will be 64 bit-- long long
  // while local ids will be 32 bit-- int

#ifdef MESH_64
  typedef long long mesh_lint_t;
  //typedef int64_t mesh_lint_t;
#else
  typedef int mesh_lint_t;
#endif

// We need to read and write Exodus files
  /// Read exodus file based on filename.
  int read_exodus(const char * filename);
  /// Write exodus file based on filename.
  int write_exodus(const char * filename);
  /// Write exodus file based on exodus id ex_id.
  int write_exodus(const int ex_id);
  /// Write exodus file based on exodus id ex_id at timestep counter and time time.
  int write_exodus(const int ex_id, const int counter, const double time);
  int write_exodus_no_elem(const int ex_id, const int counter, const double time);
  /// Create exodus file based on filename.
  int create_exodus(const char * filename, const bool use64output = false);
  /// Open exodus file based on filename.
  int open_exodus(const char * filename, WR wr);
  /// Read time from exodus file with id ex_id and timestep counter.
  int read_time_exodus(const int ex_id, const int counter, double &time);
  /// Read the last timestep index timestep from exodus file with exodus id ex_id.
  int read_last_step_exodus(const int ex_id, int &timestep);
  /// Read nodal data by variable index index at timestep index timestep.
  int read_nodal_data_exodus(const int ex_id, const int timestep, const int index, double *data);
  /// Read nodal data by variable name name at timestep index timestep.
  int read_nodal_data_exodus(const int ex_id, const int timestep, std::string name, double *data);
  /// Read the number of processors from Nemesis file with exodus id ex_id.
  int read_num_proc_nemesis(int ex_id, int *nproc);
  /// Read elem data by variable index index at timestep index timestep.
  int read_elem_data_exodus(const int ex_id, const int timestep, const int index, double *data);
  /// Read elem data by variable name name at timestep index timestep.
  int read_elem_data_exodus(const int ex_id, const int timestep, std::string name, double *data);

// We need a set of convenient functions to retrieve data from this object, and write data to it

  /// Compute the nodal adjacencies. Must be called before any call to get_nodal_adj(int i).
  void compute_nodal_adj();
  /// Compute the elemental adjacencies.
  void compute_elem_adj();
  /// Compute nodal patch elements. Must be called before any call to get_nodal_patch(int i).
  void compute_nodal_patch_old();
  void compute_nodal_patch_overlap();
  /// Return a std::vector<int> of elements (by local id) in nodal patch for node i (by local id).
  std::vector<int> get_nodal_patch(int i){return nodal_patch[i];}
  std::vector<int> get_nodal_patch_overlap(int i){return nodal_patch_overlap[i];}

  /// Add nodal data as std::vector<double> with name name
  int add_nodal_data(std::string name, std::vector<double> &data);
  /// Add nodal data as an array with name name
  int add_nodal_data(std::string name, double *data);
  /// Add nodal field with name name
  int add_nodal_field(const std::string name);
  /// Update nodal data as an array with name name
  int update_nodal_data(const std::string name, const double *data);
  /// Add an element field with name name
  int add_elem_field(const std::string name);
  /// Update element data as an array with name name
  int update_elem_data(const std::string name, const double *data);
  /// Toggle verbosity
  void set_verbose(const bool v = true);

  //#ifdef NEMESIS
  //  int get_num_global_nodes(){return ne_num_global_nodes;}
  //#else
  //int get_num_global_nodes(){return num_nodes;}
  //#endif
  /// Return the number of global nodes on all processors.
  int get_num_global_nodes();
  /// Return the value of num_nodes
  int get_num_nodes(){return num_nodes;}
  //int get_num_my_nodes(){return my_node_num_map.size(); }
  /// Return the number of nodes on this processor.
  int get_num_my_nodes(){return num_my_nodes; }
  /// Return the number of dimensions.
  int get_num_dim(){return num_dim;}
  /// Return the value of num_elem (global number of elements on all processors).
  int get_num_elem(){return num_elem;}
  /// Return the number of element blocks.
  int get_num_elem_blks(){return num_elem_blk;}
  /// Return number of elements in block blk on this processor
  int get_num_elem_in_blk(int blk){ return num_elem_in_blk[blk];}
  /// Return the number of nodes in an element in block blk
  int get_num_nodes_per_elem_in_blk(int blk){ return num_node_per_elem_in_blk[blk];}
  /// Return address of node id (by local id) in element elem (by local id) in block blk with offest offset.
  int& get_node_id(int blk, int elem, int offset){return connect[blk][elem * num_node_per_elem_in_blk[blk] + offset]; }
  /// Return global node id of local index i
  mesh_lint_t get_global_node_id(int i){ return node_num_map[i];}
  /// Return global element id of local index i
  mesh_lint_t get_global_elem_id(int i){ return elem_num_map[i];}
  /// Return node_num_map, a list of global node ids on this processor.
  std::vector<mesh_lint_t> get_node_num_map(){ return node_num_map; }
  /// Return a pointer to elem_num_map, a list of global element ids on this processor.
  std::vector<mesh_lint_t> *get_elem_num_map(){ return &elem_num_map; }
  /// Return my_node_num_map, a list of global node ids on this processor.
  std::vector<mesh_lint_t> get_my_node_num_map(){ return my_node_num_map; }
  /// Return global element_connect for element i, by local id
  std::vector<mesh_lint_t> get_elem_connect(int i){return elem_connect[i];};
  /// Return the x coord of node i
  double get_x(int i){return x[i];}    
  /// Return the y coord of node i
  double get_y(int i){return y[i];} 
  /// Return the z coord of node i
  double get_z(int i){return z[i];}
  /// Return the nodal adjacency for node i, local id, serial
  std::vector<int> get_nodal_adj(int i){return nodal_adj[i];}
  /// Return node set with id i
  std::vector<int> get_node_set(int i){return ns_node_list[i];}
  /// Return side set with id i
  std::vector<int> get_side_set(int i){return ss_side_list[i];}  
  /// Return the nodes in side set with id i, by local id
  std::vector<int> get_side_set_node_list(int i){return ss_node_list[i];}
  /// Return node id of node j in node set with id i, by local id
  int get_node_set_entry(int i, int j){return ns_node_list[i][j];}
  /// Return node id of node j in side set with id i, by local id
  int get_side_set_node_entry(int i, int j){return ss_node_list[i][j];}
  /// Return the exodus name of the elements in blok i
  std::string get_blk_elem_type(const int i){return blk_elem_type[i];}
  /// Set global_file_name to filename
  void set_global_file_name(std::string filename){global_file_name = filename;return;};
#if MESH_REFACTOR
  /// Get local id from global id
  int get_local_id(int gid);
#endif
  /// Creates sorted nodesetlists based on increasing x, y and z. Used for periodic BCs.
  void create_sorted_nodesetlists();
  /// Return sorted node set with id i
  std::vector<int> get_sorted_node_set(int i){return sorted_ns_node_list[i];}
  /// Return node id of sorted node j in node set with id i, by local id
  int get_sorted_node_set_entry(int i, int j){return sorted_ns_node_list[i][j];}
  /// Creates sorted nodelist based on increasing x, y and z. Used for projection method.
  void create_sorted_nodelist();
  /// Return sorted node list
  std::vector<int> get_sorted_node_num_map(){return sorted_node_num_map;}
  /// Creates sorted elemlist based on increasing x, y and z. Used for projection method.
  void create_sorted_elemlist();
  /// Creates sorted elemlist based on increasing y, x and z. Used for projection method.
  void create_sorted_elemlist_yxz();
  /// Return sorted node list
  std::vector<int> get_sorted_elem_num_map(){return sorted_elem_num_map;}
  int get_num_nodes_per_ns(const int i){return num_nodes_per_ns[i];}
  std::vector<std::vector<int> > connect;
  int close_exodus(int ex_id);
  bool side_set_found(int ss);
  bool node_set_found(int ns);

 private:

  /// Return the number of nodes in side set with id i   !!! is this global or local !!!
  int get_num_node_per_side(int i){return ss_ctr_list[i][0];}
  /// Return my node_num_mapi (on this processor)   !!! is this global or local !!!
  std::vector<mesh_lint_t> get_my_node_num_mapi(){ return node_mapi; }
  /// Return my node_num_mapb (on this processor)   !!! is this global or local !!!
  std::vector<mesh_lint_t> get_my_node_num_mapb(){ return node_mapb; }
  /// Return a pointer to the node ids in element elem in block blk   !!! is this global or local !!!
  int *get_elem_nodes(int blk, int elem){return &connect[blk][elem * num_node_per_elem_in_blk[blk]]; }
  /// Return the number of node sets   !!! is this global or local !!!
  int get_num_node_sets(){return num_node_sets;}
  /// Return the number of side sets   !!! is this global or local !!!
  int get_num_side_sets(){return num_side_sets;}
  /// Compute nodal patch elements.
  void compute_nodal_patch();
  /// Return the x vector   !!! is this global or local !!!
  std::vector<double> *get_x_vector(){ return &x;}
  /// Return the y vector   !!! is this global or local !!!
  std::vector<double> *get_y_vector(){ return &y;}
  /// Return the z vector   !!! is this global or local !!!
  std::vector<double> *get_z_vector(){ return &z;}
  /// Return element vertex map
  std::map<int,int> get_vertex_map(){return vertex_map;}
  /// Compute the element vertex map
  void set_vertex_map();
  /// Return number of element vertices. 
  int get_num_vertices(){return num_vertices;}
  /// Set num_my_nodes to n
  void set_my_num_nodes(int n){num_my_nodes = n;}
  /// Return boundary status of element elem in block blk
  int get_boundary_status(int blk, int elem);
  /// Return boundary status of node nodeid   !!! is this global or local !!!
  int get_node_boundary_status(int nodeid);
  /// !!! have no idea !!!
  int get_node_set_value(int i){ return node_set_map[i]; }
  /// !!! have no idea !!!
  int get_side_set_node_value(int i){ return side_set_node_map[i]; }
  std::vector<mesh_lint_t> node_num_map;

  bool verbose;
  std::string title;
  float exodus_version;

  int num_dim;
  int num_nodes;
  int num_elem;
  int num_elem_blk;
  int num_node_sets;
  int num_side_sets;
  int num_nodal_fields;
  int num_elem_fields;
  int num_vertices;
  int num_my_nodes;

  std::vector<double> x;               // x locations of node points
  std::vector<double> y;               // y locations of node points
  std::vector<double> z;               // z locations of node points

  std::vector<int> blk_ids;
  std::vector<int> num_elem_in_blk;
  std::vector<std::string> blk_elem_type;
  std::vector<int> num_node_per_elem_in_blk;
  //std::vector<std::vector<int> > connect;
  std::vector< std::vector <mesh_lint_t> > elem_connect;      

  std::vector<int> ss_ids;
  std::vector<int> num_sides_per_ss;
  std::vector<int> num_df_per_ss;
  std::vector<std::vector<int> > ss_elem_list;
  std::vector<std::vector<int> > ss_side_list;
  std::vector<std::vector<int> > ss_node_list;
  std::vector<std::vector<int> > ss_ctr_list;

  std::vector<int> ns_ids;
  std::vector<int> num_nodes_per_ns;
  std::vector<int> num_df_per_ns;
  std::vector<std::vector<int> > ns_node_list;
  std::vector<std::vector<int> > sorted_ns_node_list;
  //std::vector<std::vector<int> > ns_ctr_list;

  std::vector<std::vector<int> > nodal_adj; //cn we may only need this for epetra
  //std::vector<int> nodal_adj_idx;
  //std::vector<int> nodal_adj_array;

  std::vector<std::string> nodal_field_names;      
  std::vector<std::vector<double> > nodal_fields;

  std::vector<std::string> elem_field_names;      
  std::vector<std::vector<double> > elem_fields;

  std::map<int,int> vertex_map;      

  int write_nodal_coordinates_exodus(int ex_id);
  int write_element_blocks_exodus(int ex_id);
  int write_nodal_data_exodus(int ex_id);
  int write_nodal_data_exodus(int ex_id, int counter);
  int write_elem_data_exodus(int ex_id, int counter);
  int write_elem_data_exodus(int ex_id);
  void check_exodus_error(const int ex_err,const std::string msg);
  int get_nodal_field_index(std::string name);
  int read_nodal_field_index(const int ex_id, std::string name);
  int get_elem_field_index(std::string name);
  //std::vector<int> node_num_map;
  std::vector<mesh_lint_t> elem_num_map;

  std::vector<int> node_set_map;
  std::vector<int> side_set_node_map;
  std::vector<mesh_lint_t> my_node_num_map;

  int ne_num_global_nodes, ne_num_global_elems, ne_num_global_elem_blks,
		  ne_num_global_node_sets, ne_num_global_side_sets;

  int num_internal_nodes, num_border_nodes, num_external_nodes,
      num_internal_elems, num_border_elems, num_node_cmaps, num_elem_cmaps;

  std::vector<int> node_cmap_ids, node_cmap_node_cnts, elem_cmap_ids, elem_cmap_elem_cnts;

  std::vector<mesh_lint_t> node_mapi, node_mapb, node_mape, elem_mapi, elem_mapb;

  std::vector<int> global_ns_ids, num_global_node_counts, num_global_node_df_counts;
  std::vector<int> global_ss_ids, num_global_side_counts, num_global_side_df_counts;
  std::vector<int> global_elem_blk_ids, global_elem_blk_cnts;

  std::vector<std::vector<int> > node_ids_in_cmap, n_proc_ids_in_cmap;
  std::vector<std::vector<int> > elem_ids_in_cmap, e_side_ids_in_cmap, e_proc_ids_in_cmap;

  std::vector<std::vector<int> > nodal_patch;//[nodeid][elemnt ids in patch
  std::vector<std::vector<int> > nodal_patch_overlap;//[nodeid][elemnt ids in patch

  int proc_id, nprocs, nprocs_infile;

  int exid;

  char filetype;

  std::string global_file_name;

  bool is_global_node_local(int i);
  bool is_global_elem_local(int i);
  bool is_nodesets_sorted;
  bool is_compute_nodal_patch_overlap;
  std::vector<int> sorted_node_num_map;
  std::vector<int> sorted_elem_num_map;

};

#endif
