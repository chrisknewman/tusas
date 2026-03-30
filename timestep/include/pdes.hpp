//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////


#ifndef PDES_HPP
#define PDES_HPP


#include <boost/ptr_container/ptr_vector.hpp>
#include "basis.hpp"

#include "tools.hpp"
	
#include "Teuchos_ParameterList.hpp"

#include <Kokkos_Core.hpp>


#if defined (KOKKOS_HAVE_CUDA) || defined (KOKKOS_ENABLE_CUDA) || (KOKKOS_HAVE_HIP) || defined (KOKKOS_ENABLE_HIP)
#define TUSAS_DEVICE __device__
#else
#define TUSAS_DEVICE /**/ 
#endif

#if defined (KOKKOS_HAVE_CUDA) || defined (KOKKOS_ENABLE_CUDA)
#define TUSAS_HAVE_CUDA
#endif

#if defined(KOKKOS_HAVE_HIP) || defined (KOKKOS_ENABLE_HIP)
#define TUSAS_HAVE_HIP
#endif


/*
 * Definition for residual function. Each residual function is called at each Gauss point 
 * for each equation with this signature:
 *   NAME: name of function to call
 *   const boost::ptr_vector<Basis> &basis: an array of basis function objects indexed by equation
 *   const int &i: the current test function (row in residual vector)
 *   const double &dt_: the timestep size as prescribed in input file						
 *   const double &t_theta_: the timestep parameter as prescribed in input file
 *   const double &time: the current simulation time
 *   const int &eqn_id: the index of the current equation
 */
#define RES_FUNC_TPETRA(NAME) const double NAME(GPUBasis *basis[], \
                                                const int i, \
                                                const double dt_, \
                                                const double dtold_, \
                                                const double t_theta_, \
                                                const double t_theta2_, \
                                                const double time, \
                                                const int eqn_id, \
                                                const double vol, \
                                                const double rand)

/*
 * Definition for precondition function. Each precondition function is called at each Gauss point 
 * for each equation with this signature:
 *   NAME: name of function to call
 *   const boost::ptr_vector<Basis> &basis: an array of basis function objects indexed by equation
 *   const int &i: the current basis function (row in preconditioning matrix)
 *   const int &j: the current test function (column in preconditioning matrix)
 *   const double &dt_: the timestep size as prescribed in input file						
 *   const double &t_theta_: the timestep parameter as prescribed in input file
 *   const double &time: the current simulation time
 *   const int &eqn_id: the index of the current equation
 */
#define PRE_FUNC_TPETRA(NAME) const double NAME(GPUBasis *basis[], \
                                                const int i, \
                                                const int j, \
                                                const double dt_, \
                                                const double t_theta_, \
                                                const int eqn_id)

/*
 * Definition for post-process function. Each post-process function is called at each node for each equation at the 
 * end of each timestep with this signature:
 *   NAME: name of function to call
 *   const double *u: an array of solution values indexed by equation
 *   const double *gradu: an array of gradient values indexed by equation, coordinates (NULL unless error estimation is activated)
 *   const double *xyz: an array of coordinates indexed by equation, coordinates
 *   const double &time: the current simulation time
 */
#define PPR_FUNC(NAME) double NAME(const double *u, \
                                   const double *uold, \
                                   const double *uoldold, \
                                   const double *gradu, \
                                   const double *xyz, \
                                   const double &time, \
                                   const double &dt, \
                                   const double &dtold, \
                                   const int &eqn_id)

/*
 * Parameter function to propogate information from input file. Each parameter function is called at the beginning of each simulation.
 *   NAME: name of function to call
 *   Teuchos::ParameterList *plist: paramterlist containing information defined in input file
 */
#define PARAM_FUNC(NAME) void NAME(Teuchos::ParameterList *plist) 


namespace pdes
{


namespace kks
{
  const int n_c_max = 4;
  int n_c = 1;
}  // namespace kks

    
}  // namespace pdes


#endif  // ifndef PDES_HPP

