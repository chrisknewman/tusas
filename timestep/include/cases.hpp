//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////


#ifndef CASES_HPP
#define CASES_HPP


#include "pdes.hpp"


namespace cases
{


namespace tonks1
{


  PARAM_FUNC(param)
  {
    // if we wanted to set default values for all the 
    // parameters in pdes::kks, we could set them manually here?
    pdes::kks::param(plist);
  }

  KOKKOS_INLINE_FUNCTION
  const double mobility(const double hh) {
    return pdes::kks::M * (1. - hh) + hh;
  } 

  INI_FUNC(init_eta)
  {
    const double sqrt2 = std::sqrt(2.);
    const double sqrtw = std::sqrt(pdes::kks::w);
    const double k_eta = pdes::kks::k_eta;
    
    // this is l = xi * sqrt(2 * k_eta_) / sqrtw with xi = 1 instead of 4
    const double l = std::sqrt(2 * k_eta) / sqrtw;
    return 0.5 * (1. - tanh((x - 30.) / (l * sqrt2)));
  }

  INI_FUNC(init_c)
  {
    const double eta = init_eta(x, y, z, eqn_id, lid);
    const double hh = pdes::parabolicenergy::h(&eta);
    return pdes::parabolicenergy::c1 * hh + pdes::parabolicenergy::c2 * (1. - hh);
  }

  KOKKOS_INLINE_FUNCTION
  RES_FUNC_TPETRA(residual_eta)
  {
    return pdes::kks::pde_eta(basis, i, dt_, dtold_,
                              t_theta_, t_theta2_, time, eqn_id,
                              vol, rand, mobility);
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_eta_dp)) = residual_eta;

  KOKKOS_INLINE_FUNCTION
  RES_FUNC_TPETRA(residual_c)
  {
    return pdes::kks::pde_c(basis, i, dt_, dtold_,
                            t_theta_, t_theta2_, time, eqn_id,
                            vol, rand, mobility);
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_c_dp)) = residual_c;

  KOKKOS_INLINE_FUNCTION
  PRE_FUNC_TPETRA(prec_eta)
  {
    return pdes::kks::prec_eta(basis, i, j, dt_, t_theta_, eqn_id);
  }

  KOKKOS_INLINE_FUNCTION
  PRE_FUNC_TPETRA(prec_c)
  {
    return pdes::kks::prec_c(basis, i, j, dt_, t_theta_, eqn_id, mobility);
  }


}  // namespace tonks1

    
}  // namespace cases


#endif  // ifndef CASES_HPP

