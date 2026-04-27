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

  PARAM_FUNC(param_split)
  {
    // might be worth trying to think of a way for the user
    // to pass these values in from ModelEvaluator...
    pdes::kks::eta_start_idx = 2;
    pdes::kks::c_start_idx = 0;
    pdes::kks::mu_start_idx = 1;
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

  INI_FUNC(init_mu)
  {
    const int Neta_max = pdes::kks::Neta_max;
    const int Neta = pdes::kks::Neta;
    const int eta_start_idx = pdes::kks::eta_start_idx;

    double eta[Neta_max];
    for (int k = 0; k < Neta; ++k) {
      eta[k] = init_eta(x, y, z, k + eta_start_idx, lid);
    }
    const double hh = pdes::parabolicenergy::h(eta);
    const double c = init_c(x, y, z, eqn_id, lid);

    double ca = pdes::parabolicenergy::c1;
    double cb = pdes::parabolicenergy::c2;
    tools::solvers::solve_kks(c, hh, ca, cb,
                              pdes::parabolicenergy::dfa_dca,
                              pdes::parabolicenergy::dfb_dcb,
                              pdes::parabolicenergy::d2fa_dca2,
                              pdes::parabolicenergy::d2fb_dcb2);
    // based off eq (28) in the original KKS paper
    return pdes::parabolicenergy::dfa_dca(ca);
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
  RES_FUNC_TPETRA(residual_c_split)
  {
    return pdes::kks::pde_c_split(basis, i, dt_, dtold_,
                                  t_theta_, t_theta2_, time, eqn_id,
                                  vol, rand, mobility);
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_c_split_dp)) = residual_c_split;

  KOKKOS_INLINE_FUNCTION
  RES_FUNC_TPETRA(residual_mu)
  {
    return pdes::kks::pde_mu(basis, i, dt_, dtold_,
                             t_theta_, t_theta2_, time, eqn_id,
                             vol, rand);
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_mu_dp)) = residual_mu;

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


namespace sheng
{

  
  TUSAS_DEVICE const double initial_c_alpha = 0.05;
  TUSAS_DEVICE double r = 5e-6;  // cm
  TUSAS_DEVICE double d = 5e-6;  // cm
  TUSAS_DEVICE double S = 0.05;
  TUSAS_DEVICE double D = 0.019 * std::exp(-5840. / 673.);  // cm^2/s


  PARAM_FUNC(param)
  {
    r = plist->get<double>("r", r);
    d = plist->get<double>("d", d);
    S = plist->get<double>("S", S);
    D = plist->get<double>("D", D);

    // set base PDE params
    pdes::kks::param(plist);
    
    const int x0 = pdes::kks::x0;
    const int f0 = pdes::kks::f0;
    const int t0 = pdes::kks::t0;
    
    r /= x0;
    d /= x0;
    S *= t0;
    D = D * t0 * f0 / x0 / x0;
  }

  PARAM_FUNC(param_split)
  {
    // might be worth trying to think of a way for the user
    // to pass these values in from ModelEvaluator...
    pdes::kks::eta_start_idx = 2;
    pdes::kks::c_start_idx = 0;
    pdes::kks::mu_start_idx = 1;
  }

  
  INI_FUNC(init_eta)
  {
    const double sqrtw = std::sqrt(pdes::kks::w);
    const double rr = std::sqrt(x * x + y * y + z * z);
    
    // this is l = xi * sqrt(2 * k_eta_) / sqrtw with xi = 1 instead of 4
    const double xi = 1.;
    const double lambda = xi * std::sqrt(2 * pdes::kks::k_eta) / sqrtw;
    return 0.5 * (1. - tanh((rr - r) / lambda));
  }

  INI_FUNC(init_c)
  {
    const double eta = init_eta(x, y, z, eqn_id, lid);
    const double hh = pdes::parabolicenergy::h(&eta);
    return pdes::parabolicenergy::c1 * hh + initial_c_alpha * (1. - hh);
  }

  INI_FUNC(init_mu)
  {
    const int Neta_max = pdes::kks::Neta_max;
    const int Neta = pdes::kks::Neta;
    const int eta_start_idx = pdes::kks::eta_start_idx;

    double eta[Neta_max];
    for (int k = 0; k < Neta; ++k) {
      eta[k] = init_eta(x, y, z, k + eta_start_idx, lid);
    }
    const double hh = pdes::parabolicenergy::h(eta);
    const double c = init_c(x, y, z, eqn_id, lid);

    double ca = pdes::parabolicenergy::c1;
    double cb = pdes::parabolicenergy::c2;
    tools::solvers::solve_kks(c, hh, ca, cb,
                              pdes::parabolicenergy::dfa_dca,
                              pdes::parabolicenergy::dfb_dcb,
                              pdes::parabolicenergy::d2fa_dca2,
                              pdes::parabolicenergy::d2fb_dcb2);
    // based off eq (28) in the original KKS paper
    return pdes::parabolicenergy::dfa_dca(ca);
  }

  KOKKOS_INLINE_FUNCTION
  const double mobility(const double hh) {
    // M = D * d2f_dc2
    return D * pdes::parabolicenergy::d2fa_dca2() * pdes::parabolicenergy::d2fb_dcb2() 
             / ((1 - hh) * pdes::parabolicenergy::d2fa_dca2() + hh * pdes::parabolicenergy::d2fb_dcb2());
  }
  
  KOKKOS_INLINE_FUNCTION 
  const double S_forcing(const double y){
    return (y < d) ? S : 0;
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
  RES_FUNC_TPETRA(residual_c_split)
  {
    const double y = -(basis[0]->yy());
    const double s = S_forcing(y) * basis[0]->phi(i);
    
    return pdes::kks::pde_c_split(basis, i, dt_, dtold_,
                                  t_theta_, t_theta2_, time, eqn_id,
                                  vol, rand, mobility) - s;
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_c_split_dp)) = residual_c_split;

  KOKKOS_INLINE_FUNCTION
  RES_FUNC_TPETRA(residual_mu)
  {
    return pdes::kks::pde_mu(basis, i, dt_, dtold_,
                             t_theta_, t_theta2_, time, eqn_id,
                             vol, rand);
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_mu_dp)) = residual_mu;

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


}  // namespace sheng

    
}  // namespace cases


#endif  // ifndef CASES_HPP

