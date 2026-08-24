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


namespace mansoln
{


  TUSAS_DEVICE double x_offset = 10.;
  TUSAS_DEVICE double epsilon = 0.4;
  TUSAS_DEVICE double v = 1.;


  KOKKOS_INLINE_FUNCTION
  const double w(const double x, const double t) {
    return ((x - x_offset) - v * t) / (epsilon * std::sqrt(2));
  } 

  KOKKOS_INLINE_FUNCTION
  const double eta_mms(const double x, const double t)
  {
    return 0.5 * (1 + tanh(w(x, t)));
  }

  KOKKOS_INLINE_FUNCTION
  const double mobility(const double unused) {
    return pdes::kks::M;
  } 

  KOKKOS_INLINE_FUNCTION
  const double dg_deta(const double *etas, const int eqn_id)
  {
    const double eta = etas[eqn_id];
    return 2 * eta * (2 * eta - 1) * (eta - 1);
  }

  PARAM_FUNC(param)
  {
    pdes::kks::param(plist);
    pdes::kks::eta_start_idx = 0;

    x_offset = plist->get<double>("x_offset", x_offset);
    epsilon = plist->get<double>("epsilon", epsilon);
    v = plist->get<double>("v", v);
  }

  PARAM_FUNC(param_freeenergy_parabolic)
  {
    pdes::kks::param_freeenergy_parabolic(plist);
    pdes::kks::fe.dg_deta = &dg_deta;
  }

  INI_FUNC(init_eta)
  {
    return eta_mms(x, 0.);
  }

  KOKKOS_INLINE_FUNCTION
  RES_FUNC_TPETRA(source_eta_constmu)
  {
    const int Nt_max = pdes::kks::Nt_max;
    const int Neta_max = pdes::kks::Neta_max;
    const int Neta = pdes::kks::Neta;
    const int eta_start_idx = pdes::kks::eta_start_idx;

    const double L = pdes::kks::L;
    const double k_eta = pdes::kks::k_eta;
    const double w = pdes::kks::w;

    const double ca = pdes::kks::fe.c1a_0;
    const double cb = pdes::kks::fe.c1b_0;

    const int Nt = 3;
    const int local_id = eqn_id - eta_start_idx;

    const double phi = basis[0]->phi(i);
    const double x = basis[0]->xx();

    double eta[Nt_max];
    eta[0] = eta_mms(x, time + dt_);
    eta[1] = eta_mms(x, time);
    eta[2] = eta_mms(x, time - dtold_);

    double deta_dt, double_well, df_deta;
    double source[Nt_max];

    for (int tdx = 0; tdx < Nt; ++tdx) {
      deta_dt = ((2 * v) / (epsilon * std::sqrt(2))) * eta[tdx] * (1 - eta[tdx]);
      double_well = L * (w - k_eta / std::pow(epsilon, 2)) * pdes::kks::fe.dg_deta(&eta[tdx], local_id);
      df_deta = L * pdes::kks::fe.dh_deta(eta[tdx]) * (
                  pdes::kks::fe.fa(ca) 
                  - pdes::kks::fe.fb(cb) 
                  + pdes::kks::fe.dfa_dca(ca) * (cb - ca)
                );

      source[tdx] = (deta_dt - double_well - df_deta) * phi;
    }

    // time derivative entry is zero here because it will be added to the residual
    // by pde_eta()
    return tools::utils::ret_value(0., source, dt_, dtold_, t_theta_, t_theta2_);
  }

  KOKKOS_INLINE_FUNCTION
  RES_FUNC_TPETRA(residual_eta_constmu)
  {
    return pdes::kks::pde_eta_nokks(basis, i, dt_, dtold_,
                                    t_theta_, t_theta2_, time, eqn_id,
                                    vol, rand, mobility) +
           source_eta_constmu(basis, i, dt_, dtold_, 
                              t_theta_, t_theta2_, time, eqn_id,
                              vol, rand);
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_eta_constmu_dp)) = residual_eta_constmu;

  DBC_FUNC(dbc)
  {
    return eta_mms(x, t);
  }

  PPR_FUNC(postproc_exact_soln)
  {
    const double x = xyz[0];
    return eta_mms(x, time);
  }

  PPR_FUNC(postproc_ptwise_abs)
  {
    const double x = xyz[0];
    return std::abs(eta_mms(x, time) - u[0]);
  }


}


namespace tonks1
{


  PARAM_FUNC(param)
  {
    // if we wanted to set default values for all the 
    // parameters in pdes::kks, we could set them manually here?
    pdes::kks::param(plist);
  }

  PARAM_FUNC(param_freeenergy_parabolic)
  {
    pdes::kks::param_freeenergy_parabolic(plist);
  }
  
  PARAM_FUNC(param_freeenergy_calenergy)
  {
    pdes::kks::param_freeenergy_calenergy(plist);
  }

  PARAM_FUNC(param_split)
  {
    // might be worth trying to think of a way for the user
    // to pass these values in from ModelEvaluator...
    pdes::kks::eta_start_idx = 2;
    pdes::kks::c_start_idx = 0;
    pdes::kks::mu_start_idx = 1;
  }

  PARAM_FUNC(param_trans)
  {
    // might be worth trying to think of a way for the user
    // to pass these values in from ModelEvaluator...
    pdes::kks::eta_start_idx = 2;
    pdes::kks::c_start_idx = 1;
    pdes::kks::mu_start_idx = 0;
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
    const double hh = pdes::kks::fe.h(&eta);
    return pdes::kks::fe.c1a_0 * hh + pdes::kks::fe.c1b_0 * (1. - hh);
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
    const double hh = pdes::kks::fe.h(eta);
    const double c = init_c(x, y, z, eqn_id, lid);

    double ca = pdes::kks::fe.c1a_0;
    double cb = pdes::kks::fe.c1b_0;
    tools::solvers::solve_kks(c, hh, ca, cb,
                              pdes::kks::fe.dfa_dca,
                              pdes::kks::fe.dfb_dcb,
                              pdes::kks::fe.d2fa_dca2,
                              pdes::kks::fe.d2fb_dcb2);
    // based off eq (28) in the original KKS paper
    return pdes::kks::fe.dfa_dca(ca);
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
  RES_FUNC_TPETRA(residual_eta_lagkks)
  {
    return pdes::kks::pde_eta(basis, i, dt_, dtold_,
                              t_theta_, t_theta2_, time, eqn_id,
                              vol, rand, mobility, true);
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_eta_lagkks_dp)) = residual_eta_lagkks;

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
  RES_FUNC_TPETRA(residual_c_trans)
  {
    return pdes::kks::pde_mu(basis, i, dt_, dtold_,
                             t_theta_, t_theta2_, time, eqn_id,
                             vol, rand, true);
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_c_trans_dp)) = residual_c_trans;

  KOKKOS_INLINE_FUNCTION
  RES_FUNC_TPETRA(residual_c_trans_lagkks)
  {
    // TODO -- lagkks bug here??
    return pdes::kks::pde_mu(basis, i, dt_, dtold_,
                             t_theta_, t_theta2_, time, eqn_id,
                             vol, rand, true, true);  // should be true
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_c_trans_lagkks_dp)) = residual_c_trans_lagkks;

  KOKKOS_INLINE_FUNCTION
  RES_FUNC_TPETRA(residual_mu)
  {
    return pdes::kks::pde_mu(basis, i, dt_, dtold_,
                             t_theta_, t_theta2_, time, eqn_id,
                             vol, rand);
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_mu_dp)) = residual_mu;

    
  KOKKOS_INLINE_FUNCTION
  RES_FUNC_TPETRA(residual_mu_trans)
  {
    return pdes::kks::pde_c_split(basis, i, dt_, dtold_,
                                  t_theta_, t_theta2_, time, eqn_id,
                                  vol, rand, mobility, true);
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_mu_trans_dp)) = residual_mu_trans;

  KOKKOS_INLINE_FUNCTION
  RES_FUNC_TPETRA(residual_mu_trans_lagkks)
  {
    return pdes::kks::pde_c_split(basis, i, dt_, dtold_,
                                  t_theta_, t_theta2_, time, eqn_id,
                                  vol, rand, mobility, true, true);
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_mu_trans_lagkks_dp)) = residual_mu_trans_lagkks;

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

  KOKKOS_INLINE_FUNCTION
  PRE_FUNC_TPETRA(prec_c_trans)
  {
    return pdes::kks::prec_c_trans(basis, i, j, dt_, t_theta_, eqn_id);
  }

  KOKKOS_INLINE_FUNCTION
  PRE_FUNC_TPETRA(prec_mu_trans)
  {
    return pdes::kks::prec_mu_trans(basis, i, j, dt_, t_theta_, eqn_id, mobility);
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

  PARAM_FUNC(param_trans)
  {
    pdes::kks::eta_start_idx = 2;
    pdes::kks::c_start_idx = 1;
    pdes::kks::mu_start_idx = 0;
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
    const double hh = pdes::kks::fe.h(&eta);
    return pdes::kks::fe.c1a_0 * hh + initial_c_alpha * (1. - hh);
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
    const double hh = pdes::kks::fe.h(eta);
    const double c = init_c(x, y, z, eqn_id, lid);

    double ca = pdes::kks::fe.c1a_0;
    double cb = pdes::kks::fe.c1b_0;
    tools::solvers::solve_kks(c, hh, ca, cb,
                              pdes::kks::fe.dfa_dca,
                              pdes::kks::fe.dfb_dcb,
                              pdes::kks::fe.d2fa_dca2,
                              pdes::kks::fe.d2fb_dcb2);
    // based off eq (28) in the original KKS paper
    return pdes::kks::fe.dfa_dca(ca);
  }

  KOKKOS_INLINE_FUNCTION
  const double mobility(const double hh) {
    // M = D / d2f_dc2
    return D / pdes::kks::fe.d2f_dc2(hh, 0., 0.);
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
  RES_FUNC_TPETRA(residual_c)
  {
    const double y = -(basis[0]->yy());
    const double s = S_forcing(y) * basis[0]->phi(i);
    
    return pdes::kks::pde_c(basis, i, dt_, dtold_,
                            t_theta_, t_theta2_, time, eqn_id,
                            vol, rand, mobility) - s;
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_c_dp)) = residual_c;

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

  KOKKOS_INLINE_FUNCTION
  PRE_FUNC_TPETRA(prec_mu)
  {
    return pdes::kks::prec_mu(basis, i, j, dt_, t_theta_, eqn_id, mobility);
  }

  PPR_FUNC(postproc_mobility)
  {
    return pdes::kks::postproc_mobility(u, uold, uoldold, 
                                        gradu, xyz, time, 
                                        dt, dtold, eqn_id,
                                        mobility);
  }



}  // namespace sheng

    
}  // namespace cases


#endif  // ifndef CASES_HPP

