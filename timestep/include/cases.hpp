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
#include "tools.hpp"


namespace cases
{


namespace parabolicenergy
{
  /*
   *
   * This namespace implements parabolic free energy
   * density functions, their derivatives, and related
   * functions
   *
   * As in the kks namespace, this implementation assumes
   * two phases, an a-phase ("solid") and a b-phase ("liquid"):
   *   a corresponds to eta
   *   b corresponds to 1 - eta
   *
   * The free energy density functions are
   *   fa(ca) = Aa_ * (ca - (c1_ + delta_c1_))^2 + f1_
   *   fb(cb) = Ab_ * (cb - (c2_ + delta_c2_))^2 + f2_
   *
   * We also supply overloaded versions of relevant functions
   * where we assume that ca = cb = c for non-kks solves
   *
   */
  TUSAS_DEVICE double Aa = 2.;
  TUSAS_DEVICE double Ab = 2.;
  TUSAS_DEVICE double c1 = 0.3;
  TUSAS_DEVICE double c2 = 0.7;
  TUSAS_DEVICE double delta_c1 = 0.;
  TUSAS_DEVICE double delta_c2 = 0.;
  TUSAS_DEVICE double f1 = 0.;
  TUSAS_DEVICE double f2 = 0.;
  TUSAS_DEVICE const double alpha = 5.;  // pfhub2 coef in g
  
  TUSAS_DEVICE const int Neta_max = 4;
  TUSAS_DEVICE int Neta = 1;

  PARAM_FUNC(param)
  {
    int Np = plist->get<int>("N_ETA", Neta);
#ifdef TUSAS_HAVE_CUDA
    cudaMemcpyToSymbol(Neta, &Np, sizeof(int));
#else
    Neta = Np;
#endif
    if(Neta > Neta_max) exit(0);

    // parameters from Sheng 2022, eqns 8, 9
    Aa = plist->get<double>("Aa", Aa);
    Ab = plist->get<double>("Ab", Ab);
    c1 = plist->get<double>("c1", c1);
    c2 = plist->get<double>("c2", c2);
    delta_c1 = plist->get<double>("delta_c1", delta_c1);
    delta_c2 = plist->get<double>("delta_c2", delta_c2);
    f1 = plist->get<double>("f1", f1);
    f2 = plist->get<double>("f2", f2);
  }

  KOKKOS_INLINE_FUNCTION
  const double fa(const double ca)
  {
    // f_alpha(c_alpha) from Sheng 2022, eqn 8
    // altered to be more general as eqn 9
    return Aa * (ca - (c1 + delta_c1))
              * (ca - (c1 + delta_c1)) + f1;
  }

  KOKKOS_INLINE_FUNCTION
  const double fb(const double cb)
  {
    // f_beta(c_beta) from Sheng 2022, eqn 9
    return Ab * (cb - (c2 + delta_c2))
              * (cb - (c2 + delta_c2)) + f2;
  }

  KOKKOS_INLINE_FUNCTION
  const double dfa_dca(const double ca)
  {
    return 2 * Aa * (ca - (c1 + delta_c1));
  }

  KOKKOS_INLINE_FUNCTION
  const double dfb_dcb(const double cb)
  {
    return 2 * Ab * (cb - (c2 + delta_c2));
  }

  KOKKOS_INLINE_FUNCTION
  const double d2fa_dca2()
  {
    return 2 * Aa;
  }

  KOKKOS_INLINE_FUNCTION
  const double d2fb_dcb2()
  {
    return 2 * Ab;
  }

  KOKKOS_INLINE_FUNCTION 
  const double h(const double *eta)
  {
    double val = 0.;
    for (int i = 0; i < Neta; i++) {
      val += eta[i] * eta[i] * eta[i] 
               * (6. * eta[i] * eta[i] - 15. * eta[i] + 10.);
    }
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  double dh_deta(const double eta)
  {
    return eta * eta * ((30. * eta - 60.) * eta + 30.);
  }

  KOKKOS_INLINE_FUNCTION 
  const double dg_deta(const double *eta, const int eqn_id)
  {
    double aval = 0.;
    for (int i = 0; i < Neta; i++) {
      aval += eta[i] * eta[i];
    }
    aval = aval - eta[eqn_id]*eta[eqn_id];
    return 2. * eta[eqn_id] * (1. - eta[eqn_id]) * (1. - eta[eqn_id])  
             - 2. * eta[eqn_id] * eta[eqn_id] * (1. - eta[eqn_id])
             + 4. * alpha * eta[eqn_id] * aval;
  }

  KOKKOS_INLINE_FUNCTION 
  double f(const double c, const double *eta)
  {
    // assumes that ca and cb are the same quanitity 
    const double hh = h(eta);
    return fa(c) * hh + fb(c) * (1. - hh);
  }

  KOKKOS_INLINE_FUNCTION 
  double f(const double ca, const double cb, const double *eta)
  {
    const double hh = h(eta);
    return fa(ca) * hh + fb(cb) * (1. - hh);
  }

  KOKKOS_INLINE_FUNCTION 
  double df_dc(const double c, const double *eta)
  {
    // assumes that ca and cb are the same quanitity
    const double hh = h(eta);
    return dfa_dca(c) * hh + dfb_dcb(c) * (1. - hh);
  }

  KOKKOS_INLINE_FUNCTION 
  double df_dc(const double ca, const double cb, const double *eta)
  {
    // TODO: remove this function
    // assumes that dca/dc = dcb/dc = 1, which is not true
    // probably best to never use this function and use 
    // eq 28 from KKS instead -- leaving it here for now
    const double hh = h(eta);
    return dfa_dca(ca) * hh + dfb_dcb(cb) * (1. - hh);
  }

  KOKKOS_INLINE_FUNCTION
  double d2f_dc2(const double *eta)
  {
    // assumes that ca and cb are the same quanitity
    const double hh = h(eta);
    return d2fa_dca2() * hh + d2fb_dcb2() * (1. - hh);
  }

  KOKKOS_INLINE_FUNCTION 
  const double df_deta(const double c, const double eta)
  {
    // assumes that ca and cb are the same quanitity
    // and does **not** include the w g' term

    // dh(eta1, eta2) / deta1 is a function of eta1 only
    const double dhdeta = dh_deta(eta);
    return fa(c) * dhdeta + fb(c) * (-dhdeta);
  }

  KOKKOS_INLINE_FUNCTION 
  const double df_deta(const double ca, const double cb, const double eta)
  {
    // does not include the w g' term

    // dh(eta1, eta2) / deta1 is a function of eta1 only
    const double dhdeta = dh_deta(eta);
    return fa(ca) * dhdeta + fb(cb) * (-dhdeta)
             + dhdeta * dfb_dcb(cb) * (cb - ca);
  }


}  // namespace parabolicenergy


namespace tonks1
{
 
  TUSAS_DEVICE const int Nt_max = 3;
  
  TUSAS_DEVICE const int Nc_max = 2;
  TUSAS_DEVICE int Nc = 1;
  TUSAS_DEVICE int c_start_idx = 0;

  TUSAS_DEVICE const int Nmu_max = 2;
  TUSAS_DEVICE int Nmu = 1;
  TUSAS_DEVICE int mu_start_idx = 1;

  TUSAS_DEVICE const int Neta_max = 4;
  TUSAS_DEVICE int Neta = 1;
  TUSAS_DEVICE int eta_start_idx = 1;
  
  TUSAS_DEVICE double t0 = 1.;
  TUSAS_DEVICE double x0 = 1.;
  TUSAS_DEVICE double f0 = 1.;
  TUSAS_DEVICE double k_c = 0.;
  TUSAS_DEVICE double k_eta = 1.5;
  TUSAS_DEVICE double M = 10.;
  TUSAS_DEVICE double L = 2.;
  TUSAS_DEVICE double w = 12.;

  PARAM_FUNC(param)
  {
    int Np = plist->get<int>("N_ETA", Neta);
#ifdef TUSAS_HAVE_CUDA
    cudaMemcpyToSymbol(Neta, &Np, sizeof(int));
#else
    Neta = Np;
#endif
    if(Neta > Neta_max) exit(0);

    // nondim free energy density, J/m^3
    // generally, should be ~parabolicenergy::Aa_
    f0 = plist->get<double>("f0", f0);
    // nondim spatial scaling, m
    x0 = plist->get<double>("x0", x0);
    // nondim temporal scaling, s
    t0 = plist->get<double>("t0", t0);

    k_c = plist->get<double>("k_c", k_c);
    k_eta = plist->get<double>("k_eta", k_eta);
    M = plist->get<double>("M", M);
    L = plist->get<double>("L", L);
    w = plist->get<double>("w", w);

    // set params for free energy density
    parabolicenergy::param(plist);

    // nondimensionalize
    // note that if x0_, t0_, and f0_ are not
    // set by the user in an input file, these
    // values default to 1 and the original
    // dimensional equations are preserved
    parabolicenergy::Aa = parabolicenergy::Aa / f0;
    parabolicenergy::Ab = parabolicenergy::Ab / f0;
    parabolicenergy::f1 = parabolicenergy::f1 / f0;
    parabolicenergy::f2 = parabolicenergy::f2 / f0;
    k_c = k_c / x0 / x0 / f0;
    k_eta = k_eta / x0 / x0 / f0;
    M = M * t0 * f0 / x0 / x0;
    L = L * t0 * f0;
    w = w / f0;
  }

  KOKKOS_INLINE_FUNCTION
  const double mobility(const double hh) {
    return M * (1. - hh) + hh;
  } 

  KOKKOS_INLINE_FUNCTION 
  RES_FUNC_TPETRA(residual_eta)
  {
    const int Nt = 3;

    const int local_id = eqn_id - eta_start_idx;

    const double phi = basis[0]->phi(i);
    const double dphi_dx = basis[0]->dphidx(i);
    const double dphi_dy = basis[0]->dphidy(i);
    const double dphi_dz = basis[0]->dphidz(i);

    double c[Nt_max * Nc_max];
    double dc_dx[Nt_max * Nc_max];
    double dc_dy[Nt_max * Nc_max];
    double dc_dz[Nt_max * Nc_max];
    tools::utils::get_uu(c, Nc, Nc_max, c_start_idx, basis);
    tools::utils::get_graduu(dc_dx, dc_dy, dc_dz, Nc, Nc_max, c_start_idx, basis);

    double eta[Nt_max * Neta_max];
    double deta_dx[Nt_max * Neta_max];
    double deta_dy[Nt_max * Neta_max];
    double deta_dz[Nt_max * Neta_max];
    tools::utils::get_uu(eta, Neta, Neta_max, eta_start_idx, basis);
    tools::utils::get_graduu(deta_dx, deta_dy, deta_dz, Neta, Neta_max, eta_start_idx, basis);

    double hh[Nt_max];
    double ca[Nt_max];
    double cb[Nt_max];
    double k_divgrad_eta[Nt_max];
    double df_deta[Nt_max];
    double f[Nt_max];

    int idx = 0;
    for (int tdx = 0; tdx < Nt; ++tdx) {
      hh[tdx] = parabolicenergy::h(&eta[tdx * Neta_max]);
      
      ca[tdx] = parabolicenergy::c1;
      cb[tdx] = parabolicenergy::c2;
      idx = tools::utils::idx(tdx, local_id, Nc_max);
      tools::solvers::solve_kks(c[idx], hh[tdx], ca[tdx], cb[tdx],
                                parabolicenergy::dfa_dca,
                                parabolicenergy::dfb_dcb,
                                parabolicenergy::d2fa_dca2,
                                parabolicenergy::d2fb_dcb2);

      idx = tools::utils::idx(tdx, local_id, Neta_max);
      df_deta[tdx] = (parabolicenergy::df_deta(ca[tdx], cb[tdx], eta[idx])
                        + w * parabolicenergy::dg_deta(&eta[tdx * Neta_max], local_id)) * phi;
      k_divgrad_eta[tdx] = k_eta * (deta_dx[idx] * dphi_dx + deta_dy[idx] * dphi_dy + deta_dz[idx] * dphi_dz);

      f[tdx] = L* (k_divgrad_eta[tdx] + df_deta[tdx]);
    }

    const double deta_dt = (eta[tools::utils::idx(0, local_id, Neta_max)] 
                              - eta[tools::utils::idx(1, local_id, Neta_max)]) / dt_ * phi;

    return tools::utils::ret_value(deta_dt, f, dt_, dtold_, t_theta_, t_theta2_);
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_eta_dp)) = residual_eta;

  KOKKOS_INLINE_FUNCTION
  RES_FUNC_TPETRA(residual_c)
  {
    // number of time levels to compute
    // might want to pass this in to res func?
    const int Nt = 3;

    // test function
    const double phi = basis[0]->phi(i);
    // grad(phi)
    const double dphi_dx = basis[0]->dphidx(i);
    const double dphi_dy = basis[0]->dphidy(i);
    const double dphi_dz = basis[0]->dphidz(i);

    // populate c viewed as a "matrix"
    //   c[time_idx, c_idx]
    // but really a 1D array that
    // we can index this using
    //   utils::idx(time_idx, c_idx, Nc_max)
    double c[Nt_max * Nc_max] = {0.};
    double dc_dx[Nt_max * Nc_max] = {0.};
    double dc_dy[Nt_max * Nc_max] = {0.};
    double dc_dz[Nt_max * Nc_max] = {0.};
    tools::utils::get_uu(c, Nc, Nc_max, c_start_idx, basis);
    tools::utils::get_graduu(dc_dx, dc_dy, dc_dz, Nc, Nc_max, c_start_idx, basis);

    // populate eta viewed as a "matrix"
    //   eta[time_idx, eta_idx]
    // but really a 1D array that
    // we can index this using
    //   utils::idx(time_idx, eta_idx, Neta_max)
    double eta[Nt_max * Neta_max] = {0.};
    double deta_dx[Nt_max * Neta_max] = {0.};
    double deta_dy[Nt_max * Neta_max] = {0.};
    double deta_dz[Nt_max * Neta_max] = {0.};
    tools::utils::get_uu(eta, Neta, Neta_max, eta_start_idx, basis);
    tools::utils::get_graduu(deta_dx, deta_dy, deta_dz, Neta, Neta_max, eta_start_idx, basis);

    // define all the variables we need to calculate 
    // the residual = Mdivgrad_df_dc
    double hh[Nt_max] = {0.};
    double dh_dx[Nt_max] = {0.};
    double dh_dy[Nt_max] = {0.};
    double dh_dz[Nt_max] = {0.};
    double ca[Nt_max] = {0.};
    double cb[Nt_max] = {0.};
    double d2f_dc2[Nt_max] = {0.};
    double d2f_dcdx[Nt_max] = {0.};
    double d2f_dcdy[Nt_max] = {0.};
    double d2f_dcdz[Nt_max] = {0.};
    double Mdivgrad_df_dc[Nt_max] = {0.};


    // loop over each time level that we need data at
    int idx = 0;
    double dh_deta = 0;
    for (int tdx = 0; tdx < Nt; ++tdx) {
      // calculate h
      hh[tdx] = parabolicenergy::h(&eta[tdx * Neta_max]);

      // calculate grad h
      dh_dx[tdx] = 0.;
      dh_dy[tdx] = 0.;
      dh_dz[tdx] = 0.;
      for (int k = 0; k < Neta ; ++k) {
        idx = tools::utils::idx(tdx, k, Neta_max);
        dh_deta = parabolicenergy::dh_deta(eta[idx]);

        dh_dx[tdx] += dh_deta * deta_dx[idx];
        dh_dy[tdx] += dh_deta * deta_dy[idx];
        dh_dz[tdx] += dh_deta * deta_dz[idx];
      }

      // do the kks solve to get ca and cb
      // for the current component c_{eqn_id}
      ca[tdx] = parabolicenergy::c1;
      cb[tdx] = parabolicenergy::c2;
      tools::solvers::solve_kks(c[tools::utils::idx(tdx, eqn_id, Nc_max)],
                                hh[tdx],
                                ca[tdx],
                                cb[tdx],
                                parabolicenergy::dfa_dca,
                                parabolicenergy::dfb_dcb,
                                parabolicenergy::d2fa_dca2,
                                parabolicenergy::d2fb_dcb2);

      // calculate d2f_dc2 using KKS eq 29 
      d2f_dc2[tdx] = parabolicenergy::d2fa_dca2() * parabolicenergy::d2fb_dcb2() 
                       / ((1 - hh[tdx]) * parabolicenergy::d2fa_dca2() + hh[tdx] * parabolicenergy::d2fb_dcb2());

      // calculating grad(f_c) based on KKS eq 33, assuming M = D / f_cc
      // this also follows from eq 30 and the chain rule
      //   grad(f_c) = f_cc * h' * (cb - ca) * grad(eta) + f_cc * grad(c) 
      //             = f_cc * (cb - ca) * grad(h) + f_cc * grad(c) 
      idx = tools::utils::idx(tdx, eqn_id, Nc_max);
      d2f_dcdx[tdx] = d2f_dc2[tdx] * (cb[tdx] - ca[tdx]) * dh_dx[tdx] + d2f_dc2[tdx] * dc_dx[idx];
      d2f_dcdy[tdx] = d2f_dc2[tdx] * (cb[tdx] - ca[tdx]) * dh_dy[tdx] + d2f_dc2[tdx] * dc_dy[idx];
      d2f_dcdz[tdx] = d2f_dc2[tdx] * (cb[tdx] - ca[tdx]) * dh_dz[tdx] + d2f_dc2[tdx] * dc_dz[idx];

      // finally, calculate M * div(grad(f_c))
      Mdivgrad_df_dc[tdx] = mobility(hh[tdx]) * (d2f_dcdx[tdx] * dphi_dx
                                                 + d2f_dcdy[tdx] * dphi_dy
                                                 + d2f_dcdz[tdx] * dphi_dz);
    }  // tdx = 0, < Nt loop

    const double dc_dt = (c[tools::utils::idx(0, eqn_id, Nc_max)] 
                            - c[tools::utils::idx(1, eqn_id, Nc_max)]) / dt_ * phi;

    return tools::utils::ret_value(dc_dt, Mdivgrad_df_dc, dt_, dtold_, t_theta_, t_theta2_);
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_c_dp)) = residual_c;
  
  KOKKOS_INLINE_FUNCTION 
  PRE_FUNC_TPETRA(prec_eta)
  {
    const double phi_i = basis[0]->phi(i);
    const double phi_j = basis[0]->phi(j);

    const double dphi_dx_i = basis[0]->dphidx(i);
    const double dphi_dy_i = basis[0]->dphidy(i);
    const double dphi_dz_i = basis[0]->dphidz(i);
    const double dphi_dx_j = basis[0]->dphidx(j);
    const double dphi_dy_j = basis[0]->dphidy(j);
    const double dphi_dz_j = basis[0]->dphidz(j);

    const double divgrad = L * k_eta * (dphi_dx_i * dphi_dx_j 
                                        + dphi_dy_i * dphi_dy_j 
                                        + dphi_dz_i * dphi_dz_j);
    const double ut = phi_i * phi_j / dt_;

    return ut + t_theta_ * divgrad;
  }

  KOKKOS_INLINE_FUNCTION 
  PRE_FUNC_TPETRA(prec_c)
  {
    const double phi_i = basis[0]->phi(i);
    const double phi_j = basis[0]->phi(j);

    const double dphi_dx_i = basis[0]->dphidx(i);
    const double dphi_dy_i = basis[0]->dphidy(i);
    const double dphi_dz_i = basis[0]->dphidz(i);
    const double dphi_dx_j = basis[0]->dphidx(j);
    const double dphi_dy_j = basis[0]->dphidy(j);
    const double dphi_dz_j = basis[0]->dphidz(j);
    
    double eta[Neta_max];
    for(int k = 0; k < Neta; ++k){
      eta[k] = basis[k + eta_start_idx]->uu();
    }
    
    const double hh = parabolicenergy::h(eta);
    // note that we can skip the kks solve here only
    // because the free energy is parabolic -- the
    // second derivatives are constant
    const double d2f_dc2 = parabolicenergy::d2fa_dca2() * parabolicenergy::d2fb_dcb2() 
                             / ((1 - hh) * parabolicenergy::d2fa_dca2() + hh * parabolicenergy::d2fb_dcb2());
    const double Mdivgrad_c = mobility(hh) * d2f_dc2 * (dphi_dx_i * dphi_dx_j 
                                                        + dphi_dy_i * dphi_dy_j 
                                                        + dphi_dz_i * dphi_dz_j);
    const double ut = phi_i * phi_j / dt_;

    return ut + t_theta_ * Mdivgrad_c;
  }

  INI_FUNC(init_eta)
  {
    const double sqrt2 = std::sqrt(2.);
    const double sqrtw = std::sqrt(w);
    
    // this is l = xi * sqrt(2 * k_eta_) / sqrtw with xi = 1 instead of 4
    const double l = std::sqrt(2 * k_eta) / sqrtw;
    return 0.5 * (1. - tanh((x - 30.) / (l * sqrt2)));
  }

  INI_FUNC(init_c)
  {
    const double eta = init_eta(x, y, z, eqn_id, lid);
    const double hh = parabolicenergy::h(&eta);
    return parabolicenergy::c1 * hh + parabolicenergy::c2 * (1. - hh);
  }

}  // namespace tonks1

    
}  // namespace cases


#endif  // ifndef CASES_HPP

