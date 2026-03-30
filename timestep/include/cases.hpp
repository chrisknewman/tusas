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
  TUSAS_DEVICE double Aa_ = 2.;
  TUSAS_DEVICE double Ab_ = 2.;
  TUSAS_DEVICE double c1_ = 0.3;
  TUSAS_DEVICE double c2_ = 0.7;
  TUSAS_DEVICE double delta_c1_ = 0.;
  TUSAS_DEVICE double delta_c2_ = 0.;
  TUSAS_DEVICE double f1_ = 0.;
  TUSAS_DEVICE double f2_ = 0.;
  TUSAS_DEVICE const double alpha_ = 5.;  // pfhub2 coef in g
  
  TUSAS_DEVICE const int N_ETA_MAX = 4;
  TUSAS_DEVICE int N_ETA_ = 1;

  PARAM_FUNC(param_)
  {
    int N_p = plist->get<int>("N_ETA", N_ETA_);
#ifdef TUSAS_HAVE_CUDA
    cudaMemcpyToSymbol(N_ETA_, &N_p, sizeof(int));
#else
    N_ETA_ = N_p;
#endif
    if(N_ETA_ > N_ETA_MAX) exit(0);

    // parameters from Sheng 2022, eqns 8, 9
    Aa_ = plist->get<double>("Aa", Aa_);
    Ab_ = plist->get<double>("Ab", Ab_);
    c1_ = plist->get<double>("c1", c1_);
    c2_ = plist->get<double>("c2", c2_);
    delta_c1_ = plist->get<double>("delta_c1", delta_c1_);
    delta_c2_ = plist->get<double>("delta_c2", delta_c2_);
    f1_ = plist->get<double>("f1", f1_);
    f2_ = plist->get<double>("f2", f2_);
  }

  KOKKOS_INLINE_FUNCTION
  const double fa(const double ca)
  {
    // f_alpha(c_alpha) from Sheng 2022, eqn 8
    // altered to be more general as eqn 9
    return Aa_ * (ca - (c1_ + delta_c1_))
               * (ca - (c1_ + delta_c1_)) + f1_;
  }

  KOKKOS_INLINE_FUNCTION
  const double fb(const double cb)
  {
    // f_beta(c_beta) from Sheng 2022, eqn 9
    return Ab_ * (cb - (c2_ + delta_c2_))
               * (cb - (c2_ + delta_c2_)) + f2_;
  }

  KOKKOS_INLINE_FUNCTION
  const double dfa_dca(const double ca)
  {
    return 2 * Aa_ * (ca - (c1_ + delta_c1_));
  }

  KOKKOS_INLINE_FUNCTION
  const double dfb_dcb(const double cb)
  {
    return 2 * Ab_ * (cb - (c2_ + delta_c2_));
  }

  KOKKOS_INLINE_FUNCTION
  const double d2fa_dca2()
  {
    return 2 * Aa_;
  }

  KOKKOS_INLINE_FUNCTION
  const double d2fb_dcb2()
  {
    return 2 * Ab_;
  }

  KOKKOS_INLINE_FUNCTION 
  const double h(const double *eta)
  {
    double val = 0.;
    for (int i = 0; i < N_ETA_; i++) {
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
    for (int i = 0; i < N_ETA_; i++) {
      aval += eta[i] * eta[i];
    }
    aval = aval - eta[eqn_id]*eta[eqn_id];
    return 2. * eta[eqn_id] * (1. - eta[eqn_id]) * (1. - eta[eqn_id])  
             - 2. * eta[eqn_id] * eta[eqn_id] * (1. - eta[eqn_id])
             + 4. * alpha_ * eta[eqn_id] * aval;
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
 
  TUSAS_DEVICE const int Nt_MAX_ = 3;
  
  TUSAS_DEVICE const int N_C_MAX_ = 2;
  TUSAS_DEVICE int N_C_ = 1;
  TUSAS_DEVICE int c_start_idx_ = 0;

  TUSAS_DEVICE const int N_MU_MAX_ = 2;
  TUSAS_DEVICE int N_MU_ = 1;
  TUSAS_DEVICE int mu_start_idx_ = 1;

  TUSAS_DEVICE const int N_ETA_MAX_ = 4;
  TUSAS_DEVICE int N_ETA_ = 1;
  TUSAS_DEVICE int eta_start_idx_ = 1;
  
  TUSAS_DEVICE int eqn_off_ = 1;
  TUSAS_DEVICE const int eqn_off_split_ = 2;

  TUSAS_DEVICE int ci_ = 0;
  TUSAS_DEVICE int mui_ = 1;
  
  TUSAS_DEVICE double t0_ = 1.;
  TUSAS_DEVICE double x0_ = 1.;
  TUSAS_DEVICE double f0_ = 1.;
  TUSAS_DEVICE double k_c_ = 0.;
  TUSAS_DEVICE double k_eta_ = 1.5;
  TUSAS_DEVICE double M_ = 10.;
  TUSAS_DEVICE double L_ = 2.;
  TUSAS_DEVICE double w_ = 12.;

  PARAM_FUNC(param_)
  {
    int N_p = plist->get<int>("N_ETA", N_ETA_);
#ifdef TUSAS_HAVE_CUDA
    cudaMemcpyToSymbol(N_ETA_, &N_p, sizeof(int));
#else
    N_ETA_ = N_p;
#endif
    int eqn_off_p = plist->get<int>("OFFSET", eqn_off_);
#ifdef TUSAS_HAVE_CUDA
    cudaMemcpyToSymbol(eqn_off_, &eqn_off_p, sizeof(int));
#else
    eqn_off_ = eqn_off_p;
#endif
    if(N_ETA_ > N_ETA_MAX_) exit(0);

    // nondim free energy density, J/m^3
    // generally, should be ~parabolicenergy::Aa_
    f0_ = plist->get<double>("f0", f0_);
    // nondim spatial scaling, m
    x0_ = plist->get<double>("x0", x0_);
    // nondim temporal scaling, s
    t0_ = plist->get<double>("t0", t0_);

    k_c_ = plist->get<double>("k_c", k_c_);
    k_eta_ = plist->get<double>("k_eta", k_eta_);
    M_ = plist->get<double>("M", M_);
    L_ = plist->get<double>("L", L_);
    w_ = plist->get<double>("w", w_);

    // set params for free energy density
    parabolicenergy::param_(plist);

    // nondimensionalize
    // note that if x0_, t0_, and f0_ are not
    // set by the user in an input file, these
    // values default to 1 and the original
    // dimensional equations are preserved
    parabolicenergy::Aa_ = parabolicenergy::Aa_ / f0_;
    parabolicenergy::Ab_ = parabolicenergy::Ab_ / f0_;
    parabolicenergy::f1_ = parabolicenergy::f1_ / f0_;
    parabolicenergy::f2_ = parabolicenergy::f2_ / f0_;
    k_c_ = k_c_ / x0_ / x0_ / f0_;
    k_eta_ = k_eta_ / x0_ / x0_ / f0_;
    M_ = M_ * t0_ * f0_ / x0_ / x0_;
    L_ = L_ * t0_ * f0_;
    w_ = w_ / f0_;
  }

  KOKKOS_INLINE_FUNCTION
  const double mobility(const double hh) {
    return M_ * (1. - hh) + hh;
  } 

  KOKKOS_INLINE_FUNCTION 
  RES_FUNC_TPETRA(residual_eta)
  {
    const int Nt = 3;

    const int local_id = eqn_id - eta_start_idx_;

    const double phi = basis[0]->phi(i);
    const double dphi_dx = basis[0]->dphidx(i);
    const double dphi_dy = basis[0]->dphidy(i);
    const double dphi_dz = basis[0]->dphidz(i);

    double c[Nt_MAX_ * N_C_MAX_];
    double dc_dx[Nt_MAX_ * N_C_MAX_];
    double dc_dy[Nt_MAX_ * N_C_MAX_];
    double dc_dz[Nt_MAX_ * N_C_MAX_];
    tools::utils::get_uu(c, N_C_, N_C_MAX_, c_start_idx_, basis);
    tools::utils::get_graduu(dc_dx, dc_dy, dc_dz, N_C_, N_C_MAX_, c_start_idx_, basis);

    double eta[Nt_MAX_ * N_ETA_MAX_];
    double deta_dx[Nt_MAX_ * N_ETA_MAX_];
    double deta_dy[Nt_MAX_ * N_ETA_MAX_];
    double deta_dz[Nt_MAX_ * N_ETA_MAX_];
    tools::utils::get_uu(eta, N_ETA_, N_ETA_MAX_, eta_start_idx_, basis);
    tools::utils::get_graduu(deta_dx, deta_dy, deta_dz, N_ETA_, N_ETA_MAX_, eta_start_idx_, basis);

    double hh[Nt_MAX_];
    double ca[Nt_MAX_];
    double cb[Nt_MAX_];
    double k_divgrad_eta[Nt_MAX_];
    double df_deta[Nt_MAX_];
    double f[Nt_MAX_];

    int idx = 0;
    for (int tdx = 0; tdx < Nt; ++tdx) {
      hh[tdx] = parabolicenergy::h(&eta[tdx * N_ETA_MAX_]);
      
      ca[tdx] = parabolicenergy::c1_;
      cb[tdx] = parabolicenergy::c2_;
      idx = tools::utils::idx(tdx, local_id, N_C_MAX_);
      tools::solvers::solve_kks(c[idx], hh[tdx], ca[tdx], cb[tdx],
                                parabolicenergy::dfa_dca,
                                parabolicenergy::dfb_dcb,
                                parabolicenergy::d2fa_dca2,
                                parabolicenergy::d2fb_dcb2);

      idx = tools::utils::idx(tdx, local_id, N_ETA_MAX_);
      df_deta[tdx] = (parabolicenergy::df_deta(ca[tdx], cb[tdx], eta[idx])
                        + w_ * parabolicenergy::dg_deta(&eta[tdx * N_ETA_MAX_], local_id)) * phi;
      k_divgrad_eta[tdx] = k_eta_ * (deta_dx[idx] * dphi_dx + deta_dy[idx] * dphi_dy + deta_dz[idx] * dphi_dz);

      f[tdx] = L_* (k_divgrad_eta[tdx] + df_deta[tdx]);
    }

    const double deta_dt = (eta[tools::utils::idx(0, local_id, N_ETA_MAX_)] 
                              - eta[tools::utils::idx(1, local_id, N_ETA_MAX_)]) / dt_ * phi;

    return tools::utils::ret_value(deta_dt, f, dt_, dtold_, t_theta_, t_theta2_);

    /*// test function
    const double test = basis[0]->phi(i);
    // u, phi
    const double c[3] = {basis[ci_]->uu(), basis[ci_]->uuold(), basis[ci_]->uuoldold()};
    const double eta[3] = {basis[eqn_id]->uu(), basis[eqn_id]->uuold(), basis[eqn_id]->uuoldold()};
    const double divgradeta[3] = {L_ * k_eta_ * (basis[eqn_id]->duudx() * basis[0]->dphidx(i)
                                    + basis[eqn_id]->duudy() * basis[0]->dphidy(i)
                                    + basis[eqn_id]->duudz() * basis[0]->dphidz(i)),
                                  L_ * k_eta_ * (basis[eqn_id]->duuolddx() * basis[0]->dphidx(i)
                                    + basis[eqn_id]->duuolddy() * basis[0]->dphidy(i)
                                    + basis[eqn_id]->duuolddz() * basis[0]->dphidz(i)),
                                  L_ * k_eta_ * (basis[eqn_id]->duuoldolddx() * basis[0]->dphidx(i)
                                    + basis[eqn_id]->duuoldolddy() * basis[0]->dphidy(i)
                                    + basis[eqn_id]->duuoldolddz() * basis[0]->dphidz(i))};

    double eta_array[N_ETA_MAX_];
    double eta_array_old[N_ETA_MAX_];
    double eta_array_oldold[N_ETA_MAX_];
    for( int kk = 0; kk < N_ETA_; kk++){
      int kk_off = kk + eqn_off_;
      eta_array[kk] = basis[kk_off]->uu();
      eta_array_old[kk] = basis[kk_off]->uuold();
      eta_array_oldold[kk] = basis[kk_off]->uuoldold();
    }

    const double hh[3] = {parabolicenergy::h(eta_array),
                          parabolicenergy::h(eta_array_old),
                          parabolicenergy::h(eta_array_oldold)};
    double ca[3] = {parabolicenergy::c1_, parabolicenergy::c1_, parabolicenergy::c1_};
    double cb[3] = {parabolicenergy::c2_, parabolicenergy::c2_, parabolicenergy::c2_};
    tools::solvers::solve_kks(c[0], hh[0], ca[0], cb[0],
                   parabolicenergy::dfa_dca,
                   parabolicenergy::dfb_dcb,
                   parabolicenergy::d2fa_dca2,
                   parabolicenergy::d2fb_dcb2);
    tools::solvers::solve_kks(c[1], hh[1], ca[1], cb[1],
                   parabolicenergy::dfa_dca,
                   parabolicenergy::dfb_dcb,
                   parabolicenergy::d2fa_dca2,
                   parabolicenergy::d2fb_dcb2);
    tools::solvers::solve_kks(c[2], hh[2], ca[2], cb[2],
                   parabolicenergy::dfa_dca,
                   parabolicenergy::dfb_dcb,
                   parabolicenergy::d2fa_dca2,
                   parabolicenergy::d2fb_dcb2);

    const int k = eqn_id - eqn_off_;
    const double df_deta[3] = {L_ * (parabolicenergy::df_deta(ca[0], cb[0], eta[0])
                                 + w_ * parabolicenergy::dg_deta(eta_array, k)) * test,
                               L_ * (parabolicenergy::df_deta(ca[1], cb[1], eta[1])
                                 + w_ * parabolicenergy::dg_deta(eta_array_old, k)) * test,
                               L_ * (parabolicenergy::df_deta(ca[2], cb[2], eta[2])
                                 + w_ * parabolicenergy::dg_deta(eta_array_oldold, k)) * test};
    
    const double f[3] = {df_deta[0] + divgradeta[0],
                         df_deta[1] + divgradeta[1],
                         df_deta[2] + divgradeta[2]};

    const double ut = (eta[0] - eta[1]) / dt_ * test;

    return ut + (1. - t_theta2_) * t_theta_ * f[0]
             + (1. - t_theta2_) * (1. - t_theta_) * f[1]
             + .5 * t_theta2_ * ((2. + dt_ / dtold_) * f[1] - dt_ / dtold_ * f[2]);*/
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
    //   utils::idx(time_idx, c_idx, N_C_MAX_)
    double c[Nt_MAX_ * N_C_MAX_] = {0.};
    double dc_dx[Nt_MAX_ * N_C_MAX_] = {0.};
    double dc_dy[Nt_MAX_ * N_C_MAX_] = {0.};
    double dc_dz[Nt_MAX_ * N_C_MAX_] = {0.};
    tools::utils::get_uu(c, N_C_, N_C_MAX_, c_start_idx_, basis);
    tools::utils::get_graduu(dc_dx, dc_dy, dc_dz, N_C_, N_C_MAX_, c_start_idx_, basis);

    // populate eta viewed as a "matrix"
    //   eta[time_idx, eta_idx]
    // but really a 1D array that
    // we can index this using
    //   utils::idx(time_idx, eta_idx, N_ETA_MAX_)
    double eta[Nt_MAX_ * N_ETA_MAX_] = {0.};
    double deta_dx[Nt_MAX_ * N_ETA_MAX_] = {0.};
    double deta_dy[Nt_MAX_ * N_ETA_MAX_] = {0.};
    double deta_dz[Nt_MAX_ * N_ETA_MAX_] = {0.};
    tools::utils::get_uu(eta, N_ETA_, N_ETA_MAX_, eta_start_idx_, basis);
    tools::utils::get_graduu(deta_dx, deta_dy, deta_dz, N_ETA_, N_ETA_MAX_, eta_start_idx_, basis);

    // define all the variables we need to calculate 
    // the residual = Mdivgrad_df_dc
    double hh[Nt_MAX_] = {0.};
    double dh_dx[Nt_MAX_] = {0.};
    double dh_dy[Nt_MAX_] = {0.};
    double dh_dz[Nt_MAX_] = {0.};
    double ca[Nt_MAX_] = {0.};
    double cb[Nt_MAX_] = {0.};
    double d2f_dc2[Nt_MAX_] = {0.};
    double d2f_dcdx[Nt_MAX_] = {0.};
    double d2f_dcdy[Nt_MAX_] = {0.};
    double d2f_dcdz[Nt_MAX_] = {0.};
    double Mdivgrad_df_dc[Nt_MAX_] = {0.};


    // loop over each time level that we need data at
    int idx = 0;
    double dh_deta = 0;
    for (int tdx = 0; tdx < Nt; ++tdx) {
      // calculate h
      hh[tdx] = parabolicenergy::h(&eta[tdx * N_ETA_MAX_]);

      // calculate grad h
      dh_dx[tdx] = 0.;
      dh_dy[tdx] = 0.;
      dh_dz[tdx] = 0.;
      for (int k = 0; k < N_ETA_ ; ++k) {
        idx = tools::utils::idx(tdx, k, N_ETA_MAX_);
        dh_deta = parabolicenergy::dh_deta(eta[idx]);

        dh_dx[tdx] += dh_deta * deta_dx[idx];
        dh_dy[tdx] += dh_deta * deta_dy[idx];
        dh_dz[tdx] += dh_deta * deta_dz[idx];
      }

      // do the kks solve to get ca and cb
      // for the current component c_{eqn_id}
      ca[tdx] = parabolicenergy::c1_;
      cb[tdx] = parabolicenergy::c2_;
      tools::solvers::solve_kks(c[tools::utils::idx(tdx, eqn_id, N_C_MAX_)],
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
      idx = tools::utils::idx(tdx, eqn_id, N_C_MAX_);
      d2f_dcdx[tdx] = d2f_dc2[tdx] * (cb[tdx] - ca[tdx]) * dh_dx[tdx] + d2f_dc2[tdx] * dc_dx[idx];
      d2f_dcdy[tdx] = d2f_dc2[tdx] * (cb[tdx] - ca[tdx]) * dh_dy[tdx] + d2f_dc2[tdx] * dc_dy[idx];
      d2f_dcdz[tdx] = d2f_dc2[tdx] * (cb[tdx] - ca[tdx]) * dh_dz[tdx] + d2f_dc2[tdx] * dc_dz[idx];

      // finally, calculate M * div(grad(f_c))
      Mdivgrad_df_dc[tdx] = mobility(hh[tdx]) * (d2f_dcdx[tdx] * dphi_dx
                                                 + d2f_dcdy[tdx] * dphi_dy
                                                 + d2f_dcdz[tdx] * dphi_dz);
    }  // tdx = 0, < Nt loop

    const double dc_dt = (c[tools::utils::idx(0, eqn_id, N_C_MAX_)] 
                            - c[tools::utils::idx(1, eqn_id, N_C_MAX_)]) / dt_ * phi;

    return tools::utils::ret_value(dc_dt, Mdivgrad_df_dc, dt_, dtold_, t_theta_, t_theta2_);
  }
  TUSAS_DEVICE RES_FUNC_TPETRA((*residual_c_dp)) = residual_c;
  
  KOKKOS_INLINE_FUNCTION 
  PRE_FUNC_TPETRA(prec_eta_)
  {
    const double ut = basis[0]->phi(j) / dt_ * basis[0]->phi(i);
    const double divgrad = L_ * k_eta_ * (basis[0]->dphidx(j) * basis[0]->dphidx(i)
                           + basis[0]->dphidy(j) * basis[0]->dphidy(i)
                           + basis[0]->dphidz(j) * basis[0]->dphidz(i));
    return ut + t_theta_ * divgrad;
  }

  KOKKOS_INLINE_FUNCTION 
  PRE_FUNC_TPETRA(prec_c_)
  {
    const double test = basis[0]->phi(i);
    const double u_t = test * basis[0]->phi(j) / dt_;
    
    double eta_array[N_ETA_MAX_];
    for(int kk = 0; kk < N_ETA_; kk++){
      int kk_off = kk + eqn_off_;
      eta_array[kk] = basis[kk_off]->uu();
    }
    
    const double hh = parabolicenergy::h(eta_array);
    // note that we can skip the kks solve here only
    // because the free energy is parabolic -- the
    // second derivatives are constant
    const double d2f_dc2 = parabolicenergy::d2fa_dca2() * parabolicenergy::d2fb_dcb2() 
                             / ((1 - hh) * parabolicenergy::d2fa_dca2() + hh * parabolicenergy::d2fb_dcb2());
    const double divgradc = mobility(hh) * d2f_dc2 * (basis[0]->dphidx(j) * basis[0]->dphidx(i)
                             + basis[0]->dphidy(j) * basis[0]->dphidy(i)
                             + basis[0]->dphidz(j) * basis[0]->dphidz(i));
    return u_t + t_theta_ * divgradc;
  }

  INI_FUNC(init_eta_)
  {
    std::cout << "init eta" << std::endl;
    const double sqrt2 = std::sqrt(2.);
    const double sqrtw = std::sqrt(w_);
    
    // this is l = xi * sqrt(2 * k_eta_) / sqrtw with xi = 1 instead of 4
    const double l = std::sqrt(2 * k_eta_) / sqrtw;
    return 0.5 * (1. - tanh((x - 30.) / (l * sqrt2)));
    //return x < 30 ? 1 : 0;
  }

  INI_FUNC(init_c_)
  {
    const double eta = init_eta_(x, y, z, eqn_id, lid);
    const double hh = parabolicenergy::h(&eta);
    return parabolicenergy::c1_ * hh + parabolicenergy::c2_ * (1. - hh);
  }

}  // namespace tonks1

    
}  // namespace cases


#endif  // ifndef CASES_HPP

