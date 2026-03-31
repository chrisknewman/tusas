//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////


#ifndef TOOLS_HPP
#define TOOLS_HPP


namespace tools
{


namespace solvers
{


  double kks_tol = 1e-10;
  int kks_max_iter = 20;

  PARAM_FUNC(param)
  {
    kks_tol = plist->get<double>("kks_tol", kks_tol); 
    kks_max_iter = plist->get<int>("kks_max_iter", kks_max_iter);
  }

  KOKKOS_INLINE_FUNCTION
  const int solve_kks(const double &c1,  // in: c1
                      const double &hh,  // in: h(eta)
                      double &c1a,  // out: c1a, in, initial guess
                      double &c1b,  // out: c1b, in: initial guess 
                      const double DFA_DC1A(const double c1a),  // in: fa'(c1a)
                      const double DFB_DC1B(const double c1b),  // in: fb'(c1b)
                      const double D2FA_DC1A2(),  // in: fa''() [constant for now]
                      const double D2FB_DC1B2(),  // in: fb''() [constant for now]
                      const double &T = 0.)  // in: time
  {
    /*
     * here, we are using notation similar to that in
     * Tonks [10.1016/j.commatsci.2023.112375]:
     *   a = phase a, corresponds to eta
     *   b = phase b, corresponds to (1 - eta)
     *   fa = free energy in phase a
     *   fb = free energy in phase b
     *   c1 = molar fraction of component 1
     *   c1a = subset of c1 contained in phase a,
     *         corresponds to h
     *   c1b = subset of c1 contained in phase b,
     *         corresponds to (1 - h)
     * as implied above, note that
     *   c1 = c1a * h + c1b * (1 - h)
     */

    // meta variables
    double err2 = 0.;
    double delta_c1b = 0.;
    double delta_c1a = 0.;
    const int max_iter = kks_max_iter;
    const double tol = kks_tol;

    // initial guess for c1a and c1b
    c1a = hh * c1a;
    c1b = (1 - hh) * c1b;

    // terms for the kks solve
    double d2fa_dc1a2 = (*D2FA_DC1A2)();
    double d2fb_dc1b2 = (*D2FB_DC1B2)();
    double f1 = hh * c1a + (1 - hh) * c1b - c1;
    double f2 = (*DFA_DC1A)(c1a) - (*DFB_DC1B)(c1b);

    /* 
     * newton iteration loop
     * the function we are finding the roots (ca, cb) of is
     *   F = [ hh * c1a - (1 - hh) * c1b - c1,
     *         dfa_dc1a(c1a) - dfb_dc1b(c1b) ]
     * the jacobian in this case is
     *   J = [[ hh,          (1 - hh) ],
     *        [ d2fa_dc1a2,  -d2fb_dc1b2 ]]
     * so, the inverse is
     *   J^-1 = [[ -d2fb_dc1b2,  -(1 - hh) ],
     *           [ d2fa_dc1a2,   hh ]] / det(J)
     * then,
     *   delta = -J^-1 @ F 
     */
    for (int i = 0; i < max_iter; ++i) {
      // det(J)
      const double detjac = -hh * d2fb_dc1b2 - (1 - hh) * d2fa_dc1a2;

      // -J^-1 @ F
      delta_c1a = -(-d2fb_dc1b2 * f1 - (1 - hh) * f2) / detjac;
      delta_c1b = -(-d2fa_dc1a2 * f1 + hh * f2) / detjac;

      // new value for (c1a, c1b)
      c1a += delta_c1a;
      c1b += delta_c1b;

      // recalculate subset of terms for next iteration
      f1 = hh * c1a + (1 - hh) * c1b - c1;
      f2 = (*DFA_DC1A)(c1a) - (*DFB_DC1B)(c1b);

      // check error and return if done
      err2 = f1 * f1 + f2 * f2;
      if (err2 < tol * tol) return 0;

      // recalculate remaining terms for next iteration
      d2fa_dc1a2 = (*D2FA_DC1A2)();
      d2fb_dc1b2 = (*D2FB_DC1B2)();
    }

    // max iters exceeded
    std::cout << "#### solve_kks() failed to converge!" << std::endl
              << "#### current error = " << std::sqrt(err2) << std::endl
              << "#### tol = " << tol << std::endl;
    exit(-1);
  }


}  // namespace solvers


namespace utils
{


  KOKKOS_INLINE_FUNCTION
  int idx(const int i, const int j, const int ncols) {
    // row-major ordering
    return i * ncols + j;
  }

  KOKKOS_INLINE_FUNCTION
  void get_uu(double* uu,  // out: the array to populate
              const int N_UU,  // in: number of uu to get
              const int ncols,  // in: number of "columns" in uu
              const int first_idx,  // in: index to start at
              GPUBasis* basis[]) {  // in: basis
      for (int k = 0 ; k < N_UU; ++k) {
      uu[idx(0, k, ncols)] = basis[k + first_idx]->uu();
      uu[idx(1, k, ncols)] = basis[k + first_idx]->uuold();
      uu[idx(2, k, ncols)] = basis[k + first_idx]->uuoldold();
    }
  }

  KOKKOS_INLINE_FUNCTION
  void get_graduu(double* duu_dx,  // out: dx array to populate
                  double* duu_dy,  // out: dy array to populate
                  double* duu_dz,  // out: dz array to populate
                  const int N_UU,  // in: number of uu to get
                  const int ncols,  // in: number of "columns" in uu
                  const int first_idx,  // in: index to start at
                  GPUBasis* basis[]) {  // in: basis
    for (int k = 0 ; k < N_UU; ++k) {
      duu_dx[idx(0, k, ncols)] = basis[k + first_idx]->duudx();
      duu_dy[idx(0, k, ncols)] = basis[k + first_idx]->duudy();
      duu_dz[idx(0, k, ncols)] = basis[k + first_idx]->duudz();

      duu_dx[idx(1, k, ncols)] = basis[k + first_idx]->duuolddx();
      duu_dy[idx(1, k, ncols)] = basis[k + first_idx]->duuolddy();
      duu_dz[idx(1, k, ncols)] = basis[k + first_idx]->duuolddz();

      duu_dx[idx(2, k, ncols)] = basis[k + first_idx]->duuoldolddx();
      duu_dy[idx(2, k, ncols)] = basis[k + first_idx]->duuoldolddy();
      duu_dz[idx(2, k, ncols)] = basis[k + first_idx]->duuoldolddz();
    }
  }

  KOKKOS_INLINE_FUNCTION
  const double ret_value(const double ut,  // in: time derivative
                         const double* f,  // in: residual
                         const double dt,  // in: current dt
                         const double dtold,  // in: previous dt
                         const double t_theta,  // in: implicit/explicit weight
                         const double t_theta2) {  // in: adaptive time-step control
    return ut + (1. - t_theta2) * t_theta * f[0]
             + (1. - t_theta2) * (1. - t_theta) * f[1]
             + .5 * t_theta2 * ((2. + dt / dtold) * f[1] - dt / dtold * f[2]);
  } 

  KOKKOS_INLINE_FUNCTION
  const double ret_value(const double ut,  // in: time derivative
                         const double* f,  // in: residual
                         const double t_theta) {  //: in implicit/explicit weight
    return ut + t_theta * f[0] + (1. - t_theta) * f[1];
  }


}  // namespace utils

    
}  // namespace tools


#endif  // ifndef TOOLS_HPP

