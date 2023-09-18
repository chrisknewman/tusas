//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef FUNCTION_DEF_HPP
#define FUNCTION_DEF_HPP
#if 0
#include <boost/ptr_container/ptr_vector.hpp>
#endif

#include "basis.hpp"
	
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


/** Definition for initialization function. Each initialization function is called at each node for each equation at the beginning of the simualtaion with this signature:
- NAME:     name of function to call
- const double &x: the x-ccordinate of the node
- const double &y: the y-ccordinate of the node
- const double &z: the z-ccordinate of the node
- const int &eqn_id: the index of the current equation

*/

#define INI_FUNC(NAME)  double NAME(const double &x,\
			            const double &y,\
			            const double &z,\
				    const int &eqn_id) 


/** Definition for Dirichlet function. Each Dirichlet function is called at each node for each equation with this signature:
- NAME:     name of function to call
- const double &x: the x-ccordinate of the node
- const double &y: the y-ccordinate of the node
- const double &z: the z-ccordinate of the node
- const int &eqn_id: the index of the current equation
- const double &t: the current time

*/

#define DBC_FUNC(NAME)  double NAME(const double &x,\
			            const double &y,\
			            const double &z,\
			            const double &t) 

/** Definition for Neumann function. Each Neumann function is called at each Gauss point for the current equation with this signature:
- NAME:     name of function to call
- const Basis *basis:     basis function object for current equation
- const int &i:    the current basis function (row in residual vector)
- const double &dt_: the timestep size as prescribed in input file						
- const double &t_theta_: the timestep parameter as prescribed in input file
- const double &time: the current simulation time


*/

#define NBC_FUNC(NAME)  double NAME(const Basis *basis,\
				    const int &i,\
				    const double &dt_,\
				    const double &t_theta_,\
				    const double &time)

/** Definition for post-process function. Each post-process function is called at each node for each equation at the end of each timestep with this signature:
- NAME:     name of function to call
- const double *u: an array of solution values indexed by equation
- const double *gradu: an array of gradient values indexed by equation, coordinates (NULL unless error estimation is activated)
- const double *xyz: an array of coordinates indexed by equation, coordinates
- const double &time: the current simulation time

*/

#define PPR_FUNC(NAME)  double NAME(const double *u,\
				    const double *uold,\
				    const double *uoldold,\
				    const double *gradu,\
				    const double *xyz,\
				    const double &time,\
				    const double &dt,\
				    const double &dtold,\
				    const int &eqn_id)

/** Parameter function to propogate information from input file. Each parameter function is called at the beginning of each simulation.
- NAME:     name of function to call
- Teuchos::ParameterList *plist: paramterlist containing information defined in input file

*/

#define PARAM_FUNC(NAME) void NAME(Teuchos::ParameterList *plist) 

namespace timeadapt
{
PPR_FUNC(d2udt2_)
{
  const double uu = u[eqn_id];
  const double uuold = uold[eqn_id];
  const double uuoldold = uoldold[eqn_id];

  const double duuold = (uuold-uuoldold)/dtold;
  const double duu = (uu-uuold)/dt;
  const double d2udt2 = (duu-duuold)/dt;
  //const double ae = .5*dt*dt*d2udt2;

//   double r = 0;
//   r = abs(d2udt2/1.);
  //if (uu*uu > 1.e-8) r = abs(d2udt2/uu);
  //r = std::max(abs(d2udt2/uu),1.e-10);
  //return sqrt(2.*tol*abs(uu)/abs(d2udt2));
  //return abs(d2udt2/uu);
  //std::cout<<r<<" "<<uu*uu<<" "<<d2udt2<<std::endl;
  return .5*dt*dt*d2udt2;
}
PPR_FUNC(predictor_fe_)
{
  const double uu = u[eqn_id];
  //const double uuold = uold[eqn_id];
  //const double uuoldold = uoldold[eqn_id];
  const double uupred = gradu[eqn_id];//hack for now
  //std::cout<<eqn_id<<" "<<uold[eqn_id]<<std::endl;
  //std::cout<<eqn_id<<" "<<uu<<"  "<<uupred<<"  "<<uu - uupred<<std::endl;
  return (uu - uupred);
}
PPR_FUNC(postproc1_)
{
  const double uu = u[eqn_id];

  const double x = xyz[0];
  const double y = xyz[1];

  const double pi = 3.141592653589793;
  //d2udt2 = 4 E^(-2 \[Pi]^2 t) \[Pi]^4 Sin[\[Pi] x] Sin[\[Pi] y];

  const double uex = exp(-2.*pi*pi*time)*sin(pi*x)*sin(pi*y);
  //const double d2udt2ex = 4.*pi*pi*pi*pi*exp(-2.*pi*pi*time)*sin(pi*x)*sin(pi*y);
  //return sqrt(2.*tol*abs(uu)/abs(d2udt2));
  //return d2udt2;
  
  //return abs(d2udt2ex/uex);
  //return abs(d2udt2ex/1.);
  return uu;
}
PPR_FUNC(postproc2_)
{
  //const double uu = u[eqn_id];

  const double x = xyz[0];
  const double y = xyz[1];

  const double pi = 3.141592653589793;
  //d2udt2 = 4 E^(-2 \[Pi]^2 t) \[Pi]^4 Sin[\[Pi] x] Sin[\[Pi] y];

  const double uex = exp(-2.*pi*pi*time)*sin(pi*x)*sin(pi*y);
  //const double d2udt2ex = 4.*pi*pi*pi*pi*exp(-2.*pi*pi*time)*sin(pi*x)*sin(pi*y);
  //return sqrt(2.*tol*abs(uu)/abs(d2udt2));
  //return d2udt2;
  
  //return abs(d2udt2ex/uex);
  //return abs(d2udt2ex/1.);
  const double uuoldold = gradu[eqn_id];//hack for now
  return uuoldold;
}
PPR_FUNC(normu_)
{
  const double uu = u[eqn_id];

  return uu;
}
}//namespace timeadapt


// #define RES_FUNC_TPETRA(NAME)  double NAME(const GPUBasis * const * basis, 

#ifdef TUSAS3D
#define RES_FUNC_TPETRA(NAME)  double NAME(GPUBasisLHex * basis,\
                                    const int &i,\
                                    const double &dt_,\
                                    const double &dtold_,\
			            const double &t_theta_,\
			            const double &t_theta2_,\
                                    const double &time,\
				    const int &eqn_id,\
				    const double &vol,\
				    const double &rand)
#else
#define RES_FUNC_TPETRA(NAME)  double NAME(GPUBasisLQuad * basis,\
                                    const int &i,\
                                    const double &dt_,\
                                    const double &dtold_,\
			            const double &t_theta_,\
			            const double &t_theta2_,\
                                    const double &time,\
				    const int &eqn_id,\
				    const double &vol,\
				    const double &rand)
#endif
#ifdef TUSAS3D
#define PRE_FUNC_TPETRA(NAME)  double NAME(const GPUBasisLHex *basis, \
                                    const int &i,\
				    const int &j,\
				    const double &dt_,\
				    const double &t_theta_,\
				    const int &eqn_id)
#else
#define PRE_FUNC_TPETRA(NAME)  double NAME(const GPUBasisLQuad *basis, \
                                    const int &i,\
				    const int &j,\
				    const double &dt_,\
				    const double &t_theta_,\
				    const int &eqn_id)
#endif

#define NBC_FUNC_TPETRA(NAME)  double NAME(const GPUBasis *basis,\
				    const int &i,\
				    const double &dt_,\
                                    const double &dtold_,\
				    const double &t_theta_,\
			            const double &t_theta2_,\
				    const double &time)

typedef double (*RESFUNC1)(const GPUBasisLHex *basis,
				    const int &i,
				    const double &dt_,
                                    const double &dtold_,
				    const double &t_theta_,
			            const double &t_theta2_,
				    const double &time,
				    const int &eqn_id);


namespace tpetra{//we can just put the KOKKOS... around the other dbc_zero_ later...

namespace heat{
TUSAS_DEVICE
double k_d = 1.;
TUSAS_DEVICE
double rho_d = 1.;
TUSAS_DEVICE
double cp_d = 1.;

double k_h = 1.;
double rho_h = 1.;
double cp_h = 1.;

KOKKOS_INLINE_FUNCTION 
DBC_FUNC(dbc_zero_) 
{
  return 0.;
}

  //KOKKOS_INLINE_FUNCTION 
INI_FUNC(init_heat_test_)
{

  const double pi = 3.141592653589793;

  return sin(pi*x)*sin(pi*y);
}

KOKKOS_INLINE_FUNCTION 
  //KOKKOS_FUNCTION 
//TUSAS_DEVICE
RES_FUNC_TPETRA(residual_heat_test_)
{
  //printf("here\n");
  //printf("%lf %lf %lf\n",rho_d,cp_d,k_d);
   const double ut = rho_d*cp_d*(basis[eqn_id].uu()-basis[eqn_id].uuold())/dt_*basis[eqn_id].phi(i);
   const double f[3] = {k_d*(basis[eqn_id].dudx()*basis[eqn_id].dphidx(i)
			     + basis[eqn_id].dudy()*basis[eqn_id].dphidy(i)
			     + basis[eqn_id].dudz()*basis[eqn_id].dphidz(i)),
			k_d*(basis[eqn_id].duolddx()*basis[eqn_id].dphidx(i)
			     + basis[eqn_id].duolddy()*basis[eqn_id].dphidy(i)
			     + basis[eqn_id].duolddz()*basis[eqn_id].dphidz(i)),
			k_d*(basis[eqn_id].duoldolddx()*basis[eqn_id].dphidx(i)
			     + basis[eqn_id].duoldolddy()*basis[eqn_id].dphidy(i)
			     + basis[eqn_id].duoldolddz()*basis[eqn_id].dphidz(i))};

  return ut + (1.-t_theta2_)*t_theta_*f[0]
    + (1.-t_theta2_)*(1.-t_theta_)*f[1]
    +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
  //return 0.;
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_heat_test_dp_)) = residual_heat_test_;

TUSAS_DEVICE
double residual_heat_test_p(GPUBasisLHex * basis,
                                    const int &i,
                                    const double &dt_,
                                    const double &dtold_,
			            const double &t_theta_,
			            const double &t_theta2_,
                                    const double &time,
				    const int &eqn_id)
{
  //printf("here\n");
   const double ut = rho_d*cp_d*(basis[eqn_id].uu()-basis[eqn_id].uuold())/dt_*basis[eqn_id].phi(i);
   const double f[3] = {k_d*(basis[eqn_id].dudx()*basis[eqn_id].dphidx(i)
			     + basis[eqn_id].dudy()*basis[eqn_id].dphidy(i)
			     + basis[eqn_id].dudz()*basis[eqn_id].dphidz(i)),
			k_d*(basis[eqn_id].duolddx()*basis[eqn_id].dphidx(i)
			     + basis[eqn_id].duolddy()*basis[eqn_id].dphidy(i)
			     + basis[eqn_id].duolddz()*basis[eqn_id].dphidz(i)),
			k_d*(basis[eqn_id].duoldolddx()*basis[eqn_id].dphidx(i)
			     + basis[eqn_id].duoldolddy()*basis[eqn_id].dphidy(i)
			     + basis[eqn_id].duoldolddz()*basis[eqn_id].dphidz(i))};

  return ut + (1.-t_theta2_)*t_theta_*f[0]
    + (1.-t_theta2_)*(1.-t_theta_)*f[1]
    +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
  //return 0.;
}

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_heat_test_)
{
  return rho_d*cp_d*basis[eqn_id].phi(j)/dt_*basis[eqn_id].phi(i)
    + t_theta_*k_d*(basis[eqn_id].dphidx(j)*basis[eqn_id].dphidx(i)
       + basis[eqn_id].dphidy(j)*basis[eqn_id].dphidy(i)
		    + basis[eqn_id].dphidz(j)*basis[eqn_id].dphidz(i));
}

TUSAS_DEVICE
PRE_FUNC_TPETRA((*prec_heat_test_dp_)) = prec_heat_test_;

PARAM_FUNC(param_)
{
  double kk = plist->get<double>("k_",1.);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(k_d,&kk,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(k_d),&kk,sizeof(double),0,hipMemcpyHostToDevice);
#else
  k_d = kk;
#endif
  k_h = kk;

  double rho = plist->get<double>("rho_",1.);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(rho_d,&rho,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(rho_d),&rho,sizeof(double),0,hipMemcpyHostToDevice);
#else
  rho_d = rho;
#endif
  rho_h = rho;

  double cp = plist->get<double>("cp_",1.);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(cp_d,&cp,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(cp_d),&cp,sizeof(double),0,hipMemcpyHostToDevice);
#else
  cp_d = cp;
#endif
  cp_h = cp;
}
//double postproc_c_(const double *u, const double *gradu, const double *xyz, const double &time)
PPR_FUNC(postproc_)
{
  //exact solution is: u[x,y,t]=exp(-2 pi^2 k t)sin(pi x)sin(pi y)
  const double uu = u[0];
  const double x = xyz[0];
  const double y = xyz[1];

  const double pi = 3.141592653589793;

  const double s= exp(-2.*k_h*pi*pi*time)*sin(pi*x)*sin(pi*y);

  return s-uu;
}
}//namespace heat

// the above solution is also a solution to the nonlinear problem:
// u_t - div ( u grad u) + 2 pi^2  (1-u) + u_x^2 + u_y^2
// we replace u_x^2 + u_y^2 with a forcing term f2(x,y,t)

#if 0
KOKKOS_INLINE_FUNCTION 
double f1(const double &u)
{
  const double pi = 3.141592653589793;
  return 2.*pi*pi*u*(1.-u);
}

KOKKOS_INLINE_FUNCTION 
double f2(const double &x, const double &y, const double &t)
{
  const double pi = 3.141592653589793;
  const double pix = pi*x;
  const double piy = pi*y;
  const double pi2 = pi*pi;
  return exp(-4.*pi2*t)*pi2*(cos(piy)*cos(piy)*sin(pix)*sin(pix) + cos(pix)*cos(pix)*sin(piy)*sin(piy));
}

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_nlheatimr_test_)
{
  const double u_m = t_theta_*basis[eqn_id].uu() + (1. - t_theta_)*basis[eqn_id].uuold();
  const double dudx_m = t_theta_*basis[eqn_id].dudx() + (1. - t_theta_)*basis[eqn_id].duolddx();
  const double dudy_m = t_theta_*basis[eqn_id].dudy() + (1. - t_theta_)*basis[eqn_id].duolddy();
  const double dudz_m = t_theta_*basis[eqn_id].dudz() + (1. - t_theta_)*basis[eqn_id].duolddz();
  const double t_m = time + t_theta_*dt_;
  const double x = basis[0].xx();
  const double y = basis[0].yy();

  const double divgrad = u_m*(dudx_m*basis[eqn_id].dphidx(i) 
			      + dudy_m*basis[eqn_id].dphidy(i) 
			      + dudz_m*basis[eqn_id].dphidz(i));

  return (basis[eqn_id].uu()-basis[eqn_id].uuold())/dt_*basis[eqn_id].phi(i)
    + divgrad
    + f1(u_m)*basis[eqn_id].phi(i)
    + f2(x,y,t_m)*basis[eqn_id].phi(i);
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_nlheatimr_test_dp_)) = residual_nlheatimr_test_;


KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_nlheatcn_test_)
{
  const double u[2] = {basis[eqn_id].uu(), basis[eqn_id].uuold()};
  const double dudx[2] = {basis[eqn_id].dudx(), basis[eqn_id].duolddx()};
  const double dudy[2] = {basis[eqn_id].dudy(), basis[eqn_id].duolddy()};
  const double dudz[2] = {basis[eqn_id].dudz(), basis[eqn_id].duolddz()};
  //const double dudz_m = t_theta_*basis[eqn_id].dudz + (1. - t_theta_)*basis[eqn_id].duolddz;
  const double t[2] = {time, time+dt_};
  const double x = basis[0].xx();
  const double y = basis[0].yy();

  const double divgrad = t_theta_*
    u[0]*(dudx[0]*basis[eqn_id].dphidx(i) 
	  + dudy[0]*basis[eqn_id].dphidy(i) 
	  + dudz[0]*basis[eqn_id].dphidz(i))
    + (1. - t_theta_)*
    u[1]*(dudx[1]*basis[eqn_id].dphidx(i) 
	  + dudy[1]*basis[eqn_id].dphidy(i) 
	  + dudz[1]*basis[eqn_id].dphidz(i));

  return (basis[eqn_id].uu()-basis[eqn_id].uuold())/dt_*basis[eqn_id].phi(i)
    + divgrad
    + (t_theta_*f1(u[0])
       + (1. - t_theta_)*f1(u[1]))*basis[eqn_id].phi(i)
    + (t_theta_*f2(x,y,t[0])
       + (1. - t_theta_)*f2(x,y,t[1]))*basis[eqn_id].phi(i);
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_nlheatcn_test_dp_)) = residual_nlheatcn_test_;


KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_nlheatcn_test_)
{
  return basis[eqn_id].phi(j)/dt_*basis[eqn_id].phi(i)
    + t_theta_*basis[eqn_id].uu()
    *(basis[eqn_id].dphidx(j)*basis[eqn_id].dphidx(i)
       + basis[eqn_id].dphidy(j)*basis[eqn_id].dphidy(i)
      + basis[eqn_id].dphidz(j)*basis[eqn_id].dphidz(i));
}

TUSAS_DEVICE
PRE_FUNC_TPETRA((*prec_nlheatcn_test_dp_)) = prec_nlheatcn_test_;


//}//namespace heat

#endif

namespace farzadi3d
{
  TUSAS_DEVICE
  double absphi = 0.9997;	//1.
  //double absphi = 0.999999;	//1.
  
  TUSAS_DEVICE
  double k = 0.14;
  double k_h = 0.14;

  TUSAS_DEVICE				//0.5
  double eps = 0.0;

  TUSAS_DEVICE
  double lambda = 10.;

  TUSAS_DEVICE
  double D_liquid = 3.e-9;			//1.e-11				//m^2/s
  
  TUSAS_DEVICE
  double m = -2.6;
  TUSAS_DEVICE					//-2.6 100.
  double c_inf = 3.;				//1.
  double c_inf_h = 3.;
  
  TUSAS_DEVICE
  double G = 3.e5;
  TUSAS_DEVICE											//k/m
  double R = 0.003;
  double R_h = 0.003;
//   TUSAS_DEVICE											//m/s
//   double V = 0.003;
	
  TUSAS_DEVICE											//m/s
  double d0 = 5.e-9;				//4.e-9					//m
  
  
  // parameters to scale dimensional quantities
  TUSAS_DEVICE
  double delta_T0 = 47.9143;
  double delta_T0_h = 47.9143;

  TUSAS_DEVICE
  double w0 = 5.65675e-8;
  double w0_h = 5.65675e-8;

  TUSAS_DEVICE
  double tau0 = 6.68455e-6;
  double tau0_h = 6.68455e-6;
  
//   TUSAS_DEVICE
//   double Vp0 = .354508;

  TUSAS_DEVICE
  double l_T0 = 2823.43;
  double l_T0_h = 2823.43;

  TUSAS_DEVICE
  double D_liquid_ = 6.267;
  
  TUSAS_DEVICE
  double dT = 0.0;
  double dT_h = 0.0;
  
//   TUSAS_DEVICE
  double base_height = 15.;

//   TUSAS_DEVICE
  double amplitude = 0.2;
  
  //circle or sphere parameters
  double r = 0.5;
  double x0 = 20.0; 
  double y0 = 20.0;
  double z0 = 20.0;
  
  int C = 0;
  
  TUSAS_DEVICE
  double t_activate_farzadi_d = 0.0;
  double t_activate_farzadi_h = 0.0;

  TUSAS_DEVICE
  double interface_noise_amplitude_d = 0.0;

PARAM_FUNC(param_)
{
  double k_p = plist->get<double>("k", 0.14);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(k,&k_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(k),&k_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  k = k_p;
#endif
  k_h = k_p;

  double eps_p = plist->get<double>("eps", 0.0);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(eps,&eps_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(eps),&eps_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  eps = eps_p;
#endif
  
  double lambda_p = plist->get<double>("lambda", 10.);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(lambda,&lambda_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(lambda),&lambda_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  lambda = lambda_p;
#endif
  
  double d0_p = plist->get<double>("d0", 5.e-9);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(d0,&d0_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(d0),&d0_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  d0 = d0_p;
#endif
  
  double D_liquid_p = plist->get<double>("D_liquid", 3.e-9);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(D_liquid,&D_liquid_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(D_liquid),&D_liquid_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  D_liquid = D_liquid_p;
#endif
  
  double m_p = plist->get<double>("m", -2.6);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(m,&m_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(m),&m_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  m = m_p;
#endif
  
  double c_inf_p = plist->get<double>("c_inf", 3.);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(c_inf,&c_inf_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(c_inf),&c_inf_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  c_inf = c_inf_p;
#endif
  c_inf_h = c_inf_p;
  
  double G_p = plist->get<double>("G", 3.e5);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(G,&G_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(G),&G_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  G = G_p;
#endif
  
  double R_p = plist->get<double>("R", 0.003);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(R,&R_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(R),&R_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  R = R_p;
#endif
  R_h = R_p;

// added dT here
double dT_p = plist->get<double>("dT", 0.0);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(dT,&dT_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(dT),&dT_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  dT = dT_p;
#endif
  dT_h = dT_p;

  double base_height_p = plist->get<double>("base_height", 15.);
// #if defined(TUSAS_HAVE_CUDA)
//   cudaMemcpyToSymbol(base_height,&base_height_p,sizeof(double));
// #else
  base_height = base_height_p;
// #endif
  double amplitude_p = plist->get<double>("amplitude", 0.2);
// #if defined(TUSAS_HAVE_CUDA)
//   cudaMemcpyToSymbol(amplitude,&amplitude_p,sizeof(double));
// #else
  amplitude = amplitude_p;
// #endif

  int C_p = plist->get<int>("C", 0);
  C = C_p;

// circle or sphere parameters

double r_p = plist->get<double>("r", 0.5);
r = r_p;
double x0_p = plist->get<double>("x0", 20.0);
x0 = x0_p;
double y0_p = plist->get<double>("y0", 20.0);
y0 = y0_p;
double z0_p = plist->get<double>("z0", 20.0);
z0 = z0_p;

//double absphi_p = plist->get<double>("absphi", absphi);

  //the calculated values need local vars to work....

  //calculated values
  double w0_p = lambda_p*d0_p/0.8839;
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(w0,&w0_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(w0),&w0_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  w0 = w0_p;
#endif
  w0_h = w0_p;
  
  double tau0_p = (lambda_p*0.6267*w0_p*w0_p)/D_liquid_p;
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(tau0,&tau0_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(tau0),&tau0_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  tau0 = tau0_p;
#endif
  tau0_h = tau0_p;

//   double V_p = R_p;
// #if defined(TUSAS_HAVE_CUDA)
//   cudaMemcpyToSymbol(V,&V_p,sizeof(double));
// #else
//   V = V_p;
// #endif

//   double Vp0_p = V_p*tau0_p/w0_p;
// #if defined(TUSAS_HAVE_CUDA)
//   cudaMemcpyToSymbol(Vp0,&Vp0_p,sizeof(double));
// #else
//   Vp0 = Vp0_p;
// #endif

  double delta_T0_p = abs(m_p)*c_inf_p*(1.-k_p)/k_p;
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(delta_T0,&delta_T0_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(delta_T0),&delta_T0_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  delta_T0 = delta_T0_p;
#endif
  delta_T0_h = delta_T0_p;
  
  double l_T0_p = delta_T0_p/G_p;
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(l_T0,&l_T0_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(l_T0),&l_T0_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  l_T0 = l_T0_p;
#endif
  l_T0_h = l_T0_p;
  
  double D_liquid__p = D_liquid_p*tau0_p/(w0_p*w0_p);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(D_liquid_,&D_liquid__p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(D_liquid_),&D_liquid__p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  D_liquid_ = D_liquid__p;
#endif

double t_activate_farzadi_p = plist->get<double>("t_activate_farzadi", 0.0);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(t_activate_farzadi_d,&t_activate_farzadi_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(t_activate_farzadi_d),&t_activate_farzadi_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  t_activate_farzadi_d = t_activate_farzadi_p;
#endif
  t_activate_farzadi_h = t_activate_farzadi_p;

double interface_noise_amplitude_p = plist->get<double>("interface_noise_amplitude", 0.0);
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(interface_noise_amplitude_d,&interface_noise_amplitude_p,sizeof(double));
#elif defined(TUSAS_HAVE_HIP)
  hipMemcpyToSymbol(HIP_SYMBOL(interface_noise_amplitude_d),&interface_noise_amplitude_p,sizeof(double),0,hipMemcpyHostToDevice);
#else
  interface_noise_amplitude_d = interface_noise_amplitude_p;
#endif

}
  //see tpetra::pfhub3 for a possibly better implementation of a,ap
KOKKOS_INLINE_FUNCTION 
double a(const double &p,const double &px,const double &py,const double &pz, const double ep)
{
  double val = 1. + ep;
  val = (p*p < farzadi3d::absphi)&&(p*p > 1.-farzadi3d::absphi) ? (1.-3.*ep)*(1.+4.*ep/(1.-3.*ep)*
				    (px*px*px*px+py*py*py*py+pz*pz*pz*pz)/(px*px+py*py+pz*pz)/(px*px+py*py+pz*pz))
    : 1. + ep;
//   if(val!=val)  std::cout<<farzadi3d::absphi<<" "<<1.-farzadi3d::absphi<<" "<<p*p<<" "<<px*px+py*py+pz*pz<<" "<<val<<" "<<
// 	   (1.-3.*ep)*(1.+4.*ep/(1.-3.*ep)*
// 				    (px*px*px*px+py*py*py*py+pz*pz*pz*pz)/(px*px+py*py+pz*pz)/(px*px+py*py+pz*pz))<<std::endl;
  return val;
}

KOKKOS_INLINE_FUNCTION 
double ap(const double &p,const double &px,const double &py,const double &pz,const double &pd, const double ep)
{
  return (p*p < farzadi3d::absphi)&&(p*p > 1.-farzadi3d::absphi) ? 4.*ep*
				    (4.*pd*pd*pd*(px*px+py*py+pz*pz)-4.*pd*(px*px*px*px+py*py*py*py+pz*pz*pz*pz))
				    /(px*px+py*py+pz*pz)/(px*px+py*py+pz*pz)/(px*px+py*py+pz*pz)
    : 0.;
}
 
KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_conc_farzadi_)
{
  //right now, if explicit, we will have some problems with time derivates below
  const double dtestdx = basis[eqn_id].dphidx(i);
  const double dtestdy = basis[eqn_id].dphidy(i);
  const double dtestdz = basis[eqn_id].dphidz(i);
  const double test = basis[eqn_id].phi(i);
  const double u[3] = {basis[eqn_id].uu(),basis[eqn_id].uuold(),basis[eqn_id].uuoldold()};

  const int phi_id = eqn_id+1;
  const double phi[3] = {basis[phi_id].uu(),basis[phi_id].uuold(),basis[phi_id].uuoldold()};
  const double dphidx[3] = {basis[phi_id].dudx(),basis[phi_id].duolddx(),basis[phi_id].duoldolddx()};
  const double dphidy[3] = {basis[phi_id].dudy(),basis[phi_id].duolddy(),basis[phi_id].duoldolddy()};
  const double dphidz[3] = {basis[phi_id].dudz(),basis[phi_id].duolddz(),basis[phi_id].duoldolddz()};

  const double ut = (1. + k - (1.0 - k) * phi[0]) / 2. * (u[0] - u[1]) / dt_ * test;
  const double divgradu[3] = {D_liquid_*(1.-phi[0])/2.*(basis[eqn_id].dudx()*dtestdx + basis[eqn_id].dudy()*dtestdy + basis[eqn_id].dudz()*dtestdz),
  			      D_liquid_*(1.-phi[1])/2.*(basis[eqn_id].duolddx()*dtestdx + basis[eqn_id].duolddy()*dtestdy + basis[eqn_id].duolddz()*dtestdz),
  			      D_liquid_*(1.-phi[2])/2.*(basis[eqn_id].duoldolddx()*dtestdx + basis[eqn_id].duoldolddy()*dtestdy + basis[eqn_id].duoldolddz()*dtestdz)};//(grad u,grad phi)

  const double normd[3] = {(phi[0]*phi[0] < absphi)&&(phi[0]*phi[0] > 0.) ? 1./sqrt(dphidx[0]*dphidx[0] + dphidy[0]*dphidy[0] + dphidz[0]*dphidz[0]) : 0.,
  			   (phi[1]*phi[1] < absphi)&&(phi[1]*phi[1] > 0.) ? 1./sqrt(dphidx[1]*dphidx[1] + dphidy[1]*dphidy[1] + dphidz[1]*dphidz[1]) : 0.,
  			   (phi[2]*phi[2] < absphi)&&(phi[2]*phi[2] > 0.) ? 1./sqrt(dphidx[2]*dphidx[2] + dphidy[2]*dphidy[2] + dphidz[2]*dphidz[2]) : 0.}; //cn lim grad phi/|grad phi| may -> 1 here?

  //we need to double check these terms with temporal derivatives....
  const double phit = (phi[0]-phi[1])/dt_;
  const double j_coef[3] = {(1.+(1.-k)*u[0])/sqrt(8.)*normd[0]*phit,
  			    (1.+(1.-k)*u[1])/sqrt(8.)*normd[1]*phit,
  			    (1.+(1.-k)*u[2])/sqrt(8.)*normd[2]*phit};
  const double divj[3] = {j_coef[0]*(dphidx[0]*dtestdx + dphidy[0]*dtestdy + dphidz[0]*dtestdz),
  			  j_coef[1]*(dphidx[1]*dtestdx + dphidy[1]*dtestdy + dphidz[1]*dtestdz),
  			  j_coef[2]*(dphidx[2]*dtestdx + dphidy[2]*dtestdy + dphidz[2]*dtestdz)};

  double phitu[3] = {-.5*phit*(1.+(1.-k)*u[0])*test,
  		     -.5*phit*(1.+(1.-k)*u[1])*test,
  		     -.5*phit*(1.+(1.-k)*u[2])*test}; 
    
    //double val = ut + t_theta_*divgradu  + t_theta_*divj + t_theta_*phitu;
    //printf("%lf\n",val);

  const double f[3] = {divgradu[0] + divj[0] + phitu[0],
  		       divgradu[1] + divj[1] + phitu[1],
  		       divgradu[2] + divj[2] + phitu[2]};

  return (ut + (1.-t_theta2_)*t_theta_*f[0]
      + (1.-t_theta2_)*(1.-t_theta_)*f[1]
      +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]));
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_conc_farzadi_dp_)) = residual_conc_farzadi_;

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_phase_farzadi_)
{
  //derivatives of the test function
  const double dtestdx = basis[eqn_id].dphidx(i);
  const double dtestdy = basis[eqn_id].dphidy(i);
  const double dtestdz = basis[eqn_id].dphidz(i);
  //test function
  const double test = basis[eqn_id].phi(i);
  //u, phi
  const int u_id = eqn_id-1;
  const double u[3] = {basis[u_id].uu(),basis[u_id].uuold(),basis[u_id].uuoldold()};
  const double phi[3] = {basis[eqn_id].uu(),basis[eqn_id].uuold(),basis[eqn_id].uuoldold()};

  const double dphidx[3] = {basis[eqn_id].dudx(),basis[eqn_id].duolddx(),basis[eqn_id].duoldolddx()};
  const double dphidy[3] = {basis[eqn_id].dudy(),basis[eqn_id].duolddy(),basis[eqn_id].duoldolddy()};
  const double dphidz[3] = {basis[eqn_id].dudz(),basis[eqn_id].duolddz(),basis[eqn_id].duoldolddz()};

  const double as[3] = {a(phi[0],dphidx[0],dphidy[0],dphidz[0],eps),
  			a(phi[1],dphidx[1],dphidy[1],dphidz[1],eps),
  			a(phi[2],dphidx[2],dphidy[2],dphidz[2],eps)};

  const double divgradphi[3] = {as[0]*as[0]*(dphidx[0]*dtestdx + dphidy[0]*dtestdy + dphidz[0]*dtestdz),
  				as[1]*as[1]*(dphidx[1]*dtestdx + dphidy[1]*dtestdy + dphidz[1]*dtestdz),
  				as[2]*as[2]*(dphidx[2]*dtestdx + dphidy[2]*dtestdy + dphidz[2]*dtestdz)};//(grad u,grad phi)

  const double mob[3] = {(1.+(1.-k)*u[0])*as[0]*as[0],(1.+(1.-k)*u[1])*as[1]*as[1],(1.+(1.-k)*u[2])*as[2]*as[2]};
  const double phit = (phi[0]-phi[1])/dt_*test;

  //double curlgrad = -dgdtheta*dphidy*dtestdx + dgdtheta*dphidx*dtestdy;
  const double curlgrad[3] = {as[0]*(dphidx[0]*dphidx[0] + dphidy[0]*dphidy[0] + dphidz[0]*dphidz[0])
  			      *(ap(phi[0],dphidx[0],dphidy[0],dphidz[0],dphidx[0],eps)*dtestdx 
  				+ ap(phi[0],dphidx[0],dphidy[0],dphidz[0],dphidy[0],eps)*dtestdy 
  				+ ap(phi[0],dphidx[0],dphidy[0],dphidz[0],dphidz[0],eps)*dtestdz),
  			      as[1]*(dphidx[1]*dphidx[1] + dphidy[1]*dphidy[1] + dphidz[1]*dphidz[1])
  			      *(ap(phi[1],dphidx[1],dphidy[1],dphidz[1],dphidx[1],eps)*dtestdx 
  				+ ap(phi[1],dphidx[1],dphidy[1],dphidz[1],dphidy[1],eps)*dtestdy 
  				+ ap(phi[1],dphidx[1],dphidy[1],dphidz[1],dphidz[1],eps)*dtestdz),
  			      as[2]*(dphidx[2]*dphidx[2] + dphidy[2]*dphidy[2] + dphidz[2]*dphidz[2])
  			      *(ap(phi[2],dphidx[2],dphidy[2],dphidz[2],dphidx[2],eps)*dtestdx 
  				+ ap(phi[2],dphidx[2],dphidy[2],dphidz[2],dphidy[2],eps)*dtestdy 
  				+ ap(phi[2],dphidx[2],dphidy[2],dphidz[2],dphidz[2],eps)*dtestdz)};
    
  const double gp1[3] = {-(phi[0] - phi[0]*phi[0]*phi[0])*test,
  			 -(phi[1] - phi[1]*phi[1]*phi[1])*test,
  			 -(phi[2] - phi[2]*phi[2]*phi[2])*test};

  //note in paper eq 39 has g3 different
  //here (as implemented) our g3 = lambda*(1. - phi[0]*phi[0])*(1. - phi[0]*phi[0])
  //matches farzadi eq 10

  const double hp1u[3] = {lambda*(1. - phi[0]*phi[0])*(1. - phi[0]*phi[0])*(u[0])*test,
  			 lambda*(1. - phi[1]*phi[1])*(1. - phi[1]*phi[1])*(u[1])*test,
  			 lambda*(1. - phi[2]*phi[2])*(1. - phi[2]*phi[2])*(u[2])*test};
    
  const double f[3] = {(divgradphi[0] + curlgrad[0] + gp1[0] + hp1u[0])/mob[0],
  		       (divgradphi[1] + curlgrad[1] + gp1[1] + hp1u[1])/mob[1],
  		       (divgradphi[2] + curlgrad[2] + gp1[2] + hp1u[2])/mob[2]};

  const double val = phit 
      + (1.-t_theta2_)*t_theta_*f[0]
      + (1.-t_theta2_)*(1.-t_theta_)*f[1]
      +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);

  return mob[0]*val;
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_phase_farzadi_dp_)) = residual_phase_farzadi_;

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_phase_farzadi_uncoupled_)
{
  //test function
  const double test = basis[eqn_id].phi(i);
  //u, phi
  const int u_id = eqn_id-1;
  const double u[3] = {basis[u_id].uu(),basis[u_id].uuold(),basis[u_id].uuoldold()};
  const double phi[3] = {basis[eqn_id].uu(),basis[eqn_id].uuold(),basis[eqn_id].uuoldold()};

  const double dphidx[3] = {basis[eqn_id].dudx(),basis[eqn_id].duolddx(),basis[eqn_id].duoldolddx()};
  const double dphidy[3] = {basis[eqn_id].dudy(),basis[eqn_id].duolddy(),basis[eqn_id].duoldolddy()};
  const double dphidz[3] = {basis[eqn_id].dudz(),basis[eqn_id].duolddz(),basis[eqn_id].duoldolddz()};

  const double as[3] = {a(phi[0],dphidx[0],dphidy[0],dphidz[0],eps),
			a(phi[1],dphidx[1],dphidy[1],dphidz[1],eps),
			a(phi[2],dphidx[2],dphidy[2],dphidz[2],eps)};

  const double mob[3] = {(1.+(1.-k)*u[0])*as[0]*as[0],(1.+(1.-k)*u[1])*as[1]*as[1],(1.+(1.-k)*u[2])*as[2]*as[2]};

  const double x = basis[eqn_id].xx();
  
  
  // frozen temperature approximation: linear pulling of the temperature field
  const double xx = x*w0;

  //cn this should probablly be: (time+dt_)*tau
  const double tt[3] = {(time+dt_)*tau0,time*tau0,(time-dtold_)*tau0};

  const double g4[3] = {((dT < 0.001) ? G*(xx-R*tt[0])/delta_T0 : dT),
			     ((dT < 0.001) ? G*(xx-R*tt[1])/delta_T0 : dT),
			     ((dT < 0.001) ? G*(xx-R*tt[2])/delta_T0 : dT)};
  
  const double hp1g4[3] = {lambda*(1. - phi[0]*phi[0])*(1. - phi[0]*phi[0])*(g4[0])*test,
			 lambda*(1. - phi[1]*phi[1])*(1. - phi[1]*phi[1])*(g4[1])*test,
			 lambda*(1. - phi[2]*phi[2])*(1. - phi[2]*phi[2])*(g4[2])*test};

  const double val = tpetra::farzadi3d::residual_phase_farzadi_(basis,
								   i,
								   dt_,
								   dtold_,
								   t_theta_,
								   t_theta2_,
								   time,
								eqn_id,
								vol,
								rand);

  const double rv = val/mob[0]
    + (1.-t_theta2_)*t_theta_*hp1g4[0]/mob[0]
    + (1.-t_theta2_)*(1.-t_theta_)*hp1g4[1]/mob[1]
    +.5*t_theta2_*((2.+dt_/dtold_)*hp1g4[1]/mob[1]-dt_/dtold_*hp1g4[2]/mob[2]);

  return mob[0]*rv;
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_phase_farzadi_uncoupled_dp_)) = residual_phase_farzadi_uncoupled_;

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_phase_farzadi_coupled_)
{
  //test function
  const double test = basis[eqn_id].phi(i);
  //u, phi
  const int u_id = eqn_id-1;
  const int theta_id = eqn_id+1;
  const double u[3] = {basis[u_id].uu(),basis[u_id].uuold(),basis[u_id].uuoldold()};
  const double phi[3] = {basis[eqn_id].uu(),basis[eqn_id].uuold(),basis[eqn_id].uuoldold()};

  const double dphidx[3] = {basis[eqn_id].dudx(),basis[eqn_id].duolddx(),basis[eqn_id].duoldolddx()};
  const double dphidy[3] = {basis[eqn_id].dudy(),basis[eqn_id].duolddy(),basis[eqn_id].duoldolddy()};
  const double dphidz[3] = {basis[eqn_id].dudz(),basis[eqn_id].duolddz(),basis[eqn_id].duoldolddz()};

  const double as[3] = {a(phi[0],dphidx[0],dphidy[0],dphidz[0],eps),
			a(phi[1],dphidx[1],dphidy[1],dphidz[1],eps),
			a(phi[2],dphidx[2],dphidy[2],dphidz[2],eps)};

  const double mob[3] = {(1.+(1.-k)*u[0])*as[0]*as[0],(1.+(1.-k)*u[1])*as[1]*as[1],(1.+(1.-k)*u[2])*as[2]*as[2]};

  const double theta[3] = {basis[theta_id].uu(),basis[theta_id].uuold(),basis[theta_id].uuoldold()};
  
  const double g4[3] = {theta[0],theta[1],theta[2]};
  
  const double hp1g4[3] = {lambda*(1. - phi[0]*phi[0])*(1. - phi[0]*phi[0])*(g4[0])*test,
			 lambda*(1. - phi[1]*phi[1])*(1. - phi[1]*phi[1])*(g4[1])*test,
			 lambda*(1. - phi[2]*phi[2])*(1. - phi[2]*phi[2])*(g4[2])*test};

  const double val = tpetra::farzadi3d::residual_phase_farzadi_(basis,
								   i,
								   dt_,
								   dtold_,
								   t_theta_,
								   t_theta2_,
								   time,
								eqn_id,
								vol,
								rand);

  
  const double noise_term[3] = {interface_noise_amplitude_d*std::sqrt(dt_/vol) * rand*test * (1.0 - phi[0]*phi[0]), 0.0, 0.0};

  const double rv = val/mob[0]
    + (1.-t_theta2_)*t_theta_*(hp1g4[0]/mob[0] + noise_term[0])
    + (1.-t_theta2_)*(1.-t_theta_)*(hp1g4[1]/mob[1] + noise_term[1])
    +.5*t_theta2_*( (2.+dt_/dtold_)*hp1g4[1]/mob[1] - dt_/dtold_*hp1g4[2]/mob[2] + (2.+dt_/dtold_)*noise_term[1]-dt_/dtold_*noise_term[2] );
	
  return mob[0]*rv;
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_phase_farzadi_coupled_dp_)) = residual_phase_farzadi_coupled_;

KOKKOS_INLINE_FUNCTION
RES_FUNC_TPETRA(residual_conc_farzadi_activated_)
{
	const double val = tpetra::farzadi3d::residual_conc_farzadi_(basis,
  						 i,
  						 dt_,
  						 dtold_,
  						 t_theta_,
  						 t_theta2_,
  						 time,
								     eqn_id,
								     vol,
								     rand);
	
	const double u[2] = {basis[eqn_id].uu(),basis[eqn_id].uuold()};
	
	// Coefficient to turn Farzadi evolution off until a specified time
	const double delta = 1.0e12; 			   
	const double sigmoid_var = delta * (time-t_activate_farzadi_d/tau0);
	const double sigmoid = 0.5 * (1.0 + sigmoid_var / (std::sqrt(1.0 + sigmoid_var*sigmoid_var))); 			   

	return val * sigmoid + (u[1]-u[0]) * (1.0 - sigmoid)*basis[eqn_id].phi(i);
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_conc_farzadi_activated_dp_)) = residual_conc_farzadi_activated_;

KOKKOS_INLINE_FUNCTION
RES_FUNC_TPETRA(residual_phase_farzadi_coupled_activated_)
{
	const double val = tpetra::farzadi3d::residual_phase_farzadi_coupled_(basis,
  						 i,
  						 dt_,
  						 dtold_,
  						 t_theta_,
  						 t_theta2_,
  						 time,
									      eqn_id,
									      vol,
									      rand);
	
	const double phi[2] = {basis[eqn_id].uu(),basis[eqn_id].uuold()};
	
	// Coefficient to turn Farzadi evolution off until a specified time
	const double delta = 1.0e12; 			   
	const double sigmoid_var = delta * (time-t_activate_farzadi_d/tau0);
	const double sigmoid = 0.5 * (1.0 + sigmoid_var / (std::sqrt(1.0 + sigmoid_var*sigmoid_var))); 			   
	//if(sigmoid > 0) printf("%lf\n",sigmoid);
	return val * sigmoid + (phi[1]-phi[0]) * (1.0 - sigmoid)*basis[eqn_id].phi(i);
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_phase_farzadi_coupled_activated_dp_)) = residual_phase_farzadi_coupled_activated_;

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_phase_farzadi_)
{
  const double dtestdx = basis[eqn_id].dphidx(i);
  const double dtestdy = basis[eqn_id].dphidy(i);
  const double dtestdz = basis[eqn_id].dphidz(i);
  const double dbasisdx = basis[eqn_id].dphidx(j);
  const double dbasisdy = basis[eqn_id].dphidy(j);
  const double dbasisdz = basis[eqn_id].dphidz(j);

  const double test = basis[1].phi(i);
  
  const double dphidx = basis[1].dudx();
  const double dphidy = basis[1].dudy();
  const double dphidz = basis[1].dudz();

  const double u = basis[0].uu();
  const double phi = basis[1].uu();

  const double as = a(phi,dphidx,dphidy,dphidz,eps);

  const double m = (1.+(1.-k)*u)*as*as;
  const double phit = (basis[1].phi(j))/dt_*test;

  const double divgrad = as*as*(dbasisdx*dtestdx + dbasisdy*dtestdy + dbasisdz*dtestdz);

  return (phit + t_theta_*(divgrad)/m)*m;
}

TUSAS_DEVICE
PRE_FUNC_TPETRA((*prec_phase_farzadi_dp_)) = prec_phase_farzadi_;

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_conc_farzadi_)
{
  const double dtestdx = basis[eqn_id].dphidx(i);
  const double dtestdy = basis[eqn_id].dphidy(i);
  const double dtestdz = basis[eqn_id].dphidz(i);
  const double dbasisdx = basis[eqn_id].dphidx(j);
  const double dbasisdy = basis[eqn_id].dphidy(j);
  const double dbasisdz = basis[eqn_id].dphidz(j);

  const double test = basis[0].phi(i);
  const double divgrad = D_liquid_*(1.-basis[1].uu())/2.*(dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz);

  const int phi_id = eqn_id+1;
  const double phi = basis[phi_id].uu();
  const double u_t = (1. + k - (1.0 - k) * phi) / 2. *basis[0].phi(j)  / dt_ * test;

  return u_t + t_theta_*(divgrad);

}

TUSAS_DEVICE
PRE_FUNC_TPETRA((*prec_conc_farzadi_dp_)) = prec_conc_farzadi_;

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_conc_farzadi_exp_)
{
  //this is the explicit case with explicit phit
  const double dtestdx = basis[0].dphidx(i);
  const double dtestdy = basis[0].dphidy(i);
  const double dtestdz = basis[0].dphidz(i);
  const double test = basis[0].phi(i);
  const double u[2] = {basis[0].uu(),basis[0].uuold()};
  const double phi[2] = {basis[1].uu(),basis[1].uuold()};
  const double dphidx[2] = {basis[1].dudx(),basis[1].duolddx()};
  const double dphidy[2] = {basis[1].dudy(),basis[1].duolddy()};
  const double dphidz[2] = {basis[1].dudz(),basis[1].duolddz()};

  const double ut = (1.+k-(1.0-k)*phi[0])/2.*(u[0]-u[1])/dt_*test;
  const double divgradu[2] = {D_liquid_*(1.-phi[0])/2.*(basis[0].dudx()*dtestdx + basis[0].dudy()*dtestdy + basis[0].dudz()*dtestdz),
			      D_liquid_*(1.-phi[1])/2.*(basis[0].duolddx()*dtestdx + basis[0].duolddy()*dtestdy + basis[0].duolddz()*dtestdz)};//(grad u,grad phi)

  const double normd[2] = {(phi[0]*phi[0] < absphi)&&(phi[0]*phi[0] > 0.) ? 1./sqrt(dphidx[0]*dphidx[0] + dphidy[0]*dphidy[0] + dphidz[0]*dphidz[0]) : 0.,
			   (phi[1]*phi[1] < absphi)&&(phi[1]*phi[1] > 0.) ? 1./sqrt(dphidx[1]*dphidx[1] + dphidy[1]*dphidy[1] + dphidz[1]*dphidz[1]) : 0.}; //cn lim grad phi/|grad phi| may . 1 here?

  const double phit = (phi[0]-phi[1])/dt_;
  const double j_coef[2] = {(1.+(1.-k)*u[0])/sqrt(8.)*normd[0]*phit,
			    (1.+(1.-k)*u[1])/sqrt(8.)*normd[1]*phit};
  const double divj[2] = {j_coef[0]*(dphidx[0]*dtestdx + dphidy[0]*dtestdy + dphidz[0]*dtestdz),
			  j_coef[1]*(dphidx[1]*dtestdx + dphidy[1]*dtestdy + dphidz[1]*dtestdz)};

  double phitu[2] = {-.5*phit*(1.+(1.-k)*u[0])*test,
		     -.5*phit*(1.+(1.-k)*u[1])*test}; 
  
  //double val = ut + t_theta_*divgradu  + t_theta_*divj + t_theta_*phitu;
  //printf("%lf\n",val);

  return ut + t_theta_*(divgradu[0] + divj[0] + phitu[0]) + (1.-t_theta_)*(divgradu[1] + divj[1] + phitu[1]);
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_conc_farzadi_exp_dp_)) = residual_conc_farzadi_exp_;

INI_FUNC(init_phase_farzadi_)
{

  double h = base_height + amplitude*((double)rand()/(RAND_MAX));
  
  double c = (x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0);
  
  return (C == 0) ? (tanh((h-x)/sqrt(2.))) : ((c < r*r) ? 1. : -1.);	

}

INI_FUNC(init_phase_farzadi_test_)
{
  const double pp = 36.;
  const double ll = .2;
  const double aa = 9.;
  const double pi = 3.141592653589793;
  double r = ll*(1.+(2.+sin(y*aa*pi/pp))
		 *(2.+sin(y*aa*pi/pp/2.))
		 *(2.+sin(y*aa*pi/pp/4.)));
  double val = -1.;
  if(x < r) val = 1.;
  return val;
}

INI_FUNC(init_conc_farzadi_)
{
  return -1.;
}

PPR_FUNC(postproc_c_)
{
  // return the physical concentration
  const double uu = u[0];
  const double phi = u[1];

  return -c_inf_h*(1.+k_h-phi+k_h*phi)*(-1.-uu+k_h*uu)/2./k_h;
}

PPR_FUNC(postproc_t_)
{
  // return the physical temperature in K here
  double x = xyz[0];

  double xx = x*w0_h;
  double tt = time*tau0_h;
  return ((dT < 0.001) ? 877.3 + (xx-R_h*tt)/l_T0_h*delta_T0_h : 877.3);
}
PPR_FUNC(postproc_sigmoid_)
{
  const double delta = 1.0e12; 			   
  const double sigmoid_var = delta * (time-t_activate_farzadi_h/tau0_h);
  const double sigmoid = 0.5 * (1.0 + sigmoid_var / (std::sqrt(1.0 + sigmoid_var*sigmoid_var))); 
  return sigmoid;
}

}//namespace farzadi3d

#if 0

namespace pfhub3
{
  const double R_ = 8.;// 8.;

  TUSAS_DEVICE
  double smalld_ = 0.;
  TUSAS_DEVICE
  const double delta_ = -.3;//-.3;
  TUSAS_DEVICE
  const double D_ = 10.;
  TUSAS_DEVICE
  const double eps_ = .05;
  TUSAS_DEVICE
  const double tau0_ = 1.;
  TUSAS_DEVICE
  const double W_ = 1.;
  TUSAS_DEVICE
  const double lambda_ = D_*tau0_/.6267/W_/W_;

PARAM_FUNC(param_)
{
  //we will need to propgate this to device
  double smalld_p = plist->get<double>("smalld", smalld_);
  smalld_ = smalld_p;
}

KOKKOS_INLINE_FUNCTION 
double a(const double &p,const double &px,const double &py,const double &pz, const double ep)
{
  double val = 1. + ep;
  const double d = (px*px+py*py+pz*pz)*(px*px+py*py+pz*pz);
  val = (d > smalld_) ? (1.-3.*ep)*(1.+4.*ep/(1.-3.*ep)*(px*px*px*px+py*py*py*py+pz*pz*pz*pz)/d)
    : 1. + ep;
  //older version produced nicer dendrite
//   const double d = (px*px+py*py+pz*pz)*(px*px+py*py+pz*pz);
//   val = (d > smalld_) ? (1.-3.*ep)*(1.+4.*ep/(1.-3.*ep)*(px*px*px*px+py*py*py*py+pz*pz*pz*pz)/d)
//     : 1. + ep;

  return val;
}

KOKKOS_INLINE_FUNCTION 
double ap(const double &p,const double &px,const double &py,const double &pz,const double &pd, const double ep)
{
  //older version produced nicer dendrite  
//   const double d = (px*px+py*py+pz*pz)*(px*px+py*py+pz*pz);
//   return (d > smalld_) ? 4.*ep*
// 				    (4.*pd*pd*pd*(px*px+py*py+pz*pz)-4.*pd*(px*px*px*px+py*py*py*py+pz*pz*pz*pz))
// 				    /(px*px+py*py+pz*pz)/d
//     : 0.;
  const double d = (px*px+py*py+pz*pz)*(px*px+py*py+pz*pz);
  return (d > smalld_) ? 4.*ep*
    (4.*pd*pd*pd*(px*px+py*py+pz*pz)-4.*pd*(px*px*px*px+py*py*py*py+pz*pz*pz*pz))
    /((px*px+py*py+pz*pz)*d)
    : 0.;
}

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_heat_pfhub3_)
{
  const double ut = (basis[eqn_id].uu()-basis[eqn_id].uuold())/dt_*basis[eqn_id].phi(i);
  double divgradu[3] = {D_*(basis[eqn_id].dudx()*basis[eqn_id].dphidx(i)
			  + basis[eqn_id].dudy()*basis[eqn_id].dphidy(i)
			    + basis[eqn_id].dudz()*basis[eqn_id].dphidz(i)),
			D_*(basis[eqn_id].duolddx()*basis[eqn_id].dphidx(i)
			  + basis[eqn_id].duolddy()*basis[eqn_id].dphidy(i)
			    + basis[eqn_id].duolddz()*basis[eqn_id].dphidz(i)),
			D_*(basis[eqn_id].duoldolddx()*basis[eqn_id].dphidx(i)
			  + basis[eqn_id].duoldolddy()*basis[eqn_id].dphidy(i)
			    + basis[eqn_id].duoldolddz()*basis[eqn_id].dphidz(i))};

  const double phit[2] = {.5*(basis[1].uu()-basis[1].uuold())/dt_*basis[0].phi(i),
			  .5*(basis[1].uuold()-basis[1].uuoldold())/dt_*basis[0].phi(i)};

  double f[3];
  f[0] = -divgradu[0] + phit[0];
  f[1] = -divgradu[1] + phit[1];
  f[2] = -divgradu[2] + phit[1];

  return ut - (1.-t_theta2_)*t_theta_*f[0]
    - (1.-t_theta2_)*(1.-t_theta_)*f[1]
    -.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}
TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_heat_pfhub3_dp_)) = residual_heat_pfhub3_;

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_phase_pfhub3_)
{
  const double test = basis[eqn_id].phi(i);
  const double dtestdx = basis[eqn_id].dphidx(i);
  const double dtestdy = basis[eqn_id].dphidy(i);
  const double dtestdz = basis[eqn_id].dphidz(i);

  const double phi[3] = {basis[eqn_id].uu(),basis[eqn_id].uuold(),basis[eqn_id].uuoldold()};
  const double dphidx[3] = {basis[eqn_id].dudx(),basis[eqn_id].duolddx(),basis[eqn_id].duoldolddx()};
  const double dphidy[3] = {basis[eqn_id].dudy(),basis[eqn_id].duolddy(),basis[eqn_id].duoldolddy()};
  const double dphidz[3] = {basis[eqn_id].dudz(),basis[eqn_id].duolddz(),basis[eqn_id].duoldolddz()};

  const double as[3] = {a(phi[0],
			  dphidx[0],
			  dphidy[0],
			  dphidz[0],
			  eps_),
			a(phi[1],
			  dphidx[1],
			  dphidy[1],
			  dphidz[1],
			  eps_),
			a(phi[2],
			  dphidx[2],
			  dphidy[2],
			  dphidz[2],
			  eps_)};

  const double tau[3] = {tau0_*as[0]*as[0],tau0_*as[1]*as[1],tau0_*as[2]*as[2]};

  const double phit = (phi[0]-phi[1])/dt_*test;

  const double w[3] = {W_*as[0],W_*as[1],W_*as[2]};

//   const double divgradphi[3] = {w[0]*w[0]*(dphidx[0]*dtestdx
// 					     + dphidy[0]*dtestdy
// 					     + dphidz[0]*dtestdz),
// 				w[1]*w[1]*(dphidx[1]*dtestdx
// 					     + dphidy[1]*dtestdy
// 					     + dphidz[1]*dtestdz),
// 				w[2]*w[2]*(dphidx[2]*dtestdx
// 					     + dphidy[2]*dtestdy
// 					     + dphidz[2]*dtestdz)};
  const double divgradphi[3] = {W_*W_*(dphidx[0]*dtestdx
					     + dphidy[0]*dtestdy
					     + dphidz[0]*dtestdz),
				W_*W_*(dphidx[1]*dtestdx
					     + dphidy[1]*dtestdy
					     + dphidz[1]*dtestdz),
				W_*W_*(dphidx[2]*dtestdx
					     + dphidy[2]*dtestdy
					     + dphidz[2]*dtestdz)};

  const double wp[3] = {W_*(ap(phi[0],dphidx[0],dphidy[0],dphidz[0],dphidx[0],eps_)*dtestdx 
			    + ap(phi[0],dphidx[0],dphidy[0],dphidz[0],dphidy[0],eps_)*dtestdy 
			    + ap(phi[0],dphidx[0],dphidy[0],dphidz[0],dphidz[0],eps_)*dtestdz),
			W_*(ap(phi[1],dphidx[1],dphidy[1],dphidz[1],dphidx[1],eps_)*dtestdx 
			    + ap(phi[1],dphidx[1],dphidy[1],dphidz[1],dphidy[1],eps_)*dtestdy 
			    + ap(phi[1],dphidx[1],dphidy[1],dphidz[1],dphidz[1],eps_)*dtestdz),
			W_*(ap(phi[2],dphidx[2],dphidy[2],dphidz[2],dphidx[2],eps_)*dtestdx 
			    + ap(phi[2],dphidx[2],dphidy[2],dphidz[2],dphidy[2],eps_)*dtestdy 
			    + ap(phi[2],dphidx[2],dphidy[2],dphidz[2],dphidz[2],eps_)*dtestdz)};

  const double curlgrad[3] = {w[0]*(dphidx[0]*dphidx[0] + dphidy[0]*dphidy[0] + dphidz[0]*dphidz[0])*wp[0],
			      w[1]*(dphidx[1]*dphidx[1] + dphidy[1]*dphidy[1] + dphidz[1]*dphidz[1])*wp[1],
			      w[2]*(dphidx[2]*dphidx[2] + dphidy[2]*dphidy[2] + dphidz[2]*dphidz[2])*wp[2]};

  const double g[3] = {((phi[0]-lambda_*basis[0].uu()*(1.-phi[0]*phi[0]))*(1.-phi[0]*phi[0]))*test,
		       ((phi[1]-lambda_*basis[0].uuold()*(1.-phi[1]*phi[1]))*(1.-phi[1]*phi[1]))*test,
		       ((phi[2]-lambda_*basis[0].uuoldold()*(1.-phi[2]*phi[2]))*(1.-phi[2]*phi[2]))*test};

  double f[3];
  f[0] = -(divgradphi[0]/tau0_+curlgrad[0]/tau[0]-g[0]/tau[0]);
  f[1] = -(divgradphi[1]/tau0_+curlgrad[1]/tau[1]-g[1]/tau[1]);
  f[2] = -(divgradphi[2]/tau0_+curlgrad[2]/tau[2]-g[2]/tau[2]);

  return phit - (1.-t_theta2_)*t_theta_*f[0]
    - (1.-t_theta2_)*(1.-t_theta_)*f[1]
    -.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}
#if 0
RES_FUNC(residual_heat_pfhub3_n_)
{
  const double ut = (basis[eqn_id].uu-basis[eqn_id].uuold)/dt_*basis[eqn_id].phi[i];
  const double divgradu[3] = {D_*(basis[eqn_id].dudx*basis[eqn_id].dphidx[i]
				  + basis[eqn_id].dudy*basis[eqn_id].dphidy[i]
				  + basis[eqn_id].dudz*basis[eqn_id].dphidz[i]),
			      D_*(basis[eqn_id].duolddx*basis[eqn_id].dphidx[i]
				  + basis[eqn_id].duolddy*basis[eqn_id].dphidy[i]
				  + basis[eqn_id].duolddz*basis[eqn_id].dphidz[i]),
			      D_*(basis[eqn_id].duoldolddx*basis[eqn_id].dphidx[i]
				  + basis[eqn_id].duoldolddy*basis[eqn_id].dphidy[i]
				  + basis[eqn_id].duoldolddz*basis[eqn_id].dphidz[i])};

  const double phit[2] = {.5*(basis[1].uu-basis[1].uuold)/dt_*basis[0].phi[i],
			  .5*(basis[1].uuold-basis[1].uuoldold)/dt_*basis[0].phi[i]};

  double f[3];
  f[0] = -divgradu[0] + phit[0];
  f[1] = -divgradu[1] + phit[1];
  f[2] = -divgradu[2] + phit[1];
//   std::cout<<ut
//     + t_theta_*divgradu[0] + (1. - t_theta_)*divgradu[1]
//     - t_theta_*phit<<std::endl

//   return ut - t_theta_*f[0]
//     - (1.-t_theta_)*f[1];
  return ut - (1.-t_theta2_)*t_theta_*f[0]
    - (1.-t_theta2_)*(1.-t_theta_)*f[1]
    -.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

RES_FUNC(residual_phase_pfhub3_n_)
{
  const double test = basis[eqn_id].phi[i];
  const double dtestdx = basis[eqn_id].dphidx[i];
  const double dtestdy = basis[eqn_id].dphidy[i];
  const double dtestdz = basis[eqn_id].dphidz[i];

  const double phi[3] = {basis[eqn_id].uu,basis[eqn_id].uuold,basis[eqn_id].uuoldold};
  const double dphidx[3] = {basis[eqn_id].dudx,basis[eqn_id].duolddx,basis[eqn_id].duoldolddx};
  const double dphidy[3] = {basis[eqn_id].dudy,basis[eqn_id].duolddy,basis[eqn_id].duoldolddy};
  const double dphidz[3] = {basis[eqn_id].dudz,basis[eqn_id].duolddz,basis[eqn_id].duoldolddz};

  const double as[3] = {a(phi[0],
			  dphidx[0],
			  dphidy[0],
			  dphidz[0],
			  eps_),
			a(phi[1],
			  dphidx[1],
			  dphidy[1],
			  dphidz[1],
			  eps_),
			a(phi[2],
			  dphidx[2],
			  dphidy[2],
			  dphidz[2],
			  eps_)};

  const double tau[3] = {tau0_*as[0]*as[0],tau0_*as[1]*as[1],tau0_*as[2]*as[2]};
//   if(tau[0]!= tau[0]) std::cout<<tau[0]<<" "<<as[0]<<" "
// 			       <<dphidx[0]<<" "<<dphidy[0]<<" "<<dphidz[0]
// 			       <<" "<<phi[0]<<" "<<phi[0]*phi[0]<<std::endl;

  const double phit = (phi[0]-phi[1])/dt_*test;

  const double w[3] = {W_*as[0],W_*as[1],W_*as[2]};

//   const double divgradphi[3] = {w[0]*w[0]*(dphidx[0]*dtestdx
// 					     + dphidy[0]*dtestdy
// 					     + dphidz[0]*dtestdz),
// 				w[1]*w[1]*(dphidx[1]*dtestdx
// 					     + dphidy[1]*dtestdy
// 					     + dphidz[1]*dtestdz),
// 				w[2]*w[2]*(dphidx[2]*dtestdx
// 					     + dphidy[2]*dtestdy
// 					     + dphidz[2]*dtestdz)};

  const double divgradphi[3] = {W_*W_*(dphidx[0]*dtestdx
					     + dphidy[0]*dtestdy
					     + dphidz[0]*dtestdz),
				W_*W_*(dphidx[1]*dtestdx
					     + dphidy[1]*dtestdy
					     + dphidz[1]*dtestdz),
				W_*W_*(dphidx[2]*dtestdx
					     + dphidy[2]*dtestdy
					     + dphidz[2]*dtestdz)};

  const double wp[3] = {W_*(ap(phi[0],dphidx[0],dphidy[0],dphidz[0],dphidx[0],eps_)*dtestdx 
			    + ap(phi[0],dphidx[0],dphidy[0],dphidz[0],dphidy[0],eps_)*dtestdy 
			    + ap(phi[0],dphidx[0],dphidy[0],dphidz[0],dphidz[0],eps_)*dtestdz),
			W_*(ap(phi[1],dphidx[1],dphidy[1],dphidz[1],dphidx[1],eps_)*dtestdx 
			    + ap(phi[1],dphidx[1],dphidy[1],dphidz[1],dphidy[1],eps_)*dtestdy 
			    + ap(phi[1],dphidx[1],dphidy[1],dphidz[1],dphidz[1],eps_)*dtestdz),
			W_*(ap(phi[2],dphidx[2],dphidy[2],dphidz[2],dphidx[2],eps_)*dtestdx 
			    + ap(phi[2],dphidx[2],dphidy[2],dphidz[2],dphidy[2],eps_)*dtestdy 
			    + ap(phi[2],dphidx[2],dphidy[2],dphidz[2],dphidz[2],eps_)*dtestdz)};

  const double curlgrad[3] = {w[0]*(dphidx[0]*dphidx[0] + dphidy[0]*dphidy[0] + dphidz[0]*dphidz[0])*wp[0],
			      w[1]*(dphidx[1]*dphidx[1] + dphidy[1]*dphidy[1] + dphidz[1]*dphidz[1])*wp[1],
			      w[2]*(dphidx[2]*dphidx[2] + dphidy[2]*dphidy[2] + dphidz[2]*dphidz[2])*wp[2]};

  const double g[3] = {((phi[0]-lambda_*basis[0].uu*(1.-phi[0]*phi[0]))*(1.-phi[0]*phi[0]))*test,
		       ((phi[1]-lambda_*basis[0].uuold*(1.-phi[1]*phi[1]))*(1.-phi[1]*phi[1]))*test,
		       ((phi[2]-lambda_*basis[0].uuoldold*(1.-phi[2]*phi[2]))*(1.-phi[2]*phi[2]))*test};

//   if(tau[0]!= tau[0]) std::cout<<tau[0]<<" "<<as[0]<<" "
// 			       <<dphidx[0]<<" "<<dphidy[0]<<" "<<dphidz[0]
// 			       <<" "<<phi[0]<<" "<<phi[0]*phi[0]<<" "<<g[0]<<" "<<divgradphi[0]
// 			       <<" "<<curlgrad[0]<<std::endl;

  double f[3];
  f[0] = -(divgradphi[0]/tau0_+curlgrad[0]/tau[0]-g[0]/tau[0]);
  f[1] = -(divgradphi[1]/tau0_+curlgrad[1]/tau[1]-g[1]/tau[1]);
  f[2] = -(divgradphi[2]/tau0_+curlgrad[2]/tau[2]-g[2]/tau[2]);
//   return phit - t_theta_*f[0]
//     - (1.-t_theta_)*f[1];
  return phit - (1.-t_theta2_)*t_theta_*f[0]
    - (1.-t_theta2_)*(1.-t_theta_)*f[1]
    -.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}
#endif
TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_phase_pfhub3_dp_)) = residual_phase_pfhub3_;

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_heat_pfhub3_)
{
  const double ut = basis[eqn_id].phi(j)/dt_*basis[eqn_id].phi(i);
  const double divgradu = D_*(basis[eqn_id].dphidx(j)*basis[eqn_id].dphidx(i)
			  + basis[eqn_id].dphidy(j)*basis[eqn_id].dphidy(i)
			      + basis[eqn_id].dphidz(j)*basis[eqn_id].dphidz(i));
  return ut + t_theta_*divgradu;
}
#if 0
PRE_FUNC(prec_heat_pfhub3_n_)
{
  const double ut = basis[eqn_id].phi[j]/dt_*basis[eqn_id].phi[i];
  const double divgradu = D_*(basis[eqn_id].dphidx[j]*basis[eqn_id].dphidx[i]
			  + basis[eqn_id].dphidy[j]*basis[eqn_id].dphidy[i]
			      + basis[eqn_id].dphidz[j]*basis[eqn_id].dphidz[i]);
  return ut + t_theta_*divgradu;
}
#endif

TUSAS_DEVICE
PRE_FUNC_TPETRA((*prec_heat_pfhub3_dp_)) = prec_heat_pfhub3_;

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_phase_pfhub3_)
{
  const double test = basis[eqn_id].phi(i);
  const double dtestdx = basis[eqn_id].dphidx(i);
  const double dtestdy = basis[eqn_id].dphidy(i);
  const double dtestdz = basis[eqn_id].dphidz(i);

  const double phi = basis[eqn_id].uu();
  const double phit = basis[eqn_id].phi(j)/dt_*test;
//   const double as = a(phi,
// 		      basis[eqn_id].dudx(),
// 		      basis[eqn_id].dudy,
// 		      basis[eqn_id].dudz,
// 		      eps_);
  const double tau = tau0_;//*as*as;

  const double divgradphi = W_*W_*(basis[eqn_id].dphidx(j)*dtestdx
				   + basis[eqn_id].dphidy(j)*dtestdy
				   + basis[eqn_id].dphidz(j)*dtestdz);

  return phit
    + t_theta_*divgradphi/tau;
}
#if 0
PRE_FUNC(prec_phase_pfhub3_n_)
{
  const double test = basis[eqn_id].phi[i];
  const double dtestdx = basis[eqn_id].dphidx[i];
  const double dtestdy = basis[eqn_id].dphidy[i];
  const double dtestdz = basis[eqn_id].dphidz[i];

  const double phi = basis[eqn_id].uu;
  const double phit = basis[eqn_id].phi[j]/dt_*test;
//   const double as = tpetra::farzadi3d::a(phi,
// 					     basis[eqn_id].dudx,
// 					     basis[eqn_id].dudy,
// 					     basis[eqn_id].dudz,
// 					     eps_);
//   const double tau = tau0_*as*as;
  const double tau = tau0_;
//   const double divgradphi = W_*as*W_*as*(basis[eqn_id].dphidx[j]*dtestdx
// 					 + basis[eqn_id].dphidy[j]*dtestdy
// 					 + basis[eqn_id].dphidz[j]*dtestdz);
  const double divgradphi = W_*W_*(basis[eqn_id].dphidx[j]*dtestdx
					 + basis[eqn_id].dphidy[j]*dtestdy
					 + basis[eqn_id].dphidz[j]*dtestdz);
  return phit
    + t_theta_*divgradphi/tau;
}
#endif
TUSAS_DEVICE
PRE_FUNC_TPETRA((*prec_phase_pfhub3_dp_)) = prec_phase_pfhub3_;

INI_FUNC(init_heat_pfhub3_)
{
  return delta_;
}

INI_FUNC(init_phase_pfhub3_)
{
  double val = -1.;
  const double r = sqrt(x*x+y*y+z*z);
  //if(x*x+y*y+z*z < R_*R_) val = 1.;
  //see https://aip.scitation.org/doi/pdf/10.1063/1.5142353
  //we should have a general function for this
  //val = tanh((R_-r)/(sqrt(8.)*W_));
  val = tanh((R_-r)/(sqrt(2.)*W_));


  //should probably be:
  //val = -tanh( (x*x+y*y+z*z - R_*R_)/(sqrt(2.)*W_) );
  return val;
}

}//namespace pfhub3

namespace pfhub2
{
  TUSAS_DEVICE
  const int N_MAX = 1;
  TUSAS_DEVICE
  int N_ = 1;
  TUSAS_DEVICE
  int eqn_off_ = 1;
  TUSAS_DEVICE
  const double c0_ = .5;
  TUSAS_DEVICE
  const double eps_ = .05;
  TUSAS_DEVICE
  const double eps_eta_ = .1;
  TUSAS_DEVICE
  const double psi_ = 1.5;
  TUSAS_DEVICE
  const double rho_ = 1.414213562373095;//std::sqrt(2.);
  TUSAS_DEVICE
  const double c_alpha_ = .3;
  TUSAS_DEVICE
  const double c_beta_ = .7;
  TUSAS_DEVICE
  const double alpha_ = 5.;
  TUSAS_DEVICE
  const double k_c_ = 3.;
  TUSAS_DEVICE
  const double k_eta_ = 3.;
  TUSAS_DEVICE
  const double M_ = 5.;
  TUSAS_DEVICE
  const double L_ = 5.;
  TUSAS_DEVICE
  const double w_ = 1.;
//   double c_a[2] = {0., 0.};
//   double c_b[2] = {0., 0.};

  PARAM_FUNC(param_)
  {
    int N_p = plist->get<int>("N");
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(N_,&N_p,sizeof(int));
#else
    N_ = N_p;
#endif
    int eqn_off_p = plist->get<int>("OFFSET");
#if defined(TUSAS_HAVE_CUDA)
  cudaMemcpyToSymbol(eqn_off_,&eqn_off_p,sizeof(int));
#else
    eqn_off_ = eqn_off_p;
#endif
  }
 
KOKKOS_INLINE_FUNCTION 
  double dhdeta(const double eta)
  {
    //return 30.*eta[eqn_id]*eta[eqn_id] - 60.*eta[eqn_id]*eta[eqn_id]*eta[eqn_id] + 30.*eta[eqn_id]*eta[eqn_id]*eta[eqn_id]*eta[eqn_id];
    return 30.*eta*eta - 60.*eta*eta*eta + 30.*eta*eta*eta*eta;
  }
 
KOKKOS_INLINE_FUNCTION 
  double h(const double *eta)
  {
    double val = 0.;
    for (int i = 0; i < N_; i++){
      val += eta[i]*eta[i]*eta[i]*(6.*eta[i]*eta[i] - 15.*eta[i] + 10.);
    }
    return val;
  }
 
KOKKOS_INLINE_FUNCTION 
  double d2fdc2()
  {
    return 2.*rho_*rho_;
  }
 
KOKKOS_INLINE_FUNCTION 
  double df_alphadc(const double c)
  {
    return 2.*rho_*rho_*(c - c_alpha_);
  }
 
KOKKOS_INLINE_FUNCTION 
  double df_betadc(const double c)
  {
    return -2.*rho_*rho_*(c_beta_ - c);
  }
 
KOKKOS_INLINE_FUNCTION 
  void solve_kks(const double c, double *phi, double &ca, double &cb)//const double phi
  {
    double delta_c_a = 0.;
    double delta_c_b = 0.;
    const int max_iter = 20;
    const double tol = 1.e-8;
    const double hh = h(phi);
    //c_a[0] = (1.-hh)*c;
    ca = c - hh*(c_beta_ - c_alpha_);

    //c_b[0]=hh*c;
    cb = c - (1.-hh)*(c_beta_ - c_alpha_);

    //std::cout<<"-1"<<" "<<delta_c_b<<" "<<delta_c_a<<" "<<c_b[0]<<" "<<c_a[0]<<" "<<hh*c_b[0] + (1.- hh)*c_a[0]<<" "<<c<<std::endl;
    for(int i = 0; i < max_iter; i++){
      const double det = hh*d2fdc2() + (1.-hh)*d2fdc2();
      const double f1 = hh*cb + (1.- hh)*ca - c;
      const double f2 = df_betadc(cb) - df_alphadc(ca);
      delta_c_b = (-d2fdc2()*f1 - (1-hh)*f2)/det;
      delta_c_a = (-d2fdc2()*f1 + hh*f2)/det;
      cb = delta_c_b + cb;
      ca = delta_c_a + ca;
      //std::cout<<i<<" "<<delta_c_b<<" "<<delta_c_a<<" "<<c_b[0]<<" "<<c_a[0]<<" "<<hh*c_b[0] + (1.- hh)*c_a[0]<<" "<<c<<std::endl;
      if(delta_c_a*delta_c_a+delta_c_b*delta_c_b < tol*tol) return;
    }
//     std::cout<<"###################################  solve_kks falied to converge with delta_c_a*delta_c_a+delta_c_b*delta_c_b = "
// 	     <<delta_c_a*delta_c_a+delta_c_b*delta_c_b<<"  ###################################"<<std::endl;
    exit(0);
    return;
  }
 
KOKKOS_INLINE_FUNCTION 
  double f_alpha(const double c)
  {
    return rho_*rho_*(c - c_alpha_)*(c - c_alpha_);
  }
 
KOKKOS_INLINE_FUNCTION 
  double f_beta(const double c)
  {
    return rho_*rho_*(c_beta_ - c)*(c_beta_ - c);
  }
 
KOKKOS_INLINE_FUNCTION 
  double dgdeta(const double *eta, const int eqn_id)
  {

    double aval =0.;
    for (int i = 0; i < N_; i++){
      aval += eta[i]*eta[i];
    }
    return 2.*eta[eqn_id]*(1. - eta[eqn_id])*(1. - eta[eqn_id])  
      - 2.* eta[eqn_id]* eta[eqn_id]* (1. - eta[eqn_id]) + 
      4.*alpha_*eta[eqn_id] *aval - 4.*alpha_*eta[eqn_id]*eta[eqn_id]*eta[eqn_id];
  }

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_c_kks_)
{
  //derivatives of the test function
  const double dtestdx = basis[0].dphidx(i);
  const double dtestdy = basis[0].dphidy(i);
  //double dtestdz = basis[0].dphidz(i);
  //test function
  const double test = basis[0].phi(i);
  //u, phi
  const double c[2] = {basis[0].uu(), basis[0].uuold()};
  const double dcdx[2] = {basis[0].dudx(), basis[0].duolddx()};
  const double dcdy[2] = {basis[0].dudy(), basis[0].duolddy()};

  double dhdx[2] = {0., 0.};
  double dhdy[2] = {0., 0.};

  double c_a[2] = {0., 0.};
  double c_b[2] = {0., 0.};

  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    dhdx[0] += dhdeta(basis[kk_off].uu())*basis[kk_off].dudx();
    dhdx[1] += dhdeta(basis[kk_off].uuold())*basis[kk_off].duolddx();
    dhdy[0] += dhdeta(basis[kk_off].uu())*basis[kk_off].dudy();
    dhdy[1] += dhdeta(basis[kk_off].uuold())*basis[kk_off].duolddy();
  }

  const double ct = (c[0]-c[1])/dt_*test;

  double eta_array[N_MAX];
  double eta_array_old[N_MAX];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    eta_array[kk] = basis[kk_off].uu();
    eta_array_old[kk] = basis[kk_off].uuold();
  }

  solve_kks(c[0],eta_array,c_a[0],c_b[0]);

  const double DfDc[2] = {-c_b[0] + c_a[0],
 		    -c_b[1] + c_a[1]};

  const double D2fDc2 = 1.;
  //double D2fDc2 = 1.*d2fdc2();

  const double dfdx[2] = {DfDc[0]*dhdx[0] + D2fDc2*dcdx[0],
		    DfDc[1]*dhdx[1] + D2fDc2*dcdx[1]};
  const double dfdy[2] = {DfDc[0]*dhdy[0] + D2fDc2*dcdy[0],
		    DfDc[1]*dhdy[1] + D2fDc2*dcdy[1]};

  const double divgradc[2] = {M_*(dfdx[0]*dtestdx + dfdy[0]*dtestdy),
			M_*(dfdx[1]*dtestdx + dfdy[1]*dtestdy)};

  return ct + t_theta_*divgradc[0] + (1.-t_theta_)*divgradc[1];
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_c_kks_dp_)) = residual_c_kks_;

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_eta_kks_)
{

  //derivatives of the test function
  const double dtestdx = basis[eqn_id].dphidx(i);
  const double dtestdy = basis[eqn_id].dphidy(i);
  //double dtestdz = basis[0].dphidz(i);
  //test function
  const double test = basis[eqn_id].phi(i);
  //u, phi
  const double c[2] = {basis[0].uu(), basis[0].uuold()};

  const double eta[2] = {basis[eqn_id].uu(), basis[eqn_id].uuold()};
  const double detadx[2] = {basis[eqn_id].dudx(), basis[eqn_id].duolddx()};
  const double detady[2] = {basis[eqn_id].dudy(), basis[eqn_id].duolddy()};

  double c_a[2] = {0., 0.};
  double c_b[2] = {0., 0.};

  double eta_array[N_MAX];
  double eta_array_old[N_MAX];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    eta_array[kk] = basis[kk_off].uu();
    eta_array_old[kk] = basis[kk_off].uuold();
  }

  solve_kks(c[0],eta_array,c_a[0],c_b[0]);

  const double etat = (eta[0]-eta[1])/dt_*test;


  const double F[2] = {f_beta(c_b[0]) - f_alpha(c_a[0]) 
		 - (c_b[0] - c_a[0])*df_betadc(c_b[0]),
		 f_beta(c_b[1]) - f_alpha(c_a[1]) 
		 - (c_b[1] - c_a[1])*df_betadc(c_b[1])};

  const int k = eqn_id - eqn_off_;
  const double dfdeta[2] = {L_*(F[0]*dhdeta(eta[0]) 
				    + w_*dgdeta(eta_array,k)    )*test,
		      L_*(F[1]*dhdeta(eta[1]) 
				    + w_*dgdeta(eta_array_old,k))*test};

  const double divgradeta[2] = {L_*k_eta_
			  *(detadx[0]*dtestdx + detady[0]*dtestdy), 
			  L_*k_eta_
			  *(detadx[1]*dtestdx + detady[1]*dtestdy)};//(grad u,grad phi)
 
  return etat + t_theta_*divgradeta[0] + t_theta_*dfdeta[0] + (1.-t_theta_)*divgradeta[1] + (1.-t_theta_)*dfdeta[1];
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_eta_kks_dp_)) = residual_eta_kks_;

PPR_FUNC(postproc_c_a_)
{

  //cn will need eta_array here...
  const double cc = u[0];
  double phi = u[1];
  double c_a = 0.;
  double c_b = 0.;

  solve_kks(cc,&phi,c_a,c_b);

  return c_a;
}

PPR_FUNC(postproc_c_b_)
{
  //cn will need eta_array here...
  const double cc = u[0];
  double phi = u[1];
  double c_a = 0.;
  double c_b = 0.;

  solve_kks(cc,&phi,c_a,c_b);

  return c_b;
}
}//namespace pfhub2

namespace robin
{
  //  http://ramanujan.math.trinity.edu/rdaileda/teach/s12/m3357/lectures/lecture_2_28_short.pdf
  // 1-D robin bc test problem, time dependent
  // Solve D[u, t] - c^2 D[u, x, x] == 0
  // u(0,t) == 0
  // D[u, x] /. x -> L == -kappa u(t,L)
  // => du/dx + kappa u = g = 0
  // u(x,t) = a E^(-mu^2 t) Sin[mu x]
  // mu solution to: Tan[mu L] + mu/kappa == 0 && Pi/2 < mu < 3 Pi/2
  const double mu = 2.028757838110434;
  const double a = 10.;
  const double c = 1.;
  const double L = 1.;
  const double kappa = 1.;

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_robin_test_)
{
  //1-D robin bc test problem, 

  //derivatives of the test function
  const double dtestdx = basis[0].dphidx(i);
  const double dtestdy = basis[0].dphidy(i);
  const double dtestdz = basis[0].dphidz(i);
  //test function
  const double test = basis[0].phi(i);
  //u, phi
  const double u = basis[0].uu();
  const double uold = basis[0].uuold();

  //double a =10.;

  const double ut = (u-uold)/dt_*test;
  const double divgradu = c*c*(basis[0].dudx()*dtestdx + basis[0].dudy()*dtestdy + basis[0].dudz()*dtestdz);//(grad u,grad phi)
  //double divgradu_old = (basis[0].duolddx()*dtestdx + basis[0].duolddy()*dtestdy + basis[0].dolddz()*dtestdz);//(grad u,grad phi)
 
  const double f[3] = {c*c*(basis[0].dudx()*dtestdx + basis[0].dudy()*dtestdy + basis[0].dudz()*dtestdz),
		       c*c*(basis[0].duolddx()*dtestdx + basis[0].duolddy()*dtestdy + basis[0].duolddz()*dtestdz),
		       c*c*(basis[0].duoldolddx()*dtestdx + basis[0].duoldolddy()*dtestdy + basis[0].duoldolddz()*dtestdz)};
  return ut + (1.-t_theta2_)*t_theta_*f[0]
    + (1.-t_theta2_)*(1.-t_theta_)*f[1]
    +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_robin_test_dp_)) = residual_robin_test_;

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_robin_test_)
{
  //cn probably want to move each of these operations inside of getbasis
  //derivatives of the test function
  const double dtestdx = basis[0].dphidx(i);
  const double dtestdy = basis[0].dphidy(i);
  const double dtestdz = basis[0].dphidz(i);

  const double dbasisdx = basis[0].dphidx(j);
  const double dbasisdy = basis[0].dphidy(j);
  const double dbasisdz = basis[0].dphidz(j);
  const double test = basis[0].phi(i);
  const double divgrad = c*c*(dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz);
  //double a =10.;
  const double u_t = (basis[0].phi(j))/dt_*test;
  return u_t + t_theta_*divgrad;
}

TUSAS_DEVICE
PRE_FUNC_TPETRA((*prec_robin_test_dp_)) = prec_robin_test_;

NBC_FUNC_TPETRA(nbc_robin_test_)
{

  const double test = basis[0].phi[i];

  //du/dn + kappa u = g = 0 on L
  //(du,dv) - <du/dn,v> = (f,v)
  //(du,dv) - <g - kappa u,v> = (f,v)
  //(du,dv) - < - kappa u,v> = (f,v)
  //          ^^^^^^^^^^^^^^ return this
  const double f[3] = {-kappa*basis[0].uu*test,
		       -kappa*basis[0].uuold*test,
		       -kappa*basis[0].uuoldold*test};
  return (1.-t_theta2_)*t_theta_*f[0]
    +(1.-t_theta2_)*(1.-t_theta_)*f[1]
    +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}
INI_FUNC(init_robin_test_)
{
  return a*sin(mu*x);
}
PPR_FUNC(postproc_robin_)
{
  const double uu = u[0];
  const double x = xyz[0];
  //const double y = xyz[1];
  //const double z = xyz[2];

  const double s= a*exp(-mu*mu*time)*sin(mu*x);//c?

  return s-uu;
}
}//namespace robin

namespace autocatalytic4
{

  //https://documen.site/download/math-3795-lecture-18-numerical-solution-of-ordinary-differential-equations-goals_pdf#
  //https://media.gradebuddy.com/documents/2449908/0c88cf76-7605-4aec-b2ad-513ddbebefec.pdf

const double k1 = .0001;
const double k2 = 1.;
const double k3 = .0008;

RES_FUNC_TPETRA(residual_a_)
{
  const double test = basis[0].phi(i);
  //u, phi
  const double u = basis[0].uu();
  const double uold = basis[0].uuold();
  const double uoldold = basis[0].uuoldold();

  const double ut = (u-uold)/dt_*test;
  //std::cout<<ut<<" "<<dt_<<" "<<time<<std::endl;
 
  double f[3];
  f[0] = (-k1*u       - k2*u*basis[1].uu())*test;
  f[1] = (-k1*uold    - k2*u*basis[1].uuold())*test;
  f[2] = (-k1*uoldold - k2*u*basis[1].uuoldold())*test;

  return ut - (1.-t_theta2_)*t_theta_*f[0]
    - (1.-t_theta2_)*(1.-t_theta_)*f[1]
    -.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

RES_FUNC_TPETRA(residual_b_)
{
  const double test = basis[1].phi(i);
  //u, phi
  const double u = basis[1].uu();
  const double uold = basis[1].uuold();
  //const double uoldold = basis[1].uuoldold;
  const double a = basis[0].uu();
  const double aold = basis[0].uuold();
  const double aoldold = basis[0].uuoldold();

  const double ut = (u-uold)/dt_*test;
  double f[3];
  f[0] = (k1*a       - k2*a*u                        + 2.*k3*basis[2].uu())*test;
  f[1] = (k1*aold    - k2*aold*uold                  + 2.*k3*basis[2].uuold())*test;
  f[2] = (k1*aoldold - k2*aoldold*basis[1].uuoldold() + 2.*k3*basis[2].uuoldold())*test;

  return ut - (1.-t_theta2_)*t_theta_*f[0]
    - (1.-t_theta2_)*(1.-t_theta_)*f[1]
    -.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

RES_FUNC_TPETRA(residual_ab_)
{
  const double test = basis[1].phi(i);
  //u, phi
  const double u = basis[2].uu();
  const double uold = basis[2].uuold();
  //const double uoldold = basis[1].uuoldold;
  const double b = basis[1].uu();
  const double bold = basis[1].uuold();
  const double boldold = basis[1].uuoldold();

  const double ut = (u-uold)/dt_*test;
  double f[3];
  f[0] = (k2*b*basis[0].uu()             - k3*u)*test;
  f[1] = (k2*bold*basis[0].uuold()       - k3*uold)*test;
  f[2] = (k2*boldold*basis[0].uuoldold() - k3*basis[2].uuoldold())*test;

  return ut - (1.-t_theta2_)*t_theta_*f[0]
    - (1.-t_theta2_)*(1.-t_theta_)*f[1]
    -.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

RES_FUNC_TPETRA(residual_c_)
{
  const double test = basis[1].phi(i);
  //u, phi
  const double u = basis[3].uu();
  const double uold = basis[3].uuold();
  //const double uoldold = basis[1].uuoldold;
  const double a = basis[0].uu();
  const double aold = basis[0].uuold();
  const double aoldold = basis[0].uuoldold();

  const double ut = (u-uold)/dt_*test;
  double f[3];
  f[0] = (k1*a       + k3*basis[2].uu())*test;
  f[1] = (k1*aold    + k3*basis[2].uuold())*test;
  f[2] = (k1*aoldold + k3*basis[2].uuoldold())*test;

  return ut - (1.-t_theta2_)*t_theta_*f[0]
    - (1.-t_theta2_)*(1.-t_theta_)*f[1]
    -.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

}//namespace autocatalytic4

namespace timeonly
{
const double pi = 3.141592653589793;
  //const double lambda = 10.;//pi*pi;
const double lambda = pi*pi;

const double ff(const double &u)
{
  return -lambda*u;
}

RES_FUNC_TPETRA(residual_test_)
{
  //test function
  const double test = basis[0].phi(i);
  //u, phi
  const double u[3] = {basis[0].uu(),basis[0].uuold(),basis[0].uuoldold()};

  const double ut = (u[0]-u[1])/dt_*test;

  const double f[3] = {ff(u[0])*test,ff(u[1])*test,ff(u[2])*test};
  //std::cout<<u[1]<<"  "<<u[2]<<std::endl;
  return ut - (1.-t_theta2_)*t_theta_*f[0]
    - (1.-t_theta2_)*(1.-t_theta_)*f[1]
    -.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}
}//namespace timeonly

namespace radconvbc
{
  double h = 50.;
  double ep = .7;
  double sigma = 5.67037e-9;
  double ti = 323.;

DBC_FUNC(dbc_) 
{
  return 1173.;
}

NBC_FUNC_TPETRA(nbc_)
{
  //https://reference.wolfram.com/language/PDEModels/tutorial/HeatTransfer/HeatTransfer.html#2048120463
  //h(t-ti)+\ep\sigma(t^4-ti^4)
  const double test = basis[0].phi[i];
  const double u = basis[0].uu;
  const double uold = basis[0].uuold;
  const double uoldold = basis[0].uuoldold;
  const double f[3] = {(h*(ti-u)+ep*sigma*(ti*ti*ti*ti-u*u*u*u))*test,
		       (h*(ti-uold)+ep*sigma*(ti*ti*ti*ti-uold*uold*uold*uold))*test,
		       (h*(ti-uoldold)+ep*sigma*(ti*ti*ti*ti-uoldold*uoldold*uoldold*uoldold))*test};
  return (1.-t_theta2_)*t_theta_*f[0]
    +(1.-t_theta2_)*(1.-t_theta_)*f[1]
    +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

INI_FUNC(init_heat_)
{
  return 1173.;
}

PARAM_FUNC(param_)
{
  h = plist->get<double>("h_",50.);
  ep = plist->get<double>("ep_",.7);
  sigma = plist->get<double>("sigma_",5.67037e-9);
  ti = plist->get<double>("ti_",323.);
}
}//namespace radconvbc
#endif


namespace goldak{
TUSAS_DEVICE
const double pi_d = 3.141592653589793;
const double pi_h = 3.141592653589793;

TUSAS_DEVICE
double te_d = 1641.;
double te = 1641.;
TUSAS_DEVICE
double tl_d = 1706.;
double tl = 1706.;
TUSAS_DEVICE
double Lf_d = 2.95e5;
double Lf = 2.95e5;
TUSAS_DEVICE
double dfldu_mushy_d = 0.0;//fl=(t-te)/(tl-te);
double dfldu_mushy_h = 0.0;


TUSAS_DEVICE
double eta_d = 0.3;
double eta_h = 0.3;
TUSAS_DEVICE
double P_d = 50.;
double P_h = 50.;
TUSAS_DEVICE
double s_d = 2.;
double s_h = 2.;
TUSAS_DEVICE
double r_d = .00005;
double r_h = .00005;
TUSAS_DEVICE
double d_d = .00001;
double d_h = .00001;
TUSAS_DEVICE
double gamma_d = 0.886227;
double gamma_h = 0.886227;
TUSAS_DEVICE
double x0_d = 0.;
double x0_h = 0.;
TUSAS_DEVICE
double y0_d = 0.;
double y0_h = 0.;
TUSAS_DEVICE
double z0_d = 0.0005;
double z0_h = 0.0005;
TUSAS_DEVICE
double t_hold_d = 0.005;
double t_hold_h = 0.005;
TUSAS_DEVICE
double t_decay_d = 0.01;
double t_decay_h = 0.01;

TUSAS_DEVICE
double tau0_d = 1.;
double tau0_h = 1.;

TUSAS_DEVICE
double W0_d = 1.;
double W0_h = 1.;

TUSAS_DEVICE
double uref_d = 1693.4;
double uref_h = 1693.4;

TUSAS_DEVICE
double t0_d = 300.;
double t0_h = 300.;
TUSAS_DEVICE
double scaling_constant_d = 1.;

#if 0
KOKKOS_INLINE_FUNCTION 
void dfldt_uncoupled(GPUBasisLHex* basis, const int index, const double dt_, const double dtold_, double *a)
{
  //the latent heat term is zero outside of the mushy region (ie outside Te < T < Tl)

  const double coef = 1./tau0_d;

  const double tt[3] = {tpetra::heat::deltau_d*basis[index].uu()+uref_d,
			tpetra::heat::deltau_d*basis[index].uuold()+uref_d,
			tpetra::heat::deltau_d*basis[index].uuoldold()+uref_d};
  const double dfldu_d[3] = {((tt[0] > te_d) && (tt[0] < tl_d)) ? coef*dfldu_mushy_d : 0.0,
			     ((tt[1] > te_d) && (tt[1] < tl_d)) ? coef*dfldu_mushy_d : 0.0,
			     ((tt[2] > te_d) && (tt[2] < tl_d)) ? coef*dfldu_mushy_d : 0.0};

  a[0] = ((1. + dt_/dtold_)*(dfldu_d[0]*basis[index].uu()-dfldu_d[1]*basis[index].uuold())/dt_
                                 -dt_/dtold_*(dfldu_d[0]*basis[index].uu()-dfldu_d[2]*basis[index].uuoldold())/(dt_+dtold_)
                                 );
  a[1] = (dtold_/dt_/(dt_+dtold_)*(dfldu_d[0]*basis[index].uu())
                                 -(dtold_-dt_)/dt_/dtold_*(dfldu_d[1]*basis[index].uuold())
                                 -dt_/dtold_/(dt_+dtold_)*(dfldu_d[2]*basis[index].uuoldold())
                                 );
  a[2] = (-(1.+dtold_/dt_)*(dfldu_d[2]*basis[index].uuoldold()-dfldu_d[1]*basis[index].uuold())/dtold_
                                 +dtold_/dt_*(dfldu_d[2]*basis[index].uuoldold()-dfldu_d[0]*basis[index].uu())/(dtold_+dt_)
                                 );
  return;
}

KOKKOS_INLINE_FUNCTION 
void dfldt_coupled(GPUBasisLHex* basis, const int index, const double dt_, const double dtold_, double *a)
{
  const double coef = tpetra::heat::rho_d*Lf_d/tau0_d;
  const double dfldu_d[3] = {-.5*coef,-.5*coef,-.5*coef};

  a[0] = ((1. + dt_/dtold_)*(dfldu_d[0]*basis[index].uu()-dfldu_d[1]*basis[index].uuold())/dt_
                                 -dt_/dtold_*(dfldu_d[0]*basis[index].uu()-dfldu_d[2]*basis[index].uuoldold())/(dt_+dtold_)
                                 );
  a[1] = (dtold_/dt_/(dt_+dtold_)*(dfldu_d[0]*basis[index].uu())
                                 -(dtold_-dt_)/dt_/dtold_*(dfldu_d[1]*basis[index].uuold())
                                 -dt_/dtold_/(dt_+dtold_)*(dfldu_d[2]*basis[index].uuoldold())
                                 );
  a[2] = (-(1.+dtold_/dt_)*(dfldu_d[2]*basis[index].uuoldold()-dfldu_d[1]*basis[index].uuold())/dtold_
                                 +dtold_/dt_*(dfldu_d[2]*basis[index].uuoldold()-dfldu_d[0]*basis[index].uu())/(dtold_+dt_)
                                 );
  return;
}

KOKKOS_INLINE_FUNCTION
const double power(const double t)
{
  // t is nondimensional
  // t_hold, t_decay, and tt are dimensional
  
  const double t_hold = t_hold_d;
  const double t_decay = t_decay_d;
  const double tt = t*tau0_d;
  return (tt < t_hold) ? P_d : 
    ((tt<t_hold+t_decay) ? P_d*((t_hold+t_decay)-tt)/(t_decay)
     :0.);
}

KOKKOS_INLINE_FUNCTION 
const double qdot(const double &x, const double &y, const double &z, const double &t)
{
  // x, y, z, and t are nondimensional values
  // r, d, and p and dimensional
  // Qdot as a whole has dimensions, but that's ok since it's written in terms of non-dimensional (x,y,z,t)

  const double p = power(t);
  const double r = r_d;
  const double d = d_d;
  
  //s_d = 2 below; we can simplify this expression 5.19615=3^1.5
  const double coef = eta_d*p*5.19615/r/r/d/gamma_d/pi_d;
  const double exparg = ((W0_d*x-x0_d)*(W0_d*x-x0_d)+(W0_d*y-y0_d)*(W0_d*y-y0_d))/r/r+(W0_d*z-z0_d)*(W0_d*z-z0_d)/d/d;
  const double f = exp( -3.* exparg );

  return coef*f;
}

KOKKOS_INLINE_FUNCTION 
const double power_h(const double t)
{
  // t is nondimensional
  // t_hold, t_decay, and tt are dimensional
  
  const double t_hold = t_hold_h;
  const double t_decay = t_decay_h;
  const double tt = t*tau0_h;
  return (tt < t_hold) ? P_h : 
    ((tt<t_hold+t_decay) ? P_h*((t_hold+t_decay)-tt)/(t_decay)
     :0.);
}

KOKKOS_INLINE_FUNCTION 
const double qdot_h(const double &x, const double &y, const double &z, const double &t)
{
  // x, y, z, and t are nondimensional values
  // r, d, and p and dimensional
  // Qdot as a whole has dimensions, but that's ok since it's written in terms of non-dimensional (x,y,z,t)

  const double p = power_h(t);
  const double r = r_h;
  const double d = d_h;
  
  //s_d = 2 below; we can simplify this expression 5.19615=3^1.5
  const double coef = eta_h*p*5.19615/r/r/d/gamma_h/pi_h;
  const double exparg = ((W0_h*x-x0_h)*(W0_d*x-x0_h)+(W0_h*y-y0_h)*(W0_d*y-y0_h))/r/r+(W0_h*z-z0_h)*(W0_h*z-z0_h)/d/d;
  const double f = exp( -3.* exparg );
  
  return coef*f;
}

KOKKOS_INLINE_FUNCTION
RES_FUNC_TPETRA(residual_test_)
{
  //u_t,v + grad u,grad v + dfldt,v - qdot,v = 0

  double val = tpetra::heat::residual_heat_test_(basis,
						    i,
						    dt_,
						    dtold_,
						    t_theta_,
						    t_theta2_,
						    time,
						 eqn_id,
						 vol,
						 rand); 

  const double qd[3] = {-qdot(basis[eqn_id].xx(),basis[eqn_id].yy(),basis[eqn_id].zz(),time)*basis[eqn_id].phi(i),
			-qdot(basis[eqn_id].xx(),basis[eqn_id].yy(),basis[eqn_id].zz(),time-dt_)*basis[eqn_id].phi(i),
			-qdot(basis[eqn_id].xx(),basis[eqn_id].yy(),basis[eqn_id].zz(),time-dt_-dtold_)*basis[eqn_id].phi(i)};

  const double rv = (val 
		     + (1.-t_theta2_)*t_theta_*qd[0]
		     + (1.-t_theta2_)*(1.-t_theta_)*qd[1]
		     +.5*t_theta2_*((2.+dt_/dtold_)*qd[1]-dt_/dtold_*qd[2]));

  return rv;
	// SJD: This is different enough than the 'master' branch that I'm keeping it (but commented out)
	/*						
//   printf("%f \n",dfldt_d);
//   exit(0);
  //better 3pt derivatives, see difference.nb and inspiration at
  //https://link.springer.com/content/pdf/10.1007/BF02510406.pdf
  const double ut[3] = {dfldt_d*((1. + dt_/dtold_)*(basis[0].uu()-basis[0].uuold())/dt_
				 -dt_/dtold_*(basis[0].uu()-basis[0].uuoldold())/(dt_+dtold_)
				 )*basis[eqn_id].phi(i),
			dfldt_d*(dtold_/dt_/(dt_+dtold_)*(basis[0].uu())
				 -(dtold_-dt_)/dt_/dtold_*(basis[0].uuold())
				 -dt_/dtold_/(dt_+dtold_)*(basis[0].uuoldold())
				 )*basis[eqn_id].phi(i),
			dfldt_d*(-(1.+dtold_/dt_)*(basis[0].uuoldold()-basis[0].uuold())/dtold_
				 +dtold_/dt_*(basis[0].uuoldold()-basis[0].uu())/(dtold_+dt_)
				 )*basis[eqn_id].phi(i)};
//   const double ut[3] = {0.,0.,0.};

  const double qd[3] = {-qdot(basis[0].xx(),basis[0].yy(),basis[0].zz(),time)*basis[eqn_id].phi(i),
			-qdot(basis[0].xx(),basis[0].yy(),basis[0].zz(),time-dt_)*basis[eqn_id].phi(i),
			-qdot(basis[0].xx(),basis[0].yy(),basis[0].zz(),time-dt_-dtold_)*basis[eqn_id].phi(i)};
//   const double qd[3] = { 0.,0.,0.};

  const double rv = (val 
		     + (1.-t_theta2_)*t_theta_*qd[0]
		     + (1.-t_theta2_)*(1.-t_theta_)*qd[1]
		     +.5*t_theta2_*((2.+dt_/dtold_)*qd[1]-dt_/dtold_*qd[2])
		     + (1.-t_theta2_)*t_theta_*ut[0]
		     + (1.-t_theta2_)*(1.-t_theta_)*ut[1]
		     +.5*t_theta2_*((2.+dt_/dtold_)*ut[1]-dt_/dtold_*ut[2]));
  //const double d =tpetra::heat::rho_d*tpetra::heat::cp_d;
  return rv;///d;
  */
}

KOKKOS_INLINE_FUNCTION
RES_FUNC_TPETRA(residual_qdot_)
{
  return (basis[eqn_id].uu()-qdot(basis[eqn_id].xx(),basis[eqn_id].yy(),basis[eqn_id].zz(),time))*basis[eqn_id].phi(i);
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_test_dp_)) = residual_test_;

KOKKOS_INLINE_FUNCTION
RES_FUNC_TPETRA(residual_uncoupled_test_)
{
  //u_t,v + grad u,grad v + dfldt,v - qdot,v = 0

  double val = tpetra::goldak::residual_test_(basis,
						 i,
						 dt_,
						 dtold_,
						 t_theta_,
						 t_theta2_,
						 time,
					      eqn_id,
					      vol,
					      rand);

  double dfldu_d[3];
  dfldt_uncoupled(basis,eqn_id,dt_,dtold_,dfldu_d);

  const double dfldt[3] = {dfldu_d[0]*basis[eqn_id].phi(i),
			   dfldu_d[1]*basis[eqn_id].phi(i),
			   dfldu_d[2]*basis[eqn_id].phi(i)};
  
  const double rv = (val 
		     + (1.-t_theta2_)*t_theta_*dfldt[0]
		     + (1.-t_theta2_)*(1.-t_theta_)*dfldt[1]
		     +.5*t_theta2_*((2.+dt_/dtold_)*dfldt[1]-dt_/dtold_*dfldt[2]));
  
  //cn hack  
  //scaling_constant_d = 1./basis[eqn_id].uuold();
  return rv * scaling_constant_d;
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_uncoupled_test_dp_)) = residual_uncoupled_test_;

KOKKOS_INLINE_FUNCTION
RES_FUNC_TPETRA(residual_coupled_test_)
{
  //u_t,v + grad u,grad v + dfldt,v - qdot,v = 0

  double val = tpetra::goldak::residual_test_(basis,
						 i,
						 dt_,
						 dtold_,
						 t_theta_,
						 t_theta2_,
						 time,
					      eqn_id,
					      vol,
					      rand);

  int phi_index = 1;
  double dfldu_d[3];
  dfldt_coupled(basis,phi_index,dt_,dtold_,dfldu_d);

  const double dfldt[3] = {dfldu_d[0]*basis[eqn_id].phi(i),
			   dfldu_d[1]*basis[eqn_id].phi(i),
			   dfldu_d[2]*basis[eqn_id].phi(i)};
  
  const double rv = (val 
		     + (1.-t_theta2_)*t_theta_*dfldt[0]
		     + (1.-t_theta2_)*(1.-t_theta_)*dfldt[1]
		     +.5*t_theta2_*((2.+dt_/dtold_)*dfldt[1]-dt_/dtold_*dfldt[2]));
  
  //cn hack  
  //scaling_constant_d = 1./basis[eqn_id].uuold();
  return rv * scaling_constant_d;
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_coupled_test_dp_)) = residual_coupled_test_;

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_test_)
{
  
  const double val = tpetra::heat::prec_heat_test_(basis,
						      i,
						      j,
						      dt_,
						      t_theta_,
						      eqn_id);

  //cn hack  
  //scaling_constant_d = 1./basis[eqn_id].uuold();
  return val * scaling_constant_d;
}

INI_FUNC(init_heat_)
{
	const double t_preheat = t0_h;
    const double val = (t_preheat-uref_h)/tpetra::heat::deltau_h;
    return val;
}

DBC_FUNC(dbc_) 
{
	// The assumption here is that the desired Dirichlet BC is the initial temperature,
    // that may not be true in the future.
    const double t_preheat = t0_h;
    const double val = (t_preheat-uref_h)/tpetra::heat::deltau_h;
    return val;
}

PPR_FUNC(postproc_qdot_)
{
  //const double uu = u[0];
  const double x = xyz[0];
  const double y = xyz[1];
  const double z = xyz[2];

  return qdot_h(x,y,z,time);
}

PPR_FUNC(postproc_u_)
{
  return u[0]*tpetra::heat::deltau_h + uref_h;
}

PARAM_FUNC(param_)
{

  //std::cout<<"tpetra::goldak::param_"<<std::endl;
  //we need to set h, ep, sigma, ti in radconv params as follows:
  //h = 100 W/(m2*K)
  //ep = .3
  //sigma = 5.6704 x 10-5 g s^-3 K^-4
  //ti = 300 K

  //we need to set rho_*, k_* and cp_* in heat params 
  //and also *maybe* figure out a way to distinguish between k_lig and k_sol
  //when phasefield is coupled
  //rho_* = 8.9 g/cm^3
  //kliq = 90 W/(m*K)
  //ksol = 90 W/(m*K)
  //cpliq = 0.44 J/(g*K)
  //cpsol = 0.44 J/(g*K)

  //here we need the rest..
  //and pull fro xml
  //te = 1635.;// K
  double te_p = plist->get<double>("te_",1641.);
  #ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(te_d,&te_p,sizeof(double));
#else
  te_d = te_p;
#endif
  te = te_p;
  //tl = 1706.;// K
  double tl_p = plist->get<double>("tl_",1706.);
  #ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(tl_d,&tl_p,sizeof(double));
#else
  tl_d = tl_p;
#endif
  tl = tl_p;
  //Lf = 17.2;// kJ/mol
  double Lf_p = plist->get<double>("Lf_",2.95e5);
  #ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(Lf_d,&Lf_p,sizeof(double));
#else
  Lf_d = Lf_p;
#endif
  Lf = Lf_p; 
  
  double dfldu_mushy_p = tpetra::heat::rho_h*Lf/(tl-te);//fl=(t-te)/(tl-te);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(dfldu_mushy_d,&dfldu_mushy_p,sizeof(double));
#else
  dfldu_mushy_d = dfldu_mushy_p;
#endif
  dfldu_mushy_h = dfldu_mushy_p;

  double eta_p = plist->get<double>("eta_",0.3);//dimensionless
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(eta_d,&eta_p,sizeof(double));
#else
  eta_d = eta_p;
#endif
  eta_h = eta_p;
  double P_p = plist->get<double>("P_",50.);// W
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(P_d,&P_p,sizeof(double));
#else
  P_d = P_p;
#endif
  P_h = P_p;
  double s_p = plist->get<double>("s_",2.);//dimensionless
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(s_d,&s_p,sizeof(double));
#else
  s_d = s_p;
#endif
  s_h = s_p;
  double r_p = plist->get<double>("r_",.00005);// 50 um
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(r_d,&r_p,sizeof(double));
#else
  r_d = r_p;
#endif
  r_h = r_p;
  double d_p = plist->get<double>("d_",.00001);// 10 um
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(d_d,&d_p,sizeof(double));
#else
  d_d = d_p;
#endif
  d_h = d_p;
  //gamma_d = is gamma function
  //gamma(3/s):
  //gamma(3/2) = sqrt(pi)/2
  double gamma_p = plist->get<double>("gamma_",0.886227);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(gamma_d,&gamma_p,sizeof(double));
#else
  gamma_d = gamma_p;
#endif
  gamma_h = gamma_p;
  double x0_p = plist->get<double>("x0_",0.);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(x0_d,&x0_p,sizeof(double));
#else
  x0_d = x0_p;
#endif
  x0_h = x0_p;
  double y0_p = plist->get<double>("y0_",0.);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(y0_d,&y0_p,sizeof(double));
#else
  y0_d = y0_p;
#endif
  y0_h = y0_p;
  double z0_p = plist->get<double>("z0_",0.);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(z0_d,&z0_p,sizeof(double));
#else
  z0_d = z0_p;
#endif
  z0_h = z0_p;
  double t_hold_p = plist->get<double>("t_hold_",0.005);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(t_hold_d,&t_hold_p,sizeof(double));
#else
  t_hold_d = t_hold_p;
#endif
  t_hold_h = t_hold_p;
  double t_decay_p = plist->get<double>("t_decay_",0.01);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(t_decay_d,&t_decay_p,sizeof(double));
#else
  t_decay_d = t_decay_p;
#endif
  t_decay_h = t_decay_p;
  
  double tau0_p = plist->get<double>("tau0_",1.);
  #ifdef TUSAS_HAVE_CUDA
    cudaMemcpyToSymbol(tau0_d,&tau0_p,sizeof(double));
  #else
    tau0_d = tau0_p;
  #endif
  tau0_h = tau0_p;
  
  double W0_p = plist->get<double>("W0_",1.);
  #ifdef TUSAS_HAVE_CUDA
    cudaMemcpyToSymbol(W0_d,&W0_p,sizeof(double));
  #else
    W0_d = W0_p;
  #endif
  W0_h = W0_p;
  
  double t0_p = plist->get<double>("t0_",300.);
  #ifdef TUSAS_HAVE_CUDA
    cudaMemcpyToSymbol(t0_d,&t0_p,sizeof(double));
  #else
    t0_d = t0_p;
  #endif
  t0_h = t0_p;
  
  double scaling_constant_p = plist->get<double>("scaling_constant_",1.);
  #ifdef TUSAS_HAVE_CUDA
    cudaMemcpyToSymbol(scaling_constant_d,&scaling_constant_p,sizeof(double));
  #else
    scaling_constant_d = scaling_constant_p;
  #endif
  
  double uref = plist->get<double>("uref_",0.);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(uref_d,&uref,sizeof(double));
#else
  uref_d = uref;
#endif
  uref_h = uref;

}
#endif
}//namespace goldak
}//namespace tpetra


#endif
