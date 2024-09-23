//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef FUNCTION_DEF_HPP
#define FUNCTION_DEF_HPP

#include <boost/ptr_container/ptr_vector.hpp>
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



/** Definition for residual function. Each residual function is called at each Gauss point for each equation with this signature:
- NAME:     name of function to call
- const boost::ptr_vector<Basis> &basis:     an array of basis function objects indexed by equation
- const int &i:    the current test function (row in residual vector)
- const double &dt_: the timestep size as prescribed in input file						
- const double &t_theta_: the timestep parameter as prescribed in input file
- const double &time: the current simulation time
- const int &eqn_id: the index of the current equation


*/

#define RES_FUNC_TPETRA(NAME)  const double NAME(GPUBasis * basis[],	\
						 const int &i,		\
						 const double &dt_,	\
						 const double &dtold_,	\
						 const double &t_theta_, \
						 const double &t_theta2_, \
						 const double &time,	\
						 const int &eqn_id,	\
						 const double &vol,	\
						 const double &rand)


/** Definition for precondition function. Each precondition function is called at each Gauss point for each equation with this signature:
- NAME:     name of function to call
- const boost::ptr_vector<Basis> &basis:     an array of basis function objects indexed by equation
- const int &i:    the current basis function (row in preconditioning matrix)
- const int &j:    the current test function (column in preconditioning matrix)
- const double &dt_: the timestep size as prescribed in input file						
- const double &t_theta_: the timestep parameter as prescribed in input file
- const double &time: the current simulation time
- const int &eqn_id: the index of the current equation


*/

#define PRE_FUNC_TPETRA(NAME)  const double NAME(GPUBasis *basis[], \
						 const int &i,	    \
						 const int &j,	    \
						 const double &dt_, \
						 const double &t_theta_, \
						 const int &eqn_id)


/** Definition for initialization function. Each initialization function is called at each node for each equation at the beginning of the simualtaion with this signature:
- NAME:     name of function to call
- const double &x: the x-ccordinate of the node
- const double &y: the y-ccordinate of the node
- const double &z: the z-ccordinate of the node
- const int &eqn_id: the index of the current equation

*/

#define INI_FUNC(NAME)  const double NAME(const double &x,\
					  const double &y,	\
					  const double &z,	\
					  const int &eqn_id,	\
					  const int &lid) 


/** Definition for Dirichlet function. Each Dirichlet function is called at each node for each equation with this signature:
- NAME:     name of function to call
- const double &x: the x-ccordinate of the node
- const double &y: the y-ccordinate of the node
- const double &z: the z-ccordinate of the node
- const int &eqn_id: the index of the current equation
- const double &t: the current time

*/

#define DBC_FUNC(NAME)  const double NAME(const double &x,\
					  const double &y,	\
					  const double &z,	\
					  const double &t) 

/** Definition for Neumann function. Each Neumann function is called at each Gauss point for the current equation with this signature:
- NAME:     name of function to call
- const Basis *basis:     basis function object for current equation
- const int &i:    the current basis function (row in residual vector)
- const double &dt_: the timestep size as prescribed in input file						
- const double &t_theta_: the timestep parameter as prescribed in input file
- const double &time: the current simulation time


*/

#define NBC_FUNC_TPETRA(NAME)  const double NAME(const GPUBasis *basis,\
						 const int &i,	       \
						 const double &dt_,    \
						 const double &dtold_, \
						 const double &t_theta_, \
						 const double &t_theta2_, \
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




namespace tpetra{//we can just put the KOKKOS... around the other dbc_zero_ later...

namespace heat{
TUSAS_DEVICE
double k_d = 1.;
TUSAS_DEVICE
double rho_d = 1.;
TUSAS_DEVICE
double cp_d = 1.;
TUSAS_DEVICE
double tau0_d = 1.;
TUSAS_DEVICE
double W0_d = 1.;
TUSAS_DEVICE
double deltau_d = 1.;
TUSAS_DEVICE
double uref_d = 0.;

double k_h = 1.;
double rho_h = 1.;
double cp_h = 1.;

double tau0_h = 1.;
double W0_h = 1.;

double deltau_h = 1.;
double uref_h = 0.;

KOKKOS_INLINE_FUNCTION 
DBC_FUNC(dbc_zero_) 
{
  return 0.;
}

KOKKOS_INLINE_FUNCTION 
DBC_FUNC(dbc_sin_) 
{  
  const double pi = 3.141592653589793;

  return sin(pi*x)*sin(pi*y);
}

  //KOKKOS_INLINE_FUNCTION 
INI_FUNC(init_heat_test_)
{

  const double pi = 3.141592653589793;

  return sin(pi*x)*sin(pi*y);
}

INI_FUNC(init_zero_)
{
  return 0.;
}

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_heat_test_)
{
  //right now, it is probably best to handle nondimensionalization of temperature via:
  // theta = (T-T_s)/(T_l-T_s) external to this module by multiplication of (T_l-T_s)=delta T

  const double ut = rho_d*cp_d/tau0_d*deltau_d*(basis[eqn_id]->uu()-basis[eqn_id]->uuold())/dt_*basis[0]->phi(i);
  const double f[3] = {k_d/W0_d/W0_d*deltau_d*(basis[eqn_id]->duudx()*basis[0]->dphidx(i)
			    + basis[eqn_id]->duudy()*basis[0]->dphidy(i)
			    + basis[eqn_id]->duudz()*basis[0]->dphidz(i)),
		       k_d/W0_d/W0_d*deltau_d*(basis[eqn_id]->duuolddx()*basis[0]->dphidx(i)
			    + basis[eqn_id]->duuolddy()*basis[0]->dphidy(i)
			    + basis[eqn_id]->duuolddz()*basis[0]->dphidz(i)),
		       k_d/W0_d/W0_d*deltau_d*(basis[eqn_id]->duuoldolddx()*basis[0]->dphidx(i)
			    + basis[eqn_id]->duuoldolddy()*basis[0]->dphidy(i)
			    + basis[eqn_id]->duuoldolddz()*basis[0]->dphidz(i))};
  //std::cout<<std::scientific<<f[0]<<std::endl<<std::defaultfloat;
  return ut + (1.-t_theta2_)*t_theta_*f[0]
    + (1.-t_theta2_)*(1.-t_theta_)*f[1]
    +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_heat_test_dp_)) = residual_heat_test_;

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_heat_test_)
{
  return rho_d*cp_d/tau0_d*deltau_d*basis[0]->phi(j)/dt_*basis[0]->phi(i)
    + t_theta_*k_d/W0_d/W0_d*deltau_d*(basis[0]->dphidx(j)*basis[0]->dphidx(i)
       + basis[0]->dphidy(j)*basis[0]->dphidy(i)
       + basis[0]->dphidz(j)*basis[0]->dphidz(i));
}

TUSAS_DEVICE
PRE_FUNC_TPETRA((*prec_heat_test_dp_)) = prec_heat_test_;

PARAM_FUNC(param_)
{
  double kk = plist->get<double>("k_",1.);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(k_d,&kk,sizeof(double));
#else
  k_d = kk;
#endif
  k_h = kk;

  double rho = plist->get<double>("rho_",1.);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(rho_d,&rho,sizeof(double));
#else
  rho_d = rho;
#endif
  rho_h = rho;

  double cp = plist->get<double>("cp_",1.);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(cp_d,&cp,sizeof(double));
#else
  cp_d = cp;
#endif
  cp_h = cp;

  double tau0 = plist->get<double>("tau0_",1.);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(tau0_d,&tau0,sizeof(double));
#else
  tau0_d = tau0;
#endif
  tau0_h = tau0;

  double W0 = plist->get<double>("W0_",1.);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(W0_d,&W0,sizeof(double));
#else
  W0_d = W0;
#endif
  W0_h = W0;

  double deltau = plist->get<double>("deltau_",1.);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(deltau_d,&deltau,sizeof(double));
#else
  deltau_d = deltau;
#endif
  deltau_h = deltau;

  double uref = plist->get<double>("uref_",0.);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(uref_d,&uref,sizeof(double));
#else
  uref_d = uref;
#endif
  uref_h = uref;
}

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
  const double u_m = t_theta_*basis[eqn_id]->uu() + (1. - t_theta_)*basis[eqn_id]->uuold();
  const double dudx_m = t_theta_*basis[eqn_id]->duudx() + (1. - t_theta_)*basis[eqn_id]->duuolddx();
  const double dudy_m = t_theta_*basis[eqn_id]->duudy() + (1. - t_theta_)*basis[eqn_id]->duuolddy();
  const double dudz_m = t_theta_*basis[eqn_id]->duudz() + (1. - t_theta_)*basis[eqn_id]->duuolddz();
  const double t_m = time + t_theta_*dt_;
  const double x = basis[0]->xx();
  const double y = basis[0]->yy();

  const double divgrad = u_m*(dudx_m*basis[0]->dphidx(i) 
			      + dudy_m*basis[0]->dphidy(i) 
			      + dudz_m*basis[0]->dphidz(i));

  return (basis[eqn_id]->uu()-basis[eqn_id]->uuold())/dt_*basis[0]->phi(i)
    + divgrad
    + f1(u_m)*basis[0]->phi(i)
    + f2(x,y,t_m)*basis[0]->phi(i);
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_nlheatimr_test_dp_)) = residual_nlheatimr_test_;


KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_nlheatcn_test_)
{
  const double u[2] = {basis[eqn_id]->uu(), basis[eqn_id]->uuold()};
  const double dudx[2] = {basis[eqn_id]->duudx(), basis[eqn_id]->duuolddx()};
  const double dudy[2] = {basis[eqn_id]->duudy(), basis[eqn_id]->duuolddy()};
  const double dudz[2] = {basis[eqn_id]->duudz(), basis[eqn_id]->duuolddz()};
  //const double dudz_m = t_theta_*basis[eqn_id].duudz() + (1. - t_theta_)*basis[eqn_id].duuolddz();
  const double t[2] = {time, time+dt_};
  const double x = basis[0]->xx();
  const double y = basis[0]->yy();

  const double divgrad = t_theta_*
    u[0]*(dudx[0]*basis[0]->dphidx(i) 
	  + dudy[0]*basis[0]->dphidy(i) 
	  + dudz[0]*basis[0]->dphidz(i))
    + (1. - t_theta_)*
    u[1]*(dudx[1]*basis[0]->dphidx(i) 
	  + dudy[1]*basis[0]->dphidy(i) 
	  + dudz[1]*basis[0]->dphidz(i));

  return (basis[eqn_id]->uu()-basis[eqn_id]->uuold())/dt_*basis[0]->phi(i)
    + divgrad
    + (t_theta_*f1(u[0])
       + (1. - t_theta_)*f1(u[1]))*basis[0]->phi(i)
    + (t_theta_*f2(x,y,t[0])
       + (1. - t_theta_)*f2(x,y,t[1]))*basis[0]->phi(i);
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_nlheatcn_test_dp_)) = residual_nlheatcn_test_;


KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_nlheatcn_test_)
{
  return basis[0]->phi(j)/dt_*basis[0]->phi(i)
    + t_theta_*basis[eqn_id]->uu()
    *(basis[0]->dphidx(j)*basis[0]->dphidx(i)
       + basis[0]->dphidy(j)*basis[0]->dphidy(i)
       + basis[0]->dphidz(j)*basis[0]->dphidz(i));
}

TUSAS_DEVICE
PRE_FUNC_TPETRA((*prec_nlheatcn_test_dp_)) = prec_nlheatcn_test_;


//}//namespace heat

namespace localprojection
{
RES_FUNC_TPETRA(residual_u1_)
{
  const double test = basis[0]->phi(i);

  //std::cout<<basis[0]->uu*basis[0]->uu()+basis[1]->uu()*basis[1]->uu()<<std::endl;

  const double u2[3] = {basis[1]->uu(), basis[1]->uuold(), basis[1]->uuoldold()};
  const double f[3] = {u2[0]*test,
		       u2[1]*test,
		       u2[2]*test};
  const double ut = (basis[eqn_id]->uu()-basis[eqn_id]->uuold())/dt_*test;

  return ut + (1.-t_theta2_)*t_theta_*f[0]
    + (1.-t_theta2_)*(1.-t_theta_)*f[1]
    +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

RES_FUNC_TPETRA(residual_u2_)
{
  const double test = basis[0]->phi(i);

  const double u1[3] = {basis[0]->uu(), basis[0]->uuold(), basis[0]->uuoldold()};

  const double f[3] = {-u1[0]*test,
		       -u1[1]*test,
		       -u1[2]*test};

  const double ut = (basis[eqn_id]->uu()-basis[eqn_id]->uuold())/dt_*test;
  return ut + (1.-t_theta2_)*t_theta_*f[0]
    + (1.-t_theta2_)*(1.-t_theta_)*f[1]
    +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

INI_FUNC(init_u1_)
{
  return 1.;
}

INI_FUNC(init_u2_)
{
  return 0.;
}

PPR_FUNC(postproc_u1_)
{
  return cos(time);
}

PPR_FUNC(postproc_u2_)
{
  return sin(time);
}

PPR_FUNC(postproc_norm_)
{
  return sqrt(u[0]*u[0]+u[1]*u[1])-1.;
}

PPR_FUNC(postproc_u1err_)
{
  return cos(time)-u[0];
}

PPR_FUNC(postproc_u2err_)
{
  return sin(time)-u[1];
}


}//namespace localprojection


namespace farzadi3d
{

  TUSAS_DEVICE
  double absphi = 0.9997;	//1.
  //double absphi = 0.999999;	//1.
  
  TUSAS_DEVICE
  double k = 0.14;

  TUSAS_DEVICE				//0.5
  double eps = 0.0;

  TUSAS_DEVICE
  double lambda = 10.;
  
  TUSAS_DEVICE
  double m = -2.6;
  TUSAS_DEVICE					//-2.6 100.
  double c_inf = 3.;				//1.
  
  TUSAS_DEVICE
  double G_solid_ = 3.e5; //k/m, temperature gradient in solid
  TUSAS_DEVICE
  double G_liquid_ = 3.e5; //k/m, temperature gradient in liquid
  TUSAS_DEVICE	      
  double R = 0.003;	//m/s, speed
//   TUSAS_DEVICE										
//   double V = 0.003;
	
  TUSAS_DEVICE											//m/s
  double d0 = 5.e-9;				//4.e-9					//m
  
  
  // parameters to scale dimensional quantities
  TUSAS_DEVICE
  double delta_T0 = 47.9143;

  TUSAS_DEVICE
  double w0 = 5.65675e-8;

  TUSAS_DEVICE
  double tau0 = 6.68455e-6;
  
//   TUSAS_DEVICE
//   double Vp0 = .354508;

  TUSAS_DEVICE
  double l_T0 = 2823.43;

  TUSAS_DEVICE
  double D_liquid_ = 6.267;

  TUSAS_DEVICE
  double D_solid_ = 0.;      //m^2/s

  TUSAS_DEVICE
  double at_coef_ = 1;

  TUSAS_DEVICE
  double dT = 0.0;

  double T_ref = 877.3;
  
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
  
  double t_activate_farzadi = 0.0;
  
PARAM_FUNC(param_)
{
  double k_p = plist->get<double>("k", 0.14);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(k,&k_p,sizeof(double));
#else
  k = k_p;
#endif
  double eps_p = plist->get<double>("eps", 0.0);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(eps,&eps_p,sizeof(double));
#else
  eps = eps_p;
#endif
  double lambda_p = plist->get<double>("lambda", 10.);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(lambda,&lambda_p,sizeof(double));
#else
  lambda = lambda_p;
#endif
  double d0_p = plist->get<double>("d0", 5.e-9);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(d0,&d0_p,sizeof(double));
#else
  d0 = d0_p;
#endif

  double D_liquid_p = plist->get<double>("D_liquid", 3.e-9);

  double D_solid_p = plist->get<double>("D_solid", 0.);

  double m_p = plist->get<double>("m", -2.6);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(m,&m_p,sizeof(double));
#else
  m = m_p;
#endif
  double c_inf_p = plist->get<double>("c_inf", 3.);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(c_inf,&c_inf_p,sizeof(double));
#else
  c_inf = c_inf_p;
#endif

  double G_p = plist->get<double>("G", 3.e5);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(G_solid_,&G_p,sizeof(double));
#else
  G_solid_ = G_p;
#endif

  double Gl_p = plist->get<double>("Gl", G_solid_);//we default this to G_solid_
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(G_liquid_,&Gl_p,sizeof(double));
#else
  G_liquid_ = Gl_p;
#endif


  double R_p = plist->get<double>("R", 0.003);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(R,&R_p,sizeof(double));
#else
  R = R_p;
#endif

// added dT here
double dT_p = plist->get<double>("dT", 0.0);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(dT,&dT_p,sizeof(double));
#else
  dT = dT_p;
#endif

  double base_height_p = plist->get<double>("base_height", 15.);
// #ifdef TUSAS_HAVE_CUDA
//   cudaMemcpyToSymbol(base_height,&base_height_p,sizeof(double));
// #else
  base_height = base_height_p;
// #endif
  double amplitude_p = plist->get<double>("amplitude", 0.2);
// #ifdef TUSAS_HAVE_CUDA
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
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(w0,&w0_p,sizeof(double));
#else
  w0 = w0_p;
#endif
  double tau0_p = (lambda_p*0.6267*w0_p*w0_p)/D_liquid_p;
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(tau0,&tau0_p,sizeof(double));
#else
  tau0 = tau0_p;
#endif

//   double V_p = R_p;
// #ifdef TUSAS_HAVE_CUDA
//   cudaMemcpyToSymbol(V,&V_p,sizeof(double));
// #else
//   V = V_p;
// #endif

//   double Vp0_p = V_p*tau0_p/w0_p;
// #ifdef TUSAS_HAVE_CUDA
//   cudaMemcpyToSymbol(Vp0,&Vp0_p,sizeof(double));
// #else
//   Vp0 = Vp0_p;
// #endif

  double delta_T0_p = abs(abs(m_p)*c_inf_p*(1.-k_p)/k_p);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(delta_T0,&delta_T0_p,sizeof(double));
#else
  delta_T0 = delta_T0_p;
#endif
  double l_T0_p = delta_T0_p/G_p;
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(l_T0,&l_T0_p,sizeof(double));
#else
  l_T0 = l_T0_p;
#endif

  double D_liquid__p = D_liquid_p*tau0_p/(w0_p*w0_p);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(D_liquid_,&D_liquid__p,sizeof(double));
#else
  D_liquid_ = D_liquid__p;
#endif

  double D_solid__p = D_solid_p*tau0_p/(w0_p*w0_p);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(D_solid_,&D_solid__p,sizeof(double));
#else
  D_solid_ = D_solid__p;
#endif

double at_coef_p = plist->get<double>("at_coef", 1.);
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(at_coef_,&at_coef_p,sizeof(double));
#else
  at_coef_ = at_coef_p;
#endif


t_activate_farzadi = plist->get<double>("t_activate_farzadi", 0.0);

 T_ref = plist->get<double>("T_ref",877.3);

  //std::cout<<l_T0<<"   "<<G<<"  "<<Vp0<<"  "<<tau0<<"   "<<w0<<std::endl;
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

//the current ordering in set_test_case is conc (u), phase (phi)

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_conc_farzadi_)
{
  //right now, if explicit, we will have some problems with time derivates below
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double test = basis[0]->phi(i);
  const double u[3] = {basis[eqn_id]->uu(),basis[eqn_id]->uuold(),basis[eqn_id]->uuoldold()};

  const int phi_id = eqn_id+1;
  const double phi[3] = {basis[phi_id]->uu(),basis[phi_id]->uuold(),basis[phi_id]->uuoldold()};
  const double dphidx[3] = {basis[phi_id]->duudx(),basis[phi_id]->duuolddx(),basis[phi_id]->duuoldolddx()};
  const double dphidy[3] = {basis[phi_id]->duudy(),basis[phi_id]->duuolddy(),basis[phi_id]->duuoldolddy()};
  const double dphidz[3] = {basis[phi_id]->duudz(),basis[phi_id]->duuolddz(),basis[phi_id]->duuoldolddz()};

  const double ut = (1. + k - (1.0 - k) * phi[0]) / 2. * (u[0] - u[1]) / dt_ * test;

  const double D[3] = {D_liquid_*(1.-phi[0])/2.+D_solid_*(1.+phi[0])/2.,
		       D_liquid_*(1.-phi[1])/2.+D_solid_*(1.-phi[1])/2.,
		       D_liquid_*(1.-phi[2])/2.+D_solid_*(1.-phi[2])/2.};

  const double divgradu[3] = {D[0]*(basis[eqn_id]->duudx()*dtestdx + basis[eqn_id]->duudy()*dtestdy + basis[eqn_id]->duudz()*dtestdz),
			      D[1]*(basis[eqn_id]->duuolddx()*dtestdx + basis[eqn_id]->duuolddy()*dtestdy + basis[eqn_id]->duuolddz()*dtestdz),
			      D[2]*(basis[eqn_id]->duuoldolddx()*dtestdx + basis[eqn_id]->duuoldolddy()*dtestdy + basis[eqn_id]->duuoldolddz()*dtestdz)};//(grad u,grad phi)

  const double normd[3] = {(phi[0]*phi[0] < absphi)&&(phi[0]*phi[0] > 0.) ? 1./sqrt(dphidx[0]*dphidx[0] + dphidy[0]*dphidy[0] + dphidz[0]*dphidz[0]) : 0.,
			   (phi[1]*phi[1] < absphi)&&(phi[1]*phi[1] > 0.) ? 1./sqrt(dphidx[1]*dphidx[1] + dphidy[1]*dphidy[1] + dphidz[1]*dphidz[1]) : 0.,
			   (phi[2]*phi[2] < absphi)&&(phi[2]*phi[2] > 0.) ? 1./sqrt(dphidx[2]*dphidx[2] + dphidy[2]*dphidy[2] + dphidz[2]*dphidz[2]) : 0.}; //cn lim grad phi/|grad phi| may -> 1 here?

  //we need to double check these terms with temporal derivatives....
  const double phit = (phi[0]-phi[1])/dt_;
  const double j_coef[3] = {(1.+(1.-k)*u[0])/sqrt(8.)*normd[0]*phit,
			    (1.+(1.-k)*u[1])/sqrt(8.)*normd[1]*phit,
			    (1.+(1.-k)*u[2])/sqrt(8.)*normd[2]*phit};
  const double divj[3] = {at_coef_*j_coef[0]*(dphidx[0]*dtestdx + dphidy[0]*dtestdy + dphidz[0]*dtestdz),
			  at_coef_*j_coef[1]*(dphidx[1]*dtestdx + dphidy[1]*dtestdy + dphidz[1]*dtestdz),
			  at_coef_*j_coef[2]*(dphidx[2]*dtestdx + dphidy[2]*dtestdy + dphidz[2]*dtestdz)};

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
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  //test function
  const double test = basis[0]->phi(i);
  //u, phi
  const int u_id = eqn_id-1;
  const double u[3] = {basis[u_id]->uu(),basis[u_id]->uuold(),basis[u_id]->uuoldold()};
  const double phi[3] = {basis[eqn_id]->uu(),basis[eqn_id]->uuold(),basis[eqn_id]->uuoldold()};

  const double dphidx[3] = {basis[eqn_id]->duudx(),basis[eqn_id]->duuolddx(),basis[eqn_id]->duuoldolddx()};
  const double dphidy[3] = {basis[eqn_id]->duudy(),basis[eqn_id]->duuolddy(),basis[eqn_id]->duuoldolddy()};
  const double dphidz[3] = {basis[eqn_id]->duudz(),basis[eqn_id]->duuolddz(),basis[eqn_id]->duuoldolddz()};

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

TUSAS_DEVICE
const double gradT(const double &phi){
  return .5*(G_solid_ + G_liquid_ - G_solid_*phi + G_solid_*phi);
}

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_phase_farzadi_uncoupled_)
{
  //test function
  const double test = basis[0]->phi(i);
  //u, phi
  const int u_id = eqn_id-1;
  const double u[3] = {basis[u_id]->uu(),basis[u_id]->uuold(),basis[u_id]->uuoldold()};
  const double phi[3] = {basis[eqn_id]->uu(),basis[eqn_id]->uuold(),basis[eqn_id]->uuoldold()};

  const double dphidx[3] = {basis[eqn_id]->duudx(),basis[eqn_id]->duuolddx(),basis[eqn_id]->duuoldolddx()};
  const double dphidy[3] = {basis[eqn_id]->duudy(),basis[eqn_id]->duuolddy(),basis[eqn_id]->duuoldolddy()};
  const double dphidz[3] = {basis[eqn_id]->duudz(),basis[eqn_id]->duuolddz(),basis[eqn_id]->duuoldolddz()};

  const double as[3] = {a(phi[0],dphidx[0],dphidy[0],dphidz[0],eps),
			a(phi[1],dphidx[1],dphidy[1],dphidz[1],eps),
			a(phi[2],dphidx[2],dphidy[2],dphidz[2],eps)};

  const double mob[3] = {(1.+(1.-k)*u[0])*as[0]*as[0],(1.+(1.-k)*u[1])*as[1]*as[1],(1.+(1.-k)*u[2])*as[2]*as[2]};

  const double x = basis[0]->xx();//non dimensional x from mesh
  
  // frozen temperature approximation: linear pulling of the temperature field
  const double xx = x*w0;// dimensional x

  //cn this should probablly be: (time+dt_)*tau
  const double tt[3] = {(time+dt_)*tau0,time*tau0,(time-dtold_)*tau0};//dimensional time

  const double g4[3] = {((dT < 0.001) ? gradT(phi[0])*(xx-R*tt[0])/delta_T0 : dT),
			     ((dT < 0.001) ? gradT(phi[1])*(xx-R*tt[1])/delta_T0 : dT),
			     ((dT < 0.001) ? gradT(phi[2])*(xx-R*tt[2])/delta_T0 : dT)};
  
  const double hp1g4[3] = {lambda*(1. - phi[0]*phi[0])*(1. - phi[0]*phi[0])*(g4[0])*test,
			 lambda*(1. - phi[1]*phi[1])*(1. - phi[1]*phi[1])*(g4[1])*test,
			 lambda*(1. - phi[2]*phi[2])*(1. - phi[2]*phi[2])*(g4[2])*test};

  const double val = tpetra::farzadi3d::residual_phase_farzadi_dp_(basis,
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
  const double test = basis[0]->phi(i);
  //u, phi
  const int u_id = eqn_id-1;
  const int theta_id = eqn_id+1;
  const double u[3] = {basis[u_id]->uu(),basis[u_id]->uuold(),basis[u_id]->uuoldold()};
  const double phi[3] = {basis[eqn_id]->uu(),basis[eqn_id]->uuold(),basis[eqn_id]->uuoldold()};

  const double dphidx[3] = {basis[eqn_id]->duudx(),basis[eqn_id]->duuolddx(),basis[eqn_id]->duuoldolddx()};
  const double dphidy[3] = {basis[eqn_id]->duudy(),basis[eqn_id]->duuolddy(),basis[eqn_id]->duuoldolddy()};
  const double dphidz[3] = {basis[eqn_id]->duudz(),basis[eqn_id]->duuolddz(),basis[eqn_id]->duuoldolddz()};

  const double as[3] = {a(phi[0],dphidx[0],dphidy[0],dphidz[0],eps),
			a(phi[1],dphidx[1],dphidy[1],dphidz[1],eps),
			a(phi[2],dphidx[2],dphidy[2],dphidz[2],eps)};

  const double mob[3] = {(1.+(1.-k)*u[0])*as[0]*as[0],(1.+(1.-k)*u[1])*as[1]*as[1],(1.+(1.-k)*u[2])*as[2]*as[2]};

  const double theta[3] = {basis[theta_id]->uu(),basis[theta_id]->uuold(),basis[theta_id]->uuoldold()};
  
  const double g4[3] = {theta[0],theta[1],theta[2]};
  
  const double hp1g4[3] = {lambda*(1. - phi[0]*phi[0])*(1. - phi[0]*phi[0])*(g4[0])*test,
			 lambda*(1. - phi[1]*phi[1])*(1. - phi[1]*phi[1])*(g4[1])*test,
			 lambda*(1. - phi[2]*phi[2])*(1. - phi[2]*phi[2])*(g4[2])*test};

  const double val = tpetra::farzadi3d::residual_phase_farzadi_dp_(basis,
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
RES_FUNC_TPETRA((*residual_phase_farzadi_coupled_dp_)) = residual_phase_farzadi_coupled_;

RES_FUNC_TPETRA(residual_conc_farzadi_activated_)
{
	const double val = tpetra::farzadi3d::residual_conc_farzadi_dp_(basis,
  						 i,
  						 dt_,
  						 dtold_,
  						 t_theta_,
  						 t_theta2_,
  						 time,
  						 eqn_id,
  						 vol,
  						 rand);
	
	const double u[2] = {basis[eqn_id]->uu(),basis[eqn_id]->uuold()};
	
	// Coefficient to turn Farzadi evolution off until a specified time
	const double delta = 1.0e12; 			   
	const double sigmoid_var = delta * (time-t_activate_farzadi/tau0);
	const double sigmoid = 0.5 * (1.0 + sigmoid_var / (std::sqrt(1.0 + sigmoid_var*sigmoid_var))); 			   
	//std::cout<<val * sigmoid + (u[1]-u[0]) * (1.0 - sigmoid)*basis[0]->phi(i)<<std::endl;
	return val * sigmoid + (u[1]-u[0]) * (1.0 - sigmoid)*basis[0]->phi(i);
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_conc_farzadi_activated_dp_)) = residual_conc_farzadi_activated_;

RES_FUNC_TPETRA(residual_phase_farzadi_coupled_activated_)
{
	const double val = tpetra::farzadi3d::residual_phase_farzadi_coupled_dp_(basis,
  						 i,
  						 dt_,
  						 dtold_,
  						 t_theta_,
  						 t_theta2_,
  						 time,
  						 eqn_id,
  						 vol,
  						 rand);
	
	const double phi[2] = {basis[eqn_id]->uu(),basis[eqn_id]->uuold()};
	
	// Coefficient to turn Farzadi evolution off until a specified time
	const double delta = 1.0e12; 			   
	const double sigmoid_var = delta * (time-t_activate_farzadi/tau0);
	const double sigmoid = 0.5 * (1.0 + sigmoid_var / (std::sqrt(1.0 + sigmoid_var*sigmoid_var))); 			   

	return val * sigmoid + (phi[1]-phi[0]) * (1.0 - sigmoid)*basis[0]->phi(i);
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_phase_farzadi_coupled_activated_dp_)) = residual_phase_farzadi_coupled_activated_;

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_conc_farzadi_)
{
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double dbasisdx = basis[0]->dphidx(j);
  const double dbasisdy = basis[0]->dphidy(j);
  const double dbasisdz = basis[0]->dphidz(j);

  const double test = basis[0]->phi(i);
  const double divgrad = (D_liquid_*(1.-basis[1]->uu())/2.+D_solid_*(1.+basis[1]->uu())/2.)*(dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz);

  const int phi_id = eqn_id+1;
  const double phi = basis[phi_id]->uu();
  const double u_t = (1. + k - (1.0 - k) * phi) / 2. *basis[0]->phi(j)  / dt_ * test;

  return u_t + t_theta_*(divgrad);

}

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_phase_farzadi_)
{
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double dbasisdx = basis[0]->dphidx(j);
  const double dbasisdy = basis[0]->dphidy(j);
  const double dbasisdz = basis[0]->dphidz(j);

  const double test = basis[0]->phi(i);
  
  const double dphidx = basis[1]->duudx();
  const double dphidy = basis[1]->duudy();
  const double dphidz = basis[1]->duudz();

  const double u = basis[0]->uu();
  const double phi = basis[1]->uu();

  const double as = a(phi,dphidx,dphidy,dphidz,eps);

  const double m = (1.+(1.-k)*u)*as*as;
  const double phit = (basis[0]->phi(j))/dt_*test;

  const double divgrad = as*as*(dbasisdx*dtestdx + dbasisdy*dtestdy + dbasisdz*dtestdz);

  return (phit + t_theta_*(divgrad)/m)*m;
}

TUSAS_DEVICE
PRE_FUNC_TPETRA((*prec_phase_farzadi_dp_)) = prec_phase_farzadi_;

TUSAS_DEVICE
PRE_FUNC_TPETRA((*prec_conc_farzadi_dp_)) = prec_conc_farzadi_;

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_conc_farzadi_exp_)
{
  //this is the explicit case with explicit phit
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double test = basis[0]->phi(i);
  const double u[2] = {basis[0]->uu(),basis[0]->uuold()};
  const double phi[2] = {basis[1]->uu(),basis[1]->uuold()};
  const double dphidx[2] = {basis[1]->duudx(),basis[1]->duuolddx()};
  const double dphidy[2] = {basis[1]->duudy(),basis[1]->duuolddy()};
  const double dphidz[2] = {basis[1]->duudz(),basis[1]->duuolddz()};

  const double ut = (1.+k-(1.0-k)*phi[0])/2.*(u[0]-u[1])/dt_*test;
  const double divgradu[2] = {D_liquid_*(1.-phi[0])/2.*(basis[0]->duudx()*dtestdx + basis[0]->duudy()*dtestdy + basis[0]->duudz()*dtestdz),
			      D_liquid_*(1.-phi[1])/2.*(basis[0]->duuolddx()*dtestdx + basis[0]->duuolddy()*dtestdy + basis[0]->duuolddz()*dtestdz)};//(grad u,grad phi)

  const double normd[2] = {(phi[0]*phi[0] < absphi)&&(phi[0]*phi[0] > 0.) ? 1./sqrt(dphidx[0]*dphidx[0] + dphidy[0]*dphidy[0] + dphidz[0]*dphidz[0]) : 0.,
			   (phi[1]*phi[1] < absphi)&&(phi[1]*phi[1] > 0.) ? 1./sqrt(dphidx[1]*dphidx[1] + dphidy[1]*dphidy[1] + dphidz[1]*dphidz[1]) : 0.}; //cn lim grad phi/|grad phi| may -> 1 here?

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

  return -c_inf*(1.+k-phi+k*phi)*(-1.-uu+k*uu)/2./k;
  //normalize, ie divide by c_inf
  //return -(1.+k-phi+k*phi)*(-1.-uu+k*uu)/2./k;
}

PPR_FUNC(postproc_t_)
{
  // return the physical temperature in K here
  const double x = xyz[0];
  const double phi = u[1];

  const double xx = x*w0;
  const double tt = time*tau0;
  //return ((dT < 0.001) ? 877.3 + (xx-R*tt)/l_T0*delta_T0 : 877.3);
  return ((dT < 0.001) ? T_ref + gradT(phi)*(xx-R*tt) : T_ref);
}
}//namespace farzadi3d

namespace noise
{
  double interface_noise_amplitude_d = 0.0;

KOKKOS_INLINE_FUNCTION 
double noise_(const double &rand, const double &dt, const double &vol)
{
  return interface_noise_amplitude_d*rand*std::sqrt(dt/vol);
}

PARAM_FUNC(param_)
{
  double interface_noise_amplitude_p = plist->get<double>("interface_noise_amplitude", 0.0);
  interface_noise_amplitude_d=interface_noise_amplitude_p;
}
}//namespace noise

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
  const double ut = (basis[eqn_id]->uu()-basis[eqn_id]->uuold())/dt_*basis[0]->phi(i);
  double divgradu[3] = {D_*(basis[eqn_id]->duudx()*basis[0]->dphidx(i)
			  + basis[eqn_id]->duudy()*basis[0]->dphidy(i)
			  + basis[eqn_id]->duudz()*basis[0]->dphidz(i)),
			D_*(basis[eqn_id]->duuolddx()*basis[0]->dphidx(i)
			  + basis[eqn_id]->duuolddy()*basis[0]->dphidy(i)
			  + basis[eqn_id]->duuolddz()*basis[0]->dphidz(i)),
			D_*(basis[eqn_id]->duuoldolddx()*basis[0]->dphidx(i)
			  + basis[eqn_id]->duuoldolddy()*basis[0]->dphidy(i)
			  + basis[eqn_id]->duuoldolddz()*basis[0]->dphidz(i))};

  const double phit[2] = {.5*(basis[1]->uu()-basis[1]->uuold())/dt_*basis[0]->phi(i),
			  .5*(basis[1]->uuold()-basis[1]->uuoldold())/dt_*basis[0]->phi(i)};

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
  const double test = basis[0]->phi(i);
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);

  const double phi[3] = {basis[eqn_id]->uu(),basis[eqn_id]->uuold(),basis[eqn_id]->uuoldold()};
  const double dphidx[3] = {basis[eqn_id]->duudx(),basis[eqn_id]->duuolddx(),basis[eqn_id]->duuoldolddx()};
  const double dphidy[3] = {basis[eqn_id]->duudy(),basis[eqn_id]->duuolddy(),basis[eqn_id]->duuoldolddy()};
  const double dphidz[3] = {basis[eqn_id]->duudz(),basis[eqn_id]->duuolddz(),basis[eqn_id]->duuoldolddz()};

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

  const double g[3] = {((phi[0]-lambda_*basis[0]->uu()*(1.-phi[0]*phi[0]))*(1.-phi[0]*phi[0]))*test,
		       ((phi[1]-lambda_*basis[0]->uuold()*(1.-phi[1]*phi[1]))*(1.-phi[1]*phi[1]))*test,
		       ((phi[2]-lambda_*basis[0]->uuoldold()*(1.-phi[2]*phi[2]))*(1.-phi[2]*phi[2]))*test};

  double f[3];
  f[0] = -(divgradphi[0]/tau0_+curlgrad[0]/tau[0]-g[0]/tau[0]);
  f[1] = -(divgradphi[1]/tau0_+curlgrad[1]/tau[1]-g[1]/tau[1]);
  f[2] = -(divgradphi[2]/tau0_+curlgrad[2]/tau[2]-g[2]/tau[2]);

  return phit - (1.-t_theta2_)*t_theta_*f[0]
    - (1.-t_theta2_)*(1.-t_theta_)*f[1]
    -.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

RES_FUNC_TPETRA(residual_phase_pfhub3_noise_)
{
  double val = residual_phase_pfhub3_(basis,
				      i,
				      dt_,
				      dtold_,
				      t_theta_,
				      t_theta2_,
				      time,
				      eqn_id,
				      vol,
				      rand);
  const double phi[1] ={ basis[eqn_id]->uu()};
  const double g = (1.-phi[0]*phi[0])*(1.-phi[0]*phi[0]);
  
  //on crusher noise is noise_amplitude*rand*sqrt(dt_/vol)*test*g
  double noise[3] = {g*tpetra::noise::noise_(rand,dt_,vol)*basis[0]->phi[i],0.*basis[0]->phi[i],0.*basis[0]->phi[i]};

  double rv = (val + (1.-t_theta2_)*t_theta_*noise[0]
	  + (1.-t_theta2_)*(1.-t_theta_)*noise[1]
	       +.5*t_theta2_*((2.+dt_/dtold_)*noise[1]-dt_/dtold_*noise[2]));

  return rv;
}
/* // Comment out nemesis code
RES_FUNC(residual_heat_pfhub3_n_)
{
  const double ut = (basis[eqn_id].uu()-basis[eqn_id].uuold())/dt_*basis[eqn_id].phi(i);
  const double divgradu[3] = {D_*(basis[eqn_id].duudx()*basis[eqn_id].dphidx(i)
				  + basis[eqn_id].duudy()*basis[eqn_id].dphidy(i)
				  + basis[eqn_id].duudz()*basis[eqn_id].dphidz(i)),
			      D_*(basis[eqn_id].duuolddx()*basis[eqn_id].dphidx(i)
				  + basis[eqn_id].duuolddy()*basis[eqn_id].dphidy(i)
				  + basis[eqn_id].duuolddz()*basis[eqn_id].dphidz(i)),
			      D_*(basis[eqn_id].duuoldolddx()*basis[eqn_id].dphidx(i)
				  + basis[eqn_id].duuoldolddy()*basis[eqn_id].dphidy(i)
				  + basis[eqn_id].duuoldolddz()*basis[eqn_id].dphidz(i))};

  const double phit[2] = {.5*(basis[1].uu()-basis[1].uuold())/dt_*basis[0].phi(i),
			  .5*(basis[1].uuold()-basis[1].uuoldold())/dt_*basis[0].phi(i)};

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
  const double test = basis[eqn_id].phi(i);
  const double dtestdx = basis[eqn_id].dphidx(i);
  const double dtestdy = basis[eqn_id].dphidy(i);
  const double dtestdz = basis[eqn_id].dphidz(i);

  const double phi[3] = {basis[eqn_id].uu(),basis[eqn_id].uuold(),basis[eqn_id].uuoldold()};
  const double dphidx[3] = {basis[eqn_id].duudx(),basis[eqn_id].duuolddx(),basis[eqn_id].duuoldolddx()};
  const double dphidy[3] = {basis[eqn_id].duudy(),basis[eqn_id].duuolddy(),basis[eqn_id].duuoldolddy()};
  const double dphidz[3] = {basis[eqn_id].duudz(),basis[eqn_id].duuolddz(),basis[eqn_id].duuoldolddz()};

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

  const double g[3] = {((phi[0]-lambda_*basis[0].uu()*(1.-phi[0]*phi[0]))*(1.-phi[0]*phi[0]))*test,
		       ((phi[1]-lambda_*basis[0].uuold()*(1.-phi[1]*phi[1]))*(1.-phi[1]*phi[1]))*test,
		       ((phi[2]-lambda_*basis[0].uuoldold()*(1.-phi[2]*phi[2]))*(1.-phi[2]*phi[2]))*test};

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
*/

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_phase_pfhub3_dp_)) = residual_phase_pfhub3_;

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_phase_pfhub3_noise_dp_)) = residual_phase_pfhub3_noise_;

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_heat_pfhub3_)
{
  const double ut = basis[0]->phi(j)/dt_*basis[0]->phi(i);
  const double divgradu = D_*(basis[0]->dphidx(j)*basis[0]->dphidx(i)
			  + basis[0]->dphidy(j)*basis[0]->dphidy(i)
			      + basis[0]->dphidz(j)*basis[0]->dphidz(i));
  return ut + t_theta_*divgradu;
}

/* // Comment out nemesis code
PRE_FUNC(prec_heat_pfhub3_n_)
{
  const double ut = basis[eqn_id].phi(j)/dt_*basis[eqn_id].phi(i);
  const double divgradu = D_*(basis[eqn_id].dphidx(j)*basis[eqn_id].dphidx(i)
			  + basis[eqn_id].dphidy(j)*basis[eqn_id].dphidy(i)
			      + basis[eqn_id].dphidz(j)*basis[eqn_id].dphidz(i));
  return ut + t_theta_*divgradu;
}
*/


TUSAS_DEVICE
PRE_FUNC_TPETRA((*prec_heat_pfhub3_dp_)) = prec_heat_pfhub3_;

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_phase_pfhub3_)
{
  const double test = basis[0]->phi(i);
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);

  const double phi = basis[eqn_id]->uu();
  const double phit = basis[0]->phi(j)/dt_*test;
//   const double as = a(phi,
// 		      basis[eqn_id]->duudx(),
// 		      basis[eqn_id]->duudy(),
// 		      basis[eqn_id]->duudz(),
// 		      eps_);
  const double tau = tau0_;//*as*as;

  const double divgradphi = W_*W_*(basis[0]->dphidx(j)*dtestdx
				   + basis[0]->dphidy(j)*dtestdy
				   + basis[0]->dphidz(j)*dtestdz);

  return phit
    + t_theta_*divgradphi/tau;
}

/* // Comment out Nemesis code
PRE_FUNC(prec_phase_pfhub3_n_)
{
  const double test = basis[eqn_id].phi(i);
  const double dtestdx = basis[eqn_id].dphidx(i);
  const double dtestdy = basis[eqn_id].dphidy(i);
  const double dtestdz = basis[eqn_id].dphidz(i);

  const double phi = basis[eqn_id].uu();
  const double phit = basis[eqn_id].phi(j)/dt_*test;
//   const double as = tpetra::farzadi3d::a(phi,
// 					     basis[eqn_id].duudx(),
// 					     basis[eqn_id].duudy(),
// 					     basis[eqn_id].duudz(),
// 					     eps_);
//   const double tau = tau0_*as*as;
  const double tau = tau0_;
//   const double divgradphi = W_*as*W_*as*(basis[eqn_id].dphidx(j)*dtestdx
// 					 + basis[eqn_id].dphidy(j)*dtestdy
// 					 + basis[eqn_id].dphidz(j)*dtestdz);
  const double divgradphi = W_*W_*(basis[eqn_id].dphidx(j)*dtestdx
					 + basis[eqn_id].dphidy(j)*dtestdy
					 + basis[eqn_id].dphidz(j)*dtestdz);
  return phit
    + t_theta_*divgradphi/tau;
}
*/

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
  int eqn_off_ = 2;
  TUSAS_DEVICE
  int ci_ = 0;
  TUSAS_DEVICE
  int mui_ = 1;
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
  //const double k_c_ = 3.;
  const double k_c_ = 0.0;
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
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(N_,&N_p,sizeof(int));
#else
    N_ = N_p;
#endif
    int eqn_off_p = plist->get<int>("OFFSET");
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(eqn_off_,&eqn_off_p,sizeof(int));
#else
    eqn_off_ = eqn_off_p;
#endif
  }
 
  PARAM_FUNC(param_trans_)
  {
    int N_p = plist->get<int>("N");
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(N_,&N_p,sizeof(int));
#else
    N_ = N_p;
#endif
    int eqn_off_p = plist->get<int>("OFFSET");
#ifdef TUSAS_HAVE_CUDA
  cudaMemcpyToSymbol(eqn_off_,&eqn_off_p,sizeof(int));
#else
    eqn_off_ = eqn_off_p;
#endif
    ci_ = 1;
    mui_ = 0;
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
  double dgdeta(const double *eta, const int eqn_id){

    double aval =0.;
    for (int i = 0; i < N_; i++){
      aval += eta[i]*eta[i];
    }
    aval = aval - eta[eqn_id]* eta[eqn_id];
    return 2.*eta[eqn_id]*(1. - eta[eqn_id])*(1. - eta[eqn_id])  
      - 2.* eta[eqn_id]* eta[eqn_id]* (1. - eta[eqn_id])
      + 4.*alpha_*eta[eqn_id] *aval;
  }

KOKKOS_INLINE_FUNCTION 
double dfdc(const double c, const double *eta)
{
  const double hh = h(eta);
  return df_alphadc(c)*(1.-hh)+df_betadc(c)*hh;
}

KOKKOS_INLINE_FUNCTION 
double dfdeta(const double c, const double eta)
{
  //this does not include the w g' term

  //dh(eta1,eta2)/deta1 is a function of eta1 only
  const double dh_deta = dhdeta(eta);
  return f_alpha(c)*(-dh_deta)+f_beta(c)*dh_deta;
}

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_c_)
{
  // c_t + M grad mu grad test
  const double ut = (basis[ci_]->uu()-basis[ci_]->uuold())/dt_*basis[0]->phi(i);
  //M_ divgrad mu

  const double f[3] = {M_*(basis[mui_]->duudx()*basis[0]->dphidx(i)
			   + basis[mui_]->duudy()*basis[0]->dphidy(i)
			   + basis[mui_]->duudz()*basis[0]->dphidz(i)),
		       M_*(basis[mui_]->duuolddx()*basis[0]->dphidx(i)
			   + basis[mui_]->duuolddy()*basis[0]->dphidy(i)
			   + basis[mui_]->duuolddz()*basis[0]->dphidz(i)),
		       M_*(basis[mui_]->duuoldolddx()*basis[0]->dphidx(i)
			   + basis[mui_]->duuoldolddy()*basis[0]->dphidy(i)
			   + basis[mui_]->duuoldolddz()*basis[0]->dphidz(i))};

  return ut + (1.-t_theta2_)*t_theta_*f[0]
    + (1.-t_theta2_)*(1.-t_theta_)*f[1]
    +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_c_dp_)) = residual_c_;

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_c_kks_)
{
  //derivatives of the test function
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  //double dtestdz = basis[0]->dphidz(i);
  //test function
  const double test = basis[0]->phi(i);
  //u, phi
  const double c[2] = {basis[0]->uu(), basis[0]->uuold()};
  const double dcdx[2] = {basis[0]->duudx(), basis[0]->duuolddx()};
  const double dcdy[2] = {basis[0]->duudy(), basis[0]->duuolddy()};

  double dhdx[2] = {0., 0.};
  double dhdy[2] = {0., 0.};

  double c_a[2] = {0., 0.};
  double c_b[2] = {0., 0.};

  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    dhdx[0] += dhdeta(basis[kk_off]->uu())*basis[kk_off]->duudx();
    dhdx[1] += dhdeta(basis[kk_off]->uuold())*basis[kk_off]->duuolddx();
    dhdy[0] += dhdeta(basis[kk_off]->uu())*basis[kk_off]->duudy();
    dhdy[1] += dhdeta(basis[kk_off]->uuold())*basis[kk_off]->duuolddy();
  }

  const double ct = (c[0]-c[1])/dt_*test;

  double eta_array[N_MAX];
  double eta_array_old[N_MAX];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    eta_array[kk] = basis[kk_off]->uu();
    eta_array_old[kk] = basis[kk_off]->uuold();
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
RES_FUNC_TPETRA(residual_eta_)
{
  //test function
  const double test = basis[0]->phi(i);
  //u, phi
  const double c[3] = {basis[ci_]->uu(), basis[ci_]->uuold(), basis[ci_]->uuoldold()};

  const double eta[3] = {basis[eqn_id]->uu(), basis[eqn_id]->uuold(), basis[eqn_id]->uuoldold()};

  const double divgradeta[3] = {L_*k_eta_*(basis[eqn_id]->duudx()*basis[0]->dphidx(i)
				    + basis[eqn_id]->duudy()*basis[0]->dphidy(i)
				    + basis[eqn_id]->duudz()*basis[0]->dphidz(i)),
				L_*k_eta_*(basis[eqn_id]->duuolddx()*basis[0]->dphidx(i)
				    + basis[eqn_id]->duuolddy()*basis[0]->dphidy(i)
				    + basis[eqn_id]->duuolddz()*basis[0]->dphidz(i)),
				L_*k_eta_*(basis[eqn_id]->duuoldolddx()*basis[0]->dphidx(i)
				    + basis[eqn_id]->duuoldolddy()*basis[0]->dphidy(i)
				    + basis[eqn_id]->duuoldolddz()*basis[0]->dphidz(i))};

  double eta_array[N_MAX];
  double eta_array_old[N_MAX];
  double eta_array_oldold[N_MAX];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    eta_array[kk] = basis[kk_off]->uu();
    eta_array_old[kk] = basis[kk_off]->uuold();
    eta_array_oldold[kk] = basis[kk_off]->uuoldold();
  }
  const int k = eqn_id - eqn_off_;

  const double df_deta[3] = {L_*(dfdeta(c[0],eta[0]) + w_*dgdeta(eta_array,k))*test,
			     L_*(dfdeta(c[1],eta[1]) + w_*dgdeta(eta_array_old,k))*test,
			     L_*(dfdeta(c[2],eta[2]) + w_*dgdeta(eta_array_oldold,k))*test};
  
  const double f[3] = {df_deta[0] + divgradeta[0],
		       df_deta[1] + divgradeta[1],
		       df_deta[2] + divgradeta[2]};

  const double ut = (eta[0]-eta[1])/dt_*test;

  return ut + (1.-t_theta2_)*t_theta_*f[0]
    + (1.-t_theta2_)*(1.-t_theta_)*f[1]
    +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_eta_dp_)) = residual_eta_;

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_eta_kks_)
{

  //derivatives of the test function
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  //double dtestdz = basis[0].dphidz(i);
  //test function
  const double test = basis[0]->phi(i);
  //u, phi
  const double c[2] = {basis[0]->uu(), basis[0]->uuold()};

  const double eta[2] = {basis[eqn_id]->uu(), basis[eqn_id]->uuold()};
  const double detadx[2] = {basis[eqn_id]->duudx(), basis[eqn_id]->duuolddx()};
  const double detady[2] = {basis[eqn_id]->duudy(), basis[eqn_id]->duuolddy()};

  double c_a[2] = {0., 0.};
  double c_b[2] = {0., 0.};

  double eta_array[N_MAX];
  double eta_array_old[N_MAX];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    eta_array[kk] = basis[kk_off]->uu();
    eta_array_old[kk] = basis[kk_off]->uuold();
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

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_mu_)
{
  //-mu + df/dc +div c grad test
  const double c = basis[ci_]->uu();
  const double mu = basis[mui_]->uu();
  //const double eta = basis[2]->uu();
  const double test = basis[0]->phi(i);

  const double divgradc = k_c_*(basis[ci_]->duudx()*basis[0]->dphidx(i)
				+ basis[ci_]->duudy()*basis[0]->dphidy(i)
				+ basis[ci_]->duudz()*basis[0]->dphidz(i));

  double eta_array[N_MAX];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    eta_array[kk] = basis[kk_off]->uu();
  };

  const double df_dc = dfdc(c,eta_array)*test;

  //return dt_*(-mu*test + df_dc + divgradc);
  return -mu*test + df_dc + divgradc;
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_mu_dp_)) = residual_mu_;

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_c_)
{
  //const int ci = 1;
  //const int mui = 0;
  const double D = k_c_;
  const double test = basis[0]->phi(i);
  //const double u_t =test * basis[0]->phi(j)/dt_;
  const double divgrad = D*(basis[0]->dphidx(j)*basis[0]->dphidx(i)
       + basis[0]->dphidy(j)*basis[0]->dphidy(i)
       + basis[0]->dphidz(j)*basis[0]->dphidz(i));
  const double d2 = d2fdc2()*basis[0]->phi(j)*test;
  return divgrad + d2;
}

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_mu_)
{
  const double D = M_;
  //const double test = basis[0]->phi(i);
  //const double u_t =test * basis[0]->phi(j)/dt_;
  const double divgrad = D*(basis[0]->dphidx(j)*basis[0]->dphidx(i)
       + basis[0]->dphidy(j)*basis[0]->dphidy(i)
       + basis[0]->dphidz(j)*basis[0]->dphidz(i));
  return divgrad;
}

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_eta_)
{
  const double ut = basis[0]->phi(j)/dt_*basis[0]->phi(i);
  const double divgrad = L_*k_eta_*(basis[0]->dphidx(j)*basis[0]->dphidx(i)
       + basis[0]->dphidy(j)*basis[0]->dphidy(i)
       + basis[0]->dphidz(j)*basis[0]->dphidz(i));
//   const double eta = basis[eqn_id]->uu();
//   const double c = basis[0]->uu();
//   const double g1 = L_*(2. - 12.*eta + 12.*eta*eta)*basis[0]->phi(j)*basis[0]->phi(i);
//   const double h1 = L_*(-f_alpha(c)+f_beta(c))*(60.*eta-180.*eta*eta+120.*eta*eta*eta)*basis[0]->phi(j)*basis[0]->phi(i);
  return ut + t_theta_*divgrad;// + t_theta_*(g1+h1);
}

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_ut_)
{
  const double test = basis[0]->phi(i);
  const double u_t =test * basis[0]->phi(j)/dt_;
  return u_t;
}

PPR_FUNC(postproc_c_a_)
{

  //cn will need eta_array here...
  const double cc = u[0];
  double phi = u[2];
  double c_a = 0.;
  double c_b = 0.;

  solve_kks(cc,&phi,c_a,c_b);

  return c_a;
}

PPR_FUNC(postproc_c_b_)
{
  //cn will need eta_array here...
  const double cc = u[0];
  double phi = u[2];
  double c_a = 0.;
  double c_b = 0.;

  solve_kks(cc,&phi,c_a,c_b);

  return c_b;
}

INI_FUNC(init_c_)
{
  return c0_ + eps_*(cos(0.105*x)*cos(0.11*y)
		     + cos(0.13*x)*cos(0.087*y)*cos(0.13*x)*cos(0.087*y)
		     + cos(0.025*x-0.15*y)*cos(0.07*x-0.02*y)
		     );
}

INI_FUNC(init_eta_)
{
  const double i = (double)(eqn_id - eqn_off_ + 1);
  return eps_eta_*std::pow(cos((0.01*i)*x-4.)*cos((0.007+0.01*i)*y)
			   + cos((0.11+0.01*i)*x)*cos((0.11+0.01*i)*y)		   
			   + psi_*std::pow(cos((0.046+0.001*i)*x+(0.0405+0.001*i)*y)
					   *cos((0.031+0.001*i)*x-(0.004+0.001*i)*y),2
					   ),2
			   );
}

INI_FUNC(init_mu_)
{
  //this will need the eta_array version
  const double c = init_c_(x,y,z,eqn_id,lid);
  const double eta = init_eta_(x,y,z,2,lid);
  return dfdc(c,&eta);
//   return 0.;
}

}//namespace pfhub2

namespace cahnhilliard
{
  //this has an mms described at:
  //https://www.sciencedirect.com/science/article/pii/S0021999112007243

  //residual_*_ is a traditional formulation for [c mu] with
  //R_c = c_t - divgrad mu
  //R_mu = -mu + df/dc - divgrad c

  //residual_*_trans_ utilizes a transformation as [mu c] with
  //R_mu = c_t - divgrad mu
  //R_c = -mu + df/dc - divgrad c
  //that puts the elliptic terms on the diagonal, with better solver convergence and 
  //potential for preconditioning
  //the transformation is inspired by:
  //https://web.archive.org/web/20220201192736id_/https://publikationen.bibliothek.kit.edu/1000141249/136305383
  //https://www.sciencedirect.com/science/article/pii/S037704271930319X

  //right now, the preconditioner can probably be improved by scaling c by dt

  double M = 1.;
  double Eps = 1.;
  double alpha = 1.;//alpha >= 1
  double pi = 3.141592653589793;
  double fcoef_ = 0.;

double F(const double &x,const double &t)
{
// Sin(a*Pi*x) 
//  - M*(Power(a,2)*Power(Pi,2)*(1 + t)*Sin(a*Pi*x) - Power(a,4)*Ep*Power(Pi,4)*(1 + t)*Sin(a*Pi*x) + 
//       6*Power(a,2)*Power(Pi,2)*Power(1 + t,3)*Power(Cos(a*Pi*x),2)*Sin(a*Pi*x) - 
//       3*Power(a,2)*Power(Pi,2)*Power(1 + t,3)*Power(Sin(a*Pi*x),3))

  double a = alpha;
  return sin(a*pi*x) 
    - M*(std::pow(a,2)*std::pow(pi,2)*(1 + t)*sin(a*pi*x) - std::pow(a,4)*Eps*std::pow(pi,4)*(1 + t)*sin(a*pi*x) + 
	 6*std::pow(a,2)*std::pow(pi,2)*std::pow(1 + t,3)*std::pow(cos(a*pi*x),2)*sin(a*pi*x) - 
	 3*std::pow(a,2)*std::pow(pi,2)*std::pow(1 + t,3)*std::pow(sin(a*pi*x),3));
}
double fp(const double &u)
{
  return u*u*u - u;
}

INI_FUNC(init_c_)
{
  return sin(alpha*pi*x);
}

INI_FUNC(init_mu_)
{
  //-Sin[a \[Pi] x] + a^2 \[Pi]^2 Sin[a \[Pi] x] + Sin[a \[Pi] x]^3
  return -sin(alpha*pi*x) + alpha*alpha*pi*pi*sin(alpha*pi*x) + sin(alpha*pi*x)*sin(alpha*pi*x)*sin(alpha*pi*x);
}

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_c_)
{
  //derivatives of the test function
  double dtestdx = basis[0]->dphidx(i);
  double dtestdy = basis[0]->dphidy(i);
  double dtestdz = basis[0]->dphidz(i);
  //test function
  double test = basis[0]->phi(i);
  double c = basis[0]->uu();
  double cold = basis[0]->uuold();
  double dmudx = basis[1]->duudx();
  double dmudy = basis[1]->duudy();
  double dmudz = basis[1]->duudz();
  double dmuolddx = basis[1]->duuolddx();
  double dmuolddy = basis[1]->duuolddy();
  double dmuolddz = basis[1]->duuolddz();
  double x = basis[0]->xx();

  double ct = (c - cold)/dt_*test;
  double divgradmu = M*t_theta_*(dmudx*dtestdx + dmudy*dtestdy + dmudz*dtestdz)
    + M*(1.-t_theta_)*(dmuolddx*dtestdx + dmuolddy*dtestdy + dmuolddz*dtestdz);
  double f = t_theta_*fcoef_*F(x,time)*test + (1.-t_theta_)*fcoef_*F(x,time-dt_)*test;

  return ct + divgradmu - f;
}

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_mu_)
{
  //derivatives of the test function
  double dtestdx = basis[0]->dphidx(i);
  double dtestdy = basis[0]->dphidy(i);
  double dtestdz = basis[0]->dphidz(i);
  //test function
  double test = basis[0]->phi(i);
  double c = basis[0]->uu();
  double dcdx = basis[0]->duudx();
  double dcdy = basis[0]->duudy();
  double dcdz = basis[0]->duudz();
  double mu = basis[1]->uu();

  double mut = mu*test;
  double f = fp(c)*test;
  double divgradc = Eps*(dcdx*dtestdx + dcdy*dtestdy + dcdz*dtestdz);

  return -mut + f + divgradc;
}

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_mu_trans_)
{
  //derivatives of the test function
  double dtestdx = basis[0]->dphidx(i);
  double dtestdy = basis[0]->dphidy(i);
  double dtestdz = basis[0]->dphidz(i);
  //test function
  double test = basis[0]->phi(i);
  double c = basis[1]->uu();
  double cold = basis[1]->uuold();
  double dmudx = basis[0]->duudx();
  double dmudy = basis[0]->duudy();
  double dmudz = basis[0]->duudz();
  double dmuolddx = basis[0]->duuolddx();
  double dmuolddy = basis[0]->duuolddy();
  double dmuolddz = basis[0]->duuolddz();
  double x = basis[0]->xx();

  double ct = (c - cold)/dt_*test;
  double divgradmu = M*t_theta_*(dmudx*dtestdx + dmudy*dtestdy + dmudz*dtestdz)
    + M*(1.-t_theta_)*(dmuolddx*dtestdx + dmuolddy*dtestdy + dmuolddz*dtestdz);
  double f = t_theta_*fcoef_*F(x,time)*test + (1.-t_theta_)*fcoef_*F(x,time-dt_)*test;

  return ct + divgradmu - f;
}

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_c_trans_)
{
  //derivatives of the test function
  double dtestdx = basis[0]->dphidx(i);
  double dtestdy = basis[0]->dphidy(i);
  double dtestdz = basis[0]->dphidz(i);
  //test function
  double test = basis[0]->phi(i);
  double c = basis[1]->uu();
  double dcdx = basis[1]->duudx();
  double dcdy = basis[1]->duudy();
  double dcdz = basis[1]->duudz();
  double mu = basis[0]->uu();

  double mut = mu*test;
  double f = fp(c)*test;
  double divgradc = Eps*(dcdx*dtestdx + dcdy*dtestdy + dcdz*dtestdz);

  return -mut + f + divgradc;
}

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_mu_trans_)
{
  const double divgradmu = M*t_theta_*(basis[0]->dphidx(j)*basis[0]->dphidx(i)
       + basis[0]->dphidy(j)*basis[0]->dphidy(i)
       + basis[0]->dphidz(j)*basis[0]->dphidz(i));
  return divgradmu;
}

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_c_trans_)
{
  const double divgradc = Eps*(basis[0]->dphidx(j)*basis[0]->dphidx(i)
       + basis[0]->dphidy(j)*basis[0]->dphidy(i)
       + basis[0]->dphidz(j)*basis[0]->dphidz(i));
  return divgradc;
}

PARAM_FUNC(param_)
{
  fcoef_ = plist->get<double>("fcoef");
}
}//namespace cahnhilliard

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
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  //test function
  const double test = basis[0]->phi(i);
  //u, phi
  const double u = basis[0]->uu();
  const double uold = basis[0]->uuold();

  //double a =10.;

  const double ut = (u-uold)/dt_*test;
 
  const double f[3] = {c*c*(basis[0]->duudx()*dtestdx + basis[0]->duudy()*dtestdy + basis[0]->duudz()*dtestdz),
		       c*c*(basis[0]->duuolddx()*dtestdx + basis[0]->duuolddy()*dtestdy + basis[0]->duuolddz()*dtestdz),
		       c*c*(basis[0]->duuoldolddx()*dtestdx + basis[0]->duuoldolddy()*dtestdy + basis[0]->duuoldolddz()*dtestdz)};
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
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);

  const double dbasisdx = basis[0]->dphidx(j);
  const double dbasisdy = basis[0]->dphidy(j);
  const double dbasisdz = basis[0]->dphidz(j);
  const double test = basis[0]->phi(i);
  const double divgrad = c*c*(dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz);
  //double a =10.;
  const double u_t = (basis[0]->phi(j))/dt_*test;
  return u_t + t_theta_*divgrad;
}

TUSAS_DEVICE
PRE_FUNC_TPETRA((*prec_robin_test_dp_)) = prec_robin_test_;

NBC_FUNC_TPETRA(nbc_robin_test_)
{

  const double test = basis[0].phi(i);

  //du/dn + kappa u = g = 0 on L
  //(du,dv) - <du/dn,v> = (f,v)
  //(du,dv) - <g - kappa u,v> = (f,v)
  //(du,dv) - < - kappa u,v> = (f,v)
  //          ^^^^^^^^^^^^^^ return this
  const double f[3] = {-kappa*basis[0].uu()*test,
		       -kappa*basis[0].uuold()*test,
		       -kappa*basis[0].uuoldold()*test};
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
  const double test = basis[0]->phi(i);
  //u, phi
  const double u = basis[0]->uu();
  const double uold = basis[0]->uuold();
  const double uoldold = basis[0]->uuoldold();

  const double ut = (u-uold)/dt_*test;
  //std::cout<<ut<<" "<<dt_<<" "<<time<<std::endl;
 
  double f[3];
  f[0] = (-k1*u       - k2*u*basis[1]->uu())*test;
  f[1] = (-k1*uold    - k2*u*basis[1]->uuold())*test;
  f[2] = (-k1*uoldold - k2*u*basis[1]->uuoldold())*test;

  return ut - (1.-t_theta2_)*t_theta_*f[0]
    - (1.-t_theta2_)*(1.-t_theta_)*f[1]
    -.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

RES_FUNC_TPETRA(residual_b_)
{
  const double test = basis[0]->phi(i);
  //u, phi
  const double u = basis[1]->uu();
  const double uold = basis[1]->uuold();
  //const double uoldold = basis[1]->uuoldold();
  const double a = basis[0]->uu();
  const double aold = basis[0]->uuold();
  const double aoldold = basis[0]->uuoldold();

  const double ut = (u-uold)/dt_*test;
  double f[3];
  f[0] = (k1*a       - k2*a*u                        + 2.*k3*basis[2]->uu())*test;
  f[1] = (k1*aold    - k2*aold*uold                  + 2.*k3*basis[2]->uuold())*test;
  f[2] = (k1*aoldold - k2*aoldold*basis[1]->uuoldold() + 2.*k3*basis[2]->uuoldold())*test;

  return ut - (1.-t_theta2_)*t_theta_*f[0]
    - (1.-t_theta2_)*(1.-t_theta_)*f[1]
    -.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

RES_FUNC_TPETRA(residual_ab_)
{
  const double test = basis[0]->phi(i);
  //u, phi
  const double u = basis[2]->uu();
  const double uold = basis[2]->uuold();
  //const double uoldold = basis[1]->uuoldold();
  const double b = basis[1]->uu();
  const double bold = basis[1]->uuold();
  const double boldold = basis[1]->uuoldold();

  const double ut = (u-uold)/dt_*test;
  double f[3];
  f[0] = (k2*b*basis[0]->uu()             - k3*u)*test;
  f[1] = (k2*bold*basis[0]->uuold()       - k3*uold)*test;
  f[2] = (k2*boldold*basis[0]->uuoldold() - k3*basis[2]->uuoldold())*test;

  return ut - (1.-t_theta2_)*t_theta_*f[0]
    - (1.-t_theta2_)*(1.-t_theta_)*f[1]
    -.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

RES_FUNC_TPETRA(residual_c_)
{
  const double test = basis[0]->phi(i);
  //u, phi
  const double u = basis[3]->uu();
  const double uold = basis[3]->uuold();
  //const double uoldold = basis[1]->uuoldold();
  const double a = basis[0]->uu();
  const double aold = basis[0]->uuold();
  const double aoldold = basis[0]->uuoldold();

  const double ut = (u-uold)/dt_*test;
  double f[3];
  f[0] = (k1*a       + k3*basis[2]->uu())*test;
  f[1] = (k1*aold    + k3*basis[2]->uuold())*test;
  f[2] = (k1*aoldold + k3*basis[2]->uuoldold())*test;

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
  const double test = basis[0]->phi(i);
  //u, phi
  const double u[3] = {basis[0]->uu(),basis[0]->uuold(),basis[0]->uuoldold()};

  const double ut = (u[0]-u[1])/dt_*test;

  const double f[3] = {ff(u[0])*test,ff(u[1])*test,ff(u[2])*test};
 
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

  double tau0_h = 1.;
  double W0_h = 1.;
  
  double deltau_h = 1.;
  double uref_h = 0.;

  double scaling_constant = 1.0;

DBC_FUNC(dbc_) 
{
  return 1173.;
}

NBC_FUNC_TPETRA(nbc_)
{
  //https://reference.wolfram.com/language/PDEModels/tutorial/HeatTransfer/HeatTransfer.html#2048120463
  //h(t-ti)+\ep\sigma(t^4-ti^4) = -g(t)
  //du/dn = g
  //return g*test here

  //std::cout<<h<<" "<<ep<<" "<<sigma<<" "<<ti<<std::endl;
  const double test = basis[0].phi(i);
  const double u = deltau_h*basis[0].uu()+uref_h; // T=deltau_h*theta+uref_h
  const double uold = deltau_h*basis[0].uuold()+uref_h;
  const double uoldold = deltau_h*basis[0].uuoldold()+uref_h;
#if 1
  const double f[3] = {(h*(ti-u)+ep*sigma*(ti*ti*ti*ti-u*u*u*u))*test,
		       (h*(ti-uold)+ep*sigma*(ti*ti*ti*ti-uold*uold*uold*uold))*test,
		       (h*(ti-uoldold)+ep*sigma*(ti*ti*ti*ti-uoldold*uoldold*uoldold*uoldold))*test};
#else
  const double c = h+4.*ep*sigma*ti*ti*ti;
  const double f[3] = {(c*(ti-u))*test,
		       (c*(ti-uold))*test,
		       (c*(ti-uoldold))*test};
#endif  
  const double coef = deltau_h / W0_h;
  //std::cout<<f[0]<<" "<<f[1]<<" "<<f[2]<<std::endl;
  const double rv = (1.-t_theta2_)*t_theta_*f[0]
    +(1.-t_theta2_)*(1.-t_theta_)*f[1]
    +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
  
  return f[0] * coef * scaling_constant;
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
  deltau_h = plist->get<double>("deltau_",1.);
  uref_h = plist->get<double>("uref_",0.); 

  W0_h = plist->get<double>("W0_",1.);

  scaling_constant = plist->get<double>("scaling_constant_",1.);
//   std::cout<<"tpetra::radconvbc::param_:"<<std::endl
// 	   <<"  h     = "<<h<<std::endl
// 	   <<"  ep    = "<<ep<<std::endl
// 	   <<"  sigma = "<<sigma<<std::endl
// 	   <<"  ti    = "<<ti<<std::endl<<std::endl;
}
}//namespace radconvbc

namespace goldak{
TUSAS_DEVICE
const double pi_d = 3.141592653589793;

double te = 1641.;
double tl = 1706.;
double Lf = 2.95e5;

TUSAS_DEVICE
double dfldu_mushy_d = tpetra::heat::rho_d*Lf/(tl-te);//fl=(t-te)/(tl-te);

TUSAS_DEVICE
double eta_d = 0.3;
TUSAS_DEVICE
double P_d = 50.;
TUSAS_DEVICE
double s_d = 2.;
TUSAS_DEVICE
double r_d = .00005;
TUSAS_DEVICE
double d_d = .00001;
TUSAS_DEVICE
double gamma_d = 0.886227;
TUSAS_DEVICE
double x0_d = 0.;
TUSAS_DEVICE
double y0_d = 0.;
TUSAS_DEVICE
double z0_d = 0.;
TUSAS_DEVICE
double t_hold_d = 0.005;
TUSAS_DEVICE
double t_decay_d = 0.01;
TUSAS_DEVICE
double tau0_d = 1.;
TUSAS_DEVICE
double W0_d = 1.;

TUSAS_DEVICE
double t0_d = 300.;

TUSAS_DEVICE
double scaling_constant_d = 1.;

KOKKOS_INLINE_FUNCTION 
void dfldt_uncoupled(GPUBasis * basis[], const int index, const double dt_, const double dtold_, double *a)
{
  //the latent heat term is zero outside of the mushy region (ie outside Te < T < Tl)

  //we need device versions of deltau_h,uref_h
  const double coef = 1./tau0_d;

  const double tt[3] = {tpetra::heat::deltau_h*basis[index]->uu()+tpetra::heat::uref_h,
			tpetra::heat::deltau_h*basis[index]->uuold()+tpetra::heat::uref_h,
			tpetra::heat::deltau_h*basis[index]->uuoldold()+tpetra::heat::uref_h};
  const double dfldu_d[3] = {((tt[0] > te) && (tt[0] < tl)) ? coef*dfldu_mushy_d : 0.0,
			     ((tt[1] > te) && (tt[1] < tl)) ? coef*dfldu_mushy_d : 0.0,
			     ((tt[2] > te) && (tt[2] < tl)) ? coef*dfldu_mushy_d : 0.0};

  a[0] = ((1. + dt_/dtold_)*(dfldu_d[0]*basis[index]->uu()-dfldu_d[1]*basis[index]->uuold())/dt_
                                 -dt_/dtold_*(dfldu_d[0]*basis[index]->uu()-dfldu_d[2]*basis[index]->uuoldold())/(dt_+dtold_)
                                 );
  a[1] = (dtold_/dt_/(dt_+dtold_)*(dfldu_d[0]*basis[index]->uu())
                                 -(dtold_-dt_)/dt_/dtold_*(dfldu_d[1]*basis[index]->uuold())
                                 -dt_/dtold_/(dt_+dtold_)*(dfldu_d[2]*basis[index]->uuoldold())
                                 );
  a[2] = (-(1.+dtold_/dt_)*(dfldu_d[2]*basis[index]->uuoldold()-dfldu_d[1]*basis[index]->uuold())/dtold_
                                 +dtold_/dt_*(dfldu_d[2]*basis[index]->uuoldold()-dfldu_d[0]*basis[index]->uu())/(dtold_+dt_)
                                 );
  return;
}

KOKKOS_INLINE_FUNCTION 
void dfldt_coupled(GPUBasis * basis[], const int index, const double dt_, const double dtold_, double *a)
{
  const double coef = tpetra::heat::rho_d*tpetra::goldak::Lf/tau0_d;
  const double dfldu_d[3] = {-.5*coef,-.5*coef,-.5*coef};

  a[0] = ((1. + dt_/dtold_)*(dfldu_d[0]*basis[index]->uu()-dfldu_d[1]*basis[index]->uuold())/dt_
                                 -dt_/dtold_*(dfldu_d[0]*basis[index]->uu()-dfldu_d[2]*basis[index]->uuoldold())/(dt_+dtold_)
                                 );
  a[1] = (dtold_/dt_/(dt_+dtold_)*(dfldu_d[0]*basis[index]->uu())
                                 -(dtold_-dt_)/dt_/dtold_*(dfldu_d[1]*basis[index]->uuold())
                                 -dt_/dtold_/(dt_+dtold_)*(dfldu_d[2]*basis[index]->uuoldold())
                                 );
  a[2] = (-(1.+dtold_/dt_)*(dfldu_d[2]*basis[index]->uuoldold()-dfldu_d[1]*basis[index]->uuold())/dtold_
                                 +dtold_/dt_*(dfldu_d[2]*basis[index]->uuoldold()-dfldu_d[0]*basis[index]->uu())/(dtold_+dt_)
                                 );
  return;
}

KOKKOS_INLINE_FUNCTION 
const double P(const double t)
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

  const double p = P(t);
  const double r = r_d;
  const double d = d_d;
  
  //s_d = 2 below; we can simplify this expression 5.19615=3^1.5
  const double coef = eta_d*p*5.19615/r/r/d/gamma_d/pi_d;
  const double exparg = ((W0_d*x-x0_d)*(W0_d*x-x0_d)+(W0_d*y-y0_d)*(W0_d*y-y0_d))/r/r+(W0_d*z-z0_d)*(W0_d*z-z0_d)/d/d;
  const double f = exp( -3.* exparg );

  return coef*f;
}

KOKKOS_INLINE_FUNCTION
RES_FUNC_TPETRA(residual_test_)
{
  //u_t,v + grad u,grad v - qdot,v = 0

  double val = tpetra::heat::residual_heat_test_dp_(basis,
						    i,
						    dt_,
						    dtold_,
						    t_theta_,
						    t_theta2_,
						    time,
						    eqn_id,
						    vol,
						    rand);

  const double qd[3] = {-qdot(basis[0]->xx(),basis[0]->yy(),basis[0]->zz(),time)*basis[0]->phi(i),
			-qdot(basis[0]->xx(),basis[0]->yy(),basis[0]->zz(),time-dt_)*basis[0]->phi(i),
			-qdot(basis[0]->xx(),basis[0]->yy(),basis[0]->zz(),time-dt_-dtold_)*basis[0]->phi(i)};

  const double rv = (val 
		     + (1.-t_theta2_)*t_theta_*qd[0]
		     + (1.-t_theta2_)*(1.-t_theta_)*qd[1]
		     +.5*t_theta2_*((2.+dt_/dtold_)*qd[1]-dt_/dtold_*qd[2]));

  return rv;
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_test_dp_)) = residual_test_;

KOKKOS_INLINE_FUNCTION
RES_FUNC_TPETRA(residual_uncoupled_test_)
{
  //u_t,v + grad u,grad v + dfldt,v - qdot,v = 0

  double val = tpetra::goldak::residual_test_dp_(basis,
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

  const double dfldt[3] = {dfldu_d[0]*basis[0]->phi(i),
			   dfldu_d[1]*basis[0]->phi(i),
			   dfldu_d[2]*basis[0]->phi(i)};
  
  const double rv = (val 
		     + (1.-t_theta2_)*t_theta_*dfldt[0]
		     + (1.-t_theta2_)*(1.-t_theta_)*dfldt[1]
		     +.5*t_theta2_*((2.+dt_/dtold_)*dfldt[1]-dt_/dtold_*dfldt[2]));

  return rv * scaling_constant_d;
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_uncoupled_test_dp_)) = residual_uncoupled_test_;

KOKKOS_INLINE_FUNCTION
RES_FUNC_TPETRA(residual_coupled_test_)
{
  //u_t,v + grad u,grad v + dfldt,v - qdot,v = 0

  double val = tpetra::goldak::residual_test_dp_(basis,
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

  const double dfldt[3] = {dfldu_d[0]*basis[0]->phi(i),
			   dfldu_d[1]*basis[0]->phi(i),
			   dfldu_d[2]*basis[0]->phi(i)};
  
  const double rv = (val 
		     + (1.-t_theta2_)*t_theta_*dfldt[0]
		     + (1.-t_theta2_)*(1.-t_theta_)*dfldt[1]
		     +.5*t_theta2_*((2.+dt_/dtold_)*dfldt[1]-dt_/dtold_*dfldt[2]));
  
  return rv * scaling_constant_d;
}

TUSAS_DEVICE
RES_FUNC_TPETRA((*residual_coupled_test_dp_)) = residual_coupled_test_;

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_test_)
{
  
  const double val = tpetra::heat::prec_heat_test_dp_(basis,
						      i,
						      j,
						      dt_,
						      t_theta_,
						      eqn_id);

  return val * scaling_constant_d;
}

TUSAS_DEVICE
PRE_FUNC_TPETRA((*prec_test_dp_)) = prec_test_;

INI_FUNC(init_heat_)
{
  const double t_preheat = t0_d;
  const double val = (t_preheat-tpetra::heat::uref_h)/tpetra::heat::deltau_h;
  return val;
}

DBC_FUNC(dbc_) 
{
  // The assumption here is that the desired Dirichlet BC is the initial temperature,
  // that may not be true in the future.
  const double t_preheat = t0_d;
  const double val = (t_preheat-tpetra::heat::uref_h)/tpetra::heat::deltau_h;
  return val;
}

PPR_FUNC(postproc_qdot_)
{
  const double x = xyz[0];
  const double y = xyz[1];
  const double z = xyz[2];

  return qdot(x,y,z,time);
}

PPR_FUNC(postproc_u_)
{
  return u[0]*tpetra::heat::deltau_h + tpetra::heat::uref_h;
}

PARAM_FUNC(param_)
{
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
  te = plist->get<double>("te_",1641.);
  //tl = 1706.;// K
  tl = plist->get<double>("tl_",1706.);
  //Lf = 17.2;// kJ/mol
  Lf = plist->get<double>("Lf_",2.95e5);

  //eta_d = 0.3;//dimensionless
  eta_d = plist->get<double>("eta_",0.3);
  //P_d = 50.;// W
  P_d = plist->get<double>("P_",50.);
  //s_d = 2.;//dimensionless
  s_d = plist->get<double>("s_",2.);
  //r_d = .005;// 50 um
  r_d = plist->get<double>("r_",.00005);
  //d_d = .001;// 10 um
  d_d = plist->get<double>("d_",.00001);
  //gamma_d = is gamma function
  //gamma(3/s):
  //gamma(3/2) = sqrt(pi)/2
  gamma_d = plist->get<double>("gamma_",0.886227);
  x0_d = plist->get<double>("x0_",0.);
  y0_d = plist->get<double>("y0_",0.);
  z0_d = plist->get<double>("z0_",0.);
  t_hold_d = plist->get<double>("t_hold_",0.005);
  t_decay_d = plist->get<double>("t_decay_",0.01);
  tau0_d = plist->get<double>("tau0_",1.);
  W0_d = plist->get<double>("W0_",1.);
  
  t0_d = plist->get<double>("t0_",300.);

  dfldu_mushy_d = tpetra::heat::rho_d*Lf/(tl-te); //fl=(t-te)/(tl-te);
  
  scaling_constant_d = plist->get<double>("scaling_constant_",1.);

}
}//namespace goldak

namespace fullycoupled
{
  double hemisphere_IC_rad = 1.0;
  double hemispherical_IC_x0 = 0.0;
  double hemispherical_IC_y0 = 0.0;
  double hemispherical_IC_z0 = 0.0;
  bool hemispherical_IC = false;	
  
INI_FUNC(init_conc_farzadi_)
{
  return -1.;
}

INI_FUNC(init_phase_farzadi_)
{
  if (hemispherical_IC){
	  const double w0 = tpetra::farzadi3d::w0;
	  
	  const double dist = std::sqrt( (x-hemispherical_IC_x0/w0)*(x-hemispherical_IC_x0/w0) 
	  	+ (y-hemispherical_IC_y0/w0)*(y-hemispherical_IC_y0/w0) 
	  	+ (z-hemispherical_IC_z0/w0)*(z-hemispherical_IC_z0/w0));
	  const double r = hemisphere_IC_rad/w0 + tpetra::farzadi3d::amplitude*((double)rand()/(RAND_MAX));
	  return std::tanh( (dist-r)/std::sqrt(2.));
  }
  else {
	  double h = tpetra::farzadi3d::base_height + tpetra::farzadi3d::amplitude*((double)rand()/(RAND_MAX));
	  
	  return std::tanh((h-z)/std::sqrt(2.));
	  
	  double c = (x-tpetra::farzadi3d::x0)*(x-tpetra::farzadi3d::x0) + (y-tpetra::farzadi3d::y0)*(y-tpetra::farzadi3d::y0) + (z-tpetra::farzadi3d::z0)*(z-tpetra::farzadi3d::z0);
	  return ((tpetra::farzadi3d::C == 0) ? (tanh((h-z)/sqrt(2.))) : (c < tpetra::farzadi3d::r*tpetra::farzadi3d::r) ? 1. : -1.);	
  }
}

INI_FUNC(init_heat_)
{
  const double t_preheat = tpetra::goldak::t0_d;
  const double val = (t_preheat-tpetra::heat::uref_h)/tpetra::heat::deltau_h;
  return val;
}

DBC_FUNC(dbc_) 
{
  // The assumption here is that the desired Dirichlet BC is the initial temperature,
  // that may not be true in the future.
  const double t_preheat = tpetra::goldak::t0_d;
  const double val = (t_preheat-tpetra::heat::uref_h)/tpetra::heat::deltau_h;
  return val;
}

PPR_FUNC(postproc_t_)
{
  // return the physical temperature in K here
  const double theta = u[2];
  return theta * tpetra::heat::deltau_h + tpetra::heat::uref_h;
}

PARAM_FUNC(param_)
{
	hemispherical_IC = plist->get<bool>("hemispherical_IC", false);
	hemisphere_IC_rad = plist->get<double>("hemisphere_IC_rad", 1.0);
	hemispherical_IC_x0 = plist->get<double>("hemispherical_IC_x0", 0.0);
	hemispherical_IC_y0 = plist->get<double>("hemispherical_IC_y0", 0.0);
	hemispherical_IC_z0 = plist->get<double>("hemispherical_IC_z0", 0.0);
}
}//namespace fullycoupled

namespace quaternion
{
  //see
  //[1] LLNL-JRNL-409478 A Numerical Algorithm for the Solution of a Phase-Field Model of Polycrystalline Materials
  //M. R. Dorr, J.-L. Fattebert, M. E. Wickett, J. F. Belak, P. E. A. Turchi (2008)
  //[2] LLNL-JRNL-636233 Phase-field modeling of coring during solidification of Au-Ni alloy using quaternions and CALPHAD input
  //J. L. Fattebert, M. E. Wickett, P. E. A. Turchi May 7, 2013

  //cn open question on how to choose beta, needs study
  const double beta = 1.e-32;
  const double ff = 1.;
  const double cc = 1.;

  //cn seems it is critical that initial q boundaries are ahead of the front given by 2 H T p'(phi)/grad q
  //this term, in conjunction with beta is used to keep the q evolution flat as discussed in refs
  //right now we put the grain boundary at .064 and phi at .064 - sqrt(2)*dx
  const double halfdx = .001;
  const double dx = .002;
  //double r0 = .064 - 1.*sqrt(2.)*dx;
  double r0 = .064 + 0.*sqrt(2.)*dx;

  //const double Mq = 1.;//1/sec/pJ
  const double Mq = 3.;//1/sec/pJ
  const double Mphi = 1.;//1/sec/pJ
  //const double Mmax = .64;//1/sec/pJ
  //const double Mmin= 1.e-6;
  const double Dmax = 1000.;
  //const double Dmin = 1.e-6;
  const double epq = .0477;//(pJ/um)^1/2
  //const double epq = .0;//(pJ/um)^1/2
  const double epphi = .083852;//(pJ/um)^1/2
  //                          const double L = 2.e9;//J/m^3   =======>
  const double L = 2000.;//pJ/um^3
  const double omega = 31.25;//pJ/um^3
  //const double H = .884e-3;//pJ/K/um^2
  double H = .884e-3;//pJ/K/um^2
  //const double T = 975.;//K
  const double T = 925.;//K
  const double Tm = 1025.;//K

  const int N = 4;
  const int qid = 1;
  const int phid = 0;
  const double pi = 3.141592653589793;

  const double dist(const double &x, const double &y, const double &z,
	   const double &x0, const double &y0, const double &z0,
	   const double &r)
  {
    const double c = (x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0);
    return r-sqrt(c);
  }
  const double p(const double &phi)
  {
    return phi*phi;
  }
  const double pp(const double &phi)
  {
    return 2.*phi;
  }
  const double h(const double &phi)
  {
    return phi*phi*phi*(10.-15.*phi+6.*phi*phi);
  }
  const double hp(const double &phi)
  {
    return 3.*phi*phi*(10.-15.*phi+6.*phi*phi)+phi*phi*phi*(-15.+12.*phi);
  }
  const double g(const double &phi)
  {
    return 16.*phi*phi*(1.-phi)*(1.-phi);
  }
  const double gp(const double &phi)
  {
    return 32.*phi*(1.-3.*phi+2.*phi*phi);
  }
  const double norm( const double &u1, const double &u2, const double &u3, const double &u4)
  {
    return sqrt(u1*u1+u2*u2+u3*u3+u4*u4);
  }
  const double normsquared( const double &u1, const double &u2, const double &u3, const double &u4)
  {
    return u1*u1+u2*u2+u3*u3+u4*u4;
  }
    
KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_)
{
  //derivatives of the test function
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  //test function
  const double test = basis[0]->phi(i);
  //u, phi
  const double u[3] = {basis[eqn_id]->uu(),basis[eqn_id]->uuold(),basis[eqn_id]->uuoldold()};

  const double ut = (u[0]-u[1])/dt_*test;

  const double phi[3] = {basis[phid]->uu(),basis[phid]->uuold(),basis[phid]->uuoldold()};
  const double m[3] = {1.-h(phi[0]),1.-h(phi[1]),1.-h(phi[2])};
  double M[3] = {m[0]*(Mq-1e-6)+1e-6,m[1]*(Mq-1e-6)+1e-6,m[2]*(Mq-1e-6)+1e-6};
  //double M[3] = {Mq,Mq,Mq};

  const double ep2[3] = {epq*epq,epq*epq,epq*epq};
  //const double ep[3] = {epq*epq*p(phi[0]),epq*epq*p(phi[1]),epq*epq*p(phi[2])};

  const double divgradu[3] = {(basis[eqn_id]->duudx()*dtestdx + basis[eqn_id]->duudy()*dtestdy + basis[eqn_id]->duudz()*dtestdz),
			      (basis[eqn_id]->duuolddx()*dtestdx + basis[eqn_id]->duuolddy()*dtestdy + basis[eqn_id]->duuolddz()*dtestdz),
			      (basis[eqn_id]->duuoldolddx()*dtestdx + basis[eqn_id]->duuoldolddy()*dtestdy + basis[eqn_id]->duuoldolddz()*dtestdz)};
  const double mep[3] = {M[0]*ep2[0],M[1]*ep2[1],M[2]*ep2[2]};

  double divgraduk[3] = {0., 0., 0.};
  double normq2[3] = {0., 0., 0.};

  for(int k = qid; k < N+qid; k++){
    //sum_k q_k grad q_k dot grad test
    divgraduk[0] = divgraduk[0] + basis[k]->uu()*(basis[k]->duudx()*dtestdx + basis[k]->duudy()*dtestdy + basis[k]->duudz()*dtestdz);
    divgraduk[1] = divgraduk[1] + basis[k]->uuold()*(basis[k]->duuolddx()*dtestdx + basis[k]->duuolddy()*dtestdy + basis[k]->duuolddz()*dtestdz);
    divgraduk[2] = divgraduk[2] + basis[k]->uuoldold()*(basis[k]->duuoldolddx()*dtestdx + basis[k]->duuoldolddy()*dtestdy + basis[k]->duuoldolddz()*dtestdz);
    //sum_k q_k^2 = norm q
    normq2[0] = normq2[0] + basis[k]->uu()*basis[k]->uu();
    normq2[1] = normq2[1] + basis[k]->uuold()*basis[k]->uuold();
    normq2[2] = normq2[2] + basis[k]->uuoldold()*basis[k]->uuoldold();
  }
  //it is possible that normq2 could be zero in the interface...
  double b2 = beta;
  normq2[0] = normq2[0] + b2;
  normq2[1] = normq2[1] + b2;
  normq2[2] = normq2[2] + b2;

  //   - q/normq^2 sum q_k ep^2 grad q_k grad test
  divgraduk[0] = u[0]*divgraduk[0]/normq2[0];
  divgraduk[1] = u[1]*divgraduk[1]/normq2[1];
  divgraduk[2] = u[2]*divgraduk[2]/normq2[2];

  //we need norm grad q here
  //grad q is nonzero only within interfaces
  double normgradq[3] = {0.,0.,0.};
  for(int k = qid; k < N+qid; k++){
    normgradq[0] = normgradq[0] + basis[k]->duudx()*basis[k]->duudx() + basis[k]->duudy()*basis[k]->duudy() + basis[k]->duudz()*basis[k]->duudz();
    normgradq[1] = normgradq[1] + basis[k]->duuolddx()*basis[k]->duuolddx() + basis[k]->duuolddy()*basis[k]->duuolddy() + basis[k]->duuolddz()*basis[k]->duuolddz();
    normgradq[2] = normgradq[2] + basis[k]->duuoldolddx()*basis[k]->duuoldolddx() + basis[k]->duuoldolddy()*basis[k]->duuoldolddy() + basis[k]->duuoldolddz()*basis[k]->duuoldolddz();
  }
  //std::cout<<normgradq[0]<<std::endl;
  //b2 = sqrt(beta);
  b2 = beta;
  normgradq[0] = sqrt(b2 + normgradq[0]);
  normgradq[1] = sqrt(b2 + normgradq[1]);
  normgradq[2] = sqrt(b2 + normgradq[2]);

  double Dq[3] = {2.*T*H*p(phi[0])/normgradq[0],2.*T*H*p(phi[1])/normgradq[1],2.*T*H*p(phi[2])/normgradq[2]};

  const double mdq[3] = {M[0]*Dq[0], M[1]*Dq[1], M[2]*Dq[2]};

  const double f[3] = {(mep[0] + mdq[0])*divgradu[0] - ff*(mep[0] + mdq[0])*divgraduk[0], 
		       (mep[1] + mdq[1])*divgradu[1] - ff*(mep[1] + mdq[1])*divgraduk[1], 
		       (mep[2] + mdq[2])*divgradu[2] - ff*(mep[2] + mdq[2])*divgraduk[2]};

  double val= (ut + (1.-t_theta2_)*t_theta_*f[0]
	       + (1.-t_theta2_)*(1.-t_theta_)*f[1]
	       +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]));
//   if(val!=val){
//     std::cout<<divgradu[0]<<" "<<divgradu[1]<<" "<<divgradu[2]<<std::endl;
//   }
//   std::cout<<val<<" "<<i<<std::endl;
  //val = 0.;
  return val;
}
  
KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(precon_)
{
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);

  const double phi = basis[phid]->uu();
  const double u = basis[eqn_id]->uu();

  double m = 1.-h(phi);
  double M = m*(Mq+1e-6)+1e-6;
  //double M = Mq;
  //const double ep = epq*epq*p(phi);
  const double ep2 = epq*epq;

  const double ut = basis[0]->phi(j)/dt_*basis[0]->phi(i);
  const double divgradu = (basis[0]->dphidx(j)*basis[0]->dphidx(i)
			   + basis[0]->dphidy(j)*basis[0]->dphidy(i)
			   + basis[0]->dphidz(j)*basis[0]->dphidz(i));
  double normq2 = 0.;

  for(int k = qid; k < N+qid; k++){
    //sum_k q_k^2 = norm q
    normq2 = normq2 + basis[k]->uu()*basis[k]->uu();
  }
  //it is possible that normq2 could be zero in the interface...
  double b2 = beta;
  normq2 = normq2 + b2;

  const double divgraduk1 = -u*u*(basis[0]->dphidx(j)*basis[0]->dphidx(i)
				       + basis[0]->dphidy(j)*basis[0]->dphidy(i)
				       + basis[0]->dphidz(j)*basis[0]->dphidz(i))/normq2;

  const double divgraduk2 = -u*basis[0]->phi(j)*(basis[eqn_id]->duudx()*dtestdx + basis[eqn_id]->duudy()*dtestdy + basis[eqn_id]->duudz()*dtestdz)/normq2;

  double normgradq = 0.;
  for(int k = qid; k < N+qid; k++){
    normgradq = normgradq + basis[k]->duudx()*basis[k]->duudx() + basis[k]->duudy()*basis[k]->duudy() + basis[k]->duudz()*basis[k]->duudz();
  }
  //b2 = sqrt(beta);
  b2 = beta;

  normgradq = sqrt(b2 + normgradq);
  double Dq = 2.*T*H*p(phi)/normgradq;

  const double f = (M*ep2 + M*Dq)*divgradu + ff*(M*ep2 + M*Dq)*(divgraduk1 + divgraduk2);

  return (ut + t_theta_*f);
}

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_phase_)
{
  //derivatives of the test function
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  //test function
  const double test = basis[0]->phi(i);
  //u, phi
  const double phi[3] = {basis[eqn_id]->uu(), basis[eqn_id]->uuold(), basis[eqn_id]->uuoldold()};

  const double phit = (phi[0]-phi[1])/dt_*test;

  //M eps^2 grad phi rrad test
  const double divgradu[3] = {Mphi*epphi*epphi*(basis[eqn_id]->duudx()*dtestdx + basis[eqn_id]->duudy()*dtestdy + basis[eqn_id]->duudz()*dtestdz),
			      Mphi*epphi*epphi*(basis[eqn_id]->duuolddx()*dtestdx + basis[eqn_id]->duuolddy()*dtestdy + basis[eqn_id]->duuolddz()*dtestdz),
			      Mphi*epphi*epphi*(basis[eqn_id]->duuoldolddx()*dtestdx + basis[eqn_id]->duuoldolddy()*dtestdy + basis[eqn_id]->duuoldolddz()*dtestdz)};

  const double ww[3] = {Mphi*gp(phi[0])*test,
			Mphi*gp(phi[1])*test,
			Mphi*gp(phi[2])*test};

  const double hh[3] = {(Mphi*hp(phi[0])*L*(T-Tm)/Tm*test),
			(Mphi*hp(phi[1])*L*(T-Tm)/Tm*test),
			(Mphi*hp(phi[2])*L*(T-Tm)/Tm*test)};

  const double f[3] = {divgradu[0] + ww[0] + hh[0],
		       divgradu[1] + ww[1] + hh[1],
		       divgradu[2] + ww[2] + hh[2]};

  return phit + (1.-t_theta2_)*t_theta_*f[0]
    + (1.-t_theta2_)*(1.-t_theta_)*f[1]
    +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_phi_)
{
  const double test = basis[0]->phi(i);

  double val = residual_phase_(basis,
			       i,
			       dt_,
			       dtold_,
			       t_theta_,
			       t_theta2_,
			       time,
			       eqn_id,
			       vol,
			       rand);

  const double phi[3] = {basis[phid]->uu(),basis[phid]->uuold(),basis[phid]->uuoldold()};

  double normgradq[3] = {0.,0.,0.};
  for(int k = qid; k < N+qid; k++){
    normgradq[0] = normgradq[0] + basis[k]->duudx()*basis[k]->duudx() + basis[k]->duudy()*basis[k]->duudy() + basis[k]->duudz()*basis[k]->duudz();
    normgradq[1] = normgradq[1] + basis[k]->duuolddx()*basis[k]->duuolddx() + basis[k]->duuolddy()*basis[k]->duuolddy() + basis[k]->duuolddz()*basis[k]->duuolddz();
    normgradq[2] = normgradq[2] + basis[k]->duuoldolddx()*basis[k]->duuoldolddx() + basis[k]->duuoldolddy()*basis[k]->duuoldolddy() + basis[k]->duuoldolddz()*basis[k]->duuoldolddz();
  }
  double b2 = 0.*beta;
  normgradq[0] = sqrt(b2 + normgradq[0]);
  normgradq[1] = sqrt(b2 + normgradq[1]);
  normgradq[2] = sqrt(b2 + normgradq[2]);

  const double pq[3] = {cc*Mphi*2.*H*T*pp(phi[0])*normgradq[0]*test,
			cc*Mphi*2.*H*T*pp(phi[1])*normgradq[1]*test,
			cc*Mphi*2.*H*T*pp(phi[2])*normgradq[2]*test};

  const double epqq[3] = {0.,0.,0.};

  const double f[3] = {pq[0] + epqq[0],
		       pq[1] + epqq[1],
		       pq[2] + epqq[2]};

  return val + (1.-t_theta2_)*t_theta_*f[0]
    + (1.-t_theta2_)*(1.-t_theta_)*f[1]
    +.5*t_theta2_*((2.+dt_/dtold_)*f[1]-dt_/dtold_*f[2]);
}

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(precon_phi_)
{
  const double phit = basis[0]->phi(j)/dt_*basis[0]->phi(i);
  const double divgradphi = Mphi*epphi*epphi*(basis[0]->dphidx(j)*basis[0]->dphidx(i)
					    + basis[0]->dphidy(j)*basis[0]->dphidy(i)
					    + basis[0]->dphidz(j)*basis[0]->dphidz(i));

  const double ww = Mphi*32.*omega*(1.-6.*basis[eqn_id]->uu()+6.*basis[eqn_id]->uu()*basis[eqn_id]->uu())*basis[0]->phi(j)*basis[0]->phi(i);
  return phit + t_theta_*(divgradphi + ww);
}

INI_FUNC(initphi_)
{
  double val = 0.;
  const double x0 = 0.;
  const double x1 = .128;
  const double w = .0005;
  const double den = sqrt(2)*w;
  //if( x*x + y*y <= r0*r0 ) val = 1.;

  val = .5*(tanh((r0-sqrt(x*x + y*y))/den) + 1.);

  //if( (x-x1)*(x-x1) + y*y <= r0*r0 ) val = 1.;
  val += .5*(tanh((r0-sqrt((x-x1)*(x-x1) + y*y))/den) + 1.);
  //if( (x-x1)*(x-x1) + (y-x1)*(y-x1) <= r0*r0 ) val = 1.;
  val += .5*(tanh((r0-sqrt((x-x1)*(x-x1) + (y-x1)*(y-x1)))/den) + 1.);
  //if( x*x + (y-x1)*(y-x1) <= r0*r0 ) val = 1.;
  val += .5*(tanh((r0-sqrt(x*x + (y-x1)*(y-x1)))/den) + 1.);

  return val;
}

INI_FUNC(initphisharp_)
{
  double val = 0.;
  const double x0 = 0.;
  const double x1 = .128;
  //const double s = .001;
  //const double den = sqrt(2)*s;
  if( x*x + y*y <= r0*r0 ) val = 1.;

  //val = .5*(tanh((r0-sqrt(x*x + y*y))/den) + 1.);

  if( (x-x1)*(x-x1) + y*y <= r0*r0 ) val = 1.;
  //val += .5*(tanh((r0-sqrt((x-x1)*(x-x1) + y*y))/den) + 1.);
  if( (x-x1)*(x-x1) + (y-x1)*(y-x1) <= r0*r0 ) val = 1.;
  //val += .5*(tanh((r0-sqrt((x-x1)*(x-x1) + (y-x1)*(y-x1)))/den) + 1.);
  if( x*x + (y-x1)*(y-x1) <= r0*r0 ) val = 1.;
  //val += .5*(tanh((r0-sqrt(x*x + (y-x1)*(y-x1)))/den) + 1.);

  return val;
}

const int nqr = 50;
const double qr[50][4]=  {{  -0.576287725136916,   0.333505408602186,  -0.361632934649101,   0.652601119265578},
			  {  -0.343986940785772,   0.682277345495660,  -0.573018177692459,  -0.296345704248021},
			  {  -0.232647440127303,   0.937205535917663,  -0.031637110694253,   0.257914802356164},
			  {   0.018242774740360,   0.178106722707921,  -0.372611279703857,  -0.910552596357547},
			  {  -0.517542850695652,  -0.395916678894981,   0.563925275224169,   0.507333879245149},
			  {  -0.356107794974079,  -0.389917100062351,   0.630199242877875,   0.569210688334135},
			  {   0.409223435595249,   0.525455340662819,  -0.608126788477899,   0.431989205726369},
			  {  -0.630453117516449,  -0.631184470746428,   0.318531939032296,  -0.320425395870346},
			  {   0.864255096566982,  -0.020351386737651,  -0.465285732524850,   0.190152928519825},
			  {   0.365432754153649,  -0.526852095410034,  -0.122425598464415,  -0.757560390064294},
			  {  -0.023391493919690,   0.982971622791350,   0.098302986525391,  -0.153480127832022},
			  {   0.147741536299811,   0.400175531209581,   0.832990657776059,  -0.352361386548965},
			  {  -0.602112968970399,   0.001080903203019,  -0.188520159958563,  -0.775834359599467},
			  {  -0.070671422382090,  -0.049651701376410,   0.829575523143495,  -0.551674460175670},
			  {   0.687470613929105,  -0.653760699701202,  -0.294746850163215,  -0.114478805151857},
			  {   0.095463229772612,   0.371143231339357,   0.410153657689132,  -0.827594979852216},
			  {   0.042511119781260,  -0.920321746623667,   0.334562237442024,   0.198163560384581},
			  {  -0.528344360485014,  -0.843590300373634,   0.077018684138540,  -0.057234291755664},
			  {  -0.039692982495951,   0.077402900076849,   0.708270363097808,   0.700561454090623},
			  {   0.250279759012911,  -0.813559892230614,   0.335954670425274,   0.403255258368191},
			  {   0.353980275179600,   0.628676543465058,  -0.636025446497738,   0.273743310215892},
			  {  -0.152788863914545,   0.464351434025763,  -0.543957262514449,  -0.682014519889208},
			  {  -0.187087542711502,   0.313930481394199,   0.704355492269799,  -0.608546830345165},
			  {   0.689128595718498,  -0.494460316926392,  -0.464443106989173,  -0.254761405865069},
			  {  -0.682625906147194,   0.235690765317176,  -0.690213312597345,   0.045577609791715},
			  {   0.917329187803420,   0.044286345561380,   0.145755501351447,   0.367833134215127},
			  {   0.557850641655030,   0.638416569840060,   0.515439781907479,  -0.124694731989341},
			  {   0.603412787722204,   0.303539051185344,   0.344659352608143,   0.651894916898847},
			  {  -0.638702974525255,   0.478016200385605,  -0.492716777886467,   0.347547405817794},
			  {  -0.437944726422717,  -0.084971025937098,  -0.241076692463062,   0.861896960025268},
			  {  -0.890255128091356,  -0.365571278185609,  -0.212361908912582,   0.169428058820809},
			  {   0.283933477549684,   0.513929394485025,   0.760889651782191,  -0.276234131891003},
			  {  -0.477568098812240,  -0.547319427022375,  -0.451482872991269,  -0.518202056341261},
			  {   0.389733113033568,  -0.646237045904149,   0.626319109355931,   0.195474178248271},
			  {  -0.655398087317474,  -0.544353299842224,   0.241274666195507,   0.464671246736578},
			  {   0.645697924708023,  -0.458002399972834,  -0.519589975656267,  -0.321456449374400},
			  {  -0.009541949679743,   0.542735057151925,  -0.506206397811392,  -0.670151245428538},
			  {   0.582391235211232,   0.633132670526536,  -0.149738649013156,  -0.487382609099655},
			  {   0.358542858356779,  -0.732849985326541,  -0.029975582224899,  -0.577476737365686},
			  {   0.668212350540457,  -0.166742069822322,  -0.627939298650339,  -0.362465962466941},
			  {   0.961964726495685,   0.066134849382783,   0.220256040724483,  -0.147435827394747},
			  {  -0.499712621263083,  -0.250706230771231,  -0.825072304715864,   0.081788593298288},
			  {   0.706077234730690,   0.557270543778255,  -0.409909524385984,  -0.151256938517897},
			  {  -0.597855494383301,   0.253183533691461,   0.164180933948266,   0.742638220825645},
			  {  -0.822741088328489,  -0.364554146085352,  -0.434967223033053,   0.031636861964273},
			  {   0.441977920540021,  -0.123768371147922,  -0.379477128367265,   0.803326843261598},
			  {   0.000043384544043,  -0.936920756221932,   0.226423044168647,   0.266293258922767},
			  {  -0.028414345444403,   0.685050830453213,  -0.332176322165211,   0.647732101768257},
			  {   0.605325332554190,  -0.497801637772663,   0.485155380746398,  -0.387813135068723},
			  {   0.167327436725344,  -0.599734410519205,   0.735098691333444,   0.268235120291818}};

//q0 is -.5 lower right
INI_FUNC(initq0_)
{
  double val = .0;
  const double s = .001;
  const double den = sqrt(2)*s;

  val = .5*(1.+tanh((r0-x)/den)+tanh((y-r0)/den));
  if (val > .5) val = .5;
  return val;
}

const double s = 1.e-5;
  //could there be threading issues with r0?
INI_FUNC(initq0s_)
{
  r0 = .064;
  double val = .5;

  const double s = r0 + halfdx;

  if (x > s && y < s) val = -.5;
#if 0
  val = val * initphi_(x,
			    y,
			    z,
		       eqn_id,
		       lid);
  //cn we could further perturb here by adding a multiple of proc_id to lid in these functions if needed...
  //as a hack, we could do (pid+1)*lid in the calling code...

  //it actually seems like we do a normalization of the initial condition, so we can probably just put a random number in here for now
  //for reproducibility we can leave it for now
  //however, if we do not run with local projection on and the qs are not normalized, we will have issues

  if (val*val < s*s ) val = qr[lid%nqr][eqn_id-qid];
#endif
  return val;
}


//q1 is -.5 in upper right
INI_FUNC(initq1_)
{
  double val = .0;//.5
  const double s = .001;
  const double den = sqrt(2)*s;
  //if (x > r0 && y > r0 ) val = -.5;
  val = .5*(1.+tanh((r0-x)/den)+tanh((r0-y)/den));
  if (val > .5) val = .5;
  return val;
}

INI_FUNC(initq1s_)
{
  double val = .5;

  const double s = r0 + halfdx;

  if (x > s && y > s) val = -.5;
#if 0
  val = val * initphi_(x,
			    y,
			    z,
		       eqn_id,
		       lid); 
  if (val*val < s*s ) val = qr[(lid)%nqr][eqn_id-qid];
#endif
  return val;

}

//q2 and q3 are .5 everywhere
INI_FUNC(initq2_)
{
  double val = .5;
#if 0
  const double alpha = 1./sqrt(2.);
  val = (val - alpha) * initphi_(x,
			    y,
			    z,
				 eqn_id,
				 lid) + alpha; 

  if ((val-alpha)*(val-alpha) < s*s ) val = qr[(lid)%nqr][eqn_id-qid];
#endif
  return val;
}

INI_FUNC(initq3_)
{
  return initq2_(x,
		 y,
		 z,
		 eqn_id,
		 lid); 
}

INI_FUNC(init_)
{
  //this will be a function of eqn_id...

  const double r = .2;
  const double x0 = .5;
  const double y0 = .5;
  const double w = .01;//.025;
  //g1 is the background grain
  //g0 is black??
  const double g0[4] = {-0.707106781186547,
			-0.000000000000000,
			0.707106781186548,
			-0.000000000000000};
  const double g1[4] = {0.460849109679818,
		       0.025097870693789,
		       0.596761014095944,
		       0.656402686655941};   
  const double g2[4] = {0.659622270251558,
			0.314355942642574,
			0.581479198986258,
			-0.357716008950928};
  const double g3[4] = {0.610985843880283,
			-0.625504969359875,
			-0.474932634235629,
			-0.099392277476709};

  double scale = (g2[eqn_id]-g1[eqn_id])/2.;
  double shift = (g2[eqn_id]+g1[eqn_id])/2.;
  double d = dist(x,y,z,x0,y0,0,r);
  double val = shift + scale*tanh(d/sqrt(2.)/w);
  return val;
}

  //https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

//atan2 returns [-pi,pi]
//asin returns [-pi/2,pi/2]
//paraview expects [0,1] for an rgb value
//https://discourse.paraview.org/t/coloring-surface-by-predefined-rgb-values/6011/6
//so should we shift and scale each of these here?

//Also the page seems different today, 4-12-23

//And our convention to normalize via:
//  return (s+pi)/2./pi;
//so that each angle is between 0 and 1 can be used as RGB coloring

//Also, with the example, q=[.5 -.5 .5 .5] and roundoff, etc 
//produces arguments to atan2 with near +/- 0, producing oscillations
//we need to fix this
//ie small chages in q lead to large changes in euler angle

#if 0
EulerAngles ToEulerAngles(Quaternion q) {
    EulerAngles angles;

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    angles.roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = std::sqrt(1 + 2 * (q.w * q.y - q.x * q.z));
    double cosp = std::sqrt(1 - 2 * (q.w * q.y - q.x * q.z));
    angles.pitch = 2 * std::atan2(sinp, cosp) - M_PI / 2;

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    angles.yaw = std::atan2(siny_cosp, cosy_cosp);

    return angles;
}
#endif

PPR_FUNC(postproc_ea0_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...
  //          
  //                         w      x      y      z
  double sinr_cosp = 2. * (u[qid+0] * u[qid+1] + u[qid+2] * u[qid+3]);
  //                            x      x      y      y
  double cosr_cosp = 1. - 2. * (u[qid+1] * u[qid+1] + u[qid+2] * u[qid+2]);
  double s = std::atan2(sinr_cosp, cosr_cosp);
  return (s+pi)/2./pi;
  //return s;
}

PPR_FUNC(postproc_ea1_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...
  //                 w      y      z      x
  double sinp = 2 * (u[qid+0] * u[qid+2] - u[qid+3] * u[qid+1]);
  double s = 0.;
  if (std::abs(sinp) >= 1.){
  //if (std::abs(sinp) >= .95){
    s = std::copysign(pi / 2., sinp); // use 90 degrees if out of range
  }else{
    s = std::asin(sinp);
  }
  return (s+pi/2.)/pi;
  //return s;
}

PPR_FUNC(postproc_ea2_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...
  //                       w      z      x      y
  double siny_cosp = 2. * (u[qid+0] * u[qid+3] + u[qid+1] * u[qid+2]);
  //                            y      y      z      z
  double cosy_cosp = 1. - 2. * (u[qid+2] * u[qid+2] + u[qid+3] * u[qid+3]);
  double s = std::atan2(siny_cosp, cosy_cosp);
  return (s+pi)/2./pi;
  //return s;
}

//a possible alternative to euler angles is to just consider the quaternion as rgba color
//there may be visualization ttols that allow for this, although doesn't seem easy in 
//paraview
//rgba can be converted to rgb via the following, assuming some background color BGColor,
//and all values normalized in [0 1]:
//  Source => Target = (BGColor + Source) =
//  Target.R = ((1 - Source.A) * BGColor.R) + (Source.A * Source.R)
//  Target.G = ((1 - Source.A) * BGColor.G) + (Source.A * Source.G)
//  Target.B = ((1 - Source.A) * BGColor.B) + (Source.A * Source.B)
//
//note that in our example problem, q0 and q1 can have values in [-.5 .5]
//we should probably normalize these anyway, with the convention that q_i \in [-1 1]
//and should be normalized to [0 1] via q->(q+1)/2
//https://stackoverflow.com/questions/2049230/convert-rgba-color-to-rgb


const double bgcolor[3] = {1.,1.,1.};

PPR_FUNC(postproc_rgb_r_)
{
  const double u4 = (u[qid+3]+1.)/2.;
  const double u1 = (u[qid]+1.)/2.;
  return (1.-u4)*bgcolor[0]+u4*u1;
}
PPR_FUNC(postproc_rgb_g_)
{
  const double u4 = (u[qid+3]+1.)/2.;
  const double u2 = (u[qid+1]+1.)/2.;
  return (1.-u4)*bgcolor[1]+u4*u2;
}
PPR_FUNC(postproc_rgb_b_)
{
  const double u4 = (u[qid+3]+1.)/2.;
  const double u3 = (u[qid+2]+1.)/2.;
  return (1.-u4)*bgcolor[2]+u4*u3;
}

PPR_FUNC(postproc_mq_)
{
  const double phi = u[0];
  const double m = 1.-h(phi);
  return (m*(Mq-1e-6)+1e-6)*epq*epq;
}

PPR_FUNC(postproc_md_)
{
  const double phi = u[0];
  const double m = 1.-h(phi);
  //return (m*(Mq-1e-6)+1e-6)*2.*T*H*p(phi);
  return 2.*T*H*p(phi);
}


//other code suggests:
// // converts vec3 color to vec4 quaternion
// vec4 c2q( in vec3 c ) {
//     c = c / sqrt3; // length(c) must be <= 1.0
//     float rr = c.r*c.r;
//     float gg = c.g*c.g;
//     float bb = c.b*c.b;
//     float ww = 1.0 - sqrt(rr+gg+bb);
//     float xx = rr/(ww+1.0);
//     float yy = gg/(ww+1.0);
//     float zz = bb/(ww+1.0);
//     return vec4( sqrt( xx ), sqrt( yy ), sqrt( zz ), sqrt( ww ) );
// }

// // converts vec4 quaternion to vec3 color
// vec3 q2c( in vec4 q ) {
//     float xx = q.x*q.x;
//     float yy = q.y*q.y;
//     float zz = q.z*q.z;
//     float ww = q.w*q.w;
//     float rr = (1.0-ww)*xx;
//     float gg = (1.0-ww)*yy;
//     float bb = (1.0-ww)*zz;
//     vec3 c = vec3( sqrt( rr ), sqrt( gg ), sqrt( bb ) );
//     return c * sqrt3; // renormalize
// }



PPR_FUNC(postproc_normqold_)
{
  return uold[qid+0]*uold[qid+0]+uold[qid+1]*uold[qid+1]+uold[qid+2]*uold[qid+2]+uold[qid+3]*uold[qid+3];
}

PPR_FUNC(postproc_normq_)
{
  return u[qid+0]*u[qid+0]+u[qid+1]*u[qid+1]+u[qid+2]*u[qid+2]+u[qid+3]*u[qid+3];
}

PPR_FUNC(postproc_d_)
{
  double phi=u[0];
  return 2*H*T*p(phi);
}

PPR_FUNC(postproc_qdotqt_)
{
  double s = 0.;
  for(int k = qid; k < 4+qid; k++) s = s + u[k]*(u[k]-uold[k])/dt;
  return s;
}

PPR_FUNC(postproc_normgradq_)
{
  double s = 0.;
  for(int k = qid; k < 12+qid; k++) s = s + gradu[k]*gradu[k];
  return s;
}

PPR_FUNC(postproc_qdotqold_)
{
  double s = 0.;
  for(int k = qid; k < 4+qid; k++) s = s + u[k]*uold[k];
  return std::sqrt(s);
}

PPR_FUNC(postproc_normphi_)
{
  double s = u[0]*u[0];
  return s;
}

PARAM_FUNC(param_)
{
}

}//namespace quaternion

namespace l21d
{
const double a = 0.7071067811865475;// 1/sqrt[2]

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_)
{
  //derivatives of the test function
  //test function
  const double test = basis[0]->phi(i);
  std::cout<<basis[0]->uu()<<" "<<basis[1]->uu()<<" "<<basis[0]->uu()*basis[0]->uu()+basis[1]->uu()*basis[1]->uu()<<std::endl;
  return basis[eqn_id]->uu()*test - basis[eqn_id]->uuold()*test;
}

INI_FUNC(initq0_)
{
  return a;
}

INI_FUNC(initq1_)
{
  double val = a;

  const double s = 0.;

  if (x > s ) val = -a;
  return val;
}

}//namespace l21d

// Comment: Grain seems not to be used in tpetra setup yet
namespace grain
{

  //see
  //[1] Suwa et al, Mater. T. JIM., 44,11, (2003);
  //[2] Krill et al, Acta Mater., 50,12, (2002); 



  double L = 1.;
  double alpha = 1.;
  double beta = 1.;
  double gamma = 1.;
  double kappa = 2.;

  int N = 6;

  double pi = 3.141592653589793;

  double r(const double &x,const int &n){
    return sin(64./512.*x*n*pi);
  }

PARAM_FUNC(param_)
{
  N = plist->get<int>("numgrain");
}

KOKKOS_INLINE_FUNCTION
RES_FUNC_TPETRA(residual_)
{
  //derivatives of the test function
  double dtestdx = basis[0]->dphidx(i);
  double dtestdy = basis[0]->dphidy(i);
  double dtestdz = basis[0]->dphidz(i);
  double test = basis[0]->phi(i);

  double u = basis[eqn_id]->uu();
  double uold = basis[eqn_id]->uuold();

  double divgradu = kappa*(basis[eqn_id]->duudx()*dtestdx + basis[eqn_id]->duudy()*dtestdy + basis[eqn_id]->duudz()*dtestdz);

  double s = 0.;
  for(int k = 0; k < N; k++){
    s = s + basis[k]->uu()*basis[k]->uu();
  }
  s = s - u*u;

  return (u-uold)/dt_*test + L* ((-alpha*u + beta*u*u*u +2.*gamma*u*s)*test +  divgradu); 

}
PRE_FUNC(prec_)
{
#if 0
  //cn probably want to move each of these operations inside of getbasis
  //derivatives of the test function
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;
  double dtestdz = basis[0].dphidxi[i]*basis[0].dxidz
    +basis[0].dphideta[i]*basis[0].detadz
    +basis[0].dphidzta[i]*basis[0].dztadz;

  double dbasisdx = basis[0].dphidxi[j]*basis[0].dxidx
    +basis[0].dphideta[j]*basis[0].detadx
    +basis[0].dphidzta[j]*basis[0].dztadx;
  double dbasisdy = basis[0].dphidxi[j]*basis[0].dxidy
    +basis[0].dphideta[j]*basis[0].detady
    +basis[0].dphidzta[j]*basis[0].dztady;
  double dbasisdz = basis[0].dphidxi[j]*basis[0].dxidz
    +basis[0].dphideta[j]*basis[0].detadz
    +basis[0].dphidzta[j]*basis[0].dztadz;

  double u = basis[eqn_id].uu();
  
  double test = basis[0].phi(i);
  double divgrad = L*kappa*(dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz);
  double u_t =test * basis[0].phi(j)/dt_;
  double alphau = -test*L*alpha*basis[0].phi(j);
  double betau = 3.*u*u*basis[0].phi(j)*test*L*beta;

  double s = 0.;
  for(int k = 0; k < N; k++){
    s = s + basis[k].uu()*basis[k].uu();
  }
  s = s - u*u;

  double gammau = 2.*gamma*L*basis[0].phi(j)*s*test;

  return u_t + divgrad + betau + gammau;// + alphau ;
#endif
  return 1.;
}
PPR_FUNC(postproc_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...


  double s =0.;
  for(int j = 0; j < N; j++){
    s = s + u[j]*u[j];
  }

  return s;
}
}//namespace grain

namespace random
{

RES_FUNC_TPETRA(residual_test_)
{
  //test function
  const double test = basis[0]->phi(i);
  //printf("%d %le \n",i,rand);
  return (basis[0]->uu() - rand)*test;
}

}//namespace random

NBC_FUNC_TPETRA(nbc_one_)
{
  
  double phi = basis->phi(i);
  
  return 1.*phi;
}

namespace uehara
{
  const double phi_sol_ = 1.;
  const double phi_liq_ = 0.;

  double L = 3.e3;//J/m^3
  double m = 2.5e5;
  double a = 10.;//m^4
  double r0 = 29.55;
  double rho = 1.e3;//kg/m^3
  double c = 5.e2;//J/kg/K
  double k = 150.;//W/m/K

  //plane stress
  double giga = 1.e+9;
  double E = 200.*giga;//GPa
  //double E = 200.*1.e9;//Pa
  double nu = .3;
  double c0 = E/(1.-nu*nu);
  //it seems c1=E (1-nu)/(1+nu)/(1-2 nu)
  //double c1 = c0;
  const double c1 = E*(1.-nu)/(1.+nu)/(1.-2.*nu);
  //it seems c2 = E nu/(1+nu)/(1-2 nu)
  //double c2 = c0*nu;
  const double c2 = E*nu/(1.+nu)/(1.-2.*nu);
  //double c3 = c0*(1.-nu)/2.;//always confused about the 2 here
  const double c3 = E/(1.+nu)/2.;

  double alpha = 5.e-6;//1/K
  double beta = 1.5e-3;

  double t0 = 300.;//K

PARAM_FUNC(param_)
{
  t0 = plist->get<double>("t0",300.);
  alpha = plist->get<double>("alpha",5.e-6);
}

const double h_(const double &p)
{
  return p*(1.-p);
}

INI_FUNC(init_phase_c_2_)
{
  //this currently just puts solid on each of the corner nodes
  //which it seems is done in the paper:
  //Takuya UEHARA and Takahiro TSUJINO. Simulations on the stress evo-
  //lution and residual stress in precipitated phase using a phase field model.
  //Journal of Computational Science and Technology, 2(1):142–149, 2008.
  //https://doi.org/10.1299/jcst.2.142

  double val = phi_liq_ ;  

  //double r0 = uehara2::r0;
  double dx = 1.e-2;
  double x0 = 15.*dx;
  //double r = .9*r0*dx/2.;

  double rr = sqrt((x-x0)*(x-x0)+(y-x0)*(y-x0));

  //val = phi_liq_;

  if( rr > r0*sqrt(2.)*dx/2. ){
    val = phi_sol_;
  }
  else {
    val = phi_liq_;
  }

  return val;
}

INI_FUNC(init_heat_)
{
  const double val = t0;  

  return val;
}

RES_FUNC_TPETRA(residual_stress_x_dt_)
{
  double strain[3];//x,y,yx

  strain[0] = (basis[2]->duudx()-basis[2]->duuolddx())/dt_;
  strain[1] = (basis[3]->duudy()-basis[3]->duuolddy())/dt_;

  const double stress = c1*strain[0] + c2*strain[1];

  return stress;
}

RES_FUNC_TPETRA(residual_stress_y_dt_)
{
  double strain[3];//x,y,z,yx,zy,zx

  strain[0] = (basis[2]->duudx()-basis[2]->duuolddx())/dt_;
  strain[1] = (basis[3]->duudy()-basis[3]->duuolddy())/dt_;

  const double stress = c2*strain[0] + c1*strain[1];

  return stress;
}

RES_FUNC_TPETRA(residual_heat_)
{
  //derivatives of the test function
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double test = basis[0]->phi(i);

  const int phi_id = 0;
  const int u_id = 1;
  const double phi[3] = {basis[phi_id]->uu(),basis[phi_id]->uuold(),basis[phi_id]->uuoldold()};
  const double u = basis[u_id]->uu();
  const double uold = basis[u_id]->uuold();

  const double dudx = basis[u_id]->duudx();
  const double dudy = basis[u_id]->duudy();


  const double ut = rho*c*(u-uold)/dt_*test;
  const double divgradu = k*(dudx*dtestdx + dudy*dtestdy);
  double h = h_(phi[0]);
  h = h *h;
  //double phitu = -30.*1e12*uehara::L*h*(phi-phioldold)/2./dt_*test; 
  const double phitu = -30.*L*h*(phi[0]-phi[1])/dt_*test; 
  
  //thermal term
  const double stress = 0.*test*alpha*u*(residual_stress_x_dt_(basis, 
							    i, dt_, dtold_, t_theta_, t_theta2_,
							    time, eqn_id, vol, rand)
				      +residual_stress_y_dt_(basis, 
							     i, dt_,  dtold_,t_theta_,t_theta2_,
							     time, eqn_id, vol, rand));
  

  double rhs = divgradu + phitu + stress;

  return (ut + rhs);// /rho/c;
}

RES_FUNC_TPETRA(residual_phase_)
{
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double test = basis[0]->phi(i);

  const int phi_id = 0;
  const int u_id = 1;
  const double phi[3] = {basis[phi_id]->uu(),basis[phi_id]->uuold(),basis[phi_id]->uuoldold()};
  const double u = basis[u_id]->uu();
  const double dphidx = basis[phi_id]->duudx();
  const double dphidy = basis[phi_id]->duudy();
  const double dphidz = basis[phi_id]->duudz();
  
  const double b = 5.e-5;//m^3/J
  //b = 5.e-7;//m^3/J
  const double f = 0.;
  const double um = 400.;//K

  const double phit = m*(phi[0]-phi[1])/dt_*test;
  const double divgradphi = a*(dphidx*dtestdx + dphidy*dtestdy + dphidz*dtestdz);//(grad u,grad test)

  //double M = b*phi*(1. - phi)*(L*(um - u)/um + f);
  const double h = h_(phi[0]);
  //const double M = 100000000.*b*h*(L*(um - u)/um + f);
  const double M =   70000000.*b*h*(L*(um - u)/um + f);
  //const double M = b*h*(L*(um - u)/um + f);
  //double g = -phi*(1. - phi)*(phi - .5 + M)*test;

  const double g = -(h*(phi[0] - .5)+h*M)*test;
  const double rhs = divgradphi + g;

  return (phit + rhs);// /m;
}

RES_FUNC_TPETRA(residual_liniso_x_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double strain[3], stress[3];//x,y,yx

  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double test = basis[0]->phi(i);

  const int phi_id = 0;
  const int u_id = 1;

  //u, phi
  //double u = basis[0].uu();

  //double ut = (basis[1].uu() - basis[1].uuoldold())/dt_/2;//thermal strain
  const double ut = (basis[u_id]->uu() - basis[u_id]->uuold())/dt_;//thermal strain

  const double phi = basis[phi_id]->uu();
  double h = h_(phi);
  h = h*h;
  //double hp = 2.*(1.-phi)*(1.-phi)*phi-2.*(1.-phi)*phi*phi;//2 (1 - x)^2 x - 2 (1 - x) x^2
  // h' p_t p_x +h p_t_x
  const double strain_phi = 30.*beta*h*(phi-basis[phi_id]->uuold())/dt_;
//   double strain_phi = 0.*2.*30.*beta*(c1+c2)*(hp*(phi-basis[0].uuold())/dt_*basis[0].duudx()
// 					   +h*(basis[0].duudx()-basis[0].duuolddx())/dt_
// 					   )*test;
  
  const double ff =   alpha*ut + strain_phi;

  //strain = D displacement
  strain[0] = (basis[2]->duudx()-basis[2]->duuolddx())/dt_- ff;
  strain[1] = (basis[3]->duudy()-basis[3]->duuolddy())/dt_- ff;
  strain[2] = (basis[2]->duudy()-basis[2]->duuolddy() + basis[3]->duudx()-basis[3]->duuolddx())/dt_;// - alpha*ut - strain_phi;

  //stress =  C strain
  stress[0] = c1*strain[0] + c2*strain[1];
  // stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

  const double divgradu = (stress[0]*dtestdx + stress[2]*dtestdy)/E;//(grad u,grad phi)
 
  //std::cout<<"residual_liniso_x_test_"<<std::endl;
 
  return divgradu;
}

RES_FUNC_TPETRA(residual_liniso_y_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double strain[3], stress[3];//x,y,yx

  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double test = basis[0]->phi(i);

  const int phi_id = 0;
  const int u_id = 1;

  //u, phi
  //double u = basis[0].uu();

  //double ut = (basis[1].uu() - basis[1].uuoldold())/dt_/2;//thermal strain
  const double ut = (basis[u_id]->uu() - basis[u_id]->uuold())/dt_;//thermal strain

  const double phi = basis[phi_id]->uu();

  double h = h_(phi);
  h = h*h;
  //double hp = 2.*(1.-phi)*(1.-phi)*phi-2.*(1.-phi)*phi*phi;//2 (1 - x)^2 x - 2 (1 - x) x^2
  const double strain_phi = 30.*beta*h*(phi-basis[0]->uuold())/dt_;
//   double strain_phi = 0.*2.*30.*beta*(c1+c2)*(hp*(phi-basis[0].uuold())/dt_*basis[0].duudy()
// 					+h*(basis[0].duudy()-basis[0].duuolddy())/dt_
// 					)*test;

  double ff =   alpha*ut + strain_phi;

  strain[0] = (basis[2]->duudx()-basis[2]->duuolddx())/dt_- ff;
  strain[1] = (basis[3]->duudy()-basis[3]->duuolddy())/dt_- ff;
  strain[2] = (basis[2]->duudy()-basis[2]->duuolddy() + basis[3]->duudx()-basis[3]->duuolddx())/dt_;// - alpha*ut - strain_phi;

  //stress[0] = c1*strain[0] + c2*strain[1];
  stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

  double divgradu = (stress[1]*dtestdy + stress[2]*dtestdx)/E;//(grad u,grad phi)
  
  //std::cout<<"residual_liniso_y_test_"<<std::endl;

  return divgradu;
}

NBC_FUNC_TPETRA(conv_bc_)
{

  const double test = basis[0].phi(i);
  const double u = basis[0].uu();
  const double uw = 300.;//K
  const double h =1.e4;//W/m^2/K
  //std::cout<<h*(uw-u)*test/rho/c<<std::endl;
  return h*(uw-u)*test/rho/c;
}

PRE_FUNC_TPETRA(prec_phase_)
{
 
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double test = basis[0]->phi(i);

  const int phi_id = 0;

  const double phit = m*(basis[phi_id]->phi(j))/dt_*test;
  const double divgrad = a*(basis[0]->dphidx(j) * dtestdx + basis[0]->dphidy(j) * dtestdy + basis[0]->dphidz(j) * dtestdz);

  return (phit + t_theta_*divgrad);// /m;
}

PRE_FUNC_TPETRA(prec_heat_)
{
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double test = basis[0]->phi(i);

  const int u_id = 1;


  const double stress = test*alpha*basis[0]->phi(j)*(residual_stress_x_dt_(basis, 
 										   i, dt_, 0., t_theta_, 0.,
										   0., eqn_id, 0., 0.)
						    +residual_stress_y_dt_(basis, 
										    i, dt_,  0.,t_theta_, 0.,
										    0., eqn_id, 0., 0.)
 						    );
  
  const double divgrad = k*(basis[0]->dphidx(j) * dtestdx + basis[0]->dphidy(j) * dtestdy + basis[0]->dphidz(j) * dtestdz);
  const double u_t =rho*c*basis[u_id]->phi(j)/dt_*test;
 
  return (u_t + t_theta_*divgrad + stress);// /rho/c;
}

PRE_FUNC_TPETRA(prec_liniso_x_test_)
{
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  //const double dtestdz = basis[0]->dphidz(i);

  const double dbasisdx = basis[0]->dphidx(j); 
  const double dbasisdy = basis[0]->dphidy(j); 

  double strain[3], stress[3];//x,y,z,yx,zy,zx

  strain[0] = dbasisdx;
  strain[1] = dbasisdy;
  strain[2] = (dbasisdy + dbasisdx);

  stress[0] = c1*strain[0] + c2*strain[1];
  //stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

  const double divgradu = (stress[0]*dtestdx + stress[2]*dtestdy)/E/dt_;//(grad u,grad phi)
  //double divgradu = (stress[0]*dtestdx + stress[2]*dtestdy)/E;//(grad u,grad phi)
  
  return divgradu;
}

PRE_FUNC_TPETRA(prec_liniso_y_test_)
{

  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  //const double dtestdz = basis[0]->dphidz(i);

  const double dbasisdx = basis[0]->dphidx(j); 
  const double dbasisdy = basis[0]->dphidy(j); 

  double strain[3], stress[3];//x,y,z,yx,zy,zx

  strain[0] = dbasisdx;
  strain[1] = dbasisdy;
  strain[2] = (dbasisdy + dbasisdx);

  //stress[0] = c1*strain[0] + c2*strain[1];
  stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];


  const double divgradu = (stress[1]*dtestdy + stress[2]*dtestdx)/E/dt_;//(grad u,grad phi)
  //double divgradu = (stress[1]*dtestdy + stress[2]*dtestdx)/E;//(grad u,grad phi)

  //std::cout<<divgradu<<std::endl;
  return divgradu;
}

PPR_FUNC(postproc_stress_eq_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...


  double strain[3], stress[3];//x,y,z,yx,zy,zx

  strain[0] = gradu[0];// - alpha*u[1] - 30.*beta*h*phi;//var 0 dx
  strain[1] = gradu[4];// - alpha*u[1] - 30.*beta*h*phi;//var 1 dy
  strain[2] = gradu[1] + gradu[3];// + gradu[2];// - alpha*u[1] - 30.*beta*h*phi;

  stress[0] = c1*strain[0] + c2*strain[1];
  stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

  //von mises
  return sqrt((stress[0]-stress[1])*(stress[0]-stress[1])
	       + 3.*stress[2]*stress[2]
	       );
}


}//namespace uehara

namespace yang
{
#if 0
  //uehara properties mesh is in m
  const double m = 2.5e5;//2.5e5
  const double a = 10;//m^4
  const double L = 3.e3;//J/m^3
  const double rho = 1.e3;//kg/m^3
  const double c = 5.e2;//J/(kg K)
  const double k = 150.;//W/(m K)
#endif
  //yang mesh is in um
  const double m = 1.;
  const double a = 10.;

  double L = 2.1e9;//J/m^3
  //const double rho = 8084.;//kg/m^3
  const double rho = 8.084e-15;//kg/um^3
  const double c = 770.;//J/(kg K)
  //const double k = 10.;//W/(m K) = J/(K m s)
  const double k = 0.00001;//J/(K um s)
  //const double h =1.e4;//W/m^2/K
  const double h =1.e-8;//W/um^2/K

  const double ul = 1723;
  const double us = 1650;

  const double um = us;
  const double uw = 1650;

  double du = 1./rho/c;//rho*c;
  double dm = 1.;

  int N_ = 1;
  int eqn_off_ = 2;

  PARAM_FUNC(param_)
  {
    N_ = plist->get<int>("N",0);
    eqn_off_ = plist->get<int>("OFFSET",2);
  }

RES_FUNC_TPETRA(residual_phase_)
{
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double test = basis[0]->phi(i);

  const int phi_id = 0;
  const int u_id = 1;
  const double phi[3] = {basis[phi_id]->uu(),basis[phi_id]->uuold(),basis[phi_id]->uuoldold()};
  const double u = basis[u_id]->uu();
  const double dphidx = basis[phi_id]->duudx();
  const double dphidy = basis[phi_id]->duudy();
  const double dphidz = basis[phi_id]->duudz();
  
  const double b = 5.e-5;//m^3/J
  //const double um = 350.;//K

  const double phit = m*(phi[0]-phi[1])/dt_*test;
  const double divgradphi = a*(dphidx*dtestdx + dphidy*dtestdy + dphidz*dtestdz);//(grad u,grad test)

  const double h = tpetra::uehara::h_(phi[0]);

  //const double g = -(h*(phi[0] - .5)+h*M)*test;
  //const double gp = -(h*(phi[0] - .5))*test;
  const double gp = (2*phi[0]*(1.-3.*phi[0]+2*phi[0]*phi[0])/4.)*test;

  //const double M =   -70000000.*b*h*h*(L*(um - u)/um);
  //const double M =   70000000.*b*(6.*phi[0]-6*phi[0]*phi[0])*(L*(u - um)/um);
  //const double M =   b*h*h*(L*(u - um)/um);
  const double M =   h*h*(L*(u - um)/um);
  const double g = M*test;
  const double rhs = divgradphi + gp + g;

  return (phit + rhs)/dm;// /m;
}

RES_FUNC_TPETRA(residual_heat_)
{
  //derivatives of the test function
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double test = basis[0]->phi(i);

  const int phi_id = 0;
  const int u_id = 1;
  const double phi[3] = {basis[phi_id]->uu(),basis[phi_id]->uuold(),basis[phi_id]->uuoldold()};
  const double u = basis[u_id]->uu();
  const double uold = basis[u_id]->uuold();

  const double dudx = basis[u_id]->duudx();
  const double dudy = basis[u_id]->duudy();

  const double ut = rho*c*(u-uold)/dt_*test;
  const double divgradu = k*(dudx*dtestdx + dudy*dtestdy);
  double h = tpetra::uehara::h_(phi[0]);
  h = h *h;
  const double phitu = -0.*30.*L*h*(phi[0]-phi[1])/dt_*test; 
  
  double rhs = divgradu + phitu;

  return (ut + rhs)/du;// /rho/c;
}

RES_FUNC_TPETRA(residual_eta_)
{
  //derivatives of the test function
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double test = basis[0]->phi(i);

  const int phi_id = 0;
  const double eta = basis[eqn_id]->uu();
  const double etaold = basis[eqn_id]->uuold();
  const double phi = basis[phi_id]->uu();

  const double kg =10.;
  const double divgradu = kg*(basis[eqn_id]->duudx()*dtestdx + basis[eqn_id]->duudy()*dtestdy + basis[eqn_id]->duudz()*dtestdz);

#if 0
  double s = 0.;
  for(int k = 0; k < N; k++){
    s = s + basis[k]->uu()*basis[k]->uu();
  }
  s = s - u*u;

  return (u-uold)/dt_*test + L* ((-alpha*u + beta*u*u*u +2.*gamma*u*s)*test +  divgradu); 
#endif 
  const double mg = 1.;
  const double lg = .1/m;
  const double val = (eta-etaold)/dt_*test + lg*divgradu + lg*mg*(eta*eta*eta-eta)*test + lg*mg*2.*(1.-phi)*(1.-phi)*eta*test;
  return val/lg;
  //return m*(basis[2]->uu()-phi)*test;
}
PRE_FUNC_TPETRA(prec_eta_)
{
  //derivatives of the test function
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);

  double dbasisdx = basis[0]->dphidx(j);
  double dbasisdy = basis[0]->dphidy(j);
  double dbasisdz = basis[0]->dphidz(j);

  const double kg =10.;
  const double mg = 1.;
  const double lg = 1./m;
  
  double test = basis[0]->phi(i);
  double divgrad = lg*kg*(dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz);
  double u_t =test * basis[0]->phi(j)/dt_;

#if 0
  double alphau = -test*L*alpha*basis[0].phi(j);
  double betau = 3.*u*u*basis[0].phi(j)*test*L*beta;

  double s = 0.;
  for(int k = 0; k < N; k++){
    s = s + basis[k].uu()*basis[k].uu();
  }
  s = s - u*u;

  double gammau = 2.*gamma*L*basis[0].phi(j)*s*test;

  return u_t + divgrad + betau + gammau;// + alphau ;
#endif
  return (u_t + divgrad)*m;
    //return 1.;
}

PRE_FUNC_TPETRA(prec_phase_)
{
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double test = basis[0]->phi(i);

  const int phi_id = 0;

  const double phit = m*(basis[phi_id]->phi(j))/dt_*test;
  const double divgrad = a*(basis[0]->dphidx(j) * dtestdx + basis[0]->dphidy(j) * dtestdy + basis[0]->dphidz(j) * dtestdz);

  return (phit + t_theta_*divgrad)/dm;
}

PRE_FUNC_TPETRA(prec_heat_)
{
  const double dtestdx = basis[0]->dphidx(i);
  const double dtestdy = basis[0]->dphidy(i);
  const double dtestdz = basis[0]->dphidz(i);
  const double test = basis[0]->phi(i);

  const int u_id = 1;
  
  const double divgrad = k*(basis[0]->dphidx(j) * dtestdx + basis[0]->dphidy(j) * dtestdy + basis[0]->dphidz(j) * dtestdz);
  const double u_t = rho*c*basis[u_id]->phi(j)/dt_*test;
 
  return (u_t + t_theta_*divgrad)/du;// /rho/c;
}

NBC_FUNC_TPETRA(conv_bc_)
{

  const double test = basis[0].phi(i);
  const int u_id = 1;
  const double u = basis[u_id].uu();
  //const double uw = 300.;//K
  return h*(u-uw)*test/du;
}

  const double euler_angles[4][3] = {{ 68.929, 36.059,277.170},
				     {226.785, 41.003, 97.730},
				     {317.524, 31.789, 79.010},
				     {144.094, 23.823,222.160}};

PPR_FUNC(postproc_ea1_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...

  //right now, the background color is black, ie 0,0,0
  //might be better id white, 1,1,1
  //probably do not want an if or ternary or where
  //we could probably take care of this as an extra step in paraview

  const int col = 0;

  double r = 0.;
  for(int i = 0; i < N_; i++){
    r = r + u[i+eqn_off_]*euler_angles[i][col]/360.;
  }
  return r;
}

PPR_FUNC(postproc_ea2_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...

  //in some refs this component is normailized by 180 rather than 360

  const int col = 1;

  double r = 0.;
  for(int i = 0; i < N_; i++){
    r = r + u[i+eqn_off_]*euler_angles[i][col]/360.;
  }
  return r;
}

PPR_FUNC(postproc_ea3_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...

  const int col = 2;

  double r = 0.;
  for(int i = 0; i < N_; i++){
    r = r + u[i+eqn_off_]*euler_angles[i][col]/360.;
  }
  return r;
}

INI_FUNC(init_phase_)
{
  //this currently just puts solid on each of the corner nodes
  //which it seems is done in the paper:
  //Takuya UEHARA and Takahiro TSUJINO. Simulations on the stress evo-
  //lution and residual stress in precipitated phase using a phase field model.
  //Journal of Computational Science and Technology, 2(1):142–149, 2008.
  //https://doi.org/10.1299/jcst.2.142

  double val = 0.;  

  //double r0 = uehara2::r0;
  //double dx = 1.e-2;
  double x0 = 0.;
  //double r = .9*r0*dx/2.;

  double rr = sqrt((x-x0)*(x-x0)+(y-x0)*(y-x0));

  //val = phi_liq_;

  if( rr > 3.e-6 ){
    val = 0.;
  }
  else {
    val = 1.;
  }

  return val;
}

INI_FUNC(init_heat_)
{
  //return um;
  return uw;
}

}//namespace yang
}//namespace tpetra


#endif
