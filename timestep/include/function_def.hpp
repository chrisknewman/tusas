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

/** Definition for residual function. Each residual function is called at each Gauss point for each equation with this signature:
- NAME:     name of function to call
- const boost::ptr_vector<Basis> &basis:     an array of basis function objects indexed by equation
- const int &i:    the current test function (row in residual vector)
- const double &dt_: the timestep size as prescribed in input file						
- const double &t_theta_: the timestep parameter as prescribed in input file
- const double &time: the current simulation time
- const int &eqn_id: the index of the current equation


*/


#define RES_FUNC(NAME)  double NAME(const boost::ptr_vector<Basis> &basis,\
                                    const int &i,\
                                    const double &dt_,\
			            const double &t_theta_,\
                                    const double &time,\
				    const int &eqn_id)

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

#define PRE_FUNC(NAME)  double NAME(const boost::ptr_vector<Basis> &basis,\
                                    const int &i,\
				    const int &j,\
				    const double &dt_,\
				    const double &t_theta_,\
				    const int &eqn_id)

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
				    const double *gradu,\
				    const double *xyz,\
				    const double &time)

/** Parameter function to propogate information from input file. Each parameter function is called at the beginning of each simulation.
- NAME:     name of function to call
- Teuchos::ParameterList *plist: paramterlist containing information defined in input file

*/

#define PARAM_FUNC(NAME) void NAME(Teuchos::ParameterList *plist) 


namespace heat
{
// double residual_heat_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)

/** Residual function for heat equation test problem. */
RES_FUNC(residual_heat_test_)
{

  //for heat eqn:
  //u[x,y,t]=exp(-2 pi^2 t)sin(pi x)sin(pi y)

  //for neumann:
  //u[x,y,t]=exp( -1/4 pi^2 t)sin(pi/4 x)
  //derivatives of the test function
  double dtestdx = basis[0].dphidx[i];
  double dtestdy = basis[0].dphidy[i];
  double dtestdz = basis[0].dphidz[i];
  //test function
  double test = basis[0].phi[i];
  //u, phi
  double u = basis[0].uu;
  double uold = basis[0].uuold;

  double ut = (u-uold)/dt_*test;
  double divgradu = (basis[0].dudx*dtestdx + basis[0].dudy*dtestdy + basis[0].dudz*dtestdz);//(grad u,grad phi)
  double divgradu_old = (basis[0].duolddx*dtestdx + basis[0].duolddy*dtestdy + basis[0].duolddz*dtestdz);//(grad u,grad phi)
 
 
  return ut + t_theta_*divgradu + (1.-t_theta_)*divgradu_old;
}
//double prec_heat_test_(const boost::ptr_vector<Basis> &basis, 
//			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_heat_test_)
{
  //cn probably want to move each of these operations inside of getbasis
  //derivatives of the test function
  double dtestdx = basis[0].dphidx[i];
  double dtestdy = basis[0].dphidy[i];
  double dtestdz = basis[0].dphidz[i];

  double dbasisdx = basis[0].dphidxi[j]*basis[0].dxidx
    +basis[0].dphideta[j]*basis[0].detadx
    +basis[0].dphidzta[j]*basis[0].dztadx;
  double dbasisdy = basis[0].dphidxi[j]*basis[0].dxidy
    +basis[0].dphideta[j]*basis[0].detady
    +basis[0].dphidzta[j]*basis[0].dztady;
  double dbasisdz = basis[0].dphidxi[j]*basis[0].dxidz
    +basis[0].dphideta[j]*basis[0].detadz
    +basis[0].dphidzta[j]*basis[0].dztadz;
  double test = basis[0].phi[i];
  double divgrad = dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz;
  double u_t =test * basis[0].phi[j]/dt_;
  return u_t + t_theta_*divgrad;
}
//double init_heat_test_(const double &x,
//		 const double &y,
//		 const double &z)
INI_FUNC(init_heat_test_)
{

  double pi = 3.141592653589793;

  return sin(pi*x)*sin(pi*y);
}
}//namespace heat


double rand_phi_zero_(const double &phi, const double &random_number)
{
  return 0.;
}

//furtado
double hp1_furtado_(const double &phi,const double &c)
{
  double dH = 2.35e9;
  double Tm = 1728.;
  return -30.*dH/Tm*phi*phi*(1.-phi)*(1.-phi);
}
double hpp1_furtado_(const double &phi,const double &c)
{
  double dH = 2.35e9;
  double Tm = 1728.;
  return -30.*dH/Tm* 2.* (1. -phi) *phi* (1. - 2. *phi)   ;
}
double gp1_furtado_(const double &phi)
{
  return 2.*phi*(1.-phi)*(1.-2.*phi);
}
double gpp1_furtado_(const double &phi)
{
  return 2.*( 1. - 6.* phi + 6.* phi*phi);
}
double w_furtado_(const double &delta)
{
  return .61e8;
}
double m_furtado_(const double &theta,const double &M,const double &eps)
{
  //return 1./13.47;
  return 1.;
}
double hp2_furtado_(const double &phi)
{
  double dH = 2.35e9;
  double rho = 0.37;
  double Cp = 5.42e6;
  //return dH/rho/Cp*30.*phi*phi*(1.-phi)*(1.-phi);
  //return 1.*phi*phi*(1.-phi)*(1.-phi);
  return 1.;
}
double rand_phi_furtado_(const double &phi, const double &random_number)
{
  //double a = .025;
  double a = 75.;
  double r = 16.*a*phi*phi
    *(1.-phi)*(1.-phi);
//   double r = 256.*a*phi*phi*phi*phi
//     *(1.-phi)*(1.-phi)*(1.-phi)*(1.-phi);
  if( phi > 1. || phi < 0. ) r = 0;
  return ((double)rand()/(RAND_MAX)*2.-1.)*r;
//   return random_number*16.*r;
}

//karma
//karma has phi in [-1 +1]
double hp2_karma_(const double &phi)
{
  //return 15./8./2.*(1.-2*phi*phi+phi*phi*phi*phi);//VF
  return .5;//IVF
}
double w_karma_(const double &delta)
{
  return 1.;
}
double gp1_karma_(const double &phi)
{
  return phi*(1.-phi*phi);
}
double hp1_karma_(const double &phi,const double &c)
{
  double lambda = 1.;
  return -lambda* (1. -phi*phi) * (1. -phi*phi)  ;
}
double hpp1_karma_(const double &phi,const double &c)//precon term
{
  double dH = 2.35e9;
  double Tm = 1728.;
  //return -30.*dH/Tm* 2.* (1. -phi) *phi* (1. - 2. *phi)   ;
  return 0.   ;
}
double gpp1_karma_(const double &phi)//precon term
{
  //return 2.*( 1. - 6.* phi + 6.* phi*phi);
  return 0.;
}
double m_karma_(const double &theta,const double &M,const double &eps)
{
  double eps4 = eps;
  double a_sbar = 1. - 3.* eps4;
  double eps_prime = 4.*eps4/a_sbar;
  double t0 = 1.;
  double g = a_sbar*(1. + eps_prime * (pow(sin(theta),M)+pow(cos(theta),M)));
  return t0*g*g;
}
double gs2_karma_( const double &theta, const double &M, const double &eps, const double &psi)
{ 
  //double g = 1. + eps_ * (M_*cos(theta));
  double W_0 = 1.;
  double eps4 = eps;
  double a_sbar = 1. - 3.* eps4;
  double eps_prime = 4.*eps4/a_sbar;
  double g = W_0*a_sbar*(1. + eps_prime * (pow(sin(theta),M)+pow(cos(theta),M)));
  return g*g;
}
double dgs2_2dtheta_karma_(const double &theta, const double &M, const double &eps, const double &psi)
{
  double W_0 = 1.;
  double eps4 = eps;
  double a_sbar = 1. - 3.* eps4;
  double eps_prime = 4.*eps4/a_sbar;
  double g = W_0*a_sbar*eps*(-4.*pow(cos(theta),3)*sin(theta) + 4.*cos(theta)*pow(sin(theta),3))*
    (1. + eps*(pow(cos(theta),4) + pow(sin(theta),4)));
  return g;
}

/*
@inproceedings{inproceedings,
author = {Cummins, Sharen and J Quirk, James and Kothe, Douglas},
year = {2002},
month = {11},
pages = {},
title = {An Exploration of the Phase Field Technique for Microstructure Solidification Modeling}
}
*/
namespace cummins
{

  double delta_ = -9999.;

  double m_cummins_(const double &theta,const double &M,const double &eps)
  {
    
    //double g = 1. + eps * (cos(M*(theta)));
    double g = (4.*eps*(cos(theta)*cos(theta)*cos(theta)*cos(theta) 
			+ sin(theta)*sin(theta)*sin(theta)*sin(theta) ) -3.*eps +1.   );
    return g*g;
  }
  double hp2_cummins_(const double &phi)
  {
    return 1.;
  }
// double residual_heat_(const boost::ptr_vector<Basis> &basis, 
// 		      const int &i, 
// 		      const double &dt_, 
// 		      const double &t_theta_, 
// 		      const double &delta, 
// 		      const double &time)
RES_FUNC(residual_heat_)
{
  //derivatives of the test function
  double dtestdx = basis[eqn_id].dphidxi[i]*basis[eqn_id].dxidx
    +basis[eqn_id].dphideta[i]*basis[eqn_id].detadx
    +basis[eqn_id].dphidzta[i]*basis[eqn_id].dztadx;
  double dtestdy = basis[eqn_id].dphidxi[i]*basis[eqn_id].dxidy
    +basis[eqn_id].dphideta[i]*basis[eqn_id].detady
    +basis[eqn_id].dphidzta[i]*basis[eqn_id].dztady;
  double dtestdz = basis[eqn_id].dphidxi[i]*basis[eqn_id].dxidz
    +basis[eqn_id].dphideta[i]*basis[eqn_id].detadz
    +basis[eqn_id].dphidzta[i]*basis[eqn_id].dztadz;
  //test function
  double test = basis[0].phi[i];
  //u, phi
  double u = basis[0].uu;
  double uold = basis[0].uuold;
  double phi = basis[1].uu;
  double phiold = basis[1].uuold;

  double D_ = 4.;

  double ut = (u-uold)/dt_*test;
  double divgradu = D_*(basis[0].dudx*dtestdx + basis[0].dudy*dtestdy + basis[0].dudz*dtestdz);//(grad u,grad phi)
  // 	      double divgradu_old = D_*(basis[0].duolddx*dtestdx + basis[0].duolddy*dtestdy + basis[0].duolddz*dtestdz);//(grad u,grad phi)
  //	      double divgradu = diffusivity_(*ubasis)*(basis[0].dudx*dtestdx + basis[0].dudy*dtestdy + basis[0].dudz*dtestdz);//(grad u,grad phi)
  double divgradu_old = D_*(basis[0].duolddx*dtestdx + basis[0].duolddy*dtestdy + basis[0].duolddz*dtestdz);//(grad u,grad phi)
  double hp2 = hp2_cummins_(phi);
  double phitu = -hp2*(phi-phiold)/dt_*test; 
  hp2 = hp2_cummins_(phiold);	
  double phitu_old = -hp2*(phiold-basis[1].uuoldold)/dt_*test; 
  return (ut + t_theta_*divgradu + (1.-t_theta_)*divgradu_old + t_theta_*phitu 
						       + (1.-t_theta_)*phitu_old);
}
double theta(const double &x,const double &y)
{
  double small = 1e-9;
  double pi = 3.141592653589793;
  double t = 0.;
  double sy = 1.;
  if(y < 0.) sy = -1.;
  double n = sy*sqrt(y*y);
  //double n = y;
  //std::cout<<y<<"   "<<n<<std::endl;
//   if(abs(x) < small && y > 0. ) t = pi/2.;
//   else if(abs(x) < small && y < 0. ) t = 3.*pi/2.;
//   else t= atan(n/x);
  if(std::abs(x) < small && y > 0. ) t = pi/2.;
  if(std::abs(x) < small && y < 0. ) t = 3.*pi/2.;
  if(x > small && y >= 0.) t= atan(n/x);
  if(x > small && y <0.) t= atan(n/x) + 2.*pi;
  if(x < -small) t= atan(n/x)+ pi;

  return t;
}
double gs_cummins_(const double &theta, const double &M, const double &eps, const double &psi)
{
  double eps_0_ = 1.;
  double g = eps_0_*(4.*eps*(cos(theta)*cos(theta)*cos(theta)*cos(theta) 
			     + sin(theta)*sin(theta)*sin(theta)*sin(theta) 
			     *(1.-2.*sin(psi)*sin(psi)*cos(psi)*cos(psi))
			     ) -3.*eps +1.   );
  return g;

}
double gs2_cummins_( const double &theta, const double &M, const double &eps, const double &psi)
{ 
  double eps_0_ = 1.;
  //double g = eps_0_*(1. + eps * (cos(M*theta)));
  //   double g = eps_0_*(4.*eps*(cos(theta)*cos(theta)*cos(theta)*cos(theta) 
  // 			     + sin(theta)*sin(theta)*sin(theta)*sin(theta) ) -3.*eps +1.   );
  double g =  gs_cummins_(theta,M,eps,psi);
  return g*g;
}
double dgs2_2dtheta_cummins_(const double &theta, const double &M, const double &eps, const double &psi)
{
  double eps_0_ = 1.;
  //return -1.*eps_0_*(eps*M*(1. + eps*cos(M*(theta)))*sin(M*(theta)));

  double g = gs_cummins_(theta,M,eps,psi);

  //double dg = 4.* eps* (-4.*cos(theta)*cos(theta)*cos(theta)*sin(theta) + 4.* cos(theta)*sin(theta)*sin(theta)*sin(theta));
  double dg = 4.* eps* (-4.*cos(theta)*cos(theta)*cos(theta)*sin(theta) + 
			4.* cos(theta)*sin(theta)*sin(theta)*sin(theta)*(1.-2.*cos(psi)*cos(psi)*sin(psi)*sin(psi)));

  return g*dg;

//   return eps_0_*4.*eps*(-4.*cos(theta)*cos(theta)*cos(theta)*sin(theta) + 4.*cos(theta)*sin(theta)*sin(theta)*sin(theta))
//     *(1. - 3.*eps + 4.*eps*(cos(theta)*cos(theta)*cos(theta)*cos(theta) 
// 			    + sin(theta)*sin(theta)*sin(theta)*sin(theta)));
}
double dgs2_2dpsi_cummins_(const double &theta, const double &M, const double &eps, const double &psi)
{
  //4 eps (-4 Cos[psi]^3 Sin[psi] + 4 Cos[psi] Sin[psi]^3) Sin[theta]^4
  double g = gs_cummins_(theta,M,eps,psi);
  double dg = 4.* eps* (-4.*cos(psi)*cos(psi)*cos(psi)*sin(psi) + 4.*cos(psi)*sin(psi)*sin(psi)*sin(psi))*sin(theta)*sin(theta)*sin(theta)*sin(theta);

  return g*dg;
}
double w_cummins_(const double &delta)
{
  return 1./delta/delta;
}
double gp1_cummins_(const double &phi)
{
  return phi*(1.-phi)*(1.-2.*phi);
}
double hp1_cummins_(const double &phi,const double &c)
{
  return -c*phi*phi*(1.-phi)*(1.-phi);
}
double hpp1_cummins_(const double &phi,const double &c)
{
  return -c* 2.* (1. -phi) *phi* (1. - 2. *phi);
}
double gpp1_cummins_(const double &phi)
{
  return 1. - 6.* phi + 6.* phi*phi;
}
// double residual_phase_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_phase_)
{
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
  //test function
  double test = basis[0].phi[i];
  //u, phi
  double u = basis[0].uu;
  double uold = basis[0].uuold;
  double phi = basis[1].uu;
  double phiold = basis[1].uuold;

  double eps_ = .05;
  double M_= 4.;

  double dphidx = basis[1].dudx;
  double dphidy = basis[1].dudy;
  double dphidz = basis[1].dudz;

  double theta_0_ =0.;
  double theta_ = theta(dphidx,dphidy)-theta_0_;

  double psi_ = 0.;
  double m = m_cummins_(theta_, M_, eps_);

  double phit = m*(phi-phiold)/dt_*test;

  double gs2 = gs2_cummins_(theta_, M_, eps_,0.);

  double divgradphi = gs2*(dphidx*dtestdx + dphidy*dtestdy + dphidz*dtestdz);//(grad u,grad phi)

  double dgdtheta = dgs2_2dtheta_cummins_(theta_, M_, eps_, 0.);	
  double dgdpsi = 0.;
  double curlgrad = dgdtheta*(-dphidy*dtestdx + dphidx*dtestdy);//cn could be a wrong sign here!!!
  //+dgdpsi*(-dphidz*dtestdx + dphidx*dtestdz);
  double w = w_cummins_(delta_);//cn_delta
  double gp1 = gp1_cummins_(phi);
  double phidel2 = gp1*w*test;

  double T_m_ = 1.55;
  double T_inf_ = 1.;
  double alpha_ = 191.82;

  double hp1 = hp1_cummins_(phi,5.*alpha_/delta_);
  double phidel = hp1*(T_m_ - u)*test;
  double rhs = divgradphi + curlgrad + phidel2 + phidel;

  dphidx = basis[1].duolddx;
  dphidy = basis[1].duolddy;
  dphidz = basis[1].duolddz;
  theta_ = theta(dphidx,dphidy)-theta_0_;
  //psi_ = psi(dphidx,dphidy,dphidz);
  psi_ =0.;
  gs2 = gs2_cummins_(theta_, M_, eps_,0.);
  divgradphi = gs2*dphidx*dtestdx + gs2*dphidy*dtestdy + gs2*dphidz*dtestdz;//(grad u,grad phi)
  dgdtheta = dgs2_2dtheta_cummins_(theta_, M_, eps_, 0.);

  curlgrad = dgdtheta*(-dphidy*dtestdx + dphidx*dtestdy);
  //+dgdpsi*(-dphidz*dtestdx + dphidx*dtestdz);

  gp1 = gp1_cummins_(phiold);
  
  phidel2 = gp1*w*basis[1].phi[i];
  
  hp1 = hp1_cummins_(phiold,5.*alpha_/delta_);

  phidel = hp1*(T_m_ - uold)*test;
	      
  double rhs_old = divgradphi + curlgrad + phidel2 + phidel;

  return phit + t_theta_*rhs + (1.-t_theta_)*rhs_old;

}
// double prec_heat_(const boost::ptr_vector<Basis> &basis, 
//		 const int &i
//	  , const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_heat_)
{
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
  double test = basis[0].phi[i];
  double D_ = 4.;
  double divgrad = D_*dbasisdx * dtestdx + D_*dbasisdy * dtestdy + D_*dbasisdz * dtestdz;
  double u_t =test * basis[0].phi[j]/dt_;
  return u_t + t_theta_*divgrad;
}
//double prec_phase_(const boost::ptr_vector<Basis> &basis, 
//		 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_phase_)
{
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

  double test = basis[0].phi[i];
  
  double dphidx = basis[1].dudx;
  double dphidy = basis[1].dudy;
  double dphidz = basis[1].dudz;

  double eps_ = .05;
  double M_= 4.;

  double theta_0_ =0.;
  double theta_ = theta(dphidx,dphidy)-theta_0_;

  double m = m_cummins_(theta_, M_, eps_);

  double phit = m*(basis[1].phi[j])/dt_*test;

  double gs2 = gs2_cummins_(theta_, M_, eps_,0.);

  double divgrad = gs2*dbasisdx * dtestdx + gs2*dbasisdy * dtestdy + gs2*dbasisdz * dtestdz;

  double dg2 = dgs2_2dtheta_cummins_(theta_, M_, eps_,0.);
  double curlgrad = -dg2*(dtestdy*dphidx -dtestdx*dphidy);

  return phit + t_theta_*divgrad + t_theta_*curlgrad;
}
double R_cummins(const double &theta)
{
  double R_0_ = .3;
  double eps_ = .05;
  double M_ = 4.;
  return R_0_*(1. + eps_ * cos(M_*(theta)));
}
//double init_heat_(const double &x,
//		 const double &y,
//		 const double &z)
INI_FUNC(init_heat_)
{
  double theta_0_ = 0.;
  double t = theta(x,y) - theta_0_;
  double r = R_cummins(t);

  double T_m_ = 1.55;
  double T_inf_ = 1.;

  double val = 0.;  

  if(x*x+y*y+z*z < r*r){
    val=T_m_;
  }
  else {
    val=T_inf_;
  }
  return val;
}
//double init_heat_const_(const double &x,
//		 const double &y,
//		 const double &z)
INI_FUNC(init_heat_const_)
{

  double T_inf_ = 1.;

  return T_inf_;
}
//double init_phase_(const double &x,
//		 const double &y,
//		 const double &z)
INI_FUNC(init_phase_)
{
  double theta_0_ = 0.;
  double t = theta(x,y) - theta_0_;
  double r = R_cummins(t);

  double phi_sol_ = 1.;
  double phi_liq_ = 0.;

  double val = 0.;  

  if(x*x+y*y+z*z < r*r){
    val=phi_sol_;
  }
  else {
    val=phi_liq_;
  }
  return val;
}
PARAM_FUNC(param_)
{
  delta_ = plist->get<double>("delta");
}
}//cummins

//double init_zero_(const double &x,
//		 const double &y,
//		 const double &z)
INI_FUNC(init_zero_)
{

  return 0.;
}
//double nbc_zero_(const Basis *basis,
//	 const int &i, 
//	 const double &dt_, 
//	 const double &t_theta_,
//	 const double &time)
NBC_FUNC(nbc_zero_)
{
  
  double phi = basis->phi[i];
  
  return 0.*phi;
}

//double dbc_zero_(const double &x,
//	const double &y,
//	const double &z,
//	const double &t)
KOKKOS_INLINE_FUNCTION
DBC_FUNC(dbc_zero_)
{  
  return 0.;
}
//double nbc_one_(const Basis *basis,
//	const int &i, 
//	const double &dt_, 
//	const double &t_theta_,
//	const double &time)
NBC_FUNC(nbc_one_)
{
  
  double phi = basis->phi[i];
  
  return 1.*phi;
}
//double dbc_one_(const double &x,
//       const double &y,
//       const double &z,
//       const double &t)
DBC_FUNC(dbc_one_)
{ 
  return 1.;
}
//double dbc_ten_(const double &x,
//       const double &y,
//       const double &z,
//       const double &t)
DBC_FUNC(dbc_ten_)
{
  
  
  return 10.*dbc_one_(x,
	       y,
	       z,
	       t);
}
//double nbc_mone_(const Basis *basis,
//	 const int &i, 
//	 const double &dt_, 
//	 const double &t_theta_,
//	 const double &time)
NBC_FUNC(nbc_mone_)
{

  double phi = basis->phi[i];

  return -1.*phi;
}
//double dbc_mone_(const double &x,
//	const double &y,
//	const double &z,
//	const double &t)
DBC_FUNC(dbc_mone_)
{
  return -1.;
}

//double init_neumann_test_(const double &x,
//		 const double &y,
//		 const double &z)
INI_FUNC(init_neumann_test_)
{

  double pi = 3.141592653589793;

  return sin(pi*x);
}








namespace farzadi
{

  //it appears the mm mesh is in um
  // 1m = 1e3 mm
  // 1m = 1e6 um

  double pi = 3.141592653589793;
  double theta_0_ = 0.;

  double phi_sol_ = 1.;
  double phi_liq_ = -1.;
  double k_ =0.14;
  double eps_ = .01;
  double M_= 4.;
  double lambda = 10.;
  double c_inf = 3.;
  double D= 6.267;
  //double D_ = 3.e-9;//m^2/s
  //double D_ = .003;//mm^2/s
  double D_ = 3.e3;//um^2/s
  //double D_ = D;
  //double m = -2.6;
  double tl = 925.2;//k
  double ts = 877.3;//k
  double G = .290900;//k/um
  double R = 3000.;//um/s
  double V = R;//um/s
  double t0 = ts;
  //double t0 = 900.;//k
  double dt0 = tl-ts;
  //double d0 = 5.e-9;//m
  double d0 = 5.e-3;//um

  //for the farzadiQuad1000x360mmr.e mesh...
  double pp = 360.;
  double ll = 40.;
  double aa = 14.;

  //double init_conc_farzadi_(const double &x,
  //		 const double &y,
  //		 const double &z)
PARAM_FUNC(param_)
{
  pp = plist->get<double>("pp");
  ll = plist->get<double>("ll");
  aa = plist->get<double>("aa");
}
INI_FUNC(init_conc_farzadi_)
{

  double val = -1.;

  return val;
}
double ff(const double y)
{
  return 2.+sin(y*aa*pi/pp);
}
  //double init_phase_farzadi_(const double &x,
  //		 const double &y,
  //		 const double &z)
INI_FUNC(init_phase_farzadi_)
{
  double val = phi_liq_;
  double r = ll*(1.+ff(y)*ff(y/2.)*ff(y/4.));

  //r=.9;

  if(x < r){
    val=phi_sol_;
  }
  else {
    val=phi_liq_;
  }
  return val;
  //return 1.;
}
INI_FUNC(init_phase_rand_farzadi_)
{
  double val = phi_liq_;
  int ri = rand()%(100);//random int between 0 and 100
  double rd = (double)ri/ll;
  double r = .5+rd*std::abs(sin(y*aa*pi/pp));

  //r=.9;

  if(x < r){
    val=phi_sol_;
  }
  else {
    val=phi_liq_;
  }
  return val;
  //return 1.;
}
double tscale_(const double &x, const double &time)
{
  //x and time come in as nondimensional quantities here...
  double xx = d0*x;
  double tt = d0*d0/D_*time;
  double t = t0 + G*(xx-V*tt);
  return (t-ts)/dt0;
}

// double residual_phase_farzadi_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_phase_farzadi_)
{
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
  //test function
  double test = basis[0].phi[i];
  //u, phi
  double u = basis[0].uu;
  double uold = basis[0].uuold;
  double phi = basis[1].uu;
  double phiold = basis[1].uuold;

  double dphidx = basis[1].dudx;
  double dphidy = basis[1].dudy;

  //double theta_ = theta(basis[1].duolddx,basis[1].duolddy);
  double theta_ = cummins::theta(basis[1].dudx,basis[1].dudy);

  double m = (1+(1-k_)*u)*cummins::m_cummins_(theta_, M_, eps_);//cn we probably need u and uold here for CN...
  //double m = m_cummins_(theta_, M_, eps_);//cn we probably need u and uold here for CN...
  //double theta_old = theta(dphidx,dphidy);
  //double mold = (1+(1-k_)*uold)*m_cummins_(theta_old, M_, eps_);

  //double phit = (t_theta_*m+(1.-t_theta_)*mold)*(phi-phiold)/dt_*test;
  //double phit = m*(phi-phiold)/dt_*test;

  double gs2 = cummins::gs2_cummins_(theta_, M_, eps_,0.);
  double divgradphi = gs2*(dphidx*dtestdx + dphidy*dtestdy);//(grad u,grad phi)

  double phit = (1.+(1.-k_)*u)*gs2*(phi-phiold)/dt_*test;

  double dgdtheta = cummins::dgs2_2dtheta_cummins_(theta_, M_, eps_, 0.);	
  double dgdpsi = 0.;
  double curlgrad = dgdtheta*(-dphidy*dtestdx + dphidx*dtestdy);

  double gp1 = -(phi - phi*phi*phi);
  double phidel2 = gp1*test;

  double x = basis[0].xx;
  //double t = t0 + G*(x-R*time);
  //double t_scale = (t-ts)/dt0;
  double t_scale = tscale_(x,time);

  double hp1 = lambda*(1. - phi*phi)*(1. - phi*phi)*(u+t_scale);
  double phidel = hp1*test;
  //phidel = 0.;
  double rhs = divgradphi + curlgrad + phidel2 + phidel;

//   dphidx = basis[1].duolddx;
//   dphidy = basis[1].duolddy;
//   theta_ = theta(dphidx,dphidy);

//   gs2 = gs2_cummins_(theta_, M_, eps_,0.);
//   divgradphi = gs2*dphidx*dtestdx + gs2*dphidy*dtestdy;//(grad u,grad phi)
//   dgdtheta = dgs2_2dtheta_cummins_(theta_, M_, eps_, 0.);

//   curlgrad = dgdtheta*(-dphidy*dtestdx + dphidx*dtestdy);

//   gp1 = -(phiold-phiold*phiold*phiold);
  
//   phidel2 = gp1*test;
  
//   hp1 = lambda*(1.-phiold*phiold)*(1.-phiold*phiold)*(uold+t_scale);

//   phidel = hp1*test;
	      
//   double rhs_old = divgradphi + curlgrad + phidel2 + phidel;

  return phit + t_theta_*rhs;// + (1.-t_theta_)*rhs_old*0.;

}

// double residual_conc_farzadi_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_conc_farzadi_)
{
  //derivatives of the test function
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;
  //test function
  double test = basis[0].phi[i];
  //u, phi
  double u = basis[0].uu;
  double uold = basis[0].uuold;
  double phi = basis[1].uu;
  double phiold = basis[1].uuold;
  double dphidx = basis[1].dudx;
  double dphidy = basis[1].dudy;

  double ut = (1.+k_)/2.*(u-uold)/dt_*test;
  //ut = (u-uold)/dt_*test;
  double divgradu = D*(1.-phi)/2.*(basis[0].dudx*dtestdx + basis[0].dudy*dtestdy);//(grad u,grad phi)
  //divgradu = (basis[0].dudx*dtestdx + basis[0].dudy*dtestdy);
  double divgradu_old = D*(1.-phiold)/2*(basis[0].duolddx*dtestdx + basis[0].duolddy*dtestdy);//(grad u,grad phi)

  //j is antitrapping current
  // j grad test here... j1*dtestdx + j2*dtestdy 
  // what if dphidx*dphidx + dphidy*dphidy = 0?

  double norm = sqrt(dphidx*dphidx + dphidy*dphidy);
  double small = 1.e-12;
  
  double j_coef = 0.;
  if (small < norm) {
    j_coef = (1.+(1.-k_)*u)/sqrt(8.)/norm*(phi-phiold)/dt_;
  } 
  //j_coef = 0.;
  double j1 = j_coef*dphidx;
  double j2 = j_coef*dphidy;
  double divj = j1*dtestdx + j2*dtestdy;

  double dphiolddx = basis[1].duolddx;
  double dphiolddy = basis[1].duolddy; 
  norm = sqrt(dphidx*dphidx + dphidy*dphidy);
  j_coef = 0.;
  if (small < norm) {
    j_coef = (1.+(1.-k_)*uold)/sqrt(8.)/norm*(phiold-basis[1].uuoldold)/dt_; 
  }
  j1 = j_coef*dphidx;
  j2 = j_coef*dphidy;
  j_coef = 0.;
  double divj_old = j1 *dtestdx + j2 *dtestdy;

  double h = phi*(1.+(1.-k_)*u);
  double hold = phiold*(1. + (1.-k_)*uold);

  //double phitu = -.5*(h-hold)/dt_*test; 
  double phitu = -.5*(phi-phiold)/dt_*(1.+(1.-k_)*u)*test; 
  //phitu = 1.*test; 
//   h = hold;
//   hold = basis[1].uuoldold*(1. + (1.-k_)*basis[0].uuoldold);
//   double phitu_old = -.5*(h-hold)/dt_*test;
 
  //return ut*0.  + t_theta_*(divgradu + divj*0.) + (1.-t_theta_)*(divgradu_old + divj_old)*0. + t_theta_*phitu*0. + (1.-t_theta_)*phitu_old*0.;

  return ut + t_theta_*divgradu  + t_theta_*divj + t_theta_*phitu;
}
  //double prec_phase_farzadi_(const boost::ptr_vector<Basis> &basis, 
  //			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_phase_farzadi_)
{
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

  double test = basis[1].phi[i];
  
  double dphidx = basis[1].dudx;
  double dphidy = basis[1].dudy;
  double dphidz = basis[1].dudz;

  double u = basis[0].uu;
  double phi = basis[1].uu;


  double theta_ = cummins::theta(dphidx,dphidy)-theta_0_;

  double gs2 = cummins::gs2_cummins_(theta_, M_, eps_,0.);

  double m = (1.+(1.-k_)*0.*u)*gs2;
  double phit = m*(basis[1].phi[j])/dt_*test;
  //phit = (basis[1].phi[j])*test;

  double divgrad = gs2*dbasisdx * dtestdx + gs2*dbasisdy * dtestdy;// + gs2*dbasisdz * dtestdz;

  //double dg2 = cummins::dgs2_2dtheta_cummins_(theta_, M_, eps_,0.);
  //double curlgrad = dg2*(-dtestdx*dphidy + dtestdy*dphidx );
  //double t1 = (-1.+3.*phi*phi)*basis[1].phi[j]*test;
  //double t2 = -4.*lambda*(u + 0.)*(-phi+phi*phi*phi)*basis[1].phi[j]*test;

  //return phit + t_theta_*(divgrad + 0.*curlgrad + 0.*t1 + 0.*t2);
  return phit + t_theta_*(divgrad);
}
  //double prec_conc_farzadi_(const boost::ptr_vector<Basis> &basis, 
  //		 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_conc_farzadi_)
{
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
  double test = basis[0].phi[i];
  double divgrad = D*(1.-basis[1].uu)/2.*(dbasisdx * dtestdx + dbasisdy * dtestdy);


  //double dphidx = basis[1].dudx;
  //double dphidy = basis[1].dudy;
  //double norm = sqrt(dphidx*dphidx + dphidy*dphidy);
  //double small = 1.e-12;
  //double phi = basis[1].uu;
  //double phiold = basis[1].uuold;
  //double j_coef = 0.;
  //if (small < norm) {
  // j_coef = (1.+(1.-k_)* basis[0].phi[j])/sqrt(8.)/norm*(phi-phiold)/dt_;
  //} 
  ////j_coef = 0.;
  //double j1 = j_coef*dphidx;
  //double j2 = j_coef*dphidy;
  //double divj = j1*dtestdx + j2*dtestdy;



  double u_t =(1.+k_)/2.*test * basis[0].phi[j]/dt_;
  //double phitu = -.5*(1.-k_)*basis[0].phi[j]*test*(phi-phiold)/dt_; 
  //return u_t + t_theta_*(divgrad + 0.*divj + 0.*phitu);
  return u_t + t_theta_*(divgrad);
}
//double postproc_c_(const double *u, const double *gradu, const double *xyz, const double &time)
PPR_FUNC(postproc_c_)
{

  double uu = u[0];
  double phi = u[1];

  return -c_inf*(1.+k_-phi+k_*phi)*(-1.-uu+k_*uu)/2./k_;
}
//double postproc_t_(const double *u, const double *gradu, const double *xyz, const double &time)
PPR_FUNC(postproc_t_)
{
  // x is in nondimensional space, tscale_ takes in nondimensional and converts to um
  double x = xyz[0];

  return tscale_(x,time);
}
}//namespace farzadi

// double residual_robin_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_robin_test_)
{
  //1-D robin bc test problem, steady state
  // -d^2 u/dx^2 + a^2 u = 0
  // u(x=0) = 0; grad u = b(1-u) | x=1
  // u[x;b,a]=b cosh ( ax)/[b cosh(a)+a sinh(a)]



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
  //test function
  double test = basis[0].phi[i];
  //u, phi
  double u = basis[0].uu;
  //double uold = basis[0].uuold;

  double a =10.;

  double au = a*a*u*test;
  double divgradu = (basis[0].dudx*dtestdx + basis[0].dudy*dtestdy + basis[0].dudz*dtestdz);//(grad u,grad phi)
  //double divgradu_old = (basis[0].duolddx*dtestdx + basis[0].duolddy*dtestdy + basis[0].duolddz*dtestdz);//(grad u,grad phi)
 
 
  return divgradu + au;
}

//double prec_robin_test_(const boost::ptr_vector<Basis> &basis, 
//			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_robin_test_)
{
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
  double test = basis[0].phi[i];
  double divgrad = dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz;
  double a =10.;
  double u_t =test * a*a*basis[0].phi[j];
  return u_t + t_theta_*divgrad;
}

//double nbc_robin_test_(const Basis *basis,
//	 const int &i, 
//	 const double &dt_, 
//	 const double &t_theta_,
//	 const double &time)
NBC_FUNC(nbc_robin_test_)
{

  double test = basis->phi[i];
  double u = basis[0].uu;

  double b =1.;

  return b*(1.-u)*test;
}

namespace liniso
{
// double residual_liniso_x_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_liniso_x_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double E = 1e6;
  double nu = .3;

  double strain[6], stress[6];//x,y,z,yx,zy,zx


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
  //test function
  //double test = basis[0].phi[i];
  //u, phi
  //double u = basis[0].uu;

  strain[0] = basis[0].dudx;
  strain[1] = basis[1].dudy;
  strain[2] = basis[2].dudz;
  strain[3] = basis[0].dudy + basis[1].dudx;
  strain[4] = basis[1].dudz + basis[2].dudy;
  strain[5] = basis[0].dudz + basis[2].dudx;

  //plane stress
//   double c = E/(1.-nu*nu);
//   double c1 = c;
//   double c2 = c*nu;
//   double c3 = c*(1.-nu)/2.;//always confused about the 2 here
  //plane strain
  double c = E/(1.+nu)/(1.-2.*nu);
  double c1 = c*(1.-nu);
  double c2 = c*nu;
  double c3 = c*(1.-2.*nu)/2.;

  stress[0] = c1*strain[0] + c2*strain[1] + c2*strain[2];
  //stress[1] = c2*strain[0] + c1*strain[1] + c2*strain[2];
  //stress[2] = c2*strain[0] + c2*strain[1] + c1*strain[2];
  stress[3] = c3*strain[3]; 
  //stress[4] = c3*strain[4]; 
  stress[5] = c3*strain[5]; 

  double divgradu = stress[0]*dtestdx + stress[3]*dtestdy + stress[5]*dtestdz;//(grad u,grad phi)
 
 
  return divgradu;
}
// double residual_liniso_y_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_liniso_y_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double E = 1e6;
  double nu = .3;

  double strain[6], stress[6];//x,y,z,yx,zy,zx


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
  //test function
  //double test = basis[0].phi[i];
  //u, phi
  //double u = basis[0].uu;

  strain[0] = basis[0].dudx;
  strain[1] = basis[1].dudy;
  strain[2] = basis[2].dudz;
  strain[3] = basis[0].dudy + basis[1].dudx;
  strain[4] = basis[1].dudz + basis[2].dudy;
  strain[5] = basis[0].dudz + basis[2].dudx;

  //plane stress
//   double c = E/(1.-nu*nu);
//   double c1 = c;
//   double c2 = c*nu;
//   double c3 = c*(1.-nu)/2.;//always confused about the 2 here
  //plane strain
  double c = E/(1.+nu)/(1.-2.*nu);
  double c1 = c*(1.-nu);
  double c2 = c*nu;
  double c3 = c*(1.-2.*nu)/2.;

  //stress[0] = c1*strain[0] + c2*strain[1] + c2*strain[2];
  stress[1] = c2*strain[0] + c1*strain[1] + c2*strain[2];
  //stress[2] = c2*strain[0] + c2*strain[1] + c1*strain[2];
  stress[3] = c3*strain[3]; 
  stress[4] = c3*strain[4]; 
  //stress[5] = c3*strain[5]; 

  double divgradu = stress[1]*dtestdy + stress[3]*dtestdx + stress[4]*dtestdz;//(grad u,grad phi)
 
 
  return divgradu;
}
// double residual_liniso_z_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_liniso_z_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double E = 1e6;
  double nu = .3;

  double strain[6], stress[6];//x,y,z,yx,zy,zx


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
  //test function
  //double test = basis[0].phi[i];
  //u, phi
  //double u = basis[0].uu;

  strain[0] = basis[0].dudx;
  strain[1] = basis[1].dudy;
  strain[2] = basis[2].dudz;
  strain[3] = basis[0].dudy + basis[1].dudx;
  strain[4] = basis[1].dudz + basis[2].dudy;
  strain[5] = basis[0].dudz + basis[2].dudx;

  //plane stress
//   double c = E/(1.-nu*nu);
//   double c1 = c;
//   double c2 = c*nu;
//   double c3 = c*(1.-nu)/2.;//always confused about the 2 here
  //plane strain
  double c = E/(1.+nu)/(1.-2.*nu);
  double c1 = c*(1.-nu);
  double c2 = c*nu;
  double c3 = c*(1.-2.*nu)/2.;

  //stress[0] = c1*strain[0] + c2*strain[1] + c2*strain[2];
  //stress[1] = c2*strain[0] + c1*strain[1] + c2*strain[2];
  stress[2] = c2*strain[0] + c2*strain[1] + c1*strain[2];
  //stress[3] = c3*strain[3]; 
  stress[4] = c3*strain[4]; 
  stress[5] = c3*strain[5]; 

  double divgradu = stress[2]*dtestdz + stress[4]*dtestdy + stress[5]*dtestdx;//(grad u,grad phi)
 
 
  return divgradu;
}



  //double prec_liniso_x_test_(const boost::ptr_vector<Basis> &basis, 
  //		 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_liniso_x_test_)
{

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

  double E = 1e6;
  double nu = .3;

  double strain[6], stress[6];//x,y,z,yx,zy,zx

  //strain[0] = basis[0].dudx;
  //strain[1] = basis[1].dudy;
  //strain[2] = basis[2].dudz;
  //strain[3] = basis[0].dudy + basis[1].dudx;
  //strain[4] = basis[1].dudz + basis[2].dudy;
  //strain[5] = basis[0].dudz + basis[2].dudx;

  //strain[0] = dbasisdx;
  //strain[1] = dbasisdy;
  //strain[2] = dbasisdz;
  //strain[3] = dbasisdy + dbasisdx;
  //strain[4] = dbasisdz + dbasisdy;
  //strain[5] = dbasisdz + dbasisdx;

  strain[0] = dbasisdx;
  strain[1] = 0;
  strain[2] = 0;
  strain[3] = 0 + dbasisdy;
  strain[4] = 0;
  strain[5] = dbasisdz +  0;

  //plane stress
//   double c = E/(1.-nu*nu);
//   double c1 = c;
//   double c2 = c*nu;
//   double c3 = c*(1.-nu)/2.;//always confused about the 2 here
  //plane strain
  double c = E/(1.+nu)/(1.-2.*nu);
  double c1 = c*(1.-nu);
  double c2 = c*nu;
  double c3 = c*(1.-2.*nu)/2.;

  stress[0] = c1*strain[0] + c2*strain[1] + c2*strain[2];
  //stress[1] = c2*strain[0] + c1*strain[1] + c2*strain[2];
  //stress[2] = c2*strain[0] + c2*strain[1] + c1*strain[2];
  stress[3] = c3*strain[3]; 
  //stress[4] = c3*strain[4]; 
  stress[5] = c3*strain[5]; 

  double divgradu = stress[0]*dtestdx + stress[3]*dtestdy + stress[5]*dtestdz;//(grad u,grad phi)
 
 
  return divgradu;
}
  //double prec_liniso_y_test_(const boost::ptr_vector<Basis> &basis, 
  //		 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_liniso_y_test_)
{

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

  double E = 1e6;
  double nu = .3;

  double strain[6], stress[6];//x,y,z,yx,zy,zx

  //strain[0] = basis[0].dudx;
  //strain[1] = basis[1].dudy;
  //strain[2] = basis[2].dudz;
  //strain[3] = basis[0].dudy + basis[1].dudx;
  //strain[4] = basis[1].dudz + basis[2].dudy;
  //strain[5] = basis[0].dudz + basis[2].dudx;

  //strain[0] = dbasisdx;
  //strain[1] = dbasisdy;
  //strain[2] = dbasisdz;
  //strain[3] = dbasisdy + dbasisdx;
  //strain[4] = dbasisdz + dbasisdy;
  //strain[5] = dbasisdz + dbasisdx;
  strain[0] = 0;
  strain[1] = dbasisdy;
  strain[2] = 0;
  strain[3] = 0 + dbasisdx;
  strain[4] = dbasisdz + 0;
  strain[5] = 0 + 0;

  //plane stress
//   double c = E/(1.-nu*nu);
//   double c1 = c;
//   double c2 = c*nu;
//   double c3 = c*(1.-nu)/2.;//always confused about the 2 here
  //plane strain
  double c = E/(1.+nu)/(1.-2.*nu);
  double c1 = c*(1.-nu);
  double c2 = c*nu;
  double c3 = c*(1.-2.*nu)/2.;

  //stress[0] = c1*strain[0] + c2*strain[1] + c2*strain[2];
  stress[1] = c2*strain[0] + c1*strain[1] + c2*strain[2];
  //stress[2] = c2*strain[0] + c2*strain[1] + c1*strain[2];
  stress[3] = c3*strain[3]; 
  stress[4] = c3*strain[4]; 
  //stress[5] = c3*strain[5]; 

  double divgradu = stress[1]*dtestdy + stress[3]*dtestdx + stress[4]*dtestdz;//(grad u,grad phi)

  return divgradu;
}
  //double prec_liniso_z_test_(const boost::ptr_vector<Basis> &basis, 
  //		 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_liniso_z_test_)
{

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

  double E = 1e6;
  double nu = .3;

  double strain[6], stress[6];//x,y,z,yx,zy,zx

  //strain[0] = basis[0].dudx;
  //strain[1] = basis[1].dudy;
  //strain[2] = basis[2].dudz;
  //strain[3] = basis[0].dudy + basis[1].dudx;
  //strain[4] = basis[1].dudz + basis[2].dudy;
  //strain[5] = basis[0].dudz + basis[2].dudx;

  //strain[0] = dbasisdx;
  //strain[1] = dbasisdy;
  //strain[2] = dbasisdz;
  //strain[3] = dbasisdy + dbasisdx;
  //strain[4] = dbasisdz + dbasisdy;
  //strain[5] = dbasisdz + dbasisdx;
  strain[0] = 0;
  strain[1] = 0;
  strain[2] = dbasisdz;
  strain[3] = 0 + 0;
  strain[4] = 0 + dbasisdy;
  strain[5] = 0 + dbasisdx;

  //plane stress
//   double c = E/(1.-nu*nu);
//   double c1 = c;
//   double c2 = c*nu;
//   double c3 = c*(1.-nu)/2.;//always confused about the 2 here
  //plane strain
  double c = E/(1.+nu)/(1.-2.*nu);
  double c1 = c*(1.-nu);
  double c2 = c*nu;
  double c3 = c*(1.-2.*nu)/2.;

  //stress[0] = c1*strain[0] + c2*strain[1] + c2*strain[2];
  //stress[1] = c2*strain[0] + c1*strain[1] + c2*strain[2];
  stress[2] = c2*strain[0] + c2*strain[1] + c1*strain[2];
  //stress[3] = c3*strain[3]; 
  stress[4] = c3*strain[4]; 
  stress[5] = c3*strain[5]; 

  double divgradu = stress[2]*dtestdz + stress[4]*dtestdy + stress[5]*dtestdx;//(grad u,grad phi)


  return divgradu;
}
// double residual_linisobodyforce_y_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_linisobodyforce_y_test_)
{
  //this is taken from a test that had E=2e11; nu=.3 and body force -1e10 in the y direction;
  //we reuse the test case that has E=1e6 and scale the body force accordingly by 5e-6  

  //test function
  double test = basis[0].phi[i]; 

  double bf = -1.e10*5.e-6;

  double divgradu = residual_liniso_y_test_(basis,i,dt_,t_theta_,time,eqn_id) + bf*test;
 
  return divgradu;
}

// double residual_linisoheat_x_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_linisoheat_x_test_)
{
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double gradu = basis[3].dudx;
  double c = 1.e-6;
  double alpha = 1.e-4;;
  double E = 1.;

  double divgradu = c*residual_liniso_x_test_(basis,i,dt_,t_theta_,time,eqn_id) - alpha*E*gradu*dtestdx;
 
  return divgradu;
}


// double residual_linisoheat_y_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_linisoheat_y_test_)
{
  //test function
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;
  double gradu = basis[3].dudy;

  double c = 1.e-6;
  double alpha = 1.e-4;
  double E = 1.;


  double divgradu = c*residual_liniso_y_test_(basis,i,dt_,t_theta_,time,eqn_id) - alpha*E*gradu*dtestdy;
 
  return divgradu;
}



// double residual_linisoheat_z_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_linisoheat_z_test_)
{
  double dtestdz = basis[0].dphidxi[i]*basis[0].dxidz
    +basis[0].dphideta[i]*basis[0].detadz
    +basis[0].dphidzta[i]*basis[0].dztadz;
  double gradu = basis[3].dudz;
  double c = 1.e-6;
  double alpha = 1.e-4;
  double E = 1.;

  double divgradu = c*residual_liniso_z_test_(basis,i,dt_,t_theta_,time,eqn_id) - alpha*E*gradu*dtestdz;
 
  return divgradu;
}



// double residual_divgrad_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_divgrad_test_)
{
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
  //test function
  //double test = basis[3].phi[i];
  //u, phi
  //double u = basis[0].uu;
  //double uold = basis[0].uuold;

  double divgradu = (basis[3].dudx*dtestdx + basis[3].dudy*dtestdy + basis[3].dudz*dtestdz);//(grad u,grad phi)
  //double divgradu_old = (basis[0].duolddx*dtestdx + basis[0].duolddy*dtestdy + basis[0].duolddz*dtestdz);//(grad u,grad phi)
 
 
  return divgradu;
}
}//namespace liniso

namespace uehara
{

  double L = 3.e3;//J/m^3
  double m = 2.5e5;
  double a = 10.;//m^4
  double rho = 1.e3;//kg/m^3
  double c = 5.e2;//J/kg/K
  double k = 150.;//W/m/K
  //double k = 1.5;//W/m/K
  double r0 = 29.55;

  double giga = 1.e+9;
  double E = 200.*giga;//GPa
  //double E = 200.*1.e9;//Pa
  double nu = .3;

  //plane stress
  double c0 = E/(1.-nu*nu);
  double c1 = c0;
  double c2 = c0*nu;
  double c3 = c0*(1.-nu)/2.;//always confused about the 2 here
  //plane strain
//   double c0 = E/(1.+nu)/(1.-2.*nu);
//   double c1 = c0*(1.-nu);
//   double c2 = c0*nu;
//   double c3 = c0*(1.-2.*nu)/2.;

  double alpha = 5.e-6;//1/K
  //double alpha = 0.;//1/K
  double beta = 1.5e-3;
  //double beta = 0.;
  
  //double init_heat_(const double &x,
  //	   const double &y,
  //	   const double &z)
INI_FUNC(init_heat_)
{
  double val = 400.;  

  return val;
}

  //double init_phase_(const double &x,
  //	   const double &y,
  //	   const double &z)
INI_FUNC(init_phase_)
{

  //this is not exactly symmetric on the fine mesh
  double phi_sol_ = 1.;
  double phi_liq_ = 0.;

  double val = phi_sol_ ;  

  double dx = 1.e-2;
  double r = r0*dx;
  double r1 = (r0-1)*dx;

  //if(y > r){
  if( (x > r1 && y > r) ||(x > r && y > r1) ){
    val=phi_sol_;
  }
  else {
    val=phi_liq_;
  }

  return val;
}
  //double init_phase_c_(const double &x,
  //	   const double &y,
  //	   const double &z)
INI_FUNC(init_phase_c_)
{

  //this is not exactly symmetric on the fine mesh
  double phi_sol_ = 1.;
  double phi_liq_ = 0.;

  double val = phi_sol_ ;  

  double dx = 1.e-2;
  double r = r0*dx;

  double rr = sqrt(x*x+y*y);

  //if(y > r){
  if( rr > r0*sqrt(2.)*dx ){
    val=phi_sol_;
  }
  else {
    val=phi_liq_;
  }

  return val;
}
  //double init_heat_seed_(const double &x,
  //	   const double &y,
  //	   const double &z)
INI_FUNC(init_heat_seed_)
{

  //this is not exactly symmetric on the fine mesh
  double phi_sol_ = 300.;
  double phi_liq_ = 400.;

  double val = phi_sol_ ;  

  double dx = 1.e-2;
  double r = r0*dx;
  double r1 = (r0-1)*dx;

  //if(y > r){
  if( (x > r1 && y > r) ||(x > r && y > r1) ){
    val=phi_sol_;
  }
  else {
    val=phi_liq_;
  }

  return val;
}
  //double init_heat_seed_c_(const double &x,
  //	   const double &y,
  //	   const double &z)
INI_FUNC(init_heat_seed_c_)
{

  //this is not exactly symmetric on the fine mesh
  double phi_sol_ = 300.;
  double phi_liq_ = 400.;

  double val = phi_sol_ ;  

  double dx = 1.e-2;
  double r = r0*dx;
  double r1 = (r0-1)*dx;

  double rr = sqrt(x*x+y*y);

  //if(y > r){
  if( rr > r0*sqrt(2.)*dx ){
    val=phi_sol_;
  }
  else {
    val=phi_liq_;
  }

  return val;
}
  //double dbc_(const double &x,
  //       const double &y,
  //       const double &z,
  //       const double &t)
DBC_FUNC(dbc_)
{  
  return 300.;
}
double conv_bc_(const Basis *basis,
		 const int &i, 
		 const double &dt_, 
		 const double &t_theta_,
		 const double &time)
{

  double test = basis->phi[i];
  double u = basis->uu;
  double uw = 300.;//K
  double h =1.e4;//W/m^2/K
  
  return h*(uw-u)*test/rho/c;
}
  //double nbc_stress_(const Basis *basis,
  //	 const int &i, 
  //	 const double &dt_, 
  //	 const double &t_theta_,
  //	 const double &time)
NBC_FUNC(nbc_stress_)
{

  double test = basis->phi[i];
  
  return -alpha*300.*test;
}
// double residual_phase_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_phase_)
{
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
  //test function
  double test = basis[0].phi[i];
  //u, phi
  double phi = basis[0].uu;
  double phiold = basis[0].uuold;
  double u = basis[1].uu;
  //double uold = basis[1].uuold;

  double dphidx = basis[0].dudx;
  double dphidy = basis[0].dudy;
  double dphidz = basis[0].dudz;

  double b = 5.e-5;//m^3/J
  //b = 5.e-7;//m^3/J
  double f = 0.;
  double um = 400.;//K

  double phit = m*(phi-phiold)/dt_*test;
  double divgradphi = a*(dphidx*dtestdx + dphidy*dtestdy + dphidz*dtestdz);//(grad u,grad test)
  //double M = b*phi*(1. - phi)*(L*(um - u)/um + f);
  double M = .66*150000000.*b*phi*(1. - phi)*(L*(um - u)/um + f);
  //double g = -phi*(1. - phi)*(phi - .5 + M)*test;
  double g = -(phi*(1. - phi)*(phi - .5)+phi*(1. - phi)*M)*test;
  double rhs = divgradphi + g;

  return (phit + rhs)/m;

}


// double residual_stress_x_dt_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_stress_x_dt_)
{
  double strain[3];//x,y,yx

  strain[0] = (basis[2].dudx-basis[2].duolddx)/dt_;
  strain[1] = (basis[3].dudy-basis[3].duolddy)/dt_;

  double stress = c1*strain[0] + c2*strain[1];

  return stress;
}
// double residual_stress_y_dt_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_stress_y_dt_)
{
  double strain[3];//x,y,z,yx,zy,zx

  strain[0] = (basis[2].dudx-basis[2].duolddx)/dt_;
  strain[1] = (basis[3].dudy-basis[3].duolddy)/dt_;

  double stress = c2*strain[0] + c1*strain[1];

  return stress;
}
// double residual_heat_(const boost::ptr_vector<Basis> &basis, 
// 		      const int &i, 
// 		      const double &dt_, 
// 		      const double &t_theta_, 
// 		      const double &delta, 
// 		      const double &time)
RES_FUNC(residual_heat_)
{
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
  //test function
  double test = basis[1].phi[i];
  //u, phi
  double phi = basis[0].uu;
  double phiold = basis[0].uuold;
  double phioldold = basis[0].uuoldold;
  double u = basis[1].uu;
  double uold = basis[1].uuold;

  double dudx = basis[1].dudx;
  double dudy = basis[1].dudy;
  //double dudz = basis[1].dudz;

  double ut = rho*c*(u-uold)/dt_*test;
  double divgradu = k*(dudx*dtestdx + dudy*dtestdy);
  //double phitu = -30.*L*phi*phi*(1.-phi)*(1.-phi)*(phi-phioldold)/2./dt_*test; 
  double h = phi*phi*(1.-phi)*(1.-phi);
  double phitu = -30.*L*h*(phi-phiold)/dt_*test; 
  
  //thermal term
  double stress = test*alpha*u*(residual_stress_x_dt_(basis, 
						      i, dt_, t_theta_,
						      time, eqn_id)
				+residual_stress_y_dt_(basis, 
						       i, dt_, t_theta_,
						       time, eqn_id));
  

  double rhs = divgradu + phitu + stress;

  return (ut + rhs)/rho/c;

}
// double residual_liniso_x_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_liniso_x_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double strain[3], stress[3];//x,y,yx


  //derivatives of the test function
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;
  //test function
  double test = basis[0].phi[i];
  //u, phi
  //double u = basis[0].uu;

  //double ut = (basis[1].uu - basis[1].uuoldold)/dt_/2;//thermal strain
  double ut = (basis[1].uu - basis[1].uuold)/dt_;//thermal strain

  double phi = basis[0].uu;
  double h = phi*phi*(1.-phi)*(1.-phi);
  //double hp = 2.*(1.-phi)*(1.-phi)*phi-2.*(1.-phi)*phi*phi;//2 (1 - x)^2 x - 2 (1 - x) x^2
  // h' p_t p_x +h p_t_x
  double strain_phi = 30.*beta*h*(phi-basis[0].uuold)/dt_;
//   double strain_phi = 0.*2.*30.*beta*(c1+c2)*(hp*(phi-basis[0].uuold)/dt_*basis[0].dudx
// 					   +h*(basis[0].dudx-basis[0].duolddx)/dt_
// 					   )*test;
  
  double ff =   alpha*ut + strain_phi;

  strain[0] = (basis[2].dudx-basis[2].duolddx)/dt_- ff;
  strain[1] = (basis[3].dudy-basis[3].duolddy)/dt_- ff;
  strain[2] = (basis[2].dudy-basis[2].duolddy + basis[3].dudx-basis[3].duolddx)/dt_;// - alpha*ut - strain_phi;

  stress[0] = c1*strain[0] + c2*strain[1];
  // stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

  double divgradu = (stress[0]*dtestdx + stress[2]*dtestdy)/E;//(grad u,grad phi)
 
  //std::cout<<"residual_liniso_x_test_"<<std::endl;
 
  return divgradu;
}
// double residual_liniso_y_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_liniso_y_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double strain[3], stress[3];//x,y,z,yx,zy,zx


  //derivatives of the test function
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;
  double test = basis[0].phi[i];
  //u, phi
  //double u = basis[0].uu;

  //strain[0] = basis[2].dudx;
  //strain[1] = basis[3].dudy;
  //strain[2] = basis[2].dudy + basis[3].dudx;

  //double ut = (basis[1].uu - basis[1].uuoldold)/dt_/2.;//thermal strain
  double ut = (basis[1].uu - basis[1].uuold)/dt_;//thermal strain

  double phi = basis[0].uu;
  double h = phi*phi*(1.-phi)*(1.-phi);
  //double hp = 2.*(1.-phi)*(1.-phi)*phi-2.*(1.-phi)*phi*phi;//2 (1 - x)^2 x - 2 (1 - x) x^2
  double strain_phi = 30.*beta*h*(phi-basis[0].uuold)/dt_;
//   double strain_phi = 0.*2.*30.*beta*(c1+c2)*(hp*(phi-basis[0].uuold)/dt_*basis[0].dudy
// 					+h*(basis[0].dudy-basis[0].duolddy)/dt_
// 					)*test;

  double ff =   alpha*ut + strain_phi;

  strain[0] = (basis[2].dudx-basis[2].duolddx)/dt_- ff;
  strain[1] = (basis[3].dudy-basis[3].duolddy)/dt_- ff;
  strain[2] = (basis[2].dudy-basis[2].duolddy + basis[3].dudx-basis[3].duolddx)/dt_;// - alpha*ut - strain_phi;

  //stress[0] = c1*strain[0] + c2*strain[1];
  stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

  double divgradu = (stress[1]*dtestdy + stress[2]*dtestdx)/E;//(grad u,grad phi)
  
  //std::cout<<"residual_liniso_y_test_"<<std::endl;

  return divgradu;
}
// double residual_stress_x_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_stress_x_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double phi = basis[0].uu;
  double h = phi*phi*(1.-phi)*(1.-phi);
  double phit = (phi - basis[0].uuold)/dt_;

  double ut = (basis[1].uu - basis[1].uuold)/dt_;

  double strain[2]; 
  double stress = (basis[4].uu - basis[4].uuold)/dt_;//x,y,yx

  //test function
  double test = basis[0].phi[i];

  strain[0] = (basis[2].dudx-basis[2].duolddx)/dt_- alpha*ut - 30.*h*phit;
  strain[1] = (basis[3].dudy-basis[3].duolddy)/dt_- alpha*ut - 30.*h*phit;

  double sx = c1*strain[0] + c2*strain[1];

  //std::cout<<strain[0]<<" "<<strain[1]<<" "<<sx<<" "<<stress - sx<<" "<<(stress - sx)*test<<" "<<(stress - sx)*test/E<<std::endl;

  return (stress - sx)*test*dt_/E;
}
// double residual_stress_y_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_stress_y_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double phi = basis[0].uu;
  double h = phi*phi*(1.-phi)*(1.-phi);
  double phit = (phi - basis[0].uuold)/dt_;

  double ut = (basis[1].uu - basis[1].uuold)/dt_;

  double strain[2]; 
  double stress = (basis[5].uu - basis[5].uuold)/dt_;//x,y,yx

  //test function
  double test = basis[0].phi[i];

  strain[0] = (basis[2].dudx-basis[2].duolddx)/dt_- alpha*ut - 30.*h*phit;
  strain[1] = (basis[3].dudy-basis[3].duolddy)/dt_- alpha*ut - 30.*h*phit;

  double sy = c2*strain[0] + c1*strain[1];

  return (stress - sy)*test*dt_/E;//(grad u,grad phi)
}
// double residual_stress_xy_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_stress_xy_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  //double strain[2]; 
  double stress = (basis[6].uu - basis[6].uuold)/dt_;

  double test = basis[0].phi[i];

  //strain[0] = basis[0].dudx;
  //strain[1] = basis[1].dudy;
  double strain = (basis[2].dudy-basis[2].duolddy + basis[3].dudx-basis[3].duolddx)/dt_;

  //stress[0] = c1*strain[0] + c2*strain[1];
  //stress[1] = c2*strain[0] + c1*strain[1];
  double sxy = c3*strain;

  return (stress - sxy)*test*dt_/E;//(grad u,grad phi)
}
  //double prec_phase_(const boost::ptr_vector<Basis> &basis, 
  //		 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_phase_)
{
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

  double test = basis[0].phi[i];
  
  double phit = m*(basis[0].phi[j])/dt_*test;
  double divgrad = a*(dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz);

  return (phit + t_theta_*divgrad)/m;
}
  //double prec_heat_(const boost::ptr_vector<Basis> &basis, 
  //		 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_heat_)
{
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
  double test = basis[0].phi[i];

  double stress = test*alpha*basis[1].phi[j]*(residual_stress_x_dt_(basis, 
						      i, dt_, t_theta_,
								    0.,0)
				+residual_stress_y_dt_(basis, 
						       i, dt_, t_theta_,
						       0.,0));
  double divgrad = k*(dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz);
  double u_t =rho*c*basis[1].phi[j]/dt_*test;
 
  return (u_t + t_theta_*divgrad + stress)/rho/c;
}
  //double prec_liniso_x_test_(const boost::ptr_vector<Basis> &basis, 
  //		 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_liniso_x_test_)
{

  //cn probably want to move each of these operations inside of getbasis
  //derivatives of the test function
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;

  double dbasisdx = basis[0].dphidxi[j]*basis[0].dxidx
    +basis[0].dphideta[j]*basis[0].detadx
    +basis[0].dphidzta[j]*basis[0].dztadx;
  double dbasisdy = basis[0].dphidxi[j]*basis[0].dxidy
    +basis[0].dphideta[j]*basis[0].detady
    +basis[0].dphidzta[j]*basis[0].dztady;

  double strain[3], stress[3];//x,y,z,yx,zy,zx

  strain[0] = dbasisdx;
  strain[1] = dbasisdy;
  strain[2] = (dbasisdy + dbasisdx);

  stress[0] = c1*strain[0] + c2*strain[1];
  //stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

  double divgradu = (stress[0]*dtestdx + stress[2]*dtestdy)/E/dt_;//(grad u,grad phi)
  //double divgradu = (stress[0]*dtestdx + stress[2]*dtestdy)/E;//(grad u,grad phi)
  
  return divgradu;
}
  //double prec_liniso_y_test_(const boost::ptr_vector<Basis> &basis, 
  //		 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_liniso_y_test_)
{

  //cn probably want to move each of these operations inside of getbasis
  //derivatives of the test function
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;

  double dbasisdx = basis[0].dphidxi[j]*basis[0].dxidx
    +basis[0].dphideta[j]*basis[0].detadx
    +basis[0].dphidzta[j]*basis[0].dztadx;
  double dbasisdy = basis[0].dphidxi[j]*basis[0].dxidy
    +basis[0].dphideta[j]*basis[0].detady
    +basis[0].dphidzta[j]*basis[0].dztady;

  double strain[3], stress[3];//x,y,z,yx,zy,zx

  strain[0] = dbasisdx;
  strain[1] = dbasisdy;
  strain[2] = (dbasisdy + dbasisdx);

  //stress[0] = c1*strain[0] + c2*strain[1];
  stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];


  double divgradu = (stress[1]*dtestdy + stress[2]*dtestdx)/E/dt_;//(grad u,grad phi)
  //double divgradu = (stress[1]*dtestdy + stress[2]*dtestdx)/E;//(grad u,grad phi)

  //std::cout<<divgradu<<std::endl;
  return divgradu;
}
  //double prec_stress_test_(const boost::ptr_vector<Basis> &basis, 
  //		 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_stress_test_)
{
  double test = basis[0].phi[i];

  return test * basis[0].phi[j]/dt_*dt_/E;
}
//double postproc_stress_x_(const double *u, const double *gradu, const double *xyz, const double &time)
PPR_FUNC(postproc_stress_x_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...


  double strain[2];//x,y,z,yx,zy,zx
  double phi = u[0];
  if(phi < 0.) phi = 0.;
  if(phi > 1.) phi = 1.;
  double h = phi*phi*(1.-phi)*(1.-phi);
//   strain[0] = gradu[0] - alpha*u[1] - 30.*beta*h*phi;//var 0 dx
//   strain[1] = gradu[3] - alpha*u[1] - 30.*beta*h*phi;//var 1 dy
  strain[0] = gradu[0] - alpha*u[1];// - 30.*beta*h*phi;//var 0 dx
  strain[1] = gradu[4] - alpha*u[1];// - 30.*beta*h*phi;//var 1 dy

  return c1*strain[0] + c2*strain[1];
}
//double postproc_stress_xd_(const double *u, const double *gradu, const double *xyz, const double &time)
PPR_FUNC(postproc_stress_xd_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...


  double strain[2];//x,y,z,yx,zy,zx
//   double phi = u[0];
//   if(phi < 0.) phi = 0.;
//   if(phi > 1.) phi = 1.;
//   double h = phi*phi*(1.-phi)*(1.-phi);
//   strain[0] = gradu[0] - alpha*u[1] - 30.*beta*h*phi;//var 0 dx
//   strain[1] = gradu[3] - alpha*u[1] - 30.*beta*h*phi;//var 1 dy
  strain[0] = gradu[0];//var 0 dx
  strain[1] = gradu[4];//var 1 dy

  return c1*strain[0] + c2*strain[1];
}
//double postproc_stress_y_(const double *u, const double *gradu, const double *xyz, const double &time)
PPR_FUNC(postproc_stress_y_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...


  double strain[2];//x,y,z,yx,zy,zx
  double phi = u[0];
  if(phi < 0.) phi = 0.;
  if(phi > 1.) phi = 1.;
  double h = phi*phi*(1.-phi)*(1.-phi);
//   strain[0] = gradu[0] - alpha*u[1] - 30.*beta*h*phi;//var 0 dx
//   strain[1] = gradu[3] - alpha*u[1] - 30.*beta*h*phi;//var 1 dy
  strain[0] = gradu[0] - alpha*u[1] - 30.*beta*h*phi;//var 0 dx
  strain[1] = gradu[4] - alpha*u[1] - 30.*beta*h*phi;//var 1 dy

  return c2*strain[0] + c1*strain[1];
}
//double postproc_stress_xy_(const double *u, const double *gradu, const double *xyz, const double &time)
PPR_FUNC(postproc_stress_xy_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...


  double phi = u[0];
  if(phi < 0.) phi = 0.;
  if(phi > 1.) phi = 1.;
  double h = phi*phi*(1.-phi)*(1.-phi);

  double strain = gradu[1] + gradu[3] - alpha*u[1] - 30.*beta*h*phi;

  return c3*strain;
}
//double postproc_stress_eq_(const double *u, const double *gradu, const double *xyz, const double &time)
PPR_FUNC(postproc_stress_eq_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...


  double strain[3], stress[3];//x,y,z,yx,zy,zx
  double phi = u[0];
  if(phi < 0.) phi = 0.;
  if(phi > 1.) phi = 1.;
  double h = phi*phi*(1.-phi)*(1.-phi);

  strain[0] = gradu[0] - alpha*u[1] - 30.*beta*h*phi;//var 0 dx
  strain[1] = gradu[4] - alpha*u[1] - 30.*beta*h*phi;//var 1 dy
  strain[2] = gradu[1] + gradu[3];// - alpha*u[1] - 30.*beta*h*phi;

  stress[0] = c1*strain[0] + c2*strain[1];
  stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

//   return sqrt(((stress[0]-stress[1])*(stress[0]-stress[1])
// 	       + stress[0]*stress[0]
// 	       + stress[1]*stress[1]
// 	       + 6. *stress[2]*stress[2]
// 	       )/2.);
  return sqrt((stress[0]-stress[1])*(stress[0]-stress[1])
	       + 3.*stress[2]*stress[2]
	       );
}
//double postproc_stress_eqd_(const double *u, const double *gradu, const double *xyz, const double &time)
PPR_FUNC(postproc_stress_eqd_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...


  double strain[3], stress[3];//x,y,z,yx,zy,zx
  double phi = u[0];
  if(phi < 0.) phi = 0.;
  if(phi > 1.) phi = 1.;
  double h = phi*phi*(1.-phi)*(1.-phi);

  strain[0] = gradu[0];// - alpha*u[1] - 30.*beta*h*phi;//var 0 dx
  strain[1] = gradu[4];// - alpha*u[1] - 30.*beta*h*phi;//var 1 dy
  strain[2] = gradu[1] + gradu[3];// + gradu[2];// - alpha*u[1] - 30.*beta*h*phi;

  stress[0] = c1*strain[0] + c2*strain[1];
  stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

//   return sqrt(((stress[0]-stress[1])*(stress[0]-stress[1])
// 	       + stress[0]*stress[0]
// 	       + stress[1]*stress[1]
// 	       + 6. *stress[2]*stress[2]
// 	       )/2.);

  return sqrt((stress[0]-stress[1])*(stress[0]-stress[1])
	       + 3.*stress[2]*stress[2]
	       );
//   return (stress[0]+stress[1])/2.
//     +sqrt((stress[0]-stress[1])*(stress[0]-stress[1])/4.+stress[3]*stress[3]);
}
//double postproc_phi_(const double *u, const double *gradu, const double *xyz, const double &time)
PPR_FUNC(postproc_phi_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...


  double phi = u[0];
  if(phi < 0.) phi = 0.;
  if(phi > 1.) phi = 1.;
  return phi;
}
//double postproc_strain_(const double *u, const double *gradu, const double *xyz, const double &time)
PPR_FUNC(postproc_strain_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...


  double phi = u[0];
  double uu = u[1];
//   if(phi < 0.) phi = 0.;
//   if(phi > 1.) phi = 1.;
  double h = phi*phi*(1.-phi)*(1.-phi);
  return alpha*uu;// + 30.*beta*h*phi;
}
}//namespace uehara


namespace uehara2
{
  //double init_phase_c_(const double &x,
  //	   const double &y,
  //	   const double &z)
INI_FUNC(init_phase_c_)
{
  double phi_sol_ = 1.;
  double phi_liq_ = 0.;

  double val = phi_sol_ ;  

  double r0 = uehara::r0;
  double dx = 1.e-2;
  double x0 = 15.*dx;
  double r = .9*r0*dx/2.;

  double rr = sqrt((x-x0)*(x-x0)+(y-x0)*(y-x0));

  val=phi_liq_;

  if( rr > r0*sqrt(2.)*dx/2. ){
    val=phi_sol_;
  }
  else {
    val=phi_liq_;
  }

  return val;
}

  //double init_heat_(const double &x,
  //	   const double &y,
  //	   const double &z)
INI_FUNC(init_heat_)
{
  double val = 300.;  

  return val;
}
// double residual_heat_(const boost::ptr_vector<Basis> &basis, 
// 		      const int &i, 
// 		      const double &dt_, 
// 		      const double &t_theta_, 
// 		      const double &delta, 
// 		      const double &time)
RES_FUNC(residual_heat_)
{
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
  //test function
  double test = basis[1].phi[i];
  //u, phi
  double phi = basis[0].uu;
  double phiold = basis[0].uuold;
  double phioldold = basis[0].uuoldold;
  double u = basis[1].uu;
  double uold = basis[1].uuold;

  double dudx = basis[1].dudx;
  double dudy = basis[1].dudy;
  //double dudz = basis[1].dudz;

  double ut = uehara::rho*uehara::c*(u-uold)/dt_*test;
  double divgradu = uehara::k*(dudx*dtestdx + dudy*dtestdy);
  double h = phi*phi*(1.-phi)*(1.-phi);
  //double phitu = -30.*1e12*uehara::L*h*(phi-phioldold)/2./dt_*test; 
  double phitu = -30.*uehara::L*h*(phi-phiold)/dt_*test; 
  
  //thermal term
  double stress = test*uehara::alpha*u*(uehara::residual_stress_x_dt_(basis, 
						      i, dt_, t_theta_, 
						      time, eqn_id)
				+uehara::residual_stress_y_dt_(basis, 
						       i, dt_, t_theta_,
						       time, eqn_id));
  

  double rhs = divgradu + phitu + stress;

  return (ut + rhs)/uehara::rho/uehara::c;

}
}//namespace uehara2








namespace coupledstress
{
  double E = 1e6;
  double nu = .3;

  //plane stress
//   double c = E/(1.-nu*nu);
//   double c1 = c;
//   double c2 = c*nu;
//   double c3 = c*(1.-nu)/2.;//always confused about the 2 here
  //plane strain
  double c = E/(1.+nu)/(1.-2.*nu);
  double c1 = c*(1.-nu);
  double c2 = c*nu;
  double c3 = c*(1.-2.*nu)/2.;

// double residual_liniso_x_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_liniso_x_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double strain[3], stress[3];//x,y,yx


  //derivatives of the test function
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;
  //test function
  //double test = basis[0].phi[i];
  //u, phi
  //double u = basis[0].uu;

  strain[0] = basis[0].dudx;
  strain[1] = basis[1].dudy;
  strain[2] = basis[0].dudy + basis[1].dudx;

  stress[0] = c1*strain[0] + c2*strain[1];
  // stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

  double divgradu = stress[0]*dtestdx + stress[2]*dtestdy;//(grad u,grad phi)
 
  //std::cout<<"residual_liniso_x_test_"<<std::endl;
 
  return divgradu;
}
// double residual_liniso_y_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_liniso_y_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double strain[3], stress[3];//x,y,z,yx,zy,zx


  //derivatives of the test function
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;
  //double test = basis[0].phi[i];
  //u, phi
  //double u = basis[0].uu;

  strain[0] = basis[0].dudx;
  strain[1] = basis[1].dudy;
  strain[2] = basis[0].dudy + basis[1].dudx;

  stress[0] = c1*strain[0] + c2*strain[1];
  stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

  double divgradu = stress[1]*dtestdy + stress[2]*dtestdx;//(grad u,grad phi)
  
  //std::cout<<"residual_liniso_y_test_"<<std::endl;

  return divgradu;
}
// double residual_stress_x_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_stress_x_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double strain[3], stress[3];//x,y,yx

  //test function
  double test = basis[0].phi[i];
  //u, phi
  double sx = basis[2].uu;

  strain[0] = basis[0].dudx;
  strain[1] = basis[1].dudy;
  //strain[2] = basis[0].dudy + basis[1].dudx;

  stress[0] = c1*strain[0] + c2*strain[1];
  // stress[1] = c2*strain[0] + c1*strain[1];
  //stress[2] = c3*strain[2];

  return (sx - stress[0])*test;
}
// double residual_stress_y_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_stress_y_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double strain[3], stress[3];//x,y,z,yx,zy,zx

  double test = basis[0].phi[i];
  //u, phi
  double sy = basis[3].uu;

  strain[0] = basis[0].dudx;
  strain[1] = basis[1].dudy;
  //strain[2] = basis[0].dudy + basis[1].dudx;

  //stress[0] = c1*strain[0] + c2*strain[1];
  stress[1] = c2*strain[0] + c1*strain[1];
  //stress[2] = c3*strain[2];

  return (sy - stress[1])*test;//(grad u,grad phi)
}
// double residual_stress_xy_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_stress_xy_test_)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double strain[3], stress[3];//x,y,z,yx,zy,zx

  double test = basis[0].phi[i];
  //u, phi
  double sxy = basis[4].uu;

  //strain[0] = basis[0].dudx;
  //strain[1] = basis[1].dudy;
  strain[2] = basis[0].dudy + basis[1].dudx;

  //stress[0] = c1*strain[0] + c2*strain[1];
  //stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

  return (sxy - stress[2])*test;//(grad u,grad phi)
}
  //double prec_liniso_x_test_(const boost::ptr_vector<Basis> &basis, 
  //		 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_liniso_x_test_)
{

  //cn probably want to move each of these operations inside of getbasis
  //derivatives of the test function
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;

  double dbasisdx = basis[0].dphidxi[j]*basis[0].dxidx
    +basis[0].dphideta[j]*basis[0].detadx
    +basis[0].dphidzta[j]*basis[0].dztadx;
  double dbasisdy = basis[0].dphidxi[j]*basis[0].dxidy
    +basis[0].dphideta[j]*basis[0].detady
    +basis[0].dphidzta[j]*basis[0].dztady;

  double strain[3], stress[3];//x,y,z,yx,zy,zx

  strain[0] = dbasisdx;
  strain[1] = dbasisdy;
  strain[2] = dbasisdy + dbasisdx;

  stress[0] = c1*strain[0] + c2*strain[1];
  //stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

  double divgradu = stress[0]*dtestdx + stress[2]*dtestdy;//(grad u,grad phi)
 
  return divgradu;
}
  //double prec_liniso_y_test_(const boost::ptr_vector<Basis> &basis, 
  //		 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_liniso_y_test_)
{

  //cn probably want to move each of these operations inside of getbasis
  //derivatives of the test function
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;

  double dbasisdx = basis[0].dphidxi[j]*basis[0].dxidx
    +basis[0].dphideta[j]*basis[0].detadx
    +basis[0].dphidzta[j]*basis[0].dztadx;
  double dbasisdy = basis[0].dphidxi[j]*basis[0].dxidy
    +basis[0].dphideta[j]*basis[0].detady
    +basis[0].dphidzta[j]*basis[0].dztady;

  double strain[3], stress[3];//x,y,z,yx,zy,zx

  strain[0] = dbasisdx;
  strain[1] = dbasisdy;
  strain[2] = dbasisdy + dbasisdx;

  //stress[0] = c1*strain[0] + c2*strain[1];
  stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];


  double divgradu = stress[1]*dtestdy + stress[2]*dtestdx;//(grad u,grad phi)

  return divgradu;
}
  //double prec_stress_test_(const boost::ptr_vector<Basis> &basis, 
  //		 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_stress_test_)
{
  double test = basis[0].phi[i];

  return test * basis[0].phi[j];
}
//double postproc_stress_x_(const double *u, const double *gradu, const double *xyz, const double &time)
PPR_FUNC(postproc_stress_x_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...



  double strain[2];//x,y,z,yx,zy,zx
  strain[0] = gradu[0];//var 0 dx
  strain[1] = gradu[4];//var 1 dy

  return c1*strain[0] + c2*strain[1];
}
//double postproc_stress_y_(const double *u, const double *gradu, const double *xyz, const double &time)
PPR_FUNC(postproc_stress_y_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...

  double strain[2];//x,y,z,yx,zy,zx
  strain[0] = gradu[0];//var 0 dx
  strain[1] = gradu[4];//var 1 dy

  return c2*strain[0] + c1*strain[1];
}
//double postproc_stress_xy_(const double *u, const double *gradu, const double *xyz, const double &time)
PPR_FUNC(postproc_stress_xy_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...


  double strain = gradu[1] + gradu[3];

  return c3*strain;
}

}//namespace coupledstress

namespace laplace
{
// double residual_heat_test_(const boost::ptr_vector<Basis> &basis, 
// 			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
// 		      const double &time)
RES_FUNC(residual_heat_test_)
{

  //u[x,y,t]=exp(-2 pi^2 t)sin(pi x)sin(pi y)
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
  //test function
  double test = basis[0].phi[i];
  //u, phi
  //double u = basis[0].uu;
  //double uold = basis[0].uuold;

  //double ut = (u-uold)/dt_*test;
  double divgradu = (basis[0].dudx*dtestdx + basis[0].dudy*dtestdy + basis[0].dudz*dtestdz);//(grad u,grad phi)
  //double divgradu_old = (basis[0].duolddx*dtestdx + basis[0].duolddy*dtestdy + basis[0].duolddz*dtestdz);//(grad u,grad phi)
 
 
  return divgradu - 8.*test;
}
}//namespace laplace

namespace cahnhilliard
{
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

RES_FUNC(residual_c_)
{
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
  //test function
  double test = basis[0].phi[i];
  double c = basis[0].uu;
  double cold = basis[0].uuold;
  double mu = basis[1].uu;
  double x = basis[0].xx;

  double ct = (c - cold)/dt_*test;
  double divgradmu = M*t_theta_*(basis[1].dudx*dtestdx + basis[1].dudy*dtestdy + basis[1].dudz*dtestdz)
    + M*(1.-t_theta_)*(basis[1].duolddx*dtestdx + basis[1].duolddy*dtestdy + basis[1].duolddz*dtestdz);
  double f = t_theta_*fcoef_*F(x,time)*test + (1.-t_theta_)*fcoef_*F(x,time-dt_)*test;

  return ct + divgradmu - f;
}

RES_FUNC(residual_mu_)
{
  //derivatives of the test function
  double dtestdx = basis[1].dphidxi[i]*basis[1].dxidx
    +basis[1].dphideta[i]*basis[1].detadx
    +basis[1].dphidzta[i]*basis[1].dztadx;
  double dtestdy = basis[1].dphidxi[i]*basis[1].dxidy
    +basis[1].dphideta[i]*basis[1].detady
    +basis[1].dphidzta[i]*basis[1].dztady;
  double dtestdz = basis[1].dphidxi[i]*basis[1].dxidz
    +basis[1].dphideta[i]*basis[1].detadz
    +basis[1].dphidzta[i]*basis[1].dztadz;
  //test function
  double test = basis[1].phi[i];
  double c = basis[0].uu;
  double mu = basis[1].uu;

  double mut = mu*test;
  double f = fp(c)*test;
  double divgradc = Eps*(basis[0].dudx*dtestdx + basis[0].dudy*dtestdy + basis[0].dudz*dtestdz);

  return mut - f - divgradc;
}
PARAM_FUNC(param_)
{
  fcoef_ = plist->get<double>("fcoef");
}



}//namespace cahnhilliard

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

INI_FUNC(init_)
{
  //return .001*r(x,eqn_id)*r(x,N-eqn_id)*(y,eqn_id)*r(y,N-eqn_id);
  return ((rand() % 100)/50.-1.)*.001;
}
RES_FUNC(residual_)
{
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
  double test = basis[0].phi[i];

  double u = basis[eqn_id].uu;
  double uold = basis[eqn_id].uuold;

  double divgradu = kappa*(basis[eqn_id].dudx*dtestdx + basis[eqn_id].dudy*dtestdy + basis[eqn_id].dudz*dtestdz);

  double s = 0.;
  for(int k = 0; k < N; k++){
    s = s + basis[k].uu*basis[k].uu;
  }
  s = s - u*u;

  return (u-uold)/dt_*test + L* ((-alpha*u + beta*u*u*u +2.*gamma*u*s)*test +  divgradu); 

}
PRE_FUNC(prec_)
{
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

  double u = basis[eqn_id].uu;
  
  double test = basis[0].phi[i];
  double divgrad = L*kappa*(dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz);
  double u_t =test * basis[0].phi[j]/dt_;
  double alphau = -test*L*alpha*basis[0].phi[j];
  double betau = 3.*u*u*basis[0].phi[j]*test*L*beta;

  double s = 0.;
  for(int k = 0; k < N; k++){
    s = s + basis[k].uu*basis[k].uu;
  }
  s = s - u*u;

  double gammau = 2.*gamma*L*basis[0].phi[j]*s*test;

  return u_t + divgrad + betau + gammau;// + alphau ;
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

namespace periodic
{

RES_FUNC(residual_)
{
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
  double test = basis[0].phi[i];

  double u = basis[0].uu;
  double uold = basis[0].uuold;
  double x = basis[0].xx;

  double f = sin(11.*x);

  double divgradu = basis[0].dudx*dtestdx + basis[0].dudy*dtestdy + basis[0].dudz*dtestdz;

  return (u-uold)/dt_*test + divgradu - f*test; 

}


}//namespace periodic


namespace kundin
{
  //double vf = 1e5;// J/mol-at / J/m^3
  //double lf = 1.;//m/m    length conversion
  double lf = 1000.;//mm/m    length conversion
  double lf2 = lf*lf;
  double lf3 = lf2*lf;
  double W  = 0.00000007*lf;//m
  double dx = 0.000000032*lf; //m
  double lx = dx*320.; //m

  int N = 6;
  double a_1 = sqrt(2.)/3.;
  double a_2 = 1.175;
  double sigma = 0.12/lf2;//J/m^2
  double eps_4 = .05;

  //is this the correct conversion? the paper is confusing
  double x_fact = (1.e5)/lf3;// (J/mol-at/mf)   should convert J/mol-at to J/m^3
  double XA_L [6] = { 1.2*x_fact, 1.3*x_fact, .9*x_fact, 3.*x_fact, 3.*x_fact, 3.*x_fact };//J/mol-at/mf^2
  double deltaA_SLT0 [6] = {.01645, .034356, -.0098313, -.0412, -.00545,  .00005};//mf
  double CAeq_ST0 [6] =    {.20645, .219356,  .0201687,  .0098,  .00355,  .00505};//mf
  double CAeq_LT0 [6] =    {.19,    .185,     .03,       .051,   .009,    .005};//mf
  double kA_L [6] =        {3.48,   1.48,     1.48,      2.48,   1.51,    11.21};
  double mA_Sp [6] =       {9118.,  4365.8,  -15257.4,  -3640.8,-27522.9, 3700000.};//K/mf
  double mA_Lp [6] =       {2620.,  2936.,   -10309.,   -1468.7,-18131.3, 330000.};//K/mf
  double T0 = 1635.15;//K
  //double T0 = 1243.;//K
  //double T0 = 1362.;//C
  double d_fact = (1.e-10)*lf2;
  double DA_L [6] = {8.9843*d_fact, 9.0398*d_fact, 10.759*d_fact, 10.526*d_fact, 10.992*d_fact, 11.092*d_fact};//m^2/s
  double q_fact =  (1.e5)*lf;
  double QA_L [6] = {5.615*q_fact,20.3*q_fact, .967*q_fact, 68.9*q_fact, .977*q_fact, .000123*q_fact};//(J/mol-at)/(J/m^3) s/m^2
  double tau = 8.12e-8;//s


  double Tdot [4] = {28745.6, 14372.8, 7186.39,3593.2};//K/s
  //double lambda = 18.3;//J/(s m K) keep as meters
  double a = 4.7e-6 * lf2;//m^2/s
  //douple pi = 3.14159;


double df(const double p)
{
  return 2.* (1. - p)*(1. - p)*p - 2.* (1. - p)*p*p;
}
double dg(const double p)
{
  return p*p*p* (-15. + 12. *p) + 3 *p*p* (10. - 15.* p + 6 *p*p);
}
double CAeq_L(const int i,const double T){
  return CAeq_LT0[i] + (T-T0)/mA_Lp[i];
}
double CAeq_S(const int i,const double T){
  return CAeq_ST0[i] + (T-T0)/mA_Sp[i];
}
double deltaA_SL(const int i,const double T){
  double k_S = 1./kA_L[i];
  //return deltaA_SLT0[i] + (T0-T)*(1.-k_S)/mA_Lp[i];
  return CAeq_S(i,T) - CAeq_L(i,T);
}
  double temp(const double time, const double z){
  double G_z = Tdot[0]/2./a*z;//K/m(mm)
  //return T0 - Tdot[0]*time + G_z*z;
  //return T0 +5.*(z- .1*time);
  //return T0 - 1.e2*time+5.*z;
  return  T0 - 150.;
}
double deltaG_ch(const double * CA, const double p, const double T)
{

  double s = 0.;
  for (int i = 0; i < N; i++){
    double CAeq = CAeq_S(i,T)*p + CAeq_L(i,T)*(1. - p);
    double d = (p+(1.-p)*kA_L[i]);
    s += XA_L[i]*deltaA_SL(i,T)*(CA[i]-CAeq)/d;
  }
  return s;
}
double fn (const double p){
  return 16.*p*p*(1.-p)*(1.-p);
}
double XA_bar(const int i, const double p){
  //this should be evaluated at p = .5
  double XA_S = kA_L[i]*XA_L[i];
  //double val = (1.-p)/XA_L[i] + p /XA_S;
  double val = (1.-.5)/XA_L[i] + .5 /XA_S;
  return 1./val;
}
double QA_L_(const int i,const double p,const double T){
  return XA_bar(i,p)*deltaA_SL(i,T)*deltaA_SL(i,T)/DA_L[i];
}
double tau_(const double p,const double T){
  double t =0.;  
  for (int i = 0; i < N; i++){
    t += QA_L_(i,p,T);
  }
  t = t*a_1*a_2*W*W*W/sigma;
  return t;
}
  //double a_small =  5.e-3;//   => px < 1e-9
double a_small =  5.e-9;//   => px < 1e-9
inline const double a_s_(const double px, const double py, const double pz, const double ep){
    // in 2d, equivalent to: a_s = 1 + ep cos 4 theta
    double px2 = px*px;
    double py2 = py*py;
    double pz2 = pz*pz;
    double norm4 = (px2 + py2 + pz2 )*(px2 + py2 + pz2 );
    //double small = 5.e-3;//   => px < 1e-9
    if( norm4 < a_small ) return 1. + ep;
    //return 1.-3.*ep + 4.*ep*(px2*px2 + py2*py2 + pz2*pz2)/norm4;
    return 1.;
  }
inline const double da_s_dpx(const double px, const double py, const double pz, const double ep){
    // (16 ep x (-y^4 - z^4 + x^2 y^2 + x^2 z^2))/(x^2 + y^2 + z^2)^3
    double px2 = px*px;
    double py2 = py*py;
    double pz2 = pz*pz;
    double norm4 = (px2 + py2 + pz2 )*(px2 + py2 + pz2 );
    double norm6 = norm4*(px2 + py2 + pz2 );
    //double small = 5.e-3;//   => px < 1e-9
    if( norm4 < a_small ) return 0.;
    //return 16.*ep*px*(- py2*py2 - pz2*pz2 + px2*py2 + px2*pz2)/norm6;
    return 0.;
  }
inline const double da_s_dpy(const double px, const double py, const double pz, const double ep){
    // -((16 ep y (x^4 - x^2 y^2 - y^2 z^2 + z^4))/(x^2 + y^2 + z^2)^3)
    double px2 = px*px;
    double py2 = py*py;
    double pz2 = pz*pz;
    double norm4 = (px2 + py2 + pz2 )*(px2 + py2 + pz2 );
    double norm6 = norm4*(px2 + py2 + pz2 );
    //double small = 5.e-3;//   => px < 1e-9
    if( norm4 < a_small ) return 0.;
    //return - 16.*ep*py*(px2*px2 + pz2*pz2 - px2*py2 - py2*pz2)/norm6;
    return 0.;
  }
inline const double da_s_dpz(const double px, const double py, const double pz, const double ep){
    //-((16 ep z (x^4 + y^4 - x^2 z^2 - y^2 z^2))/(x^2 + y^2 + z^2)^3)
    double px2 = px*px;
    double py2 = py*py;
    double pz2 = pz*pz;
    double norm4 = (px2 + py2 + pz2 )*(px2 + py2 + pz2 );
    double norm6 = norm4*(px2 + py2 + pz2 );
    //double small = 5.e-3;//   => px < 1e-9
    if( norm4 < a_small ) return 0.;
    //return - 16.*ep*pz*(px2*px2 + py2*py2 - px2*pz2 - py2*pz2)/norm6;
    return 0.;
  }
  
  
RES_FUNC(phiresidual_)
{
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
  double test = basis[0].phi[i];

  int phi_id = eqn_id;
  double phi[2] = {basis[phi_id].uu, basis[phi_id].uuold};
  double phi_x[2] = {basis[phi_id].dudx, basis[phi_id].duolddx};
  double phi_y[2] = {basis[phi_id].dudy, basis[phi_id].duolddy};
  double phi_z[2] = {basis[phi_id].dudz, basis[phi_id].duolddz};

  double y = basis[phi_id].yy;

  double a_s[2] = {a_s_(phi_x[0], phi_y[0], phi_z[0], eps_4), 
		   a_s_(phi_x[1], phi_y[1], phi_z[1], eps_4)};
  //a_s[0]=1;a_s[1]=1;
  double a_spx[2] = {da_s_dpx(phi_x[0], phi_y[0], phi_z[0], eps_4), 
		     da_s_dpx(phi_x[1], phi_y[1], phi_z[1], eps_4)};
  double a_spy[2] = {da_s_dpy(phi_x[0], phi_y[0], phi_z[0], eps_4), 
		     da_s_dpy(phi_x[1], phi_y[1], phi_z[1], eps_4)};
  double a_spz[2] = {da_s_dpz(phi_x[0], phi_y[0], phi_z[0], eps_4), 
		     da_s_dpz(phi_x[1], phi_y[1], phi_z[1], eps_4)};

  tau = tau_(phi[0],temp(time,y));
  tau=2.e-5;
  //tau=tau/1000.;
  double phit = (tau*phi[0]-tau*phi[1])/dt_*test;

  double divgrad = t_theta_*W*W*a_s[0]*a_s[0]*(phi_x[0]*dtestdx + phi_y[0]*dtestdy + phi_z[0]*dtestdz)
             +(1.-t_theta_)*W*W*a_s[1]*a_s[1]*(phi_x[1]*dtestdx + phi_y[1]*dtestdy + phi_z[1]*dtestdz);
 
  double normphi2[2] = {phi_x[0]*phi_x[0] + phi_y[0]*phi_y[0] + phi_z[0]*phi_z[0],
			phi_x[1]*phi_x[1] + phi_y[1]*phi_y[1] + phi_z[1]*phi_z[1]};

  double curlgrad = t_theta_*W*W*normphi2[0]*a_s[0]*(a_spx[0]*dtestdx + a_spy[0]*dtestdy + a_spz[0]*dtestdz)
              +(1.-t_theta_)*W*W*normphi2[1]*a_s[1]*(a_spx[1]*dtestdx + a_spy[1]*dtestdy + a_spz[1]*dtestdz);

  double dfdp = (t_theta_*df(phi[0])+(1.-t_theta_)*df(phi[1]))*test;

  double CA[6];
  double deltaG_ch_[2];
  for (int i = 0; i < N; i++) CA[i] =  basis[i].uu;
  double T = temp(time,y);
  deltaG_ch_[0] = deltaG_ch(CA,phi[0],T);

  for (int i = 0; i < N; i++) CA[i] =  basis[i].uuold;
  T = temp(time - dt_,y);
  deltaG_ch_[1] = deltaG_ch(CA,phi[1],T);

  double dgdp = -a_1*W/sigma*(t_theta_*dg(phi[0])*deltaG_ch_[0]
			+(1.-t_theta_)*dg(phi[1])*deltaG_ch_[1])*test;
//   if(phi[1] >0 && phi[1]<1)
//     std::cout<<dgdp<<" "<<dfdp<<" "<<phi[1]<<std::endl;

  return (phit + divgrad + curlgrad + dfdp + 10.*dgdp)/tau;// /tau/10.;

}
PRE_FUNC(phiprec_)
{
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
  double test = basis[0].phi[i];

  int phi_id = eqn_id;
  double phi =  basis[phi_id].uu;
  double phi_x = basis[phi_id].dudx;
  double phi_y = basis[phi_id].dudy;
  double phi_z = basis[phi_id].dudz;

  //tau = tau_(phi,temp(0.));
  tau = 2.e-5;
  double phit = tau*basis[phi_id].phi[j]/dt_*test;
  double a_s = a_s_(phi_x, phi_y, phi_z, eps_4);
  double divgrad = t_theta_*W*W*a_s*a_s*(dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz);

  return (phit + t_theta_*divgrad)/tau;// /tau/10.;
}
RES_FUNC(cresidual_)
{
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
  double test = basis[0].phi[i];

  double y = basis[eqn_id].yy;

  double CA[2] = {basis[eqn_id].uu, 
		  basis[eqn_id].uuold};
  double CA_x[2] = {basis[eqn_id].dudx,
		    basis[eqn_id].duolddx};
  double CA_y[2] = {basis[eqn_id].dudy,
		    basis[eqn_id].duolddy};
  double CA_z[2] = {basis[eqn_id].dudz,
		    basis[eqn_id].duolddz};
  double phi[2] = {basis[6].uu,
		   basis[6].uuold};
  double phi_x[2] = {basis[6].dudx,
		     basis[6].duolddx};
  double phi_y[2] = {basis[6].dudy,
		     basis[6].duolddy};
  double phi_z[2] = {basis[6].dudz,
		     basis[6].duolddz}; 

  double T[2] = {temp(time,y),
		 temp(time-dt_,y)};
  double D_L = DA_L[eqn_id];
  //double D_S = 0.;
  double D_S = D_L;
  double k_L =kA_L[eqn_id];

  double CAt = (CA[0] - CA[1])/dt_*test;

//   double CAeq[2] = {CAeq_S(eqn_id,T)*phi[0] + CAeq_L(eqn_id,T)*(1.- phi[0]),
// 		    CAeq_S(eqn_id,T)*phi[1] + CAeq_L(eqn_id,T)*(1.- phi[1])};
  
  double CAeq_x[2] = {CAeq_S(eqn_id,T[0])*phi_x[0] + CAeq_L(eqn_id,T[0])*(- phi_x[0]),
		      CAeq_S(eqn_id,T[1])*phi_x[1] + CAeq_L(eqn_id,T[1])*(- phi_x[1])};
  double CAeq_y[2] = {CAeq_S(eqn_id,T[0])*phi_y[0] + CAeq_L(eqn_id,T[0])*(- phi_y[0]),
		      CAeq_S(eqn_id,T[1])*phi_y[1] + CAeq_L(eqn_id,T[1])*(- phi_y[1])};
  double CAeq_z[2] = {CAeq_S(eqn_id,T[0])*phi_z[0] + CAeq_L(eqn_id,T[0])*(- phi_z[0]),
		      CAeq_S(eqn_id,T[1])*phi_z[1] + CAeq_L(eqn_id,T[1])*(- phi_z[1])};

  double CA_xd[2] = {CA_x[0] - CAeq_x[0],
		     CA_x[1] - CAeq_x[1]};
  double CA_yd[2] = {CA_y[0] - CAeq_y[0],
		     CA_y[1] - CAeq_y[1]};
  double CA_zd[2] = {CA_z[0] - CAeq_z[0],
		     CA_z[1] - CAeq_z[1]};

  //the denominator *should* be inside the gradient operator
  //however, most implementations factor it out like this
  double coef[2] = {(D_S*phi[0] + D_L*(1.-phi[0])*k_L)/(phi[0] + (1. - phi[0])*k_L),
		    (D_S*phi[1] + D_L*(1.-phi[1])*k_L)/(phi[1] + (1. - phi[1])*k_L)};
//   double coef[2] = {D_L,
// 		    D_L};
  double divgrad = t_theta_*coef[0]*(CA_xd[0]*dtestdx + CA_yd[0]*dtestdy + CA_zd[0]*dtestdz)
    +(1.-t_theta_)*coef[1]*(CA_xd[1]*dtestdx + CA_yd[1]*dtestdy + CA_zd[1]*dtestdz);

  double delta_SL[2] = {deltaA_SL(eqn_id,T[0]),
			deltaA_SL(eqn_id,T[1])};
  double normphi[2] = {sqrt(phi_x[0]*phi_x[0] + phi_y[0]*phi_y[0] + phi_z[0]*phi_z[0]),
		       sqrt(phi_x[1]*phi_x[1] + phi_y[1]*phi_y[1] + phi_z[0]*phi_z[1])};
  double small = 1.e-12;
  double coef2[2] = {0.,0.};
  if (normphi[0] > small) coef2[0] = W/sqrt(2.)*delta_SL[0]*(phi[0] - phi[1])/dt_/normphi[0];
  if (normphi[1] > small) coef2[1] = W/sqrt(2.)*delta_SL[1]*(phi[1] - basis[6].uuoldold)/dt_/normphi[1];
  double trap = t_theta_*coef2[0]*(phi_x[0]*dtestdx + phi_y[0]*dtestdy + phi_z[0]*dtestdz)
       + (1. - t_theta_)*coef2[1]*(phi_x[1]*dtestdx + phi_y[1]*dtestdy + phi_z[1]*dtestdz);

  //std::cout<<CAt<<" "<<divgrad<<"     :   "
  //std::cout<<CA<<" "<<CAold<<" "<<CA-CAold<<" "<<eqn_id<<" "<<basis[eqn_id].uu<<std::endl;
  return (CAt + divgrad + 0.*trap);
  //return CA[0]-CAeq_ST0[eqn_id];
}
PRE_FUNC(cprec_)
{
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;
  double dtestdz = basis[0].dphidxi[i]*basis[0].dxidz
    +basis[0].dphideta[i]*basis[0].detadz
    +basis[0].dphidzta[i]*basis[0].dztadz;

  double dbasisdx = basis[eqn_id].dphidxi[j]*basis[eqn_id].dxidx
    +basis[eqn_id].dphideta[j]*basis[eqn_id].detadx
    +basis[eqn_id].dphidzta[j]*basis[eqn_id].dztadx;
  double dbasisdy = basis[eqn_id].dphidxi[j]*basis[eqn_id].dxidy
    +basis[eqn_id].dphideta[j]*basis[eqn_id].detady
    +basis[eqn_id].dphidzta[j]*basis[eqn_id].dztady;
  double dbasisdz = basis[eqn_id].dphidxi[j]*basis[eqn_id].dxidz
    +basis[eqn_id].dphideta[j]*basis[eqn_id].detadz
    +basis[eqn_id].dphidzta[j]*basis[eqn_id].dztadz;
  double test = basis[0].phi[i];

  double CAt = basis[eqn_id].phi[j]/dt_*test;
  double CA = basis[eqn_id].phi[j];
  double CA_x = dbasisdx;
  double CA_y = dbasisdy;
  double CA_z = dbasisdz;
  double phi = basis[6].uu;
  double phi_x = basis[6].dudx;
  double phi_y = basis[6].dudy;
  double phi_z = basis[6].dudz;

  //double T = temp(0.);
  double D_L = DA_L[eqn_id];
  double k_L = kA_L[eqn_id];
  double den = (phi + (1. - phi)*k_L);
  double den_x = phi_x - k_L*phi_x;
  double den_y = phi_y - k_L*phi_y;
  double den_z = phi_z - k_L*phi_z;
  double num = CA;
  double num_x = CA_x;
  double num_y = CA_y;
  double num_z = CA_z;

  double CA_xd = (num_x*den - num*den_x)/den/den;
  double CA_yd = (num_y*den - num*den_y)/den/den;
  double CA_zd = (num_z*den - num*den_z)/den/den;
  double D_S = 0.;
  //double D_S = D_L;
  double coef = (D_S*phi + D_L*(1.-phi)*k_L)/den;
  double divgrad = coef*(CA_x*dtestdx + CA_y*dtestdy + CA_z*dtestdz);

  return (CAt + t_theta_*divgrad);
}
double init(const double x)
{
  double pi = 3.141592653589793;
  double val = lx/7.*sin(7.*pi*x/lx);
  return 10.*(std::abs(val)-.0005);
}
INI_FUNC(cinit_)
{
  double val = CAeq_LT0[eqn_id];
  if(y < init(x)) val=CAeq_ST0[eqn_id];
  //std::cout<<eqn_id<<" "<<val<<std::endl;
  return val;
}
DBC_FUNC(dbc0_)
{  
  return CAeq_LT0[0];
}
DBC_FUNC(dbc1_)
{  
  return CAeq_LT0[1];
}
DBC_FUNC(dbc2_)
{  
  return CAeq_LT0[2];
}
DBC_FUNC(dbc3_)
{  
  return CAeq_LT0[3];
}
DBC_FUNC(dbc4_)
{  
  return CAeq_LT0[4];
}
DBC_FUNC(dbc5_)
{  
  return CAeq_LT0[5];
}
INI_FUNC(phiinit_)
{

  double phi_sol_ = 1.;
  double phi_liq_ = 0.;

  double val = phi_liq_;  

  if(y < init(x)) val=phi_sol_;
  return val;
}
PPR_FUNC(postproc_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...


  double y = xyz[1];
  return temp(time,y);
}

}//namespace kundin


namespace truchas
{
  // time is ms space is mm
  double rho_ = 7.57e-3;   // g/mm^3   density
  double cp_ = 750.65;    // (g-mm^2/ms^2)/g-K   specific heat

  //we can interp this as in truchas
  double k_ = .0213; //(g-mm^2/ms^3)/mm-K    thermal diffusivity
  double l_ = 2.1754e5; //g-mm^2/ms^2/g       latent heat
  double w_ =  2.4; 
  //double eps_ = .001;//       anisotropy strength
  double eps_ = 000547723;
  double m_ = .002149;// K ms/mm
  //double t_m_ = 1678.;
  double t_m_ = 1450.;

  double zmin_ = -.036;
  double dz_ = .0002;

  double get_k_liq_(const double temp){
    double c1 = 8.92e-3;
    double c2 = 1.474e-5;
    double r =273;
    return c1+c2*(temp-r);
    //return k_;
  }
  double get_k_sol_(const double temp){
    double c1 = 12.3e-3;
    double c2 = 1.472e-5;
    double r =273;
    return c1+c2*(temp-r);
    //return k_;
  }
RES_FUNC(residual_heat_)
{

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
  //test function
  double test = basis[0].phi[i];
  //u, phi
  double u = basis[0].uu;
  double uold = basis[0].uuold;

  double ut = rho_*cp_*(u-uold)/dt_*test;
  double divgradu = get_k_liq_(basis[0].uu)*(basis[0].dudx*dtestdx + basis[0].dudy*dtestdy + basis[0].dudz*dtestdz);//(grad u,grad phi)
  //double divgradu_old = k_*(basis[0].duolddx*dtestdx + basis[0].duolddy*dtestdy + basis[0].duolddz*dtestdz);//(grad u,grad phi)
 
 
  return (ut + t_theta_*divgradu);// /rho_/cp_;// + (1.-t_theta_)*divgradu_old;
}
//double prec_heat_test_(const boost::ptr_vector<Basis> &basis, 
//			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
PRE_FUNC(prec_heat_)
{
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
  double test = basis[0].phi[i];
  double divgrad = get_k_liq_(basis[0].uu)*(dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz);
  double u_t =rho_*cp_*test * basis[0].phi[j]/dt_;
  return (u_t + t_theta_*divgrad);// /rho_/cp_;;
}
RES_FUNC(residual_phase_)
{

  //derivatives of the test function
  double dtestdx = basis[1].dphidxi[i]*basis[1].dxidx
    +basis[1].dphideta[i]*basis[1].detadx
    +basis[1].dphidzta[i]*basis[1].dztadx;
  double dtestdy = basis[1].dphidxi[i]*basis[1].dxidy
    +basis[1].dphideta[i]*basis[1].detady
    +basis[1].dphidzta[i]*basis[1].dztady;
  double dtestdz = basis[1].dphidxi[i]*basis[1].dxidz
    +basis[1].dphideta[i]*basis[1].detadz
    +basis[1].dphidzta[i]*basis[1].dztadz;
  //test function
  double test = basis[1].phi[i];
  //u, phi
  double u = basis[1].uu;
  double uold = basis[1].uuold;

  double ut = (u-uold)/dt_*test;
  double divgradu = m_*eps_*eps_*(basis[1].dudx*dtestdx + basis[1].dudy*dtestdy + basis[1].dudz*dtestdz);//(grad u,grad phi)
  double g = 2.*m_*w_*u*(1.-u)*(1.-2.*u);
  double p = 30.*m_*l_*(t_m_-basis[0].uu)/t_m_*u*u*(1.-u)*(1.-u);
 
  return (ut + t_theta_*divgradu+ t_theta_*g + t_theta_*p)/eps_/eps_/m_;
}
PRE_FUNC(prec_phase_)
{
  //cn probably want to move each of these operations inside of getbasis
  //derivatives of the test function
  double dtestdx = basis[1].dphidxi[i]*basis[1].dxidx
    +basis[1].dphideta[i]*basis[1].detadx
    +basis[1].dphidzta[i]*basis[1].dztadx;
  double dtestdy = basis[1].dphidxi[i]*basis[1].dxidy
    +basis[1].dphideta[i]*basis[1].detady
    +basis[1].dphidzta[i]*basis[1].dztady;
  double dtestdz = basis[1].dphidxi[i]*basis[1].dxidz
    +basis[1].dphideta[i]*basis[1].detadz
    +basis[1].dphidzta[i]*basis[1].dztadz;

  double dbasisdx = basis[1].dphidxi[j]*basis[1].dxidx
    +basis[1].dphideta[j]*basis[1].detadx
    +basis[1].dphidzta[j]*basis[1].dztadx;
  double dbasisdy = basis[1].dphidxi[j]*basis[1].dxidy
    +basis[1].dphideta[j]*basis[1].detady
    +basis[1].dphidzta[j]*basis[1].dztady;
  double dbasisdz = basis[1].dphidxi[j]*basis[1].dxidz
    +basis[1].dphideta[j]*basis[1].detadz
    +basis[1].dphidzta[j]*basis[1].dztadz;
  double test = basis[1].phi[i];
  double divgrad = m_*eps_*eps_*(dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz);
  double u_t = test * basis[1].phi[j]/dt_;
  return (u_t + t_theta_*divgrad)/eps_/eps_/m_;
}
INI_FUNC(init_phase_)
{

  double phi_sol_ = 1.;
  double phi_liq_ = 0.;

  double val = phi_liq_;  

  double c = 1.5;//1.1;

  if ( z < zmin_ + c*dz_) {
    val = ((rand() % 100)/50.-1.)*1.5;
    if (val < 0.) val =0.;
    if (val > 1.) val =1.;
  }

  return val;
}

}//namespace truchas

namespace rotate
{
//t0 is rotation in x-y plane (wrt to z axis) only
//there is no derivative coded at this time
//note that to rotate around the x or y axis, is more complicated;
//requiring a rotation matrix in cartesian coordinates to be applied
const double a_sr_(const double px, const double py, const double pz, const double ep, const double t0){
    // in 2d, equivalent to: a_s = 1 + ep cos 4 (theta - t0)
    double nn = sqrt(px*px + py*py + pz*pz );
    double small = 5.e-3;//   => px < 1e-9
    //double small = a_small;
    if( nn < small*small ) return 1. + ep;

    double xx = px/nn;
    double yy = py/nn;
    double zz = pz/nn;
    double aa = .75 + (xx*xx*xx*xx + yy*yy*yy*yy -.75)*cos(4.*t0) + (xx*xx*xx*yy-yy*yy*yy*xx)*sin(4.*t0);

    return 1.-3.*ep + 4.*ep*(aa + zz*zz*zz*zz);
  }
const double da_sr_dpx(const double px, const double py, const double pz, const double ep, const double t0){
  return 0.;
}

}//namespace rotate
  //double a_small =  5.e-3;//   => px < 1e-9

namespace takaki {

  double a_small =  5.e-9;//   => px < 1e-9
  const double umperm_ = 1.e6; //1e6 um = 1 m

  const double ep4_=.02;
  const double w0_ = .9375; //um
  const double T0_ = 931.2; //K
  const double G_ = .2; //K/um
  const double Vp_ = 50.; //um/s
  const double tau0_ =1.;
  const double c0_ = .00245; //atm frac
  const double m_ = -620; //K/atm frac
  const double k_ = .14; //atm frac
  double lT_ = -m_*(1.- k_)*c0_/(k_*G_); //should be um
  double gamma_ = .24e-6/umperm_;// K/m -> K/um
  //double d0_ = k_*gamma_/(-m_*(1.-k_)*c0_);
  double d0_ = 1.3e-2;//um
  const double a1_ = 0.88388;
  double lambdastar_ = a1_*w0_/d0_;
  //const double abar_ = 1.;
  const double Dl_ = (3.e-9)*umperm_*umperm_; //m^2/s -> um^2/s
  //const double Ds_ = (2.e-12)*umperm_*umperm_; //m^2/s -> um^2/s
  const double Ds_ = Dl_;
  double a_ = (1.-k_*Ds_/Dl_)/2./std::sqrt(2.);


double dfdp(const double p){
  return -p + p*p*p;
}
double dgdp(const double p){
  return (1.-p*p)*(1.-p*p);
}
double q(const double p){
  return (k_*Ds_+Dl_+(k_*Ds_-Dl_)*p)/2./Dl_;;
}

  //see anisotropy_new.nb
  //rotation about z axis or x-y plane
const double rpx_z(const double px, const double py, const double pz, const double tz){
  double c = cos(tz);
  double s = sin(tz);
  return c*px + s*py;
}
const double drpx_zdpx(const double px, const double py, const double pz, const double tz){
  return cos(tz);
}
const double rpy_z(const double px, const double py, const double pz, const double tz){
  double c = cos(tz);
  double s = sin(tz);
  return -s*px + c*py;
}
const double drpy_zdpy(const double px, const double py, const double pz, const double tz){
  return cos(tz);
}
const double rpz_z(const double px, const double py, const double pz, const double tz){
  return pz;
}
const double drpz_zdpz(const double px, const double py, const double pz, const double tz){
  return 1.;
}
  //full 3d rotation about z axis, tz, y axis, ty, and x axis, tx
const double rpx_f(const double px, const double py, const double pz, const double tz, const double ty, const double tx){
  double cy = cos(ty);
  double cz = cos(tz);
  double sy = sin(ty);
  double sz = sin(tz);
  return cy*cz*px + cy*sz*py + sy*pz;
}
const double drpx_fdpx(const double px, const double py, const double pz, const double tz, const double ty, const double tx){
  return rpx_f(1.,0.,0.,tz,ty,tx);
}
const double rpy_f(const double px, const double py, const double pz, const double tz, const double ty, const double tx){
  double cx = cos(tx);
  double cy = cos(ty);
  double cz = cos(tz);
  double sx = sin(tx);
  double sy = sin(ty);
  double sz = sin(tz);
  return (-cz*sx*sy - cx*sz)*px + (cx*cz-sx*sy*sz)*py + cy*sx*pz;
}
const double drpy_fdpy(const double px, const double py, const double pz, const double tz, const double ty, const double tx){
  return rpy_f(0.,1.,0.,tz,ty,tx);
}
const double rpz_f(const double px, const double py, const double pz, const double tz, const double ty, const double tx){
  double cx = cos(tx);
  double cy = cos(ty);
  double cz = cos(tz);
  double sx = sin(tx);
  double sy = sin(ty);
  double sz = sin(tz);
  return (-cx*cz*sy + sx*sz)*px + (-cz*sx - cx*sy*sz)*py  + cx*cy*pz;
}
const double drpz_fdpz(const double px, const double py, const double pz, const double tz, const double ty, const double tx){
  return rpz_f(0.,0.,1.,tz,ty,tx);
}




const double a_s_(const double px, const double py, const double pz, const double ep){
    // in 2d, equivalent to: a_s = 1 + ep cos 4 theta
    double px2 = px*px;
    double py2 = py*py;
    double pz2 = pz*pz;
    double norm4 = (px2 + py2 + pz2 )*(px2 + py2 + pz2 );
    //double small = 5.e-3;//   => px < 1e-9
    if( norm4 < a_small ) return 1. + ep;
    //return 1.-3.*ep + 4.*ep*(px2*px2 + py2*py2 + pz2*pz2)/norm4;
    return 1.;
  }
const double da_s_dpx(const double px, const double py, const double pz, const double ep){
    // (16 ep x (-y^4 - z^4 + x^2 y^2 + x^2 z^2))/(x^2 + y^2 + z^2)^3
    double px2 = px*px;
    double py2 = py*py;
    double pz2 = pz*pz;
    double norm4 = (px2 + py2 + pz2 )*(px2 + py2 + pz2 );
    double norm6 = norm4*(px2 + py2 + pz2 );
    //double small = 5.e-3;//   => px < 1e-9
    if( norm4 < a_small ) return 0.;
    //return 16.*ep*px*(- py2*py2 - pz2*pz2 + px2*py2 + px2*pz2)/norm6;
    return 0.;
  }
const double da_s_dpy(const double px, const double py, const double pz, const double ep){
    // -((16 ep y (x^4 - x^2 y^2 - y^2 z^2 + z^4))/(x^2 + y^2 + z^2)^3)
    double px2 = px*px;
    double py2 = py*py;
    double pz2 = pz*pz;
    double norm4 = (px2 + py2 + pz2 )*(px2 + py2 + pz2 );
    double norm6 = norm4*(px2 + py2 + pz2 );
    //double small = 5.e-3;//   => px < 1e-9
    if( norm4 < a_small ) return 0.;
    //return - 16.*ep*py*(px2*px2 + pz2*pz2 - px2*py2 - py2*pz2)/norm6;
    return 0.;
  }
const double da_s_dpz(const double px, const double py, const double pz, const double ep){
    //-((16 ep z (x^4 + y^4 - x^2 z^2 - y^2 z^2))/(x^2 + y^2 + z^2)^3)
    double px2 = px*px;
    double py2 = py*py;
    double pz2 = pz*pz;
    double norm4 = (px2 + py2 + pz2 )*(px2 + py2 + pz2 );
    double norm6 = norm4*(px2 + py2 + pz2 );
    //double small = 5.e-3;//   => px < 1e-9
    if( norm4 < a_small ) return 0.;
    //return - 16.*ep*pz*(px2*px2 + py2*py2 - px2*pz2 - py2*pz2)/norm6;
    return 0.;
  }

const double w_(const double px, const double py, const double pz, const double ep){
  double a2 = a_s_(px, py, pz, ep);
  return w0_*a2*a2;
}
const double dw_dpx(const double px, const double py, const double pz, const double ep){
  return 2.*w0_*a_s_(px, py, pz, ep)*da_s_dpx(px, py, pz, ep);
}
const double dw_dpy(const double px, const double py, const double pz, const double ep){
  return 2.*w0_*a_s_(px, py, pz, ep)*da_s_dpy(px, py, pz, ep);
}
const double dw_dpz(const double px, const double py, const double pz, const double ep){
  return 2.*w0_*a_s_(px, py, pz, ep)*da_s_dpz(px, py, pz, ep);
}

double temp(const double time, const double y){
  return  T0_ + G_*(y-Vp_*time);
}

double uprime(const double time, const double y){
  return  (y-Vp_*time)/lT_;
}

const double tau_(const double px, const double py, const double pz, const double ep){
  double a2 = a_s_(px, py, pz, ep);
  return tau0_*a2*a2;
}

INI_FUNC(init_conc_)
{
  //return c0_;//cn should be u0
  return 0.;
}

INI_FUNC(init_phase_)
{
  double dx = .75;
  double lfo = 230.*dx;
  double luo = 300.*dx; 
  double val = -1.;
  double r2 = x*x + y*y;

  double rr2 =.75*.75*dx*dx;

  if(r2 < rr2 ){
    val=1.;
  }

  r2 = (x - lfo)*(x - lfo) + y*y;

  if(r2 < rr2 ){
    val=1.;
  }
  r2 = (x - (lfo+luo))*(x - (lfo+luo)) + y*y;

  if(r2 < rr2 ){
    val=1.;
  }

  return val;
}

RES_FUNC(residual_conc_)
{
  //derivatives of the test function
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;
  //test function
  double test = basis[0].phi[i];
  //u, phi
  //double u = basis[0].uu;
  double u = basis[0].uuold;
  //double uold = basis[0].uuold;
  double uold = basis[0].uuoldold;
  //double phi = basis[1].uu;
  double phi = basis[1].uuold;
  //double phiold = basis[1].uuold;
  double phiold = basis[1].uuoldold;
  //double dphidx = basis[1].dudx;
  double dphidx = basis[1].duolddx;
  //double dphidy = basis[1].dudy;
  double dphidy = basis[1].duolddy;

  double dtc = (1.+k_-(1.-k_)*phi)/2.;
  //double ut = dtc*(u-uold)/dt_*test;
  double ut = dtc*(basis[0].uu-basis[0].uuold)/dt_*test;
  //ut = (u-uold)/dt_*test;
  double divgradu = Dl_*q(phi)*(basis[0].dudx*dtestdx + basis[0].dudy*dtestdy);//(grad u,grad phi)
  //double divgradu_old = D*(1.-phiold)/2*(basis[0].duolddx*dtestdx + basis[0].duolddy*dtestdy);//(grad u,grad phi)

  //j is antitrapping current
  // j grad test here... j1*dtestdx + j2*dtestdy 
  // what if dphidx*dphidx + dphidy*dphidy = 0?

  double norm = sqrt(dphidx*dphidx + dphidy*dphidy);
  double small = 1.e-12;
  
  double j_coef = 0.;
  if (small < norm) {
    j_coef = a_*w0_*(1.+(1.-k_)*u)/norm*(phi-phiold)/dt_;
  } 
  //j_coef = 0.;
  double j1 = j_coef*dphidx;
  double j2 = j_coef*dphidy;
  double divj = j1*dtestdx + j2*dtestdy;
  double phitu = -.5*(phi-phiold)/dt_*(1.+(1.-k_)*u)*test; 
  //phitu = 1.*test; 
//   h = hold;
//   hold = basis[1].uuoldold*(1. + (1.-k_)*basis[0].uuoldold);
//   double phitu_old = -.5*(h-hold)/dt_*test;
 
  //return ut*0.  + t_theta_*(divgradu + divj*0.) + (1.-t_theta_)*(divgradu_old + divj_old)*0. + t_theta_*phitu*0. + (1.-t_theta_)*phitu_old*0.;

  //std::cout<<"u: "<<ut<<" "<< t_theta_*divgradu  <<" "<< 0.*t_theta_*divj <<" "<< t_theta_*phitu<<"  :  "<<dtc<<std::endl;
  return (ut + t_theta_*divgradu  + 0.*t_theta_*divj + t_theta_*phitu)/dtc*dt_;
}
RES_FUNC(residual_phase_)
{
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
  //test function
  double test = basis[0].phi[i];
  //u, phi
  //double u = basis[0].uu;
  double u = basis[0].uuold;
  //double uold = basis[0].uuold;
  double uold = basis[0].uuoldold;
  //double phi = basis[1].uu;
  double phi = basis[1].uuold;
  //double phiold = basis[1].uuold;
  double phiold = basis[1].uuoldold;

  //double dphidx = basis[1].dudx;
  double dphidx = basis[1].duolddx;
  //double dphidy = basis[1].dudy;
  double dphidy = basis[1].duolddy;

  double y = basis[0].yy;
  double up = uprime(y,time);
  double dtc = tau_(dphidx,dphidy,0.,ep4_)*(1.-(1.-k_)*up);
  //double phit = dtc*(phi-phiold)/dt_*test;
  double phit = dtc*(basis[1].uu-basis[1].uuold)/dt_*test;

  double w = w_(dphidx,dphidy,0.,ep4_);
  double divgrad = w*w*(dphidx*dtestdx + dphidy*dtestdy);//(grad u,grad phi)

  double norm2 = dphidx*dphidx + dphidy*dphidy;
  double curlgrad = w*norm2*(dw_dpx(dphidx,dphidy,0.,ep4_)*dtestdx + dw_dpy(dphidx,dphidy,0.,ep4_)*dtestdy);

  double df = dfdp(phi)*test;
  double dg = lambdastar_*dgdp(phi)*(u + up)*test; 

  //std::cout<<"phi: "<<phit <<" "<< divgrad <<" "<< curlgrad <<" "<< df <<" "<<dg<<" "<<d0_<<"  :  "<<dtc<<std::endl;
  return (phit + t_theta_*(divgrad + curlgrad + df +dg))/dtc*dt_;
}


}//namespace takaki

namespace allencahn {


double kappa = 0.0004;
const double pi = 3.141592653589793;
const double A1 = 0.0075;
const double B1 = 8.0*pi;
const double A2 = 0.03;
const double B2 = 22.0*pi;
const double C2 = 0.0625*pi;



PARAM_FUNC(param_)
{
  //kappa = plist->get<double>("kappa");
}

double alpha(double t, double x) {

   double alpha_result;
   alpha_result = A1*t*sin(B1*x) + A2*sin(B2*x + C2*t) + 0.25;
   return alpha_result;

}

double eta(double a, double y) {

   double eta_result;
   eta_result = -1.0L/2.0L*tanh((1.0L/2.0L)*sqrt(2)*(-a + y)/sqrt(kappa)) + 1.0L/2.0L;
   return eta_result;

}

double eta0(double a, double y) {

   double eta0_result;
   eta0_result = -1.0L/2.0L*tanh((1.0L/2.0L)*sqrt(2)*(-a + y)/sqrt(kappa)) + 1.0L/2.0L;
   return eta0_result;

}

double dadt(double t, double x) {

   double dadt_result;
   dadt_result = A1*sin(B1*x) + A2*C2*cos(B2*x + C2*t);
   return dadt_result;

}

double dadx(double t, double x) {

   double dadx_result;
   dadx_result = A1*B1*t*cos(B1*x) + A2*B2*cos(B2*x + C2*t);
   return dadx_result;

}

double d2adx2(double t, double x) {

   double d2adx2_result;
   d2adx2_result = -A1*pow(B1, 2)*t*sin(B1*x) - A2*pow(B2, 2)*sin(B2*x + C2*t);
   return d2adx2_result;

}

double source(double a, double t, double x, double y) {

   double source_result;
   source_result = (1.0L/4.0L)*(-pow(tanh((1.0L/2.0L)*sqrt(2)*(-a + y)/sqrt(kappa)), 2) + 1)*(-2*sqrt(kappa)*pow(A1*B1*t*cos(B1*x) + A2*B2*cos(B2*x + C2*t), 2)*tanh((1.0L/2.0L)*sqrt(2)*(-a + y)/sqrt(kappa)) - sqrt(2)*kappa*(-A1*pow(B1, 2)*t*sin(B1*x) - A2*pow(B2, 2)*sin(B2*x + C2*t)) + sqrt(2)*(A1*sin(B1*x) + A2*C2*cos(B2*x + C2*t)))/sqrt(kappa);
   return source_result;

}
RES_FUNC(residual_)
{
  //derivatives of the test function
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;
  //test function
  double test = basis[0].phi[i];

  double u[2] = {basis[0].uu, basis[0].uuold};
  double u_x[2] = {basis[0].dudx, basis[0].duolddx};
  double u_y[2] = {basis[0].dudy, basis[0].duolddy};

  double ut = (u[0]-u[1])/dt_*test;

  double divgrad = t_theta_*kappa*(u_x[0]*dtestdx + u_y[0]*dtestdy)
             +(1.-t_theta_)*kappa*(u_x[1]*dtestdx + u_y[1]*dtestdy);
  double fp = (-t_theta_*4.*u[0]*(u[0]-1.)*(u[0]-.5)
	  -(1.-t_theta_)*4.*u[1]*(u[1]-1.)*(u[1]-.5))*test;
  
  double x = basis[0].xx;
  double y = basis[0].yy;
  double t[2] = {time, time - dt_};
  double a[2] = {alpha(t[0],x),alpha(t[1],x)};
  double s = (-t_theta_*source(a[0],t[0],x,y)
         -(1.-t_theta_)*source(a[1],t[1],x,y))*test;

  return ut + divgrad + fp + s;
}
PRE_FUNC(prec_)
{
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
  double test = basis[0].phi[i];
  
  double u_t = test * basis[0].phi[j]/dt_;

  double divgrad = t_theta_*kappa*(dbasisdx * dtestdx + dbasisdy * dtestdy);

  return (u_t + divgrad);
}
INI_FUNC(init_)
{
  double a = alpha(0.,x);
  double val = eta0(a,y);
  return val;
}
PPR_FUNC(postproc_)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...


  double x = xyz[0];
  double y = xyz[1];
  double a = alpha(time,x);
  double val = eta(a,y);

  return val;
}
PPR_FUNC(postproc_error)
{
  //u is u0,u1,...
  //gradu is dee0/dx,dee0/dy,dee0/dz,dee1/dx,dee1/dy,dee1/dz...


  // x is in nondimensional space, tscale_ takes in nondimensional and converts to um
  double x = xyz[0];
  double y = xyz[1];
  double a = alpha(time,x);
  double val = (u[0]-eta(a,y));
  //double val = (u[0]-eta(a,y))*(u[0]-eta(a,y));

  return val;
}

}//namespace allencahn
namespace pfhub2 {

  int N_ = 1;
  int eqn_off_ =1;
  const double c0_ = .5;
  const double eps_ = .05;
  const double eps_eta_ = .1;
  const double psi_ = 1.5;
  const double rho_ = std::sqrt(2.);
  const double c_alpha_ = .3;
  const double c_beta_ = .7;
  const double alpha_ = 5.;
  const double k_c_ = 3.;
  const double k_eta_ = 3.;
  const double M_ = 5.;
  const double L_ = 5.;
  const double w_ = 1.;
  double c_a[2] = {0., 0.};
  double c_b[2] = {0., 0.};

  PARAM_FUNC(param_)
  {
    N_ = plist->get<int>("N");
    eqn_off_ = plist->get<int>("OFFSET");
  }

  double f_alpha(const double c){
    return rho_*rho_*(c - c_alpha_)*(c - c_alpha_);
  }

  double df_alphadc(const double c){
    return 2.*rho_*rho_*(c - c_alpha_);
  }

  double f_beta(const double c){
    return rho_*rho_*(c_beta_ - c)*(c_beta_ - c);
  }

  double df_betadc(const double c){
    return -2.*rho_*rho_*(c_beta_ - c);
  }

  double d2fdc2(){
    return 2.*rho_*rho_;
  }

  double h(const double *eta){
    double val = 0.;
    for (int i = 0; i < N_; i++){
      val += eta[i]*eta[i]*eta[i]*(6.*eta[i]*eta[i] - 15.*eta[i] + 10.);
    }
    return val;
  }

  //double dhdeta(const double *eta, const int eqn_id){
  double dhdeta(const double eta){

    //return 30.*eta[eqn_id]*eta[eqn_id] - 60.*eta[eqn_id]*eta[eqn_id]*eta[eqn_id] + 30.*eta[eqn_id]*eta[eqn_id]*eta[eqn_id]*eta[eqn_id];
    return 30.*eta*eta - 60.*eta*eta*eta + 30.*eta*eta*eta*eta;
  }

//   double d2hdeta2(const double *eta, const int eqn_id){

//     return 60.*eta[eqn_id] - 180.*eta[eqn_id]*eta[eqn_id] + 120.*eta[eqn_id]*eta[eqn_id]*eta[eqn_id];
//   }

  double g(const double *eta){

    double aval =0.;
    for (int i = 0; i < N_; i++){
      aval += eta[i]*eta[i];
    }
    
    double val = 0.;
    for (int i = 0; i < N_; i++){
      val += eta[i]*eta[i]*(1.-eta[i])*(1.-eta[i]) + alpha_*eta[i]*eta[i]*aval - alpha_*eta[i]*eta[i]*eta[i]*eta[i];
    }
    return val;
  }

  double dgdeta(const double *eta, const int eqn_id){

    double aval =0.;
    for (int i = 0; i < N_; i++){
      aval += eta[i]*eta[i];
    }
    return 2.*eta[eqn_id]*(1. - eta[eqn_id])*(1. - eta[eqn_id])  - 2.* eta[eqn_id]* eta[eqn_id]* (1. - eta[eqn_id]) + 
      4.*alpha_*eta[eqn_id] *aval - 4.*alpha_*eta[eqn_id]*eta[eqn_id]*eta[eqn_id];
  }

//   double d2gdeta2(const double *eta, const int eqn_id){

//     double aval =0.;
//     for (int i = 0; i < N_; i++){
//       aval += eta[i]*eta[i];
//     }
//     return 2. - 12.*eta[eqn_id] + 12.*eta[eqn_id]*eta[eqn_id] + alpha_*4.*aval - 4.*alpha_*eta[eqn_id]*eta[eqn_id];
//   }



  //void solve_kks(double &c_a, double &c_b, const double c, double phi)//const double phi
  void solve_kks(const double c, double *phi)//const double phi
  {
    double delta_c_a = 0.;
    double delta_c_b = 0.;
    int max_iter = 20;
    double tol = 1.e-8;
    double hh = h(phi);
    //c_a[0] = (1.-hh)*c;
    c_a[0] = c - hh*(c_beta_ - c_alpha_);

    //c_b[0]=hh*c;
    c_b[0]= c - (1.-hh)*(c_beta_ - c_alpha_);

    //std::cout<<"-1"<<" "<<delta_c_b<<" "<<delta_c_a<<" "<<c_b[0]<<" "<<c_a[0]<<" "<<hh*c_b[0] + (1.- hh)*c_a[0]<<" "<<c<<std::endl;
    for(int i = 0; i < max_iter; i++){
      double det = hh*d2fdc2() + (1.-hh)*d2fdc2();
      double f1 = hh*c_b[0] + (1.- hh)*c_a[0] - c;
      double f2 = df_betadc(c_b[0]) - df_alphadc(c_a[0]);
      delta_c_b = (-d2fdc2()*f1 - (1-hh)*f2)/det;
      delta_c_a = (-d2fdc2()*f1 + hh*f2)/det;
      c_b[0] = delta_c_b + c_b[0];
      c_a[0] = delta_c_a + c_a[0];
      //std::cout<<i<<" "<<delta_c_b<<" "<<delta_c_a<<" "<<c_b[0]<<" "<<c_a[0]<<" "<<hh*c_b[0] + (1.- hh)*c_a[0]<<" "<<c<<std::endl;
      if(delta_c_a*delta_c_a+delta_c_b*delta_c_b < tol*tol) return;
    }
    std::cout<<"###################################  solve_kks falied to converge with delta_c_a*delta_c_a+delta_c_b*delta_c_b = "
	     <<delta_c_a*delta_c_a+delta_c_b*delta_c_b<<"  ###################################"<<std::endl;
    //if(delta_c_a*delta_c_a+delta_c_b*delta_c_b > 0) exit(0);
    exit(0);
    return;
  }

RES_FUNC(residual_c_)
{

  //derivatives of the test function
  double dtestdx = basis[0].dphidx[i];
  double dtestdy = basis[0].dphidy[i];
  //double dtestdz = basis[0].dphidz[i];
  //test function
  double test = basis[0].phi[i];
  //u, phi
  double c[2] = {basis[0].uu, basis[0].uuold};
  double dcdx[2] = {basis[0].dudx, basis[0].duolddx};
  double dcdy[2] = {basis[0].dudy, basis[0].duolddy};

  double dhdx[2] = {0., 0.};
  double dhdy[2] = {0., 0.};

  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    dhdx[0] += dhdeta(basis[kk_off].uu)*basis[kk_off].dudx;
    dhdx[1] += dhdeta(basis[kk_off].uuold)*basis[kk_off].duolddx;
    dhdy[0] += dhdeta(basis[kk_off].uu)*basis[kk_off].dudy;
    dhdy[1] += dhdeta(basis[kk_off].uuold)*basis[kk_off].duolddy;
  }

  double ct = (c[0]-c[1])/dt_*test;

  double DfDc[2] = {df_betadc(c[0])-df_alphadc(c[0]),
		    df_betadc(c[1])-df_alphadc(c[1])};

  double D2fDc2 = d2fdc2();

  double dfdx[2] = {DfDc[0]*dhdx[0] + D2fDc2*dcdx[0],
		    DfDc[1]*dhdx[1] + D2fDc2*dcdx[1]};
  double dfdy[2] = {DfDc[0]*dhdy[0] + D2fDc2*dcdy[0],
		    DfDc[1]*dhdy[1] + D2fDc2*dcdy[1]};

  double divgradc[2] = {M_*(dfdx[0]*dtestdx + dfdy[0]*dtestdy),
			M_*(dfdx[1]*dtestdx + dfdy[1]*dtestdy)};

  return ct + t_theta_*divgradc[0] + (1.-t_theta_)*divgradc[1];
}

PRE_FUNC(prec_c_)
{
  //cn probably want to move each of these operations inside of getbasis
  //derivatives of the test function
  double dtestdx = basis[0].dphidx[i];
  double dtestdy = basis[0].dphidy[i];
  double dtestdz = basis[0].dphidz[i];

  double dbasisdx = basis[0].dphidxi[j]*basis[0].dxidx
    +basis[0].dphideta[j]*basis[0].detadx
    +basis[0].dphidzta[j]*basis[0].dztadx;
  double dbasisdy = basis[0].dphidxi[j]*basis[0].dxidy
    +basis[0].dphideta[j]*basis[0].detady
    +basis[0].dphidzta[j]*basis[0].dztady;
  double dbasisdz = basis[0].dphidxi[j]*basis[0].dxidz
    +basis[0].dphideta[j]*basis[0].detadz
    +basis[0].dphidzta[j]*basis[0].dztadz;
  double test = basis[0].phi[i];
  double divgrad = dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz;
  double u_t =test * basis[0].phi[j]/dt_;
  double D2fDc2 = d2fdc2();
  return u_t + t_theta_*M_*D2fDc2*divgrad;
}


RES_FUNC(residual_eta_)
{

  //derivatives of the test function
  double dtestdx = basis[0].dphidx[i];
  double dtestdy = basis[0].dphidy[i];
  //double dtestdz = basis[0].dphidz[i];
  //test function
  double test = basis[eqn_id].phi[i];
  //u, phi
  double c[2] = {basis[0].uu, basis[0].uuold};

  double eta[2] = {basis[eqn_id].uu, basis[eqn_id].uuold};
  double detadx[2] = {basis[eqn_id].dudx, basis[eqn_id].duolddx};
  double detady[2] = {basis[eqn_id].dudy, basis[eqn_id].duolddy};

  double eta_array[N_];
  double eta_array_old[N_];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    eta_array[kk] = basis[kk_off].uu;
    eta_array_old[kk] = basis[kk_off].uuold;
  }

  double etat = (eta[0]-eta[1])/dt_*test;

  double F[2] = {f_beta(c[0])-f_alpha(c[0]),
		 f_beta(c[1])-f_alpha(c[1])};

  int k = eqn_id - eqn_off_;
  double dfdeta[2] = {L_*(F[0]*dhdeta(eta[0]) + w_*dgdeta(eta_array,k))*test,
		      L_*(F[1]*dhdeta(eta[1]) + w_*dgdeta(eta_array_old,k))*test};

  double divgradeta[2] = {L_*k_eta_*(detadx[0]*dtestdx + detady[0]*dtestdy), 
			  L_*k_eta_*(detadx[1]*dtestdx + detady[1]*dtestdy)};//(grad u,grad phi)
 
  return etat + t_theta_*divgradeta[0] + t_theta_*dfdeta[0] + (1.-t_theta_)*divgradeta[1] + (1.-t_theta_)*dfdeta[1];
}

PRE_FUNC(prec_eta_)
{
  //derivatives of the test function
  double dtestdx = basis[0].dphidx[i];
  double dtestdy = basis[0].dphidy[i];
  double dtestdz = basis[0].dphidz[i];

  double dbasisdx = basis[0].dphidxi[j]*basis[0].dxidx
    +basis[0].dphideta[j]*basis[0].detadx
    +basis[0].dphidzta[j]*basis[0].dztadx;
  double dbasisdy = basis[0].dphidxi[j]*basis[0].dxidy
    +basis[0].dphideta[j]*basis[0].detady
    +basis[0].dphidzta[j]*basis[0].dztady;
  double dbasisdz = basis[0].dphidxi[j]*basis[0].dxidz
    +basis[0].dphideta[j]*basis[0].detadz
    +basis[0].dphidzta[j]*basis[0].dztadz;
  double test = basis[0].phi[i];
  double divgrad = dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz;
  double u_t =test * basis[0].phi[j]/dt_;
  return u_t + t_theta_*L_*k_eta_*divgrad;
}

RES_FUNC(residual_c_kks_)
{

  //derivatives of the test function
  double dtestdx = basis[0].dphidx[i];
  double dtestdy = basis[0].dphidy[i];
  //double dtestdz = basis[0].dphidz[i];
  //test function
  double test = basis[0].phi[i];
  //u, phi
  double c[2] = {basis[0].uu, basis[0].uuold};
  double dcdx[2] = {basis[0].dudx, basis[0].duolddx};
  double dcdy[2] = {basis[0].dudy, basis[0].duolddy};

  double dhdx[2] = {0., 0.};
  double dhdy[2] = {0., 0.};

  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    dhdx[0] += dhdeta(basis[kk_off].uu)*basis[kk_off].dudx;
    dhdx[1] += dhdeta(basis[kk_off].uuold)*basis[kk_off].duolddx;
    dhdy[0] += dhdeta(basis[kk_off].uu)*basis[kk_off].dudy;
    dhdy[1] += dhdeta(basis[kk_off].uuold)*basis[kk_off].duolddy;
  }

  double ct = (c[0]-c[1])/dt_*test;

  double eta_array[N_];
  double eta_array_old[N_];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    eta_array[kk] = basis[kk_off].uu;
    eta_array_old[kk] = basis[kk_off].uuold;
  }

  solve_kks(c[0],eta_array);

  double DfDc[2] = {-c_b[0] + c_a[0],
 		    -c_b[1] + c_a[1]};

  double D2fDc2 = 1.;
  //double D2fDc2 = 1.*d2fdc2();

  double dfdx[2] = {DfDc[0]*dhdx[0] + D2fDc2*dcdx[0],
		    DfDc[1]*dhdx[1] + D2fDc2*dcdx[1]};
  double dfdy[2] = {DfDc[0]*dhdy[0] + D2fDc2*dcdy[0],
		    DfDc[1]*dhdy[1] + D2fDc2*dcdy[1]};

  double divgradc[2] = {M_*(dfdx[0]*dtestdx + dfdy[0]*dtestdy),
			M_*(dfdx[1]*dtestdx + dfdy[1]*dtestdy)};

  return ct + t_theta_*divgradc[0] + (1.-t_theta_)*divgradc[1];
}

RES_FUNC(residual_eta_kks_)
{

  //derivatives of the test function
  double dtestdx = basis[eqn_id].dphidx[i];
  double dtestdy = basis[eqn_id].dphidy[i];
  //double dtestdz = basis[0].dphidz[i];
  //test function
  double test = basis[eqn_id].phi[i];
  //u, phi
  double c[2] = {basis[0].uu, basis[0].uuold};

  double eta[2] = {basis[eqn_id].uu, basis[eqn_id].uuold};
  double detadx[2] = {basis[eqn_id].dudx, basis[eqn_id].duolddx};
  double detady[2] = {basis[eqn_id].dudy, basis[eqn_id].duolddy};

  double eta_array[N_];
  double eta_array_old[N_];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    eta_array[kk] = basis[kk_off].uu;
    eta_array_old[kk] = basis[kk_off].uuold;
  }

  solve_kks(c[0],eta_array);

  double etat = (eta[0]-eta[1])/dt_*test;


  double F[2] = {f_beta(c_b[0])-f_alpha(c_a[0]) - (c_b[0] - c_a[0])*df_betadc(c_b[0]),
		 f_beta(c_b[1])-f_alpha(c_a[1]) - (c_b[1] - c_a[1])*df_betadc(c_b[1])};

  int k = eqn_id - eqn_off_;
  double dfdeta[2] = {L_*(F[0]*dhdeta(eta[0]) + w_*dgdeta(eta_array,k)    )*test,
		      L_*(F[1]*dhdeta(eta[1]) + w_*dgdeta(eta_array_old,k))*test};

  double divgradeta[2] = {L_*k_eta_*(detadx[0]*dtestdx + detady[0]*dtestdy), 
			  L_*k_eta_*(detadx[1]*dtestdx + detady[1]*dtestdy)};//(grad u,grad phi)
 
  return etat + t_theta_*divgradeta[0] + t_theta_*dfdeta[0] + (1.-t_theta_)*divgradeta[1] + (1.-t_theta_)*dfdeta[1];
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
  const double i = (double)(eqn_id -eqn_off_ + 1);
  return eps_eta_*std::pow(cos((0.01*i)*x-4.)*cos((0.007+0.01*i)*y)
			   + cos((0.11+0.01*i)*x)*cos((0.11+0.01*i)*y)		   
			   + psi_*std::pow(cos((0.046+0.001*i)*x+(0.0405+0.001*i)*y)
					   *cos((0.031+0.001*i)*x-(0.004+0.001*i)*y),2
					   ),2
			   );
}

RES_FUNC(residual_c_alpha_)
{

  double test = basis[eqn_id].phi[i];
  double c = basis[0].uu;
  double ca = basis[1].uu;
  double cb = basis[2].uu;
  double eta_array[N_];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    eta_array[kk] = basis[kk_off].uu;
  }

  return (h(eta_array)*cb + (1.- h(eta_array))*ca - c)*test;
}

PRE_FUNC(prec_c_alpha_)
{
  double test = basis[0].phi[i];
  double eta_array[N_];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    eta_array[kk] = basis[kk_off].uu;
  }
  return (1.-h(eta_array)) * basis[eqn_id].phi[j]*test;
  //return basis[eqn_id].phi[j]*test;
}

  RES_FUNC(residual_c_beta_)
{
  double test = basis[eqn_id].phi[i];
  double ca = basis[1].uu;
  double cb = basis[2].uu;
  return (df_betadc(cb) - df_alphadc(ca))*test;
}

PRE_FUNC(prec_c_beta_)
{
  double test = basis[eqn_id].phi[i];
  double c_b = basis[eqn_id].phi[j];
  //return df_betadc(c_b)*c_b*test;
  return c_b*test;
 }

INI_FUNC(init_c_alpha_)
{
   double c = init_c_(x,y,z,0);

  double eta_array[N_];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    eta_array[kk] = init_eta_(x,y,z,kk_off);
  }

  return c-h(eta_array)*(c_beta_ - c_alpha_);
}

INI_FUNC(init_c_beta_)
{
  double c = init_c_(x,y,z,0);

  double eta_array[N_];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    eta_array[kk] = init_eta_(x,y,z,kk_off);
  }

  return c+(1-h(eta_array))*(c_beta_ - c_alpha_);
}

PPR_FUNC(postproc_c_b_)
{
  //cn will need eta_array here...
  double cc = u[0];
  double phi = u[1];

  solve_kks(cc,&phi);

  return c_b[0];
}
PPR_FUNC(postproc_c_a_)
{

  //cn will need eta_array here...
  double cc = u[0];
  double phi = u[1];

  solve_kks(cc,&phi);

  return c_a[0];
}

RES_FUNC(residual_c_kkspp_)
{

  //derivatives of the test function
  double dtestdx = basis[0].dphidx[i];
  double dtestdy = basis[0].dphidy[i];
  //double dtestdz = basis[0].dphidz[i];
  //test function
  double test = basis[0].phi[i];
  //u, phi
  double c[2] = {basis[0].uu, basis[0].uuold};
  double dcdx[2] = {basis[0].dudx, basis[0].duolddx};
  double dcdy[2] = {basis[0].dudy, basis[0].duolddy};

  double dhdx[2] = {0., 0.};
  double dhdy[2] = {0., 0.};

  double eta_array[N_];
  double eta_array_old[N_];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
//     dhdx[0] += dhdeta(basis[kk_off].uu)*basis[kk_off].dudx;
//     dhdx[1] += dhdeta(basis[kk_off].uuold)*basis[kk_off].duolddx;
//     dhdy[0] += dhdeta(basis[kk_off].uu)*basis[kk_off].dudy;
//     dhdy[1] += dhdeta(basis[kk_off].uuold)*basis[kk_off].duolddy;
    eta_array[kk] = basis[kk_off].uu;
    eta_array_old[kk] = basis[kk_off].uuold;
  }

  double ct = (c[0]-c[1])/dt_*test;

//   double ca[2] = {basis[1].uu, basis[1].uuold};
//   double cb[2] = {basis[2].uu, basis[2].uuold};

//   double DfDc[2] = {cb[0] - ca[0],
// 		    cb[1] - ca[1]};

  double D2fDc2 = 1.;
  //double D2fDc2 = 1.*d2fdc2();

//   double dfdx[2] = {M_*(DfDc[0]*dhdx[0] + D2fDc2*dcdx[0]),
// 		    M_*(DfDc[1]*dhdx[1] + D2fDc2*dcdx[1])};]
//   double dfdy[2] = {M_*(DfDc[0]*dhdy[0] + D2fDc2*dcdy[0]),
// 		    M_*(DfDc[1]*dhdy[1] + D2fDc2*dcdy[1])};

//   double divgradc[2] = {dfdx[0]*dtestdx + dfdy[0]*dtestdy,
// 			dfdx[1]*dtestdx + dfdy[1]*dtestdy};

  //double eta[1] =  {basis[3].uu};

  double divgradc[2] = { M_*((1.-h(eta_array))*basis[1].dudx + h(eta_array)*basis[2].dudx)*dtestdx
			+M_*((1.-h(eta_array))*basis[1].dudy + h(eta_array)*basis[2].dudy)*dtestdy,
			 M_*((1.-h(eta_array_old))*basis[1].duolddx + h(eta_array_old)*basis[2].duolddx)*dtestdx
			+M_*((1.-h(eta_array_old))*basis[1].duolddy + h(eta_array_old)*basis[2].duolddy)*dtestdy
			};


  return ct + t_theta_*divgradc[0] + (1.-t_theta_)*divgradc[1];
}

RES_FUNC(residual_eta_kkspp_)
{

  //derivatives of the test function
  double dtestdx = basis[eqn_id].dphidx[i];
  double dtestdy = basis[eqn_id].dphidy[i];
  //double dtestdz = basis[0].dphidz[i];
  //test function
  double test = basis[eqn_id].phi[i];
  //u, phi
  double c[2] = {basis[0].uu, basis[0].uuold};

  double eta[2] = {basis[eqn_id].uu, basis[eqn_id].uuold};
  double detadx[2] = {basis[eqn_id].dudx, basis[eqn_id].duolddx};
  double detady[2] = {basis[eqn_id].dudy, basis[eqn_id].duolddy};

  double eta_array[N_];
  double eta_array_old[N_];
  for( int kk = 0; kk < N_; kk++){
    int kk_off = kk + eqn_off_;
    eta_array[kk] = basis[kk_off].uu;
    eta_array_old[kk] = basis[kk_off].uuold;
  }

  double etat = (eta[0]-eta[1])/dt_*test;

  double ca[2] = {basis[1].uu, basis[1].uuold};
  double cb[2] = {basis[2].uu, basis[2].uuold};

//   double F[2] = {f_beta(c[0])-f_alpha(c[0]),
// 		 f_beta(c[1])-f_alpha(c[1])};
  double F[2] = {f_beta(cb[0])-f_alpha(ca[0]) - (cb[0] - ca[0])*df_betadc(cb[0]),
 		 f_beta(cb[1])-f_alpha(ca[1]) - (cb[1] - ca[1])*df_betadc(cb[1])};

  int k = eqn_id - eqn_off_;
  double dfdeta[2] = {L_*(F[0]*dhdeta(eta[0]) + w_*dgdeta(eta_array,k))*test,
		      L_*(F[1]*dhdeta(eta[1]) + w_*dgdeta(eta_array_old,k))*test};

  double divgradeta[2] = {L_*k_eta_*(detadx[0]*dtestdx + detady[0]*dtestdy), 
			  L_*k_eta_*(detadx[1]*dtestdx + detady[1]*dtestdy)};//(grad u,grad phi)
 
  return etat + t_theta_*divgradeta[0] + t_theta_*dfdeta[0] + (1.-t_theta_)*divgradeta[1] + (1.-t_theta_)*dfdeta[1];
}

}//namespace pfhub2


#define RES_FUNC_TPETRA(NAME)  double NAME(const GPUBasis *basis, \
                                    const int &i,\
                                    const double &dt_,\
			            const double &t_theta_,\
                                    const double &time,\
				    const int &eqn_id)

#define PRE_FUNC_TPETRA(NAME)  double NAME(const GPUBasis *basis, \
                                    const int &i,\
				    const int &j,\
				    const double &dt_,\
				    const double &t_theta_,\
				    const int &eqn_id)


namespace tpetra{//we can just put the KOKKOS... around the other dbc_zero_ later...
  //namespace heat{




#ifdef KOKKOS_HAVE_CUDA
__device__
#endif
double k_d = 2.;

double k_h = 2.;

KOKKOS_INLINE_FUNCTION 
DBC_FUNC(dbc_zero_) 
{
  return 0.;
}

KOKKOS_INLINE_FUNCTION 
INI_FUNC(init_heat_test_)
{

  double pi = 3.141592653589793;

  return sin(pi*x)*sin(pi*y);
}

KOKKOS_INLINE_FUNCTION 
RES_FUNC_TPETRA(residual_heat_test_)
{
  return (basis[eqn_id].uu-basis[eqn_id].uuold)/dt_*basis[eqn_id].phi[i]
    + t_theta_*k_d*(basis[eqn_id].dudx*basis[eqn_id].dphidx[i]
       + basis[eqn_id].dudy*basis[eqn_id].dphidy[i]
       + basis[eqn_id].dudz*basis[eqn_id].dphidz[i])
    +(1. - t_theta_)*k_d*(basis[eqn_id].duolddx*basis[eqn_id].dphidx[i]
		   + basis[eqn_id].duolddy*basis[eqn_id].dphidy[i]
		   + basis[eqn_id].duolddz*basis[eqn_id].dphidz[i]);
}
#ifdef KOKKOS_HAVE_CUDA
__device__ RES_FUNC_TPETRA((*residual_heat_test_dp_)) = residual_heat_test_;
#else
RES_FUNC_TPETRA((*residual_heat_test_dp_)) = residual_heat_test_;
#endif

KOKKOS_INLINE_FUNCTION 
PRE_FUNC_TPETRA(prec_heat_test_)
{
  return basis[eqn_id].phi[j]/dt_*basis[eqn_id].phi[i]
    + t_theta_*k_d*(basis[eqn_id].dphidx[j]*basis[eqn_id].dphidx[i]
       + basis[eqn_id].dphidy[j]*basis[eqn_id].dphidy[i]
       + basis[eqn_id].dphidz[j]*basis[eqn_id].dphidz[i]);
}

#ifdef KOKKOS_HAVE_CUDA
__device__ PRE_FUNC_TPETRA((*prec_heat_test_dp_)) = prec_heat_test_;
#else
PRE_FUNC_TPETRA((*prec_heat_test_dp_)) = prec_heat_test_;
#endif

PARAM_FUNC(param_)
{
  double kk = plist->get<double>("k_",1.);

#ifdef KOKKOS_HAVE_CUDA
  cudaMemcpyToSymbol(k_d,&kk,sizeof(double));
#else
  k_d = kk;
#endif
  k_h = kk;
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

//}//namespace heat


namespace farzadi3d
{

  const double absphi = 0.999999;	//1.
  
  double pi = 3.141592653589793;

  double k = 0.14;					//0.5
  double eps = 0.0;
  double lambda = 10.;
  double D_liquid = 3.e-9;			//1.e-11				//m^2/s
  
  double m = -2.6;					//-2.6 100.
  double c_inf = 3.;				//1.
  
  double G = 3.e5;											//k/m
  double R = 0.003;											//m/s
  double V = R;												//m/s
  double d0 = 5.e-9;				//4.e-9					//m
  
  
  // parameters to scale dimensional quantities
  double delta_T0 = abs(m)*c_inf*(1.-k)/k;
  double w0 = lambda*d0/0.8839;
  double tau0 = (lambda*0.6267*w0*w0)/D_liquid;
  
  double Vp0 = V*tau0/w0; 
  double l_T0 = delta_T0/(G*w0);
  double D_liquid_ = D_liquid*tau0/(w0*w0);
  
  double base_height = 15.;
  double amplitude = 0.2;
  
   //double tl = 925.2;//k
  //double ts = 877.3;//k
  //double t0 = ts;
  //double t0 = 900.;//k
  //double dt0 = tl-ts;
  
  //for the farzadiQuad1000x360mmr.e mesh...
  //double pp = 360.;
  //double ll = 40.;
  //double aa = 14.;
  
  
  	PARAM_FUNC(param_)
	{
	  k = plist->get<double>("k", 0.14);
	  eps = plist->get<double>("eps", 0.0);
	  lambda = plist->get<double>("lambda", 10.);
	  d0 = plist->get<double>("d0", 5.e-9);
	  D_liquid = plist->get<double>("D_liquid", 3.e-9);
	  m = plist->get<double>("m", -2.6);
	  c_inf = plist->get<double>("c_inf", 3.);
	  
	  G = plist->get<double>("G", 3.e5);
	  R = plist->get<double>("R", 0.003);
	  base_height = plist->get<double>("base_height", 15.);
	  amplitude = plist->get<double>("amplitude", 0.2);
	  
	  //pp = plist->get<double>("pp");
	  //ll = plist->get<double>("ll");
	  //aa = plist->get<double>("aa");
	}


	//double ff(const double y)
	//{ 

	  //return 2. + sin(y*aa*M_PI/pp);
	//}


double a(const double &p,const double &px,const double &py,const double &pz)
{
  return (p*p < absphi) ? (1.-3.*eps)*(1.+4.*eps/(1.-3.*eps)*
				    (px*px*px*px+py*py*py*py+pz*pz*pz*pz)/(px*px+py*py+pz*pz)/(px*px+py*py+pz*pz))
    : 1. + eps;
}

double ap(const double &p,const double &px,const double &py,const double &pz,const double &pd)
{
  return (p*p < absphi) ? 4.*eps*
				    (4.*pd*pd*pd*(px*px+py*py+pz*pz)-4.*pd*(px*px*px*px+py*py*py*py+pz*pz*pz*pz))
				    /(px*px+py*py+pz*pz)/(px*px+py*py+pz*pz)/(px*px+py*py+pz*pz)
    : 0.;
}

RES_FUNC_TPETRA(residual_phase_farzadi_)
{
  //derivatives of the test function
  const double dtestdx = basis[eqn_id].dphidx[i];
  const double dtestdy = basis[eqn_id].dphidy[i];
  const double dtestdz = basis[eqn_id].dphidz[i];
  //test function
  const double test = basis[0].phi[i];
  //u, phi
  const double u = basis[0].uu;
  const double uold = basis[0].uuold;
  const double phi = basis[1].uu;
  const double phiold = basis[1].uuold;

  const double dphidx = basis[1].dudx;
  const double dphidy = basis[1].dudy;
  const double dphidz = basis[1].dudz;

  //double theta_ = theta(basis[1].duolddx,basis[1].duolddy);
  //double theta_ = cummins::theta(basis[1].dudx,basis[1].dudy);

  //double m = (1.+(1.-farzadi::k_)*u)*cummins::m_cummins_(theta_, farzadi::M_, eps);//cn we probably need u and uold here for CN...
  //double m = m_cummins_(theta_, M_, eps_);//cn we probably need u and uold here for CN...
  //double theta_old = theta(dphidx,dphidy);
  //double mold = (1+(1-k_)*uold)*m_cummins_(theta_old, M_, eps_);

  //double phit = (t_theta_*m+(1.-t_theta_)*mold)*(phi-phiold)/dt_*test;
  //double phit = m*(phi-phiold)/dt_*test;

  //double gs2 = cummins::gs2_cummins_(theta_, farzadi::M_, eps,0.);
  const double as = a(phi,dphidx,dphidy,dphidz);
  //const double as = 1.;
  //double gs2 = as*as;

  const double divgradphi = as*as*(dphidx*dtestdx + dphidy*dtestdy + dphidz*dtestdz);//(grad u,grad phi)

  const double phit = (1.+(1.-k)*u)*as*as*(phi-phiold)/dt_*test;

  //double dgdtheta = cummins::dgs2_2dtheta_cummins_(theta_, farzadi::M_, eps, 0.);	
  //double dgdpsi = 0.;
  //double curlgrad = -dgdtheta*dphidy*dtestdx + dgdtheta*dphidx*dtestdy;
  const double curlgrad = as*(dphidx*dphidx + dphidy*dphidy + dphidz*dphidz)
    *(ap(phi,dphidx,dphidy,dphidz,dphidx)*dtestdx + ap(phi,dphidx,dphidy,dphidz,dphidy)*dtestdy + ap(phi,dphidx,dphidy,dphidz,dphidz)*dtestdz);

  const double gp1 = -(phi - phi*phi*phi);
  const double phidel2 = gp1*test;

  const double x = basis[0].xx;
  
  
  // frozen temperature approximation: linear pulling of the temperature field
  double xx = x*w0;
  double tt = time*tau0;
  double t_scale = (xx-Vp0*tt)/l_T0;
  
  // need to plot t_scale
  //double t_scale = farzadi::tscale_(x,time);

  const double hp1 = lambda*(1. - phi*phi)*(1. - phi*phi)*(u+t_scale);
  const double phidel = hp1*test;
  
  const double rhs = divgradphi + curlgrad + phidel2 + phidel;
  
  //double val = phit + t_theta_*rhs;
  //printf("%lf\n",val);

  return phit + t_theta_*rhs;	// + (1.-t_theta_)*rhs_old*0.;
}

RES_FUNC_TPETRA(residual_conc_farzadi_)
{
  //right now, if explicit, we will have some problems with time derivates below
  const double dtestdx = basis[eqn_id].dphidx[i];
  const double dtestdy = basis[eqn_id].dphidy[i];
  const double dtestdz = basis[eqn_id].dphidz[i];
  const double test = basis[0].phi[i];
  const double u = basis[0].uu;
  const double uold = basis[0].uuold;
  const double phi = basis[1].uu;
  const double phiold = basis[1].uuold;
  const double dphidx = basis[1].dudx;
  const double dphidy = basis[1].dudy;
  const double dphidz = basis[1].dudz;

  const double ut = (1.+k)/2.*(u-uold)/dt_*test;
  const double divgradu = D_liquid_*(1.-phi)/2.*(basis[0].dudx*dtestdx + basis[0].dudy*dtestdy + basis[0].dudz*dtestdz);//(grad u,grad phi)

  const double normd = (phi*phi < absphi) ? 1./sqrt(dphidx*dphidx + dphidy*dphidy + dphidz*dphidz) : 0.; //cn lim grad phi/|grad phi| may -> 1 here?

  const double j_coef = (1.+(1.-k)*u)/sqrt(8.)*normd*(phi-phiold)/dt_;
  const double divj = j_coef*(dphidx*dtestdx + dphidy*dtestdy + dphidz*dtestdz);

  double phitu = -.5*(phi-phiold)/dt_*(1.+(1.-k)*u)*test; 
  
  //double val = ut + t_theta_*divgradu  + t_theta_*divj + t_theta_*phitu;
  //printf("%lf\n",val);

  return ut + t_theta_*divgradu  + t_theta_*divj + t_theta_*phitu;
}
//do not use now
PRE_FUNC_TPETRA(prec_phase_farzadi_)
{
  const double dtestdx = basis[eqn_id].dphidx[i];
  const double dtestdy = basis[eqn_id].dphidy[i];
  const double dtestdz = basis[eqn_id].dphidz[i];
  const double dbasisdx = basis[eqn_id].dphidx[j];
  const double dbasisdy = basis[eqn_id].dphidy[j];
  const double dbasisdz = basis[eqn_id].dphidz[j];

  const double test = basis[1].phi[i];
  
  const double dphidx = basis[1].dudx;
  const double dphidy = basis[1].dudy;
  const double dphidz = basis[1].dudz;

  const double u = basis[0].uu;
  const double phi = basis[1].uu;

  const double as = a(phi,dphidx,dphidy,dphidz);

  const double m = (1.+(1.-k)*u*0.)*as*as;
  const double phit = m*(basis[1].phi[j])/dt_*test;

  const double divgrad = as*as*(dbasisdx*dtestdx + dbasisdy*dtestdy + dbasisdz*dtestdz);

  return phit + t_theta_*(divgrad);
}
//do not use now
PRE_FUNC_TPETRA(prec_conc_farzadi_)
{
  const double dtestdx = basis[eqn_id].dphidx[i];
  const double dtestdy = basis[eqn_id].dphidy[i];
  const double dtestdz = basis[eqn_id].dphidz[i];
  const double dbasisdx = basis[eqn_id].dphidx[j];
  const double dbasisdy = basis[eqn_id].dphidy[j];
  const double dbasisdz = basis[eqn_id].dphidz[j];

  const double test = basis[0].phi[i];
  const double divgrad = D_liquid_*(1.-basis[1].uu)/2.*(dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz);

  const double u_t =(1.+k)/2.*test * basis[0].phi[j]/dt_;

  return u_t + t_theta_*(divgrad);

}

INI_FUNC(init_phase_farzadi_)
{
  //double r = ll*(1.+ff(y)*ff(y/2.)*ff(y/4.));
  //double rz = (1.+ff(z)*ff(z/2.)*ff(z/4.))/9.;
  //return (x < r*rz) ? 1. : -1.;
  
  double r = base_height + amplitude*((double)rand()/(RAND_MAX));
  return (tanh((r-x)/sqrt(2.)));	

}

INI_FUNC(init_conc_farzadi_)
{	
  return -1.;
}
}//namespace farzadi3d
}//namespace tpetra









namespace puga
{

RES_FUNC(residual_)
{
  return 0.;
}

PRE_FUNC(prec_)
{
  return 0.;
}

INI_FUNC(init_)
{
  return 0.;
}

PARAM_FUNC(param_)
{
  //delta_ = plist->get<double>("delta");
}

}//namespace puga

#endif
