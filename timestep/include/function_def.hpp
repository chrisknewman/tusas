#ifndef FUNCTION_DEF_HPP
#define FUNCTION_DEF_HPP
#include <boost/ptr_container/ptr_vector.hpp>
#include "basis.hpp"

//cummins
double gs_cummins_(const double &theta, const double &M, const double &eps, const double &psi)
{
  double eps_0_ = 1.;
  double g = eps_0_*(4.*eps*(cos(theta)*cos(theta)*cos(theta)*cos(theta) 
			     + sin(theta)*sin(theta)*sin(theta)*sin(theta) 
			     *(1.-2.*sin(psi)*sin(psi)*cos(psi)*cos(psi))
			     ) -3.*eps +1.   );
  return g;

}
double hp1_cummins_(const double &phi,const double &c)
{
  return -c*phi*phi*(1.-phi)*(1.-phi);
}
double hpp1_cummins_(const double &phi,const double &c)
{
  return -c* 2.* (1. -phi) *phi* (1. - 2. *phi);
}
double w_cummins_(const double &delta)
{
  return 1./delta/delta;
}
double m_cummins_(const double &theta,const double &M,const double &eps)
{

  //double g = 1. + eps * (cos(M*(theta)));
  double g = (4.*eps*(cos(theta)*cos(theta)*cos(theta)*cos(theta) 
			     + sin(theta)*sin(theta)*sin(theta)*sin(theta) ) -3.*eps +1.   );
  return g*g;
}
double rand_phi_zero_(const double &phi, const double &random_number)
{
  return 0.;
}
double gp1_cummins_(const double &phi)
{
  return phi*(1.-phi)*(1.-2.*phi);
}
double gpp1_cummins_(const double &phi)
{
  return 1. - 6.* phi + 6.* phi*phi;
}
double hp2_cummins_(const double &phi)
{
  return 1.;
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

//cn some experimental stuff here

//double diffusivity_heat_(const double &theta, const double &M, const double &eps, const double &psi)
double residual_heat_(const boost::ptr_vector<Basis> &basis, 
		      const int &i, 
		      const double &dt_, 
		      const double &t_theta_, 
		      const double &delta, 
		      const double &time)
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
  if(abs(x) < small && y > 0. ) t = pi/2.;
  if(abs(x) < small && y < 0. ) t = 3.*pi/2.;
  if(x > small && y >= 0.) t= atan(n/x);
  if(x > small && y <0.) t= atan(n/x) + 2.*pi;
  if(x < -small) t= atan(n/x)+ pi;

  return t;
}
double residual_phase_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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
  double w = w_cummins_(delta);
  double gp1 = gp1_cummins_(phi);
  double phidel2 = gp1*w*test;

  double T_m_ = 1.55;
  double T_inf_ = 1.;
  double alpha_ = 191.82;

  double hp1 = hp1_cummins_(phi,5.*alpha_/delta);
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
  
  hp1 = hp1_cummins_(phiold,5.*alpha_/delta);

  phidel = hp1*(T_m_ - uold)*test;
	      
  double rhs_old = divgradphi + curlgrad + phidel2 + phidel;

  return phit + t_theta_*rhs + (1.-t_theta_)*rhs_old;

}
double prec_heat_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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
double prec_phase_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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
double init_heat_(const double &x,
			 const double &y,
			 const double &z)
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
double init_heat_const_(const double &x,
			 const double &y,
			 const double &z)
{

  double T_inf_ = 1.;

  return T_inf_;
}
double init_phase_(const double &x,
			 const double &y,
			 const double &z)
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
double residual_heat_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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
  double u = basis[0].uu;
  double uold = basis[0].uuold;

  double ut = (u-uold)/dt_*test;
  double divgradu = (basis[0].dudx*dtestdx + basis[0].dudy*dtestdy + basis[0].dudz*dtestdz);//(grad u,grad phi)
  double divgradu_old = (basis[0].duolddx*dtestdx + basis[0].duolddy*dtestdy + basis[0].duolddz*dtestdz);//(grad u,grad phi)
 
 
  return ut + t_theta_*divgradu + (1.-t_theta_)*divgradu_old;
}
double prec_heat_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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
  double u_t =test * basis[0].phi[j]/dt_;
  return u_t + t_theta_*divgrad;
}
double init_heat_test_(const double &x,
			 const double &y,
			 const double &z)
{

  double pi = 3.141592653589793;

  return sin(pi*x)*sin(pi*y);
}
double init_zero_(const double &x,
			 const double &y,
			 const double &z)
{

  return 0.;
}
double nbc_zero_(const Basis *basis,
		 const int &i, 
		 const double &dt_, 
		 const double &t_theta_,
		 const double &time)
{
  
  double phi = basis->phi[i];
  
  return 0.*phi;
}
double dbc_zero_(const double &x,
		const double &y,
		const double &z,
		const double &t)
{
  
  
  return 0.;
}
double nbc_one_(const Basis *basis,
		const int &i, 
		const double &dt_, 
		const double &t_theta_,
		const double &time)
{
  
  double phi = basis->phi[i];
  
  return 1.*phi;
}
double dbc_one_(const double &x,
	       const double &y,
	       const double &z,
	       const double &t)
{
  
  
  return 1.;
}
double dbc_ten_(const double &x,
	       const double &y,
	       const double &z,
	       const double &t)
{
  
  
  return 10.*dbc_one_(x,
	       y,
	       z,
	       t);
}
double nbc_mone_(const Basis *basis,
		 const int &i, 
		 const double &dt_, 
		 const double &t_theta_,
		 const double &time)
{

  double phi = basis->phi[i];

  return -1.*phi;
}
double dbc_mone_(const double &x,
		const double &y,
		const double &z,
		const double &t)
{


  return -1.;
}

double postproc_null_(const double &u)
{
  return u;
}


double init_neumann_test_(const double &x,
			 const double &y,
			 const double &z)
{

  double pi = 3.141592653589793;

  return sin(pi*x);
}








//farzadi
double init_conc_farzadi_(const double &x,
			 const double &y,
			 const double &z)
{

  double val = -1.;

  return val;
}
double init_phase_farzadi_(const double &x,
			 const double &y,
			 const double &z)
{
  //cn something wierd with greater than 8 procs here....
  double pi = 3.141592653589793;
  double theta_0_ = 0.;

  double phi_sol_ = 1.;
  double phi_liq_ = -1.;

  double val = phi_liq_;  

  double r = 10.+ 10.*abs(sin(y*7.*pi/636.408));//cn this will be a perturbation

  if(x < r){
    val=phi_sol_;
  }
  else {
    val=phi_liq_;
  }
  return val;
  //return 1.;
}
double residual_phase_farzadi_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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

  double k =0.14;
  double eps_ = .01;
  double M_= 4.;
  double lambda = 10.;

  double dphidx = basis[1].dudx;
  double dphidy = basis[1].dudy;

  double theta_ = theta(basis[1].duolddx,basis[1].duolddy);

  double m = (1+(1-k)*u)*m_cummins_(theta_, M_, eps_);//cn we probably need u and uold here for CN...
  //double m = m_cummins_(theta_, M_, eps_);//cn we probably need u and uold here for CN...
  double theta_old = theta(dphidx,dphidy);
  double mold = (1+(1-k)*uold)*m_cummins_(theta_old, M_, eps_);

  double phit = (t_theta_*m+(1.-t_theta_)*mold)*(phi-phiold)/dt_*test;
  //phit = (phi-phiold)/dt_*test;
  double gs2 = gs2_cummins_(theta_, M_, eps_,0.);
  double divgradphi = gs2*(dphidx*dtestdx + dphidy*dtestdy);//(grad u,grad phi)

  double dgdtheta = dgs2_2dtheta_cummins_(theta_, M_, eps_, 0.);	
  double dgdpsi = 0.;
  double curlgrad = dgdtheta*(-dphidy*dtestdx + dphidx*dtestdy);//farzadi paper has the sign changed here...

  double gp1 = -(phi - phi*phi*phi);
  double phidel2 = gp1*test;

  //double m = -2.6;
  //double c_inf = 3.;
  double tl = 925.2;
  double ts = 877.3;
  double G = 290900.;
  double R = .003;
  double V = .003;
  double t0 = 0.;
  double x = basis[0].xx;//cn this is in x space, maybe it should be gauss pt?
  double t = tl + G*(x-V*time);
  t=900.;
  double t_scale = (t-ts)/(tl-ts);

  double hp1 = lambda*(1. - phi*phi)*(1. - phi*phi)*(u+t_scale);
  double phidel = hp1*test;
  //phidel = 0.;
  double rhs = divgradphi + curlgrad + phidel2 + phidel;
  rhs = divgradphi + phidel;
  dphidx = basis[1].duolddx;
  dphidy = basis[1].duolddy;
  theta_ = theta(dphidx,dphidy);

  gs2 = gs2_cummins_(theta_, M_, eps_,0.);
  divgradphi = gs2*dphidx*dtestdx + gs2*dphidy*dtestdy;//(grad u,grad phi)
  dgdtheta = dgs2_2dtheta_cummins_(theta_, M_, eps_, 0.);

  curlgrad = dgdtheta*(-dphidy*dtestdx + dphidx*dtestdy);

  gp1 = -(phiold-phiold*phiold*phiold);
  
  phidel2 = gp1*test;
  
  hp1 = lambda*(1.-phiold*phiold)*(1.-phiold*phiold)*(uold+t_scale);

  phidel = hp1*test;
	      
  double rhs_old = divgradphi + curlgrad + phidel2 + phidel;

  return dt_*phit + dt_*t_theta_*rhs;// + (1.-t_theta_)*rhs_old*0.;

}

double residual_conc_farzadi_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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

  double k =0.14;
  //double D_ = .6267/.8839*10.*10.;
  double D_ = .6267*10.;//a_2*lambda

  double ut = (1.+k)/2.*(u-uold)/dt_*test;
  //ut = (u-uold)/dt_*test;
  double divgradu = D_*(1.-phi)/2.*(basis[0].dudx*dtestdx + basis[0].dudy*dtestdy);//(grad u,grad phi)
  //divgradu = (basis[0].dudx*dtestdx + basis[0].dudy*dtestdy);
  double divgradu_old = D_*(1.-phiold)/2*(basis[0].duolddx*dtestdx + basis[0].duolddy*dtestdy);//(grad u,grad phi)

  //j is antitrapping current
  // j grad test here... j1*dtestdx + j2*dtestdy 
  // what if dphidx*dphidx + dphidy*dphidy = 0?

  double norm = sqrt(dphidx*dphidx + dphidy*dphidy);
  
  double j_coef = 0.;
  if (0 < norm) {
    j_coef = (1.+(1.-k)*u)/sqrt(8.)/norm*(phi-phiold)/dt_;
  } 
  //j_coef = 0.;
  double j1 = j_coef*dphidx;
  double j2 = j_coef*dphidy;
  double divj = j1*dtestdx + j2*dtestdy;

  double dphiolddx = basis[1].duolddx;
  double dphiolddy = basis[1].duolddy; 
  norm = sqrt(dphidx*dphidx + dphidy*dphidy);
  j_coef = 0.;
  if (1.e-6 < norm) {
    j_coef = (1.+(1.-k)*uold)/sqrt(8.)/norm*(phiold-basis[1].uuoldold)/dt_; 
  }
  j1 = j_coef*dphidx;
  j2 = j_coef*dphidy;
  //j_coef = 0.;
  double divj_old = j1 *dtestdx + j2 *dtestdy;

  double h = phi*(1+(1.-k)*u);
  double hold = phiold + (1.-k)*uold;

  double phitu = -.5*(h-hold)/dt_*test; 
  //phitu = 1.*test; 
  h = hold;
  hold = basis[1].uuoldold*(1. + (1.-k)*basis[0].uuoldold);
  double phitu_old = -.5*(h-hold)/dt_*test;
 
  //return ut*0.  + t_theta_*(divgradu + divj*0.) + (1.-t_theta_)*(divgradu_old + divj_old)*0. + t_theta_*phitu*0. + (1.-t_theta_)*phitu_old*0.;

  return dt_*ut + dt_*t_theta_*divgradu  + 0.*dt_*t_theta_*divj + dt_*t_theta_*phitu;
}
double residual_c_farzadi_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
{
  double u = basis[0].uu;
  double phi = basis[1].uu;
  double c = basis[2].uu;
  double test = basis[2].phi[i];
  double k = 0.14;
  double c_inf = 3.;

  return dt_*(2.*k*c+c_inf*(1.+k-phi+k*phi)*(-1-u+k*u))*test;
}
double prec_phase_farzadi_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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

  double k =0.14;
  double eps_ = .01;
  double M_= 4.;

  double theta_0_ =0.;
  double theta_ = theta(dphidx,dphidy)-theta_0_;

  double m = t_theta_*(1.+(1.-k)*basis[0].phi[j])*m_cummins_(theta_, M_, eps_);
  //double m = t_theta_*m_cummins_(theta_, M_, eps_);

  double phit = m*(basis[1].phi[j])/dt_*test;
  //phit = (basis[1].phi[j])/dt_*test;
  double gs2 = gs2_cummins_(theta_, M_, eps_,0.);

  double divgrad = gs2*dbasisdx * dtestdx + gs2*dbasisdy * dtestdy + gs2*dbasisdz * dtestdz;

  double dg2 = dgs2_2dtheta_cummins_(theta_, M_, eps_,0.);
  double curlgrad = -dg2*(dtestdy*dphidx -dtestdx*dphidy);

  //return phit + t_theta_*divgrad + t_theta_*curlgrad;

  return dt_*phit + dt_*t_theta_*divgrad;
}
double prec_conc_farzadi_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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
  double k =0.14;
  //double D_ = .6267/.8839*10.*10.;
  double D_ = .6267*10.;
  double divgrad = D_*(1.-basis[1].phi[j])/2.*(dbasisdx * dtestdx + dbasisdy * dtestdy);
  //double divgrad = D_/2.*(dbasisdx * dtestdx + dbasisdy * dtestdy);
  double u_t =(1.+k)/2.*test * basis[0].phi[j]/dt_;
  return dt_*u_t + dt_*t_theta_*divgrad;
}
double prec_c_farzadi_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
{
  double test = basis[2].phi[i];
  double k = 0.14;
  double val =0.;
  //val = 2.*k*basis[2].phi[j]*test;
  if(i == j ) val=1.;
  return val;
}


double residual_robin_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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

double prec_robin_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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

double nbc_robin_test_(const Basis *basis,
		 const int &i, 
		 const double &dt_, 
		 const double &t_theta_,
		 const double &time)
{

  double test = basis->phi[i];
  double u = basis[0].uu;

  double b =1.;

  return b*(1.-u)*test;
}

namespace liniso
{
double residual_liniso_x_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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
double residual_liniso_y_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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
double residual_liniso_z_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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



double prec_liniso_x_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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
double prec_liniso_y_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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
double prec_liniso_z_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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
double residual_linisobodyforce_y_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
{
  //this is taken from a test that had E=2e11; nu=.3 and body force -1e10 in the y direction;
  //we reuse the test case that has E=1e6 and scale the body force accordingly by 5e-6  

  //test function
  double test = basis[0].phi[i]; 

  double bf = -1.e10*5.e-6;

  double divgradu = residual_liniso_y_test_(basis,i,dt_,t_theta_,delta,time) + bf*test;
 
  return divgradu;
}

double residual_linisoheat_x_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
{
  double dtestdx = basis[0].dphidxi[i]*basis[0].dxidx
    +basis[0].dphideta[i]*basis[0].detadx
    +basis[0].dphidzta[i]*basis[0].dztadx;
  double gradu = basis[3].dudx;
  double c = 1.e-6;
  double alpha = 1.e-4;;
  double E = 1.;

  double divgradu = c*residual_liniso_x_test_(basis,i,dt_,t_theta_,delta,time) - alpha*E*gradu*dtestdx;
 
  return divgradu;
}


double residual_linisoheat_y_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
{
  //test function
  double dtestdy = basis[0].dphidxi[i]*basis[0].dxidy
    +basis[0].dphideta[i]*basis[0].detady
    +basis[0].dphidzta[i]*basis[0].dztady;
  double gradu = basis[3].dudy;

  double c = 1.e-6;
  double alpha = 1.e-4;
  double E = 1.;


  double divgradu = c*residual_liniso_y_test_(basis,i,dt_,t_theta_,delta,time) - alpha*E*gradu*dtestdy;
 
  return divgradu;
}



double residual_linisoheat_z_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
{
  double dtestdz = basis[0].dphidxi[i]*basis[0].dxidz
    +basis[0].dphideta[i]*basis[0].detadz
    +basis[0].dphidzta[i]*basis[0].dztadz;
  double gradu = basis[3].dudz;
  double c = 1.e-6;
  double alpha = 1.e-4;
  double E = 1.;

  double divgradu = c*residual_liniso_z_test_(basis,i,dt_,t_theta_,delta,time) - alpha*E*gradu*dtestdz;
 
  return divgradu;
}



double residual_divgrad_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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
  double a = 10.e-1;//m^4
  double rho = 1.e3;//kg/m^3
  double c = 5.e2;//J/kg/K
  double k = 150.;//W/m/K
  //double k = 1.5;//W/m/K
  double r0 = 29.5;

  //double E = 200.e9;//GPa
  double E = 200.;//GPa
  double nu = .3;

  //plane stress
//   double c = E/(1.-nu*nu);
//   double c1 = c;
//   double c2 = c*nu;
//   double c3 = c*(1.-nu)/2.;//always confused about the 2 here
  //plane strain
  double c0 = E/(1.+nu)/(1.-2.*nu);
  double c1 = c0*(1.-nu);
  double c2 = c0*nu;
  double c3 = c0*(1.-2.*nu)/2.;

  double alpha = 5.e-6;//1/K
  double beta = 1.5e-3;
  
double init_heat_(const double &x,
		   const double &y,
		   const double &z)
{
  double val = 400.;  

  return val;
}
double init_heat_1_(const double &x,
		   const double &y,
		   const double &z)
{
  double phi_sol_ = 300.;
  double phi_liq_ = 400.;

  double val = phi_sol_ ;  

  double dx = 1.e-2;
  double r = r0*dx;

  //if(y > r){
  if(x > r && y > r){
    val=phi_sol_;
  }
  else {
    val=phi_liq_;
  }
  return val;
}
double init_phase_(const double &x,
		   const double &y,
		   const double &z)
{
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
double dbc_(const double &x,
	       const double &y,
	       const double &z,
	       const double &t)
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
double residual_phase_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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
  double M = 200000000.*b*phi*(1. - phi)*(L*(um - u)/um + f);
  double g = -phi*(1. - phi)*(phi - .5 + M)*test;
  double rhs = divgradphi + g;

  return (phit + t_theta_*rhs)/m;

}
double residual_heat_(const boost::ptr_vector<Basis> &basis, 
		      const int &i, 
		      const double &dt_, 
		      const double &t_theta_, 
		      const double &delta, 
		      const double &time)
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
  double u = basis[1].uu;
  double uold = basis[1].uuold;

  double dudx = basis[1].dudx;
  double dudy = basis[1].dudy;
  double dudz = basis[1].dudz;

  double ut = rho*c*(u-uold)/dt_*test;
  double divgradu = k*(dudx*dtestdx + dudy*dtestdy + dudz*dtestdz);
  double phitu = -30.*L*phi*phi*(1.-phi)*(1.-phi)*(phi-phiold)/dt_*test; 
  double rhs = divgradu + phitu;

  return (ut + t_theta_*rhs)/rho/c;

}
double residual_liniso_x_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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

  double ut = (basis[1].uu - basis[1].uuold)/dt_;//thermal strain

  double phi = basis[0].uu;
  double strain_phi = 30.*beta*phi*phi*(1.-phi)*(1.-phi)*(phi-basis[0].uuold)/dt_;

  strain[0] = (basis[2].dudx-basis[2].duolddx)/dt_ - alpha*ut - strain_phi;
  strain[1] = (basis[3].dudy-basis[3].duolddy)/dt_ - alpha*ut - strain_phi;
  strain[2] = (basis[2].dudy-basis[2].duolddy + basis[3].dudx-basis[3].duolddx)/dt_ - alpha*ut - strain_phi;

  stress[0] = c1*strain[0] + c2*strain[1];
  // stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

  double divgradu = stress[0]*dtestdx + stress[2]*dtestdy;//(grad u,grad phi)
 
  //std::cout<<"residual_liniso_x_test_"<<std::endl;
 
  return divgradu;
}
double residual_liniso_y_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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

  //strain[0] = basis[2].dudx;
  //strain[1] = basis[3].dudy;
  //strain[2] = basis[2].dudy + basis[3].dudx;

  double ut = (basis[1].uu - basis[1].uuold)/dt_;//thermal strain

  double phi = basis[0].uu;
  double strain_phi = 30.*beta*phi*phi*(1.-phi)*(1.-phi)*(phi-basis[0].uuold)/dt_;
  
  strain[0] = (basis[2].dudx-basis[2].duolddx)/dt_ - alpha*ut - strain_phi;
  strain[1] = (basis[3].dudy-basis[3].duolddy)/dt_ - alpha*ut - strain_phi;
  strain[2] = (basis[2].dudy-basis[2].duolddy + basis[3].dudx-basis[3].duolddx)/dt_ - alpha*ut - strain_phi;

  stress[0] = c1*strain[0] + c2*strain[1];
  stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

  double divgradu = stress[1]*dtestdy + stress[2]*dtestdx;//(grad u,grad phi)
  
  //std::cout<<"residual_liniso_y_test_"<<std::endl;

  return divgradu;
}
double residual_stress_x_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double strain[3], stress[3];//x,y,yx

  //test function
  double test = basis[0].phi[i];
  //u, phi
  double sx = basis[4].uu;

  strain[0] = basis[2].dudx;
  strain[1] = basis[3].dudy;
  //strain[2] = basis[0].dudy + basis[1].dudx;

  stress[0] = c1*strain[0] + c2*strain[1];
  // stress[1] = c2*strain[0] + c1*strain[1];
  //stress[2] = c3*strain[2];

  return (sx - stress[0])*test;
}
double residual_stress_y_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double strain[3], stress[3];//x,y,z,yx,zy,zx

  double test = basis[0].phi[i];
  //u, phi
  double sy = basis[5].uu;

  strain[0] = basis[2].dudx;
  strain[1] = basis[3].dudy;
  //strain[2] = basis[0].dudy + basis[1].dudx;

  //stress[0] = c1*strain[0] + c2*strain[1];
  stress[1] = c2*strain[0] + c1*strain[1];
  //stress[2] = c3*strain[2];

  return (sy - stress[1])*test;//(grad u,grad phi)
}
double residual_stress_xy_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
{
  //3-D isotropic x-displacement based solid mech, steady state
  //strong form: sigma = stress  eps = strain
  // d^T sigma = d^T B D eps == 0

  double strain[3], stress[3];//x,y,z,yx,zy,zx

  double test = basis[0].phi[i];
  //u, phi
  double sxy = basis[6].uu;

  //strain[0] = basis[0].dudx;
  //strain[1] = basis[1].dudy;
  strain[2] = basis[2].dudy + basis[3].dudx;

  //stress[0] = c1*strain[0] + c2*strain[1];
  //stress[1] = c2*strain[0] + c1*strain[1];
  stress[2] = c3*strain[2];

  return (sxy - stress[2])*test;//(grad u,grad phi)
}
double prec_phase_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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
double prec_heat_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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

  double divgrad = k*(dbasisdx * dtestdx + dbasisdy * dtestdy + dbasisdz * dtestdz);
  double u_t =rho*c*basis[1].phi[j]/dt_*test;
  return (u_t + t_theta_*divgrad)/rho/c;
}
double prec_liniso_x_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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

  double divgradu = stress[0]*dtestdx + stress[2]*dtestdy;//(grad u,grad phi)
 
  return divgradu;
}
double prec_liniso_y_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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


  double divgradu = stress[1]*dtestdy + stress[2]*dtestdx;//(grad u,grad phi)

  return divgradu;
}
double prec_stress_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
{
  double test = basis[0].phi[i];

  return test * basis[0].phi[j];
}

}//namespace uehara










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

double residual_liniso_x_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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
double residual_liniso_y_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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
double residual_stress_x_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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
double residual_stress_y_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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
double residual_stress_xy_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const double &dt_, const double &t_theta_, const double &delta, 
		      const double &time)
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
double prec_liniso_x_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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
double prec_liniso_y_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
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
double prec_stress_test_(const boost::ptr_vector<Basis> &basis, 
			 const int &i, const int &j, const double &dt_, const double &t_theta_, const double &delta)
{
  double test = basis[0].phi[i];

  return test * basis[0].phi[j];
}


}//namespace coupledstress
#endif
