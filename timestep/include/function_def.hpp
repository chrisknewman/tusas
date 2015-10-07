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
			 const int &i, const double &dt_, const double &t_theta_, const double &delta)
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
			 const int &i, const double &dt_, const double &t_theta_, const double &delta)
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
  double curlgrad = dgdtheta*(-dphidy*dtestdx + dphidx*dtestdy);
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
    //(*u)[numeqs_*nn+1]=phi_sol_;
  }
  else {
    val=T_inf_;
    //(*u)[numeqs_*nn+1]=phi_liq_;
  }
  return val;
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
			 const int &i, const double &dt_, const double &t_theta_, const double &delta)
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
double dirichlet_zero_(const double &x,
			 const double &y,
			 const double &z)
{


  return 0.;
}

#endif
