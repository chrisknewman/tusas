//cummins
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

  double g = 1. + eps * (cos(M*(theta)));
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
  return 1./13.47;
}
double hp2_furtado_(const double &phi)
{
  double dH = 2.35e9;
  double rho = 0.37;
  double Cp = 5.42e6;
  return dH/rho/Cp*30.*phi*phi*(1.-phi)*(1.-phi);
}
double rand_phi_furtado_(const double &phi, const double &random_number)
{
  double a = .025;
//   return ((double)rand()/(RAND_MAX)*2.-1.)*16.*a*phi*phi
// 		*(1.-phi)*(1.-phi);
//   return random_number*16.*a*phi*phi
// 		*(1.-phi)*(1.-phi);
  return 0.;
}
