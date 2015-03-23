double hp1_cummins_(const double &phi,const double &c)
{
  return -c*phi*phi*(1.-phi)*(1.-phi);
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
double rand_phi_cummins_(const double &phi)
{
  return 0.;
}
double rand_phi_furtado_(const double &phi)
{
  double a = .01;
  return ((double)rand()/(RAND_MAX)*2.-1.)*16.*a*phi*phi
		*(1.-phi)*(1.-phi);
}
double gp1_cummins_(const double &phi)
{
  return phi*(1.-phi)*(1.-2.*phi);
}
double hp2_cummins_(const double &phi)
{
  return 1.;
}
