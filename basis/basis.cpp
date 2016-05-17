#include <cmath>
#include <iostream>

#include "basis.hpp"

void Basis::setN(int N, double *abscissa, double *weight){

 if ( N == 2 ) {
    abscissa[0] = -1.0L/sqrt(3.0L);
    abscissa[1] =  1.0L/sqrt(3.0L);
    weight[0] = 1.0;
    weight[1] = 1.0;
  } else if ( N == 3 ) {
    abscissa[0] = -sqrt(15.0L)/5.0L;
    abscissa[1] =  0.0;
    abscissa[2] =  sqrt(15.0L)/5.0L;
    weight[0] = 5.0L/9.0L;
    weight[1] = 8.0L/9.0L;
    weight[2] = 5.0L/9.0L;
  } else if ( N == 4 ) {
    abscissa[0] =  -sqrt(525.0L+70.0L*sqrt(30.0L))/35.0L;
    abscissa[1] =  -sqrt(525.0L-70.0L*sqrt(30.0L))/35.0L;
    abscissa[2] =   sqrt(525.0L-70.0L*sqrt(30.0L))/35.0L;
    abscissa[3] =   sqrt(525.0L+70.0L*sqrt(30.0L))/35.0L;
    weight[0] = (18.0L-sqrt(30.0L))/36.0L;
    weight[1] = (18.0L+sqrt(30.0L))/36.0L;
    weight[2] = (18.0L+sqrt(30.0L))/36.0L;
    weight[3] = (18.0L-sqrt(30.0L))/36.0L;
  } else if ( N == 5 ) {
    abscissa[0] = -sqrt(245.0L+14.0L*sqrt(70.0L))/21.0L;
    abscissa[1] = -sqrt(245.0L-14.0L*sqrt(70.0L))/21.0L;
    abscissa[2] = 0.0;
    abscissa[3] = sqrt(245.0L-14.0L*sqrt(70.0L))/21.0L;
    abscissa[4] = sqrt(245.0L+14.0L*sqrt(70.0L))/21.0L;
    weight[0] = (322.0L-13.0L*sqrt(70.0L))/900.0L;
    weight[1] = (322.0L+13.0L*sqrt(70.0L))/900.0L;
    weight[2] = 128.0L/225.0L;
    weight[3] = (322.0L+13.0L*sqrt(70.0L))/900.0L;
    weight[4] = (322.0L-13.0L*sqrt(70.0L))/900.0L;
  } else {
    std::cout<<"WARNING: only 1 < N < 6 gauss points supported at this time, defaulting to N = 2"<<std::endl;
    abscissa[0] = -1.0L/sqrt(3.0L);
    abscissa[1] =  1.0L/sqrt(3.0L);
    weight[0] = 1.0;
    weight[1] = 1.0;
  }
}

// Constructor

BasisLTri::BasisLTri(int n){
  ngp = n;
  phi = new double[3];
  dphidxi = new double[3];
  dphideta = new double[3];
  dphidzta = new double[3];
  dphidx = new double[3];
  dphidy = new double[3];
  dphidz = new double[3];
  abscissa = new double[ngp];//number guass pts
  weight = new double[ngp];
  if( ngp == 1 ){
    abscissa[0] = 1.0L / 3.0L;
    weight[0] = .5;
  }else if (ngp == 3 ) {
    abscissa[0] = 1.0L / 2.0L;
    abscissa[1] = 1.0L / 2.0L;
    abscissa[2] = 0.;
    weight[0] = 1.0L / 6.0L;
    weight[1] = 1.0L / 6.0L;
    weight[2] = 1.0L / 6.0L;
  }else {
    std::cout<<"void BasisLTri::getBasis(int gp, double *x, double *y, double *u, double *uold)"<<std::endl
	     <<"   only ngp = 1 supported at this time"<<std::endl;
  }
}

// Destructor
BasisLTri::~BasisLTri() {
  delete [] phi;
  delete [] dphidxi;
  delete [] dphideta;
  delete [] dphidzta;
  delete [] dphidx;
  delete [] dphidy;
  delete [] dphidz;
  delete [] abscissa;
  delete [] weight;
}


void BasisLTri::getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold){

  int N = 3;  
  // gp irrelevent and unused

// one gauss point at center
  if ( 1 == ngp ){
    xi = eta = abscissa[0];
    wt = weight[0];
  }
  if ( 3 == ngp ){
    if(0 == gp){
      xi = abscissa[0];
      eta = abscissa[0];
      wt = weight[0];
    }else if (1 == gp){
      xi = abscissa[0];
      eta = abscissa[2];
      wt = weight[1];
    }else if (2 == gp){
      xi = abscissa[2];
      eta = abscissa[0];
      wt = weight[2];
    }
  }

  // Calculate basis function and derivatives at nodel pts
  phi[0]=(1.0-xi-eta);
  phi[1]= xi;
  phi[2]= eta;
  dphidxi[0]=-1.0;
  dphidxi[1]=1.0;
  dphidxi[2]=0.0;
  dphideta[0]=-1.0;
  dphideta[1]=0.0;
  dphideta[2]=1.0;
  dphidzta[0]=0.0;
  dphidzta[1]=0.0;
  dphidzta[2]=0.0;
  
  // Caculate basis function and derivative at GP.

  double dxdxi = 0;
  double dxdeta = 0;
  double dydxi = 0;
  double dydeta = 0;

  for (int i=0; i < N; i++) {
    dxdxi += dphidxi[i] * x[i];
    dxdeta += dphideta[i] * x[i];
    dydxi += dphidxi[i] * y[i];
    dydeta += dphideta[i] * y[i];
  }

  jac = dxdxi * dydeta - dxdeta * dydxi;

  dxidx = dydeta / jac;
  dxidy = -dxdeta / jac;
  dxidz = 0.;
  detadx = -dydxi / jac;
  detady = dxdxi / jac;
  detadz = 0.;
  dztadx = 0.;
  dztady = 0.;
  dztadz = 0.;

  xx=0.0;
  yy=0.0;
  zz=0.;
  uu=0.0;
  dudx=0.0;
  dudy=0.0;
  dudz=0.0;
  uuold = 0.;
  uuoldold = 0.;
  duolddx = 0.;
  duolddy = 0.;
  duolddz = 0.;
  duoldolddx = 0.;
  duoldolddy = 0.;
  duoldolddz = 0.;
  // x[i] is a vector of node coords, x(j, k) 
  for (int i=0; i < N; i++) {
    xx += x[i] * phi[i];
    yy += y[i] * phi[i];
    dphidx[i] = dphidxi[i]*dxidx+dphideta[i]*detadx;
    dphidy[i] = dphidxi[i]*dxidy+dphideta[i]*detady;
    dphidz[i] = 0.;
    if( u ){
      uu += u[i] * phi[i];
      //      dudx += u[i] * (dphidxi[i]*dxidx+dphideta[i]*detadx);
      dudx += u[i] * dphidx[i];
      dudy += u[i]* dphidy[i];
    }
    if( uold ){
      uuold += uold[i] * phi[i];
      duolddx += uold[i] * dphidx[i];
      duolddy += uold[i]* dphidy[i];
    }
    if( uoldold ){
      uuoldold += uoldold[i] * phi[i];
      duoldolddx += uoldold[i] * dphidx[i];
      duoldolddy += uoldold[i]* dphidy[i];
    }
  }

  return;
}


// Constructor
BasisLQuad::BasisLQuad(int n) :sngp(n){
  //sngp = 2;// number of Gauss points
  if ( 2 != sngp ){
    std::cout<<"BasisLQuad only supported for n = 2 at this time"<<std::endl<<std::endl<<std::endl;
    exit(0);
  }
  ngp = sngp*sngp;
  phi = new double[4];//number of nodes
  dphidxi = new double[4];
  dphideta = new double[4];
  dphidzta = new double[4];
  dphidx = new double[4];
  dphidy = new double[4];
  dphidz = new double[4];
  abscissa = new double[sngp];//number guass pts
  weight = new double[sngp];
  setN(sngp, abscissa, weight);

}

// Destructor
BasisLQuad::~BasisLQuad() {
  delete [] phi;
  delete [] dphidxi;
  delete [] dphideta;
  delete [] dphidzta;
  delete [] dphidx;
  delete [] dphidy;
  delete [] dphidz;
  delete [] abscissa;
  delete [] weight;
}

void BasisLQuad::getBasis(const int gp,const  double *x, const  double *y,  const double *z,const  double *u,const  double *uold,const  double *uoldold) {

  //cn this is horrible, need a better way than having if statements in here
  if(4 == ngp){
    if(0 == gp){
      xi = abscissa[0];
      eta = abscissa[0];
      wt = weight[0] * weight[0];
    }else if (1 == gp){
      xi = abscissa[1];
      eta = abscissa[0];
      wt = weight[0] * weight[1];
    }else if (2 == gp){
      xi = abscissa[1];
      eta = abscissa[1];
      wt = weight[1] * weight[1];
    }else if (3 == gp){
      xi = abscissa[0];
      eta = abscissa[1];
      wt = weight[0] * weight[1];
    }
  }
  if(9 == ngp){ 
    double ab[3];// = new double[3];
    ab[0] = abscissa[0];
    ab[1] = abscissa[2];
    ab[2] = abscissa[1];
    
    double w[3];// = new double[3];
    w[0] = weight[0];
    w[1] = weight[2];
    w[2] = weight[1];
    switch (gp){
    case 0:
      xi = ab[0];
      eta = ab[0];
      wt = w[0] * w[0];
      break;
    case 1:
      xi = ab[1];
      eta = ab[0];
      wt = w[1] * w[0];
      break;
    case 2:
      xi = ab[1];
      eta = ab[1];
      wt = w[1] * w[1];
      break;
    case 3:
      xi = ab[0];
      eta = ab[1];
      wt = w[0] * w[1];
      break;
    case 4:
      xi = ab[2];
      eta = ab[0];
      wt = w[2] * w[0];
      break;
    case 5:
      xi = ab[1];
      eta = ab[2];
      wt = w[1] * w[2];
      break;
    case 6:	
      xi = ab[2];
      eta = ab[1];
      wt = w[2] * w[1];
      break;
    case 7:
      xi = ab[0];
      eta = ab[2];
      wt = w[0] * w[2];
      break;
    case 8:
      xi = ab[2];
      eta = ab[2];
      wt = w[2] * w[2];
      break;
    default :
      printf("gp = %d\n", gp);
    }
  }
  // Calculate basis function and derivatives at nodal pts
  phi[0]=(1.0-xi)*(1.0-eta)/4.0;
  phi[1]=(1.0+xi)*(1.0-eta)/4.0;
  phi[2]=(1.0+xi)*(1.0+eta)/4.0;
  phi[3]=(1.0-xi)*(1.0+eta)/4.0;

  dphidxi[0]=-(1.0-eta)/4.0;
  dphidxi[1]= (1.0-eta)/4.0;
  dphidxi[2]= (1.0+eta)/4.0;
  dphidxi[3]=-(1.0+eta)/4.0;

  dphideta[0]=-(1.0-xi)/4.0;
  dphideta[1]=-(1.0+xi)/4.0;
  dphideta[2]= (1.0+xi)/4.0;
  dphideta[3]= (1.0-xi)/4.0;
  
  // Caculate basis function and derivative at GP.
  //std::cout<<x[0]<<" "<<x[1]<<" "<<x[2]<<" "<<x[3]<<std::endl;
  double dxdxi  = .25*( (x[1]-x[0])*(1.-eta)+(x[2]-x[3])*(1.+eta) );
  double dxdeta = .25*( (x[3]-x[0])*(1.- xi)+(x[2]-x[1])*(1.+ xi) );
  double dydxi  = .25*( (y[1]-y[0])*(1.-eta)+(y[2]-y[3])*(1.+eta) );
  double dydeta = .25*( (y[3]-y[0])*(1.- xi)+(y[2]-y[1])*(1.+ xi) );
  double dzdxi  = .25*( (z[1]-z[0])*(1.-eta)+(z[2]-z[3])*(1.+eta) );
  double dzdeta = .25*( (z[3]-z[0])*(1.- xi)+(z[2]-z[1])*(1.+ xi) );

  //jac = dxdxi * dydeta - dxdeta * dydxi;
  jac = sqrt( (dzdxi * dxdeta - dxdxi * dzdeta)*(dzdxi * dxdeta - dxdxi * dzdeta)
	     +(dydxi * dzdeta - dzdxi * dydeta)*(dydxi * dzdeta - dzdxi * dydeta)
	     +(dxdxi * dydeta - dxdeta * dydxi)*(dxdxi * dydeta - dxdeta * dydxi));

  dxidx = dydeta / jac;
  dxidy = -dxdeta / jac;
  dxidz = 0.;
  detadx = -dydxi / jac;
  detady = dxdxi / jac;
  detadz =0.;
  dztadx =0.;
  dztady =0.;
  dztadz =0.;
  // Caculate basis function and derivative at GP.
  xx=0.0;
  yy=0.0;
  zz=0.0;
  uu=0.0;
  uuold=0.0;
  uuoldold=0.0;
  dudx=0.0;
  dudy=0.0;
  dudz=0.0;
  duolddx = 0.;
  duolddy = 0.;
  duolddz = 0.;
  duoldolddx = 0.;
  duoldolddy = 0.;
  duoldolddz = 0.;
  // x[i] is a vector of node coords, x(j, k) 
  for (int i=0; i < 4; i++) {
    xx += x[i] * phi[i];
    yy += y[i] * phi[i];
    dphidx[i] = dphidxi[i]*dxidx+dphideta[i]*detadx;
    dphidy[i] = dphidxi[i]*dxidy+dphideta[i]*detady;
    dphidz[i] = 0.0;
    dphidzta[i]= 0.0;
    if( u ){
      uu += u[i] * phi[i];
      dudx += u[i] * dphidx[i];
      dudy += u[i]* dphidy[i];
    }
    if( uold ){
      uuold += uold[i] * phi[i];
      duolddx += uold[i] * dphidx[i];
      duolddy += uold[i]* dphidy[i];
    }
    if( uoldold ){
      uuoldold += uoldold[i] * phi[i];
      duoldolddx += uoldold[i] * dphidx[i];
      duoldolddy += uoldold[i]* dphidy[i];
    }
  }

  return;
}


//Constructor
BasisQTri::BasisQTri() {
  phi = new double[6];
  dphidxi = new double[6];
  dphideta = new double[6];
  dphidzta = new double[6];
  dphidx = new double[6];
  dphidy = new double[6];
  dphidz = new double[6];
  ngp = 3;
  weight = new double[ngp];
  abscissa = new double[ngp];
  abscissa[0] = 1.0L / 2.0L;
  abscissa[1] = 1.0L / 2.0L;
  abscissa[2] = 0.;
  weight[0] = 1.0L / 6.0L;
  weight[1] = 1.0L / 6.0L;
  weight[2] = 1.0L / 6.0L;
}

//Destructor
BasisQTri::~BasisQTri() {
  delete [] phi;
  delete [] dphidxi;
  delete [] dphideta;
  delete [] dphidzta;
  delete [] dphidx;
  delete [] dphidy;
  delete [] dphidz;
  delete [] weight;
  delete [] abscissa;
}

void BasisQTri::getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold){

  int N = 6;
  wt = 1.0L / 6.0L;
  if (gp==0) {xi = 2.0L/3.0L; eta=1.0L/6.0L;}
  if (gp==1) {xi = 1.0L/6.0L; eta=2.0L/3.0L;}
  if (gp==2) {xi = 1.0L/6.0L; eta=1.0L/6.0L;}

  // Calculate basis function and derivatives at nodel pts
  phi[0]=2.0 * (1.0 - xi - eta) * (0.5 - xi - eta);
  phi[1]= 2.0 * xi * (xi - 0.5);
  phi[2]= 2.0 * eta * (eta - 0.5);
  phi[3]=4.0 * (1.0 - xi - eta) * xi;
  phi[4]= 4.0 * xi * eta;
  phi[5]= 4.0 * (1.0 - xi - eta) * eta;
  dphidxi[0]=-2.0 * (0.5 - xi - eta) - 2.0 * (1.0 - xi - eta);
  dphidxi[1]= 2.0 * (xi - 0.5) + 2.0 * xi;
  dphidxi[2]= 0.0;
  dphidxi[3]=-4.0 * xi + 4.0 * (1.0 - xi - eta);
  dphidxi[4]= 4.0 * eta;
  dphidxi[5]= -4.0 * eta;
  dphideta[0]=-2.0 * (0.5 - xi - eta) - 2.0 * (1.0 - xi - eta);
  dphideta[1]= 0.0;
  dphideta[2]= 2.0 * eta + 2.0 * (eta - 0.5);
  dphideta[3]=-4.0 * xi;
  dphideta[4]= 4.0 * xi;
  dphideta[5]= 4.0 * (1.0 - xi - eta) - 4.0 * eta;
  dphidzta[0]= 0.0;
  dphidzta[1]= 0.0;
  dphidzta[2]= 0.0;
  dphidzta[3]= 0.0;
  dphidzta[4]= 0.0;
  dphidzta[5]= 0.0;
  
  // Caculate basis function and derivative at GP.
  double dxdxi = 0;
  double dxdeta = 0;
  double dydxi = 0;
  double dydeta = 0;

  for (int i=0; i < N; i++) {
    dxdxi += dphidxi[i] * x[i];
    dxdeta += dphideta[i] * x[i];
    dydxi += dphidxi[i] * y[i];
    dydeta += dphideta[i] * y[i];
  }

  jac = dxdxi * dydeta - dxdeta * dydxi;

  dxidx = dydeta / jac;
  dxidy = -dxdeta / jac;
  dxidz = 0.;
  detadx = -dydxi / jac;
  detady = dxdxi / jac;
  detadz = 0.;
  dztadx = 0.;
  dztady = 0.;
  dztadz = 0.;
  xx=0.0;
  yy=0.0;
  zz=0.0;
  uu=0.0;
  uuold = 0.;
  uuoldold = 0.;

  dudx=0.0;
  dudy=0.0;
  dudz=0.0;

  duolddx = 0.;
  duolddy = 0.;
  duolddz = 0.;
  duoldolddx = 0.;
  duoldolddy = 0.;
  duoldolddz = 0.;
  for (int i=0; i < N; i++) {
    xx += x[i] * phi[i];
    yy += y[i] * phi[i];
    dphidx[i] = dphidxi[i]*dxidx+dphideta[i]*detadx;
    dphidy[i] = dphidxi[i]*dxidy+dphideta[i]*detady;
    dphidz[i] = 0.;
    if( u ){
      uu += u[i] * phi[i];
      dudx += u[i] * dphidx[i];
      dudy += u[i]* dphidy[i];
    }
    if( uold ){
      uuold += uold[i] * phi[i];
      duolddx += uold[i] * dphidx[i];
      duolddy += uold[i]* dphidy[i];
    }
    if( uoldold ){
      uuoldold += uoldold[i] * phi[i];
      duoldolddx += uoldold[i] * dphidx[i];
      duoldolddy += uoldold[i]* dphidy[i];
    }
  }

  return;
}

// Constructor
BasisQQuad::BasisQQuad() {
  sngp = 3;
  ngp = sngp*sngp; // number of Gauss points
  phi = new double[9];
  dphidxi = new double[9];
  dphideta = new double[9];
  dphidzta = new double[9];
  dphidx = new double[9];
  dphidy = new double[9];
  dphidz = new double[9];
  abscissa = new double[sngp];
  weight = new double[sngp];
  setN(sngp, abscissa, weight);
}

// Destructor
BasisQQuad::~BasisQQuad() {
  delete [] phi;
  delete [] dphidxi;
  delete [] dphideta;
  delete [] dphidzta;
  delete [] dphidx;
  delete [] dphidy;
  delete [] dphidz;
  delete [] abscissa;
  delete [] weight;
}

void BasisQQuad::getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold){


  //std::cout<<"getBasis(int gp, double *x, double *y, double *u, double *uold) "<<gp<<std::endl;
  //printf("starting getBasis\n");
//   abscissa[0] = -1.0/sqrt(3.0);
//   abscissa[1] =  1.0/sqrt(3.0);
//   weight[0] = 1.0;
//   weight[1] = 1.0;

//printf("x:::%f %f %f %f %f %f %f %f %f\n",x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]);
//printf("y:::%f %f %f %f %f %f %f %f %f\n\n\n",y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7],y[8]);

// reindex abscissa and weight so it makes sense  
  double ab[3];// = new double[3];
  ab[0] = abscissa[0];
  ab[1] = abscissa[2];
  ab[2] = abscissa[1];

  double w[3];// = new double[3];
  w[0] = weight[0];
  w[1] = weight[2];
  w[2] = weight[1];

  
  switch (gp){
    case 0:
      xi = ab[0];
      eta = ab[0];
      wt = w[0] * w[0];
      break;
   case 1:
      xi = ab[1];
      eta = ab[0];
      wt = w[1] * w[0];
      break;
   case 2:
      xi = ab[1];
      eta = ab[1];
      wt = w[1] * w[1];
      break;
   case 3:
      xi = ab[0];
      eta = ab[1];
      wt = w[0] * w[1];
      break;
   case 4:
      xi = ab[2];
      eta = ab[0];
      wt = w[2] * w[0];
      break;
   case 5:
      xi = ab[1];
      eta = ab[2];
      wt = w[1] * w[2];
      break;
   case 6:	
      xi = ab[2];
      eta = ab[1];
      wt = w[2] * w[1];
      break;
   case 7:
      xi = ab[0];
      eta = ab[2];
      wt = w[0] * w[2];
      break;
   case 8:
      xi = ab[2];
      eta = ab[2];
      wt = w[2] * w[2];
      break;
   default :
      printf("gp = %d\n", gp);
  }
 
//printf("xi = %f, eta = %f \n",xi,eta);

/*if(0 == gp){
    xi = absciss[0];
    eta = abscissa[0];
    wt = weight[0] * weight[0];
  }else if (1 == gp){
    xi = abscissa[1];
    eta = abscissa[0];
    wt = weight[0] * weight[1];
  }else if (2 == gp){
    xi = abscissa[1];
    eta = abscissa[1];
    wt = weight[1] * weight[1];
  }else if (3 == gp){
    xi = abscissa[0];
    eta = abscissa[1];
    wt = weight[0] * weight[1];
  }*/

  // Calculate basis function and derivatives at nodal pts
/*phi[0]=(1.0-xi)*(1.0-eta)/4.0;
  phi[1]=(1.0+xi)*(1.0-eta)/4.0;
  phi[2]=(1.0+xi)*(1.0+eta)/4.0;
  phi[3]=(1.0-xi)*(1.0+eta)/4.0;*/

  

  double phi1_xi = -xi*(1-xi)/2;
  double phi2_xi = xi*(1+xi)/2;
  double phi3_xi = 1-xi*xi;

  double phi1_eta = -eta*(1-eta)/2;
  double phi2_eta = eta*(1+eta)/2;
  double phi3_eta = 1-eta*eta;
 
  //double phi_xi[3] = [phi1_xi,phi2_xi,phi3_xi];
  //double phi_eta[3] = [phi1_eta,phi2_eta,phi3_eta];

  //printf("gp = %d\n",gp);
  //printf("_xi %f %f %f:\n", phi1_xi, phi2_xi, phi3_xi);
  //printf("_eta %f %f %f:\n\n", phi1_eta, phi2_eta, phi3_eta);

  phi[0] = (-xi*(1-xi)/2)*(-eta*(1-eta)/2);
  phi[1] = (xi*(1+xi)/2)*(-eta*(1-eta)/2);
  phi[2] = (xi*(1+xi)/2)*(eta*(1+eta)/2);
  phi[3] = (-xi*(1-xi)/2)*(eta*(1+eta)/2);
  
  phi[4] = (1-xi*xi)*(-eta*(1-eta)/2);
  phi[5] = (xi*(1+xi)/2)*(1-eta*eta);
  phi[6] = (1-xi*xi)*(eta*(1+eta)/2);
  phi[7] = (-xi*(1-xi)/2)*(1-eta*eta);
  
  phi[8] = (1-xi*xi)*(1-eta*eta);

  double dphi1dxi = (-1+2*xi)/2;
  double dphi2dxi = (1+2*xi)/2;
  double dphi3dxi = (-2*xi);

  double dphi1deta = (-1+2*eta)/2;
  double dphi2deta = (1+2*eta)/2;
  double dphi3deta = -2*eta;
  
  
  dphidxi[0] = ((-1+2*xi)/2)*(-eta*(1-eta)/2);
  dphidxi[1] = ((1+2*xi)/2)*(-eta*(1-eta)/2);
  dphidxi[2] = ((1+2*xi)/2)*(eta*(1+eta)/2);
  dphidxi[3] = ((-1+2*xi)/2)*(eta*(1+eta)/2);
  dphidxi[4] = (-2*xi)*(-eta*(1-eta)/2);
  dphidxi[5] = ((1+2*xi)/2)*(1-eta*eta);
  dphidxi[6] = (-2*xi)*(eta*(1+eta)/2);
  dphidxi[7] = ((-1+2*xi)/2)*(1-eta*eta);
  dphidxi[8] = (-2*xi)*(1-eta*eta);


  dphideta[0] = (-xi*(1-xi)/2)*((-1+2*eta)/2);
  dphideta[1] = (xi*(1+xi)/2)*((-1+2*eta)/2);
  dphideta[2] = (xi*(1+xi)/2)*((1+2*eta)/2);
  dphideta[3] = (-xi*(1-xi)/2)*((1+2*eta)/2);
  dphideta[4] = (1-xi*xi)*((-1+2*eta)/2);
  dphideta[5] = (xi*(1+xi)/2)*(-2*eta);
  dphideta[6] = (1-xi*xi)*((1+2*eta)/2);
  dphideta[7] = (-xi*(1-xi)/2)*(-2*eta);
  dphideta[8] = (1-xi*xi)*(-2*eta);

 
/*dphidx[0]=-(1.0-eta)/4.0;
  dphidx[1]= (1.0-eta)/4.0;
  dphidx[2]= (1.0+eta)/4.0;
  dphidx[3]=-(1.0+eta)/4.0;*/

/*dphide[0]=-(1.0-xi)/4.0;
  dphide[1]=-(1.0+xi)/4.0;
  dphide[2]= (1.0+xi)/4.0;
  dphide[3]= (1.0-xi)/4.0;*/
  
  // Caculate basis function and derivative at GP.
  /*double dxdxi  = .25*( (x[1]-x[0])*(1.-eta)+(x[2]-x[3])*(1.+eta) );
  double dxdeta = .25*( (x[3]-x[0])*(1.- xi)+(x[2]-x[1])*(1.+ xi) );
  double dydxi  = .25*( (y[1]-y[0])*(1.-eta)+(y[2]-y[3])*(1.+eta) );
  double dydeta = .25*( (y[3]-y[0])*(1.- xi)+(y[2]-y[1])*(1.+ xi) );*/

  double dxdxi  = dphi1dxi *(x[0]*phi1_eta+x[3]*phi2_eta + x[7]*phi3_eta) + 
                   dphi2dxi *(x[1]*phi1_eta+x[2]*phi2_eta + x[5]*phi3_eta) + 
                   dphi3dxi *(x[4]*phi1_eta+x[6]*phi2_eta + x[8]*phi3_eta); 
 
  double dydxi =  dphi1dxi *(y[0]*phi1_eta+y[3]*phi2_eta + y[7]*phi3_eta) + 
                   dphi2dxi *(y[1]*phi1_eta+y[2]*phi2_eta + y[5]*phi3_eta) + 
                   dphi3dxi *(y[4]*phi1_eta+y[6]*phi2_eta + y[8]*phi3_eta); 
  
  double dxdeta = dphi1deta *(x[0]*phi1_xi+x[1]*phi2_xi + x[4]*phi3_xi) + 
                   dphi2deta *(x[3]*phi1_xi+x[2]*phi2_xi + x[6]*phi3_xi) + 
                   dphi3deta *(x[7]*phi1_xi+x[5]*phi2_xi + x[8]*phi3_xi); 
 
  double dydeta = dphi1deta *(y[0]*phi1_xi+y[1]*phi2_xi + y[4]*phi3_xi) + 
                   dphi2deta *(y[3]*phi1_xi+y[2]*phi2_xi + y[6]*phi3_xi) + 
                   dphi3deta *(y[7]*phi1_xi+y[5]*phi2_xi + y[8]*phi3_xi); 
 
/*printf("%f + %f + %f\n",dphi1dxi *(x[0]*phi1_eta+x[3]*phi2_eta + x[7]*phi3_eta),dphi2dxi *(x[1]*phi1_eta+x[2]*phi2_eta + x[5]*phi3_eta),dphi3dxi *(x[4]*phi1_eta+x[6]*phi2_eta + x[8]*phi3_eta));*/

//  printf("%f %f %f %f\n",dxdxi,dydeta,dxdeta,dydxi);
  jac = dxdxi * dydeta - dxdeta * dydxi;
  //printf("jacobian = %f\n\n",jac);
  if (jac <= 0){
    printf("\n\n\n\n\n\nnonpositive jacobian \n\n\n\n\n\n");
  }
 
//printf("_deta : %f %f %f\n",phi1_eta,phi2_eta,phi3_eta);
 // printf("_deta : %f %f %f\n",dphi1deta,dphi2deta,dphi3deta);
//printf("_xi : %f %f %f\n\n",dphi1dxi,dphi2dxi,dphi3dxi);

  dxidx = dydeta / jac;
  dxidy = -dxdeta / jac;
  dxidz = 0.;
  detadx = -dydxi / jac;
  detady = dxdxi / jac;
  detadz =0.;
  dztadx =0.;
  dztady =0.;
  dztadz =0.;

  xx=0.0;
  yy=0.0;
  zz=0.0;
  uu=0.0;
  uuold=0.0;
  uuoldold=0.0;
  dudx=0.0;
  dudy=0.0;
  dudz=0.0;
  duolddx = 0.;
  duolddy = 0.;
  duolddz = 0.;
  duoldolddx = 0.;
  duoldolddy = 0.;
  duoldolddz = 0.;
  for (int i=0; i < 9; i++) {
    xx += x[i] * phi[i];
    yy += y[i] * phi[i];
    dphidx[i] = dphidxi[i]*dxidx+dphideta[i]*detadx;
    dphidy[i] = dphidxi[i]*dxidy+dphideta[i]*detady;
    dphidz[i] = 0.0;
    dphidzta[i]= 0.0;
    if( u ){
      uu += u[i] * phi[i];
      dudx += u[i] * dphidx[i];
      dudy += u[i]* dphidy[i];
    }
    if( uold ){
      uuold += uold[i] * phi[i];
      duolddx += uold[i] * dphidx[i];
      duolddy += uold[i]* dphidy[i];
    }
    if( uoldold ){
      uuoldold += uoldold[i] * phi[i];
      duoldolddx += uoldold[i] * dphidx[i];
      duoldolddy += uoldold[i]* dphidy[i];
    }
  }

  //printf("getBasis done\n");
  return;
}

//  3D basis...

// Constructor
BasisLHex::BasisLHex(int n) : sngp(n){
  ngp = 8;
  phi = new double[ngp];
  dphidxi = new double[ngp];
  dphideta = new double[ngp];
  dphidzta = new double[ngp];
  abscissa = new double[sngp];
  weight = new double[sngp];
  setN(sngp, abscissa, weight);
}

BasisLHex::BasisLHex(){
  sngp = 4;
  ngp = 8;
  phi = new double[ngp];
  dphidxi = new double[ngp];
  dphideta = new double[ngp];
  dphidzta = new double[ngp];
  abscissa = new double[sngp];
  weight = new double[sngp];
  setN(sngp, abscissa, weight);
}

// Destructor
BasisLHex::~BasisLHex() {
  delete [] phi;
  delete [] dphideta;
  delete [] dphidxi;
  delete [] dphidzta;
  delete [] abscissa;
  delete [] weight;
}

// Calculates a linear 3D basis
void BasisLHex::getBasis(const int gp, const double *x, const double *y) {
  std::cout<<"BasisLHex::getBasis(int gp, double *x, double *y) is not implemented"<<std::endl;
  exit(0);
}

void BasisLHex::getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold){

  if(0 == gp){//cn N = 2 gp for now
    xi = abscissa[0];  // 0, 0, 0
    eta = abscissa[0];
    zta = abscissa[0];
    wt = weight[0] * weight[0] * weight[0];
  }else if (1 == gp){
    xi = abscissa[1]; // 1, 0, 0
    eta = abscissa[0];
    zta = abscissa[0];
    wt = weight[0] * weight[1] * weight[0];
  }else if (2 == gp){
    xi = abscissa[1]; // 1, 1, 0
    eta = abscissa[1];
    zta = abscissa[0];
    wt = weight[1] * weight[1] * weight[0];
  }else if (3 == gp){
    xi = abscissa[0];  //0, 1, 0
    eta = abscissa[1];
    zta = abscissa[0];
    wt = weight[0] * weight[1] * weight[0];
  } else if(4 == gp){
    xi = abscissa[0];  // 0, 0, 1
    eta = abscissa[0];
    zta = abscissa[1];
    wt = weight[0] * weight[0] * weight[1];
  }else if (5 == gp){
    xi = abscissa[1]; // 1, 0, 1
    eta = abscissa[0];
    zta = abscissa[1];
    wt = weight[0] * weight[1] * weight[1];
  }else if (6 == gp){
    xi = abscissa[1]; // 1, 1, 1
    eta = abscissa[1];
    zta = abscissa[1];
    wt = weight[1] * weight[1] * weight[1];
  }else if (7 == gp){
    xi = abscissa[0];  //0, 1, 1
    eta = abscissa[1];
    zta = abscissa[1];
    wt = weight[0] * weight[1] * weight[1];
  }

  // Calculate basis function and derivatives at nodal pts
   phi[0]   =  0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 - zta);
   phi[1]   =  0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 - zta);
   phi[2]   =  0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 - zta);
   phi[3]   =  0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 - zta);
   phi[4]   =  0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 + zta);
   phi[5]   =  0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 + zta);
   phi[6]   =  0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 + zta);
   phi[7]   =  0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 + zta);

   // ksi-derivatives
   dphidxi[0] = -0.125 * (1.0 - eta) * (1.0 - zta);
   dphidxi[1] =  0.125 * (1.0 - eta) * (1.0 - zta);
   dphidxi[2] =  0.125 * (1.0 + eta) * (1.0 - zta);
   dphidxi[3] = -0.125 * (1.0 + eta) * (1.0 - zta);
   dphidxi[4] = -0.125 * (1.0 - eta) * (1.0 + zta);
   dphidxi[5] =  0.125 * (1.0 - eta) * (1.0 + zta);
   dphidxi[6] =  0.125 * (1.0 + eta) * (1.0 + zta);
   dphidxi[7] = -0.125 * (1.0 + eta) * (1.0 + zta);

   // eta-derivatives
   dphideta[0] = -0.125 * (1.0 - xi) * (1.0 - zta);
   dphideta[1] = -0.125 * (1.0 + xi) * (1.0 - zta);
   dphideta[2] =  0.125 * (1.0 + xi) * (1.0 - zta);
   dphideta[3] =  0.125 * (1.0 - xi) * (1.0 - zta);
   dphideta[4] = -0.125 * (1.0 - xi) * (1.0 + zta);
   dphideta[5] = -0.125 * (1.0 + xi) * (1.0 + zta);
   dphideta[6] =  0.125 * (1.0 + xi) * (1.0 + zta);
   dphideta[7] =  0.125 * (1.0 - xi) * (1.0 + zta);

   // zeta-derivatives
   dphidzta[0] = -0.125 * (1.0 - xi) * (1.0 - eta);
   dphidzta[1] = -0.125 * (1.0 + xi) * (1.0 - eta);
   dphidzta[2] = -0.125 * (1.0 + xi) * (1.0 + eta);
   dphidzta[3] = -0.125 * (1.0 - xi) * (1.0 + eta);
   dphidzta[4] =  0.125 * (1.0 - xi) * (1.0 - eta);
   dphidzta[5] =  0.125 * (1.0 + xi) * (1.0 - eta);
   dphidzta[6] =  0.125 * (1.0 + xi) * (1.0 + eta);
   dphidzta[7] =  0.125 * (1.0 - xi) * (1.0 + eta);
  
  // Caculate basis function and derivative at GP.
  double dxdxi  = 0.125*( (x[1]-x[0])*(1.-eta)*(1.-zta) + (x[2]-x[3])*(1.+eta)*(1.-zta) 
	+ (x[5]-x[4])*(1.-eta)*(1.+zta) + (x[6]-x[7])*(1.+eta)*(1.+zta) );
  double dxdeta = 0.125*( (x[3]-x[0])*(1.- xi)*(1.-zta) + (x[2]-x[1])*(1.+ xi)*(1.-zta) 
	+ (x[7]-x[4])*(1.- xi)*(1.+zta) + (x[6]-x[5])*(1.+ xi)*(1.+zta) );
  double dxdzta = 0.125*( (x[4]-x[0])*(1.- xi)*(1.-eta) + (x[5]-x[1])*(1.+ xi)*(1.-eta)
	+ (x[6]-x[2])*(1.+ xi)*(1.+eta) + (x[7]-x[3])*(1.- xi)*(1.+eta) );

  double dydxi  = 0.125*( (y[1]-y[0])*(1.-eta)*(1.-zta) + (y[2]-y[3])*(1.+eta)*(1.-zta)
	+ (y[5]-y[4])*(1.-eta)*(1.+zta) + (y[6]-y[7])*(1.+eta)*(1.+zta) );
  double dydeta = 0.125*( (y[3]-y[0])*(1.- xi)*(1.-zta) + (y[2]-y[1])*(1.+ xi)*(1.-zta) 
	+ (y[7]-y[4])*(1.- xi)*(1.+zta) + (y[6]-y[5])*(1.+ xi)*(1.+zta) );
  double dydzta = 0.125*( (y[4]-y[0])*(1.- xi)*(1.-eta) + (y[5]-y[1])*(1.+ xi)*(1.-eta)
	+ (y[6]-y[2])*(1.+ xi)*(1.+eta) + (y[7]-y[3])*(1.- xi)*(1.+eta) );

  double dzdxi  = 0.125*( (z[1]-z[0])*(1.-eta)*(1.-zta) + (z[2]-z[3])*(1.+eta)*(1.-zta)
	+ (z[5]-z[4])*(1.-eta)*(1.+zta) + (z[6]-z[7])*(1.+eta)*(1.+zta) );
  double dzdeta = 0.125*( (z[3]-z[0])*(1.- xi)*(1.-zta) + (z[2]-z[1])*(1.+ xi)*(1.-zta) 
	+ (z[7]-z[4])*(1.- xi)*(1.+zta) + (z[6]-z[5])*(1.+ xi)*(1.+zta) );
  double dzdzta = 0.125*( (z[4]-z[0])*(1.- xi)*(1.-eta) + (z[5]-z[1])*(1.+ xi)*(1.-eta)
	+ (z[6]-z[2])*(1.+ xi)*(1.+eta) + (z[7]-z[3])*(1.- xi)*(1.+eta) );

  jac = dxdxi*(dydeta*dzdzta - dydzta*dzdeta) - dxdeta*(dydxi*dzdzta - dydzta*dzdxi) 
	                                      + dxdzta*(dydxi*dzdeta - dydeta*dzdxi);


  dxidx =  (-dydzta*dzdeta + dydeta*dzdzta) / jac;
  dxidy =  ( dxdzta*dzdeta - dxdeta*dzdzta) / jac;
  dxidz =  (-dxdzta*dydeta + dxdeta*dydzta) / jac;

  detadx =  ( dydzta*dzdxi - dydxi*dzdzta) / jac;
  detady =  (-dxdzta*dzdxi + dxdxi*dzdzta) / jac;
  detadz =  ( dxdzta*dydxi - dxdxi*dydzta) / jac;

  dztadx =  ( dydxi*dzdeta - dydeta*dzdxi) / jac;
  dztady =  (-dxdxi*dzdeta + dxdeta*dzdxi) / jac;
  dztadz =  ( dxdxi*dydeta - dxdeta*dydxi) / jac;
  // Caculate basis function and derivative at GP.
  xx=0.0;
  yy=0.0;
  zz=0.0;
  uu=0.0;
  uuold=0.0;
  uuoldold=0.0;
  dudx=0.0;
  dudy=0.0;
  dudz=0.0;
  duolddx = 0.;
  duolddy = 0.;
  duolddz = 0.;
  duoldolddx = 0.;
  duoldolddy = 0.;
  duoldolddz = 0.;
  // x[i] is a vector of node coords, x(j, k) 
  for (int i=0; i < 8; i++) {
    xx += x[i] * phi[i];
    yy += y[i] * phi[i];
    zz += z[i] * phi[i];
    if( u ){
      uu += u[i] * phi[i];
      dudx += u[i] * (dphidxi[i]*dxidx+dphideta[i]*detadx+dphidzta[i]*dztadx);
      dudy += u[i] * (dphidxi[i]*dxidy+dphideta[i]*detady+dphidzta[i]*dztady);
      dudz += u[i] * (dphidxi[i]*dxidz+dphideta[i]*detadz+dphidzta[i]*dztadz);
    }
    if( uold ){
      uuold += uold[i] * phi[i];
      duolddx += uold[i] * (dphidxi[i]*dxidx+dphideta[i]*detadx+dphidzta[i]*dztadx);
      duolddy += uold[i] * (dphidxi[i]*dxidy+dphideta[i]*detady+dphidzta[i]*dztady);
      duolddz += uold[i] * (dphidxi[i]*dxidz+dphideta[i]*detadz+dphidzta[i]*dztadz);
      //exit(0);
    }
    if( uoldold ){
      uuoldold += uoldold[i] * phi[i];
      duoldolddx += uoldold[i] * (dphidxi[i]*dxidx+dphideta[i]*detadx+dphidzta[i]*dztadx);
      duoldolddy += uoldold[i] * (dphidxi[i]*dxidx+dphideta[i]*detadx+dphidzta[i]*dztadx);
      duoldolddz += uoldold[i] * (dphidxi[i]*dxidx+dphideta[i]*detadx+dphidzta[i]*dztadx);
    }
  }
  return;
}


BasisLTet::BasisLTet(int n) : sngp(n){
  ngp = 8;
  phi = new double[ngp];
  dphidxi = new double[ngp];
  dphideta = new double[ngp];
  dphidzta = new double[ngp];
  abscissa = new double[sngp];
  weight = new double[sngp];
  setN(sngp, abscissa, weight);
  exit(0);
}

BasisLTet::BasisLTet(){
  //cn we can have 1 or 4 guass points; 4 will be default
#if 0
  sngp = 1;
  ngp = 1;
#endif
  sngp = 2;
  ngp = 4;
  phi = new double[ngp];
  dphidxi = new double[ngp];
  dphideta = new double[ngp];
  dphidzta = new double[ngp];
  abscissa = new double[sngp];
  weight = new double[sngp];
  //setN(sngp, abscissa, weight);
#if 0
  abscissa[0] = 0.25000000;
  weight[0] = 0.16666667;
#endif
  abscissa[0] = 0.13819660;
  abscissa[1] = 0.58541020;
  weight[0] = 0.041666666667;
  weight[1] = 0.041666666667;
}

// Destructor
BasisLTet::~BasisLTet() {
  delete [] phi;
  delete [] dphideta;
  delete [] dphidxi;
  delete [] dphidzta;
  delete [] abscissa;
  delete [] weight;
}

// Calculates a linear 3D basis
void BasisLTet::getBasis(const int gp, const double *x, const double *y) {
  std::cout<<"BasisLTet::getBasis(int gp, double *x, double *y) is not implemented"<<std::endl;
  exit(0);
}

void BasisLTet::getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold){

  if(0 == gp){
    xi = abscissa[0];  // 0, 0, 0
    eta = abscissa[0];
    zta = abscissa[0];
    wt = weight[0] * weight[0] * weight[0];
  }else if (1 == gp){
    xi = abscissa[1]; // 1, 0, 0
    eta = abscissa[0];
    zta = abscissa[0];
    wt = weight[0] * weight[1] * weight[0];
  }else if (2 == gp){
    xi = abscissa[0]; // 1, 1, 0
    eta = abscissa[1];
    zta = abscissa[0];
    wt = weight[1] * weight[1] * weight[0];
  }else if (3 == gp){
    xi = abscissa[0];  //0, 1, 0
    eta = abscissa[0];
    zta = abscissa[1];
    wt = weight[0] * weight[1] * weight[0];
  } 

  // Calculate basis function and derivatives at nodal pts
   phi[0]   =  1.0 - xi  - eta - zta;
   phi[1]   =  xi;
   phi[2]   =  eta;
   phi[3]   =  zta;

   // ksi-derivatives
   dphidxi[0] = -1.;
   dphidxi[1] =  1.;;
   dphidxi[2] =  0.;
   dphidxi[3] =  0.;

   // eta-derivatives
   dphideta[0] = -1.;
   dphideta[1] =  0.;
   dphideta[2] =  1.;
   dphideta[3] =  0.;

   // zeta-derivatives
   dphidzta[0] = -1.;
   dphidzta[1] =  0.;
   dphidzta[2] =  0.;
   dphidzta[3] =  1.;
  
  // Caculate basis function and derivative at GP.
  double dxdxi  = (x[1]-x[0]);
  double dxdeta = (x[2]-x[0]);
  double dxdzta = (x[3]-x[0]);

  double dydxi  = (y[1]-y[0]);
  double dydeta = (y[2]-y[0]); 
  double dydzta = (y[3]-y[0]);

  double dzdxi  = (z[1]-z[0]);
  double dzdeta = (z[2]-z[0]);
  double dzdzta = (z[3]-z[0]);

  jac = dxdxi*(dydeta*dzdzta - dydzta*dzdeta) - dxdeta*(dydxi*dzdzta - dydzta*dzdxi) 
	                                      + dxdzta*(dydxi*dzdeta - dydeta*dzdxi);


  dxidx =  (-dydzta*dzdeta + dydeta*dzdzta) / jac;
  dxidy =  ( dxdzta*dzdeta - dxdeta*dzdzta) / jac;
  dxidz =  (-dxdzta*dydeta + dxdeta*dydzta) / jac;

  detadx =  ( dydzta*dzdxi - dydxi*dzdzta) / jac;
  detady =  (-dxdzta*dzdxi + dxdxi*dzdzta) / jac;
  detadz =  ( dxdzta*dydxi - dxdxi*dydzta) / jac;

  dztadx =  ( dydxi*dzdeta - dydeta*dzdxi) / jac;
  dztady =  (-dxdxi*dzdeta + dxdeta*dzdxi) / jac;
  dztadz =  ( dxdxi*dydeta - dxdeta*dydxi) / jac;
  // Caculate basis function and derivative at GP.
  xx=0.0;
  yy=0.0;
  zz=0.0;
  uu=0.0;
  uuold=0.0;
  uuoldold=0.0;
  dudx=0.0;
  dudy=0.0;
  dudz=0.0;
  duolddx = 0.;
  duolddy = 0.;
  duolddz = 0.;
  duoldolddx = 0.;
  duoldolddy = 0.;
  duoldolddz = 0.;
  // x[i] is a vector of node coords, x(j, k) 
  for (int i=0; i < ngp; i++) {
    xx += x[i] * phi[i];
    yy += y[i] * phi[i];
    zz += z[i] * phi[i];
    if( u ){
      uu += u[i] * phi[i];
      dudx += u[i] * (dphidxi[i]*dxidx+dphideta[i]*detadx+dphidzta[i]*dztadx);
      dudy += u[i] * (dphidxi[i]*dxidy+dphideta[i]*detady+dphidzta[i]*dztady);
      dudz += u[i] * (dphidxi[i]*dxidz+dphideta[i]*detadz+dphidzta[i]*dztadz);
    }
    if( uold ){
      uuold += uold[i] * phi[i];
      duolddx += uold[i] * (dphidxi[i]*dxidx+dphideta[i]*detadx+dphidzta[i]*dztadx);
      duolddy += uold[i] * (dphidxi[i]*dxidy+dphideta[i]*detady+dphidzta[i]*dztady);
      duolddz += uold[i] * (dphidxi[i]*dxidz+dphideta[i]*detadz+dphidzta[i]*dztadz);
      //exit(0);
    }
    if( uoldold ){
      uuoldold += uoldold[i] * phi[i];
      duoldolddx += uoldold[i] * (dphidxi[i]*dxidx+dphideta[i]*detadx+dphidzta[i]*dztadx);
      duoldolddy += uoldold[i] * (dphidxi[i]*dxidy+dphideta[i]*detady+dphidzta[i]*dztady);
      duoldolddz += uoldold[i] * (dphidxi[i]*dxidz+dphideta[i]*detadz+dphidzta[i]*dztadz);
    }
  }
  return;
}


// Constructor
BasisLBar::BasisLBar(int n) :sngp(n){
  //sngp = 2;// number of Gauss points
  if ( 2 != sngp ){
    std::cout<<"BasisLBar only supported for n = 2 at this time"<<std::endl<<std::endl<<std::endl;
    exit(0);
  }
  ngp = sngp;
  phi = new double[2];//number of nodes
  dphidxi = new double[2];
  dphideta = new double[2];
  dphidzta = new double[2];
  dphidx = new double[2];
  dphidy = new double[2];
  dphidz = new double[2];
  abscissa = new double[sngp];//number guass pts
  weight = new double[sngp];
  setN(sngp, abscissa, weight);
  //std::cout<<abscissa[0]<<" "<<abscissa[1]<<" "<<weight[0]<<" "<<weight[1]<<std::endl;
}

// Destructor
BasisLBar::~BasisLBar() {
  delete [] phi;
  delete [] dphidxi;
  delete [] dphideta;
  delete [] dphidzta;
  delete [] dphidx;
  delete [] dphidy;
  delete [] dphidz;
  delete [] abscissa;
  delete [] weight;
}

void BasisLBar::getBasis(const int gp,const  double *x, const  double *y,  const double *z,const  double *u,const  double *uold,const  double *uoldold) {

  //cn this is horrible, need a better way than having if statements in here
  if(2 == ngp){
    if(0 == gp){
      xi = abscissa[0];
      wt = weight[0];
    }else if (1 == gp){
      xi = abscissa[1];
      wt = weight[0];
    }
  }
  // Calculate basis function and derivatives at nodal pts
  phi[0]=(1.0-xi)/2.0;
  phi[1]=(1.0+xi)/2.0;

  dphidxi[0]= -1.0/2.0;
  dphidxi[1]= 1.0/2.0;

  dphideta[0]= 0.0;
  dphideta[1]= 0.0;
  dphidzta[0] = 0.0;
  dphidzta[1]= 0.0;
  
  // Caculate basis function and derivative at GP.
  //std::cout<<x[0]<<" "<<x[1]<<" "<<x[2]<<" "<<x[3]<<std::endl;
  double dxdxi  = .5* (x[1]-x[0]);
  //double dxdeta = .25*( (x[3]-x[0])*(1.- xi)+(x[2]-x[1])*(1.+ xi) );
  //double dydxi  = .25*( (y[1]-y[0])*(1.-eta)+(y[2]-y[3])*(1.+eta) );
  double dydxi  = .5*(y[1]-y[0]);
  //double dydeta = .25*( (y[3]-y[0])*(1.- xi)+(y[2]-y[1])*(1.+ xi) );
  double dzdxi  = .5*(z[1]-z[0]);

  //cn not sure about this still
  jac = sqrt(dxdxi*dxdxi+dydxi*dydxi+dzdxi*dzdxi);

  dxidx = 1. / dxdxi;
  dxidy = 1. / dydxi;
  dxidz = 1. / dzdxi;
  detadx = 0.;
  detady = 0.;
  detadz =0.;
  dztadx =0.;
  dztady =0.;
  dztadz =0.;
  // Caculate basis function and derivative at GP.
  xx=0.0;
  yy=0.0;
  zz=0.0;
  uu=0.0;
  uuold=0.0;
  uuoldold=0.0;
  dudx=0.0;
  dudy=0.0;
  dudz=0.0;
  duolddx = 0.;
  duolddy = 0.;
  duolddz = 0.;
  duoldolddx = 0.;
  duoldolddy = 0.;
  duoldolddz = 0.;
  // x[i] is a vector of node coords, x(j, k) 
  for (int i=0; i < 2; i++) {
    xx += x[i] * phi[i];
    yy += y[i] * phi[i];
    zz += z[i] * phi[i];
    dphidx[i] = dphidxi[i]*dxidx;
    dphidy[i] = dphidxi[i]*dxidy;
    dphidz[i] = dphidxi[i]*dxidz;
    if( u ){
      //uu += u[i] * phi[i];
      //dudx += u[i] * dphidx[i];
      //dudy += u[i]* dphidy[i];
    }
    if( uold ){
      //uuold += uold[i] * phi[i];
      //duolddx += uold[i] * dphidx[i];
      //duolddy += uold[i]* dphidy[i];
    }
    if( uoldold ){
      //uuoldold += uoldold[i] * phi[i];
      //duoldolddx += uoldold[i] * dphidx[i];
      //duoldolddy += uoldold[i]* dphidy[i];
    }
  }
  return;
}


// Constructor
BasisQBar::BasisQBar(int n) :sngp(n){
  //sngp = 3;// number of Gauss points
  if ( 3 != sngp ){
    std::cout<<"BasisQBar only supported for n = 3 at this time"<<std::endl<<std::endl<<std::endl;
    exit(0);
  }
  ngp = sngp;
  phi = new double[3];//number of nodes
  dphidxi = new double[3];
  dphideta = new double[3];
  dphidzta = new double[3];
  dphidx = new double[3];
  dphidy = new double[3];
  dphidz = new double[3];
  abscissa = new double[sngp];//number guass pts
  weight = new double[sngp];
  setN(sngp, abscissa, weight);
  //std::cout<<abscissa[0]<<" "<<abscissa[1]<<" "<<abscissa[2]<<" "<<weight[0]<<" "<<weight[1]<<" "<<weight[2]<<std::endl;
}

// Destructor
BasisQBar::~BasisQBar() {
  delete [] phi;
  delete [] dphidxi;
  delete [] dphideta;
  delete [] dphidzta;
  delete [] dphidx;
  delete [] dphidy;
  delete [] dphidz;
  delete [] abscissa;
  delete [] weight;
}

void BasisQBar::getBasis(const int gp,const  double *x, const  double *y,  const double *z,const  double *u,const  double *uold,const  double *uoldold) {

  //cn this is horrible, need a better way than having if statements in here
  if(3 == ngp){
    if(0 == gp){
      xi = abscissa[0];
      wt = weight[0];
    }else if (1 == gp){
      xi = abscissa[1];
      wt = weight[1];
    }else if (2 == gp){
      xi = abscissa[2];
      wt = weight[2];
    }
  }else{
    std::cout<<"BasisQBar only supported for n = 3 at this time"<<std::endl<<std::endl<<std::endl;
    exit(0);
  }
  // Calculate basis function and derivatives at nodal pts
  phi[0]=-xi*(1.0-xi)/2.0;
  phi[1]=xi*(1.0+xi)/2.0;
  phi[2]=(1.0-xi*xi);

  dphidxi[0]=-1.0/2.0 + xi;
  dphidxi[1]= 1.0/2.0 + xi;
  dphidxi[2]=-2.0*xi;

  dphideta[0]= 0.0;
  dphideta[1]= 0.0;
  dphideta[2]= 0.0;
  dphidzta[0] = 0.0;
  dphidzta[1]= 0.0;
  dphidzta[2]= 0.0;
  
  // Caculate basis function and derivative at GP.
  double dxdxi  = dphidxi[0]*x[0] + dphidxi[1]*x[1] + dphidxi[2]*x[2];
  double dydxi  = dphidxi[0]*y[0] + dphidxi[1]*y[1] + dphidxi[2]*y[2];
  double dzdxi  = dphidxi[0]*z[0] + dphidxi[1]*z[1] + dphidxi[2]*z[2];
    
  jac = sqrt(dxdxi*dxdxi+dydxi*dydxi+dzdxi*dzdxi);

  dxidx = 1. / dxdxi;
  dxidy = 1. / dydxi;
  dxidz = 1. / dzdxi;
  detadx = 0.;
  detady = 0.;
  detadz =0.;
  dztadx =0.;
  dztady =0.;
  dztadz =0.;
  // Caculate basis function and derivative at GP.
  xx=0.0;
  yy=0.0;
  zz=0.0;
  uu=0.0;
  uuold=0.0;
  uuoldold=0.0;
  dudx=0.0;
  dudy=0.0;
  dudz=0.0;
  duolddx = 0.;
  duolddy = 0.;
  duolddz = 0.;
  duoldolddx = 0.;
  duoldolddy = 0.;
  duoldolddz = 0.;
  // x[i] is a vector of node coords, x(j, k) 
  for (int i=0; i < 3; i++) {
    xx += x[i] * phi[i];
    yy += y[i] * phi[i];
    zz += z[i] * phi[i];
    dphidx[i] = dphidxi[i]*dxidx;
    dphidy[i] = dphidxi[i]*dxidy;
    dphidz[i] = dphidxi[i]*dxidz;
    if( u ){
      //uu += u[i] * phi[i];
      //dudx += u[i] * dphidx[i];
      //dudy += u[i]* dphidy[i];
    }
    if( uold ){
      //uuold += uold[i] * phi[i];
      //duolddx += uold[i] * dphidx[i];
      //duolddy += uold[i]* dphidy[i];
    }
    if( uoldold ){
      //uuoldold += uoldold[i] * phi[i];
      //duoldolddx += uoldold[i] * dphidx[i];
      //duoldolddy += uoldold[i]* dphidy[i];
    }
  }
  return;
}
