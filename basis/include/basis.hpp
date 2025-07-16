//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef BASIS_H
#define BASIS_H
	
#include <Kokkos_Core.hpp>

#include <cmath>

// gaw // KOKKOS_INLINE_FUNCTION is defined as __host__ __device__ inline
// Use KOKKOS_INLINE_FUNCTION instead of TUSAS_CUDA_CALLABLE_MEMBER, want
// to inline functions defined in header
#ifdef KOKKOS_HAVE_CUDA
#define TUSAS_CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define TUSAS_CUDA_CALLABLE_MEMBER
#endif

#define TUSAS_MAX_NUMEQS 6

#define TUSAS_FMA
KOKKOS_INLINE_FUNCTION
double FMA(double A, double B, double C)
{
#ifdef TUSAS_FMA
  return std::fma(A,B,C);
#else
  return A*B+C;
#endif
}


/// Base class for computation of finite element basis.
/** All basis classes inherit from this. */
class Basis {

 public:

  /// Constructor
  Basis():stype("unknown"){};

  /// Destructor
  virtual ~Basis(){};

  /// Evaluate the basis functions at the specified gauss point
  /** Evaluate the 2D basis functions at Gauss point gp given x, y. This function needs to be called before any accessor function.*/
  virtual void getBasis(const int gp, ///< current Gauss point (input)
			const double *x, ///< x array (input)
			const double *y ///< y array (input)
			){getBasis(gp, x, y, NULL, NULL, NULL, NULL);};
  /// Evaluate the basis functions at the specified gauss point
  /** Evaluate the 3D basis functions at Gauss point gp given x, y, z.  This function needs to be called before any accessor function.*/
  virtual void getBasis(const int gp, ///< current Gauss point (input) 
			const double *x, ///< x array (input) 
			const double *y, ///< y array (input) 
			const double *z ///< z array (input)
			){getBasis(gp, x, y, z, NULL, NULL, NULL);};
  /// Evaluate the basis functions at the specified gauss point
  /** Evaluate the 3D basis functions at Gauss point gp given x, y, z and interpolate u.  This function needs to be called before any accessor function.*/
  virtual void getBasis(const int gp,  ///< current Gauss point (input)
			const double *x,  ///< x array (input)
			const double *y, ///< y array (input) 
			const double *z, ///< z array (input)
			const double *u ///< u (solution)  array (input)
			){getBasis(gp, x, y, z, u, NULL, NULL);};
  /// Evaluate the basis functions at the specified gauss point
  /** Evaluate the 3D basis functions at Gauss point gp given x, y, z and interpolate u, uold.  This function needs to be called before any accessor function.*/
  virtual void getBasis(const int gp,  ///< current Gauss point (input)
			const double *x,  ///< x array (input)
			const double *y, ///< y array (input) 
			const double *z, ///< z array (input) 
			const double *u, ///< u (solution)  array (input) 
			const double *uold ///< uold (solution)  array (input)
			){getBasis(gp, x, y, z, u, uold, NULL);};
  /// Evaluate the basis functions at the specified gauss point
  /** Evaluate the 3D basis functions at Gauss point gp given x, y, z and interpolate u, uold, uoldold.  This function needs to be called before any accessor function.*/
  virtual void getBasis( const int gp,  ///< current Gauss point (input) 
			 const double *x,  ///< x array (input) 
			 const double *y, ///< y array (input)  
			 const double *z, ///< z array (input)  
			 const double *u, ///< u (solution)  array (input)  
			 const double *uold, ///< uold (solution)  array (input) 
			 const double *uoldold///< uoldold (solution)  array (input)
			 ) {return;};

  /// Evaluate the basis functions at the specified at point xx, yy ,zz
  /** Evaluate the 3D basis functions at  xx, yy, zz and interpolate u.  True if xx, yy, zz is in this element. Returns val = u(xx,yy,zz)*/
  virtual bool evalBasis( const double *x,  ///< x array (input) 
			  const double *y, ///< y array (input)  
			  const double *z, ///< z array (input)  
			  const double *u, ///< u (solution)  array (input)  
			  const double xx_, ///< xx 
			  const double yy_,///< yy
			  const double zz_,///< zz
			  double &val ///< return val
			 ){return false;};

  /// Set the number of Gauss points.
  void setN(const int N, ///< number of Gauss points (input)
	    double *abscissa, ///< abscissa array
	    double *weight ///< weight array
	    ){

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
};    

  /// Required for particular implementation
  virtual Basis* clone() const {return new Basis(*this);};
  /// Required for particular implementation
  virtual const char * type() const {return stype.c_str();};

 public:
  // Variables that are calculated at the gauss point
  /// Access number of Gauss points.
  int ngp;
  /// Access value of basis function at the current Gauss point.
  double *phi; 
  /// Access value of dphi / dxi  at the current Gauss point.
  double *dphidxi;
  /// Access value of dphi / deta  at the current Gauss point.
  double *dphideta;
  /// Access value of dphi / dzta  at the current Gauss point.
  double *dphidzta;
  
  /// Access value of dxi / dx  at the current Gauss point.
  double dxidx;
  /// Access value of dxi / dy  at the current Gauss point.
  double dxidy;
  /// Access value of dxi / dz  at the current Gauss point.
  double dxidz;

  /// Access value of deta / dx  at the current Gauss point.
  double detadx;
  /// Access value of deta / dy  at the current Gauss point.
  double detady;
  /// Access value of deta / dz  at the current Gauss point.
  double detadz;

  /// Access value of dzta / dx  at the current Gauss point.
  double dztadx;
  /// Access value of dzta / dy  at the current Gauss point.
  double dztady;
  /// Access value of dzta / dz  at the current Gauss point.
  double dztadz;

  /// Access value of the Gauss weight  at the current Gauss point.
  double wt;
  /// Access value of the mapping Jacobian.
  double jac;

  /// Access value of u at the current Gauss point.
  double uu;
  /// Access value of du / dx at the current Gauss point.
  double dudx;
  /// Access value of du / dy at the current Gauss point.
  double dudy;
  /// Access value of du / dz at the current Gauss point.
  double dudz;

  /// Access value of u_old at the current Gauss point.
  double uuold;
  /// Access value of du_old / dx at the current Gauss point.
  double duolddx;
  /// Access value of du_old / dy at the current Gauss point.
  double duolddy;
  /// Access value of du_old / dz at the current Gauss point.
  double duolddz;

  /// Access value of u_old_old at the current Gauss point.
  double uuoldold;
  /// Access value of du_old_old / dx at the current Gauss point.
  double duoldolddx;
  /// Access value of du_old_old / dy at the current Gauss point.
  double duoldolddy;
  /// Access value of du_old_old / dz at the current Gauss point.
  double duoldolddz;

  /// Access value of x coordinate in real space at the current Gauss point.
  double xx;
  /// Access value of y coordinate in real space at the current Gauss point.
  double yy;
  /// Access value of z coordinate in real space at the current Gauss point.
  double zz;

  /// Access value of the derivative of the basis function wrt to x at the current Gauss point.
  double * dphidx;
  /// Access value of the derivative of the basis function wrt to y at the current Gauss point.
  double * dphidy;
  /// Access value of the derivative of the basis function wrt to z at the current Gauss point.
  double * dphidz;

  /// Access volume of current element.
  const double vol() const {return volp;};
protected:
  /// Access a pointer to the coordinates of the Gauss points in canonical space.
  double *abscissa;
  /// Access a pointer to the Gauss weights.
  double *weight;
  /// Access a pointer to the xi coordinate at each Gauss point.
  double *xi;
  /// Access a pointer to the eta coordinate at each Gauss point.
  double *eta;
  /// Access a pointer to the zta coordinate at each Gauss point.
  double *zta;
  /// Access the number of Gauss weights.
  double *nwt;

  std::string stype;

  double volp;
  double canonical_vol;
};

/// Implementation of 2-D bilinear triangle element.
/** 3 node element with number of Gauss points specified in constructor, defaults to 1. */
class BasisLTri : public Basis {

 public:

  /// Constructor
  /** Number of Gauss points = sngp, defaults to 1. */
  BasisLTri(int n = 1){
  canonical_vol = .5;
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

  xi  = new double[ngp];
  eta = new double[ngp];
  nwt  = new double[ngp];

  if( ngp == 1 ){
    abscissa[0] = 1.0L / 3.0L;
    weight[0] = .5;

    xi[0] = eta[0] = abscissa[0];
    nwt[0] = weight[0];

  }else if (ngp == 3 ) {
    abscissa[0] = 1.0L / 2.0L;
    abscissa[1] = 1.0L / 2.0L;
    abscissa[2] = 0.;
    weight[0] = 1.0L / 6.0L;
    weight[1] = 1.0L / 6.0L;
    weight[2] = 1.0L / 6.0L;

    xi[0]  = abscissa[0];
    eta[0] = abscissa[0];
    nwt[0]  = weight[0];

    xi[1]  = abscissa[0];
    eta[1] = abscissa[2];
    nwt[1]  = weight[1];

    xi[2]  = abscissa[2];
    eta[2] = abscissa[0];
    nwt[2]  = weight[2];
  }else {
    std::cout<<"void BasisLTri::getBasis(int gp, double *x, double *y, double *u, double *uold)"<<std::endl
	     <<"   only ngp = 1 and 3 supported at this time"<<std::endl;
  }
};

  /// Destructor
  virtual ~BasisLTri(){
  delete [] phi;
  delete [] dphidxi;
  delete [] dphideta;
  delete [] dphidzta;
  delete [] dphidx;
  delete [] dphidy;
  delete [] dphidz;
  delete [] abscissa;
  delete [] weight;

  delete [] xi;
  delete [] eta;
  delete [] nwt;
};

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
					    ){
  int N = 3;  
  // gp irrelevent and unused


  // Calculate basis function and derivatives at nodel pts
  phi[0]=(1.0-xi[gp]-eta[gp]);
  phi[1]= xi[gp];
  phi[2]= eta[gp];

  wt = nwt[gp];

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
  double dzdxi = 0;
  double dzdeta = 0;

  for (int i=0; i < N; i++) {
    dxdxi += dphidxi[i] * x[i];
    dxdeta += dphideta[i] * x[i];
    dydxi += dphidxi[i] * y[i];
    dydeta += dphideta[i] * y[i];
    dzdxi += dphidxi[i] * z[i];
    dzdeta += dphideta[i] * z[i];
  }

  //jac = dxdxi * dydeta - dxdeta * dydxi;
  jac = sqrt( (dzdxi * dxdeta - dxdxi * dzdeta)*(dzdxi * dxdeta - dxdxi * dzdeta)
	     +(dydxi * dzdeta - dzdxi * dydeta)*(dydxi * dzdeta - dzdxi * dydeta)
	     +(dxdxi * dydeta - dxdeta * dydxi)*(dxdxi * dydeta - dxdeta * dydxi));
  volp = jac*canonical_vol;
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
    zz += z[i] * phi[i];
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
};

  BasisLTri* clone() const{ return new BasisLTri(*this); };

 public:
};

/// Implementation of 2-D bilinear quadrilateral element.
/** 4 node element with number of Gauss points specified in constructor, defaults to 4. */
class BasisLQuad : public Basis {

 public:

  /// Constructor
  /** Number of Gauss points = sngp, defaults to 4 (sngp refers to 1 dimension of a tensor product, ie sngp = 2 is really 4 Gauss points). */
  BasisLQuad(int n = 2) :sngp(n){
  canonical_vol = 4.;
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

  xi  = new double[ngp];
  eta = new double[ngp];
  nwt  = new double[ngp];


  if(2 == sngp){

    //cn right now, changing the order when ngp = 4 breaks all the quad tests
    //cn so we leave it for now...
    xi[0]  = abscissa[0];
    eta[0] = abscissa[0];
    nwt[0]  = weight[0] * weight[0];

    xi[1]  = abscissa[1];
    eta[1] = abscissa[0];
    nwt[1]  = weight[0] * weight[1];

    xi[2]  = abscissa[1];
    eta[2] = abscissa[1];
    nwt[2]  = weight[1] * weight[1];

    xi[3]  = abscissa[0];
    eta[3] = abscissa[1];
    nwt[3]  = weight[0] * weight[1];
  }
  else{
    int c = 0;
    for( int i = 0; i < sngp; i++ ){
      for( int j = 0; j < sngp; j++ ){
	xi[i+j+c]  = abscissa[i];
	eta[i+j+c] = abscissa[j];
	nwt[i+j+c]  = weight[i] * weight[j];
      }
      c = c + sngp - 1;
    }
  }
};

  /// Destructor
  ~BasisLQuad(){
  delete [] phi;
  delete [] dphidxi;
  delete [] dphideta;
  delete [] dphidzta;
  delete [] dphidx;
  delete [] dphidy;
  delete [] dphidz;
  delete [] abscissa;
  delete [] weight;

  delete [] xi;
  delete [] eta;
  delete [] nwt;
};

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
		 ) {

  // Calculate basis function and derivatives at nodal pts
  phi[0]=(1.0-xi[gp])*(1.0-eta[gp])/4.0;
  phi[1]=(1.0+xi[gp])*(1.0-eta[gp])/4.0;
  phi[2]=(1.0+xi[gp])*(1.0+eta[gp])/4.0;
  phi[3]=(1.0-xi[gp])*(1.0+eta[gp])/4.0;

  dphidxi[0]=-(1.0-eta[gp])/4.0;
  dphidxi[1]= (1.0-eta[gp])/4.0;
  dphidxi[2]= (1.0+eta[gp])/4.0;
  dphidxi[3]=-(1.0+eta[gp])/4.0;

  dphideta[0]=-(1.0-xi[gp])/4.0;
  dphideta[1]=-(1.0+xi[gp])/4.0;
  dphideta[2]= (1.0+xi[gp])/4.0;
  dphideta[3]= (1.0-xi[gp])/4.0;
  
  // Caculate basis function and derivative at GP.
  //std::cout<<x[0]<<" "<<x[1]<<" "<<x[2]<<" "<<x[3]<<std::endl;
  double dxdxi  = .25*( (x[1]-x[0])*(1.-eta[gp])+(x[2]-x[3])*(1.+eta[gp]) );
  double dxdeta = .25*( (x[3]-x[0])*(1.- xi[gp])+(x[2]-x[1])*(1.+ xi[gp]) );
  double dydxi  = .25*( (y[1]-y[0])*(1.-eta[gp])+(y[2]-y[3])*(1.+eta[gp]) );
  double dydeta = .25*( (y[3]-y[0])*(1.- xi[gp])+(y[2]-y[1])*(1.+ xi[gp]) );
  double dzdxi  = .25*( (z[1]-z[0])*(1.-eta[gp])+(z[2]-z[3])*(1.+eta[gp]) );
  double dzdeta = .25*( (z[3]-z[0])*(1.- xi[gp])+(z[2]-z[1])*(1.+ xi[gp]) );



  wt = nwt[gp];

  //jac = dxdxi * dydeta - dxdeta * dydxi;
  jac = sqrt( (dzdxi * dxdeta - dxdxi * dzdeta)*(dzdxi * dxdeta - dxdxi * dzdeta)
	     +(dydxi * dzdeta - dzdxi * dydeta)*(dydxi * dzdeta - dzdxi * dydeta)
	     +(dxdxi * dydeta - dxdeta * dydxi)*(dxdxi * dydeta - dxdeta * dydxi));

  volp = jac*canonical_vol;

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
    zz += z[i] * phi[i];
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
};

  /// Evaluate the basis functions at the specified at point xx, yy ,zz
  /** Evaluate the 3D basis functions at  xx, yy, zz and interpolate u.  True if xx, yy, zz is in this element. Returns val = u(xx,yy,zz)*/
  bool evalBasis( const double *x,  ///< x array (input) 
		  const double *y, ///< y array (input)  
		  const double *z, ///< z array (input)  
		  const double *u, ///< u (solution)  array (input)  
		  const double xx_, ///< xx 
		  const double yy_,///< yy
		  const double zz_,///< zz
		  double &val ///< return val
		  ){
  //for each direction we solve the a nonlinear least squares system for the barycentric coords

  // F1(xi,eta) = -xx + sum_k x(k)*phi(k) == 0
  // The Newton system gives J(v) dv = -F(v) is reformulated as:
  // J(v)^T J(v) dv = -J(v)^T F(v), where inv(J^T J) is calculated explicitly 

  const int niter = 3;
  const int nnode = 4;

  //initial iterate 
  double xi_ = 0.;
  double eta_ = 0.;

  for(int i = 0; i < niter; i++){
    // Calculate basis function and derivatives at iterate
    phi[0]=(1.0-xi_)*(1.0-eta_)/4.0;
    phi[1]=(1.0+xi_)*(1.0-eta_)/4.0;
    phi[2]=(1.0+xi_)*(1.0+eta_)/4.0;
    phi[3]=(1.0-xi_)*(1.0+eta_)/4.0;
    
    dphidxi[0]=-(1.0-eta_)/4.0;
    dphidxi[1]= (1.0-eta_)/4.0;
    dphidxi[2]= (1.0+eta_)/4.0;
    dphidxi[3]=-(1.0+eta_)/4.0;
    
    dphideta[0]=-(1.0-xi_)/4.0;
    dphideta[1]=-(1.0+xi_)/4.0;
    dphideta[2]= (1.0+xi_)/4.0;
    dphideta[3]= (1.0-xi_)/4.0;    
    
    double f0 = -xx_;
    double f1 = -yy_;
    double f2 = -zz_;
    double j00 = 0.;
    double j01 = 0.;
    double j10 = 0.;
    double j11 = 0.;
    double j20 = 0.;
    double j21 = 0.;

    //residual F and elements of J
    for(int k = 0; k < nnode; k++){
      f0 = f0 +x[k]*phi[k];
      f1 = f1 +y[k]*phi[k];
      f2 = f2 +z[k]*phi[k];
      j00=j00 +x[k]*dphidxi[k];
      j01=j01+x[k]*dphideta[k];
      j10=j10+y[k]*dphidxi[k];
      j11=j11+y[k]*dphideta[k];
      j20=j20+z[k]*dphidxi[k];
      j21=j21+z[k]*dphideta[k];
    }//k
    
    //J(v)^T F(v)
    double jtf0=j00*f0+j10*f1+j20*f2;
    double jtf1=j01*f0+j11*f1+j21*f2;
    
    // inv(J^T J)
    double a=j00*j00+j10*j10+j20*j20;
    double b=j00*j01+j10*j11+j20*j21;
    double d=j01*j01+j11*j11+j21*j21;
    double deti=a*d-b*b;

    // inv(J^T J) * -J(v)^T F(v)
    double dxi=-(d/deti*jtf0-b/deti*jtf1);
    double deta=-(-b/deti*jtf0+a/deti*jtf1);

    xi_=dxi+xi_;
    eta_=deta+eta_;
    
  }//i
  //std::cout<<xi_<<" "<<eta_<<std::endl<<std::endl;

  //cn hack here, need to think about a reasonable number here
  double small = 1e-8;

  if( (std::fabs(xi_) > (1.+small))||(std::fabs(eta_) > (1.+small)) ) return false;
  
  phi[0]=(1.0-xi_)*(1.0-eta_)/4.0;
  phi[1]=(1.0+xi_)*(1.0-eta_)/4.0;
  phi[2]=(1.0+xi_)*(1.0+eta_)/4.0;
  phi[3]=(1.0-xi_)*(1.0+eta_)/4.0;

  val = 0.;
  for(int k = 0; k < nnode; k++){
    val = val + u[k]*phi[k];
  }//k

  return true;
};

  BasisLQuad* clone() const{ return new BasisLQuad(*this); };

 public:
  // Variables that are calculated at the gauss point
  int sngp;
};

/// Implementation of 2-D biquadratic triangle element.
/** 6 node element with 3 Gauss points. */
class BasisQTri : public Basis {

 public:

  /// Constructor
  BasisQTri(int n = 3){
  canonical_vol = .5;
  ngp = n;
  phi = new double[6];
  dphidxi = new double[6];
  dphideta = new double[6];
  dphidzta = new double[6];
  dphidx = new double[6];
  dphidy = new double[6];
  dphidz = new double[6];
  weight = new double[ngp];
  abscissa = new double[ngp];
  abscissa[0] = 2.0L / 3.0L;
  abscissa[1] = 1.0L / 6.0L;
  abscissa[2] = 0.;
  weight[0] = 1.0L / 6.0L;
  weight[1] = 1.0L / 6.0L;
  weight[2] = 1.0L / 6.0L;

  xi  = new double[ngp];
  eta = new double[ngp];
  nwt  = new double[ngp];
#if 0
  wt = 1.0L / 6.0L;
  if (gp==0) {xi = 2.0L/3.0L; eta=1.0L/6.0L;}
  if (gp==1) {xi = 1.0L/6.0L; eta=2.0L/3.0L;}
  if (gp==2) {xi = 1.0L/6.0L; eta=1.0L/6.0L;}
#endif
  if( 3 == ngp ){
    xi[0]  = abscissa[0];
    eta[0] = abscissa[1];
    nwt[0]  = weight[0];
    xi[1]  = abscissa[1]; 
    eta[1] = abscissa[0];
    nwt[1]  = weight[1];
    xi[2]  = abscissa[1]; 
    eta[2] = abscissa[1];
    nwt[2]  = weight[2];
  }
  else {
    std::cout<<"void BasisQTri::getBasis(int gp, double *x, double *y, double *u, double *uold)"<<std::endl
	     <<"   only ngp =  3 supported at this time"<<std::endl;
  } 
};

  /// Destructor
  ~BasisQTri() {
  delete [] phi;
  delete [] dphidxi;
  delete [] dphideta;
  delete [] dphidzta;
  delete [] dphidx;
  delete [] dphidy;
  delete [] dphidz;
  delete [] weight;
  delete [] abscissa;

  delete [] xi;
  delete [] eta;
  delete [] nwt;
};

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
					    ){
  int N = 6;


  // Calculate basis function and derivatives at nodel pts
  phi[0]=2.0 * (1.0 - xi[gp] - eta[gp]) * (0.5 - xi[gp] - eta[gp]);
  phi[1]= 2.0 * xi[gp] * (xi[gp] - 0.5);
  phi[2]= 2.0 * eta[gp] * (eta[gp] - 0.5);
  phi[3]=4.0 * (1.0 - xi[gp] - eta[gp]) * xi[gp];
  phi[4]= 4.0 * xi[gp] * eta[gp];
  phi[5]= 4.0 * (1.0 - xi[gp] - eta[gp]) * eta[gp];
  dphidxi[0]=-2.0 * (0.5 - xi[gp] - eta[gp]) - 2.0 * (1.0 - xi[gp] - eta[gp]);
  dphidxi[1]= 2.0 * (xi[gp] - 0.5) + 2.0 * xi[gp];
  dphidxi[2]= 0.0;
  dphidxi[3]=-4.0 * xi[gp] + 4.0 * (1.0 - xi[gp] - eta[gp]);
  dphidxi[4]= 4.0 * eta[gp];
  dphidxi[5]= -4.0 * eta[gp];
  dphideta[0]=-2.0 * (0.5 - xi[gp] - eta[gp]) - 2.0 * (1.0 - xi[gp] - eta[gp]);
  dphideta[1]= 0.0;
  dphideta[2]= 2.0 * eta[gp] + 2.0 * (eta[gp] - 0.5);
  dphideta[3]=-4.0 * xi[gp];
  dphideta[4]= 4.0 * xi[gp];
  dphideta[5]= 4.0 * (1.0 - xi[gp] - eta[gp]) - 4.0 * eta[gp];
  dphidzta[0]= 0.0;
  dphidzta[1]= 0.0;
  dphidzta[2]= 0.0;
  dphidzta[3]= 0.0;
  dphidzta[4]= 0.0;
  dphidzta[5]= 0.0;
  
  wt = nwt[gp];


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
  volp = jac*canonical_vol;
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
};


  BasisQTri* clone() const{ return new BasisQTri(*this); };

 public:

};

/// Implementation of 2-D biquadratic quadrilateral element.
/** 9 node element with 9 Gauss points. */
class BasisQQuad : public Basis {

 public:

  /// Constructor
  BasisQQuad(int n = 3) :sngp(n){
  canonical_vol = 4.;
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

  xi  = new double[ngp];
  eta = new double[ngp];
  nwt  = new double[ngp]; 

  double ab[3];// = new double[3];
  ab[0] = abscissa[0];
  ab[1] = abscissa[2];
  ab[2] = abscissa[1];

  double w[3];// = new double[3];
  w[0] = weight[0];
  w[1] = weight[2];
  w[2] = weight[1];
 
  if(3 == sngp){

    xi[0] = ab[0];
    eta[0]  = ab[0];
    nwt[0]  = w[0] * w[0];
    xi[1] = ab[1];
    eta[1] = ab[0];
    nwt[1] = w[1] * w[0];
    xi[2] = ab[1];
    eta[2]  = ab[1];
    nwt[2]  = w[1] * w[1];
    xi[3] = ab[0];
    eta[3]  = ab[1];
    nwt[3]  = w[0] * w[1];
    xi[4] = ab[2];
    eta[4] = ab[0];
    nwt[4] = w[2] * w[0];
    xi[5] = ab[1];
    eta[5] = ab[2];
    nwt[5] = w[1] * w[2];
    xi[6] = ab[2];
    eta[6] = ab[1];
    nwt[6] = w[2] * w[1];
    xi[7] = ab[0];
    eta[7] = ab[2];
    nwt[7] = w[0] * w[2];
    xi[8] = ab[2];
    eta[8] = ab[2];
    nwt[8] = w[2] * w[2];
  }
  else{ 
    int c = 0;
    for( int i = 0; i < sngp; i++ ){
      for( int j = 0; j < sngp; j++ ){
	xi[i+j+c]  = abscissa[i];
	eta[i+j+c] = abscissa[j];
	nwt[i+j+c]  = weight[i] * weight[j];
      }
      c = c + sngp - 1;
    }
  }
};

  /// Destructor
  ~BasisQQuad() {
  delete [] phi;
  delete [] dphidxi;
  delete [] dphideta;
  delete [] dphidzta;
  delete [] dphidx;
  delete [] dphidy;
  delete [] dphidz;
  delete [] abscissa;
  delete [] weight;

  delete [] xi;
  delete [] eta;
  delete [] nwt;
};

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
					    ) {
  const double phi0_xi = -xi[gp]*(1-xi[gp])/2;
  const double phi1_xi = xi[gp]*(1+xi[gp])/2;
  const double phi2_xi = 1-xi[gp]*xi[gp];

  const double phi0_eta = -eta[gp]*(1-eta[gp])/2;
  const double phi1_eta = eta[gp]*(1+eta[gp])/2;
  const double phi2_eta = 1-eta[gp]*eta[gp];
 
  //double phi_xi[3] = [phi1_xi,phi2_xi,phi3_xi];
  //double phi_eta[3] = [phi1_eta,phi2_eta,phi3_eta];

  //printf("gp = %d\n",gp);
  //printf("_xi %f %f %f:\n", phi1_xi, phi2_xi, phi3_xi);
  //printf("_eta %f %f %f:\n\n", phi1_eta, phi2_eta, phi3_eta);

  phi[0] = phi0_xi * phi0_eta;
  phi[1] = phi1_xi * phi0_eta;
  phi[2] = phi1_xi * phi1_eta;
  phi[3] = phi0_xi * phi1_eta;
  
  phi[4] = phi2_xi * phi0_eta;
  phi[5] = phi1_xi * phi2_eta;
  phi[6] = phi2_xi * phi1_eta;
  phi[7] = phi0_xi * phi2_eta;
  
  phi[8] = phi2_xi * phi2_eta;

  const double dphi0dxi = (-1+2*xi[gp])/2;
  const double dphi1dxi = (1+2*xi[gp])/2;
  const double dphi2dxi = (-2*xi[gp]);

  const double dphi0deta = (-1+2*eta[gp])/2;
  const double dphi1deta = (1+2*eta[gp])/2;
  const double dphi2deta = -2*eta[gp];
  
  
  dphidxi[0] = dphi0dxi * phi0_eta;
  dphidxi[1] = dphi1dxi * phi0_eta;
  dphidxi[2] = dphi1dxi * phi1_eta;
  dphidxi[3] = dphi0dxi * phi1_eta;
  
  dphidxi[4] = dphi2dxi * phi0_eta;
  dphidxi[5] = dphi1dxi * phi2_eta;
  dphidxi[6] = dphi2dxi * phi1_eta;
  dphidxi[7] = dphi0dxi * phi2_eta;
  
  dphidxi[8] = dphi2dxi * phi2_eta;


  dphideta[0] = phi0_xi * dphi0deta;
  dphideta[1] = phi1_xi * dphi0deta;
  dphideta[2] = phi1_xi * dphi1deta;
  dphideta[3] = phi0_xi * dphi1deta;
  
  dphideta[4] = phi2_xi * dphi0deta;
  dphideta[5] = phi1_xi * dphi2deta;
  dphideta[6] = phi2_xi * dphi1deta;
  dphideta[7] = phi0_xi * dphi2deta;
  
  dphideta[8] = phi2_xi * dphi2deta;

  wt = nwt[gp];


 
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

  double dxdxi  = dphi0dxi *(x[0]*phi0_eta+x[3]*phi1_eta + x[7]*phi2_eta) + 
                   dphi1dxi *(x[1]*phi0_eta+x[2]*phi1_eta + x[5]*phi2_eta) + 
                   dphi2dxi *(x[4]*phi0_eta+x[6]*phi1_eta + x[8]*phi2_eta); 
 
  double dydxi =  dphi0dxi *(y[0]*phi0_eta+y[3]*phi1_eta + y[7]*phi2_eta) + 
                   dphi1dxi *(y[1]*phi0_eta+y[2]*phi1_eta + y[5]*phi2_eta) + 
                   dphi2dxi *(y[4]*phi0_eta+y[6]*phi1_eta + y[8]*phi2_eta); 
  
  double dxdeta = dphi0deta *(x[0]*phi0_xi+x[1]*phi1_xi + x[4]*phi2_xi) + 
                   dphi1deta *(x[3]*phi0_xi+x[2]*phi1_xi + x[6]*phi2_xi) + 
                   dphi2deta *(x[7]*phi0_xi+x[5]*phi1_xi + x[8]*phi2_xi); 
 
  double dydeta = dphi0deta *(y[0]*phi0_xi+y[1]*phi1_xi + y[4]*phi2_xi) + 
                   dphi1deta *(y[3]*phi0_xi+y[2]*phi1_xi + y[6]*phi2_xi) + 
                   dphi2deta *(y[7]*phi0_xi+y[5]*phi1_xi + y[8]*phi2_xi); 
 
/*printf("%f + %f + %f\n",dphi1dxi *(x[0]*phi1_eta+x[3]*phi2_eta + x[7]*phi3_eta),dphi2dxi *(x[1]*phi1_eta+x[2]*phi2_eta + x[5]*phi3_eta),dphi3dxi *(x[4]*phi1_eta+x[6]*phi2_eta + x[8]*phi3_eta));*/

//  printf("%f %f %f %f\n",dxdxi,dydeta,dxdeta,dydxi);
  jac = dxdxi * dydeta - dxdeta * dydxi;
  //printf("jacobian = %f\n\n",jac);
//   if (jac <= 0){
//     printf("\n\n\n\n\n\nnonpositive jacobian \n\n\n\n\n\n");
//   }
 
//printf("_deta : %f %f %f\n",phi1_eta,phi2_eta,phi3_eta);
 // printf("_deta : %f %f %f\n",dphi1deta,dphi2deta,dphi3deta);
//printf("_xi : %f %f %f\n\n",dphi1dxi,dphi2dxi,dphi3dxi);

  volp = jac*canonical_vol;

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
};    

  BasisQQuad* clone() const{ return new BasisQQuad(*this); };

 public:
  // Variables that are calculated at the gauss point
  int sngp;
};


/// Implementation of 3-D bilinear hexahedral element.
/** 8 node element with number of Gauss points specified in constructor, defaults to 8. */
class BasisLHex : public Basis {

 public:

  /// Constructor
  /** Number of Gauss points = sngp (sngp refers to 1 dimension of a tensor product, ie sngp = 2 is really 8 Gauss points). */
  BasisLHex(int n = 2): sngp(n){
  canonical_vol = 8.;
  ngp = sngp*sngp*sngp;
  phi = new double[ngp];
  dphidxi = new double[8];
  dphideta = new double[8];
  dphidzta = new double[8];
  dphidx = new double[8];
  dphidy = new double[8];
  dphidz = new double[8];

  abscissa = new double[sngp];
  weight = new double[sngp];
  setN(sngp, abscissa, weight);

  xi  = new double[ngp];
  eta = new double[ngp];
  zta = new double[ngp];
  nwt  = new double[ngp];

  if ( 2 == sngp ){
    xi[0] = abscissa[0];  // 0, 0, 0
    eta[0] = abscissa[0];
    zta[0] = abscissa[0];
    nwt[0] = weight[0] * weight[0] * weight[0];
    xi[1] = abscissa[1]; // 1, 0, 0
    eta[1] = abscissa[0];
    zta[1] = abscissa[0];
    nwt[1] = weight[0] * weight[1] * weight[0];
    xi[2] = abscissa[1]; // 1, 1, 0
    eta[2] = abscissa[1];
    zta[2] = abscissa[0];
    nwt[2] = weight[1] * weight[1] * weight[0];
    xi[3] = abscissa[0];  //0, 1, 0
    eta[3] = abscissa[1];
    zta[3] = abscissa[0];
    nwt[3] = weight[0] * weight[1] * weight[0];
    xi[4] = abscissa[0];  // 0, 0, 1
    eta[4] = abscissa[0];
    zta[4] = abscissa[1];
    nwt[4] = weight[0] * weight[0] * weight[1];
    xi[5] = abscissa[1]; // 1, 0, 1
    eta[5] = abscissa[0];
    zta[5] = abscissa[1];
    nwt[5] = weight[0] * weight[1] * weight[1];
    xi[6] = abscissa[1]; // 1, 1, 1
    eta[6] = abscissa[1];
    zta[6] = abscissa[1];
    nwt[6] = weight[1] * weight[1] * weight[1];
    xi[7] = abscissa[0];  //0, 1, 1
    eta[7] = abscissa[1];
    zta[7] = abscissa[1];
    nwt[7] = weight[0] * weight[1] * weight[1];
  }
  else{
    int c = 0;
    for( int i = 0; i < sngp; i++ ){
      for( int j = 0; j < sngp; j++ ){
	for( int k = 0; k < sngp; k++ ){
	  xi[i+j+k+c]  = abscissa[i];
	  eta[i+j+k+c] = abscissa[j];
	  zta[i+j+k+c] = abscissa[k];
	  nwt[i+j+k+c]  = weight[i] * weight[j] * weight[k]; 
	}   
	c = c + sngp - 1;
      }
      c = c + sngp - 1;
    }
  }
};

  /// Destructor
  ~BasisLHex() {
  delete [] phi;
  delete [] dphideta;
  delete [] dphidxi;
  delete [] dphidzta;
  delete [] dphidx;
  delete [] dphidy;
  delete [] dphidz;
  delete [] abscissa;
  delete [] weight;

  delete [] xi;
  delete [] eta;
  delete [] zta;
  delete [] nwt;
};

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
		 ) 
  { // Calculate basis function and derivatives at nodal pts
   phi[0]   =  0.125 * (1.0 - xi[gp]) * (1.0 - eta[gp]) * (1.0 - zta[gp]);
   phi[1]   =  0.125 * (1.0 + xi[gp]) * (1.0 - eta[gp]) * (1.0 - zta[gp]);
   phi[2]   =  0.125 * (1.0 + xi[gp]) * (1.0 + eta[gp]) * (1.0 - zta[gp]);
   phi[3]   =  0.125 * (1.0 - xi[gp]) * (1.0 + eta[gp]) * (1.0 - zta[gp]);
   phi[4]   =  0.125 * (1.0 - xi[gp]) * (1.0 - eta[gp]) * (1.0 + zta[gp]);
   phi[5]   =  0.125 * (1.0 + xi[gp]) * (1.0 - eta[gp]) * (1.0 + zta[gp]);
   phi[6]   =  0.125 * (1.0 + xi[gp]) * (1.0 + eta[gp]) * (1.0 + zta[gp]);
   phi[7]   =  0.125 * (1.0 - xi[gp]) * (1.0 + eta[gp]) * (1.0 + zta[gp]);

   // ksi-derivatives
   dphidxi[0] = -0.125 * (1.0 - eta[gp]) * (1.0 - zta[gp]);
   dphidxi[1] =  0.125 * (1.0 - eta[gp]) * (1.0 - zta[gp]);
   dphidxi[2] =  0.125 * (1.0 + eta[gp]) * (1.0 - zta[gp]);
   dphidxi[3] = -0.125 * (1.0 + eta[gp]) * (1.0 - zta[gp]);
   dphidxi[4] = -0.125 * (1.0 - eta[gp]) * (1.0 + zta[gp]);
   dphidxi[5] =  0.125 * (1.0 - eta[gp]) * (1.0 + zta[gp]);
   dphidxi[6] =  0.125 * (1.0 + eta[gp]) * (1.0 + zta[gp]);
   dphidxi[7] = -0.125 * (1.0 + eta[gp]) * (1.0 + zta[gp]);

   // eta-derivatives
   dphideta[0] = -0.125 * (1.0 - xi[gp]) * (1.0 - zta[gp]);
   dphideta[1] = -0.125 * (1.0 + xi[gp]) * (1.0 - zta[gp]);
   dphideta[2] =  0.125 * (1.0 + xi[gp]) * (1.0 - zta[gp]);
   dphideta[3] =  0.125 * (1.0 - xi[gp]) * (1.0 - zta[gp]);
   dphideta[4] = -0.125 * (1.0 - xi[gp]) * (1.0 + zta[gp]);
   dphideta[5] = -0.125 * (1.0 + xi[gp]) * (1.0 + zta[gp]);
   dphideta[6] =  0.125 * (1.0 + xi[gp]) * (1.0 + zta[gp]);
   dphideta[7] =  0.125 * (1.0 - xi[gp]) * (1.0 + zta[gp]);

   // zeta-derivatives
   dphidzta[0] = -0.125 * (1.0 - xi[gp]) * (1.0 - eta[gp]);
   dphidzta[1] = -0.125 * (1.0 + xi[gp]) * (1.0 - eta[gp]);
   dphidzta[2] = -0.125 * (1.0 + xi[gp]) * (1.0 + eta[gp]);
   dphidzta[3] = -0.125 * (1.0 - xi[gp]) * (1.0 + eta[gp]);
   dphidzta[4] =  0.125 * (1.0 - xi[gp]) * (1.0 - eta[gp]);
   dphidzta[5] =  0.125 * (1.0 + xi[gp]) * (1.0 - eta[gp]);
   dphidzta[6] =  0.125 * (1.0 + xi[gp]) * (1.0 + eta[gp]);
   dphidzta[7] =  0.125 * (1.0 - xi[gp]) * (1.0 + eta[gp]);
  
  // Caculate basis function and derivative at GP.
  double dxdxi  = 0.125*( (x[1]-x[0])*(1.-eta[gp])*(1.-zta[gp]) + (x[2]-x[3])*(1.+eta[gp])*(1.-zta[gp]) 
	+ (x[5]-x[4])*(1.-eta[gp])*(1.+zta[gp]) + (x[6]-x[7])*(1.+eta[gp])*(1.+zta[gp]) );
  double dxdeta = 0.125*( (x[3]-x[0])*(1.- xi[gp])*(1.-zta[gp]) + (x[2]-x[1])*(1.+ xi[gp])*(1.-zta[gp]) 
	+ (x[7]-x[4])*(1.- xi[gp])*(1.+zta[gp]) + (x[6]-x[5])*(1.+ xi[gp])*(1.+zta[gp]) );
  double dxdzta = 0.125*( (x[4]-x[0])*(1.- xi[gp])*(1.-eta[gp]) + (x[5]-x[1])*(1.+ xi[gp])*(1.-eta[gp])
	+ (x[6]-x[2])*(1.+ xi[gp])*(1.+eta[gp]) + (x[7]-x[3])*(1.- xi[gp])*(1.+eta[gp]) );

  double dydxi  = 0.125*( (y[1]-y[0])*(1.-eta[gp])*(1.-zta[gp]) + (y[2]-y[3])*(1.+eta[gp])*(1.-zta[gp])
	+ (y[5]-y[4])*(1.-eta[gp])*(1.+zta[gp]) + (y[6]-y[7])*(1.+eta[gp])*(1.+zta[gp]) );
  double dydeta = 0.125*( (y[3]-y[0])*(1.- xi[gp])*(1.-zta[gp]) + (y[2]-y[1])*(1.+ xi[gp])*(1.-zta[gp]) 
	+ (y[7]-y[4])*(1.- xi[gp])*(1.+zta[gp]) + (y[6]-y[5])*(1.+ xi[gp])*(1.+zta[gp]) );
  double dydzta = 0.125*( (y[4]-y[0])*(1.- xi[gp])*(1.-eta[gp]) + (y[5]-y[1])*(1.+ xi[gp])*(1.-eta[gp])
	+ (y[6]-y[2])*(1.+ xi[gp])*(1.+eta[gp]) + (y[7]-y[3])*(1.- xi[gp])*(1.+eta[gp]) );

  double dzdxi  = 0.125*( (z[1]-z[0])*(1.-eta[gp])*(1.-zta[gp]) + (z[2]-z[3])*(1.+eta[gp])*(1.-zta[gp])
	+ (z[5]-z[4])*(1.-eta[gp])*(1.+zta[gp]) + (z[6]-z[7])*(1.+eta[gp])*(1.+zta[gp]) );
  double dzdeta = 0.125*( (z[3]-z[0])*(1.- xi[gp])*(1.-zta[gp]) + (z[2]-z[1])*(1.+ xi[gp])*(1.-zta[gp]) 
	+ (z[7]-z[4])*(1.- xi[gp])*(1.+zta[gp]) + (z[6]-z[5])*(1.+ xi[gp])*(1.+zta[gp]) );
  double dzdzta = 0.125*( (z[4]-z[0])*(1.- xi[gp])*(1.-eta[gp]) + (z[5]-z[1])*(1.+ xi[gp])*(1.-eta[gp])
	+ (z[6]-z[2])*(1.+ xi[gp])*(1.+eta[gp]) + (z[7]-z[3])*(1.- xi[gp])*(1.+eta[gp]) );

  wt = nwt[gp];

  jac = dxdxi*(dydeta*dzdzta - dydzta*dzdeta) - dxdeta*(dydxi*dzdzta - dydzta*dzdxi) 
	                                      + dxdzta*(dydxi*dzdeta - dydeta*dzdxi);

  volp = jac*canonical_vol;
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
    dphidx[i] = dphidxi[i]*dxidx+dphideta[i]*detadx+dphidzta[i]*dztadx;
    dphidy[i] = dphidxi[i]*dxidy+dphideta[i]*detady+dphidzta[i]*dztady;
    dphidz[i] = dphidxi[i]*dxidz+dphideta[i]*detadz+dphidzta[i]*dztadz;
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
}; 

  /// Evaluate the basis functions at the specified at point xx, yy ,zz
  /** Evaluate the 3D basis functions at  xx, yy, zz and interpolate u.  True if xx, yy, zz is in this element. Returns val = u(xx,yy,zz)*/
  bool evalBasis( const double *x,  ///< x array (input) 
		  const double *y, ///< y array (input)  
		  const double *z, ///< z array (input)  
		  const double *u, ///< u (solution)  array (input)  
		  const double xx_, ///< xx 
		  const double yy_,///< yy
		  const double zz_,///< zz
		  double &val ///< return val
		  ){
  //for each direction we solve the a nonlinear system for the barycentric coords

  // F1(xi,eta) = -xx + sum_k x(k)*phi(k) == 0
  // The Newton system gives J(v) dv = -F(v) , where inv(J) is calculated explicitly 

  const int niter = 3;
  const int nnode = 8;

  //initial iterate 
  double xi_ = 0.0;
  double eta_ = 0.0;
  double zta_ = 0.0;

  for(int i = 0; i < niter; i++){
    // Calculate basis function and derivatives at iterate
    phi[0]   =  0.125 * (1.0 - xi_) * (1.0 - eta_) * (1.0 - zta_);
    phi[1]   =  0.125 * (1.0 + xi_) * (1.0 - eta_) * (1.0 - zta_);
    phi[2]   =  0.125 * (1.0 + xi_) * (1.0 + eta_) * (1.0 - zta_);
    phi[3]   =  0.125 * (1.0 - xi_) * (1.0 + eta_) * (1.0 - zta_);
    phi[4]   =  0.125 * (1.0 - xi_) * (1.0 - eta_) * (1.0 + zta_);
    phi[5]   =  0.125 * (1.0 + xi_) * (1.0 - eta_) * (1.0 + zta_);
    phi[6]   =  0.125 * (1.0 + xi_) * (1.0 + eta_) * (1.0 + zta_);
    phi[7]   =  0.125 * (1.0 - xi_) * (1.0 + eta_) * (1.0 + zta_);
    
    dphidxi[0] = -0.125 * (1.0 - eta_) * (1.0 - zta_);
    dphidxi[1] =  0.125 * (1.0 - eta_) * (1.0 - zta_);
    dphidxi[2] =  0.125 * (1.0 + eta_) * (1.0 - zta_);
    dphidxi[3] = -0.125 * (1.0 + eta_) * (1.0 - zta_);
    dphidxi[4] = -0.125 * (1.0 - eta_) * (1.0 + zta_);
    dphidxi[5] =  0.125 * (1.0 - eta_) * (1.0 + zta_);
    dphidxi[6] =  0.125 * (1.0 + eta_) * (1.0 + zta_);
    dphidxi[7] = -0.125 * (1.0 + eta_) * (1.0 + zta_);
    
    dphideta[0] = -0.125 * (1.0 - xi_) * (1.0 - zta_);
    dphideta[1] = -0.125 * (1.0 + xi_) * (1.0 - zta_);
    dphideta[2] =  0.125 * (1.0 + xi_) * (1.0 - zta_);
    dphideta[3] =  0.125 * (1.0 - xi_) * (1.0 - zta_);
    dphideta[4] = -0.125 * (1.0 - xi_) * (1.0 + zta_);
    dphideta[5] = -0.125 * (1.0 + xi_) * (1.0 + zta_);
    dphideta[6] =  0.125 * (1.0 + xi_) * (1.0 + zta_);
    dphideta[7] =  0.125 * (1.0 - xi_) * (1.0 + zta_);
    
    dphidzta[0] = -0.125 * (1.0 - xi_) * (1.0 - eta_);
    dphidzta[1] = -0.125 * (1.0 + xi_) * (1.0 - eta_);
    dphidzta[2] = -0.125 * (1.0 + xi_) * (1.0 + eta_);
    dphidzta[3] = -0.125 * (1.0 - xi_) * (1.0 + eta_);
    dphidzta[4] =  0.125 * (1.0 - xi_) * (1.0 - eta_);
    dphidzta[5] =  0.125 * (1.0 + xi_) * (1.0 - eta_);
    dphidzta[6] =  0.125 * (1.0 + xi_) * (1.0 + eta_);
    dphidzta[7] =  0.125 * (1.0 - xi_) * (1.0 + eta_);
  
    double f0 = -xx_;
    double f1 = -yy_;
    double f2 = -zz_;
    double j00 = 0.;
    double j01 = 0.;
    double j02 = 0.;
    double j10 = 0.;
    double j11 = 0.;
    double j12 = 0.;
    double j20 = 0.;
    double j21 = 0.;
    double j22 = 0.;

    //residual F and elements of J
    for(int k = 0; k < nnode; k++){
      f0 = f0 +x[k]*phi[k];
      f1 = f1 +y[k]*phi[k];
      f2 = f2 +z[k]*phi[k];
      j00=j00 +x[k]*dphidxi[k];
      j01=j01 +x[k]*dphideta[k];
      j02=j02 +x[k]*dphidzta[k];
      j10=j10 +y[k]*dphidxi[k];
      j11=j11 +y[k]*dphideta[k];
      j12=j12 +y[k]*dphidzta[k];
      j20=j20 +z[k]*dphidxi[k];
      j21=j21 +z[k]*dphideta[k];
      j22=j22 +z[k]*dphidzta[k];
    }//k

    double deti=j00*j11*j22 + j10*j21*j02 + j20*j01*j12 - j00*j21*j12 - j20*j11*j02 - j10*j01*j22;
    
    double a00=j11*j22-j12*j21;
    double a01=j02*j21-j01*j22;
    double a02=j01*j12-j02*j11;
    
    double a10=j12*j20-j10*j22;
    double a11=j00*j22-j02*j20;
    double a12=j02*j10-j00*j12;
    
    double a20=j10*j21-j11*j20;
    double a21=j01*j20-j00*j21;
    double a22=j00*j11-j01*j10;
    
    double dxi= -(a00*f0+a01*f1+a02*f2)/deti;
    double deta= -(a10*f0+a11*f1+a12*f2)/deti;
    double dzta= -(a20*f0+a21*f1+a22*f2)/deti;
    
    xi_=dxi+xi_;
    eta_=deta+eta_;
    zta_=dzta+zta_;
    
  }//i
  //std::cout<<xi_<<" "<<eta_<<" "<<zta_<<std::endl;

  //cn hack here, need to think about a reasonable number here
  double small = 1e-8;

  if( (std::fabs(xi_) > (1.+small))||(std::fabs(eta_) > (1.+small))||(std::fabs(zta_) > (1.+small)) ) return false;

  phi[0]   =  0.125 * (1.0 - xi_) * (1.0 - eta_) * (1.0 - zta_);
  phi[1]   =  0.125 * (1.0 + xi_) * (1.0 - eta_) * (1.0 - zta_);
  phi[2]   =  0.125 * (1.0 + xi_) * (1.0 + eta_) * (1.0 - zta_);
  phi[3]   =  0.125 * (1.0 - xi_) * (1.0 + eta_) * (1.0 - zta_);
  phi[4]   =  0.125 * (1.0 - xi_) * (1.0 - eta_) * (1.0 + zta_);
  phi[5]   =  0.125 * (1.0 + xi_) * (1.0 - eta_) * (1.0 + zta_);
  phi[6]   =  0.125 * (1.0 + xi_) * (1.0 + eta_) * (1.0 + zta_);
  phi[7]   =  0.125 * (1.0 - xi_) * (1.0 + eta_) * (1.0 + zta_);
    
  val = 0.;
  for(int k = 0; k < nnode; k++){
    val = val + u[k]*phi[k];
  }//k

  return true;
};
 
  BasisLHex* clone() const{ return new BasisLHex(*this); };

  // Calculates the values of u and x at the specified gauss point
  void getBasis(const int gp,    ///< current Gauss point (input)
		const double *x,    ///< x array (input) 
		const double *y   ///< y array (input) 
		){
  std::cout<<"BasisLHex::getBasis(int gp, double *x, double *y) is not implemented"<<std::endl;
  //exit(0);
};

 public:
  // Variables that are calculated at the gauss point
  int sngp;
};

/// Implementation of 3-D bilinear tetrahedral element.
/** 4 node element with 4 Gauss points. */
class BasisLTet : public Basis {

 public:

  /// Constructor
  BasisLTet(){
  //cn we can have 1 or 4 guass points; 4 will be default
  canonical_vol = 1./6.;
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
  dphidx = new double[ngp];
  dphidy = new double[ngp];
  dphidz = new double[ngp];
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

  xi  = new double[ngp];
  eta = new double[ngp];
  zta = new double[ngp];
  nwt  = new double[ngp];

#if 0
  if(0 == gp){
    xi = abscissa[0];  // 0, 0, 0
    eta = abscissa[0];
    zta = abscissa[0];
    //wt = weight[0] * weight[0] * weight[0];
  }else if (1 == gp){
    xi = abscissa[1]; // 1, 0, 0
    eta = abscissa[0];
    zta = abscissa[0];
    //wt = weight[0] * weight[1] * weight[0];
  }else if (2 == gp){
    xi = abscissa[0]; // 1, 1, 0
    eta = abscissa[1];
    zta = abscissa[0];
    //wt = weight[1] * weight[1] * weight[0];
  }else if (3 == gp){
    xi = abscissa[0];  //0, 1, 0
    eta = abscissa[0];
    zta = abscissa[1];
    //wt = weight[0] * weight[1] * weight[0];
  } 
  wt=weight[0];//cn each wt = .25/6=0.041666666667
#endif

  xi[0] = abscissa[0];  // 0, 0, 0
  eta[0]  = abscissa[0];
  zta[0]  = abscissa[0];
  nwt[0]  = weight[0];
  xi[1] = abscissa[1]; // 1, 0, 0
  eta[1] = abscissa[0];
  zta[1] = abscissa[0];
  nwt[1]  = weight[0];
  xi[2]  = abscissa[0]; // 1, 1, 0
  eta[2]  = abscissa[1];
  zta[2]  = abscissa[0];
  nwt[2]  = weight[0];
  xi[3]= abscissa[0];  //0, 1, 0
  eta[3] = abscissa[0];
  zta [3]= abscissa[1];
  nwt[3]  = weight[0];
};
//   BasisLTet(int sngp);

  /// Destructor
  ~BasisLTet(){
  delete [] phi;
  delete [] dphideta;
  delete [] dphidxi;
  delete [] dphidzta;
  delete [] dphidx;
  delete [] dphidy;
  delete [] dphidz;
  delete [] abscissa;
  delete [] weight;

  delete [] xi;
  delete [] eta;
  delete [] zta;
  delete [] nwt;
};

  BasisLTet* clone() const{ return new BasisLTet(*this); };

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
		 )
{
  // Calculate basis function and derivatives at nodal pts
  phi[0]   =  1.0 - xi[gp]  - eta[gp] - zta[gp];
  phi[1]   =  xi[gp];
  phi[2]   =  eta[gp];
  phi[3]   =  zta[gp];
  
  
  // ksi-derivatives
  dphidxi[0] = -1.;
  dphidxi[1] =  1.;
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

  wt = nwt[gp];
  
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
  volp = jac*canonical_vol;

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
    dphidx[i] = dphidxi[i]*dxidx+dphideta[i]*detadx+dphidzta[i]*dztadx;
    dphidy[i] = dphidxi[i]*dxidy+dphideta[i]*detady+dphidzta[i]*dztady;
    dphidz[i] = dphidxi[i]*dxidz+dphideta[i]*detadz+dphidzta[i]*dztadz;
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
};  

  // Calculates the values of u and x at the specified gauss point
  void getBasis(const int gp,     ///< current Gauss point (input)
		const double *x,     ///< x array (input)
		const double *y   ///< y array (input) 
		) {
  std::cout<<"BasisLTet::getBasis(int gp, double *x, double *y) is not implemented"<<std::endl;
  //exit(0);
};

 public:
  // Variables that are calculated at the gauss point
  int sngp;
};


/// Implementation of 1-D linear bar element.
/** 2 node element with number of Gauss points specified in constructor, defaults to 2. */
class BasisLBar : public Basis {

 public:

  /// Constructor
  /** Number of Gauss points = sngp; default 2. */
  BasisLBar(int n = 2) :sngp(n){
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

  xi  = new double[ngp];
  nwt  = new double[ngp];

  for(int i = 0; i < sngp; i++){
    xi[i] = abscissa[i];
    nwt[i] = weight[i];
  }
};

  /// Destructor
  ~BasisLBar() {
  delete [] phi;
  delete [] dphidxi;
  delete [] dphideta;
  delete [] dphidzta;
  delete [] dphidx;
  delete [] dphidy;
  delete [] dphidz;
  delete [] abscissa;
  delete [] weight;

  delete [] xi;
  delete [] nwt;
};

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
		 )
  {
  // Calculate basis function and derivatives at nodal pts
  phi[0]=(1.0-xi[gp])/2.0;
  phi[1]=(1.0+xi[gp])/2.0;

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

  wt = nwt[gp];  

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
};  

  BasisLBar* clone() const{ return new BasisLBar(*this); };

 public:
  // Variables that are calculated at the gauss point
  int sngp;
};



/// Implementation of 1-D quadratic bar element.
/** 3 node element with number of Gauss points specified in constructor, defaults to 3. */
class BasisQBar : public Basis {

 public:

  /// Constructor
  /** Number of Gauss points = sngp; default 3. */
  BasisQBar(int n = 3) :sngp(n){
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

  xi  = new double[ngp];
  nwt  = new double[ngp];

  for(int i = 0; i < sngp; i++){
    xi[i] = abscissa[i];
    nwt[i] = weight[i];
  }
};

  /// Destructor
  ~BasisQBar(){
  delete [] phi;
  delete [] dphidxi;
  delete [] dphideta;
  delete [] dphidzta;
  delete [] dphidx;
  delete [] dphidy;
  delete [] dphidz;
  delete [] abscissa;
  delete [] weight;

  delete [] xi;
  delete [] nwt;
};

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
		 )
  {
  // Calculate basis function and derivatives at nodal pts
  phi[0]=-xi[gp]*(1.0-xi[gp])/2.0;
  phi[1]=xi[gp]*(1.0+xi[gp])/2.0;
  phi[2]=(1.0-xi[gp]*xi[gp]);

  dphidxi[0]=-1.0/2.0 + xi[gp];
  dphidxi[1]= 1.0/2.0 + xi[gp];
  dphidxi[2]=-2.0*xi[gp];
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
  
  wt = nwt[gp];  
  
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
};  

  BasisQBar* clone() const{ return new BasisQBar(*this); };

 public:
  // Variables that are calculated at the gauss point
  int sngp;
};

class OMPBasisLQuad{
public:
  int sngp;
  /// Access number of Gauss points.
  int ngp;


OMPBasisLQuad(){
  sngp =2;
  ngp = sngp*sngp;
  //phi = new double[4];//number of nodes
  //dphidxi = new double[4];
  //dphideta = new double[4];
  //dphidzta = new double[4];
  //dphidx = new double[4];
  //dphidy = new double[4];
  //dphidz = new double[4];
  //abscissa = new double[sngp];//number guass pts
  //weight = new double[sngp];
  //setN(sngp, abscissa, weight);
  abscissa[0] = -1.0L/1.732050807568877;
  abscissa[1] =  1.0L/1.732050807568877;
  weight[0] = 1.0;
  weight[1] = 1.0;

  //xi  = new double[ngp];
  //eta = new double[ngp];
  //nwt  = new double[ngp];


  //cn right now, changing the order when ngp = 4 breaks all the quad tests
  //cn so we leave it for now...
  xi[0]  = abscissa[0];
  eta[0] = abscissa[0];
  nwt[0]  = weight[0] * weight[0];
  
  xi[1]  = abscissa[1];
  eta[1] = abscissa[0];
  nwt[1]  = weight[0] * weight[1];
  
  xi[2]  = abscissa[1];
  eta[2] = abscissa[1];
  nwt[2]  = weight[1] * weight[1];
  
  xi[3]  = abscissa[0];
  eta[3] = abscissa[1];
  nwt[3]  = weight[0] * weight[1];
  // Calculate basis function and derivatives at nodal pts
  for (int i=0; i < 4; i++) {
    phi1[0][i]=(1.0-xi[i])*(1.0-eta[i])/4.0;
    phi1[1][i]=(1.0+xi[i])*(1.0-eta[i])/4.0;
    phi1[2][i]=(1.0+xi[i])*(1.0+eta[i])/4.0;
    phi1[3][i]=(1.0-xi[i])*(1.0+eta[i])/4.0;

    dphidxi1[0][i]=-(1.0-eta[i])/4.0;
    dphidxi1[1][i]= (1.0-eta[i])/4.0;
    dphidxi1[2][i]= (1.0+eta[i])/4.0;
    dphidxi1[3][i]=-(1.0+eta[i])/4.0;

    dphideta1[0][i]=-(1.0-xi[i])/4.0;
    dphideta1[1][i]=-(1.0+xi[i])/4.0;
    dphideta1[2][i]= (1.0+xi[i])/4.0;
    dphideta1[3][i]= (1.0-xi[i])/4.0;
  }
  
}

KOKKOS_INLINE_FUNCTION
void getBasis(const int gp,const  double x[4], const  double y[4],  const double z[4],const  double u[4],const  double uold[4]) const {
  return;
}

void getBasis(const int gp,const  double x[4], const  double y[4],  const double z[4],const  double u[4],const  double uold[4],const  double uoldold[4]) {

  // Caculate basis function and derivative at GP.
  double dxdxi = .25*( (x[1]-x[0])*(1.-eta[gp])+(x[2]-x[3])*(1.+eta[gp]) );
  double dxdeta = .25*( (x[3]-x[0])*(1.- xi[gp])+(x[2]-x[1])*(1.+ xi[gp]) );
  double dydxi  = .25*( (y[1]-y[0])*(1.-eta[gp])+(y[2]-y[3])*(1.+eta[gp]) );
  double dydeta = .25*( (y[3]-y[0])*(1.- xi[gp])+(y[2]-y[1])*(1.+ xi[gp]) );
  double dzdxi   = .25*( (z[1]-z[0])*(1.-eta[gp])+(z[2]-z[3])*(1.+eta[gp]) );
  double dzdeta  = .25*( (z[3]-z[0])*(1.- xi[gp])+(z[2]-z[1])*(1.+ xi[gp]) );

  wt = nwt[gp];

  jac = dxdxi * dydeta - dxdeta * dydxi;

//   jac = sqrt( (dzdxi * dxdeta - dxdxi * dzdeta)*(dzdxi * dxdeta - dxdxi * dzdeta)
// 	     +(dydxi * dzdeta - dzdxi * dydeta)*(dydxi * dzdeta - dzdxi * dydeta)
// 	     +(dxdxi * dydeta - dxdeta * dydxi)*(dxdxi * dydeta - dxdeta * dydxi));

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
    xx += x[i] * phi1[i][gp];
    yy += y[i] * phi1[i][gp];
    zz += z[i] * phi1[i][gp];
    dphidx[i] = dphidxi1[i][gp]*dxidx+dphideta1[i][gp]*detadx;
    dphidy[i] = dphidxi1[i][gp]*dxidy+dphideta1[i][gp]*detady;
    dphidz[i] = 0.0;
    dphidzta[i]= 0.0;
 
    uu += u[i] * phi1[i][gp];
    dudx += u[i] * dphidx[i];
    dudy += u[i]* dphidy[i];
    
    uuold += uold[i] * phi1[i][gp];
    duolddx += uold[i] * dphidx[i];
    duolddy += uold[i]* dphidy[i];
    
    uuoldold += uoldold[i] * phi1[i][gp];
    duoldolddx += uoldold[i] * dphidx[i];
    duoldolddy += uoldold[i]* dphidy[i];
  }
  

  return;
}

  // Variables that are calculated at the gauss point
  /// Access value of basis function at the current Gauss point.
  double phi1[4][4]; 
  /// Access value of dphi / dxi  at the current Gauss point.
  double dphidxi1[4][4];
  /// Access value of dphi / deta  at the current Gauss point.
  double dphideta1[4][4];
  /// Access value of dphi / dzta  at the current Gauss point.
  double dphidzta[4];
  
  /// Access value of dxi / dx  at the current Gauss point.
  double dxidx;
  /// Access value of dxi / dy  at the current Gauss point.
  double dxidy;
  /// Access value of dxi / dz  at the current Gauss point.
  double dxidz;

  /// Access value of deta / dx  at the current Gauss point.
  double detadx;
  /// Access value of deta / dy  at the current Gauss point.
  double detady;
  /// Access value of deta / dz  at the current Gauss point.
  double detadz;

  /// Access value of dzta / dx  at the current Gauss point.
  double dztadx;
  /// Access value of dzta / dy  at the current Gauss point.
  double dztady;
  /// Access value of dzta / dz  at the current Gauss point.
  double dztadz;

  /// Access value of the Gauss weight  at the current Gauss point.
  double wt;
  /// Access value of the mapping Jacobian.
  double jac;

  /// Access value of u at the current Gauss point.
  double uu;
  /// Access value of du / dx at the current Gauss point.
  double dudx;
  /// Access value of du / dy at the current Gauss point.
  double dudy;
  /// Access value of du / dz at the current Gauss point.
  double dudz;

  /// Access value of u_old at the current Gauss point.
  double uuold;
  /// Access value of du_old / dx at the current Gauss point.
  double duolddx;
  /// Access value of du_old / dy at the current Gauss point.
  double duolddy;
  /// Access value of du_old / dz at the current Gauss point.
  double duolddz;

  /// Access value of u_old_old at the current Gauss point.
  double uuoldold;
  /// Access value of du_old_old / dx at the current Gauss point.
  double duoldolddx;
  /// Access value of du_old_old / dy at the current Gauss point.
  double duoldolddy;
  /// Access value of du_old_old / dz at the current Gauss point.
  double duoldolddz;

  /// Access value of x coordinate in real space at the current Gauss point.
  double xx;
  /// Access value of y coordinate in real space at the current Gauss point.
  double yy;
  /// Access value of z coordinate in real space at the current Gauss point.
  double zz;

  /// Access value of the derivative of the basis function wrt to x at the current Gauss point.
  double dphidx[4];
  /// Access value of the derivative of the basis function wrt to y at the current Gauss point.
  double dphidy[4];
  /// Access value of the derivative of the basis function wrt to z at the current Gauss point.
  double dphidz[4];
  //protected:
  /// Access a pointer to the coordinates of the Gauss points in canonical space.
  double abscissa[4];
  /// Access a pointer to the Gauss weights.
  double weight[4];
  /// Access a pointer to the xi coordinate at each Gauss point.
  double xi[4];
  //double *xi;
  /// Access a pointer to the eta coordinate at each Gauss point.
  double eta[4];
  /// Access a pointer to the zta coordinate at each Gauss point.
  double zta[4];
  /// Access the number of Gauss weights.
  double nwt[4];


};


// Current max element: QHex
#define BASIS_NODES_PER_ELEM 27
#define BASIS_SNODES_PER_ELEM 3
// NGP = Total number of Gauss points in element
// SNGP = Total number of Gauss points in tensor product element per dimension
// TNGP = Total number of Gauss points in a triangle/tetrahedron
#define BASIS_NGP_PER_ELEM 64
#define BASIS_SNGP_PER_ELEM 4
#define BASIS_TNGP_PER_ELEM 5

class Unified {
public:
#ifdef KOKKOS_HAVE_CUDA
/** Allocate instances in CPU/GPU unified memory */
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }
  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }

/** Allocate all arrays in CPU/GPU unified memory */
  void* operator new[] (std::size_t size) {
    void *ptr; 
    cudaMallocManaged(&ptr,size);
    cudaDeviceSynchronize();
    return ptr;
  }
  void operator delete[] (void* ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
#endif
};

// Reference element basis classes
// Returns evaluation of reference basis functions, reference coords at quadrature points
class GPURefBasis{

public:

  KOKKOS_INLINE_FUNCTION
  GPURefBasis(){
  };
  KOKKOS_INLINE_FUNCTION
  virtual ~GPURefBasis(){
  };
  
  /// Public access points for internal functions
  KOKKOS_INLINE_FUNCTION
  const int canonical_vol() const {return canonical_volp;};
  KOKKOS_INLINE_FUNCTION
  const int ngp() const {return ngpp;};
  KOKKOS_INLINE_FUNCTION
  const int nbn() const {return nbnp;};
  KOKKOS_INLINE_FUNCTION
  const int pd() const {return pdp;};
  KOKKOS_INLINE_FUNCTION  
  const double nwt(const int i) const {return nwtp[i];};
  KOKKOS_INLINE_FUNCTION
  const double phinew(const int i, const int j) const {return phinewp[i][j];};
  KOKKOS_INLINE_FUNCTION
  const double dphidxinew(const int i, const int j) const {return dphidxinewp[i][j];};
  KOKKOS_INLINE_FUNCTION
  const double dphidetanew(const int i, const int j) const {return dphidetanewp[i][j];};
  KOKKOS_INLINE_FUNCTION
  const double dphidztanew(const int i, const int j) const {return dphidztanewp[i][j];};
  KOKKOS_INLINE_FUNCTION
  const double xi(const int i) const {return xip[i];};
  KOKKOS_INLINE_FUNCTION
  const double eta(const int i) const {return etap[i];};
  KOKKOS_INLINE_FUNCTION
  const double zta(const int i) const {return ztap[i];};

protected:
  double canonical_volp;

  /// Access number of basis nodes and polynomial degree
  int nbnp;
  int pdp;

  /// Access the number of Gauss points and the corresponding weights.
  int ngpp;
  double nwtp[BASIS_NGP_PER_ELEM];
  
  /// Access a pointer to the xi, eta, zta coordinates at each Gauss point.
  double xip[BASIS_NGP_PER_ELEM];
  double etap[BASIS_NGP_PER_ELEM];
  double ztap[BASIS_NGP_PER_ELEM];
  
  /// Assess phi at Gauss point for each phi; including its derivatives
  double phinewp[BASIS_NGP_PER_ELEM][BASIS_NODES_PER_ELEM];
  double dphidxinewp[BASIS_NGP_PER_ELEM][BASIS_NODES_PER_ELEM];
  double dphidetanewp[BASIS_NGP_PER_ELEM][BASIS_NODES_PER_ELEM];
  double dphidztanewp[BASIS_NGP_PER_ELEM][BASIS_NODES_PER_ELEM];
};

class GPURefTensorProductBasis:public GPURefBasis{
public:
  KOKKOS_INLINE_FUNCTION
  GPURefTensorProductBasis(){
  };
  KOKKOS_INLINE_FUNCTION
  virtual ~GPURefTensorProductBasis(){
  };

protected:
  /// Access a pointer to the coordinates of the Gauss points per dimension
  double abscissa[BASIS_SNGP_PER_ELEM];
  /// Access a pointer to the Gauss weights.
  double weight[BASIS_SNGP_PER_ELEM];
  /// Access number of Gauss points per dimension
  //note that quadrature is exact for polynomials of degree 2*sngp - 1
  int sngp;
 
  /// 1D basis functions and their derivatives (currently up to quadratic)
  double xibasisatqp[BASIS_SNODES_PER_ELEM];
  double etabasisatqp[BASIS_SNODES_PER_ELEM];
  double ztabasisatqp[BASIS_SNODES_PER_ELEM];
  double dxibasisatqp[BASIS_SNODES_PER_ELEM];
  double detabasisatqp[BASIS_SNODES_PER_ELEM];
  double dztabasisatqp[BASIS_SNODES_PER_ELEM];

  // Functions to evaluate 1D coordinates at Gauss points
  KOKKOS_INLINE_FUNCTION
  void xibasisFunctions1D(const int gp){
    if (pd() == 1){
      xibasisatqp[0] = (1 - xip[gp])/2.;
      xibasisatqp[1] = (1 + xip[gp])/2.;
      dxibasisatqp[0] = -1/2.;
      dxibasisatqp[1] = 1/2.;
    }else if (pd() == 2){
      xibasisatqp[0] = -xip[gp]*(1 - xip[gp])/2.;
      xibasisatqp[1] = xip[gp]*(1 + xip[gp])/2.;
      xibasisatqp[2] = 1 - xip[gp]*xip[gp];
      dxibasisatqp[0] = xip[gp] - 1/2.;
      dxibasisatqp[1] = xip[gp] + 1/2.;
      dxibasisatqp[2] = -2*xip[gp];
    }
  }

  KOKKOS_INLINE_FUNCTION
  void etabasisFunctions1D(const int gp){
    if (pd() == 1){
      etabasisatqp[0] = (1 - etap[gp])/2.;
      etabasisatqp[1] = (1 + etap[gp])/2.;
      detabasisatqp[0] = -1/2.;
      detabasisatqp[1] = 1/2.;
    }else if (pd() == 2){
      etabasisatqp[0] = -etap[gp]*(1 - etap[gp])/2.;
      etabasisatqp[1] = etap[gp]*(1 + etap[gp])/2.;
      etabasisatqp[2] = 1 - etap[gp]*etap[gp];
      detabasisatqp[0] = etap[gp] - 1/2.;
      detabasisatqp[1] = etap[gp] + 1/2.;
      detabasisatqp[2] = -2*etap[gp];
    }
  }

  KOKKOS_INLINE_FUNCTION
  void ztabasisFunctions1D(const int gp){
    if (pd() == 1){
      ztabasisatqp[0] = (1 - ztap[gp])/2.;
      ztabasisatqp[1] = (1 + ztap[gp])/2.;
      dztabasisatqp[0] = -1/2.;
      dztabasisatqp[1] = 1/2.;
    }else if (pd() == 2){
      ztabasisatqp[0] = -ztap[gp]*(1 - ztap[gp])/2.;
      ztabasisatqp[1] = ztap[gp]*(1 + ztap[gp])/2.;
      ztabasisatqp[2] = 1 - ztap[gp]*ztap[gp];
      dztabasisatqp[0] = ztap[gp] - 1/2.;
      dztabasisatqp[1] = ztap[gp] + 1/2.;
      dztabasisatqp[2] = -2*ztap[gp];
    }
  }

  /// 1D quadrature rule
  KOKKOS_INLINE_FUNCTION
  int computeQuadratureData(const int n){
    sngp = n;
    if( 3 == n){
      abscissa[0] = -3.872983346207417/5.0;
      abscissa[1] =  0.0;
      abscissa[2] =  3.872983346207417/5.0;
      weight[0] = 5.0/9.0;
      weight[1] = 8.0/9.0;
      weight[2] = 5.0/9.0;
    } else if ( 4 == n ) {
      abscissa[0] =  -30.13977090579184/35.0;
      abscissa[1] =  -11.89933652546997/35.0;
      abscissa[2] =   11.89933652546997/35.0;
      abscissa[3] =   30.13977090579184/35.0;
      weight[0] = (18.0-5.477225575051661)/36.0;
      weight[1] = (18.0+5.477225575051661)/36.0;
      weight[2] = (18.0+5.477225575051661)/36.0;
      weight[3] = (18.0-5.477225575051661)/36.0;
    } else {
      sngp = 2;
      if ( 2 != n ) {
        std::cout<<"WARNING: only 1 < N < 5 gauss points supported at this time, defaulting to N = 2"<<std::endl;
      }
      abscissa[0] = -1.0/1.732050807568877;
      abscissa[1] =  1.0/1.732050807568877;
      weight[0] = 1.0;
      weight[1] = 1.0;
    }
    return sngp;
  }
};

class GPURefBasisBar:public GPURefTensorProductBasis{
public:
  KOKKOS_INLINE_FUNCTION
  GPURefBasisBar(){
  };
  KOKKOS_INLINE_FUNCTION
  virtual ~GPURefBasisBar(){
  };

protected:
  // Set up reference element coordinate values at quadrature points
  KOKKOS_INLINE_FUNCTION
  void computeCoordinateData(const int sngp){
    for(int i = 0; i < sngp; i++){
      xip[i] = abscissa[i];
      nwtp[i] = weight[i];
    }
  }

  KOKKOS_INLINE_FUNCTION
  void computeBasisFunctions(const int ngpp, const int nbnp){
    for(int gp = 0; gp < ngpp; gp++){
      // Compute basis function evaluations at coordinate
      xibasisFunctions1D(gp);
      for(int bn = 0; bn < nbnp; bn++){
          phinewp[gp][bn] = xibasisatqp[bn];
          dphidxinewp[gp][bn] = dxibasisatqp[bn];
          dphidetanewp[gp][bn] = 0;
          dphidztanewp[gp][bn] = 0;
      }
    }
  }
};

class GPURefBasisLBar:public GPURefBasisBar{
public:

  KOKKOS_INLINE_FUNCTION
  GPURefBasisLBar(const int n = 2){
    pdp = 1;
    nbnp = pdp + 1;
    canonical_volp = 2.;
    sngp = computeQuadratureData(n);
    computeCoordinateData(sngp);

    // Set up reference element basis function values at quadrature points
    ngpp = sngp;
    computeBasisFunctions(ngpp, nbnp);
  }
  
  KOKKOS_INLINE_FUNCTION
  ~GPURefBasisLBar(){}
};

class GPURefBasisQBar:public GPURefBasisBar{
public:

  KOKKOS_INLINE_FUNCTION
  GPURefBasisQBar(const int n = 3){
    pdp = 2;
    nbnp = pdp + 1;
    canonical_volp = 2.;
    sngp = computeQuadratureData(n);
    computeCoordinateData(sngp);

    // Set up reference element basis function values at quadrature points
    ngpp = sngp;
    computeBasisFunctions(ngpp, nbnp);
  }
  
  KOKKOS_INLINE_FUNCTION
  ~GPURefBasisQBar(){}
};

class GPURefBasisQuad:public GPURefTensorProductBasis{
public:
  KOKKOS_INLINE_FUNCTION
  GPURefBasisQuad(){
  };
  KOKKOS_INLINE_FUNCTION
  virtual ~GPURefBasisQuad(){
  };

protected:
  // Set up reference element coordinate values at quadrature points
  KOKKOS_INLINE_FUNCTION
  void computeCoordinateData(const int sngp){
    int c = 0;
    for( int i = 0; i < sngp; i++ ){
      for( int j = 0; j < sngp; j++ ){
	      //std::cout<<i+j+c<<"   "<<i<<"   "<<j<<std::endl;
	      xip[i+j+c]  = abscissa[i];
	      etap[i+j+c] = abscissa[j];
	      nwtp[i+j+c]  = weight[i] * weight[j];
      }
      c = c + sngp - 1;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void computeBasisFunctions(const int ngpp, const int nbnp, const int nl[2][9]){
    for(int gp = 0; gp < ngpp; gp++){
      // Compute basis function evaluations at gauss point coordinate
      xibasisFunctions1D(gp);
      etabasisFunctions1D(gp);
      for(int bn = 0; bn < nbnp; bn++){
          phinewp[gp][bn] = xibasisatqp[nl[0][bn]]*etabasisatqp[nl[1][bn]];
          dphidxinewp[gp][bn] = dxibasisatqp[nl[0][bn]]*etabasisatqp[nl[1][bn]];
          dphidetanewp[gp][bn] = xibasisatqp[nl[0][bn]]*detabasisatqp[nl[1][bn]];
          dphidztanewp[gp][bn] = 0;
      }
    }
  }
};

class GPURefBasisLQuad:public GPURefBasisQuad{
public:
  KOKKOS_INLINE_FUNCTION
  GPURefBasisLQuad(const int n = 2){
    pdp = 1;
    nbnp = (pdp+1)*(pdp+1);
    canonical_volp = 4.;
    sngp = computeQuadratureData(n);
    computeCoordinateData(sngp);

    // Set up reference element basis function values at quadrature points
    ngpp = sngp*sngp;
    // Follow exodus node numbering
    // (first row for xi, second for eta basis functions)
    int nl[2][9] = {{0, 1, 1, 0}, {0, 0, 1, 1}};
    computeBasisFunctions(ngpp, nbnp, nl);
  }
  
  KOKKOS_INLINE_FUNCTION
  ~GPURefBasisLQuad(){}
};

class GPURefBasisQQuad:public GPURefBasisQuad{
public:

  KOKKOS_INLINE_FUNCTION
  GPURefBasisQQuad(const int n = 3){
    pdp = 2;
    nbnp = (pdp+1)*(pdp+1);
    canonical_volp = 4.;
    sngp = computeQuadratureData(n);
    computeCoordinateData(sngp);

    // Set up reference element basis function values at quadrature points
    ngpp = sngp*sngp;
    int nl[2][9] = {{0, 1, 1, 0, 2, 1, 2, 0, 2},
                    {0, 0, 1, 1, 0, 2, 1, 2, 2}};
    computeBasisFunctions(ngpp, nbnp, nl);
  }
  
  KOKKOS_INLINE_FUNCTION
  ~GPURefBasisQQuad(){}
};

class GPURefBasisHex:public GPURefTensorProductBasis{
public:
  KOKKOS_INLINE_FUNCTION
  GPURefBasisHex(){
  };
  KOKKOS_INLINE_FUNCTION
  virtual ~GPURefBasisHex(){
  };

protected:
  // Set up reference element coordinate values at quadrature points
  KOKKOS_INLINE_FUNCTION
  void computeCoordinateData(const int sngp){
    int c = 0;
    for( int i = 0; i < sngp; i++ ){
      for( int j = 0; j < sngp; j++ ){
	      for( int k = 0; k < sngp; k++ ){
	        //std::cout<<i+j+k+c<<"   "<<i<<"   "<<j<<"   "<<k<<std::endl;
	        xip[i+j+k+c]  = abscissa[i];
	        etap[i+j+k+c] = abscissa[j];
	        ztap[i+j+k+c] = abscissa[k];
	        nwtp[i+j+k+c]  = weight[i] * weight[j] * weight[k]; 
	      }   
	      c = c + sngp - 1;
      }
      c = c + sngp - 1;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void computeBasisFunctions(const int ngpp, const int nbnp, const int nl[3][27]){
    for(int gp = 0; gp < ngpp; gp++){
      // Compute basis function evaluations at coordinate
      xibasisFunctions1D(gp);
      etabasisFunctions1D(gp);
      ztabasisFunctions1D(gp);
      for(int bn = 0; bn < nbnp; bn++){
        phinewp[gp][bn] = xibasisatqp[nl[0][bn]]*etabasisatqp[nl[1][bn]]*ztabasisatqp[nl[2][bn]];
        dphidxinewp[gp][bn] = dxibasisatqp[nl[0][bn]]*etabasisatqp[nl[1][bn]]*ztabasisatqp[nl[2][bn]];
        dphidetanewp[gp][bn] = xibasisatqp[nl[0][bn]]*detabasisatqp[nl[1][bn]]*ztabasisatqp[nl[2][bn]];
        dphidztanewp[gp][bn] = xibasisatqp[nl[0][bn]]*etabasisatqp[nl[1][bn]]*dztabasisatqp[nl[2][bn]];
      }
    }
  }

};

class GPURefBasisLHex:public GPURefBasisHex{
public:
  KOKKOS_INLINE_FUNCTION
  GPURefBasisLHex(const int n = 2){
    pdp = 1;
    nbnp = (pdp+1)*(pdp+1)*(pdp+1);
    canonical_volp = 8.;
    sngp = computeQuadratureData(n);
    computeCoordinateData(sngp);
  
   // Set up reference element basis function values at quadrature points
    ngpp = sngp*sngp*sngp;
    int nl[3][27] = {{0, 1, 1, 0, 0, 1, 1, 0},
                    {0, 0, 1, 1, 0, 0, 1, 1},
                    {0, 0, 0, 0, 1, 1, 1, 1}};
    computeBasisFunctions(ngpp, nbnp, nl);
  }
  
  KOKKOS_INLINE_FUNCTION
  ~GPURefBasisLHex(){}
};

class GPURefBasisQHex:public GPURefBasisHex{
public:
  KOKKOS_INLINE_FUNCTION
  GPURefBasisQHex(const int n = 3){
    pdp = 2;
    nbnp = (pdp+1)*(pdp+1)*(pdp+1);
    canonical_volp = 8.;
    sngp = computeQuadratureData(n);
    computeCoordinateData(sngp);
  
    // Set up reference element basis function values at quadrature points
    ngpp = sngp*sngp*sngp;
    // Check https://sandialabs.github.io/seacas-docs/exodusII-new.pdf, Figure 4.14
    int nl[3][27] = {{0, 1, 1, 0, 0, 1, 1, 0, 2, 1, 2, 0, 0, 1, 1, 0, 2, 1, 2, 0, 2, 2, 2, 0, 1, 2, 2},
                     {0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 1, 2, 0, 0, 1, 1, 0, 2, 1, 2, 2, 2, 2, 2, 2, 0, 1},
                     {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 2, 0, 1, 2, 2, 2, 2}};
    computeBasisFunctions(ngpp, nbnp, nl);
  }
  
  KOKKOS_INLINE_FUNCTION
  ~GPURefBasisQHex(){}
};

class GPURefBasisTri:public GPURefBasis{
  public:
  KOKKOS_INLINE_FUNCTION
  GPURefBasisTri(){
  };
  KOKKOS_INLINE_FUNCTION
  virtual ~GPURefBasisTri(){
  };

protected:
  /// Access a pointer to the coordinates of the Gauss points per dimension
  double abscissa[BASIS_TNGP_PER_ELEM];
  /// Access a pointer to the Gauss weights.
  double weight[BASIS_TNGP_PER_ELEM];

  int ngp; 

  /// Triangle quadrature rule
  /// abscissae contain coordinates for both xi and eta, need to combine appropriately in computeCoordinateData
  KOKKOS_INLINE_FUNCTION
  int computeQuadratureData(const int n){
    ngp = n;
    if( 4 == n){
      abscissa[0] = 1/3.;
      abscissa[1] = 1/5.;
      abscissa[2] = 3/5.;
      weight[0] = -9/32.;
      weight[1] = 25/96.;
    } else if ( 3 == n ) {
      abscissa[0] = 1/6.;
      abscissa[1] = 2/3.;
      weight[0] = 1/6.;
      // GAW This is the usual Gaussian quadrature, and was also used for QTri in old code. However, for LTri, old code used
      // abscissa[0] = 1/2.
      // abscissa[1] = 0.
    } else {
      ngp = 1;
      if ( 1 != n ) {
        std::cout<<"WARNING: only 1, 3 or 4 Gauss points supported at this time, defaulting to 1."<<std::endl;
      }
      abscissa[0] = 1/3.;
      weight[0] = 1/2.;
    }
    return ngp;
  }

  // Set up reference element coordinate values at quadrature points
  KOKKOS_INLINE_FUNCTION
  void computeCoordinateData(const int ngp){
    if( ngp == 1 ){
      xip[0] = abscissa[0];
      etap[0] = abscissa[0];
      nwtp[0] = weight[0];
    }else if (ngp == 3 ) {
      xip[0]  = abscissa[0];
      etap[0] = abscissa[0];
      nwtp[0]  = weight[0];
      xip[1]  = abscissa[1];
      etap[1] = abscissa[0];
      nwtp[1]  = weight[0];
      xip[2]  = abscissa[0];
      etap[2] = abscissa[1];
      nwtp[2]  = weight[0];
    }else if (ngp == 4) {
      xip[0]  = abscissa[0];
      etap[0] = abscissa[0];
      nwtp[0]  = weight[0];
      xip[1]  = abscissa[2];
      etap[1] = abscissa[1];
      nwtp[1]  = weight[1];
      xip[2]  = abscissa[1];
      etap[2] = abscissa[2];
      nwtp[2]  = weight[1];
      xip[3]  = abscissa[1];
      etap[3] = abscissa[1];
      nwtp[3]  = weight[1];
    }
  }

  KOKKOS_INLINE_FUNCTION
  virtual void computeBasisFunctions(){}
};

class GPURefBasisLTri:public GPURefBasisTri{
public:
  KOKKOS_INLINE_FUNCTION
  GPURefBasisLTri(const int n = 1){
    pdp = 1;
    nbnp = 3;
    canonical_volp = 0.5;
    ngpp = computeQuadratureData(n);
    computeCoordinateData(ngpp);

    // Set up reference element basis function values at quadrature points
    // Follow exodus node numbering
    computeBasisFunctions(ngpp);
  }

  KOKKOS_INLINE_FUNCTION
  ~GPURefBasisLTri(){}
private:
  KOKKOS_INLINE_FUNCTION
  virtual void computeBasisFunctions(const int ngpp){
    for(int gp = 0; gp < ngpp; gp++){
      phinewp[gp][0] = (1. - xip[gp] - etap[gp]);
      phinewp[gp][1] = xip[gp];
      phinewp[gp][2] = etap[gp];
      dphidxinewp[gp][0] = -1.;
      dphidxinewp[gp][1] = 1.;
      dphidxinewp[gp][2] = 0.;
      dphidetanewp[gp][0] = -1.;
      dphidetanewp[gp][1] = 0.;
      dphidetanewp[gp][2] = 1.;
      dphidztanewp[gp][0] = 0.;
      dphidztanewp[gp][1] = 0.;
      dphidztanewp[gp][2] = 0.;
    }
  }
};

class GPURefBasisQTri:public GPURefBasisTri{
public:
  KOKKOS_INLINE_FUNCTION
  GPURefBasisQTri(const int n = 3){
    pdp = 2;
    nbnp = 6;
    canonical_volp = 0.5;
    ngpp = computeQuadratureData(n);
    computeCoordinateData(ngpp);

    // Set up reference element basis function values at quadrature points
    // Follow exodus node numbering
    computeBasisFunctions(ngpp);
  }

  KOKKOS_INLINE_FUNCTION
  ~GPURefBasisQTri(){}
private:
  KOKKOS_INLINE_FUNCTION
  virtual void computeBasisFunctions(const int ngpp){
    for(int gp = 0; gp < ngpp; gp++){
      phinewp[gp][0]= 2. * (1. - xip[gp] - etap[gp]) * (0.5 - xip[gp] - etap[gp]);
      phinewp[gp][1]= 2. * xip[gp] * (xip[gp] - 0.5);
      phinewp[gp][2]= 2. * etap[gp] * (etap[gp] - 0.5);
      phinewp[gp][3]= 4. * (1. - xip[gp] - etap[gp]) * xip[gp];
      phinewp[gp][4]= 4. * xip[gp] * etap[gp];
      phinewp[gp][5]= 4. * (1. - xip[gp] - etap[gp]) * etap[gp];
      dphidxinewp[gp][0]=-2. * (0.5 - xip[gp] - etap[gp]) - 2. * (1. - xip[gp] - etap[gp]);
      dphidxinewp[gp][1]= 2. * (xip[gp] - 0.5) + 2.0 * xip[gp];
      dphidxinewp[gp][2]= 0.;
      dphidxinewp[gp][3]=-4. * xip[gp] + 4. * (1. - xip[gp] - etap[gp]);
      dphidxinewp[gp][4]= 4. * etap[gp];
      dphidxinewp[gp][5]= -4. * etap[gp];
      dphidetanewp[gp][0]=-2. * (0.5 - xip[gp] - etap[gp]) - 2. * (1. - xip[gp] - etap[gp]);
      dphidetanewp[gp][1]= 0.;
      dphidetanewp[gp][2]= 2. * etap[gp] + 2. * (etap[gp] - 0.5);
      dphidetanewp[gp][3]=-4. * xip[gp];
      dphidetanewp[gp][4]= 4. * xip[gp];
      dphidetanewp[gp][5]= 4. * (1. - xip[gp] - etap[gp]) - 4. * etap[gp];
      dphidztanewp[gp][0]= 0.;
      dphidztanewp[gp][1]= 0.;
      dphidztanewp[gp][2]= 0.;
      dphidztanewp[gp][3]= 0.;
      dphidztanewp[gp][4]= 0.;
      dphidztanewp[gp][5]= 0.;
    }
  }
};

class GPURefBasisTet:public GPURefBasis{
public:
  KOKKOS_INLINE_FUNCTION
  GPURefBasisTet(){
  };
  KOKKOS_INLINE_FUNCTION
  virtual ~GPURefBasisTet(){
  };

protected:
  /// Access a pointer to the coordinates of the Gauss points per dimension
  double abscissa[BASIS_TNGP_PER_ELEM];
  /// Access a pointer to the Gauss weights.
  double weight[BASIS_TNGP_PER_ELEM];

  int ngp; 

  /// Tetrahedron quadrature rule
  /// abscissae contain coordinates for both xi and eta, need to combine appropriately in computeCoordinateData
  KOKKOS_INLINE_FUNCTION
  int computeQuadratureData(const int n){
    ngp = n;
    if ( 5 == n) {
      abscissa[0] = 0.25;
      abscissa[1] = 0.5;
      abscissa[2] = 0.1666666666666667;
      weight[0] = -0.1333333333333333;
      weight[1] = 0.075;
    } else {
      ngp = 4;
      if ( 4 != n ) {
        std::cout<<"WARNING: only 4 or 5 Gauss points supported at this time for tetrahedra, defaulting to 4."<<std::endl;
      }
      abscissa[0] = 0.13819660;
      abscissa[1] = 0.58541020;
      weight[0] = 0.041666666667;
    }
    return ngp;
  }

  // Set up reference element coordinate values at quadrature points
  KOKKOS_INLINE_FUNCTION
  void computeCoordinateData(const int ngp){
    if ( 4 == ngp ) {
      xip[0] = abscissa[0];  // 0, 0, 0
      etap[0]  = abscissa[0];
      ztap[0]  = abscissa[0];
      nwtp[0]  = weight[0];
      xip[1] = abscissa[1]; // 1, 0, 0
      etap[1] = abscissa[0];
      ztap[1] = abscissa[0];
      nwtp[1]  = weight[0];
      xip[2]  = abscissa[0]; // 1, 1, 0
      etap[2]  = abscissa[1];
      ztap[2]  = abscissa[0];
      nwtp[2]  = weight[0];
      xip[3]= abscissa[0];  //0, 1, 0
      etap[3] = abscissa[0];
      ztap[3]= abscissa[1];
      nwtp[3]  = weight[0];
    } else if ( 5 == ngp ) {
      xip[0] = abscissa[0];
      etap[0] = abscissa[0];
      ztap[0] = abscissa[0];
      nwtp[0] = weight[0];
      xip[1] = abscissa[1];
      etap[1] = abscissa[2];
      ztap[1] = abscissa[2];
      nwtp[1] = weight[1];
      xip[2] = abscissa[2];
      etap[2] = abscissa[2];
      ztap[2] = abscissa[2];
      nwtp[2] = weight[1];
      xip[3] = abscissa[2];
      etap[3] = abscissa[2];
      ztap[3] = abscissa[1];
      nwtp[3] = weight[1];
      xip[4] = abscissa[2];
      etap[4] = abscissa[1];
      ztap[4] = abscissa[2];
      nwtp[4] = weight[1];
    }
  }

  KOKKOS_INLINE_FUNCTION
  virtual void computeBasisFunctions(){}
};

class GPURefBasisLTet:public GPURefBasisTet{
public:
  KOKKOS_INLINE_FUNCTION
  GPURefBasisLTet(const int n = 4){
    pdp = 1;
    nbnp = 4;
    canonical_volp = 1/6.;
    ngpp = computeQuadratureData(n);
    computeCoordinateData(ngpp);

    // Set up reference element basis function values at quadrature points
    // Follow exodus node numbering
    computeBasisFunctions(ngpp);
  }

  KOKKOS_INLINE_FUNCTION
  ~GPURefBasisLTet(){}
private:
  KOKKOS_INLINE_FUNCTION
  virtual void computeBasisFunctions(const int ngpp){
    for(int gp = 0; gp < ngpp; gp++){
      phinewp[gp][0] = 1. - xip[gp] - etap[gp] - ztap[gp];
      phinewp[gp][1] = xip[gp];
      phinewp[gp][2] = etap[gp];
      phinewp[gp][3] = ztap[gp];
      dphidxinewp[gp][0] = -1.;
      dphidxinewp[gp][1] =  1.;
      dphidxinewp[gp][2] =  0.;
      dphidxinewp[gp][3] =  0.;
      dphidetanewp[gp][0] = -1.;
      dphidetanewp[gp][1] =  0.;
      dphidetanewp[gp][2] =  1.;
      dphidetanewp[gp][3] =  0.;
      dphidztanewp[gp][0] = -1.;
      dphidztanewp[gp][1] =  0.;
      dphidztanewp[gp][2] =  0.;
      dphidztanewp[gp][3] =  1.;
    }
  }
};

class GPURefBasisQTet:public GPURefBasisTet{
public:
  KOKKOS_INLINE_FUNCTION
  GPURefBasisQTet(const int n = 5){
    pdp = 1;
    nbnp = 10;
    canonical_volp = 1/6.;
    ngpp = computeQuadratureData(n);
    computeCoordinateData(ngpp);

    // Set up reference element basis function values at quadrature points
    // Follow exodus node numbering
    computeBasisFunctions(ngpp);
  }

  KOKKOS_INLINE_FUNCTION
  ~GPURefBasisQTet(){}
private:

  // Exodus corner ordering:
  // 1 at origin
  // 2 at xi = 1
  // 3 at eta = 1
  // 4 at zta = 1
  KOKKOS_INLINE_FUNCTION
  virtual void computeBasisFunctions(const int ngpp){
    for(int gp = 0; gp < ngpp; gp++){
      phinewp[gp][0] = (1 - xip[gp] - etap[gp] - ztap[gp])*(1 - 2*(xip[gp] + etap[gp] + ztap[gp]));
      phinewp[gp][1] = xip[gp]*(2*xip[gp] - 1);
      phinewp[gp][2] = etap[gp]*(2*etap[gp] - 1);
      phinewp[gp][3] = ztap[gp]*(2*ztap[gp] - 1);
      phinewp[gp][4] = 4*(1 - xip[gp] - etap[gp] - ztap[gp])*xip[gp];
      phinewp[gp][5] = 4*xip[gp]*etap[gp];
      phinewp[gp][6] = 4*etap[gp]*(1 - xip[gp] - etap[gp] - ztap[gp]);
      phinewp[gp][7] = 4*(1 - xip[gp] - etap[gp] - ztap[gp])*ztap[gp];
      phinewp[gp][8] = 4*xip[gp]*ztap[gp];
      phinewp[gp][9] = 4*etap[gp]*ztap[gp];

      dphidxinewp[gp][0] = 4*(xip[gp] + etap[gp] + ztap[gp]) - 3;
      dphidxinewp[gp][1] = 4*xip[gp] - 1;
      dphidxinewp[gp][2] = 0;
      dphidxinewp[gp][3] = 0;
      dphidxinewp[gp][4] = 4*(1 - 2*xip[gp] - etap[gp] - ztap[gp]);
      dphidxinewp[gp][5] = 4*etap[gp];
      dphidxinewp[gp][6] = -4*etap[gp];
      dphidxinewp[gp][7] = -4*ztap[gp];
      dphidxinewp[gp][8] = 4*ztap[gp];
      dphidxinewp[gp][9] = 0;

      dphidetanewp[gp][0] = 4*(xip[gp] + etap[gp] + ztap[gp]) - 3;
      dphidetanewp[gp][1] = 0;
      dphidetanewp[gp][2] = 4*etap[gp] - 1;
      dphidetanewp[gp][3] = 0;
      dphidetanewp[gp][4] = -4*xip[gp];
      dphidetanewp[gp][5] = 4*xip[gp];
      dphidetanewp[gp][6] = 4*(1 - xip[gp] - 2*etap[gp] - ztap[gp]);
      dphidetanewp[gp][7] = -4*ztap[gp];
      dphidetanewp[gp][8] = 0;
      dphidetanewp[gp][9] = 4*ztap[gp];

      dphidztanewp[gp][0] = 4*(xip[gp] + etap[gp] + ztap[gp]) - 3;
      dphidztanewp[gp][1] = 0;
      dphidztanewp[gp][2] = 0;
      dphidztanewp[gp][3] = 4*ztap[gp] - 1;
      dphidztanewp[gp][4] = -4*xip[gp];
      dphidztanewp[gp][5] = 0;
      dphidztanewp[gp][6] = -4*etap[gp];
      dphidztanewp[gp][7] = 4*(1 - xip[gp] - etap[gp] - 2*ztap[gp]);
      dphidztanewp[gp][8] = 4*xip[gp];
      dphidztanewp[gp][9] = 4*etap[gp];
    }
  }
};

// Mesh element basis classes
// Returns (weighted) Jacobian, basis functions and fields evaluated at quadrature points
class GPUBasis{

public:

  KOKKOS_INLINE_FUNCTION
  GPUBasis(){
  };
  KOKKOS_INLINE_FUNCTION
  virtual ~GPUBasis(){
  };

  // Function to compute data related to element's vertices' coordinates
  KOKKOS_INLINE_FUNCTION
  virtual void computeElemData( const double x[BASIS_NODES_PER_ELEM], 
						   const double y[BASIS_NODES_PER_ELEM],  
						   const double z[BASIS_NODES_PER_ELEM]) {};

  // Function to build cell coordinates and basis functions at Gauss point (given coord dof values)
  // Also returns weight times Jacobian
  KOKKOS_INLINE_FUNCTION
  virtual double getCoordsBasisWJac(const int gp,
						                        const double x[BASIS_NODES_PER_ELEM], 
						                        const double y[BASIS_NODES_PER_ELEM],  
						                        const double z[BASIS_NODES_PER_ELEM]) {};

  // Function to build u, uold, uoldold at Gauss point (given u dof values)
  // Takes phi basis data as input, since getCoordsBasisJacobian may not be
  // called by each basis object
  KOKKOS_INLINE_FUNCTION
  void getField(const int gp,
                const GPUBasis * basis,
                const double u[BASIS_NODES_PER_ELEM],
                const double uold[BASIS_NODES_PER_ELEM],
                const double uoldold[BASIS_NODES_PER_ELEM]) {
#if 0  
  uup=0.0;
  uuoldp=0.0;
  uuoldoldp=0.0;
  duudxp=0.0;
  duudyp=0.0;
  duudzp=0.0;
  duuolddxp = 0.;
  duuolddyp = 0.;
  duuolddzp = 0.;
  duuoldolddxp = 0.;
  duuoldolddyp = 0.;
  duuoldolddzp = 0.;
  // `basisp` is reference basis, `basis` is GPUBasis instance for first equation,
  // containing (non-reference) element basis function values at Gauss point
  // dphidz is not computed for 2D elements at the moment! Instead, it is set 0.
  for (int i=0; i < basisp->nbn(); i++) {
    if( u ){
      uup += u[i] * basis->phi(i);
      duudxp += u[i] * basis->dphidx(i);
      duudyp += u[i] * basis->dphidy(i);
      duudzp += u[i] * basis->dphidz(i);
    }
    if( uold ){
      uuoldp += uold[i] * basis->phi(i);
      duuolddxp += uold[i] * basis->dphidx(i);
      duuolddyp += uold[i] * basis->dphidy(i);
      duuolddzp += uold[i] * basis->dphidz(i);
    }
    if( uoldold ){
      uuoldoldp += uoldold[i] * basis->phi(i);
      duuoldolddxp += uoldold[i] * basis->dphidx(i);
      duuoldolddyp += uoldold[i] * basis->dphidy(i);
      duuoldolddzp += uoldold[i] * basis->dphidz(i);
    }
    }
#else
  // Initialize all accumulators
  uup = uuoldp = uuoldoldp = 0.0;
  duudxp = duudyp = duudzp = 0.0;
  duuolddxp = duuolddyp = duuolddzp = 0.0;
  duuoldolddxp = duuoldolddyp = duuoldolddzp = 0.0;

  const int nbn = basisp->nbn();

  const bool has_u = (u != nullptr);
  const bool has_uold = (uold != nullptr);
  const bool has_uoldold = (uoldold != nullptr);

  for (int i = 0; i < nbn; ++i) {
    const double phi     = basis->phi(i);
    const double dphidx  = basis->dphidx(i);
    const double dphidy  = basis->dphidy(i);
    const double dphidz  = basis->dphidz(i);

    if (has_u) {
      uup     = FMA(u[i], phi, uup);
      duudxp  = FMA(u[i], dphidx, duudxp);
      duudyp  = FMA(u[i], dphidy, duudyp);
      duudzp  = FMA(u[i], dphidz, duudzp);
    }

    if (has_uold) {
      uuoldp      = FMA(uold[i], phi, uuoldp);
      duuolddxp   = FMA(uold[i], dphidx, duuolddxp);
      duuolddyp   = FMA(uold[i], dphidy, duuolddyp);
      duuolddzp   = FMA(uold[i], dphidz, duuolddzp);
    }

    if (has_uoldold) {
      uuoldoldp       = FMA(uoldold[i], phi, uuoldoldp);
      duuoldolddxp    = FMA(uoldold[i], dphidx, duuoldolddxp);
      duuoldolddyp    = FMA(uoldold[i], dphidy, duuoldolddyp);
      duuoldolddzp    = FMA(uoldold[i], dphidz, duuoldolddzp);
    }
  }
#endif
  }

  /// Access volume of current element.
  KOKKOS_INLINE_FUNCTION const double vol() const {return volp;};
  /// Acces number of Gauss points
  KOKKOS_INLINE_FUNCTION const double ngp() const {return ngpp;};

  /// Access value of x coordinate in real space at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double xx() const {return xxp;};
  /// Access value of y coordinate in real space at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double yy() const {return yyp;};
  /// Access value of z coordinate in real space at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double zz() const {return zzp;};

  /// Access value of the basis function at the current Gauss point
  KOKKOS_INLINE_FUNCTION
  const double phi(const int i) const {return phip[i];};
  /// Access value of the derivative of the basis function wrt to x at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double dphidx(const int i) const {return dphidxp[i];};
  /// Access value of the derivative of the basis function wrt to y at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double dphidy(const int i) const {return dphidyp[i];};
  /// Access value of the derivative of the basis function wrt to z at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double dphidz(const int i) const {return dphidzp[i];};

  /// Access value of u at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double uu() const {return uup;};
  /// Access value of du / dx at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double duudx() const {return duudxp;};
  /// Access value of du / dy at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double duudy() const {return duudyp;};
  /// Access value of du / dz at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double duudz() const {return duudzp;};

  /// Access value of u_old at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double uuold() const {return uuoldp;};
  /// Access value of du_old / dx at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double duuolddx() const {return duuolddxp;};
  /// Access value of du_old / dy at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double duuolddy() const {return duuolddyp;};
  /// Access value of du_old / dz at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double duuolddz() const {return duuolddzp;};

  /// Access value of u_old_old at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double uuoldold() const {return uuoldoldp;};
  /// Access value of du_old_old / dx at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double duuoldolddx() const {return duuoldolddxp;};
  /// Access value of du_old_old / dy at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double duuoldolddy() const {return duuoldolddyp;};
  /// Access value of du_old_old / dz at the current Gauss point.
  KOKKOS_INLINE_FUNCTION
  const double duuoldolddz() const {return duuoldolddzp;};

protected:
  /// Access value for reference basis class
  const GPURefBasis * basisp;

  /// Access value of the mapping Jacobian, volume, and number of Gauss points
  double jac;
  double volp;
  double ngpp;

  /// Internal values
  double phip[BASIS_NODES_PER_ELEM];
  double dphidxp[BASIS_NODES_PER_ELEM];
  double dphidyp[BASIS_NODES_PER_ELEM];
  double dphidzp[BASIS_NODES_PER_ELEM];

  double xi;
  double eta;
  double zta;

  double uup;
  double duudxp;
  double duudyp;
  double duudzp;
  double uuoldp;
  double duuolddxp;
  double duuolddyp;
  double duuolddzp;
  double uuoldoldp;
  double duuoldolddxp;
  double duuoldolddyp;
  double duuoldolddzp;
  double xxp;
  double yyp;
  double zzp;

  /// Access value of da/db at Gauss point, a = xi, eta, zta, b = x, y, z
  double dxidx;
  double dxidy;
  double dxidz;
  double detadx;
  double detady;
  double detadz;
  double dztadx;
  double dztady;
  double dztadz;

  double dxdxi;
  double dxdeta;
  double dxdzta;
  double dydxi;
  double dydeta;
  double dydzta;
  double dzdxi;
  double dzdeta;
  double dzdzta;
};

class GPUBasisBar:public GPUBasis{
public:
  KOKKOS_INLINE_FUNCTION
  GPUBasisBar(const GPURefBasis *basis){
    basisp = basis;
    ngpp = basisp->ngp();
  };
  KOKKOS_INLINE_FUNCTION
  virtual ~GPUBasisBar(){
  };

  KOKKOS_INLINE_FUNCTION
  void computeElemData(const double x[BASIS_NODES_PER_ELEM], 
						           const double y[BASIS_NODES_PER_ELEM],  
						           const double z[BASIS_NODES_PER_ELEM]) {
    //std::cout<<"lbar 1"<<std::endl;
    nodaldiff[0] = x[1]-x[0];
    nodaldiff[1] = y[1]-y[0];
    nodaldiff[2] = z[1]-z[0];
    //std::cout<<nodaldiff[0]<<" "<<nodaldiff[1]<<" "<<nodaldiff[2]<<std::endl;
    //std::cout<<"lbar 2"<<std::endl;
}

  KOKKOS_INLINE_FUNCTION
  double getCoordsBasisWJac(const int gp,
						                const double x[BASIS_NODES_PER_ELEM], 
						                const double y[BASIS_NODES_PER_ELEM],  
						                const double z[BASIS_NODES_PER_ELEM]) {
#if 0
  // Calculate partial coordinate derivatives for Jacobian and phi derivatives
  // we could make these protected, ove this jac and volp to computeElemData as an optimization
  // Reference cell goes from -1 to 1, so need factor of .5
  dxdxi  = .5*nodaldiff[0];
  dydxi  = .5*nodaldiff[1];
  dzdxi  = .5*nodaldiff[2];

  // For 1D bar embedded in 3D, the Jacobian transformation adds a
  // factor of sqrt(det(J^T J)), for J = (dx/dxi, dy/dxi, dz/dxi)
  jac = sqrt(dxdxi*dxdxi+dydxi*dydxi+dzdxi*dzdxi);
  volp = jac*basisp->canonical_vol();
  dxidx = 1. / dxdxi;
  dxidy = 1. / dydxi;
  dxidz = 1. / dzdxi;

  // Caculate basis function and derivative at GP.
  xxp = 0.0;
  yyp = 0.0;
  zzp = 0.0;

  for (int i=0; i < basisp->nbn(); i++) {
    phip[i]=basisp->phinew(gp, i);
    dphidxp[i] = basisp->dphidxinew(gp, i)*dxidx;
    dphidyp[i] = basisp->dphidxinew(gp, i)*dxidy;
    dphidzp[i] = basisp->dphidxinew(gp, i)*dxidz;

    xxp += x[i] * phip[i];
    yyp += y[i] * phip[i];
    zzp += z[i] * phip[i];
  }

  return jac*basisp->nwt(gp);
#else
  // Compute derivatives of position with respect to reference coordinate
  const double dxdxi = 0.5 * nodaldiff[0];
  const double dydxi = 0.5 * nodaldiff[1];
  const double dzdxi = 0.5 * nodaldiff[2];

  // Compute Jacobian and inverse components
  const double jac = std::sqrt(dxdxi * dxdxi + dydxi * dydxi + dzdxi * dzdxi);
  volp = jac * basisp->canonical_vol();

  const double dxidx = 1.0 / dxdxi;
  const double dxidy = 1.0 / dydxi;
  const double dxidz = 1.0 / dzdxi;

  // Initialize global coordinates at Gauss point
  xxp = yyp = zzp = 0.0;

  const int nbn = basisp->nbn();

  for (int i = 0; i < nbn; ++i) {
    const double phi      = basisp->phinew(gp, i);
    const double dphidxi  = basisp->dphidxinew(gp, i);

    phip[i]     = phi;
    dphidxp[i]  = dphidxi * dxidx;
    dphidyp[i]  = dphidxi * dxidy;
    dphidzp[i]  = dphidxi * dxidz;

    xxp = FMA(x[i], phi, xxp);
    yyp = FMA(y[i], phi, yyp);
    zzp = FMA(z[i], phi, zzp);
  }

  return jac * basisp->nwt(gp);
#endif
  }

protected:
  /// difference in nodal coordinates
  double nodaldiff[3];
};

class GPUBasisQuad:public GPUBasis{
public:
  KOKKOS_INLINE_FUNCTION
  GPUBasisQuad(const GPURefBasis *basis){
    basisp = basis;
    ngpp = basisp->ngp();
  };
  KOKKOS_INLINE_FUNCTION
  virtual ~GPUBasisQuad(){
  };

  KOKKOS_INLINE_FUNCTION
  void computeElemData(const double x[BASIS_NODES_PER_ELEM],
						           const double y[BASIS_NODES_PER_ELEM],  
						           const double z[BASIS_NODES_PER_ELEM]) {
    // Compute differences between the corners; needed for partial derivatives and Jacobian
    // First four nodes are always the corners in exo ordering, regardless of polynomial degree
    nodaldiff[0] = x[1]-x[0];
    nodaldiff[1] = x[3]-x[0];
    nodaldiff[2] = y[1]-y[0];
    nodaldiff[3] = y[3]-y[0];
    nodaldiff[4] = z[1]-z[0];
    nodaldiff[5] = z[3]-z[0];

    nodaldiff[6] = x[2]-x[3];
    nodaldiff[7] = x[2]-x[1];
    nodaldiff[8] = y[2]-y[3];
    nodaldiff[9] = y[2]-y[1];
    nodaldiff[10] = z[2]-z[3];
    nodaldiff[11] = z[2]-z[1];
    //std::cout<<"lquad 2"<<std::endl;
  }

  KOKKOS_INLINE_FUNCTION
  double getCoordsBasisWJac(const int gp,
						                const double x[BASIS_NODES_PER_ELEM], 
						                const double y[BASIS_NODES_PER_ELEM],  
						                const double z[BASIS_NODES_PER_ELEM]) {

#if 0
  // Calculate partial coordinate derivatives for Jacobian and phi derivatives
  // we could make these protected, ove this jac and volp to computeElemData as an optimization
  xi = basisp->xi(gp);
  eta = basisp->eta(gp);

  dxdxi  = .25*( (nodaldiff[0])*(1.-eta)+(nodaldiff[6])*(1.+eta) );
  dxdeta = .25*( (nodaldiff[1])*(1.-xi)+(nodaldiff[7])*(1.+xi) );
  dydxi  = .25*( (nodaldiff[2])*(1.-eta)+(nodaldiff[8])*(1.+eta) );
  dydeta = .25*( (nodaldiff[3])*(1.-xi)+(nodaldiff[9])*(1.+xi) );
  dzdxi  = .25*( (nodaldiff[4])*(1.-eta)+(nodaldiff[10])*(1.+eta) );
  dzdeta = .25*( (nodaldiff[5])*(1.-xi)+(nodaldiff[11])*(1.+xi) );

  // For 2D quadrilateral embedded in 3D, the Jacobian transformation adds a
  // factor of sqrt(det(J^T J)); by the Cauchy-Binet formula, this in turn
  // is equal to the 2-norm of the 2x2 sub-determinants of J.
  jac = sqrt( (dzdxi * dxdeta - dxdxi * dzdeta)*(dzdxi * dxdeta - dxdxi * dzdeta)
	     +(dydxi * dzdeta - dzdxi * dydeta)*(dydxi * dzdeta - dzdxi * dydeta)
	     +(dxdxi * dydeta - dxdeta * dydxi)*(dxdxi * dydeta - dxdeta * dydxi));
  volp = jac*basisp->canonical_vol();

  // Only need derivatives dphidy, dphidx for interior integrals in xy-plane;
  // boundary integrals in xyz space only need phi, jac, x, y, z.
  // Currently does not work for interior integrals not in xy-plane!
  dxidx = dydeta / jac;
  dxidy = -dxdeta / jac;
  detadx = -dydxi / jac;
  detady = dxdxi / jac;

  // Caculate basis function and derivative at GP.
  xxp = 0.0;
  yyp = 0.0;
  zzp = 0.0;

  // x[i] is a vector of node coords, x(j, k)
  for (int i=0; i < basisp->nbn(); i++) {
    phip[i]=basisp->phinew(gp, i);
    dphidxp[i] = basisp->dphidxinew(gp, i)*dxidx+basisp->dphidetanew(gp, i)*detadx;
    dphidyp[i] = basisp->dphidxinew(gp, i)*dxidy+basisp->dphidetanew(gp, i)*detady;
    dphidzp[i] = 0.0;

    xxp += x[i] * phip[i];
    yyp += y[i] * phip[i];
    zzp += z[i] * phip[i];
  }

  return jac*basisp->nwt(gp);
#else
  // Retrieve reference coordinates
  const double xi   = basisp->xi(gp);
  const double eta  = basisp->eta(gp);

  const double one_minus_xi  = 1.0 - xi;
  const double one_plus_xi   = 1.0 + xi;
  const double one_minus_eta = 1.0 - eta;
  const double one_plus_eta  = 1.0 + eta;

  // Compute partial derivatives of coordinates w.r.t. xi, eta
  const double dxdxi  = 0.25 * (nodaldiff[0] * one_minus_eta + nodaldiff[6] * one_plus_eta);
  const double dxdeta = 0.25 * (nodaldiff[1] * one_minus_xi  + nodaldiff[7] * one_plus_xi);
  const double dydxi  = 0.25 * (nodaldiff[2] * one_minus_eta + nodaldiff[8] * one_plus_eta);
  const double dydeta = 0.25 * (nodaldiff[3] * one_minus_xi  + nodaldiff[9] * one_plus_xi);
  const double dzdxi  = 0.25 * (nodaldiff[4] * one_minus_eta + nodaldiff[10] * one_plus_eta);
  const double dzdeta = 0.25 * (nodaldiff[5] * one_minus_xi  + nodaldiff[11] * one_plus_xi);

  // Compute Jacobian using 2-norm of 2x2 sub-determinants (Cauchy-Binet)
  const double t1 = dzdxi * dxdeta - dxdxi * dzdeta;
  const double t2 = dydxi * dzdeta - dzdxi * dydeta;
  const double t3 = dxdxi * dydeta - dxdeta * dydxi;

  const double jac = std::sqrt(t1 * t1 + t2 * t2 + t3 * t3);
  volp = jac * basisp->canonical_vol();

  // Inverse Jacobian terms for transforming derivatives
  const double dxidx   = dydeta / jac;
  const double dxidy   = -dxdeta / jac;
  const double detadx  = -dydxi / jac;
  const double detady  = dxdxi / jac;

  // Initialize physical coordinate accumulators
  xxp = yyp = zzp = 0.0;

  const int nbn = basisp->nbn();

  for (int i = 0; i < nbn; ++i) {
    const double phi       = basisp->phinew(gp, i);
    const double dphidxi   = basisp->dphidxinew(gp, i);
    const double dphideta  = basisp->dphidetanew(gp, i);

    phip[i]      = phi;
    dphidxp[i]   = FMA(dphidxi, dxidx, dphideta * detadx);
    dphidyp[i]   = FMA(dphidxi, dxidy, dphideta * detady);
    dphidzp[i]   = 0.0; // No z-variation for 2D basis in xy plane

    xxp = FMA(x[i], phi, xxp);
    yyp = FMA(y[i], phi, yyp);
    zzp = FMA(z[i], phi, zzp);
  }

  return jac * basisp->nwt(gp);
#endif
  }

protected:
  /// difference in nodal coordinates
  double nodaldiff[12];
};

class GPUBasisHex:public GPUBasis{
public:
  KOKKOS_INLINE_FUNCTION
  GPUBasisHex(const GPURefBasis *basis){
    basisp = basis;
    ngpp = basisp->ngp();
  };
  KOKKOS_INLINE_FUNCTION
  virtual ~GPUBasisHex(){
  };

  KOKKOS_INLINE_FUNCTION
  void computeElemData(const double x[BASIS_NODES_PER_ELEM], 
						           const double y[BASIS_NODES_PER_ELEM],  
						           const double z[BASIS_NODES_PER_ELEM]) {
    nodaldiff[0] = x[1]-x[0];
    nodaldiff[1] = x[3]-x[0];
    nodaldiff[2] = x[4]-x[0];
    nodaldiff[3] = y[1]-y[0];
    nodaldiff[4] = y[3]-y[0];
    nodaldiff[5] = y[4]-y[0];
    nodaldiff[6] = z[1]-z[0];
    nodaldiff[7] = z[3]-z[0];
    nodaldiff[8] = z[4]-z[0];

    nodaldiff[9] = x[2]-x[3];
    nodaldiff[10] = x[2]-x[1];
    nodaldiff[11] = x[5]-x[1];
    nodaldiff[12] = y[2]-y[3];
    nodaldiff[13] = y[2]-y[1];
    nodaldiff[14] = y[5]-y[1];
    nodaldiff[15] = z[2]-z[3];
    nodaldiff[16] = z[2]-z[1];
    nodaldiff[17] = z[5]-z[1];

    nodaldiff[18] = x[5]-x[4];
    nodaldiff[19] = x[7]-x[4];
    nodaldiff[20] = x[6]-x[2];
    nodaldiff[21] = y[5]-y[4];
    nodaldiff[22] = y[7]-y[4];
    nodaldiff[23] = y[6]-y[2];
    nodaldiff[24] = z[5]-z[4];
    nodaldiff[25] = z[7]-z[4];
    nodaldiff[26] = z[6]-z[2];

    nodaldiff[27] = x[6]-x[7];
    nodaldiff[28] = x[6]-x[5];
    nodaldiff[29] = x[7]-x[3];
    nodaldiff[30] = y[6]-y[7];
    nodaldiff[31] = y[6]-y[5];
    nodaldiff[32] = y[7]-y[3];
    nodaldiff[33] = z[6]-z[7];
    nodaldiff[34] = z[6]-z[5];
    nodaldiff[35] = z[7]-z[3];
    //std::cout<<"lhex"<<std::endl;
}

  KOKKOS_INLINE_FUNCTION
  double getCoordsBasisWJac(const int gp,
                            const double x[BASIS_NODES_PER_ELEM], 
                            const double y[BASIS_NODES_PER_ELEM],  
                            const double z[BASIS_NODES_PER_ELEM]) {
#if 0
  // Calculate partial coordinate derivatives for Jacobian and phi derivatives
  // we could make these protected, ove this jac and volp to computeElemData as an optimization
  xi = basisp->xi(gp);
  eta = basisp->eta(gp);
  zta = basisp->zta(gp);

  // Caculate basis function and derivative at GP.
  dxdxi  = 0.125*( (nodaldiff[0])*(1.-eta)*(1.-zta) + (nodaldiff[9])*(1.+eta)*(1.-zta) 
	    + (nodaldiff[18])*(1.-eta)*(1.+zta) + (nodaldiff[27])*(1.+eta)*(1.+zta) );
  dxdeta = 0.125*( (nodaldiff[1])*(1.- xi)*(1.-zta) + (nodaldiff[10])*(1.+ xi)*(1.-zta) 
	    + (nodaldiff[19])*(1.- xi)*(1.+zta) + (nodaldiff[28])*(1.+ xi)*(1.+zta) );
  dxdzta = 0.125*( (nodaldiff[2])*(1.- xi)*(1.-eta) + (nodaldiff[11])*(1.+ xi)*(1.-eta)
	    + (nodaldiff[20])*(1.+ xi)*(1.+eta) + (nodaldiff[29])*(1.- xi)*(1.+eta) );
  dydxi  = 0.125*( (nodaldiff[3])*(1.-eta)*(1.-zta) + (nodaldiff[12])*(1.+eta)*(1.-zta)
	    + (nodaldiff[21])*(1.-eta)*(1.+zta) + (nodaldiff[30])*(1.+eta)*(1.+zta) );
  dydeta = 0.125*( (nodaldiff[4])*(1.- xi)*(1.-zta) + (nodaldiff[13])*(1.+ xi)*(1.-zta) 
	    + (nodaldiff[22])*(1.- xi)*(1.+zta) + (nodaldiff[31])*(1.+ xi)*(1.+zta) );
  dydzta = 0.125*( (nodaldiff[5])*(1.- xi)*(1.-eta) + (nodaldiff[14])*(1.+ xi)*(1.-eta)
	    + (nodaldiff[23])*(1.+ xi)*(1.+eta) + (nodaldiff[32])*(1.- xi)*(1.+eta) );
  dzdxi  = 0.125*( (nodaldiff[6])*(1.-eta)*(1.-zta) + (nodaldiff[15])*(1.+eta)*(1.-zta)
	    + (nodaldiff[24])*(1.-eta)*(1.+zta) + (nodaldiff[33])*(1.+eta)*(1.+zta) );
  dzdeta = 0.125*( (nodaldiff[7])*(1.- xi)*(1.-zta) + (nodaldiff[16])*(1.+ xi)*(1.-zta) 
	    + (nodaldiff[25])*(1.- xi)*(1.+zta) + (nodaldiff[34])*(1.+ xi)*(1.+zta) );
  dzdzta = 0.125*( (nodaldiff[8])*(1.- xi)*(1.-eta) + (nodaldiff[17])*(1.+ xi)*(1.-eta)
	    + (nodaldiff[26])*(1.+ xi)*(1.+eta) + (nodaldiff[35])*(1.- xi)*(1.+eta) );
  
  jac = dxdxi*(dydeta*dzdzta - dydzta*dzdeta) - dxdeta*(dydxi*dzdzta - dydzta*dzdxi) 
    + dxdzta*(dydxi*dzdeta - dydeta*dzdxi);
    
  //std::cout<<jac<<" "<<wt<<" "<<gp<<std::endl;
  volp = jac*basisp->canonical_vol();
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
  xxp=0.0;
  yyp=0.0;
  zzp=0.0;

  // x[i] is a vector of node coords, x(j, k) 
  for (int i=0; i < basisp->nbn(); i++) {
    phip[i]=basisp->phinew(gp, i);
    dphidxp[i] = basisp->dphidxinew(gp, i)*dxidx+basisp->dphidetanew(gp, i)*detadx+basisp->dphidztanew(gp, i)*dztadx;
    dphidyp[i] = basisp->dphidxinew(gp, i)*dxidy+basisp->dphidetanew(gp, i)*detady+basisp->dphidztanew(gp, i)*dztady;
    dphidzp[i] = basisp->dphidxinew(gp, i)*dxidz+basisp->dphidetanew(gp, i)*detadz+basisp->dphidztanew(gp, i)*dztadz;
    
    xxp += x[i] * phip[i];
    yyp += y[i] * phip[i];
    zzp += z[i] * phip[i];
  }

  return jac*basisp->nwt(gp);
#else
  // Get reference coordinates at Gauss point
  const double xi  = basisp->xi(gp);
  const double eta = basisp->eta(gp);
  const double zta = basisp->zta(gp);

  const double one_minus_xi  = 1.0 - xi;
  const double one_plus_xi   = 1.0 + xi;
  const double one_minus_eta = 1.0 - eta;
  const double one_plus_eta  = 1.0 + eta;
  const double one_minus_zta = 1.0 - zta;
  const double one_plus_zta  = 1.0 + zta;

  // Compute coordinate derivatives ∂x/∂ξ, ∂x/∂η, ∂x/∂ζ, etc.
  auto blend = [](double a, double b) { return 0.125 * a * b; };

  const double dxdxi   = blend(nodaldiff[0],  one_minus_eta * one_minus_zta) + 
                         blend(nodaldiff[9],  one_plus_eta  * one_minus_zta) +
                         blend(nodaldiff[18], one_minus_eta * one_plus_zta) + 
                         blend(nodaldiff[27], one_plus_eta  * one_plus_zta);

  const double dxdeta  = blend(nodaldiff[1],  one_minus_xi * one_minus_zta) + 
                         blend(nodaldiff[10], one_plus_xi  * one_minus_zta) +
                         blend(nodaldiff[19], one_minus_xi * one_plus_zta) + 
                         blend(nodaldiff[28], one_plus_xi  * one_plus_zta);

  const double dxdzta  = blend(nodaldiff[2],  one_minus_xi * one_minus_eta) + 
                         blend(nodaldiff[11], one_plus_xi  * one_minus_eta) +
                         blend(nodaldiff[20], one_plus_xi  * one_plus_eta) + 
                         blend(nodaldiff[29], one_minus_xi * one_plus_eta);

  const double dydxi   = blend(nodaldiff[3],  one_minus_eta * one_minus_zta) + 
                         blend(nodaldiff[12], one_plus_eta  * one_minus_zta) +
                         blend(nodaldiff[21], one_minus_eta * one_plus_zta) + 
                         blend(nodaldiff[30], one_plus_eta  * one_plus_zta);

  const double dydeta  = blend(nodaldiff[4],  one_minus_xi * one_minus_zta) + 
                         blend(nodaldiff[13], one_plus_xi  * one_minus_zta) +
                         blend(nodaldiff[22], one_minus_xi * one_plus_zta) + 
                         blend(nodaldiff[31], one_plus_xi  * one_plus_zta);

  const double dydzta  = blend(nodaldiff[5],  one_minus_xi * one_minus_eta) + 
                         blend(nodaldiff[14], one_plus_xi  * one_minus_eta) +
                         blend(nodaldiff[23], one_plus_xi  * one_plus_eta) + 
                         blend(nodaldiff[32], one_minus_xi * one_plus_eta);

  const double dzdxi   = blend(nodaldiff[6],  one_minus_eta * one_minus_zta) + 
                         blend(nodaldiff[15], one_plus_eta  * one_minus_zta) +
                         blend(nodaldiff[24], one_minus_eta * one_plus_zta) + 
                         blend(nodaldiff[33], one_plus_eta  * one_plus_zta);

  const double dzdeta  = blend(nodaldiff[7],  one_minus_xi * one_minus_zta) + 
                         blend(nodaldiff[16], one_plus_xi  * one_minus_zta) +
                         blend(nodaldiff[25], one_minus_xi * one_plus_zta) + 
                         blend(nodaldiff[34], one_plus_xi  * one_plus_zta);

  const double dzdzta  = blend(nodaldiff[8],  one_minus_xi * one_minus_eta) + 
                         blend(nodaldiff[17], one_plus_xi  * one_minus_eta) +
                         blend(nodaldiff[26], one_plus_xi  * one_plus_eta) + 
                         blend(nodaldiff[35], one_minus_xi * one_plus_eta);

  // Compute Jacobian determinant (scalar triple product)
  const double jac = 
    dxdxi  * (dydeta * dzdzta - dydzta * dzdeta) -
    dxdeta * (dydxi  * dzdzta - dydzta * dzdxi ) +
    dxdzta * (dydxi  * dzdeta - dydeta * dzdxi );

  volp = jac * basisp->canonical_vol();

  // Inverse Jacobian: entries for transforming gradient basis
  const double inv_jac = 1.0 / jac;

  const double dxidx   = inv_jac * (-dydzta * dzdeta + dydeta * dzdzta);
  const double dxidy   = inv_jac * ( dxdzta * dzdeta - dxdeta * dzdzta);
  const double dxidz   = inv_jac * (-dxdzta * dydeta + dxdeta * dydzta);

  const double detadx  = inv_jac * ( dydzta * dzdxi  - dydxi  * dzdzta);
  const double detady  = inv_jac * (-dxdzta * dzdxi  + dxdxi  * dzdzta);
  const double detadz  = inv_jac * ( dxdzta * dydxi  - dxdxi  * dydzta);

  const double dztadx  = inv_jac * ( dydxi  * dzdeta - dydeta * dzdxi );
  const double dztady  = inv_jac * (-dxdxi  * dzdeta + dxdeta * dzdxi );
  const double dztadz  = inv_jac * ( dxdxi  * dydeta - dxdeta * dydxi );

  // Compute global coordinates and gradients at Gauss point
  xxp = yyp = zzp = 0.0;

  const int nbn = basisp->nbn();
  for (int i = 0; i < nbn; ++i) {
    const double phi      = basisp->phinew(gp, i);
    const double dphidxi  = basisp->dphidxinew(gp, i);
    const double dphideta = basisp->dphidetanew(gp, i);
    const double dphidzta = basisp->dphidztanew(gp, i);

    phip[i] = phi;

    dphidxp[i] = FMA(dphidxi, dxidx,  FMA(dphideta, detadx,  dphidzta * dztadx));
    dphidyp[i] = FMA(dphidxi, dxidy,  FMA(dphideta, detady,  dphidzta * dztady));
    dphidzp[i] = FMA(dphidxi, dxidz,  FMA(dphideta, detadz,  dphidzta * dztadz));

    xxp = FMA(x[i], phi, xxp);
    yyp = FMA(y[i], phi, yyp);
    zzp = FMA(z[i], phi, zzp);
  }

  return jac * basisp->nwt(gp);
#endif
  }

protected:
  /// difference in nodal coordinates
  double nodaldiff[36];
};

class GPUBasisTri:public GPUBasis{
public:
  KOKKOS_INLINE_FUNCTION
  GPUBasisTri(const GPURefBasis *basis){
    basisp = basis;
    ngpp = basisp->ngp();
  };
  KOKKOS_INLINE_FUNCTION
  virtual ~GPUBasisTri(){
  };

  KOKKOS_INLINE_FUNCTION
  void computeElemData(const double x[BASIS_NODES_PER_ELEM],
						           const double y[BASIS_NODES_PER_ELEM],  
						           const double z[BASIS_NODES_PER_ELEM]) {
    nodaldiff[0] = x[1]-x[0];
    nodaldiff[1] = x[2]-x[0];
    nodaldiff[2] = y[1]-y[0];
    nodaldiff[3] = y[2]-y[0];
    nodaldiff[4] = z[1]-z[0];
    nodaldiff[5] = z[2]-z[0];
  }

  KOKKOS_INLINE_FUNCTION
  double getCoordsBasisWJac(const int gp,
						                const double x[BASIS_NODES_PER_ELEM], 
						                const double y[BASIS_NODES_PER_ELEM],  
						                const double z[BASIS_NODES_PER_ELEM]) {
#if 0
  // Calculate partial coordinate derivatives for Jacobian and phi derivatives
  dxdxi  = nodaldiff[0];
  dxdeta = nodaldiff[1];
  dydxi  = nodaldiff[2];
  dydeta = nodaldiff[3];
  dzdxi  = nodaldiff[4];
  dzdeta = nodaldiff[5];

  // For 2D triangle embedded in 3D, situation as for 2D quadrilateral
  jac = sqrt( (dzdxi * dxdeta - dxdxi * dzdeta)*(dzdxi * dxdeta - dxdxi * dzdeta)
	     +(dydxi * dzdeta - dzdxi * dydeta)*(dydxi * dzdeta - dzdxi * dydeta)
	     +(dxdxi * dydeta - dxdeta * dydxi)*(dxdxi * dydeta - dxdeta * dydxi));
  volp = jac*basisp->canonical_vol();
  // As for quads, currently does not work for interior integrals not in xy-plane!
  dxidx = dydeta / jac;
  dxidy = -dxdeta / jac;
  detadx = -dydxi / jac;
  detady = dxdxi / jac;

  // Caculate basis function and derivative at GP.
  xxp = 0.0;
  yyp = 0.0;
  zzp = 0.0;

  // x[i] is a vector of node coords, x(j, k)
  for (int i=0; i < basisp->nbn(); i++) {
    phip[i]=basisp->phinew(gp, i);
    dphidxp[i] = basisp->dphidxinew(gp, i)*dxidx+basisp->dphidetanew(gp, i)*detadx;
    dphidyp[i] = basisp->dphidxinew(gp, i)*dxidy+basisp->dphidetanew(gp, i)*detady;
    dphidzp[i] = 0.0;

    xxp += x[i] * phip[i];
    yyp += y[i] * phip[i];
    zzp += z[i] * phip[i];
  }

  return jac*basisp->nwt(gp);
#else
  // Load precomputed geometric differentials
  const double dxdxi   = nodaldiff[0];
  const double dxdeta  = nodaldiff[1];
  const double dydxi   = nodaldiff[2];
  const double dydeta  = nodaldiff[3];
  const double dzdxi   = nodaldiff[4];
  const double dzdeta  = nodaldiff[5];

  // Compute Jacobian using 2-norm of 2x2 sub-determinants (Cauchy-Binet)
  const double t1 = dzdxi * dxdeta - dxdxi * dzdeta;
  const double t2 = dydxi * dzdeta - dzdxi * dydeta;
  const double t3 = dxdxi * dydeta - dxdeta * dydxi;

  const double jac = std::sqrt(t1 * t1 + t2 * t2 + t3 * t3);
  volp = jac * basisp->canonical_vol();

  // Inverse Jacobian terms (for projection of gradient to physical space)
  const double inv_jac = 1.0 / jac;
  const double dxidx   =  dydeta * inv_jac;
  const double dxidy   = -dxdeta * inv_jac;
  const double detadx  = -dydxi  * inv_jac;
  const double detady  =  dxdxi  * inv_jac;

  // Initialize physical coordinate accumulators
  xxp = yyp = zzp = 0.0;

  const int nbn = basisp->nbn();
  for (int i = 0; i < nbn; ++i) {
    const double phi       = basisp->phinew(gp, i);
    const double dphidxi   = basisp->dphidxinew(gp, i);
    const double dphideta  = basisp->dphidetanew(gp, i);

    phip[i] = phi;

    // Use fused multiply-add for gradient projection
    dphidxp[i] = FMA(dphidxi, dxidx, dphideta * detadx);
    dphidyp[i] = FMA(dphidxi, dxidy, dphideta * detady);
    dphidzp[i] = 0.0; // No derivative in z for 2D triangle in 3D

    // Interpolate physical coordinates
    xxp = FMA(x[i], phi, xxp);
    yyp = FMA(y[i], phi, yyp);
    zzp = FMA(z[i], phi, zzp);
  }

  return jac * basisp->nwt(gp);
#endif
  }

  /// difference in nodal coordinates
  double nodaldiff[6];

};

class GPUBasisTet:public GPUBasis{
public:
  KOKKOS_INLINE_FUNCTION
  GPUBasisTet(const GPURefBasis *basis){
    basisp = basis;
    ngpp = basisp->ngp();
  };
  KOKKOS_INLINE_FUNCTION
  virtual ~GPUBasisTet(){
  };

  KOKKOS_INLINE_FUNCTION
  void computeElemData(const double x[BASIS_NODES_PER_ELEM],
						           const double y[BASIS_NODES_PER_ELEM],  
						           const double z[BASIS_NODES_PER_ELEM]) {
    nodaldiff[0] = x[1]-x[0];
    nodaldiff[1] = x[2]-x[0];
    nodaldiff[2] = x[3]-x[0];
    nodaldiff[3] = y[1]-y[0];
    nodaldiff[4] = y[2]-y[0];
    nodaldiff[5] = y[3]-y[0];
    nodaldiff[6] = z[1]-z[0];
    nodaldiff[7] = z[2]-z[0];
    nodaldiff[8] = z[3]-z[0];
  }

  KOKKOS_INLINE_FUNCTION
  double getCoordsBasisWJac(const int gp,
						                const double x[BASIS_NODES_PER_ELEM], 
						                const double y[BASIS_NODES_PER_ELEM],  
						                const double z[BASIS_NODES_PER_ELEM]) {
#if 0
  // Calculate partial coordinate derivatives for Jacobian and phi derivatives
  dxdxi  = nodaldiff[0];
  dxdeta = nodaldiff[1];
  dxdzta = nodaldiff[2];
  dydxi  = nodaldiff[3];
  dydeta = nodaldiff[4];
  dydzta = nodaldiff[5];
  dzdxi  = nodaldiff[6];
  dzdeta = nodaldiff[7];
  dzdzta = nodaldiff[8];

  jac = dxdxi*(dydeta*dzdzta - dydzta*dzdeta) - dxdeta*(dydxi*dzdzta - dydzta*dzdxi)
    + dxdzta*(dydxi*dzdeta - dydeta*dzdxi);
  volp = jac*basisp->canonical_vol();
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
  xxp = 0.0;
  yyp = 0.0;
  zzp = 0.0;

  // x[i] is a vector of node coords, x(j, k)
  for (int i=0; i < basisp->nbn(); i++) {
    phip[i]=basisp->phinew(gp, i);
    dphidxp[i] = basisp->dphidxinew(gp, i)*dxidx+basisp->dphidetanew(gp, i)*detadx+basisp->dphidztanew(gp, i)*dztadx;
    dphidyp[i] = basisp->dphidxinew(gp, i)*dxidy+basisp->dphidetanew(gp, i)*detady+basisp->dphidztanew(gp, i)*dztady;
    dphidzp[i] = basisp->dphidxinew(gp, i)*dxidz+basisp->dphidetanew(gp, i)*detadz+basisp->dphidztanew(gp, i)*dztadz;

    xxp += x[i] * phip[i];
    yyp += y[i] * phip[i];
    zzp += z[i] * phip[i];
  }

  return jac*basisp->nwt(gp);
#else
  // Load geometric derivatives
  const double dxdxi   = nodaldiff[0];
  const double dxdeta  = nodaldiff[1];
  const double dxdzta  = nodaldiff[2];
  const double dydxi   = nodaldiff[3];
  const double dydeta  = nodaldiff[4];
  const double dydzta  = nodaldiff[5];
  const double dzdxi   = nodaldiff[6];
  const double dzdeta  = nodaldiff[7];
  const double dzdzta  = nodaldiff[8];

  // Compute Jacobian determinant (scalar triple product)
  const double jac = 
      dxdxi  * (dydeta * dzdzta - dydzta * dzdeta) -
      dxdeta * (dydxi  * dzdzta - dydzta * dzdxi ) +
      dxdzta * (dydxi  * dzdeta - dydeta * dzdxi );

  volp = jac * basisp->canonical_vol();

  // Inverse Jacobian transformation components
  const double inv_jac = 1.0 / jac;

  const double dxidx   = inv_jac * (-dydzta * dzdeta + dydeta * dzdzta);
  const double dxidy   = inv_jac * ( dxdzta * dzdeta - dxdeta * dzdzta);
  const double dxidz   = inv_jac * (-dxdzta * dydeta + dxdeta * dydzta);

  const double detadx  = inv_jac * ( dydzta * dzdxi  - dydxi  * dzdzta);
  const double detady  = inv_jac * (-dxdzta * dzdxi  + dxdxi  * dzdzta);
  const double detadz  = inv_jac * ( dxdzta * dydxi  - dxdxi  * dydzta);

  const double dztadx  = inv_jac * ( dydxi  * dzdeta - dydeta * dzdxi );
  const double dztady  = inv_jac * (-dxdxi  * dzdeta + dxdeta * dzdxi );
  const double dztadz  = inv_jac * ( dxdxi  * dydeta - dxdeta * dydxi );

  // Initialize interpolated coordinates
  xxp = yyp = zzp = 0.0;

  const int nbn = basisp->nbn();
  for (int i = 0; i < nbn; ++i) {
    const double phi       = basisp->phinew(gp, i);
    const double dphidxi   = basisp->dphidxinew(gp, i);
    const double dphideta  = basisp->dphidetanew(gp, i);
    const double dphidzta  = basisp->dphidztanew(gp, i);

    phip[i] = phi;

    dphidxp[i] = FMA(dphidxi, dxidx, 
                    FMA(dphideta, detadx, 
                      dphidzta * dztadx));

    dphidyp[i] = FMA(dphidxi, dxidy, 
                    FMA(dphideta, detady, 
                      dphidzta * dztady));

    dphidzp[i] = FMA(dphidxi, dxidz, 
                    FMA(dphideta, detadz, 
                      dphidzta * dztadz));

    xxp = FMA(x[i], phi, xxp);
    yyp = FMA(y[i], phi, yyp);
    zzp = FMA(z[i], phi, zzp);
  }

  return jac * basisp->nwt(gp);
#endif
  }

  /// difference in nodal coordinates
  double nodaldiff[9];

};

// GAW TO DOs:
// - QTet still needs testing! Easy to get exodus ordering wrong
// - Maybe do base classes  GPUBasis2D, GPUBasis3D from which GPUBasisTri/Quad  and GPUBasisHex/Tet inherit.
// -- Difference Tri/Quad and Tet/Hex only lies in computeElemData and dx,y,z/dxi,eta

#endif
