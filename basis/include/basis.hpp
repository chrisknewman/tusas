//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef _NOX_EXAMPLE_EPETRA_LINEAR_BASIS_H
#define _NOX_EXAMPLE_EPETRA_LINEAR_BASIS_H
	
#include <Kokkos_Core.hpp>

#ifdef KOKKOS_HAVE_CUDA
#define TUSAS_CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define TUSAS_CUDA_CALLABLE_MEMBER
#endif

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
};

/// Implementation of 2-D bilinear triangle element.
/** 3 node element with number of Gauss points specified in constructor, defaults to 1. */
class BasisLTri : public Basis {

 public:

  /// Constructor
  /** Number of Gauss points = sngp, defaults to 1. */
  BasisLTri(int n = 1){
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
  double phi1_xi = -xi[gp]*(1-xi[gp])/2;
  double phi2_xi = xi[gp]*(1+xi[gp])/2;
  double phi3_xi = 1-xi[gp]*xi[gp];

  double phi1_eta = -eta[gp]*(1-eta[gp])/2;
  double phi2_eta = eta[gp]*(1+eta[gp])/2;
  double phi3_eta = 1-eta[gp]*eta[gp];
 
  //double phi_xi[3] = [phi1_xi,phi2_xi,phi3_xi];
  //double phi_eta[3] = [phi1_eta,phi2_eta,phi3_eta];

  //printf("gp = %d\n",gp);
  //printf("_xi %f %f %f:\n", phi1_xi, phi2_xi, phi3_xi);
  //printf("_eta %f %f %f:\n\n", phi1_eta, phi2_eta, phi3_eta);

  phi[0] = (-xi[gp]*(1-xi[gp])/2)*(-eta[gp]*(1-eta[gp])/2);
  phi[1] = (xi[gp]*(1+xi[gp])/2)*(-eta[gp]*(1-eta[gp])/2);
  phi[2] = (xi[gp]*(1+xi[gp])/2)*(eta[gp]*(1+eta[gp])/2);
  phi[3] = (-xi[gp]*(1-xi[gp])/2)*(eta[gp]*(1+eta[gp])/2);
  
  phi[4] = (1-xi[gp]*xi[gp])*(-eta[gp]*(1-eta[gp])/2);
  phi[5] = (xi[gp]*(1+xi[gp])/2)*(1-eta[gp]*eta[gp]);
  phi[6] = (1-xi[gp]*xi[gp])*(eta[gp]*(1+eta[gp])/2);
  phi[7] = (-xi[gp]*(1-xi[gp])/2)*(1-eta[gp]*eta[gp]);
  
  phi[8] = (1-xi[gp]*xi[gp])*(1-eta[gp]*eta[gp]);

  double dphi1dxi = (-1+2*xi[gp])/2;
  double dphi2dxi = (1+2*xi[gp])/2;
  double dphi3dxi = (-2*xi[gp]);

  double dphi1deta = (-1+2*eta[gp])/2;
  double dphi2deta = (1+2*eta[gp])/2;
  double dphi3deta = -2*eta[gp];
  
  
  dphidxi[0] = ((-1+2*xi[gp])/2)*(-eta[gp]*(1-eta[gp])/2);
  dphidxi[1] = ((1+2*xi[gp])/2)*(-eta[gp]*(1-eta[gp])/2);
  dphidxi[2] = ((1+2*xi[gp])/2)*(eta[gp]*(1+eta[gp])/2);
  dphidxi[3] = ((-1+2*xi[gp])/2)*(eta[gp]*(1+eta[gp])/2);
  dphidxi[4] = (-2*xi[gp])*(-eta[gp]*(1-eta[gp])/2);
  dphidxi[5] = ((1+2*xi[gp])/2)*(1-eta[gp]*eta[gp]);
  dphidxi[6] = (-2*xi[gp])*(eta[gp]*(1+eta[gp])/2);
  dphidxi[7] = ((-1+2*xi[gp])/2)*(1-eta[gp]*eta[gp]);
  dphidxi[8] = (-2*xi[gp])*(1-eta[gp]*eta[gp]);


  dphideta[0] = (-xi[gp]*(1-xi[gp])/2)*((-1+2*eta[gp])/2);
  dphideta[1] = (xi[gp]*(1+xi[gp])/2)*((-1+2*eta[gp])/2);
  dphideta[2] = (xi[gp]*(1+xi[gp])/2)*((1+2*eta[gp])/2);
  dphideta[3] = (-xi[gp]*(1-xi[gp])/2)*((1+2*eta[gp])/2);
  dphideta[4] = (1-xi[gp]*xi[gp])*((-1+2*eta[gp])/2);
  dphideta[5] = (xi[gp]*(1+xi[gp])/2)*(-2*eta[gp]);
  dphideta[6] = (1-xi[gp]*xi[gp])*((1+2*eta[gp])/2);
  dphideta[7] = (-xi[gp]*(1-xi[gp])/2)*(-2*eta[gp]);
  dphideta[8] = (1-xi[gp]*xi[gp])*(-2*eta[gp]);

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
//   if (jac <= 0){
//     printf("\n\n\n\n\n\nnonpositive jacobian \n\n\n\n\n\n");
//   }
 
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

#define BASIS_NODES_PER_ELEM 27

class GPUBasis{

public:

  TUSAS_CUDA_CALLABLE_MEMBER GPUBasis(){};
  TUSAS_CUDA_CALLABLE_MEMBER virtual ~GPUBasis(){};
  TUSAS_CUDA_CALLABLE_MEMBER virtual void getBasis(const int gp,
						   const double x[BASIS_NODES_PER_ELEM], 
						   const double y[BASIS_NODES_PER_ELEM],  
						   const double z[BASIS_NODES_PER_ELEM],
						   const double u[BASIS_NODES_PER_ELEM],
						   const double uold[BASIS_NODES_PER_ELEM],
						   const double uoldold[BASIS_NODES_PER_ELEM]) {};

  
    /// Access number of Gauss points.
  int ngp;

  double phi[BASIS_NODES_PER_ELEM];
  double dphidxi[BASIS_NODES_PER_ELEM];
  double dphideta[BASIS_NODES_PER_ELEM];

  /// Access value of u at the current Gauss point.
  double uu;
  /// Access value of u_old at the current Gauss point.
  double uuold;
  /// Access value of du / dx at the current Gauss point.
  double dudx;
  /// Access value of du / dy at the current Gauss point.
  double dudy;
  /// Access value of dxi / dx  at the current Gauss point.
  double dxidx;
  /// Access value of dxi / dy  at the current Gauss point.
  double dxidy;
  /// Access value of deta / dx  at the current Gauss point.
  double detadx;
  /// Access value of deta / dy  at the current Gauss point.
  double detady;
  /// Access value of the derivative of the basis function wrt to x at the current Gauss point.
  double dphidx[BASIS_NODES_PER_ELEM];
  /// Access value of the derivative of the basis function wrt to y at the current Gauss point.
  double dphidy[BASIS_NODES_PER_ELEM];

  /// Access value of the Gauss weight  at the current Gauss point.
  double wt;
  /// Access value of the mapping Jacobian.
  double jac;

  // Variables that are calculated at the gauss point
  /// Access number of Gauss points.
  int sngp;
  /// Access value of dphi / dzta  at the current Gauss point.
  double dphidzta[BASIS_NODES_PER_ELEM];
  
  /// Access value of dxi / dz  at the current Gauss point.
  double dxidz;

  /// Access value of deta / dz  at the current Gauss point.
  double detadz;

  /// Access value of dzta / dx  at the current Gauss point.
  double dztadx;
  /// Access value of dzta / dy  at the current Gauss point.
  double dztady;
  /// Access value of dzta / dz  at the current Gauss point.
  double dztadz;

  /// Access value of du / dz at the current Gauss point.
  double dudz;

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

  /// Access value of the derivative of the basis function wrt to z at the current Gauss point.
  double dphidz[BASIS_NODES_PER_ELEM];
  
protected:
  /// Access a pointer to the coordinates of the Gauss points in canonical space.
  double abscissa[BASIS_NODES_PER_ELEM];
  /// Access a pointer to the Gauss weights.
  double weight[BASIS_NODES_PER_ELEM];
  /// Access a pointer to the xi coordinate at each Gauss point.
  double xi[BASIS_NODES_PER_ELEM];
  //double *xi;
  /// Access a pointer to the eta coordinate at each Gauss point.
  double eta[BASIS_NODES_PER_ELEM];
  /// Access a pointer to the zta coordinate at each Gauss point.
  double zta[BASIS_NODES_PER_ELEM];
  /// Access the number of Gauss weights.
  double nwt[BASIS_NODES_PER_ELEM];


};

class GPUBasisLQuad:public GPUBasis{
public:

  TUSAS_CUDA_CALLABLE_MEMBER GPUBasisLQuad(){
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
  abscissa[0] = -1.0/1.732050807568877;
  abscissa[1] =  1.0/1.732050807568877;
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
  }
  
  TUSAS_CUDA_CALLABLE_MEMBER ~GPUBasisLQuad(){}
  TUSAS_CUDA_CALLABLE_MEMBER void getBasis(const int gp,const  double x[BASIS_NODES_PER_ELEM], const  double y[BASIS_NODES_PER_ELEM],  const double z[BASIS_NODES_PER_ELEM],const  double u[BASIS_NODES_PER_ELEM],const  double uold[BASIS_NODES_PER_ELEM],const  double uoldold[BASIS_NODES_PER_ELEM]) {

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
//     if( uoldold ){
//       uuoldold += uoldold[i] * phi[i];
//       duoldolddx += uoldold[i] * dphidx[i];
//       duoldolddy += uoldold[i]* dphidy[i];
//     }
  }
  
  return;
  }

};

#endif

