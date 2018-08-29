//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef _NOX_EXAMPLE_EPETRA_LINEAR_BASIS_H
#define _NOX_EXAMPLE_EPETRA_LINEAR_BASIS_H

/// Base class for computation of finite element basis.
/** All basis classes inherit from this. */
class Basis {

 public:

  /// Constructor
  Basis(){};

  /// Destructor
  virtual ~Basis(){}

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
			 ){exit(0);};
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
			 ){exit(0);};

  /// Set the number of Gauss points.
  void setN(const int N, ///< number of Gauss points (input)
	    double *abscissa, ///< abscissa array
	    double *weight ///< weight array
	    );    

  /// Required for particular implementation
  virtual Basis* clone() const {exit(0);};
  /// Required for particular implementation
  virtual char type() {exit(0);};

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
};

/// Implementation of 2-D bilinear triangle element.
/** 3 node element with number of Gauss points specified in constructor, defaults to 1. */
class BasisLTri : public Basis {

 public:

  /// Constructor
  /** Number of Gauss points = sngp, defaults to 1. */
  BasisLTri(int sngp = 1);

  /// Destructor
  virtual ~BasisLTri();

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
		 );

  BasisLTri* clone() const{ return new BasisLTri(*this); }
  char type() { return 1; }

 public:
};

/// Implementation of 2-D bilinear quadrilateral element.
/** 4 node element with number of Gauss points specified in constructor, defaults to 4. */
class BasisLQuad : public Basis {

 public:

  /// Constructor
  /** Number of Gauss points = sngp, defaults to 4 (sngp refers to 1 dimension of a tensor product, ie sngp = 2 is really 4 Gauss points). */
  BasisLQuad(int sngp = 2);

  /// Destructor
  ~BasisLQuad();

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
		 );

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
		  );

  BasisLQuad* clone() const{ return new BasisLQuad(*this); }
  char type() { return 1; }

 public:
  // Variables that are calculated at the gauss point
  int sngp;
};

/// Implementation of 2-D biquadratic triangle element.
/** 6 node element with 3 Gauss points. */
class BasisQTri : public Basis {

 public:

  /// Constructor
  BasisQTri(int ngp = 3);

  /// Destructor
  ~BasisQTri();

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
		 );

  BasisQTri* clone() const{ return new BasisQTri(*this); }
  char type() { return 1; }

 public:

};

/// Implementation of 2-D biquadratic quadrilateral element.
/** 9 node element with 9 Gauss points. */
class BasisQQuad : public Basis {

 public:

  /// Constructor
  BasisQQuad(int sngp = 3);

  /// Destructor
  ~BasisQQuad();

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
		 );    

  BasisQQuad* clone() const{ return new BasisQQuad(*this); }
  char type() { return 1; }

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
  BasisLHex(int sngp = 2);

  /// Destructor
  ~BasisLHex();

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
		 ); 

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
		  );
 
  BasisLHex* clone() const{ return new BasisLHex(*this); }
  char type() { return 1; }

  // Calculates the values of u and x at the specified gauss point
  void getBasis(const int gp,    ///< current Gauss point (input)
		const double *x,    ///< x array (input) 
		const double *y   ///< y array (input) 
		);

 public:
  // Variables that are calculated at the gauss point
  int sngp;
};

/// Implementation of 3-D bilinear tetrahedral element.
/** 4 node element with 4 Gauss points. */
class BasisLTet : public Basis {

 public:

  /// Constructor
  BasisLTet();
//   BasisLTet(int sngp);

  /// Destructor
  ~BasisLTet();

  BasisLTet* clone() const{ return new BasisLTet(*this); }
  char type() { return 1; }

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
		 );  

  // Calculates the values of u and x at the specified gauss point
  void getBasis(const int gp,     ///< current Gauss point (input)
		const double *x,     ///< x array (input)
		const double *y   ///< y array (input) 
		);

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
  BasisLBar(int sngp = 2);

  /// Destructor
  ~BasisLBar();

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
		 );  

  BasisLBar* clone() const{ return new BasisLBar(*this); }
  char type() { return 1; }

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
  BasisQBar(int sngp = 3);

  /// Destructor
  ~BasisQBar();

  void getBasis( const int gp,   ///< current Gauss point (input) 
		 const double *x,   ///< x array (input) 
		 const double *y,   ///< y array (input) 
		 const double *z,   ///< z array (input)
		 const double *u,  ///< u (solution)  array (input)  
		 const double *uold,  ///< uold (solution)  array (input)  
		 const double *uoldold ///< uoldold (solution)  array (input)
		 );  

  BasisQBar* clone() const{ return new BasisQBar(*this); }
  char type() { return 1; }

 public:
  // Variables that are calculated at the gauss point
  int sngp;
};class OMPBasisLQuad{
public:
  int sngp;


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
  /// Access number of Gauss points.
  int ngp;
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

#endif

