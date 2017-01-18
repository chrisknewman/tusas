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
  /** Evaluate the 2D basis functions at Guass point gp given x, y. This function needs to be called before any accessor function.*/
  virtual void getBasis(const int gp, const double *x, const double *y){getBasis(gp, x, y, NULL, NULL, NULL, NULL);};
  /// Evaluate the basis functions at the specified gauss point
  /** Evaluate the 3D basis functions at Guass point gp given x, y, z.  This function needs to be called before any accessor function.*/
  virtual void getBasis(const int gp, const double *x, const double *y, const double *z){getBasis(gp, x, y, z, NULL, NULL, NULL);};
  /// Evaluate the basis functions at the specified gauss point
  /** Evaluate the 3D basis functions at Guass point gp given x, y, z and interpolate u.  This function needs to be called before any accessor function.*/
  virtual void getBasis(const int gp, const double *x, const double *y, const double *z, const double *u){getBasis(gp, x, y, z, u, NULL, NULL);};
  /// Evaluate the basis functions at the specified gauss point
  /** Evaluate the 3D basis functions at Guass point gp given x, y, z and interpolate u, uold.  This function needs to be called before any accessor function.*/
  virtual void getBasis(const int gp, const double *x, const double *y, const double *z, const double *u, const double *uold){getBasis(gp, x, y, z, u, uold, NULL);};
  /// Evaluate the basis functions at the specified gauss point
  /** Evaluate the 3D basis functions at Guass point gp given x, y, z and interpolate u, uold, uoldold.  This function needs to be called before any accessor function.*/
  virtual void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold){exit(0);};

  /// Set the number of Guass points.
  void setN(int N, double *abscissa, double *weight);    

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
  /// Access a pointer to the coordinates of the Guass points in canonical space.
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

  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);

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

  BasisLQuad* clone() const{ return new BasisLQuad(*this); }
  char type() { return 1; }

  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);

 public:
  // Variables that are calculated at the gauss point
  int sngp;
};

/// Implementation of 2-D biquadratic triangle element.
/** 6 node element with 3 Gauss points. */
class BasisQTri : public Basis {

 public:

  /// Constructor
  BasisQTri();

  /// Destructor
  ~BasisQTri();

  BasisQTri* clone() const{ return new BasisQTri(*this); }
  char type() { return 1; }

  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);


 public:

};

/// Implementation of 2-D biquadratic quadrilateral element.
/** 9 node element with 9 Gauss points. */
class BasisQQuad : public Basis {

 public:

  /// Constructor
  BasisQQuad();

  /// Destructor
  ~BasisQQuad();    

  BasisQQuad* clone() const{ return new BasisQQuad(*this); }
  char type() { return 1; }

  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);



 public:
  // Variables that are calculated at the gauss point
  int sngp;
};


/// Implementation of 3-D bilinear hexahedral element.
/** 8 node element with number of Gauss points specified in constructor, defaults to 8. */
class BasisLHex : public Basis {

 public:

  /// Constructor
  /** Default constructor with 8 Gauss points). */
  BasisLHex();
  /// Constructor
  /** Number of Gauss points = sngp (sngp refers to 1 dimension of a tensor product, ie sngp = 2 is really 8 Gauss points). */
  BasisLHex(int sngp);

  /// Destructor
  ~BasisLHex();

  BasisLHex* clone() const{ return new BasisLHex(*this); }
  char type() { return 1; }

  // Calculates the values of u and x at the specified gauss point
  void getBasis(const int gp, const double *x, const double *y);

  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);

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

  // Calculates the values of u and x at the specified gauss point
  void getBasis(const int gp, const double *x, const double *y);

  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);


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

  BasisLBar* clone() const{ return new BasisLBar(*this); }
  char type() { return 1; }


  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);



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

  BasisQBar* clone() const{ return new BasisQBar(*this); }
  char type() { return 1; }


  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);



 public:
  // Variables that are calculated at the gauss point
  int sngp;
};

#endif

