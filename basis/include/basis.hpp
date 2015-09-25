// 1D Linear basis function for finite element method

#ifndef _NOX_EXAMPLE_EPETRA_LINEAR_BASIS_H
#define _NOX_EXAMPLE_EPETRA_LINEAR_BASIS_H

class Basis {

 public:

  // Constructor
  Basis(){}

  // Destructor
  virtual ~Basis(){}

  // Calculates the values of u and x at the specified gauss point
  virtual void getBasis(const int gp, const double *x, const double *y){getBasis(gp, x, y, NULL, NULL, NULL, NULL);};
  virtual void getBasis(const int gp, const double *x, const double *y, const double *z){getBasis(gp, x, y, z, NULL, NULL, NULL);};

  virtual void getBasis(const int gp, const double *x, const double *y, const double *z, const double *u){getBasis(gp, x, y, z, u, NULL, NULL);};
  virtual void getBasis(const int gp, const double *x, const double *y, const double *z, const double *u, const double *uold){getBasis(gp, x, y, z, u, uold, NULL);};
  virtual void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold) = 0;

  void setN(int N, double *abscissa, double *weight);

 public:
  // Variables that are calculated at the gauss point
  int ngp;
  double *phi, *dphidxi, *dphideta, *dphidzta, *abscissa, *weight; 
  double xi, eta, zta, wt, jac;
  double dxidx, dxidy,dxidz, detadx, detady, detadz, dztadx, dztady, dztadz;
  double uu, xx, yy, zz, dudx, dudy, dudz;
  double uuold, duolddx, duolddy, duolddz;
  double uuoldold, duoldolddx, duoldolddy, duoldolddz;
  double * dphidx;
  double * dphidy;
  double * dphidz;
};

class BasisLTri : public Basis {

 public:

  // Constructor
  BasisLTri(int sngp = 1);

  // Destructor
  virtual ~BasisLTri();

  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);


 public:
  // Variables that are calculated at the gauss point
};

class BasisLQuad : public Basis {

 public:

  // Constructor
  BasisLQuad(int sngp = 2);

  // Destructor
  virtual ~BasisLQuad();

  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);



 public:
  // Variables that are calculated at the gauss point
  int sngp;
};

class BasisQTri : public Basis {

 public:

  // Constructor
  BasisQTri();

  // Destructor
  virtual ~BasisQTri();

  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);


 public:

};

class BasisQQuad : public Basis {

 public:

  // Constructor
  BasisQQuad();

  // Destructor
  virtual ~BasisQQuad();

  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);



 public:
  // Variables that are calculated at the gauss point
  int sngp;
};


class BasisLHex : public Basis {

 public:

  // Constructor
  BasisLHex();
  BasisLHex(int sngp);

  // Destructor
  virtual ~BasisLHex();

  // Calculates the values of u and x at the specified gauss point
  void getBasis(const int gp, const double *x, const double *y);

  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);



 public:
  // Variables that are calculated at the gauss point
  int sngp;
};

class BasisLTet : public Basis {

 public:

  // Constructor
  BasisLTet();
  BasisLTet(int sngp);

  // Destructor
  virtual ~BasisLTet();

  // Calculates the values of u and x at the specified gauss point
  void getBasis(const int gp, const double *x, const double *y);

  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);


 public:
  // Variables that are calculated at the gauss point
  int sngp;
};
#endif

