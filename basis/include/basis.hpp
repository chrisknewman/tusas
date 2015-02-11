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
  virtual void getBasis(int gp, double *x, double *y) = 0;
  virtual void getBasis(int gp, double *x, double *y, double *u) = 0;
  virtual void getBasis(int gp, double *x, double *y, double *u, double *uold)=0;
  virtual void getBasis(int gp, double *x, double *y, double *z, double *u, double *uold)=0;
  //template <typename T, size_t size>
    //void setN(int N, T (&abscissa)[size], T (&weight)[size]);
  void setN(int N, double *abscissa, double *weight);

 public:
  // Variables that are calculated at the gauss point
  int ngp;
  double *phi, *dphidxi, *dphideta, *dphidzta, *abscissa, *weight; 
  double xi, eta, zta, wt, jac;
  double dxidx, dxidy,dxidz, detadx, detady, detadz, dztadx, dztady, dztadz;
  double uu, xx, yy, zz, dudx, dudy, dudz;
  double uuold, duolddx, duolddy, duolddz;
  double * dphidx;
  double * dphidy;
};

class BasisLTri : public Basis {

 public:

  // Constructor
  BasisLTri(int sngp = 1);

  // Destructor
  virtual ~BasisLTri();

  // Calculates the values of u and x at the specified gauss point
  virtual void getBasis(int gp, double *x, double *y);
  void getBasis(int gp, double *x, double *y, double *u);
  virtual void getBasis(int gp, double *x, double *y, double *u, double *uold);
  virtual void getBasis(int gp, double *x, double *y, double *z, double *u, double *uold);

 public:
  // Variables that are calculated at the gauss point
};

class BasisLQuad : public Basis {

 public:

  // Constructor
  BasisLQuad(int sngp = 2);

  // Destructor
  virtual ~BasisLQuad();

  // Calculates the values of u and x at the specified gauss point
  virtual void getBasis(int gp, double *x, double *y);
  void getBasis(int gp, double *x, double *y, double *u);
  virtual void getBasis(int gp, double *x, double *y, double *u, double *uold);
  virtual void getBasis(int gp, double *x, double *y, double *z, double *u, double *uold);

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

  // Calculates the values of u and x at the specified gauss point
  virtual void getBasis(int gp, double *x, double *y, double *u);
  virtual void getBasis(int gp, double *x, double *y);
  virtual void getBasis(int gp, double *x, double *y, double *u, double *uold);
  virtual void getBasis(int gp, double *x, double *y, double *z, double *u, double *uold);

 public:

};

class BasisQQuad : public Basis {

 public:

  // Constructor
  BasisQQuad();

  // Destructor
  virtual ~BasisQQuad();

  // Calculates the values of u and x at the specified gauss point
  virtual void getBasis(int gp, double *x, double *y);
  virtual void getBasis(int gp, double *x, double *y, double *u);
  virtual void getBasis(int gp, double *x, double *y, double *u, double *uold);
  virtual void getBasis(int gp, double *x, double *y, double *z, double *u, double *uold);

 public:
  // Variables that are calculated at the gauss point
  int sngp;
};


// samet adds 3D stuff.
class BasisLHex : public Basis {

 public:

  // Constructor
  BasisLHex();
  BasisLHex(int sngp);

  // Destructor
  virtual ~BasisLHex();

  // Calculates the values of u and x at the specified gauss point
  virtual void getBasis(int gp, double *x, double *y);
  virtual void getBasis(int gp, double *x, double *y, double *z);
  virtual void getBasis(int gp, double *x, double *y, double *z, double *u);
  virtual void getBasis(int gp, double *x, double *y, double *z, double *u, double *uold);

 public:
  // Variables that are calculated at the gauss point
  int sngp;
};
#endif

