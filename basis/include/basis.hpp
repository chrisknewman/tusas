//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



// 1D Linear basis function for finite element method

#ifndef _NOX_EXAMPLE_EPETRA_LINEAR_BASIS_H
#define _NOX_EXAMPLE_EPETRA_LINEAR_BASIS_H

class Basis {

 public:

  // Constructor
  Basis(){};

  // Destructor
  virtual ~Basis(){}

  // Calculates the values of u and x at the specified gauss point
  virtual void getBasis(const int gp, const double *x, const double *y){getBasis(gp, x, y, NULL, NULL, NULL, NULL);};
  virtual void getBasis(const int gp, const double *x, const double *y, const double *z){getBasis(gp, x, y, z, NULL, NULL, NULL);};

  virtual void getBasis(const int gp, const double *x, const double *y, const double *z, const double *u){getBasis(gp, x, y, z, u, NULL, NULL);};
  virtual void getBasis(const int gp, const double *x, const double *y, const double *z, const double *u, const double *uold){getBasis(gp, x, y, z, u, uold, NULL);};
  virtual void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold){exit(0);};

  void setN(int N, double *abscissa, double *weight);    

  virtual Basis* clone() const {exit(0);};
  virtual char type() {exit(0);};

 public:
  // Variables that are calculated at the gauss point
  int ngp;
  double *phi, *dphidxi, *dphideta, *dphidzta;
  double dxidx, dxidy,dxidz, detadx, detady, detadz, dztadx, dztady, dztadz;
  double wt, jac;
  double uu, xx, yy, zz, dudx, dudy, dudz;
  double uuold, duolddx, duolddy, duolddz;
  double uuoldold, duoldolddx, duoldolddy, duoldolddz;
  double * dphidx;
  double * dphidy;
  double * dphidz;
protected:
  double *abscissa, *weight;
  double *xi, *eta, *zta, *nwt;
};

class BasisLTri : public Basis {

 public:

  // Constructor
  BasisLTri(int sngp = 1);

  // Destructor
  virtual ~BasisLTri();

  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);

  BasisLTri* clone() const{ return new BasisLTri(*this); }
  char type() { return 1; }

 public:
  // Variables that are calculated at the gauss point
};

class BasisLQuad : public Basis {

 public:

  // Constructor
  BasisLQuad(int sngp = 2);

  // Destructor
  ~BasisLQuad();

  BasisLQuad* clone() const{ return new BasisLQuad(*this); }
  char type() { return 1; }


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
  ~BasisQTri();

  BasisQTri* clone() const{ return new BasisQTri(*this); }
  char type() { return 1; }

  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);


 public:

};

class BasisQQuad : public Basis {

 public:

  // Constructor
  BasisQQuad();

  // Destructor
  ~BasisQQuad();    

  BasisQQuad* clone() const{ return new BasisQQuad(*this); }
  char type() { return 1; }

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

class BasisLTet : public Basis {

 public:

  // Constructor
  BasisLTet();
//   BasisLTet(int sngp);

  // Destructor
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


class BasisLBar : public Basis {

 public:

  // Constructor
  BasisLBar(int sngp = 2);

  // Destructor
  ~BasisLBar();

  BasisLBar* clone() const{ return new BasisLBar(*this); }
  char type() { return 1; }


  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);



 public:
  // Variables that are calculated at the gauss point
  int sngp;
};



class BasisQBar : public Basis {

 public:

  // Constructor
  BasisQBar(int sngp = 3);

  // Destructor
  ~BasisQBar();

  BasisQBar* clone() const{ return new BasisQBar(*this); }
  char type() { return 1; }


  void getBasis( const int gp,  const double *x,  const double *y,  const double *z,  const double *u,  const double *uold,  const double *uoldold);



 public:
  // Variables that are calculated at the gauss point
  int sngp;
};

#endif

