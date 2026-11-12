//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////


#ifndef RANDOM_DISTRIBUTION_H
#define RANDOM_DISTRIBUTION_H

#include "Mesh.h"

class random_distribution
{
public:
  random_distribution(Mesh *mesh,  ///< mesh object
		      const int ltpquadorder  ///< quadrature order
		      );

  ~random_distribution();

  void compute_random(const int nt);

  std::vector<std::vector<double> > get_gauss_vals() const {return gauss_val;}

  const double get_gauss_val(const int i, const int ig) const {return gauss_val[i][ig];}

  void compute_correlation() const;

  void print() const;

private:
  /// number of Gauss points in element
  int ngp;
  /// random values at Gauss points
  std::vector<std::vector<double> > gauss_val;
  /// number of elements on this proc
  int num_elem;

};

#endif
