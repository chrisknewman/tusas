//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef INTERPFLUX_H
#define INTERPFLUX_H

#include <vector>

// Epetra support
#include "Epetra_Comm.h"

//teuchos support
#include <Teuchos_RCP.hpp>

//#define TUSAS_INTERPFLUX

class interpflux
{
public:
  /// Constructor.
  interpflux(const Teuchos::RCP<const Epetra_Comm>& comm, 
	     const std::string timefileString );

  /// Destructor.
  ~interpflux();
  bool interp_time(const double time);

  double theta_;
  int timeindex_;

private:

  const Teuchos::RCP<const Epetra_Comm>  comm_;
  const std::string timefileString_;

  std::vector<double> data;

  void read_file();

};

#endif
