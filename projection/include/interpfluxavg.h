//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef INTERPFLUXAVG_H
#define INTERPFLUXAVG_H

#include <vector>

// Epetra support
#include "Epetra_Comm.h"

//teuchos support
#include <Teuchos_RCP.hpp>

//#define TUSAS_INTERPFLUXAVG

class interpfluxavg
{
public:
  /// Constructor.
  interpfluxavg(const Teuchos::RCP<const Epetra_Comm>& comm, 
	     const std::string datafileString );

  /// Destructor.
  ~interpfluxavg();
  bool get_source_value(const double time, const int index, double &val);

private:

  const Teuchos::RCP<const Epetra_Comm>  comm_;
  const std::string datafileString_;

  const int stride_;
  int timeindex_;
  std::vector<double> data;

  void read_file();

};

#endif
