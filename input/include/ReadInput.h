//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef READ_INPUT_H
#define READ_INPUT_H
	
#include "Teuchos_ParameterList.hpp"

void readParametersFromFile(    int argc, char *argv[], Teuchos::ParameterList &paramList ,int mypid);

#endif
