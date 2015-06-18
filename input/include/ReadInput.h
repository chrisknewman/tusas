#ifndef READ_INPUT_H
#define READ_INPUT_H
	
#include "Teuchos_ParameterList.hpp"

void readParametersFromFile(    int argc, char *argv[], Teuchos::ParameterList &paramList ,int mypid);

#endif
