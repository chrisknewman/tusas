//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#include "interpflux.h"

#include <iostream>

interpflux::interpflux(const Teuchos::RCP<const Epetra_Comm>& comm, 
	     const std::string timefileString) :
  comm_(comm),
  timefileString_(timefileString)
{
  timeindex_ = 0;
  theta_ = 0.;
  read_file();
  //exit(0);
}


interpflux::~interpflux()
{
}

void interpflux::read_file()
{
  //cn right now there is some kind of conflict with std::getline and nvcc
  //   I am #if ing it out for now, since this is used with truchas coupling
  exit(0);
#if 0
  int mypid = comm_->MyPID();

  if(mypid == 0) {
    std::ifstream ifile(timefileString_, std::ios::in);
      
    //cn this ignores comments
    std::string line;
    while (std::getline(ifile, line))
      {
	if (line[0] != '#' )
	  {
	    std::istringstream iss(line);
	    float num; // The number in the line
	    
	    //while the iss is a number 
	    while ((iss >> num))
	      {
		//look at the number
		data.push_back(num);
	      }
	  }
      }
  }    
#endif
  return;
}
  
bool interpflux::interp_time(const double time)
{
  bool found = false;

  int mypid = comm_->MyPID();

  if(mypid == 0) {
    //std::cout<<data.size()<<std::endl;
    for( unsigned int i = timeindex_; i < data.size(); ++i){
      //std::cout<<i<<" "<<data[i]<<std::endl;
      if(time < data[i]) {
	timeindex_ = i-1;
	double time_n = data[timeindex_];
	theta_ = (time - time_n)/(data[i] - time_n);

	found = true;
	break;
      }//if
    }//i


  }//if

  comm_->Broadcast(&timeindex_, (int)1, (int)0 );
  comm_->Broadcast(&theta_, (int)1, (int)0 );	
  comm_->Barrier();
  //std::cout<<val<<std::endl;

  return found;
}
