//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#include "interpfluxavg.h"

#include <iostream>

interpfluxavg::interpfluxavg(const Teuchos::RCP<const Epetra_Comm>& comm, 
	     const std::string datafileString) :
  comm_(comm),
  datafileString_(datafileString),
  stride_(7)
{
  timeindex_ = 0;
  read_file();
}


interpfluxavg::~interpfluxavg()
{
}

void interpfluxavg::read_file()
{
  //cn right now there is some kind of conflict with std::getline and nvcc
  //   I am #if ing it out for now, since this is used with truchas coupling
  exit(0);
#if 0
  int mypid = comm_->MyPID();

  if(mypid == 0) {
    std::ifstream ifile(datafileString_, std::ios::in);
      
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
  
bool interpfluxavg::get_source_value(const double time, const int index, double &val)
{
  val = -99999999999999.;

  bool found = false;

  int mypid = comm_->MyPID();

  if(mypid == 0) {
    //std::cout<<data.size()<<std::endl;
    for( unsigned int i = timeindex_; i < data.size(); i=i+stride_){
      //std::cout<<i<<" "<<data[i]<<std::endl;
      if(time < data[i]) {
	timeindex_ = i - stride_;
	double time_n = data[timeindex_];
	double theta = (time - time_n)/(data[i] - time_n);

	val = (1.-theta) * data[timeindex_ + 1 + index] + theta * data[i + 1 + index];

// 	std::cout<<i<<" "<<timeindex_<<" "<<theta<<" "<<data[timeindex_ + 1 + index]<<
// 	  " "<<data[i + 1 + index]<<" "<<val<<std::endl;
	found = true;
	break;
      }//if
    }//i


  }//if

  comm_->Broadcast(&val, (int)1, (int)0 );	
  comm_->Barrier();
  //std::cout<<val<<std::endl;

  return found;
}
