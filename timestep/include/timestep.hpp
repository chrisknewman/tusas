//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef TIMESTEP_HPP
#define TIMESTEP_HPP 


template<class Scalar>
class timestep 
{

public:
  timestep():start_time(0.0),start_step(0){};
  ~timestep(){};
  virtual void initialize() = 0;
  virtual void advance() = 0;
  virtual void finalize() = 0;
  virtual void write_exodus() = 0;

  virtual int get_start_step(){return start_step;};
  virtual double get_start_time(){return start_time;};
  
  protected:
  //Teuchos::ParameterList paramList_;
  double start_time;
  int start_step;
  
};

#endif
