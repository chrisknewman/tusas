//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef TIMESTEP_HPP
#define TIMESTEP_HPP 

/// Base class for timestepping methods.
template<class Scalar>
class timestep 
{

public:
  /// Constructor
  timestep():start_time(0.0),start_step(0),cur_step(0){};
  /// Destructor
  virtual ~timestep(){};
  /// Initialize
  /** Initial conditions. Write initial conditions to output exodusII file. */
  virtual void initialize() = 0;
  /// Advance one timestep.
  virtual double advance() = 0;
  /// Finalize
  /** Write final timestep to output exodusII file. Cleanup. */ 
  virtual void finalize() = 0;
  /// Write solution to exodusII file.
  virtual void write_exodus() = 0;
  /// Return the current number of timesteps taken.
  virtual int64_t get_cur_step(){return cur_step;};
  /// Return the timestep index for restart.
  virtual int64_t get_start_step(){return start_step;};
  /// Return the timestep for restart.
  virtual double get_start_time(){return start_time;};
  
  protected:
  //Teuchos::ParameterList paramList_;
  /// Start timestep for restart.
  double start_time;
  /// Timestep index for restart.
  int64_t start_step;
  /// Count of total number of timesteps taken.
  int64_t cur_step;
  
};

#endif
