#ifndef TIMESTEP_HPP
#define TIMESTEP_HPP 

class timestep 
{

public:
  timestep(){};
  ~timestep(){};
  virtual void initialize(){};
  virtual void advance(){};
  virtual void finalize(){};
  
};

#endif
