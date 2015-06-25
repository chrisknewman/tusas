#ifndef TIMESTEP_HPP
#define TIMESTEP_HPP 


template<class Scalar>
class timestep 
{

public:
  timestep(){};
  ~timestep(){};
  virtual void initialize() = 0;
  virtual void advance() = 0;
  virtual void finalize() = 0;
  virtual void write_exodus() = 0;
  
  //private:
  //Teuchos::ParameterList paramList_;
  
};

#endif
