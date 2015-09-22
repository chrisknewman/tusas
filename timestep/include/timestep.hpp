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
