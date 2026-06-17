//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
/////////////////////////////////////////////////////////////////////////////


#include "NOX_LineSearch_Generic.H"
#include "NOX_LineSearch_UserDefinedFactory.H"
#include "NOX_GlobalData.H"
#include "NOX_Utils.H"

#include "NOX_Abstract_Vector.H"
#include "NOX_Thyra_Vector.H"
#include "Thyra_TpetraThyraWrappers.hpp"
#include "Teuchos_RCP.hpp"

// Forward declarations
namespace NOX {
  class Utils;
}

class BoundedBacktracking : public NOX::LineSearch::Generic {
private:
  typedef Tpetra::Vector<>::global_ordinal_type global_ordinal_type;
  typedef Tpetra::Vector<>::local_ordinal_type local_ordinal_type;
  typedef Tpetra::Vector<>::node_type node_type;
  typedef Tpetra::Vector<>::scalar_type scalar_type;
  typedef Tpetra::Vector<scalar_type, local_ordinal_type,
			 global_ordinal_type, node_type> vector_type;

  typedef Thyra::TpetraOperatorVectorExtraction<scalar_type,int> ConverterT;

  int numeqs_;
  std::vector<double> lowerboundedVals_;
  std::vector<double> upperboundedVals_;
  std::vector<int> boundedEqnids_;
  double minStep_;
  double stepReduction_;
  int maxIters_;
  bool checkDescent_;  // Also check for function decrease?
  Teuchos::RCP<NOX::Utils> utils;
  
public:
  BoundedBacktracking(const Teuchos::RCP<NOX::GlobalData>& gd,
      const Teuchos::RCP<Teuchos::ParameterList>& params) 
  {
    utils = gd->getUtils();
 
    numeqs_ = params->get("Num Equations",-99);
 
    Teuchos::Array<int> eqnIds = 
      params->get<Teuchos::Array<int>>("Bounded Equation IDs");
    boundedEqnids_.assign(eqnIds.begin(), eqnIds.end());

    Teuchos::Array<double> lowervals = 
      params->get<Teuchos::Array<double>>("Lower Bounds");
    lowerboundedVals_.assign(lowervals.begin(), lowervals.end());

    Teuchos::Array<double> uppervals = 
      params->get<Teuchos::Array<double>>("Upper Bounds");
    upperboundedVals_.assign(uppervals.begin(), uppervals.end());

    if (lowerboundedVals_.size() != upperboundedVals_.size() || 
        lowerboundedVals_.size() != boundedEqnids_.size() ) {
      throw std::runtime_error(
          "BoundedBacktracking: Array sizes must match!");
    }

    minStep_ = params->get("Minimum Step", 1.0e-12);
    stepReduction_ = params->get("Step Reduction Factor", 0.5);
    maxIters_ = params->get("Max Iterations", 20);
    checkDescent_ = params->get("Check Descent", true);

    if (utils->isPrintType(NOX::Utils::Details)) {
      utils->out() << "BoundedBacktracking: " 
                   << boundedEqnids_.size() << " bounded equations\n";
    }
  }
  
  //reset does not seem to get called...NOX implementations seem to have constructor
  //call reset with all the initialization in reset
  bool reset(const Teuchos::RCP<NOX::GlobalData>& gd,
             const Teuchos::RCP<NOX::Abstract::Group>& grp,
             const Teuchos::RCP<NOX::StatusTest::Generic>& t) {
   
    return true;
  }
  
  bool compute(NOX::Abstract::Group& newGrp,
               double& step,
               const NOX::Abstract::Vector& dir,
               const NOX::Solver::Generic& s) {
    
    const NOX::Abstract::Group& oldGrp = s.getPreviousSolutionGroup();
    const NOX::Abstract::Vector& oldX = oldGrp.getX();
    double oldF = oldGrp.getNormF();
    
    step = 1.0;  // Start with full Newton step
    
    for (int iter = 0; iter < maxIters_; iter++) {
      // Compute new solution: x_new = x_old + step * dir
      newGrp.computeX(oldGrp, dir, step);
      const NOX::Abstract::Vector& newX = newGrp.getX();

      if (utils->isPrintType(NOX::Utils::InnerIteration))
	{
	  utils->out() << "\n" << NOX::Utils::fill(72) << "\n"
		       << "-- BoundedBacktracking Line Search -- \n";
	}
      
      // Check if bounded elements are within bounds
      bool boundsOK = checkBounds(newX);

      if (boundsOK) {
        // Optionally check for sufficient decrease
        if (checkDescent_) {
          newGrp.computeF();
          double newF = newGrp.getNormF();
	  if (utils->isPrintType(NOX::Utils::InnerIteration))
	    {
	      utils->out() << std::setw(3) << iter << ":";
	      utils->out() << " step = " << utils->sciformat(step);
	      utils->out() << " old f = " << utils->sciformat(oldF);
	      utils->out() << " new f = " << utils->sciformat(newF);
	      utils->out() << std::endl;
	    }
          // Armijo condition with alpha = 1e-4
          if (newF < oldF * (1.0 - 1.0e-4 * step)) {
	    if (utils->isPrintType(NOX::Utils::InnerIteration))
	      {
		utils->out() << NOX::Utils::fill(72) << "\n\n";
	      }
            return true;  // Accept step
          }
        } else {
          // Accept based on bounds alone
          newGrp.computeF();  // Still need to compute residual
	  if (utils->isPrintType(NOX::Utils::InnerIteration))
	    {
	      utils->out() << NOX::Utils::fill(72) << "\n\n";
	    }
          return true;
        }
      }
      
      // Reduce step and try again
      step *= stepReduction_;
      if (step < minStep_) {
        std::cout << "Warning: Step size below minimum in bounded backtracking"
                  << std::endl;
        return false;
      }
    }
    
    return false;  // Failed to find acceptable step
  }
  
private:
  bool checkBounds(const NOX::Abstract::Vector& x) {

    //std::cout<<"BoundedBacktracking::checkBounds"<<std::endl;

    const Thyra::VectorBase<double> * xx = 
      &(dynamic_cast<const NOX::Thyra::Vector&>(x).getThyraVector());

    Thyra::ConstDetachedSpmdVectorView<double> x_vec(xx->col(0));

    Teuchos::ArrayRCP<const scalar_type> vals = x_vec.values();

    const size_t localLength = vals.size();

    // Loop over ALL unknowns once
    for (size_t lid = 0; lid < localLength; lid++) {
      
      // Determine which equation this unknown belongs to
      int eqn = lid % numeqs_;
      
      // Check if this equation is bounded
      for (int k = 0; k < boundedEqnids_.size(); k++) {
        if (eqn == boundedEqnids_[k]) {
          // This equation is bounded - check it
          if (vals[lid] < lowerboundedVals_[k] || 
              vals[lid] > upperboundedVals_[k]) {
            
            if (utils->isPrintType(NOX::Utils::Details)) {
              int node = lid / numeqs_;
              utils->out() << "  Out of bounds at node " << node 
			   << ", equation " << eqn 
			   << ": value = " << vals[lid]
			   << ", bounds = [" << lowerboundedVals_[k] 
			   << ", " << upperboundedVals_[k] << "]\n";
            }
            
            return false;
          }
          break;  // Found the equation, stop searching
        }
      }
    }
    return true;
  }
};

class BoundedBacktrackingFactory : public NOX::LineSearch::UserDefinedFactory {
private:
  
public:
  BoundedBacktrackingFactory() {
    //std::cout<<"BoundedBacktrackingFactory::BoundedBacktrackingFactory"<<std::endl;
}
  
  Teuchos::RCP<NOX::LineSearch::Generic> 
  buildLineSearch(const Teuchos::RCP<NOX::GlobalData>& gd,
                  Teuchos::ParameterList& params) const {
    //std::cout<<"buildLineSearch"<<std::endl;
    return Teuchos::rcp(new BoundedBacktracking(gd,
        Teuchos::rcpFromRef(params)));
  }
};
