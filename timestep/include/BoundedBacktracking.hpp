//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
/////////////////////////////////////////////////////////////////////////////

#ifndef BOUNDEDBACKTRACKING_HPP
#define BOUNDEDBACKTRACKING_HPP

#include <cmath>
#include "NOX_LineSearch_Generic.H"
#include "NOX_LineSearch_UserDefinedFactory.H"
#include "NOX_GlobalData.H"
#include "NOX_Utils.H"

#include "NOX_Abstract_Vector.H"
#include "NOX_Thyra_Vector.H"
#include "Thyra_TpetraThyraWrappers.hpp"
#include "Teuchos_RCP.hpp"
#include "VariableBounds.hpp"

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

  VariableBounds bounds_;
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
 
    const int numeqs = params->get("Num Equations",-99);

    Teuchos::Array<int> eqnIds =
      params->get<Teuchos::Array<int>>("Bounded Equation IDs");
    Teuchos::Array<double> lowervals =
      params->get<Teuchos::Array<double>>("Lower Bounds");
    Teuchos::Array<double> uppervals =
      params->get<Teuchos::Array<double>>("Upper Bounds");

    bounds_ = VariableBounds(numeqs,
                             std::vector<int>(eqnIds.begin(), eqnIds.end()),
                             std::vector<double>(lowervals.begin(), lowervals.end()),
                             std::vector<double>(uppervals.begin(), uppervals.end()),
                             "BoundedBacktracking");

    minStep_ = params->get("Minimum Step", 1.0e-12);
    stepReduction_ = params->get("Step Reduction Factor", 0.5);
    maxIters_ = params->get("Max Iterations", 20);
    checkDescent_ = params->get("Check Descent", true);


    if (utils->isPrintType(NOX::Utils::Details)) {
      utils->out() << "BoundedBacktracking:\n" 
                   << "  bounded equations = "<<bounds_.size() <<"\n"
		   << "  minStep_ = "<<minStep_ <<"\n"
		   << "  stepReduction_ = "<<stepReduction_ <<"\n"
		   << "  maxIters_ = "<<maxIters_ <<"\n"
		   << "  checkDescent_ = "<<checkDescent_ <<"\n";
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
          // Armijo condition with alpha = 1e-4 (false for NaN)
          if (newF < oldF * (1.0 - 1.0e-4 * step)) {
	    if (utils->isPrintType(NOX::Utils::InnerIteration))
	      {
		utils->out() << NOX::Utils::fill(72) << "\n\n";
	      }
            return true;  // Accept step
          }
        } else {
          // Accept based on bounds alone, as long as the residual is finite
          newGrp.computeF();  // Still need to compute residual
          if (std::isfinite(newGrp.getNormF())) {
	    if (utils->isPrintType(NOX::Utils::InnerIteration))
	      {
		utils->out() << NOX::Utils::fill(72) << "\n\n";
	      }
            return true;
          }
        }
      }
      
      // Reduce step and try again
      step *= stepReduction_;
      if (step < minStep_) {
        if (utils->isPrintType(NOX::Utils::Warning))
          utils->out() << "BoundedBacktracking: step size below minimum ("
                       << utils->sciformat(minStep_) << ")" << std::endl;
        break;
      }
    }

    // Failed to find an acceptable step. Restore the previous solution and
    // return step = 0 so NOX reports a line search failure rather than
    // accepting an out-of-bounds "recovery step".
    if (utils->isPrintType(NOX::Utils::Warning))
      utils->out() << "BoundedBacktracking: failed to find an acceptable step"
                   << std::endl;
    // Copying the group brings along the already computed F, avoiding a
    // residual evaluation.
    step = 0.0;
    newGrp = oldGrp;
    return false;
  }
  
private:
  bool checkBounds(const NOX::Abstract::Vector& x) {
    Teuchos::RCP<const Thyra::VectorBase<double> > xx =
      dynamic_cast<const NOX::Thyra::Vector&>(x).getThyraRCPVector();
    Teuchos::RCP<const vector_type> x_vec = ConverterT::getConstTpetraVector(xx);
    return bounds_.check(*x_vec, *(x_vec->getMap()->getComm()),
                         utils->isPrintType(NOX::Utils::Details));
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

#endif // BOUNDEDBACKTRACKING_HPP
