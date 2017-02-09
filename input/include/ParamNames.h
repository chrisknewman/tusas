//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#ifndef PARAMNAMES_H
#define PARAMNAMES_H

#include <string>

//parameters in the input file
/// Timestep size.
std::string const TusasdtNameString = "dt";
/// Timestep size.
std::string const TusasdtDocString = "timestep size (double): default .001";
/// Number of timesteps.
std::string const TusasntNameString = "nt";
/// Number of timesteps.
std::string const TusasntDocString = "number of timesteps (int): default 140";
/// ExodusII mesh file.
std::string const TusasmeshNameString = "meshfile";
/// ExodusII mesh file.
std::string const TusasmeshDocString = "mesh file name (string)";
/// Testcase.
std::string const TusastestNameString = "testcase";
/// Testcase.
std::string const TusastestDocString = "test case (string): cummins (default); pool; multi; furtado; karma ";
/// Preconditioner
std::string const TusaspreconNameString = "preconditioner";
/// Preconditioner
std::string const TusaspreconDocString = "preconditioner (bool): true (default); false";
/// Theta for timestep method.
std::string const TusasthetaNameString = "theta";
/// Theta for timestep method.
std::string const TusasthetaDocString = "theta (double): 0 ee; 1 ie (default); .5 cn";
/// Method.
std::string const TusasmethodNameString = "method";
/// Method.
std::string const TusasmethodDocString = "method (string): heat; phaseheat; nemesis (default)";
/// Nox relative residual tolerance.
std::string const TusasnoxrelresNameString = "noxrelres";
/// Nox relative residual tolerance.
std::string const TusasnoxrelresDocString = "nox relative residual tolerance (double): default 1e-6";
/// Nox maximum number of iterations.
std::string const TusasnoxmaxiterNameString = "noxmaxiter";
/// Nox maximum number of iterations.
std::string const TusasnoxmaxiterDocString = "nox max number iterations (int): default 200";
/// Write exodusII output every int steps.
std::string const TusasoutputfreqNameString = "outputfreq";
/// Write exodusII output every int steps.
std::string const TusasoutputfreqDocString = "output frequency (int): default 1e10";
/// ML parameters list.
std::string const TusasmlNameString = "ML";
/// ML parameters list.
std::string const TusasmlDocString = "over ride ML parameters";
/// Linear solver parameters list.
std::string const TusaslsNameString = "Linear Solver";
/// Linear solver parameters list.
std::string const TusaslsDocString = "over ride linear solver parameters";
/// JFNK parameters list.
std::string const TusasjfnkNameString = "JFNK Solver";
/// JFNK parameters list.
std::string const TusasjfnkDocString = "over ride JFNK solver; probably should not be changed";
/// Nonlinear solver parameters list.
std::string const TusasnlsNameString = "Nonlinear Solver";
/// Nonlinear solver parameters list.
std::string const TusasnlsDocString = "over ride nonlinear solver parameters";
/// Delta factor for phasefield.
std::string const TusasdeltafactorNameString = "deltafactor";
/// Delta factor for phasefield.
std::string const TusasdeltafactorDocString = "multiplicaton factor for delta (double): default 0.5";
/// Create error estimator for this variable.
std::string const TusaserrorestimatorNameString = "errorestimator";
/// Create error estimator for this variable.
std::string const TusaserrorestimatorDocString = "variables to estimate error for, {} corrsponds to none, {0,3} corresponds to 1 and 3 (string): default none";

//other parameters not in the input file
/// Restart.
std::string const TusasrestartNameString = "restart";
/// Restart.
std::string const TusasrestartDocString = "restart (bool): false (default); true";

#endif
