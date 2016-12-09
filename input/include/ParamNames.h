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

//using namespace std;

//parameters in the input file
std::string const TusasdtNameString = "dt";
std::string const TusasdtDocString = "timestep size (double): default .001";
std::string const TusasntNameString = "nt";
std::string const TusasntDocString = "number of timesteps (int): default 140";
std::string const TusasmeshNameString = "meshfile";
std::string const TusasmeshDocString = "mesh file name (string)";
std::string const TusastestNameString = "testcase";
std::string const TusastestDocString = "test case (string): cummins (default); pool; multi; furtado; karma ";
std::string const TusaspreconNameString = "preconditioner";
std::string const TusaspreconDocString = "preconditioner (bool): true (default); false";
std::string const TusasthetaNameString = "theta";
std::string const TusasthetaDocString = "theta (double): 0 ee; 1 ie (default); .5 cn";
std::string const TusasmethodNameString = "method";
std::string const TusasmethodDocString = "method (string): heat; phaseheat; nemesis (default)";
std::string const TusasnoxrelresNameString = "noxrelres";
std::string const TusasnoxrelresDocString = "nox relative residual tolerance (double): default 1e-6";
std::string const TusasnoxmaxiterNameString = "noxmaxiter";
std::string const TusasnoxmaxiterDocString = "nox max number iterations (int): default 200";
std::string const TusasoutputfreqNameString = "outputfreq";
std::string const TusasoutputfreqDocString = "output frequency (int): default 1e10";

std::string const TusasmlNameString = "ML";
std::string const TusasmlDocString = "over ride ML parameters";
std::string const TusaslsNameString = "Linear Solver";
std::string const TusaslsDocString = "over ride linear solver parameters";
std::string const TusasjfnkNameString = "JFNK Solver";
std::string const TusasjfnkDocString = "over ride JFNK solver; probably should not be changed";
std::string const TusasnlsNameString = "Nonlinear Solver";
std::string const TusasnlsDocString = "over ride nonlinear solver parameters";

std::string const TusasdeltafactorNameString = "deltafactor";
std::string const TusasdeltafactorDocString = "multiplicaton factor for delta (double): default 0.5";

std::string const TusaserrorestimatorNameString = "errorestimator";
std::string const TusaserrorestimatorDocString = "variables to estimate error for, {} corrsponds to none, {0,3} corresponds to 1 and 3 (string): default none";

//other parameters not in the input file
std::string const TusasrestartNameString = "restart";
std::string const TusasrestartDocString = "restart (bool): false (default); true";

#endif
