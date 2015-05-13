#ifndef PARAMNAMES_H
#define PARAMNAMES_H

#include <string>

//using namespace std;

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
std::string const TusasmethodDocString = "method (string): heat; phaseheat (default); multi";
std::string const TusasnoxrelresNameString = "noxrelres";
std::string const TusasnoxrelresDocString = "nox relative residual tolerance (double): default 1e-6";
std::string const TusasnoxmaxiterNameString = "noxmaxiter";
std::string const TusasnoxmaxiterDocString = "nox max number iterations (int): default 200";
std::string const TusasoutputfreqNameString = "outputfreq";
std::string const TusasoutputfreqDocString = "output frequency (int): default 1";
#endif
