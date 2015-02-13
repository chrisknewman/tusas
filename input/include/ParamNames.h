#ifndef PARAMNAMES_H
#define PARAMNAMES_H

#include <string>

//using namespace std;

std::string const TusasdtNameString = "dt";
std::string const TusasdtDocString = "timestep size";
std::string const TusasntNameString = "nt";
std::string const TusasntDocString = "number of timesteps";
std::string const TusasmeshNameString = "meshfile";
std::string const TusasmeshDocString = "mesh file name";
std::string const TusastestNameString = "testcase";
std::string const TusastestDocString = "test case ";
std::string const TusaspreconNameString = "preconditioner";
std::string const TusaspreconDocString = "preconditioner on off";
std::string const TusasthetaNameString = "theta";
std::string const TusasthetaDocString = "theta: 0 ee; 1 ie; .5 cn";
std::string const TusasmethodNameString = "method";
std::string const TusasmethodDocString = "method: heat; phaseheat";
#endif
