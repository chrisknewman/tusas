#include <string>

#include "ReadInput.h"
#include "ParamNames.h"


void readParametersFromFile(    int argc, char *argv[], Teuchos::ParameterList &paramList )
{
  //string   filename                = "meshes/tri24.e"    ;
  //string   filename                = "meshes/tri96.e"    ;
  //string   filename                = "meshes/tri384.e"    ;
  //string   filename                = "meshes/quad16.e"    ;
  //string   filename                = "meshes/quad64.e"    ;
  //string   filename                = "meshes/quad256.e"    ;
  //string   filename                = "meshes/quad1024.e"    ;
  //string   filename                = "meshes/quad4096.e"    ;
  //string   filename                = "meshes/dendquad300.e"    ;
  //string   filename                = "meshes/dendquad300_h.e"    ;
  //string   filename                = "meshes/dendquad300_q.e"    ;
  //string   filename                = "meshes/dendquad600.e"    ;


  paramList.set(TusasmeshNameString,"meshes/dendquad300_q.e",TusasmeshDocString);
  //paramList.set(TusasmeshNameString,"meshes/dendquad300_h.e",TusasmeshDocString);
  paramList.set(TusasdtNameString,(double).001,TusasdtDocString);
  paramList.set(TusasntNameString,(int)140,TusasntDocString);
  //std::cout<<paramList<<std::endl<<std::endl;
};
