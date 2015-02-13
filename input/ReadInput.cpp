#include <string>

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
//	#include "Teuchos_StandardParameterEntryValidators.hpp"

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


  //set defaults here
  //paramList.set(TusasmeshNameString,"meshes/hex64_3d.e",TusasmeshDocString);
  //paramList.set(TusasmeshNameString,"meshes/dendquad300_q3d.e",TusasmeshDocString);
  paramList.set(TusasmeshNameString,"meshes/dendquad300_q.e",TusasmeshDocString);

  paramList.set(TusasdtNameString,(double).001,TusasdtDocString);
  //paramList.set(TusasdtNameString,(double).00001,TusasdtDocString);

  paramList.set(TusasntNameString,(int)140,TusasntDocString);
  //paramList.set(TusasntNameString,(int)1,TusasntDocString);
  //paramList.set(TusasntNameString,(int)14000,TusasntDocString);
  //std::cout<<paramList<<std::endl<<std::endl;
  paramList.set(TusastestNameString,"cummins",TusastestDocString);

  paramList.set(TusasthetaNameString,(double)1.,TusasthetaDocString);
  //paramList.set(TusasthetaNameString,(double)0.,TusasthetaDocString);

  paramList.set(TusaspreconNameString,(bool)true,TusaspreconDocString);
  //paramList.set(TusaspreconNameString,(bool)false,TusaspreconDocString);

  paramList.set(TusasmethodNameString,"phaseheat",TusasmethodDocString);

  //read/overwrite here

  // read parameters from xml file
  std::string inputFileName = "";
  Teuchos::CommandLineProcessor  clp(false); // Don't throw exceptions
  clp.setOption( "input-file", &inputFileName, "The XML file to read into a parameter list" );
  clp.setDocString( "Document string for this program. Right now, not much going on here." );
  Teuchos::CommandLineProcessor::EParseCommandLineReturn parse_return = clp.parse(argc,argv);

  if( parse_return != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL ) {
    std::cout << "Default parameter values:\n" << std::endl;
    paramList.print(std::cout, 2, true, true );
    exit(parse_return);
  }

  if(inputFileName.length()) {
    std::cout << "\nReading a parameter list from the XML file \""<<inputFileName<<"\" ...\n";
    using Teuchos::inOutArg;
    Teuchos::updateParametersFromXmlFile(inputFileName,inOutArg(paramList));
  } else {
    // no file message here about defaults
    std::cout << "No input file specified, or not read successfully. Default values will be used.\n" << std::endl;
    std::cout << "Default values:\n" << std::endl;
  }
  paramList.print(std::cout, 2, true, true );
  //exit(0);

};
