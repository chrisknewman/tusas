#include <string>

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
//	#include "Teuchos_StandardParameterEntryValidators.hpp"
#include <ml_MultiLevelPreconditioner.h>

#include "ReadInput.h"
#include "ParamNames.h"


void readParametersFromFile(    int argc, char *argv[], Teuchos::ParameterList &paramList, int mypid )
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

  //paramList.set(TusasmethodNameString,"phaseheat",TusasmethodDocString);
  paramList.set(TusasmethodNameString,"nemesis",TusasmethodDocString);

  paramList.set(TusasnoxrelresNameString,(double)1.e-6,TusasnoxrelresDocString);

  paramList.set(TusasnoxmaxiterNameString,(int)200,TusasnoxmaxiterDocString);

  paramList.set(TusasoutputfreqNameString,(int)(1e10),TusasoutputfreqDocString);

  //paramList.set(TusasrestartstepNameString,(int)0,TusasrestartstepDocString);

  paramList.set(TusasrestartNameString,(bool)false,TusasrestartDocString);

  paramList.set(TusasdeltafactorNameString,(double) .5,TusasdtDocString);

  Teuchos::ParameterList MLList;
  MLList = paramList.sublist ( TusasmlNameString, false );
  ML_Epetra::SetDefaults("SA",paramList.sublist (TusasmlNameString ));
    //MLList.set("coarse: max size",(int)128);
    MLList.set("cycle applications",(int)2);
//     MLList.set("prec type","full-MGV");
//     MLList.set("smoother: type","Chebyshev");
    MLList.set("smoother: type","Jacobi");
    MLList.set("smoother: sweeps",(int)2); 
//     MLList.set("smoother: damping factor", 1.0);

//    MLList.set("coarse: type","Chebyshev");
//     MLList.set("coarse: type","Jacobi"); 
//     MLList.set("coarse: sweeps",4);
    
//     MLList.set("coarse: damping factor", 1.0);
    
//     MLList.set("ML output",10);

    MLList.set("PDE equations",2);


  //read/overwrite here

  // read parameters from xml file
  std::string inputFileName = "";
  Teuchos::CommandLineProcessor  clp(false); // Don't throw exceptions
  clp.setOption( "input-file", &inputFileName, "The XML file to read into a parameter list" );
  bool restart = false;
  clp.setOption( "restart","norestart", &restart );
  clp.setDocString( "Document string for this program. Right now, not much going on here." );
  Teuchos::CommandLineProcessor::EParseCommandLineReturn parse_return = clp.parse(argc,argv);

  if( parse_return != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL ) {
    if( 0 == mypid ){
      std::cout << "Default parameter values:\n" << std::endl;
      paramList.print(std::cout, 2, true, true );
    }
    exit(parse_return);
  }

  if(inputFileName.length()) {
    if( 0 == mypid )
      std::cout << "\nReading a parameter list from the XML file \""<<inputFileName<<"\" ...\n";
    using Teuchos::inOutArg;
    Teuchos::updateParametersFromXmlFile(inputFileName,inOutArg(paramList));
  } else {
    // no file message here about defaults
    std::cout << "No input file specified, or not read successfully. Default values will be used.\n" << std::endl;
    std::cout << "Default values:\n" << std::endl;
  }

//   if((0 != paramList.get<int>(TusasntNameString))%(paramList.get<int>(TusasoutputfreqNameString)))
//     paramList.set(TusasoutputfreqNameString,(int)1,TusasoutputfreqDocString);

  paramList.set(TusasrestartNameString,restart,TusasrestartDocString);

  if( 0 == mypid ){
    paramList.print(std::cout, 2, true, true );
    std::cout<<std::endl<<"Initial parameter list completed."<<std::endl<<std::endl<<std::endl;
  }

  //exit(0);

};
