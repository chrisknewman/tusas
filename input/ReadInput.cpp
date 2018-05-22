//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////



#include <string>

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include <Teuchos_RCP.hpp>
#include "Teuchos_TimeMonitor.hpp"
//	#include "Teuchos_StandardParameterEntryValidators.hpp"
#include <ml_MultiLevelPreconditioner.h>

#include "ReadInput.h"
#include "ParamNames.h"


void readParametersFromFile(    int argc, char *argv[], Teuchos::ParameterList &paramList, int mypid )
{
  Teuchos::RCP<Teuchos::Time> ts_time_read = Teuchos::TimeMonitor::getNewTimer("Total Read Input Time");
  Teuchos::TimeMonitor ReadTimer(*ts_time_read);
  //set defaults here
  paramList.set(TusasmeshNameString,"meshes/dendquad300_q.e",TusasmeshDocString);

  paramList.set(TusasdtNameString,(double).001,TusasdtDocString);
  //paramList.set(TusasdtNameString,(double).00001,TusasdtDocString);

  paramList.set(TusasntNameString,(int)140,TusasntDocString);
  //paramList.set(TusasntNameString,(int)1,TusasntDocString);
  //paramList.set(TusasntNameString,(int)14000,TusasntDocString);
  //std::cout<<paramList<<"\n"<<std::endl;
  paramList.set(TusastestNameString,"cummins",TusastestDocString);

  paramList.set(TusasthetaNameString,(double)1.,TusasthetaDocString);
  //paramList.set(TusasthetaNameString,(double)0.,TusasthetaDocString);

  //paramList.set(TusaspreconNameString,(bool)true,TusaspreconDocString);
  paramList.set(TusaspreconNameString,(bool)false,TusaspreconDocString);

  //paramList.set(TusasmethodNameString,"phaseheat",TusasmethodDocString);
  paramList.set(TusasmethodNameString,"nemesis",TusasmethodDocString);

  paramList.set(TusasnoxrelresNameString,(double)1.e-6,TusasnoxrelresDocString);

  paramList.set(TusasnoxmaxiterNameString,(int)200,TusasnoxmaxiterDocString);

  paramList.set(TusasoutputfreqNameString,(int)(1e10),TusasoutputfreqDocString);

  //paramList.set(TusasrestartstepNameString,(int)0,TusasrestartstepDocString);

  paramList.set(TusasrestartNameString,(bool)false,TusasrestartDocString);

  paramList.set(TusaserrorestimatorNameString,"{}",TusaserrorestimatorDocString);

  paramList.set(TusasltpquadordNameString,(int)2,TusasltpquadordDocString);

  paramList.set(TusasqtpquadordNameString,(int)3,TusasqtpquadordDocString);

  paramList.set(TusasltriquadordNameString,(int)1,TusasltriquadordDocString);

  paramList.set(TusasqtriquadordNameString,(int)3,TusasqtriquadordDocString);

  paramList.set(TusasexaConstitNameString,(bool)false,TusasexaConstitDocString);

  //ML parameters
  Teuchos::ParameterList *MLList;
  MLList = &paramList.sublist ( TusasmlNameString, false );
  ML_Epetra::SetDefaults("SA",paramList.sublist (TusasmlNameString ));
    //MLList.set("coarse: max size",(int)128);
  //    MLList->set("cycle applications",(int)2);
//     MLList.set("prec type","full-MGV");
//     MLList.set("smoother: type","Chebyshev");
    MLList->set("smoother: type","Jacobi");
    MLList->set("smoother: sweeps",(int)2); 
//     MLList.set("smoother: damping factor", 1.0);

//    MLList.set("coarse: type","Chebyshev");
//     MLList.set("coarse: type","Jacobi"); 
//     MLList.set("coarse: sweeps",4);
    
//     MLList.set("coarse: damping factor", 1.0);
    
//     MLList.set("ML output",10);

//    MLList->set("PDE equations",1);

  //Linear solver parameters
  Teuchos::ParameterList *LSList;
  LSList = &paramList.sublist(TusaslsNameString,false);
#if 0
  LSList->set("Linear Solver Type", "Belos");
  //lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Pseudo Block GMRES").set("Num Blocks",1);
  //lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Pseudo Block GMRES").set("Maximum Restarts",200);
  //lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Psuedo Block GMRES").set("Output Frequency",1);
#else
  LSList->set("Linear Solver Type", "AztecOO");
  LSList->sublist("Linear Solver Types").sublist("AztecOO").sublist("Forward Solve").sublist("AztecOO Settings").set("Output Frequency",1);
  //lsparams->sublist("Linear Solver Types").sublist("AztecOO").sublist("Forward Solve").sublist("AztecOO Settings").sublist("AztecOO Preconditioner", "None");
#endif
  LSList->set("Preconditioner Type", "None");

  //jfnk params
  Teuchos::ParameterList *JFNKList;
  JFNKList = &paramList.sublist(TusasjfnkNameString,false);
  JFNKList->set("Difference Type","Forward");
  JFNKList->set("lambda",1.0e-4);

  //nonlinear solver parameters
  Teuchos::ParameterList *NLSList;
  NLSList = &paramList.sublist(TusasnlsNameString,false);
  NLSList->set("Nonlinear Solver", "Line Search Based");
  Teuchos::ParameterList& searchParams = NLSList->sublist("Line Search");
  //searchParams.set("Method", "Full Step");
  //searchParams.set("Method", "Interval Halving");
  //searchParams.set("Method", "Polynomial");
  //searchParams.set("Method", "Backtrack");
  //searchParams.set("Method", "NonlinearCG");
  //searchParams.set("Method", "Quadratic");
  //searchParams.set("Method", "More'-Thuente");

  //Teuchos::ParameterList& btParams = NLSList->sublist("Backtrack");
  //btParams.set("Default Step",1.0);
  //btParams.set("Max Iters",20);
  //btParams.set("Minimum Step",1e-6);
  //btParams.set("Recovery Step",1e-3);

  NLSList->sublist("Direction").sublist("Newton").set("Forcing Term Method", "Type 2");
  NLSList->sublist("Direction").sublist("Newton").set("Forcing Term Initial Tolerance", 1.0e-1);
  NLSList->sublist("Direction").sublist("Newton").set("Forcing Term Maximum Tolerance", 1.0e-2);
  NLSList->sublist("Direction").sublist("Newton").set("Forcing Term Minimum Tolerance", 1.0e-5);//1.0e-6
  NLSList->sublist("Direction").sublist("Newton").set("Forcing Term Alpha", 1.5);
  NLSList->sublist("Direction").sublist("Newton").set("Forcing Term Gamma", .9);

  //read/overwrite here

  // read parameters from xml file
  std::string inputFileName = "tusas.xml";
  Teuchos::CommandLineProcessor  clp(false); // Don't throw exceptions
  clp.setOption( "input-file", &inputFileName, "The XML file to read into a parameter list" );
  bool restart = false;
  clp.setOption( "restart","norestart", &restart );
  bool skipdecomp = false;
  clp.setOption( "skipdecomp","noskipdecomp", &skipdecomp );
  bool writedecomp = false;
  clp.setOption( "writedecomp","nowritedecomp", &writedecomp );

  clp.setDocString( "Document string for this program. Right now, not much going on here." );
  Teuchos::CommandLineProcessor::EParseCommandLineReturn parse_return = clp.parse(argc,argv);

  if( parse_return != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL ) {
    if( 0 == mypid ){
      std::cout << "Default parameter values:\n" << "\n";
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
    std::cout << "No input file specified, or not read successfully. Default values will be used.\n" << "\n";
    std::cout << "Default values:\n" << "\n";
  }

//   if((0 != paramList.get<int>(TusasntNameString))%(paramList.get<int>(TusasoutputfreqNameString)))
//     paramList.set(TusasoutputfreqNameString,(int)1,TusasoutputfreqDocString);

  paramList.set(TusasrestartNameString,restart,TusasrestartDocString);
  paramList.set(TusasskipdecompNameString,skipdecomp,TusasskipdecompDocString);
  paramList.set(TusaswritedecompNameString,writedecomp,TusaswritedecompDocString);

  if( 0 !=(LSList->get<std::string>("Linear Solver Type")).compare("AztecOO") ) {
    //LSList->sublist("Linear Solver Types").sublist("AztecOO").sublist("Forward Solve").sublist("AztecOO Settings").remove("Output Frequency");
    LSList->sublist("Linear Solver Types").remove("AztecOO");
  }
 
  if( 0 == mypid ){
    paramList.print(std::cout, 2, true, true );
    std::cout<<"\n"<<"Initial parameter list completed."<<"\n"<<"\n"<<"\n";
  }

  //exit(0);

};
