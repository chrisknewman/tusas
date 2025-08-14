//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
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
  Teuchos::RCP<Teuchos::Time> ts_time_read = Teuchos::TimeMonitor::getNewTimer("Tusas: Total Read Input Time");
  Teuchos::TimeMonitor ReadTimer(*ts_time_read);
  //set defaults here
  paramList.set(TusasmeshNameString,"meshes/dendquad300_q.e",TusasmeshDocString);

  paramList.set(TusasdtNameString,(double).001,TusasdtDocString);
  //paramList.set(TusasdtNameString,(double).00001,TusasdtDocString);

  paramList.set(TusasntNameString,(int)0,TusasntDocString);
  paramList.set(Tusasnt64NameString,(long long)-99LL,Tusasnt64DocString);
  //paramList.set(TusasntNameString,(int)1,TusasntDocString);
  //paramList.set(TusasntNameString,(int)14000,TusasntDocString);
  //std::cout<<paramList<<"\n"<<std::endl;
  paramList.set(TusastestNameString,"cummins",TusastestDocString);

  paramList.set(TusasthetaNameString,(double)1.,TusasthetaDocString);
  //paramList.set(TusasthetaNameString,(double)0.,TusasthetaDocString);

  paramList.set(TusaspreconNameString,(bool)false,TusaspreconDocString);

  paramList.set(TusasleftScalingNameString,(bool)false,TusasleftScalingDocString);

  paramList.set(TusasmethodNameString,"nemesis",TusasmethodDocString);

  paramList.set(TusasnoxrelresNameString,(double)1.e-6,TusasnoxrelresDocString);

  paramList.set(TusasnoxmaxiterNameString,(int)200,TusasnoxmaxiterDocString);

  paramList.set(TusasnoxacceptNameString,(bool)false,TusasnoxacceptDocString);

  paramList.set(TusasnoxforcestepNameString,(bool)false,TusasnoxforcestepDocString);

  paramList.set(TusasoutputfreqNameString,(int)(1e9),TusasoutputfreqDocString);

  //paramList.set(TusasrestartstepNameString,(int)0,TusasrestartstepDocString);

  paramList.set(TusasrestartNameString,(bool)false,TusasrestartDocString);

  paramList.set(TusaserrorestimatorNameString,"{}",TusaserrorestimatorDocString);

  paramList.set(TusasltpquadordNameString,(int)2,TusasltpquadordDocString);

  paramList.set(TusasqtpquadordNameString,(int)3,TusasqtpquadordDocString);

  paramList.set(TusasltriquadordNameString,(int)1,TusasltriquadordDocString);

  paramList.set(TusasqtriquadordNameString,(int)3,TusasqtriquadordDocString);

  paramList.set(TusasltetquadordNameString,(int)4,TusasltetquadordDocString);

  paramList.set(TusasqtetquadordNameString,(int)5,TusasqtetquadordDocString);

  paramList.set(TusasexaConstitNameString,(bool)false,TusasexaConstitDocString);

  paramList.set(TusasestimateTimestepNameString,(bool)false,TusasestimateTimestepDocString);

  paramList.set(TusasinitialSolveNameString,(bool)false,TusasinitialSolveDocString);

  //paramList.set(TusasdecompmethodNameString,"INERTIAL",TusasdecompmethodDocString);

  paramList.set(Tusasusenemesis64bitNameString,(bool)false,Tusasusenemesis64bitDocString);

  paramList.set(TusasadaptiveTimestepNameString,(bool)false,TusasadaptiveTimestepDocString);

  paramList.set(TusasrandomDistributionNameString,(bool)false,TusasrandomDistributionDocString);

  paramList.set(TusaspredmaxiterNameString,(int)20,TusaspredmaxiterDocString);//not sure where to put this, the residual tol is set in atslist

  paramList.set(TusasprintNormsNameString,(bool)false,TusasprintNormsDocString);

  Teuchos::ParameterList *ATSList;
  ATSList = &paramList.sublist(TusasatslistNameString,(bool)false);
  ATSList->set(TusasatsmaxiterNameString,1);
  ATSList->set(TusasatsatolNameString,1.e-2);
  ATSList->set(TusasatsrtolNameString,0.0);
  ATSList->set(TusasatssfNameString,.9);
  ATSList->set(TusasatsrmaxNameString,2.0);
  ATSList->set(TusasatsrminNameString,.5);
  ATSList->set(TusasatsepsNameString,1.e-10);
  ATSList->set(TusasatsmaxdtNameString,1.e-1);
  ATSList->set(TusasatstypeNameString,"predictor corrector",TusasatstypeDocString);
  ATSList->set(TusaspredmaxiterNameString,(int)20,TusaspredmaxiterDocString);
  //ATSList->set(TusaspredrelresNameString,paramList.get<double>(TusasnoxrelresNameString),TusaspredrelresDocString);//not sure where to put this, the residual tol is set in atslist

  //ML parameters for ML and MueLu
  Teuchos::ParameterList *MLList;
  MLList = &paramList.sublist ( TusasmlNameString, false );

  //cn right now this does not make sense as the xml file, namely the method string has not been read--need to fix
  if(paramList.get<std::string> (TusasmethodNameString)  == "nemesis"){
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
  }

  //Linear solver parameters
  Teuchos::ParameterList *LSList;
  LSList = &paramList.sublist(TusaslsNameString,false);

  LSList->set("Linear Solver Type", "Belos");
  //lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Pseudo Block GMRES").set("Num Blocks",1);
  //lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Pseudo Block GMRES").set("Maximum Restarts",200);
  //lsparams->sublist("Linear Solver Types").sublist("Belos").sublist("Solver Types").sublist("Psuedo Block GMRES").set("Output Frequency",1);

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
  std::string outputPathName = "decomp/";
  clp.setOption("output-path",&outputPathName,"Output path (must exist)");

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

  if((true == paramList.get<bool>(TusasadaptiveTimestepNameString))
     &&(0. < paramList.get<double>(TusasthetaNameString)) ){
    paramList.set(TusasestimateTimestepNameString,(bool)true);
  } else {
    paramList.set(TusasadaptiveTimestepNameString,(bool)false);
    //paramList.set(TusasestimateTimestepNameString,(bool)false);
    //should print something here
//     if( 0 == mypid )
//       std::cout <<"   Adaptive timestep set to false"
// 		<<std::endl<<std::endl;
  }
  if((true == paramList.get<bool>(TusasestimateTimestepNameString))
     && ATSList->get<std::string> (TusasatstypeNameString) == "second derivative"){
    paramList.set(TusasinitialSolveNameString,(bool)true);    
  }
//   if((0 != paramList.get<int>(TusasntNameString))%(paramList.get<int>(TusasoutputfreqNameString)))
//     paramList.set(TusasoutputfreqNameString,(int)1,TusasoutputfreqDocString);

  paramList.set(TusasrestartNameString,restart,TusasrestartDocString);
  paramList.set(TusasskipdecompNameString,skipdecomp,TusasskipdecompDocString);
  paramList.set(TusaswritedecompNameString,writedecomp,TusaswritedecompDocString);

  paramList.set(TusasoutputpathNameString,outputPathName,TusasoutputpathDocString);

  if( 0 ==(LSList->get<std::string>("Linear Solver Type")).compare("AztecOO") ) {
    if( 0 == mypid ){
      std::cout<<"Linear Solver Type: AztecOO no longer supported, please use Belos."<<"\n"<<"\n"<<"\n";
      LSList->sublist("Linear Solver Types").remove("AztecOO",true);
    }
  }

  //also want to remove ml if methodname is Tpetra
  bool remove_ml = !( paramList.get<bool> (TusaspreconNameString) )
    || (paramList.get<std::string> (TusasmethodNameString)  == "tpetra");
  if( remove_ml ){
    paramList.remove(TusasmlNameString);
    //exit(0);
  }

  if( 0 == mypid ){
    paramList.print(std::cout, 2, true, true );
    std::cout<<"\n"<<"Initial parameter list completed."<<"\n"<<"\n"<<"\n";
  }

  //need to somehow fix this such that TusasnoxrelresNameString is default but can be overridden
  if( !ATSList->isParameter(TusaspredrelresNameString) )
    ATSList->set(TusaspredrelresNameString,paramList.get<double>(TusasnoxrelresNameString));

//   Teuchos::ParameterList *problemList;
//   problemList = &paramList.sublist ( "ProblemParams", false );
//   problemList->set("Echo ProblemParams",(bool)false);
  //exit(0);

};
