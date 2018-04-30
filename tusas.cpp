//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Los Alamos National Security, LLC.  This file is part of the
//  Tusas code (LA-CC-17-001) and is subject to the revised BSD license terms
//  in the LICENSE file found in the top-level directory of this distribution.
//
//////////////////////////////////////////////////////////////////////////////


// NOX Objects
#include "NOX.H"
#include "NOX_Thyra.H"

// Trilinos Objects
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Map.h"
#include "Epetra_LinearProblem.h"

#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_FancyOStream.hpp"

#include "Stratimikos_DefaultLinearSolverBuilder.hpp"
#include "Thyra_LinearOpWithSolveFactoryHelpers.hpp"
#include "Thyra_SpmdVectorBase.hpp"

#include "ModelEvaluatorHEAT.hpp"
#include "ModelEvaluatorPHASE_HEAT.hpp"
#include "ModelEvaluatorNEMESIS.hpp"

#include "Mesh.h"

#include "ParamNames.h"
#include "ReadInput.h"
#include "tusas.h"

#if 0
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>

extern char **environ;
#endif

using namespace std;

int decomp(const int mypid, const int numproc, const std::string& infile, std::string& outfile, const bool restart, const bool skipdecomp, const bool writedecomp, const Epetra_Comm * comm);
int do_sys_call(const char* command, char * const arg[] = NULL );
int join(const int mypid, const int numproc);
void print_disclaimer(const int mypid);
void print_copyright(const int mypid);

int main(int argc, char *argv[])
{
  Teuchos::TimeMonitor::zeroOutTimers();
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  // Create a communicator for Epetra objects
#ifdef HAVE_MPI
  Epetra_MpiComm Comm( MPI_COMM_WORLD );
#else
  Epetra_SerialComm Comm;
#endif

  int mypid = Comm.MyPID();
  int numproc = Comm.NumProc();
  
  print_disclaimer(mypid);
  print_copyright(mypid);

  Teuchos::ParameterList paramList;

  readParametersFromFile(argc, argv, paramList, mypid );

  Mesh * in_mesh = new Mesh(mypid,numproc,false);
  if(1 == numproc ){
    in_mesh->read_exodus((paramList.get<std::string> (TusasmeshNameString) ).c_str());
  }
  else {
    if( paramList.get<std::string> (TusasmethodNameString)  != "nemesis") {
      std::cout<<"More than 1 proc only implemented for nemesis class now."<<"\n";
      exit(0);
    }
    std::string pfile;
    Comm.Barrier();
    decomp(mypid, numproc, paramList.get<std::string> (TusasmeshNameString), pfile, paramList.get<bool> (TusasrestartNameString), paramList.get<bool> (TusasskipdecompNameString), paramList.get<bool> (TusaswritedecompNameString),&Comm);
    Comm.Barrier();
    
    in_mesh->read_exodus(pfile.c_str());
    //exit(0);
  }
  in_mesh->set_global_file_name(paramList.get<std::string> (TusasmeshNameString) );

  double dt = paramList.get<double> (TusasdtNameString);
  int numSteps = paramList.get<int> (TusasntNameString);

  timestep<double> * model;
  if( paramList.get<std::string> (TusasmethodNameString)  == "phaseheat") {
    model = new ModelEvaluatorPHASE_HEAT<double>(Teuchos::rcp(&Comm,false),in_mesh,paramList);
  }
  else if ( paramList.get<std::string> (TusasmethodNameString)  == "heat") {
    model = new ModelEvaluatorHEAT<double>(Teuchos::rcp(&Comm,false),in_mesh,paramList);
  }
  else if( paramList.get<std::string> (TusasmethodNameString)  == "nemesis") {
    model = new ModelEvaluatorNEMESIS<double>(Teuchos::rcp(&Comm,false),in_mesh,paramList);
  }
  else {
    std::cout<<"Invalid method."<<"\n"<<"\n";
    exit(0);
  }

  model->initialize();

  //cn these values are not updated correctly during a restart
  double curTime = model->get_start_time();
  //cn these values are not updated correctly during a restart
  int elapsedSteps = model->get_start_step();
  //cn these values are not updated correctly during a restart
  double endTime = curTime + ((double)numSteps-elapsedSteps)*dt;

  while ( ( curTime <= endTime ) && ( elapsedSteps < numSteps ) ) {
    model->advance();
    curTime += dt;
    elapsedSteps++;
    if(0 == mypid){
      cout<< endl << "Time step " <<elapsedSteps <<" of "<<numSteps
	  << "  ( "<<(float)elapsedSteps/(float)numSteps*100. <<" % )   t = "
	  <<curTime<<"  t final = "<<endTime<< endl<<endl<<endl;
    }
    if(0 == elapsedSteps%(paramList.get<int> (TusasoutputfreqNameString)) &&
       elapsedSteps != numSteps){
      if(0 == mypid) std::cout<<"Writing exodus file : timestep :"<<elapsedSteps<<"\n";
      
      model->write_exodus();
    }
  }

  model->finalize();
  
  Comm.Barrier();

  if(1 != numproc ) join(mypid, numproc);

  Teuchos::TimeMonitor::summarize();

  delete model;
  delete in_mesh;
  return 0;
}

int decomp(const int mypid, 
	   const int numproc, 
	   const std::string& infile, 
	   std::string& outfile, 
	   const bool restart, 
	   const bool skipdecomp, 
	   const bool writedecomp, 
	   const Epetra_Comm * comm){

  std::string decompPath="decomp/";
  std::string nemStr = "tusas_nemesis";

  if( 0 == mypid && !restart && !skipdecomp){
    std::cout<<"Entering decomp: PID "<<mypid<<" NumProcs "<<numproc<<"\n"<<"\n";

    std::ofstream decompfile;
    if( writedecomp ){
      std::cout<<"writedecomp started."<<"\n"<<"\n";
      std::string decompFile="./decompscript";
      decompfile.open(decompFile.c_str());
      decompfile
	<<"#!/bin/bash"<<"\n";
    }
    
    std::string rmdirStr = "rm";//+" -r "+decompPath;//+";mkdir "+decompPath;
    char * rmdirArg[] = {(char*)"rm",(char*)"-rf",const_cast<char*>((decompPath).c_str()),(char*)NULL};
    if( writedecomp ){
      decompfile
	<<rmdirArg[0]<<" "<<rmdirArg[1]<<" "<<rmdirArg[2]<<"\n";
    } 
    else {
      //if(-1 == system(comStr.c_str()) ){
      if(0 != do_sys_call(rmdirStr.c_str(), rmdirArg) ){
	std::cout<<"Error removing directory: "<<decompPath<<"\n";
	exit(0);
      }
    }

    std::string comStr = "mkdir";//+decompPath;
    char * comArg[] = {(char*)"mkdir",const_cast<char*>((decompPath).c_str()),(char*)NULL};
    if( writedecomp ){
      decompfile
	<<comArg[0]<<" "<<comArg[1]<<"\n";
    }
    else {
      if(-1 == do_sys_call(comStr.c_str(), comArg) ){
	std::cout<<"Error creating directory: "<<decompPath<<"\n";
	exit(0);
      }
    }
    std::cout<<"  Creating decomp dir: "<<comStr<<" "<<comArg[1]<<"\n";

    for( int i = 0; i < numproc; i++){
      std::string numStr = std::to_string(i+1);
      std::string mkdirStr = "mkdir";//+decompPath+numStr;
      char * mkdirArg[] = {(char*)"mkdir",const_cast<char*>((decompPath+numStr).c_str()),(char*)NULL};
      if( writedecomp ){
	decompfile
	  <<mkdirArg[0]<<" "<<mkdirArg[1]<<"\n";
      }
      else {
	//if(-1 == system(mkdirStr.c_str()) ){
	if(-1 == do_sys_call(mkdirStr.c_str(), mkdirArg) ){
	  std::cout<<"Error creating directory: "<<numStr<<"\n";
	  exit(0);
	}
      }
      std::cout<<"  Creating decomp dirs: "<<mkdirStr<<" "<<mkdirArg[1]<<"\n";
    }

    std::string trilinosPath=TRILINOS_DIR;
    std::string nemFile =decompPath+nemStr+".nemI";
    std::string sliceStr = trilinosPath+"/bin/nem_slice";//+" -e -m mesh="+std::to_string(numproc)+" -l inertial -o "+nemFile+" "+infile;

    if( writedecomp ){
      std::string sliceFile="./input-ldbl";
      std::ofstream slicefile;
      slicefile.open(sliceFile.c_str());
      slicefile 
	<<"OUTPUT NEMESISI FILE = "<<nemFile<<"\n" 
	<<"GRAPH TYPE			= ELEMENTAL"<<"\n" 
	<<"DECOMPOSITION METHOD		= INERTIAL"<<"\n" 
	<<"MACHINE DESCRIPTION = MESH="<<std::to_string(numproc)<<"\n";
      slicefile.close();
      char * sliceArg[] = {(char*)"nem_slice",(char*)"-a",const_cast<char*>((sliceFile).c_str()),const_cast<char*>((infile).c_str()),(char*)NULL};
    
      decompfile
	<<sliceStr<<" "<<sliceArg[1]<<" "<<sliceArg[2]<<" "<<sliceArg[3]<<"\n";
    }
    else {
      std::string sliceFile=decompPath+"input-ldbl";
      std::ofstream slicefile;
      slicefile.open(sliceFile.c_str());
      slicefile 
	<<"OUTPUT NEMESISI FILE = "<<nemFile<<"\n"
	<<"GRAPH TYPE			= ELEMENTAL"<<"\n"
	<<"DECOMPOSITION METHOD		= INERTIAL"<<"\n" 
	<<"MACHINE DESCRIPTION = MESH="<<std::to_string(numproc)<<"\n";
      slicefile.close();
      char * sliceArg[] = {(char*)"nem_slice",(char*)"-a",const_cast<char*>((sliceFile).c_str()),const_cast<char*>((infile).c_str()),(char*)NULL};
      std::cout<<"  Running nemslice command: "<<sliceStr<<" "<<sliceArg[1]<<" "<<sliceArg[2]<<" "<<sliceArg[3]<<"\n";

	//if(-1 == system(sliceStr.c_str()) ){
      if(-1 == do_sys_call(sliceStr.c_str(),sliceArg) ){
	std::cout<<"Error running nemslice: "<<sliceStr<<"\n";
	exit(0);
      }
    }

    if( writedecomp ){
      std::string spreadFile="./nem_spread.inp";
      std::ofstream spreadfile;
      spreadfile.open(spreadFile.c_str());
      spreadfile 
	<<"Input FEM file		= "<<infile<<"\n" 
	<<"LB file         	= "<<nemFile<<"\n" 
	<<"Restart Time list	= off"<<"\n"  
	<<"Parallel Disk Info	= number="<<std::to_string(numproc)<<"\n" 
	<<"Parallel file location	= root=./"<<decompPath<<", subdir=."<<"\n";
      spreadfile.close();
      std::string spreadStr = trilinosPath+"/bin/nem_spread";//+spreadFile;
      char * spreadArg[] = {(char*)"nem_spread",(char*)"nem_spread.inp",(char*)NULL};

      decompfile
	<<spreadStr<<" "<<spreadArg[1]<<"\n";
    }
    else {
      std::string spreadFile=decompPath+"nem_spread.inp";
      std::ofstream spreadfile;
      spreadfile.open(spreadFile.c_str());
      spreadfile 
	<<"Input FEM file		= "<<infile<<"\n" 
	<<"LB file         	= "<<nemFile<<"\n" 
	<<"Restart Time list	= off"<<"\n" 
	<<"Parallel Disk Info	= number="<<std::to_string(numproc)<<"\n" 
	<<"Parallel file location	= root=./"<<decompPath<<", subdir=."<<"\n" ;
      spreadfile.close();
      std::string spreadStr = trilinosPath+"/bin/nem_spread";//+spreadFile;
      char * spreadArg[] = {(char*)"nem_spread",const_cast<char*>(spreadFile.c_str()),(char*)NULL};
      std::cout<<"  Running nemspread command: "<<spreadStr <<" "<<spreadArg[1]<<"\n";
      //if(-1 == system(spreadStr.c_str()) ){
      if(-1 == do_sys_call(spreadStr.c_str(), spreadArg) ){
	std::cout<<"Error running nemspread: "<<spreadStr<<"\n";
	exit(0);
      }
    }

    if( writedecomp ){
      decompfile.close();
    }
  }//if( 0 == mypid && !restart)
  //if(  0 == mypid  && writedecomp ){
  comm->Barrier();
  if( writedecomp ){
#ifdef HAVE_MPI
  (void) MPI_Finalize ();
#endif
    if( 0 == mypid ){
      std::cout<<"writedecomp completed."<<"\n"<<"\n";
    }
    exit(0);
  }
  std::string mypidstring;

  if( numproc < 10 ){
    mypidstring = std::to_string(mypid);
  }
  if( numproc > 9 && numproc < 100 ){
    if ( mypid < 10 ){
      mypidstring = std::to_string(0)+std::to_string(mypid);
    }
    else{
      mypidstring = std::to_string(mypid);
    }
  }//if
  if( numproc > 99 && numproc < 1000 ){
    if ( mypid < 10 ){
      mypidstring = std::to_string(0)+std::to_string(0)+std::to_string(mypid);
    }
    else if ( mypid > 9 && mypid < 100 ){
      mypidstring = std::to_string(0)+std::to_string(mypid);
    }
    else{
      mypidstring = std::to_string(mypid);
    }
  }//if
  if( numproc > 999 && numproc < 10000 ){
    if ( mypid < 10 ){
      mypidstring = std::to_string(0)+std::to_string(0)+std::to_string(0)+std::to_string(mypid);
    }
    else if ( mypid > 9 && mypid < 100 ){
      mypidstring = std::to_string(0)+std::to_string(0)+std::to_string(mypid);
    }
    else if ( mypid > 99 && mypid < 1000 ){
      mypidstring = std::to_string(0)+std::to_string(mypid);
    }
    else{
      mypidstring = std::to_string(mypid);
    }
  }//if

  outfile=decompPath+std::to_string(mypid+1)+"/"+nemStr+".par."+std::to_string(numproc)+"."+mypidstring;

  if( 0 == mypid  && !restart){
    std::cout<<"\n"<<"Exiting decomp"<<"\n"<<"\n";
  }
  return 0;
}
int join(const int mypid, const int numproc)
{
  if( 0 == mypid ){
    std::cout<<"Entering join: PID "<<mypid<<" NumProcs "<<numproc<<"\n"<<"\n";
    std::string decompPath="./decomp/";
    std::string trilinosPath=TRILINOS_DIR;
    std::string comStr = trilinosPath+"bin/epu";// -auto -add_processor_id "+decompPath+"results.e."+std::to_string(numproc)+".000";
    char * comArg[] = {(char*)"epu",(char*)"-auto", (char*)"-add_processor_id",const_cast<char*>((decompPath+"results.e."+std::to_string(numproc)+".000").c_str()),(char*)NULL};
 
    std::cout<<"Running epu command: "<<comStr<<" "<<comArg[1]<<" "<<comArg[2]<<" "<<comArg[3]<<"\n";
    //if(-1 == system(comStr.c_str()) ){
    if(-1 == do_sys_call(comStr.c_str(), comArg) ){
      std::cout<<"Error running epu: "<<comStr<<"\n";
      exit(0);
    }
  }
  return 0;
}
int do_sys_call(const char* command, char * const arg[] )
{
  int status = -99;
  int * s = &status;
  int pid = vfork();
  //std::cout<<"pid = "<<pid<<"\n";
  int err = 0;
  if( 0 == pid ) {
    err = execvp(command,arg);
    //err = execve(command,arg,environ);
    _exit(0);
  }
  else {
    wait(s);
  }
  //std::cout<<"err = "<<err<<"\n";
  return err;
}
void print_disclaimer(const int mypid)
{
  if(0 == mypid){
    std::cout<<"\n"
	     <<"Tusas Version 1.0 is registered with Los Alamos National Laboratory (LANL) as LA-CC-17-001."
	     <<"\n"<<"\n";
  }
}
void print_copyright(const int mypid)
{
  if(0 == mypid){
    std::cout<<"\n"
	     <<"Copyright (c) 2016, Los Alamos National Security, LLC"<<"\n"
	     <<"\n"
	     <<"All rights reserved."<<"\n"
	     <<"\n"
	     <<"This software was produced under U.S. Government contract DE-AC52-06NA25396"<<"\n"
	     <<"for Los Alamos National Laboratory (LANL), which is operated by Los Alamos"<<"\n"
	     <<"National Security, LLC for the U.S. Department of Energy. The U.S. Government"<<"\n"
	     <<"has rights to use, reproduce, and distribute this software.  NEITHER THE"<<"\n" 
	     <<"GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS"<<"\n" 
	     <<"OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software"<<"\n" 
	     <<"is modified to produce derivative works, such modified software should be"<<"\n" 
	     <<"clearly marked, so as not to confuse it with the version available from LANL."<<"\n"
	     <<"\n"
	     <<"Additionally, redistribution and use in source and binary forms, with or"<<"\n" 
	     <<"without modification, are permitted provided that the following conditions"<<"\n" 
	     <<"are met:"
	     <<"\n"
	     <<"1. Redistributions of source code must retain the above copyright notice,"<<"\n" 
	     <<"   this list of conditions and the following disclaimer."<<"\n"
	     <<"\n"
	     <<"2. Redistributions in binary form must reproduce the above copyright notice,"<<"\n" 
	     <<"   this list of conditions and the following disclaimer in the documentation"<<"\n" 
	     <<"   and/or other materials provided with the distribution."<<"\n"
	     <<"\n"
	     <<"3. Neither the name of Los Alamos National Security, LLC, Los Alamos National"<<"\n" 
	     <<"   Laboratory, LANL, the U.S. Government, nor the names of its contributors"<<"\n"
	     <<"   may be used to endorse or promote products derived from this software"<<"\n" 
	     <<"   without specific prior written permission."<<"\n"
	     <<"\n"
	     <<"THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND"<<"\n" 
	     <<"CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,"<<"\n" 
	     <<"BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS"<<"\n" 
	     <<"FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS"<<"\n"
	     <<"NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,"<<"\n" 
	     <<"INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT"<<"\n" 
	     <<"NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,"<<"\n" 
	     <<"DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY"<<"\n" 
	     <<"THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT"<<"\n" 
	     <<"(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF"<<"\n" 
	     <<"THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."<<"\n"
	     <<"\n"<<"\n";
  }
}
