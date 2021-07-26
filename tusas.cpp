//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Triad National Security, LLC.  This file is part of the
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

#include <Kokkos_Core.hpp>

#include "ModelEvaluatorNEMESIS.hpp"
#include "ModelEvaluatorTPETRA.hpp"

#include "Mesh.h"

#include "ParamNames.h"
#include "ReadInput.h"
#include "tusas.h"

#include <sys/wait.h>
#if 0
#include <unistd.h>
#include <spawn.h>

extern char **environ;
#endif

using namespace std;

int decomp(const int mypid, const int numproc, const std::string& infile, std::string& outfile, const bool restart, const bool skipdecomp, const bool writedecomp, const bool usenemesis64, const Epetra_Comm * comm);
int do_sys_call(const char* command, char * const arg[] = NULL );
int join(const int mypid, const int numproc, const bool skipdecomp);
void print_disclaimer(const int mypid);
void print_copyright(const int mypid);
void write_timers();
std::string getmypidstring(const int mypid, const int numproc);

int main(int argc, char *argv[])
{
  Teuchos::TimeMonitor::zeroOutTimers();
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
 //  auto comm_ = Teuchos::DefaultComm<int>::getComm();
//   comm_->describe(*(Teuchos::VerboseObjectBase::getDefaultOStream()),Teuchos::EVerbosityLevel::VERB_EXTREME );
  
  Kokkos::initialize(argc, argv);
#ifdef TUSAS_KOKKOS_PRINT_CONFIG
  Kokkos::print_configuration( std::cout , false );
#endif
  Teuchos::RCP<Teuchos::Time> ts_time_total = Teuchos::TimeMonitor::getNewTimer("Tusas: Total Run Time");
  Teuchos::ParameterList paramList;

  // Create a communicator for Epetra objects
#ifdef HAVE_MPI
  Epetra_MpiComm Comm( MPI_COMM_WORLD );
#else
  Epetra_SerialComm Comm;
#endif
  
  int mypid;
  int numproc;
    
  Mesh * in_mesh;
  std::string pfile;
  int dval = 0;
  {
    mypid = Comm.MyPID();
    numproc = Comm.NumProc();

    Teuchos::TimeMonitor TotalTimer(*ts_time_total);
    print_disclaimer(mypid);
    print_copyright(mypid);
    
    readParametersFromFile(argc, argv, paramList, mypid );

    if( 1 != numproc && (paramList.get<std::string> (TusasmethodNameString)  != "nemesis"
			 && paramList.get<std::string> (TusasmethodNameString)  != "tpetra")) {
//     if( 1 != numproc && (paramList.get<std::string> (TusasmethodNameString)  != "nemesis")) {
      std::cout<<"More than 1 proc only implemented for nemesis class now."<<"\n";
      return EXIT_FAILURE;
    }
    if( 1 == numproc && paramList.get<bool> (TusaswritedecompNameString) ) {
      std::cout<<"More than 1 proc required for writedecomp option."<<"\n";
      return EXIT_FAILURE;
    }

    if(1 == numproc ){
      pfile = paramList.get<std::string> (TusasmeshNameString);
    }
    else {
      Comm.Barrier();
      dval = decomp(mypid, 
		    numproc, 
		    paramList.get<std::string> (TusasmeshNameString), 
		    pfile, 
		    paramList.get<bool> (TusasrestartNameString), 
		    paramList.get<bool> (TusasskipdecompNameString), 
		    paramList.get<bool> (TusaswritedecompNameString),
		    paramList.get<bool> (Tusasusenemesis64bitNameString),
		    &Comm);      
      Comm.Barrier();
    }
  }
  if(0 != dval) {
    write_timers();
    Kokkos::finalize();
    return 0;
  }
 
  {
    Teuchos::TimeMonitor TotalTimer(*ts_time_total); 
    in_mesh = new Mesh(mypid,numproc,false);
    in_mesh->read_exodus(pfile.c_str());
    in_mesh->set_global_file_name(paramList.get<std::string> (TusasmeshNameString) );
    
    double dt = paramList.get<double> (TusasdtNameString);
    int numSteps = paramList.get<int> (TusasntNameString);
    
    timestep<double> * model;

    if( paramList.get<std::string> (TusasmethodNameString)  == "tpetra") {
      model = new ModelEvaluatorTPETRA<double>(Teuchos::rcp(&Comm,false),in_mesh,paramList);
    }
#ifdef TUSAS_HAVE_CUDA
#else
    else if( paramList.get<std::string> (TusasmethodNameString)  == "nemesis") {
      model = new ModelEvaluatorNEMESIS<double>(Teuchos::rcp(&Comm,false),in_mesh,paramList);
    }
#endif
    else {
      std::cout<<"Invalid method."<<"\n"<<"\n";
      return EXIT_FAILURE;
    }
    
    model->initialize();
    
    double curTime = model->get_start_time();
    int elapsedSteps = model->get_start_step();
    double endTime = curTime + ((double)numSteps-elapsedSteps)*dt;
    
    while ( ( curTime <= endTime ) && ( elapsedSteps < numSteps ) ) {
      double dtnew = 0.;

      dtnew = model->advance();
      
      //this will be dtold
      curTime += dtnew;
      elapsedSteps++;

      if(0 == mypid){
	cout<< endl << "Time step " <<elapsedSteps <<" of "<<numSteps
	  //<< "  ( "<<(float)elapsedSteps/(float)numSteps*100. <<" % )   t = "
	    << "  ( "<<(float)curTime/(float)endTime*100. <<" % )   t = "
	    <<curTime<<"  t final = "<<endTime<< endl<<endl<<endl;
      }
      if(0 == elapsedSteps%(paramList.get<int> (TusasoutputfreqNameString)) &&
	 elapsedSteps != numSteps){
	if(0 == mypid) std::cout<<"Writing exodus file : timestep :"<<elapsedSteps<<"\n";
	
	model->write_exodus();
      }
    }//while
    
    model->finalize();
    
    Comm.Barrier();
    
    if(1 != numproc ) {
      if(paramList.get<std::string> (TusasmethodNameString)  == "nemesis") join(mypid, numproc, 
		    paramList.get<bool> (TusasskipdecompNameString));
      if(paramList.get<std::string> (TusasmethodNameString)  == "tpetra") join(mypid, numproc, 
		    paramList.get<bool> (TusasskipdecompNameString));
    }

    delete model;
    delete in_mesh;
  }
  write_timers();
  Kokkos::finalize();
#ifdef TUSAS_HAVE_SUMMIT_MPI
  MPI_Finalize();  //enable this to force an abort on summit (summit ibm-mpi hangs here...)
#endif
  return 0;
}

int decomp(const int mypid, 
	   const int numproc, 
	   const std::string& infile, 
	   std::string& outfile, 
	   const bool restart, 
	   const bool skipdecomp, 
	   const bool writedecomp, 
	   const bool usenemesis64,
	   const Epetra_Comm * comm){

  //return 1 for writedecomp; return 0 otherwise
  //probably need to clean this up to return -1 on error, rather than exit(0)


  std::string decompPath="decomp/";
  std::string nemStr = "tusas_nemesis";
  Teuchos::RCP<Teuchos::Time> ts_time_decomp = Teuchos::TimeMonitor::getNewTimer("Tusas: Total Decomp Time");
  Teuchos::TimeMonitor DecompTimer(*ts_time_decomp);
  if( 0 == mypid ){
    if( !restart && !skipdecomp){
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
	std::cout<<"  Creating decomp dir: "<<comStr<<" "<<comArg[1]<<"\n";
	if(-1 == do_sys_call(comStr.c_str(), comArg) ){
	  std::cout<<"Error creating directory: "<<decompPath<<"\n";
	  exit(0);
	}
      }
      
      for( int i = 0; i < numproc; i++){
	std::string numStr = std::to_string(i+1);
	std::string mkdirStr = "mkdir";//+decompPath+numStr;
	char * mkdirArg[] = {(char*)"mkdir",const_cast<char*>((decompPath+numStr).c_str()),(char*)NULL};
	if( writedecomp ){
	  decompfile
	    <<mkdirArg[0]<<" "<<mkdirArg[1]<<"\n";
	}
	else {
	  //std::cout<<"  Creating decomp dirs: "<<mkdirArg[1]<<"\n";
	  std::cout<<"  Creating decomp dirs: "<<decompPath+numStr<<"\n";
	  //if(-1 == system(mkdirStr.c_str()) ){
	  if(-1 == do_sys_call(mkdirStr.c_str(), mkdirArg) ){
	    std::cout<<"Error creating directory: "<<numStr<<"\n";
	    exit(0);
	  }
	}
      }//for
      
      std::string trilinosPath=TRILINOS_DIR;
      std::string nemFile =decompPath+nemStr+".nemI";
      std::string sliceStr = trilinosPath+"/bin/nem_slice";//+" -e -m mesh="+std::to_string(numproc)+" -l inertial -o "+nemFile+" "+infile;
      
      //hack for 64 bit nemesis
      std::string decompMethod;
      std::string use64Str;
      if( usenemesis64 ){
	decompMethod = "LINEAR";
	use64Str = "-64";
      }
      else {
	decompMethod = "INERTIAL";
	use64Str = "";
      };

      if( writedecomp ){
	std::string sliceFile="./input-ldbl";
	std::ofstream slicefile;
	slicefile.open(sliceFile.c_str());
	slicefile 
	  <<"OUTPUT NEMESISI FILE = "<<nemFile<<"\n" 
	  <<"GRAPH TYPE			= ELEMENTAL"<<"\n" 
	  <<"DECOMPOSITION METHOD		= "<<decompMethod<<"\n" 
	  <<"MACHINE DESCRIPTION = MESH="<<std::to_string(numproc)<<"\n";
	slicefile.close();
	char * sliceArg[] = {(char*)"nem_slice",const_cast<char*>((use64Str).c_str()),(char*)"-a",const_cast<char*>((sliceFile).c_str()),const_cast<char*>((infile).c_str()),(char*)NULL};
	
	decompfile
	  <<sliceStr<<" "<<sliceArg[1]<<" "<<sliceArg[2]<<" "<<sliceArg[3]<<" "<<sliceArg[4]<<"\n";
      }
      else {
	std::string sliceFile=decompPath+"input-ldbl";
	std::ofstream slicefile;
	slicefile.open(sliceFile.c_str());
	slicefile 
	  <<"OUTPUT NEMESISI FILE = "<<nemFile<<"\n"
	  <<"GRAPH TYPE			= ELEMENTAL"<<"\n"
	  <<"DECOMPOSITION METHOD		= "<<decompMethod<<"\n" 
	  <<"MACHINE DESCRIPTION = MESH="<<std::to_string(numproc)<<"\n";
	slicefile.close();
	char * sliceArg[] = {(char*)"nem_slice",const_cast<char*>((use64Str).c_str()),(char*)"-a",const_cast<char*>((sliceFile).c_str()),const_cast<char*>((infile).c_str()),(char*)NULL};
	std::cout<<"  Running nemslice command: "<<sliceStr<<" "<<sliceArg[1]<<" "<<sliceArg[2]<<" "<<sliceArg[3]<<" "<<sliceArg[4]<<"\n";
	
	//if(-1 == system(sliceStr.c_str()) ){
	if(-1 == do_sys_call(sliceStr.c_str(),sliceArg) ){
	  std::cout<<"Error running nemslice: "<<sliceStr<<"\n";
	  exit(0);
	}
      }//if
      
      if( writedecomp ){
	std::string spreadFile="./nem_spread.inp";
	std::ofstream spreadfile;
	spreadfile.open(spreadFile.c_str());
	spreadfile 
	  <<"Input FEM file		= "<<infile<<"\n" 
	  <<"LB file         	= "<<nemFile<<"\n" 
	  <<"Debug      	= 0"<<"\n"  
	  <<"Restart Info	= off"<<"\n"  
	  <<"Parallel Disk Info	= number="<<std::to_string(numproc)<<"\n" 
	  <<"Parallel file location	= root=./"<<decompPath<<", subdir=."<<"\n";
	spreadfile.close();
	std::string spreadStr = trilinosPath+"/bin/nem_spread";//+spreadFile;
	char * spreadArg[] = {(char*)"nem_spread",const_cast<char*>((use64Str).c_str()),(char*)"nem_spread.inp",(char*)NULL};
	
	decompfile
	  <<spreadStr<<" "<<spreadArg[1]<<" "<<spreadArg[2]<<"\n";
      }//if
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
	char * spreadArg[] = {(char*)"nem_spread",const_cast<char*>((use64Str).c_str()),const_cast<char*>(spreadFile.c_str()),(char*)NULL};
	std::cout<<"  Running nemspread command: "<<spreadStr <<" "<<spreadArg[1]<<" "<<spreadArg[1]<<"\n";
	//if(-1 == system(spreadStr.c_str()) ){
	if(-1 == do_sys_call(spreadStr.c_str(), spreadArg) ){
	  std::cout<<"Error running nemspread: "<<spreadStr<<"\n";
	  exit(0);
	}
      }
      
      if( writedecomp ){
	decompfile.close();
      }
    }//if( !restart && !skipdecomp)
  }//if(  0 == mypid )
  
  comm->Barrier();
  if( writedecomp ){
#ifdef HAVE_MPI
    //(void) MPI_Finalize ();
#endif
    if( 0 == mypid ){
      std::cout<<"writedecomp completed."<<"\n"<<"\n";
    }
    //exit(0);
    return 1;
  }
  std::string mypidstring(getmypidstring(mypid,numproc));

  outfile=decompPath+std::to_string(mypid+1)+"/"+nemStr+".par."+std::to_string(numproc)+"."+mypidstring;

  if( 0 == mypid  && !restart){
    std::cout<<"\n"<<"Exiting decomp"<<"\n"<<"\n";
  }
  return 0;
}
int join(const int mypid, const int numproc, 
	   const bool skipdecomp)
{
  if( 0 == mypid ){
    std::cout<<"Entering join: PID "<<mypid<<" NumProcs "<<numproc<<"\n"<<"\n";
    std::string decompPath="./decomp/";
    std::string trilinosPath=TRILINOS_DIR;
    std::string comStr = trilinosPath+"bin/epu";// -auto -add_processor_id "+decompPath+"results.e."+std::to_string(numproc)+".000";
//     char * comArg[] = {(char*)"epu",(char*)"-auto", (char*)"-add_processor_id",
// 		       const_cast<char*>((decompPath+"results.e."+std::to_string(numproc)+".000").c_str()),(char*)NULL};
    char * comArg[] = {(char*)"epu",(char*)"-auto", (char*)"-add_processor_id",
		       const_cast<char*>(("decomp/results.e."+std::to_string(numproc)+".000").c_str()),(char*)NULL};
 
    std::ofstream epufile;
    std::string epuFile="./epuscript";
    epufile.open(epuFile.c_str());
    epufile<<"#!/bin/bash"<<"\n";
    //epufile<<comStr<<" "<<comArg[1]<<" "<<comArg[2]<<" "<<comArg[3]<<"\n";
    epufile<<comStr<<" "<<comArg[1]<<" "<<comArg[2]<<" "<<"decomp/results.e."<<std::to_string(numproc)<<".000"<<"\n";
    epufile.close();
    if( !skipdecomp){
      //if(-1 == system(comStr.c_str()) ){
      std::cout<<"Running epu command: "<<comStr<<" "<<comArg[1]<<" "<<comArg[2]<<" "<<comArg[3]<<"\n";
      if(-1 == do_sys_call(comStr.c_str(), comArg) ){
	std::cout<<"Error running epu: "<<comStr<<"\n";
	return -1;
      }
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
    std::cout<<"\n"<<"\n"<<"\n"
	     <<"Tusas Version 1.0 is registered with Los Alamos National Laboratory (LANL) as LA-CC-17-001."
	     <<"\n"<<"\n";
  }
}
void print_copyright(const int mypid)
{
  if(0 == mypid){
    std::cout<<"\n"
	     <<"Copyright (c) 2016, Triad National Security, LLC"<<"\n"
	     <<"\n"
	     <<"All rights reserved."<<"\n"
	     <<"\n"
	     <<"This software was produced under U.S. Government contract 89233218CNA000001"<<"\n"
	     <<"for Los Alamos National Laboratory (LANL), which is operated by Triad"<<"\n"
	     <<"National Security, LLC for the U.S. Department of Energy. The U.S. Government"<<"\n"
	     <<"has rights to use, reproduce, and distribute this software.  NEITHER THE"<<"\n" 
	     <<"GOVERNMENT NOR TRIAD NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS"<<"\n" 
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
	     <<"3. Neither the name of Triad National Security, LLC, Los Alamos National"<<"\n" 
	     <<"   Laboratory, LANL, the U.S. Government, nor the names of its contributors"<<"\n"
	     <<"   may be used to endorse or promote products derived from this software"<<"\n" 
	     <<"   without specific prior written permission."<<"\n"
	     <<"\n"
	     <<"THIS SOFTWARE IS PROVIDED BY TRIAD NATIONAL SECURITY, LLC AND"<<"\n" 
	     <<"CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,"<<"\n" 
	     <<"BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS"<<"\n" 
	     <<"FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL TRIAD"<<"\n"
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
void write_timers()
{ 
  Teuchos::TimeMonitor::summarize(std::cout,false,true,false);
  std::ofstream timefile;
  timefile.open("time.dat");
  Teuchos::TimeMonitor::summarize(timefile,false,true,false);
  timefile.close();
}

std::string getmypidstring(const int mypid, const int numproc)
{
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
  if( numproc > 9999 && numproc < 100000 ){
    if ( mypid < 10 ){
      mypidstring = std::to_string(0)+std::to_string(0)+std::to_string(0)+std::to_string(0)+std::to_string(mypid);
    }
    else if ( mypid > 9 && mypid < 100 ){
      mypidstring = std::to_string(0)+std::to_string(0)+std::to_string(0)+std::to_string(mypid);
    }
    else if ( mypid > 99 && mypid < 1000 ){
      mypidstring = std::to_string(0)+std::to_string(0)+std::to_string(mypid);
    }
    else if ( mypid > 999 && mypid < 10000 ){
      mypidstring = std::to_string(0)+std::to_string(mypid);
    }
    else{
      mypidstring = std::to_string(mypid);
    }
  }//if

  return mypidstring;
}
