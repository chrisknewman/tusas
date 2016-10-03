
#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>

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

using namespace std;

int decomp(const int mypid, const int numproc, const std::string& infile, std::string& outfile, const bool restart, const Epetra_Comm * comm);
int join(const int mypid, const int numproc);

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

  Teuchos::ParameterList paramList;

  readParametersFromFile(argc, argv, paramList, mypid );

  Mesh * in_mesh = new Mesh(mypid,numproc,false);

  if(1 == numproc ){
    in_mesh->read_exodus((paramList.get<std::string> (TusasmeshNameString) ).c_str());
  }
  else {
    if( paramList.get<std::string> (TusasmethodNameString)  != "nemesis") {
      std::cout<<"More than 1 proc only implemented for nemesis class now."<<std::endl;
      exit(0);
    }
    std::string pfile;
    decomp(mypid, numproc, paramList.get<std::string> (TusasmeshNameString), pfile, paramList.get<bool> (TusasrestartNameString),&Comm);
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
    std::cout<<"Invalid method."<<std::endl<<std::endl;
    exit(0);
  }

  model->initialize();
  double curTime = model->get_start_time(); 
  int elapsedSteps = model->get_start_step();
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
      if(0 == mypid) std::cout<<"Writing exodus file : timestep :"<<elapsedSteps<<std::endl;
      
      model->write_exodus();
    }
  }

  model->finalize();
  
  if(1 != numproc ) join(mypid, numproc);

  Teuchos::TimeMonitor::summarize();

  delete model;
  delete in_mesh;
}

int decomp(const int mypid, const int numproc, const std::string& infile, std::string& outfile, const bool restart, const Epetra_Comm * comm){
  std::string decompPath="decomp/";
  std::string nemStr = "tusas_nemesis";
  if( 0 == mypid && !restart){
    std::cout<<"Entering decomp: PID "<<mypid<<" NumProcs "<<numproc<<std::endl<<std::endl;

    std::string comStr = "rm -r "+decompPath+";mkdir "+decompPath;
    if(-1 == system(comStr.c_str()) ){
      std::cout<<"Error creating directory: "<<decompPath<<std::endl;
      exit(0);
    }
    std::cout<<"  Creating decomp dir: "<<comStr<<std::endl;

    for( int i = 0; i < numproc; i++){
      std::string numStr = std::to_string(i+1);
      //std::string mkdirStr = "rm -rf "+decompPath+numStr+";mkdir "+decompPath+numStr;
      std::string mkdirStr = "mkdir "+decompPath+numStr;
      if(-1 == system(mkdirStr.c_str()) ){
	std::cout<<"Error creating directory: "<<numStr<<std::endl;
	exit(0);
      }
      std::cout<<"  Creating decomp dirs: "<<mkdirStr<<std::endl;
    }

    //std::string trilinosPath="/Users/cnewman/src/trilinos-11.12.1-Source/GCC_4_9_1_MPI_OMP_DBG/";
    std::string trilinosPath=TRILINOS_DIR;
    //std::string trilinosPath=getenv("TRILINOS_DIR");
    std::string nemFile =decompPath+nemStr+".nemI";
    std::string sliceStr = trilinosPath+"/bin/nem_slice -e -m mesh="+std::to_string(numproc)+" -l inertial -o "+
      nemFile+" "+infile;
    std::cout<<"  Running nemslice command: "<<sliceStr <<std::endl;
    if(-1 == system(sliceStr.c_str()) ){
      std::cout<<"Error running nemslice: "<<sliceStr<<std::endl;
      exit(0);
    }
    std::string spreadFile=decompPath+"nem_spread.inp";
    std::ofstream spreadfile;
    spreadfile.open(spreadFile.c_str());
    spreadfile 
      <<"Input FEM file		= "<<infile<<std::endl 
      <<"LB file         	= "<<nemFile<<std::endl 
      <<"Restart Time list	= off"<<std::endl 
      <<"Parallel Disk Info	= number="<<std::to_string(numproc)<<std::endl 
      <<"Parallel file location	= root=./"<<decompPath<<", subdir=.";
    spreadfile.close();
    std::string spreadStr = trilinosPath+"/bin/nem_spread "+spreadFile;
    std::cout<<"  Running nemspread command: "<<spreadStr <<std::endl;
    if(-1 == system(spreadStr.c_str()) ){
      std::cout<<"Error running nemspread: "<<spreadStr<<std::endl;
      exit(0);
    }
  }
  comm->Barrier();
  std::string mypidstring;
#if 0
  if ( numproc > 9 && mypid < 10 ){
    mypidstring = std::to_string(0)+std::to_string(mypid);
  }
  else{
    mypidstring = std::to_string(mypid);
  }
#endif

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
  }
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
  }
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
  }



  outfile=decompPath+std::to_string(mypid+1)+"/"+nemStr+".par."+std::to_string(numproc)+"."+mypidstring;
  //std::cout<<outfile<<std::endl;

  if( 0 == mypid  && !restart){
    std::cout<<std::endl<<"Exiting decomp"<<std::endl<<std::endl;
  }
  //exit(0);
  return 0;
}
int join(const int mypid, const int numproc)
{
  if( 0 == mypid ){
    std::cout<<"Entering join: PID "<<mypid<<" NumProcs "<<numproc<<std::endl<<std::endl;
    std::string decompPath="decomp/";
    std::string trilinosPath=TRILINOS_DIR;
    //std::string trilinosPath=getenv("TRILINOS_DIR");
    std::string comStr = trilinosPath+"/bin/epu -auto -add_processor_id "+decompPath+"results.e."+std::to_string(numproc)+".000";
    
    std::cout<<"Running epu command: "<<comStr <<std::endl;
    if(-1 == system(comStr.c_str()) ){
      std::cout<<"Error running epu: "<<comStr<<std::endl;
      exit(0);
    }
  }
}
