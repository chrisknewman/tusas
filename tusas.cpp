
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
#include "ModelEvaluatorPHASE_HEAT_Exp.hpp"

#include "Mesh.h"

#include "ParamNames.h"
#include "readInput.h"


using namespace std;

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

  if(1 != Comm.NumProc() ) exit(0);

  Teuchos::ParameterList paramList;
  readParametersFromFile(argc, argv, paramList );

  //cn we should maybe control implicit/explicit via the old fashioned theta formulation;
  //cn and only have the one phase heat class

  Mesh * in_mesh = new Mesh(0,false);

  //in_mesh->read_exodus(&(paramList.get<std::string> (TusasmeshNameString) )[0]);
  in_mesh->read_exodus((paramList.get<std::string> (TusasmeshNameString) ).c_str());

  //we want end dt = .14 here; dt=.001   for dendquad300.e and ModelEvaluatorPHASE_HEAT
  //                           dt=.0001  for dendquad600.e and ModelEvaluatorPHASE_HEAT
  //                           dt=.00001 for dendquad300.e and ModelEvaluatorPHASE_HEAT_Exp
  //                           dt=.000001 for dendquad600.e and ModelEvaluatorPHASE_HEAT_Exp

  double dt = paramList.get<double> (TusasdtNameString);
  timestep<double> * model;
  if( paramList.get<std::string> (TusasmethodNameString)  == "phaseheat") {
    model = new ModelEvaluatorPHASE_HEAT<double>(Teuchos::rcp(&Comm,false),in_mesh,paramList);
  }
  else if ( paramList.get<std::string> (TusasmethodNameString)  == "heat") {
    model = new ModelEvaluatorHEAT<double>(Teuchos::rcp(&Comm,false),in_mesh,dt);
  }
  else {
    std::cout<<"Invalid method."<<std::endl<<std::endl;
    exit(0);
  //timestep<double> * model = new ModelEvaluatorPHASE_HEAT_Exp<double>(Teuchos::rcp(&Comm,false),in_mesh,dt);
  }

  int numSteps = paramList.get<int> (TusasntNameString);
  double curTime = 0.0; 
  double endTime = (double)numSteps*dt;
  int elapsedSteps =0;

  model->initialize();

  while ( ( curTime <= endTime ) && ( elapsedSteps < numSteps ) ) {
    model->advance();
    curTime += dt;
    elapsedSteps++;
    cout<< endl << "Time step " <<elapsedSteps <<" of "<<numSteps
	<< "  ( "<<(float)elapsedSteps/(float)numSteps*100. <<" % )   t = "
	<<curTime<<"  t final = "<<endTime<< endl<<endl<<endl;
    
  }

  model->finalize();
  
  delete in_mesh;
  delete model;
  Teuchos::TimeMonitor::summarize();
}
