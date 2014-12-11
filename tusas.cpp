
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

#include "Mesh.h"


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

  Mesh * in_mesh = new Mesh(0,false);
  //string   filename                = "meshes/tri24.e"    ;
  //string   filename                = "meshes/tri96.e"    ;
  //string   filename                = "meshes/tri384.e"    ;
  //string   filename                = "meshes/quad16.e"    ;
  //string   filename                = "meshes/quad64.e"    ;
  string   filename                = "meshes/quad256.e"    ;
  //string   filename                = "meshes/quad1024.e"    ;
  //string   filename                = "meshes/quad4096.e"    ;
  in_mesh->read_exodus(&filename[0]);

  double dt = .0125;
  int numSteps = 4;

  // Create the model evaluator object
  timestep<double> * model = new ModelEvaluatorHEAT<double>(Teuchos::rcp(&Comm,false),in_mesh,dt);
  
  double curTime = 0.0; 
  double endTime = (float)numSteps*dt;
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
  Teuchos::TimeMonitor::summarize();
}
