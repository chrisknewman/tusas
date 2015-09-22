#include <stdlib.h> 

int main ()
{

   system ("../tusas --input-file=HeatQuad/tusas.xml");
   system ( "mv results.e HeatQuad/");

   system ("../tusas --input-file=HeatHex/tusas.xml");
   system ( "mv results.e HeatHex/");

   system ("../tusas --input-file=HeatHexNoPrec/tusas.xml");
   system ( "mv results.e HeatHexNoPrec/");

   system ("../tusas --input-file=HeatTet/tusas.xml");
   system ( "mv results.e HeatTet/");

   system ("../tusas --input-file=PhaseHeatQuadImp/tusas.xml");
   system ( "mv results.e PhaseHeatQuadImp/");

   system ("../tusas --input-file=PhaseHeatQuadExp/tusas.xml");
   system ( "mv results.e PhaseHeatQuadExp/");

   system ("../tusas --input-file=PhaseHeatQuad/tusas.xml");
   system ( "mv results.e PhaseHeatQuad/");

   system ("../tusas --input-file=PhaseHeatQuadNoPrec/tusas.xml");
   system ( "mv results.e PhaseHeatQuadNoPrec/");

   system ("../tusas --input-file=PhaseHeatTris/tusas.xml");
   system ( "mv results.e PhaseHeatTris/");

   //system ("/usr/local/gcc/4.9.2/openmpi-1.8.4/bin/mpirun -np 4 ../tusas --input-file=PhaseHeatQuadPar/tusas.xml");
   system ("mpirun -np 4 ../tusas --input-file=PhaseHeatQuadPar/tusas.xml");
   system ( "mv results.e PhaseHeatQuadPar/");

   //system ("/usr/local/gcc/4.9.2/openmpi-1.8.4/bin/mpirun -np 4 ../tusas --input-file=PhaseHeatQuadParNoPrec/tusas.xml");
   system ("mpirun -np 4 ../tusas --input-file=PhaseHeatQuadParNoPrec/tusas.xml");
   system ( "mv results.e PhaseHeatQuadParNoPrec/");

   return 0;

}
