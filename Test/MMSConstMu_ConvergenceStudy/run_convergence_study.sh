#!/bin/bash


LOG=log.txt

EXEDIR=$1
if [[ $(echo $EXEDIR | cut -c ${#EXEDIR}) == '/' ]]; then
  EXEDIR=$(echo $EXEDIR | sed 's/.$//')
fi

TUSAS=$EXEDIR/tusas
if [[ ! -e $TUSAS ]]; then
  echo "$TUSAS not found."
  exit 1
fi
echo "--- Found tusas executable: $TUSAS" | tee $LOG

MPIRUN=$2
if [[ ! $(which $MPIRUN) ]]; then
  echo "MPI runner $MPIRUN not found."
  exit 1
fi
echo "--- Running MPI with $MPIRUN." | tee -a $LOG

NPROCS=8  # need to change epuscript too!
RUNTUSAS="$MPIRUN -n $NPROCS $TUSAS --kokkos-num-threads=1"


CONFDIR=./configs
OUTDIR=./out
CONFS='mansoln_1x2000_dt1e-1_theta1
       mansoln_1x2000_dt5e-2_theta1
       mansoln_1x2000_dt25e-3_theta1
       mansoln_1x2000_dt125e-4_theta1
       mansoln_1x2000_dt625e-5_theta1
       mansoln_1x2000_dt3125e-6_theta1
       mansoln_1x4000_dt1e-1_theta1
       mansoln_1x4000_dt5e-2_theta1
       mansoln_1x4000_dt25e-3_theta1
       mansoln_1x4000_dt125e-4_theta1
       mansoln_1x4000_dt625e-5_theta1
       mansoln_1x4000_dt3125e-6_theta1
       mansoln_1x8000_dt1e-1_theta1
       mansoln_1x8000_dt5e-2_theta1
       mansoln_1x8000_dt25e-3_theta1
       mansoln_1x8000_dt125e-4_theta1
       mansoln_1x8000_dt625e-5_theta1
       mansoln_1x8000_dt3125e-6_theta1'


for CONF in $CONFS; do
  INPUT=$CONFDIR/$CONF.xml
  OUTPUT=$OUTDIR/$CONF.e
  RMSOUT=$OUTDIR/RMS_$CONF.dat
  
  # clean up
  rm -rf results.e decomp/ decompscript nem_spread.inp input-ldbl *.dat

  echo "--- RUNNING: $RUNTUSAS --input-file=$INPUT --writedecomp" | tee -a $LOG
  $RUNTUSAS --input-file=$INPUT --writedecomp &>> $LOG
  bash decompscript &>> $LOG

  echo "--- RUNNING: $RUNTUSAS --input-file=$INPUT --skipdecomp" | tee -a $LOG
  $RUNTUSAS --input-file=$INPUT --skipdecomp &>> $LOG
  bash epuscript &>> $LOG

  echo "--- RUNNING: mv results.e $OUTPUT" | tee -a $LOG
  mv results.e $OUTPUT

  echo "--- RUNNING: mv rms1.dat $RMSOUT" | tee -a $LOG
  mv rms1.dat $RMSOUT
done

