#!/bin/bash


calculate() { printf "%s\n" "$@" | bc -l; }
sci2float() { printf "%f\n" "$@"; }
float2int() { printf "%.0f\n" "$@"; }


# BEGIN user configurable variables
LOG=log.txt
CONFDIR=./configs
OUTDIR=./out

BASENAME=mms-constmu
MESHES='1x2000
        1x4000
        1x8000'
DTS='1e-1
     5e-2
     25e-3
     125e-4
     625e-5
     3125e-6'
THETAS='1'
BCS='dirichlet'
# END user configurable variables


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


for MESH in $MESHES; do for DT in $DTS; do for THETA in $THETAS; do for BC in $BCS; do
  export MESH=$MESH; export DT=$DT; export THETA=$THETA; export BC=$BC
  export NT=$(float2int $(calculate "1 / $(sci2float $DT)"))

  CONF=${BASENAME}-${BC}_mesh@${MESH}_dt@${DT}_theta@${THETA}
  INPUT=$CONFDIR/$CONF.xml
  OUTPUT=$OUTDIR/$CONF.e
  RMSOUT=$OUTDIR/RMS_$CONF.dat

  # write config to file
  cat ${BASENAME}_TEMPLATE.xml | envsubst > $CONFDIR/$CONF.xml
  
  # clean up previous run
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

done; done; done; done

