#!/bin/bash


calculate() { printf "%s\n" "$@" | bc -l; }
sci2float() { printf "%f\n" "$@"; }
float2int() { printf "%.0f\n" "$@"; }


#### BEGIN user configurable variables
LOG=log.txt
CONFDIR=./configs
OUTDIR=./out

BASENAMES='mms-eta-constcab
           mms-coupled-constcab'
MESHES='1x2000
        1x4000
        1x8000'
DTS='1e-1
     5e-2
     25e-3
     125e-4
     625e-5
     3125e-6'
THETAS='1.0
        0.5'
BCS='dirichlet
     neumann'
#### END user configurable variables


VARS='eta
      c
      mu'


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

if [[ ! -d $CONFDIR ]]; then
    mkdir $CONFDIR
fi
if [[ ! -d $OUTDIR ]]; then
    mkdir $OUTDIR
fi

NPROCS=8  # need to change epuscript too!
RUNTUSAS="$MPIRUN -n $NPROCS $TUSAS --kokkos-num-threads=1"


for BASENAME in $BASENAMES; do for MESH in $MESHES; do for DT in $DTS; do for THETA in $THETAS; do for BC in $BCS; do
  export BASENAME=$BASENAME; export MESH=$MESH; export DT=$DT; export THETA=$THETA; export BC=$BC
  export NT=$(float2int $(calculate "1 / $(sci2float $DT)"))
  export TESTCASE="$BASENAME-$BC"
  if [[ $BASENAME == @('mms-coupled-constcab'|'other') ]]; then
    export USEPREC='true'
  else
    export USEPREC='false'
  fi

  CONF=${BASENAME}_bc@${BC}_mesh@${MESH}_dt@${DT}_theta@${THETA}
  INPUT=$CONFDIR/$CONF.xml
  OUTPUT=$OUTDIR/$CONF.e

  # clean up previous run
  rm -rf results.e decomp/ decompscript nem_spread.inp input-ldbl *.dat *_rms*.dat

  # write config to file
  cat mms_TEMPLATE.xml | envsubst > $CONFDIR/$CONF.xml
  
  echo "--- RUNNING: $RUNTUSAS --input-file=$INPUT --writedecomp" | tee -a $LOG
  $RUNTUSAS --input-file=$INPUT --writedecomp &>> $LOG
  bash decompscript &>> $LOG

  echo "--- RUNNING: $RUNTUSAS --input-file=$INPUT --skipdecomp" | tee -a $LOG
  $RUNTUSAS --input-file=$INPUT --skipdecomp &>> $LOG
  bash epuscript &>> $LOG

  echo "--- RUNNING: mv results.e $OUTPUT" | tee -a $LOG
  mv results.e $OUTPUT

  for VAR in $VARS; do
    RMSFILE=$(ls | grep ${VAR}_rms*.dat)
    RMSOUT=$OUTDIR/RMS_${VAR}_${CONF}.dat
    
    if [[ -e $RMSFILE ]]; then
      echo "--- RUNNING: mv $RMSFILE $RMSOUT" | tee -a $LOG
      mv $RMSFILE $RMSOUT
    fi
  done

done; done; done; done; done

