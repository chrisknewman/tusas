#!/bin/bash


calculate() { printf "%s\n" "$@" | bc -l; }
sci2float() { printf "%.15f\n" "$@"; }
float2int() { printf "%.0f\n" "$@"; }


#echo "sci2float 1e-5  = $(sci2float '1e-5')"
#echo "sci2float 5.e-7 = $(sci2float '5e-7')"
#echo "sci2float 1e-5 / 5.e-7 = $(calculate "$(sci2float '1e-5') / $(sci2float '5e-7')")"
#exit


#### BEGIN user configurable variables
LOG=log.txt
CONFDIR=./configs
OUTDIR=./out

#CASENAMES='mms-eta-constcab-dirichlet
#           mms-eta-constcab-neumann
#           mms-coupled-constcab-dirichlet
#           mms-coupled-constcab-neumann'
CASENAMES='mms-coupled-constcab-dirichlet
           mms-coupled-constcab-neumann'
MESHES='1x2000
        1x4000
        1x8000'
THETAS='1.0
        0.5'
VARS='eta
      c
      mu'
#### END user configurable variables


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


for CASENAME in $CASENAMES; do for MESH in $MESHES; do for THETA in $THETAS; do
  if [[ $CASENAME == *coupled* ]]; then
    export RELRES='1.e-8'
    STOPTIME='1e-5'
    DTS='1e-6
         5e-7
         25e-8
         125e-9
         625e-10
         3125e-11'
  else
    export RELRES='1.e-11'
    STOPTIME='1e-0'
    DTS='1e-1
         5e-2
         25e-3
         125e-4
         625e-5
         3125e-6'
  fi

  for DT in $DTS; do
    export CASENAME=$CASENAME; export MESH=$MESH; export DT=$DT; export THETA=$THETA
    export NT=$(float2int $(calculate "$(sci2float $STOPTIME) / $(sci2float $DT)"))
    if [[ $CASENAME == *coupled* ]]; then
      export USEPREC='true'
    else
      export USEPREC='false'
    fi

    CONF=${CASENAME}_mesh@${MESH}_dt@${DT}_theta@${THETA}
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
      RMSOUT=$OUTDIR/RMS_${CONF}_var@${VAR}.dat
        
      if [[ -e $RMSFILE ]]; then
        echo "--- RUNNING: mv $RMSFILE $RMSOUT" | tee -a $LOG
        mv $RMSFILE $RMSOUT
      fi
    done
  done
done; done; done

