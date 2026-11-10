#!/bin/bash


calculate() { printf "%s\n" "$@" | bc -l; }
sci2float() { printf "%.15f\n" "$@"; }
float2int() { printf "%.0f\n" "$@"; }
getl2diff() {
    LINE=($(cat $1 | grep "$2 " | grep "L2 norm of diff="))

    count=0
    for WORD in ${LINE[@]}; do
        count=$((count + 1))
        if [[ $WORD == 'diff=' ]]; then
            break
        fi
    done

    echo "${LINE[$count]}"
}


#### BEGIN user configurable variables
LOG=log.txt
CONFDIR=./configs
OUTDIR=./out

CASENAMES=(
    'mms-eta-constcab-neumann'
    'mms-coupled-constcab-neumann'
)
MESHES=('1x250' '1x500' '1x1000' '1x2000')
THETAS=('1.0' '0.5')
VARS=('eta' 'c' 'mu')
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

EXODIFF=$TRILINOS_DIR/bin/exodiff
if [[ ! -e $EXODIFF ]]; then
    echo "$EXODIFF not found. Is \$TRILINOS_DIR set?"
    exit 1
fi
echo "--- Found exodiff executable: $EXODIFF" | tee -a $LOG

if [[ ! -d $CONFDIR ]]; then
    mkdir $CONFDIR
fi
if [[ ! -d $OUTDIR ]]; then
    mkdir $OUTDIR
fi

NPROCS=8  # need to change epuscript too!
RUNTUSAS="$MPIRUN -n $NPROCS $TUSAS --kokkos-num-threads=1"


for CASENAME in ${CASENAMES[@]}; do for MESH in ${MESHES[@]}; do for THETA in ${THETAS[@]}; do
  if [[ $CASENAME == *coupled* ]]; then
    export USEPREC='true'
    export RELRES='1.e-11'
    STOPTIME='1e-0'
    DTS=('1e-1' '5e-2' '25e-3' '125e-4' '625e-5' '3125e-6')
  else
    export USEPREC='false'
    export RELRES='1.e-11'
    STOPTIME='1e-0'
    DTS=('1e-1' '5e-2' '25e-3' '125e-4' '625e-5' '3125e-6')
  fi

  for DT in ${DTS[@]}; do
    export CASENAME=$CASENAME; export MESH=$MESH; export DT=$DT; export THETA=$THETA
    export NT=$(float2int $(calculate "$(sci2float $STOPTIME) / $(sci2float $DT)"))

    CONF=case@${CASENAME}_mesh@${MESH}_dt@${DT}_theta@${THETA}
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

    if [[ $DT != ${DTS[0]} ]]; then
        PREVCONF=case@${CASENAME}_mesh@${MESH}_dt@${PREVDT}_theta@${THETA}
        PREVFILE=$OUTDIR/$PREVCONF.e
        if [[ -e $PREVFILE ]]; then
            echo "--- RUNNING: $EXODIFF -steps last -l2norms $OUTPUT $PREVFILE"
            $EXODIFF -steps last -l2norms $OUTPUT $PREVFILE &>> l2diff.dat

            for VAR in ${VARS[@]}; do
                DIFFOUT=$OUTDIR/DIFF_${CONF}_var@${VAR}.dat
                DIFF=$(getl2diff l2diff.dat $VAR)
                if [[ -n $DIFF ]]; then
                    echo "$DIFF" > $DIFFOUT
                fi
            done
        else
            echo "### WARNING: $PREVFILE does not exist. Skipping exodiff run."
        fi
    fi
    PREVDT=$DT

    for VAR in ${VARS[@]}; do
      RMSFILE=$(ls | grep ${VAR}_rms*.dat)

      if [[ -n $RMSFILE ]]; then
          RMSOUT=$OUTDIR/RMS_${CONF}_var@${VAR}.dat

          NLINES=$(wc -l < $RMSFILE)
          if [[ '1' == $NLINES ]]; then
              echo "### ERROR: $RMSFILE has only $NLINES line(s)." | tee -a $LOG
          elif [[ -e $RMSFILE ]]; then
            echo "--- RUNNING: mv $RMSFILE $RMSOUT" | tee -a $LOG
            mv $RMSFILE $RMSOUT
          fi
      fi
    done
  done
done; done; done

