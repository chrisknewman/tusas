# Building and Running Tusas

Throuhgout the setup process we will be assuming that the directory where Tusas will be cloned to is `TUSAS_BASE`. For example to install Tusas and it's third party libaries in your home directory run

    export TUSAS_BASE=$HOME/my-tusas


## Setting Up Third Party Libraries (TPL)

The following steps show how to install the TPL. Here we will be using [spack](https://spack.io/) to install the libaries. Spack is not required, so if you want you can install the TPL manually. 


Set the location you want spack to be installed

    export SPACK_USER_CONFIG_PATH=${TUSAS_BASE}/spack
    export SPACK_USER_CACHE_PATH=${TUSAS_BASE}/.spack

Clone and source spack     

    cd ${TUSAS_BASE}
    git clone https://github.com/spack/spack.git
    source ${SPACK_USER_CONFIG_PATH}/share/spack/setup-env.sh

Use spack to install third party libaries 

    spack install -j 20 lmod
    spack install -j 20 openmpi
    spack install -j 20 netcdf-c +parallel-netcdf
    spack install -j 20 trilinos +hdf5 +exodus +nox +thyra +ml +openmp +zoltan +zoltan2 +pamgen +chaco +stratimikos +boost +isorropia 

If for some reason some of the packages don't install you may need to a [mirror](https://spack.readthedocs.io/en/latest/mirrors.html) to spack, for example 

    spack mirror add mirror-a https://mirror/link/that/you/want
    spack mirror add local-b file://path/to/my/local/mirror


## Tusas

### Loading TPL

First we will get a copy of Tusas 

    cd ${TUSAS_BASE}
    git clone https://github.com/chrisknewman/tusas.git

Next make sure that the libaries are loaded, if spack was used then run the following

    source ${SPACK_USER_CONFIG_PATH}/share/spack/setup-env.sh
    source $(spack location -i lmod)/lmod/lmod/init/bash
    source ${SPACK_USER_CONFIG_PATH}/share/spack/setup-env.sh

Yes the one command should be run both before and after. Now load the needed modules

    module load cmake 
    module load openmpi
    module load netcdf-c
    module load parallel-netcdf
    module load trilinos

### Compiling

Now to compile Tusas 

    cd ${TUSA_BASE}/tusas 
    mkdir build 
    cd build 
    cmake \
        -DTRILINOS_DIR=$(spack location -i trilinos) \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DTUSAS_CXX_FLAGS="-DTUSAS_KOKKOS_PRINT_CONFIG -w" \
        -DCMAKE_CXX_STANDARD=14 \
        ..
    make -j 8

### Running

From inside the directory where tusas was built you can run, 

    ./tusas --input-file=tusas.xml 
