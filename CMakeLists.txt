cmake_minimum_required(VERSION 3.23.0)

# define the Tusas project without enabled 
# languages for now -- we need to find which 
# compilers were used to build the version of
# Trilinos we are linking to first
project(Tusas NONE)

# find Trilinos
# note that the find_package call gives us access to 
# the variables in TrilinosConfig.cmake
if (EXISTS ${TRILINOS_DIR}/lib/cmake/Trilinos)
    set(Trilinos_DIR ${TRILINOS_DIR}/lib/cmake/Trilinos)
elseif (EXISTS ${TRILINOS_DIR}/lib64/cmake/Trilinos)
    set(Trilinos_DIR ${TRILINOS_DIR}/lib64/cmake/Trilinos)
else()
    message(SEND_ERROR "Could not find Trilinos cmake library")
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Trilinos REQUIRED)

# set our compilers to the same used to build Trilinos
set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
set(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})

# set compiler flags
set(CMAKE_CXX_FLAGS
  "${Trilinos_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS} ${TUSAS_CXX_FLAGS}"
)
set(CMAKE_C_FLAGS
  "${Trilinos_C_COMPILER_FLAGS} ${CMAKE_C_FLAGS}"
)

# now check for working compilers
enable_language(C)
enable_language(CXX)

# add an execuatble target
add_executable(tusas)

# best practice is to write sources out explicitly
# since this only needs to be done once and is
# faster/safer than globbing or pulling in 
# whole directories
target_sources(tusas
  PRIVATE
    basis/basis.cpp
    elem_color/elem_color.cpp 
    error_estimator/error_estimator.cpp
    input/ReadInput.cpp
    mesh/Mesh.C
    periodic_bc/periodic_bc.cpp 
    post_process/post_process.cpp
    projection/interpflux.cpp 
    projection/interpfluxavg.cpp 
    projection/projection.cpp 
    random_distribution/random_distribution.cpp
    tusas.cpp

  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
      basis/include
      elem_color/include
      error_estimator/include
      input/include
      mesh/include
      periodic_bc/include
      post_process/include
      preconditioner/include
      projection/include
      random_distribution/include
      timestep/include
      utilities/include
    FILES
      basis/include/basis.hpp 
      elem_color/include/elem_color.h 
      error_estimator/include/error_estimator.h
      input/include/ReadInput.h 
      mesh/include/Mesh.h
      periodic_bc/include/periodic_bc.h 
      post_process/include/post_process.h 
      preconditioner/include/preconditioner.hpp
      preconditioner/include/preconditioner_def.hpp
      projection/include/interpflux.h 
      projection/include/interpfluxavg.h 
      projection/include/projection.h 
      random_distribution/include/random_distribution.h
      timestep/include/ModelEvaluatorTPETRA.hpp
      timestep/include/ModelEvaluatorTPETRA_def.hpp
      utilities/include/greedy_tie_break.hpp
)

# set libaries we are linking to
target_link_libraries(tusas
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  ${Trilinos_EXTRA_LD_FLAGS}
)

# not sure what this is doing, but there is 
# probably a more standard way if we care
configure_file(tusas.h.in tusas.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})


#---------------------------------------------------------------------------------
# Test set up
#---------------------------------------------------------------------------------

enable_testing()
set(TUSAS_TEST_DIR "Test")
set(
  TUSAS_RUN_COMMAND "mpirun" CACHE STRING 
  "Set command for running tests: mpirun (default), srun, jsrun, etc..." 
)

file(
  COPY
    ${TRILINOS_DIR}/bin/exodiff
  DESTINATION
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# use -DUSE_BASH on mac with clang and openmp, 
# as DYLD_LIBRARY_PATH does not get propogated 
# to child processes in shells found in /bin/*
set(TUSAS_BASH 0)
if (DEFINED ${USE_BASH})
  set(TUSAS_BASH 1)
endif()

# list all test cases here
set(TUSAS_TESTS 
  AutoCatalytic 
  AutoCatalyticT
  CahnHilliardQuadPar
  CahnHilliardTriQPar
  CoupledStressQuad
  Farzadi
  FarzadiSphere
  FullyCoupled
  Goldak
  HeatHex
  HeatHexNoPrec
  HeatHexQT
  HeatHexT
  HeatQuad
  HeatQuadQ
  HeatTet
  HeatTetQ
  HeatTri
  LinIsoBodyForceHexPrec
  LinIsoHeatHexPar
  LinIsoHex
  LocalProjectionT
  NLHeatHexCN
  NLHeatHexIMR
  NeumannHex
  NeumannHexQParT
  NeumannQuad
  NeumannQuadQPar
  NeumannTetPar
  NeumannTetQT
  NeumannTetT
  NeumannTriPar
  NeumannTriQ
  NonlinearLog0D
  PeriodicHexPar
  PeriodicQuad
  Pfhub2
  Pfhub2Split
  PhaseHeatQuad
  PhaseHeatQuadExp
  PhaseHeatQuadImp
  PhaseHeatQuadNoPrec
  PhaseHeatQuadPar
  PhaseHeatQuadParNoPrec
  PhaseHeatTris
  RadConvBC
  RestartFaultT
  RestartPar
  RobinHex
  RobinQuad
  Thermal
  Tonks1kks
  Uehara1
  UnstructuredHex
  WriteSkipDecomp
)

# add all test cases from TUSAS_TESTS
foreach(TUSAS_TEST IN LISTS TUSAS_TESTS)
  add_test(
    NAME ${TUSAS_TEST}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${TUSAS_TEST_DIR}/${TUSAS_TEST}
    COMMAND
      $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
    COMMAND_EXPAND_LISTS
  )
endforeach()

