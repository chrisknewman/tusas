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
set(DIR_TEST "Test")
set(TUSAS_RUN_COMMAND "mpirun" CACHE STRING "Set command for running tests: mpirun (default), srun, jsrun, etc..." )

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

# add all test cases here
# might want to do this in a loop
add_test(
  NAME HeatQuad
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/HeatQuad
  COMMAND
    $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME HeatQuadQ
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/HeatQuadQ
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME HeatHexNoPrec
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/HeatHexNoPrec
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND} 
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME HeatTet
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/HeatTet
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME HeatTetQ
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/HeatTetQ
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME PhaseHeatQuadExp
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/PhaseHeatQuadExp
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME PhaseHeatQuad
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/PhaseHeatQuad
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME PhaseHeatTris
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/PhaseHeatTris
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME PhaseHeatQuadPar
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/PhaseHeatQuadPar
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME PhaseHeatQuadParNoPrec
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/PhaseHeatQuadParNoPrec
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME NeumannQuad
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/NeumannQuad
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME NeumannTriPar
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/NeumannTriPar
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME NeumannQuadQPar
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/NeumannQuadQPar
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME NeumannTriQ
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/NeumannTriQ
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME NeumannHex
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/NeumannHex
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME NeumannHexQParT
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/NeumannHexQParT
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME NeumannTetPar
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/NeumannTetPar
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME NeumannTetT
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/NeumannTetT
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME NeumannTetQT
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/NeumannTetQT
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME RobinQuad
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/RobinQuad
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME RobinHex
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/RobinHex
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME LinIsoHex
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/LinIsoHex
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME LinIsoBodyForceHexPrec
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/LinIsoBodyForceHexPrec
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME LinIsoHeatHexPar
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/LinIsoHeatHexPar
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME CoupledStressQuad
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/CoupledStressQuad
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME HeatTri
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/HeatTri
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME Uehara1
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/Uehara1
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME Farzadi
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/Farzadi
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME CahnHilliardQuadPar
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/CahnHilliardQuadPar
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME CahnHilliardTriQPar
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/CahnHilliardTriQPar
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME RestartPar
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/RestartPar
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME RestartFaultT
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/RestartFaultT
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME WriteSkipDecomp
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/WriteSkipDecomp
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME HeatHexT
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/HeatHexT
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME HeatHexQT
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/HeatHexQT
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME NLHeatHexCN
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/NLHeatHexCN
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME NLHeatHexIMR
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/NLHeatHexIMR
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME AutoCatalyticT
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/AutoCatalyticT
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME RadConvBC
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/RadConvBC
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME LocalProjectionT
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/LocalProjectionT
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME Thermal
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/Thermal
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME FarzadiSphere
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/FarzadiSphere
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME Goldak
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/Goldak
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME FullyCoupled
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/FullyCoupled
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME UnstructuredHex
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/UnstructuredHex
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME Pfhub2
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/Pfhub2
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME Pfhub2Split
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/Pfhub2Split
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

add_test(
  NAME Tonks1kks
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Test/Tonks1kks
  COMMAND $<1:bash> $<1:run_test> ${CMAKE_CURRENT_BINARY_DIR} ${TUSAS_RUN_COMMAND}
  COMMAND_EXPAND_LISTS
)

