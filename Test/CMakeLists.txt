cmake_minimum_required (VERSION 2.6)

SET( DIR_6 ${TRILINOS_DIR}lib )
INCLUDE( ${DIR_6}/cmake/Trilinos/TrilinosConfig.cmake )

SET( CMAKE_C_COMPILER ${Trilinos_C_COMPILER} )
SET( CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} )

project ( TusasTest )

#add_executable ( run_tusas run_tusas.cpp )
#add_executable ( compare compare.cpp )
#add_executable ( extra extra.cpp )


#add_custom_target ( run_tusas_run
#		  COMMAND run_tusas
#                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#                  COMMENT "Running Tusas Now" 
#		  SOURCES run_tusas.cpp )

#add_custom_target ( compare_run
#		  COMMAND compare
#		  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#		  COMMENT "Comparing to Gold Values"
#		  SOURCES compare.cpp )

#add_custom_target ( extra_run
#                  COMMAND extra
#                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#                  COMMENT "Renaming Output"
#                  SOURCES extra.cpp )


#add_dependencies ( compare  run_tusas_run )
#add_dependencies ( extra compare_run )

enable_testing()

add_test( NAME HeatQuad  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HeatQuad COMMAND run_test )

add_test( NAME HeatHex  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HeatHex COMMAND run_test )

add_test( NAME HeatHexNoPrec WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HeatHexNoPrec COMMAND run_test )

add_test( NAME HeatTet  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HeatTet COMMAND run_test )

add_test( NAME PhaseHeatQuadImp  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PhaseHeatQuadImp COMMAND run_test )

add_test( NAME PhaseHeatQuadExp  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PhaseHeatQuadExp COMMAND run_test )

add_test( NAME PhaseHeatQuad  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PhaseHeatQuad COMMAND run_test )

add_test( NAME PhaseHeatQuadNoPrec  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PhaseHeatQuadNoPrec COMMAND run_test )

add_test( NAME PhaseHeatTris  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PhaseHeatTris COMMAND run_test )

add_test( NAME PhaseHeatQuadPar  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PhaseHeatQuadPar COMMAND run_test )

add_test( NAME PhaseHeatQuadParNoPrec  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PhaseHeatQuadParNoPrec COMMAND run_test )

add_test( NAME NeumannQuad  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/NeumannQuad COMMAND run_test )

add_test( NAME NeumannTriPar  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/NeumannTriPar COMMAND run_test )

add_test( NAME NeumannQuadQPar  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/NeumannQuadQPar COMMAND run_test )

add_test( NAME NeumannTriQ  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/NeumannTriQ COMMAND run_test )

add_test( NAME NeumannHex  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/NeumannHex COMMAND run_test )

add_test( NAME NeumannTetPar  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/NeumannTetPar COMMAND run_test )


#add_test( NAME RestartPar  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/RestartPar COMMAND run_test )
