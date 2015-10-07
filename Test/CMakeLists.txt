cmake_minimum_required (VERSION 2.6)

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
#add_test( HeatHexNoPrec ${CMAKE_CURRENT_SOURCE_DIR}/compare )
#set_tests_properties ( HeatHexNoPrec PROPERTIES PASS_REGULAR_EXPRESSION " HeatHexNoPrec: P" )

add_test( NAME HeatTet  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HeatTet COMMAND run_test )
#add_test( HeatTet ${CMAKE_CURRENT_SOURCE_DIR}/compare )
#set_tests_properties ( HeatTet PROPERTIES PASS_REGULAR_EXPRESSION " HeatTet: P" )

add_test( NAME PhaseHeatQuadImp  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PhaseHeatQuadImp COMMAND run_test )
#add_test( PhaseHeatQuadImp ${CMAKE_CURRENT_SOURCE_DIR}/compare )
#set_tests_properties ( PhaseHeatQuadImp PROPERTIES PASS_REGULAR_EXPRESSION " PhaseHeatQuadImp: P" )

add_test( NAME PhaseHeatQuadExp  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PhaseHeatQuadExp COMMAND run_test )
#add_test( PhaseHeatQuadExp ${CMAKE_CURRENT_SOURCE_DIR}/compare )
#set_tests_properties ( PhaseHeatQuadExp PROPERTIES PASS_REGULAR_EXPRESSION " PhaseHeatQuadExp: P" )

add_test( NAME PhaseHeatQuad  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PhaseHeatQuad COMMAND run_test )
#add_test( PhaseHeatQuad ${CMAKE_CURRENT_SOURCE_DIR}/compare )
#set_tests_properties ( PhaseHeatQuad PROPERTIES PASS_REGULAR_EXPRESSION " PhaseHeatQuad: P" )

add_test( NAME PhaseHeatQuadNoPrec  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PhaseHeatQuadNoPrec COMMAND run_test )
#add_test( PhaseHeatQuadNoPrec ${CMAKE_CURRENT_SOURCE_DIR}/compare )
#set_tests_properties ( PhaseHeatQuadNoPrec PROPERTIES PASS_REGULAR_EXPRESSION " PhaseHeatQuadNoPrec: P" )

add_test( NAME PhaseHeatTris  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PhaseHeatTris COMMAND run_test )
#add_test( PhaseHeatTris ${CMAKE_CURRENT_SOURCE_DIR}/compare )
#set_tests_properties ( PhaseHeatTris PROPERTIES PASS_REGULAR_EXPRESSION " PhaseHeatTris: P" )

add_test( NAME PhaseHeatQuadPar  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PhaseHeatQuadPar COMMAND run_test )
#add_test( PhaseHeatQuadPar ${CMAKE_CURRENT_SOURCE_DIR}/compare )
#set_tests_properties ( PhaseHeatQuadPar PROPERTIES PASS_REGULAR_EXPRESSION " PhaseHeatQuadPar: P" )

add_test( NAME PhaseHeatQuadParNoPrec  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/PhaseHeatQuadParNoPrec COMMAND run_test )
#add_test( PhaseHeatQuadParNoPrec ${CMAKE_CURRENT_SOURCE_DIR}/compare )
#set_tests_properties ( PhaseHeatQuadParNoPrec PROPERTIES PASS_REGULAR_EXPRESSION " PhaseHeatQuadParNoPrec: P" )
