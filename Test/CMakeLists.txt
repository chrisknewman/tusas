cmake_minimum_required (VERSION 2.6)

project ( TusasTest )

add_executable ( run_tusas run_tusas.cpp )
add_executable ( compare compare.cpp )
add_executable ( extra extra.cpp )

add_custom_target ( run_tusas_run
		  COMMAND run_tusas
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "Running Tusas Now" 
		  SOURCES run_tusas.cpp )

add_custom_target ( compare_run
		  COMMAND compare
		  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		  COMMENT "Comparing to Gold Values"
		  SOURCES compare.cpp )

add_custom_target ( extra_run
                  COMMAND extra
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "Renaming Output"
                  SOURCES extra.cpp )



add_dependencies ( compare  run_tusas_run )
add_dependencies ( extra compare_run )

enable_testing()

add_test( HeatQuad ${CMAKE_CURRENT_SOURCE_DIR}/compare )
set_tests_properties ( HeatQuad PROPERTIES PASS_REGULAR_EXPRESSION " HeatQuad: P" )

add_test( HeatHex ${CMAKE_CURRENT_SOURCE_DIR}/compare )
set_tests_properties ( HeatHex PROPERTIES PASS_REGULAR_EXPRESSION " HeatHex: P" )

add_test( HeatHexNoPrec ${CMAKE_CURRENT_SOURCE_DIR}/compare )
set_tests_properties ( HeatHexNoPrec PROPERTIES PASS_REGULAR_EXPRESSION " HeatHexNoPrec: P" )

add_test( HeatTet ${CMAKE_CURRENT_SOURCE_DIR}/compare )
set_tests_properties ( HeatTet PROPERTIES PASS_REGULAR_EXPRESSION " HeatTet: P" )

add_test( PhaseHeatQuadImp ${CMAKE_CURRENT_SOURCE_DIR}/compare )
set_tests_properties ( PhaseHeatQuadImp PROPERTIES PASS_REGULAR_EXPRESSION " PhaseHeatQuadImp: P" )

add_test( PhaseHeatQuadExp ${CMAKE_CURRENT_SOURCE_DIR}/compare )
set_tests_properties ( PhaseHeatQuadExp PROPERTIES PASS_REGULAR_EXPRESSION " PhaseHeatQuadExp: P" )

add_test( PhaseHeatQuad ${CMAKE_CURRENT_SOURCE_DIR}/compare )
set_tests_properties ( PhaseHeatQuad PROPERTIES PASS_REGULAR_EXPRESSION " PhaseHeatQuad: P" )

add_test( PhaseHeatQuadNoPrec ${CMAKE_CURRENT_SOURCE_DIR}/compare )
set_tests_properties ( PhaseHeatQuadNoPrec PROPERTIES PASS_REGULAR_EXPRESSION " PhaseHeatQuadNoPrec: P" )

add_test( PhaseHeatTris ${CMAKE_CURRENT_SOURCE_DIR}/compare )
set_tests_properties ( PhaseHeatTris PROPERTIES PASS_REGULAR_EXPRESSION " PhaseHeatTris: P" )

add_test( PhaseHeatQuadPar ${CMAKE_CURRENT_SOURCE_DIR}/compare )
set_tests_properties ( PhaseHeatQuadPar PROPERTIES PASS_REGULAR_EXPRESSION " PhaseHeatQuadPar: P" )

add_test( PhaseHeatQuadParNoPrec ${CMAKE_CURRENT_SOURCE_DIR}/compare )
set_tests_properties ( PhaseHeatQuadParNoPrec PROPERTIES PASS_REGULAR_EXPRESSION " PhaseHeatQuadParNoPrec: P" )
