#
# Course: High Performance Computing 2021/2022
# 
# Lecturer: Francesco Moscato	fmoscato@unisa.it
#
# Group:
# Carratù	 Arianna 	0622701696	a.carratu18@studenti.unisa.it               
# Di Rienzo	 Matteo		0622701818 	m.dirienzo1@studenti.unisa.it
# Gambardella	 Giuseppe 	0622701666 	g.gambardella23@studenti.unisa.it
#
# Assignment: CONTEST-MPI: 3 Students GROUPS
#
# Parallelize and Evaluate Performances of "COUNTING SORT" Algorithm, by using MPI.
#
# Copyright (C) 2022 - All Rights Reserved
#
# This file is part of Assignment2.
#
# Assignment2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Assignment2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Assignment2.  If not, see <http://www.gnu.org/licenses/>.
#

cmake_minimum_required(VERSION 3.9)
project(CommonAssignmentMPI01
	DESCRIPTION "Counting Sort implementation with MPICH"
	VERSION 0.1
	LANGUAGES C
	)

# Generate comile_commands.json for semantic completion in YCM
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# DEBUG flag
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG")

enable_testing()

add_executable(program_parallel_V1 src/main_counting.c src/utils.c include/utils.h)
# Added versions
add_executable(program_parallel_V2 src/main_counting.c src/utils.c include/utils.h)
add_executable(program_parallel_V3 src/main_counting.c src/utils.c include/utils.h)
add_executable(program_parallel_V4 src/main_counting.c src/utils.c include/utils.h)

add_executable(program_seq src/main_seq.c src/utils.c include/utils.h)
add_executable(init_vect src/init_vector.c src/utils.c include/utils.h)
add_executable(program_test src/main_counting.c src/utils.c include/utils.h)


target_include_directories(program_parallel_V1 PRIVATE include)
# Added versions
target_include_directories(program_parallel_V2 PRIVATE include)
target_include_directories(program_parallel_V3 PRIVATE include)
target_include_directories(program_parallel_V4 PRIVATE include)

target_include_directories(program_seq PRIVATE include)
target_include_directories(init_vect PRIVATE include)
target_include_directories(program_test PRIVATE include)

find_package(MPI REQUIRED)
if(MPI_C_FOUND)
		message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
		target_link_libraries(program_parallel_V1 PUBLIC MPI::MPI_C)
		# Added versions
		target_link_libraries(program_parallel_V2 PUBLIC MPI::MPI_C)
		target_link_libraries(program_parallel_V3 PUBLIC MPI::MPI_C)
		target_link_libraries(program_parallel_V4 PUBLIC MPI::MPI_C)

        target_link_libraries(program_test PUBLIC MPI::MPI_C)
endif()

target_compile_options(program_seq PRIVATE -O3)
target_compile_options(program_parallel_V1 PRIVATE -O3)
target_compile_options(program_parallel_V2 PRIVATE -O3)
target_compile_options(program_parallel_V3 PRIVATE -O3)
target_compile_options(program_parallel_V4 PRIVATE -O3)

target_compile_features(program_parallel_V1 PRIVATE c_std_11)
# Added versions
target_compile_features(program_parallel_V2 PRIVATE c_std_11)
target_compile_features(program_parallel_V3 PRIVATE c_std_11)
target_compile_features(program_parallel_V4 PRIVATE c_std_11)

target_compile_features(program_seq PRIVATE c_std_11)
target_compile_features(program_test PRIVATE c_std_11)

target_compile_definitions(program_parallel_V1 PRIVATE -DNO_OUTPUT )
target_compile_definitions(program_parallel_V2 PRIVATE -DNO_OUTPUT )
target_compile_definitions(program_parallel_V3 PRIVATE -DNO_OUTPUT )
target_compile_definitions(program_parallel_V4 PRIVATE -DNO_OUTPUT )

target_compile_definitions(program_parallel_V1 PRIVATE -DVERSION=1 )
target_compile_definitions(program_parallel_V2 PRIVATE -DVERSION=2 )
target_compile_definitions(program_parallel_V3 PRIVATE -DVERSION=3 )
target_compile_definitions(program_parallel_V4 PRIVATE -DVERSION=4 )
target_compile_definitions(program_test PRIVATE -DVERSION=3 )
# target_compile_definitions(program_parallel_V1 PRIVATE -DVERSION=1)

# ---------------------------------------- TEST -----------------------------------------
add_executable(sorting_test test/test.c src/utils.c)
target_include_directories(sorting_test PRIVATE include)

add_test(NAME sorting_test
	COMMAND sorting_test )

# ------------------------------------ DOCUMENTATION ------------------------------------
find_package(Doxygen
             REQUIRED dot)

if(DOXYGEN_FOUND)
	set(DOXYGEN_GENERATE_XML YES)
	doxygen_add_docs(
		docs
		./src/utils.c ./include/utils.h 
		)
endif()

# ---------------------------------- GENERATE MEASURES ----------------------------------
add_custom_target(
	 generate_measures
	 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/measures/measures.bash ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating measures [${CMAKE_CURRENT_SOURCE_DIR}/measures/measures.bash ${CMAKE_CURRENT_BINARY_DIR}]"
    DEPENDS	
		program_parallel_V1
		program_parallel_V2
		program_parallel_V3
		program_parallel_V4
		program_seq
)

# --------------------------------- MEASURES EXTRACTION ---------------------------------
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_program(PipEnv pipenv REQUIRED)

add_custom_target(
	 install_pipenv
	 COMMAND PIPENV_PIPFILE=${CMAKE_CURRENT_SOURCE_DIR}/Pipfile ${PipEnv} install --ignore-pipfile
    COMMENT "Installing pipenv dependencies [${PipEnv} install]"
    DEPENDS	${CMAKE_CURRENT_SOURCE_DIR}/Pipfile.lock
)

add_custom_target(
    extract_measures
    COMMAND PIPENV_PIPFILE=${CMAKE_CURRENT_SOURCE_DIR}/Pipfile ${PipEnv} run python ${CMAKE_CURRENT_SOURCE_DIR}/measures/extract.py
    COMMENT "Generating version header [${PipEnv} run python ${CMAKE_CURRENT_SOURCE_DIR}/measures/extract.py]"
    DEPENDS	
		install_pipenv
		${CMAKE_CURRENT_SOURCE_DIR}/measures/extract.py
)

