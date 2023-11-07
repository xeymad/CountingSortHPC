#
# Course: High Performance Computing 2020/2021
# 
# Lecturer: Francesco Moscato	fmoscato@unisa.it
#
# Group:
# Capitani	Giuseppe	0622701085	g.capitani@studenti.unisa.it               
# Falanga	Armando	0622701140  a.falanga13@studenti.unisa.it 
# Terrone	Luigi		0622701071  l.terrone2@studenti.unisa.it
#
# This file is part of EsameHPC.
#
# CommonAssignment1 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CommonAssignment1 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CommonAssignment1.  If not, see <http://www.gnu.org/licenses/>.
#

cmake_minimum_required(VERSION 3.9)
project(CommonAssignment1
	DESCRIPTION "Matrix dot Vector implementation with OpenMP"
	VERSION 0.1
	LANGUAGES C
	)

enable_testing()

add_executable(program_O1 src/main.c src/countingsort.c include/countingsort.h)
add_executable(program_O2 src/main.c src/countingsort.c include/countingsort.h)
add_executable(program_O3 src/main.c src/countingsort.c include/countingsort.h)
add_executable(program_seq_O0 src/main.c src/countingsort.c include/countingsort.h)
add_executable(program_seq_O1 src/main.c src/countingsort.c include/countingsort.h)
add_executable(program_seq_O2 src/main.c src/countingsort.c include/countingsort.h)
add_executable(program_seq_O3 src/main.c src/countingsort.c include/countingsort.h)

target_include_directories(program_O1 PRIVATE include)
target_include_directories(program_O2 PRIVATE include)
target_include_directories(program_O3 PRIVATE include)
target_include_directories(program_seq_O0 PRIVATE include)
target_include_directories(program_seq_O1 PRIVATE include)
target_include_directories(program_seq_O2 PRIVATE include)
target_include_directories(program_seq_O3 PRIVATE include)

find_package(OpenMP)
if(OpenMP_C_FOUND)
        target_link_libraries(program_O1 PRIVATE OpenMP::OpenMP_C)
        target_link_libraries(program_O2 PRIVATE OpenMP::OpenMP_C)
        target_link_libraries(program_O3 PRIVATE OpenMP::OpenMP_C)
endif()

target_compile_options(program_O1 PRIVATE -O1)
target_compile_options(program_O2 PRIVATE -O2)
target_compile_options(program_O3 PRIVATE -O3)
target_compile_options(program_seq_O0 PRIVATE -O0)
target_compile_options(program_seq_O1 PRIVATE -O1)
target_compile_options(program_seq_O2 PRIVATE -O2)
target_compile_options(program_seq_O3 PRIVATE -O3)

target_compile_features(program_O1 PRIVATE c_std_11)
target_compile_features(program_O2 PRIVATE c_std_11)
target_compile_features(program_O3 PRIVATE c_std_11)
target_compile_features(program_seq_O0 PRIVATE c_std_11)
target_compile_features(program_seq_O1 PRIVATE c_std_11)
target_compile_features(program_seq_O2 PRIVATE c_std_11)
target_compile_features(program_seq_O3 PRIVATE c_std_11)

set(ELEMENT_TYPE int CACHE STRING "Type of the elements array")
target_compile_definitions(program_O1 PRIVATE -DELEMENT_TYPE=${ELEMENT_TYPE})
target_compile_definitions(program_O2 PRIVATE -DELEMENT_TYPE=${ELEMENT_TYPE})
target_compile_definitions(program_O3 PRIVATE -DELEMENT_TYPE=${ELEMENT_TYPE})
target_compile_definitions(program_seq_O0 PRIVATE -DELEMENT_TYPE=${ELEMENT_TYPE})
target_compile_definitions(program_seq_O1 PRIVATE -DELEMENT_TYPE=${ELEMENT_TYPE})
target_compile_definitions(program_seq_O2 PRIVATE -DELEMENT_TYPE=${ELEMENT_TYPE})
target_compile_definitions(program_seq_O3 PRIVATE -DELEMENT_TYPE=${ELEMENT_TYPE})


# ---------------------------------------- TEST -----------------------------------------
add_executable(counting_test test/test.c src/countingsort.c)
target_include_directories(counting_test PRIVATE include)
target_compile_definitions(counting_test PRIVATE -DELEMENT_TYPE=${ELEMENT_TYPE})

add_test(NAME counting_test
	COMMAND counting_test )
# ------------------------------------ DOCUMENTATION ------------------------------------
find_package(Doxygen
             REQUIRED dot)

if(DOXYGEN_FOUND)
	set(DOXYGEN_GENERATE_XML YES)
    set(DOXYGEN_GENERATE_LATEX YES)
	doxygen_add_docs(
		docs
		./src/countingsort.c ./include/countingsort.h ./test/test.c 
		)
endif()

# ---------------------------------- GENERATE MEASURES ----------------------------------
add_custom_target(
	 generate_measures
	 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/measures/measures.bash ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating measures [${CMAKE_CURRENT_SOURCE_DIR}/measures/measures.bash ${CMAKE_CURRENT_BINARY_DIR}]"
    DEPENDS	
		program_O1
		program_O2
		program_O3
		program_seq_O0
		program_seq_O1
		program_seq_O2
		program_seq_O3
)

add_custom_target(
	 change_ref
	 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/measures/measures.bash -p
	 COMMENT "Change graph ref [${CMAKE_CURRENT_SOURCE_DIR}/measures/measures.bash -p]"
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

