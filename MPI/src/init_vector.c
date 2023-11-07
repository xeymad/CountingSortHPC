/* 
 * Course: High Performance Computing A.A. 2021/2022
 * 
 * Lecturer: Moscato Francesco 	fmoscato@unisa.it
 *
 * Group:
 * Carratù	 Arianna 	0622701696	a.carratu18@studenti.unisa.it               
 * Di Rienzo	 Matteo		0622701818 	m.dirienzo1@studenti.unisa.it
 * Gambardella	 Giuseppe 	0622701666 	g.gambardella23@studenti.unisa.it
 * 
 * Assignment: CONTEST-MPI: 3 Students GROUPS
 *
 * Parallelize and Evaluate Performances of "COUNTING SORT" Algorithm, by using MPI.
 *
 * Copyright (C) 2022 - All Rights Reserved
 *
 * This file is part of Assignment2.
 *
 * Assignment2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Assignment2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Assignment2.  If not, see <http://www.gnu.org/licenses/>.
 */
 
/**
	@file init_vector.c
*/
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "../include/utils.h"

#define FILE_A "unordered_v.bin"
#define ELEMENT_TYPE int


int main(int argc, char**argv){
	char *file_name = FILE_A; 
	if(argc < 2 || argc > 3){
		fprintf(stderr,"Wrong param number\n"
		"Usage: %s array_size (file_name)\n",argv[0]);
		exit(EXIT_FAILURE);
	}

	if(argc == 3)
		file_name = argv[2];
	
	int *array;
	int array_size = atoi(argv[1]);
	assert(array_size>0);
    
    //Generates and array
	generateArrayRange(&array,array_size, 50, -50);
	FILE*file_a;
    
    //Creates a file
	file_a = fopen(file_name,"w");
    assert(file_a != NULL);    
    
    //Writes the array in the file
	fwrite(array,sizeof(int),array_size,file_a);

	fclose(file_a);
	exit(EXIT_SUCCESS);
}
