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
	@file main_seq.c
*/

// C program to implement the Counting Sort
// Algorithm using MPI
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include "../include/utils.h"

//#define FNAME "unordered_v.bin"

int main(int argc, char* argv[])
{
    if (argc != 2) {
		printf("Desired number of arguments are not their "
			"in argv....\n");
		printf("2 files required first one input and "
			"second one output....\n");
		exit(-1);
	}

    int array_size = atoi(argv[1]);
    double read_time = 0.0, global_counting_time = 0.0, global_elapsed =0.0;
    int* vet;
    char* fname = "unordered_v.bin";
    
    //Start the timer for reading 
    STARTTIME(1);
    readIntArrayFromFile(fname, &vet, array_size);
    //printIntArray(vet,array_size);
    //Stops the timer    
    ENDTIME(1, read_time);
    
    //Start the timer for the sorting
    STARTTIME(2);
	counting_sort(vet,array_size);
    //Stops the timer  
	ENDTIME(2, global_counting_time);

    //Total time
    global_elapsed = global_counting_time + read_time;
    
    //printIntArray(vet,array_size);
    printf("%d,1,%f,%f,%f\n",array_size, read_time, global_counting_time, global_elapsed);

    //Deallocation
    destroyArray(vet);
    return 0;
}
