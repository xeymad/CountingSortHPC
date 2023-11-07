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
 * Assignment: CONTEST-OMP: 3 Students GROUPS
 *
 * Parallelize and Evaluate Performances of "COUNTING SORT" Algorithm, by using OpenMP.
 *
 * Copyright (C) 2021 - All Rights Reserved
 *
 * This file is part of Assignment1.
 *
 * Assignment1 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Assignment1 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Assignment1.  If not, see <http://www.gnu.org/licenses/>.
 */
 
 /**
	@file main.c
*/

#include"countingsort.h"
#include<stdio.h>
#include<stdlib.h>


#ifdef _OPENMP
	#include <omp.h>
#else 
    #define omp_set_num_threads(threads) ;
#endif


int main(int argc,char** argv){

	if((argc < 3)){
		printf("ERROR! Usage: ./main array_size threads");	
		exit(1);
	}

	int array_size = atoi(argv[1]);
	int threads = atoi(argv[2]);
    omp_set_num_threads(threads);
    ELEMENT_TYPE *a;

	double time_sort = 0.0, time_init = 0.0;

	STARTTIME(1);
	generate(&a,array_size);
	ENDTIME(1, time_init);
    
	STARTTIME(2);
	counting_sort(a,array_size);
	ENDTIME(2, time_sort);

    

    printf("%d;%d;%f;%f\n", array_size, threads, time_init, time_sort);
	
	free(a);
	return 0;
}
