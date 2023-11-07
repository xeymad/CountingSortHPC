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
	@file utils.h
*/

#include<time.h>
#define STARTTIME(id)                           \
  clock_t start_time_42_##id, end_time_42_##id; \
  start_time_42_##id = clock()

#define ENDTIME(id, x)        \
  end_time_42_##id = clock(); \
  x = ((double)(end_time_42_##id - start_time_42_##id)) / CLOCKS_PER_SEC

#define ELEMENT_TYPE int

void maxmin(ELEMENT_TYPE*a, int len,ELEMENT_TYPE *max, ELEMENT_TYPE *min);
void counting_sort(ELEMENT_TYPE* a, int len);
int* merge(int* arr1, int n1, int* arr2, int n2);
void generateArray(int** v, int len);
void generateArrayRange(ELEMENT_TYPE** v, int len, int upper, int lower); 
void destroyArray(ELEMENT_TYPE*v);
void readIntArrayFromFile(char *, ELEMENT_TYPE**, int);
void printIntArray(int* array, int len);
