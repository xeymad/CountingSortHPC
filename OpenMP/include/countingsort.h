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

#include<time.h>

//#define ELEMENT_TYPE unsigned long long int
/* Token concatenation used */
#define STARTTIME(id)                           \
  clock_t start_time_42_##id, end_time_42_##id; \
  start_time_42_##id = clock()

#define ENDTIME(id, x)        \
  end_time_42_##id = clock(); \
  x = ((double)(end_time_42_##id - start_time_42_##id)) / CLOCKS_PER_SEC
/** macros to get execution time: both macros have to be in the same scope
*   define a double variable to use in ENDTIME before STARTTIME:
*   double x;
*   the variable will hold the execution time in seconds.
*/
void counting_sort(ELEMENT_TYPE*,  int);
void generate(ELEMENT_TYPE**, int);
void maxmin(ELEMENT_TYPE*, int ,ELEMENT_TYPE *, ELEMENT_TYPE *);

