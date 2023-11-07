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
	@file test.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "countingsort.h"

#ifdef _OPENMP
	#include <omp.h>
#else 
    #define omp_set_num_threads(threads) ;
#endif


/**
 * @brief This function checks if the two arrays are equal.
 *
 * @param a        pointer to the first array to check.
 * @param b        pointer to the second array to check.
 * @param len      array size.
 */
int int_array_equals(ELEMENT_TYPE *a, ELEMENT_TYPE *b, int len){
    int i;
    for(i=0;i<len;i++)
        if(a[i]!=b[i])
            return 0;
    return 1;
}

int main(int argc,char **argv){
    /*                  DECLARATIONS                    */

    int array_size=30;
    omp_set_num_threads(4);

    ELEMENT_TYPE a1[]={342,280,186,359,294,116,295,92,225,
    260,-326,-186,259,22,126,350,-70,153,108,-188,
    362,-130,-324,-59,17,-270,111,-122,-275,-348};

    ELEMENT_TYPE a2[]={-220,396,362,-315,39,-322,-245,-137,
    -274,-334,201,313,-249,-246,250,363,-29,-163,
    -365,-285,245,7,54,-40,-305,239,63,-11,-221,-84};

    ELEMENT_TYPE a3[]={-31,149,392,-166,-240,32,-38,301,391,
    -249,228,125,234,-231,375,-138,-138,-44,-206,304,
    -146,-31,235,368,343,-138,317,-288,383,136};

    ELEMENT_TYPE a4[]={62,302,222,-305,203,131,317,142,-185,
    64,-112,398,-377,354,347,-302,59,-189,21,198,
    -392,-338,-281,-186,147,-269,211,118,197,99};

    ELEMENT_TYPE a5[]={183,-277,-266,-107,368,22,65,-218,-371,
    319,325,-14,388,-181,77,207,190,-240,198,248,
    97,380,104,-204,-149,-126,2,381,-350,236};

    ELEMENT_TYPE expected_result1[] = {-348,-326,-324,-275,-270,-188,
    -186,-130,-122,-70,-59,17,22,92,108,111,116,126,153,186,
    225,259,260,280,294,295,342,350,359,362};
    
    ELEMENT_TYPE expected_result2[] = {-365,-334,-322,-315,-305,-285,
    -274,-249,-246,-245,-221,-220,-163,-137,-84,-40,-29,
    -11,7,39,54,63,201,239,245,250,313,362,363,396};

    ELEMENT_TYPE expected_result3[] = {-288,-249,-240,-231,-206,-166,
    -146,-138,-138,-138,-44,-38,-31,-31,32,125,136,149,228,234,235,
    301,304,317,343,368,375,383,391,392};

    ELEMENT_TYPE expected_result4[] = {-392,-377,-338,-305,-302,-281,
    -269,-189,-186,-185,-112,21,59,62,64,99,118,131,142,147,
    197,198,203,211,222,302,317,347,354,398};

    ELEMENT_TYPE expected_result5[] = {-371,-350,-277,-266,-240,-218,
    -204,-181,-149,-126,-107,-14,2,22,65,77,97,104,183,190,
    198,207,236,248,319,325,368,380,381,388};

    /*                      TESTING                          */
    counting_sort(a1,array_size);
    counting_sort(a2,array_size);
    counting_sort(a3,array_size);
    counting_sort(a4,array_size);
    counting_sort(a5,array_size);

    assert(int_array_equals(a1,expected_result1,array_size));

    assert(int_array_equals(a2,expected_result2,array_size));

    assert(int_array_equals(a3,expected_result3,array_size));

    assert(int_array_equals(a4,expected_result4,array_size));

    assert(int_array_equals(a5,expected_result5,array_size));
}
