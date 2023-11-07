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
	@file countingsort.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "countingsort.h"

/**
 * @brief This function calculates the maximum and minimum of the array passed as an argument.
 *
 * @param a        pointer to the array used in the counting sort on which to calculate the minimum and maximum.
 * @param len      array size.
 * @param max      pointer to the variable used to store the maximum.
 * @param min	   pointer to the variable used to store the minimum.
 */
void maxmin(ELEMENT_TYPE*a, int len,ELEMENT_TYPE *max, ELEMENT_TYPE *min){
	if(len<1) 
		return;
	*max=a[0]; *min=a[0];
	ELEMENT_TYPE local_min,local_max,i;
	#pragma omp parallel default(none) private(i,local_min,local_max) shared(len,a,max,min)
	{
		local_max=a[0];
		local_min=a[0];
		#pragma omp for 
		for(i=1;i<len;i++){
			if(a[i]<local_min)
				local_min=a[i];
			if(a[i]>local_max)
				local_max=a[i];
		}
		#pragma omp critical
		{
			if(local_min<*min)
				*min=local_min;
			if(local_max>*max)
				*max=local_max;
		}
	}
}

/**
 * @brief This function sorts the array 'a' by implementing the counting sort algorithm.
 * @param a           pointer to the array to be sorted.
 * @param len         array size. 
 * @see   https://it.wikipedia.org/wiki/Counting_sort
 */

void counting_sort(ELEMENT_TYPE* a, int len){
    if(len<1 || a==NULL) 
		return;
    ELEMENT_TYPE min,max;
    //trova il minimo e il massimo del vettore
    maxmin(a,len,&max,&min);

    int lenc = (max-min+1);
    ELEMENT_TYPE *c = (ELEMENT_TYPE*)malloc(lenc* sizeof(ELEMENT_TYPE));
        if (c == NULL)
		perror("Memory Allocation - c");
    //inizializza a zero gli elementi di C
    int t;
    #pragma omp parallel for default(none)  shared(lenc,c) private(t)
    for(t =0; t< lenc; t++)
		c[t] = 0;
    	
    //aumenta il numero di volte che si e' incontrato il valore
    int q;
    #pragma omp parallel for default(none) shared(a,c,len,min) private(q)  
    for(q=0; q< len; q++)
    	c[a[q] - min] = c[a[q] - min] + 1;
	
	int r;
	int k = 0;		 //indice per l'array A
	//#pragma omp parallel for shared(lenc) private(a,c,r) num_threads(threads)
	for(r =0; r< lenc; r++)
    	for(;c[r] > 0;c[r]--)
    		a[k++] = r + min;		//scrive C[i] volte il valore (i + min) nell'array A
    free(c);
}


/**
 * @brief This function generates an array of the given size.
 * @param v           pointer to the array used to store the generated array.
 * @param len         array size. 
 */

void generate(ELEMENT_TYPE** v, int len) {
    ELEMENT_TYPE *a= (ELEMENT_TYPE*)malloc(len* sizeof(ELEMENT_TYPE));
    if (a == NULL)
		perror("Memory Allocation - a");
    int i;
    #pragma omp parallel for default(none) shared(len,a) private(i)  
    for (i = 0; i < len; i++) {
        a[i] = len-i;
    }
    *v=a;
}
