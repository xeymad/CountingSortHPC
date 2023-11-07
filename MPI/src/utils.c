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
	@file utils.c
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

#define ELEMENT_TYPE int

/**
 * @brief This function calculates the maximum and minimum of the array passed as an argument.
 *
 * @param array        pointer to the array used in the counting sort on which to calculate the minimum and maximum.
 * @param len      array size.
 * @param max      pointer to the variable used to store the maximum.
 * @param min	   pointer to the variable used to store the minimum.
 */
void maxmin(ELEMENT_TYPE *array, int len,ELEMENT_TYPE *max, ELEMENT_TYPE *min){
	if(len<1)
		return;
    *max = array[0];
    *min = array[0];
    int i;
    for(i = 1 ; i < len; i++){
        if (array[i]>*max)
            *max=array[i];
        if (array[i]<*min)
            *min=array[i];
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
    for(t =0; t< lenc; t++)
		c[t] = 0;
    	
    //aumenta il numero di volte che si e' incontrato il valore
    int q;
    for(q=0; q< len; q++)
    	c[a[q] - min] = c[a[q] - min] + 1;
	
	int r;
	int k = 0;		 //indice per l'array A
	
	for(r =0; r< lenc; r++)
    	for(;c[r] > 0;c[r]--)
    		a[k++] = r + min;		//scrive C[i] volte il valore (i + min) nell'array A
    free(c);
}

/**
 * @brief This function generates a reversed order array of len elements
 * @param v           pointer of the array
 * @param len         array size. 
 */
void generateArray(ELEMENT_TYPE** v, int len) {
    ELEMENT_TYPE *a= (ELEMENT_TYPE*)malloc(len* sizeof(ELEMENT_TYPE));
    if (a == NULL)
		perror("Memory Allocation - a");
    int i;
    for (i = 0; i < len; i++) {
        a[i] = len-i;
    }
    *v=a;
}

/**
 * @brief This function generates an array in which the elements are contained in the range [lower, upper]
 * @param v           pointer of the array
 * @param len         array size. 
 * @param upper       upper value bound
 * @param lower       lower value bound
 */
void generateArrayRange(ELEMENT_TYPE** v, int len, int upper, int lower) {
    ELEMENT_TYPE *a= (ELEMENT_TYPE*)malloc(len* sizeof(ELEMENT_TYPE));
    if (a == NULL)
		perror("Memory Allocation - a");
    int i;
    int k = 0;
	int val = 100;
    for (i = len; i > 0; i--) {
        a[k++] = (i % (upper - lower + 1)) + lower;
    }
    *v=a;
}

/**
 * @brief Destroys the array
* @param v             pointer of the array to destroy
 */
void destroyArray(ELEMENT_TYPE* v){
	free(v);
	v=NULL;
}

/**
 * @brief Prints the array of integer
 * @param array             pointer of array to be printed     
 * @param len               lenght of the array 
 */
void printIntArray(int* array, int len){
  for(int i=0;i<len;i++)
    printf("%d ",array[i]);
  printf("\n");
}

/**
 * @brief Reads an integer array from file
 * @param fname             filename
 * @param v             buffer of the read array   
 * @param len               lenght of the array 
 */
void readIntArrayFromFile(char *fname, ELEMENT_TYPE** v, int len){
	FILE *fp;
	if ((fp=fopen(fname, "r"))==NULL){
		perror("Can't open file");
		exit(1);
	}
	
	ELEMENT_TYPE *a= (ELEMENT_TYPE*)malloc(len* sizeof(ELEMENT_TYPE));
    if (a == NULL)
		perror("Memory Allocation - a");    
	
	int r = fread(a, sizeof(ELEMENT_TYPE), len, fp);
	*v=a;
}
