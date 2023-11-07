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
	@file test.c
*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <assert.h>
#include "../include/utils.h"
#define N 10


int intArrayEquals(int *a,int len_a, int *b, int len_b){
    if(len_a!=len_b)
        return 0;
    for(int i=0;i<len_a;i++)
        if(a[i]!=b[i]) return 0;
    return 1;
}

int main(int argc, char**argv){
    char* input_file = "in.bin";
    char* output_file = "out.bin";
    int buffer[N];
    int unordered[] = {-40, -41, -42, -43, -44, -45, -46, -47, -48, -49};
    int ordered[] = {-49, -48, -47, -46, -45, -44, -43, -42, -41, -40};
    //Test init_vect
    int pid = fork();
    assert(pid>=0);
    if(pid==0){
        char string[10];
        snprintf(string, sizeof(string), "%d", N);
        char *init_vector[]={"../build/init_vect",string,input_file,NULL};
        execvp(init_vector[0],init_vector);
    }
    wait(NULL);
    FILE *fp = fopen(input_file,"r");
    assert(fp!=NULL);
    fread(buffer, sizeof(int), sizeof(buffer), fp);
    assert(intArrayEquals(buffer,10,unordered,10));
    printf("init_vect SUCCESS\n");
    fclose(fp);
    //Test mpi_counting_vect
    pid = fork();
    assert(pid>=0);
    if(pid==0){
        char string[10];
        snprintf(string, sizeof(string), "%d", N);
        char *mpi_main_counting[]={"../build/program_test",string,input_file,output_file,NULL};
        execvp(mpi_main_counting[0],mpi_main_counting);
    }
    wait(NULL);
    fp = fopen(output_file,"r");
    assert(fp!=NULL);
    fread(buffer, sizeof(int), sizeof(buffer), fp);
    //printIntArray(buffer,N);
    assert(intArrayEquals(buffer,N,ordered,N));
    printf("mpi_counting_vect SUCCESS\n");
    fclose(fp);
    remove(input_file);
    remove(output_file);

    return EXIT_SUCCESS;
}
