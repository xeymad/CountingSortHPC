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
	@file main_counting.c
*/

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <assert.h>
#include "../include/utils.h"
#include <string.h>

#define DEFAULT_INFILE "unordered_v.bin"
#define DEFAULT_OUTFILE "ordered_v.bin"


/**
 * @brief This function implements the vectorial sum between two vectors. It is used as a MPI_Operation
 *
 * @param invec    first array to be summed
 * @param inoutvec second array to be summed, it also is the output of the sum
 * @param len      lenght of the arrays
 * @param dtype	   MPI_Datatype, required for MPI_Operation
 */
void sumVectors(int *invec, int *inoutvec, int *len, MPI_Datatype *dtype)
{
    for (int i=0; i<*len; i++ ) 
        inoutvec[i] += invec[i];
}

int main(int argc,char **argv){
    /*                  DECLARATIONS                    */
    double  read_time_start, read_time_stop, read_time, global_read_time;
    double counting_time_start, counting_time_stop, counting_time, global_counting_time;
    double global_elapsed;
    int *array;
    int *chunk;
    int min,max;
    int size, rank;
    int c_len;
	int chunk_size;
    int* global_counting;
    char *in_file_name = DEFAULT_INFILE;
    char *out_file_name = DEFAULT_OUTFILE;
    MPI_File fp;

    //Initialize the MPI Environment
    int rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS) {
		fprintf(stderr,"Error in creating MPI program.\nTerminating...\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
        exit(EXIT_FAILURE);
	}
    //Get size and rank
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    //Checks command line arguments
    if ((argc < 2  || argc>4) && rank==0) {
		fprintf(stderr,"Usage: %s array_size (input_file_name) (output_file_name)\n",argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 0);
		exit(EXIT_FAILURE);
	}
    if(argc==3)
        in_file_name = argv[2];
    if(argc==4){
        in_file_name = argv[2];
        out_file_name = argv[3];
    }
    if(access(in_file_name,F_OK) && rank==0){
        fprintf(stderr,"No input file detected.\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
        exit(EXIT_FAILURE);
    }
    //Read array size and check if is greater than 0
    int array_size = atoi(argv[1]);
    if(array_size<=0 && rank==0){
        fprintf(stderr,"array_size must be a nonnegative integer.\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
        exit(EXIT_FAILURE);
    }
    //Wait to all checks before start the application
    MPI_Barrier(MPI_COMM_WORLD);

    //Calculate chunk size for each process
    chunk_size = (int) array_size / size;
    int own_chunk_size = rank==size-1 ? array_size-rank*chunk_size : chunk_size;

	//Chunck allocation for each process
	chunk = (int *)malloc(own_chunk_size * sizeof(int));

    //Array Reading
    read_time_start = MPI_Wtime();
    MPI_File_open(MPI_COMM_WORLD, in_file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);

    //Multiple versions of file reading
#if VERSION == 1
    MPI_Datatype chunk_v;
	MPI_Type_contiguous(own_chunk_size,MPI_INT,&chunk_v);
	MPI_Type_commit(&chunk_v);
	MPI_File_read_ordered(fp,chunk,1,chunk_v,MPI_STATUS_IGNORE);
#elif VERSION == 2
    MPI_Datatype chunk_v;
	MPI_Type_contiguous(own_chunk_size,MPI_INT,&chunk_v);
	MPI_Type_commit(&chunk_v);
	int displacement = rank * chunk_size * sizeof(int);
	MPI_File_seek(fp, displacement, MPI_SEEK_SET);
	MPI_File_read_all(fp, chunk, 1, chunk_v, MPI_STATUS_IGNORE);
#elif VERSION == 3
	int displacement = rank * chunk_size * sizeof(int);
	MPI_File_set_view(fp, displacement, MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
	MPI_File_read_all(fp, chunk, own_chunk_size, MPI_INT, MPI_STATUS_IGNORE);
#elif VERSION == 4
    MPI_Datatype chunk_v;
	MPI_Type_contiguous(own_chunk_size,MPI_INT,&chunk_v);
	MPI_Type_commit(&chunk_v);
	int displacement = rank * chunk_size * sizeof(int);
	MPI_File_set_view(fp, displacement, MPI_BYTE, chunk_v, "native", MPI_INFO_NULL);
	MPI_File_read(fp, chunk, 1, chunk_v, MPI_STATUS_IGNORE);
#endif
    MPI_File_close(&fp);
    read_time_stop = MPI_Wtime();
    read_time = read_time_stop- read_time_start;

    //Counting sort algorithm
    counting_time_start = MPI_Wtime();
    //Calculating local max and local min for each chunk    
    int local_max = INT_MIN,local_min = INT_MAX;
    if(own_chunk_size>0)
        maxmin(chunk,own_chunk_size,&local_max,&local_min);

    //Calculating global max and global min through a reduction
    MPI_Allreduce(&local_min,&min,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(&local_max,&max,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    c_len = max - min + 1;
    
    int *local_counting = (int *)malloc(c_len*sizeof(int));
    for(int i = 0; i<c_len; i++)
        local_counting[i]=0;

    //Each process computes its own local_counting
    for(int i = 0; i<own_chunk_size; i++)
        local_counting[chunk[i]- min] = local_counting[chunk[i] - min] + 1;

    //Computes the sum of each local_counting and sends it to rank 0
    if(rank == 0)
        global_counting = (int *)malloc(c_len*sizeof(int));

    // Create vectorial sum operation for later reduction
    MPI_Op vect_sum;
    MPI_Op_create((MPI_User_function *)sumVectors, 1, &vect_sum );
    MPI_Reduce(local_counting,global_counting,c_len,MPI_INT,vect_sum,0, MPI_COMM_WORLD);
    
    //Orders the array
    if(rank==0){
        array = (int*)malloc(array_size*sizeof(int));
        int r;
        int k = array_size-1;		
        for(int r = c_len-1; r >= 0; r--)
            for(;global_counting[r] > 0;global_counting[r]--)
              array[k--] = r + min;
    }

    //Stops the time for the counting sort algorithm
    counting_time_stop = MPI_Wtime();
    counting_time = counting_time_stop - counting_time_start;
    //Takes the max time for each process
    MPI_Reduce(&counting_time,&global_counting_time,1,MPI_DOUBLE,MPI_MAX,0, MPI_COMM_WORLD);
    MPI_Reduce(&read_time, &global_read_time, 1 ,MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if(rank == 0){
        global_elapsed = global_counting_time + global_read_time;
        //printIntArray(array, array_size);
    
    //Save the array in output file if required
    #ifndef NO_OUTPUT        
          FILE *out = fopen(out_file_name,"w");
          fwrite(array,sizeof(int),array_size,out);
    #endif
        destroyArray(global_counting);        
        destroyArray(array);
		printf("%d,%d,%f,%f,%f\n",array_size, size, global_read_time, global_counting_time, global_elapsed);
    }

    //Deallocation
    destroyArray(chunk);
    destroyArray(local_counting);
   
    MPI_Op_free(&vect_sum);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
