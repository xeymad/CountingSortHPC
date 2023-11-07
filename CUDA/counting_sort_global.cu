/* 
 * Course: High Performance Computing A.A. 2021/2022
 * 
 * Lecturer: Moscato Francesco  fmoscato@unisa.it
 *
 * Group:
 * Carratù  Arianna  0622701696 a.carratu18@studenti.unisa.it               
 * Di Rienzo  Matteo  0622701818  m.dirienzo1@studenti.unisa.it
 * Gambardella  Giuseppe  0622701666  g.gambardella23@studenti.unisa.it
 * 
 * Assignment: CONTEST-CUDA: 3 Students GROUPS
 *
 * Parallelize and Evaluate Performances of "COUNTING SORT" Algorithm, by using CUDA.
 *
 * Copyright (C) 2022 - All Rights Reserved
 *
 * This file is part of Assignment3.
 *
 * Assignment3 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Assignment3 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Assignment3.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <stdio.h>
#include <assert.h>
#define epsilon (float)1e-5
#define DATA float
#define THREADxBLOCK 128

/*
* Nvidia Geforce 750 Ti
* COMPUTE CAPABILITY 5.0 Maxwell
* Max Threads / SM: 2048
* Max Threads Blocks / SM: 32
* FP32 CUDA CORES: 4608
*/

void printIntArray(int* array, int len){
  for(int i=0;i<len;i++)
    printf("%d ",array[i]);
  printf("\n");
}

int intArrayEquals(int *a,int len_a, int *b, int len_b){
    if(len_a!=len_b)
        return 0;
    for(int i=0;i<len_a;i++)
        if(a[i]!=b[i]) return 0;
    return 1;
}

void maxmin(int *array, int len,int *max, int *min){
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

void hostCountingSort(int* a, int len){
  if(len<1 || a==NULL) 
    return;
  int min,max;
  maxmin(a,len,&max,&min);
  int c_len = (max-min+1);
  int *c = (int*)malloc(c_len* sizeof(int));
  assert(c!=NULL);
  int t;
  for(t = 0; t< c_len; t++)
  c[t] = 0;
  int q;
  for(q=0; q< len; q++)
    c[a[q] - min] = c[a[q] - min] + 1;
  int r;
  int k = len-1;		
  for(r = c_len-1; r >= 0; r--)
      for(;c[r] > 0;c[r]--)
        a[k--] = r + min;
  free(c);
}

__global__ void kernelCountingArraySum(int *unordered_array,int array_size, int *counting_array, int min) {
  // index for linear array
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Increment counting array
  // More info: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd
  if(i<array_size)
    atomicAdd(&counting_array[unordered_array[i] - min],1);
  // // assuming all elements are different.
  // counting_array[unordered_array[i] - min]++;
}

void deviceCountingSort(int *array, int array_size) {
  float gflops;
  int *deviceArray;

  // CUDA grid management
  int gridsize = array_size/THREADxBLOCK;
  if(gridsize*THREADxBLOCK<array_size) {
    gridsize=gridsize+1;
  }
  dim3 dimGrid(gridsize,1);
  dim3 dimBlock(array_size<THREADxBLOCK ? array_size:THREADxBLOCK, 1);
  printf("Gridsize: %d\n", gridsize);

  // Find max and min in array
  int min,max;
  maxmin(array,array_size,&max,&min);

  // Create counting_array
  int c_len = (max-min+1);
  int *counting_array = (int*)malloc(c_len*sizeof(int));
  assert(counting_array!=NULL);
  int *dev_counting_array;

  // allocate array in GPU.
  cudaMalloc(&deviceArray, array_size*sizeof(int));
  cudaMemcpy(deviceArray, array, array_size*sizeof(int), cudaMemcpyHostToDevice);
  // allocate dev_counting_array in GPU. 
  cudaMalloc(&dev_counting_array, c_len*sizeof(int));
  cudaMemset(dev_counting_array, 0, c_len*sizeof(int));

  // cudaGetLastError call to reset previous CUDA errors
  cudaError_t mycudaerror;
  mycudaerror = cudaGetLastError();

  // Create start and stop CUDA events 
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // kernel launch
  kernelCountingArraySum<<<dimGrid, dimBlock>>>(deviceArray,array_size, dev_counting_array, min);

  // device synchronization and cudaGetLastError call
  mycudaerror = cudaGetLastError() ;
  if(mycudaerror != cudaSuccess)  {
    fprintf(stderr,"%s\n",cudaGetErrorString(mycudaerror)) ;
    exit(1);
  }

  // event record, synchronization, elapsed time and destruction
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed;
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed = elapsed/1000.f; // convert to seconds
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //Calculate Gflops
  printf("kernelCountingArraySum elapsed time %fs \n", elapsed);
  gflops = 1.0e-9*6*THREADxBLOCK/elapsed;
  printf("Gflops CountingArraySum: %.4f\n", gflops);

  // copy back results from device
  cudaMemcpy(counting_array, dev_counting_array, c_len*sizeof(int), cudaMemcpyDeviceToHost);

  // free memory on device
  cudaFree(deviceArray);
  cudaFree(dev_counting_array);

  // final phase of counting sort algorithm
  int r;
  int k = array_size-1;
  for(r = c_len-1; r >= 0; r--)
    for(;counting_array[r] > 0;counting_array[r]--)
      array[k--] = r + min;
  free(counting_array);
}

// main
int main(int argc, char** argv) {

  int array_size;
  int *unordered_array,*ordered_array_GPU;
  if(argc<2) {
    fprintf(stderr,"Usage: %s array_size\n",argv[0]);
    exit(1);
  }
  array_size=atoi(argv[1]);
  if(array_size<1) {
    fprintf(stderr,"Error array_size=%d, must be > 0\n",array_size);
    exit(1);
  }
  //Initialize the unordered array
  unordered_array = (int *)malloc(array_size*sizeof(int));
  ordered_array_GPU = (int *)malloc(array_size*sizeof(int));
  assert(unordered_array!=NULL);
  //Initialize reversed array
  int i;
  for (i = 0; i < array_size; i++)
    unordered_array[i] = ordered_array_GPU[i] = array_size-i;
  unordered_array[0] = -12;
  ordered_array_GPU[0]= -12;
  hostCountingSort(unordered_array,array_size);
  deviceCountingSort(ordered_array_GPU,array_size);
  assert(intArrayEquals(unordered_array,array_size,ordered_array_GPU,array_size));
  printf("Test passed.\n");
  return EXIT_SUCCESS;
}