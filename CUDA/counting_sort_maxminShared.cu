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
#define THREADxBLOCK 128

/*
* NVIDIA GEFORCE 750 Ti
* COMPUTE CAPABILITY 5.3 MAXWELL
* Max Threads / SM: 2048
* Max Threads Blocks / SM: 32
* FP32 CUDA CORES: 3072
*/

void HandleError( cudaError_t err )
{
  if(err != cudaSuccess)  {
    fprintf(stderr,"%s\n",cudaGetErrorString(err)) ;
    exit(1);
  }
}

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

__global__ void kernelMaxMin(int *array,int array_size,int *maxmin_out){
  // BANK CONFLICTS
  extern __shared__ int sdata[];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  int limit = array_size - blockIdx.x * blockDim.x;
  sdata[tid] = array[i];
  __syncthreads();
  int index = tid*2, next_index = index+1;
  if(index< blockDim.x && next_index<limit && sdata[index]<sdata[next_index]){
    int swap = sdata[index];
    sdata[index] = sdata[next_index];
    sdata[next_index] = swap;
  }
  __syncthreads();
  for (unsigned int s=2; s < blockDim.x; s *= 2) {
    index = 2 * s * tid;
    next_index = index + 1;
    if (index < blockDim.x && next_index+s<limit) {
    //Do maxmin reduction of first two elements
      if(sdata[index]<sdata[index+s])
        sdata[index] = sdata[index+s];
      if(sdata[next_index]>sdata[next_index+s])
        sdata[next_index] = sdata[next_index+s];
    }
    __syncthreads();
  }
  // Write the result to global memory
  if (tid==0){
    int offset = blockIdx.x*2;
    maxmin_out[offset]=sdata[0];
    maxmin_out[offset+1]=sdata[1];
  }
}

__global__ void kernelCountingArraySum(int *unordered_array, int *counting_array,int array_size, int min) {
  extern __shared__ int sdata[];
  // index for linear array
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  // copy into shared mem
  sdata[tid] = unordered_array[i];
  //__syncthreads();
  if(i<array_size){
    // Increment counting array
    // More info: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd
    atomicAdd(&counting_array[sdata[tid] - min],1);
  }
  // // assuming all elements are different.
  // counting_array[unordered_array[i] - min]++;
}

void deviceCountingSort(int *array, int array_size) {
  cudaEvent_t start, stop;
  float gflops;

  int max,min;
  int *deviceArray;
  int gridsize = array_size/THREADxBLOCK;
  if(gridsize*THREADxBLOCK<array_size) {
    gridsize=gridsize+1;
  }
  int allocation = array_size%2==0?array_size:array_size+1;
  dim3 dimGrid(gridsize,1);
  dim3 dimBlock(THREADxBLOCK, 1);
  printf("gridsize:%d dimblock:%d\n",gridsize,dimBlock.x);
  int size = sizeof(int);
  int *dev_maxmin_out;
  int *maxmin_out = (int*)malloc(gridsize*2*size);
  assert(maxmin_out!=NULL);
  printf("The allocation is %d\n",allocation);
  cudaMalloc(&deviceArray,allocation*size);
  HandleError(cudaMemcpy(deviceArray,array,array_size*size,cudaMemcpyHostToDevice));
  int delta = allocation - array_size;
  if(allocation>array_size){
    // Copy array[0] to the last position
    HandleError(cudaMemcpy(deviceArray+array_size,&array[0],delta*size,cudaMemcpyHostToDevice));
  }
  cudaMalloc(&dev_maxmin_out,gridsize*2*size);
  // Create start and stop CUDA events 
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // Launch maxmin kernel
  kernelMaxMin<<<dimGrid,dimBlock,THREADxBLOCK*size>>>(deviceArray,allocation,dev_maxmin_out);
  // event record, synchronization, elapsed time and destruction
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed_maxmin;
  cudaEventElapsedTime(&elapsed_maxmin, start, stop);
  elapsed_maxmin = elapsed_maxmin/1000.f; // convert to seconds
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // calculate Gflops
  printf("kernelMaxMin elapsed time %fs \n", elapsed_maxmin);
  gflops = 1.0e-9*180*THREADxBLOCK/elapsed_maxmin;
  printf("Gflops MaxMin: %.4f\n", gflops);
  HandleError(cudaMemcpy(maxmin_out,dev_maxmin_out,gridsize*2*size,cudaMemcpyDeviceToHost));
  cudaFree(dev_maxmin_out);
  maxmin(maxmin_out,gridsize*2,&max,&min);
  free(maxmin_out);
  printf("CPU max is %d and min %d\n",max,min);
  // Create counting_array
  int c_len = max-min+1;
  int *counting_array = (int*)malloc(c_len*size);
  assert(counting_array!=NULL);
  int *dev_counting_array;
  // allocate dev_counting_array in GPU. 
  cudaMalloc(&dev_counting_array, c_len*size);
  cudaMemset(dev_counting_array, 0, c_len*size);
  // cudaGetLastError call to reset previous CUDA errors
  HandleError(cudaGetLastError());
  // Create start and stop CUDA events 
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // kernel launch
  kernelCountingArraySum<<<dimGrid, dimBlock, THREADxBLOCK*size>>>(deviceArray, dev_counting_array,allocation,min);
  // device synchronization and cudaGetLastError call
  HandleError(cudaGetLastError());
  // event record, synchronization, elapsed time and destruction
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed;
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed = elapsed/1000.f; // convert to seconds
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // calculate Gflops
  printf("kernelCountingArraySum elapsed time %fs \n", elapsed);
  gflops = 1.0e-9*6*THREADxBLOCK/elapsed;
  printf("Gflops CountingArraySum: %.4f\n", gflops);
  // copy back results from device
  cudaMemcpy(counting_array, dev_counting_array, c_len*sizeof(int), cudaMemcpyDeviceToHost);
  // free memory on device
  cudaFree(deviceArray);
  cudaFree(dev_counting_array);
  // if allocation is greater then array_size decrement last element.
  if(allocation>array_size)
    counting_array[array[0]-min]--;
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
  //printIntArray(ordered_array_GPU,array_size);
  assert(intArrayEquals(unordered_array,array_size,ordered_array_GPU,array_size));
  printf("Test passed.\n");
  return EXIT_SUCCESS;
}
