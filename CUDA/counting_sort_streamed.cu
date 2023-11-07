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

template <unsigned int blockSize>
__device__ void warpReduceMax(volatile int* sdata, unsigned int tid) {
  if (blockSize >= 64) sdata[tid] = sdata[tid + 32] > sdata[tid] ? sdata[tid + 32] : sdata[tid];
  if (blockSize >= 32) sdata[tid] = sdata[tid + 16] > sdata[tid] ? sdata[tid + 16] : sdata[tid];
  if (blockSize >= 16) sdata[tid] = sdata[tid + 8] > sdata[tid] ? sdata[tid + 8] : sdata[tid];
  if (blockSize >= 8) sdata[tid] = sdata[tid + 4] > sdata[tid] ? sdata[tid + 4] : sdata[tid];
  if (blockSize >= 4) sdata[tid] = sdata[tid + 2] > sdata[tid] ? sdata[tid + 2] : sdata[tid];
  if (blockSize >= 2) sdata[tid] = sdata[tid + 1] > sdata[tid] ? sdata[tid + 1] : sdata[tid];
}

template <unsigned int blockSize>
__global__ void kernelMax(int *g_idata, int *g_odata){
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  sdata[tid] = g_idata[i] > g_idata[i+blockDim.x] ? g_idata[i] : g_idata[i+blockDim.x];
  __syncthreads();
  if (blockSize >= 512) { 
    if (tid < 256)
      sdata[tid] = sdata[tid + 256] > sdata[tid] ? sdata[tid + 256] : sdata[tid]; 
    __syncthreads(); 
  }
  if (blockSize >= 256) { 
    if (tid < 128) 
      sdata[tid] = sdata[tid + 128] > sdata[tid] ? sdata[tid + 128] : sdata[tid]; 
    __syncthreads();
  }
  if (blockSize >= 128) { 
    if (tid < 64)
      sdata[tid] = sdata[tid + 64] > sdata[tid] ? sdata[tid + 64] : sdata[tid]; 
    __syncthreads(); 
  }
  if (tid < 32) 
    warpReduceMax<blockSize>(sdata, tid);
  if (tid == 0) 
    g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__device__ void warpReduceMin(volatile int* sdata, unsigned int tid) {
  if (blockSize >= 64) sdata[tid] = sdata[tid + 32] < sdata[tid] ? sdata[tid + 32] : sdata[tid];
  if (blockSize >= 32) sdata[tid] = sdata[tid + 16] < sdata[tid] ? sdata[tid + 16] : sdata[tid];
  if (blockSize >= 16) sdata[tid] = sdata[tid + 8] < sdata[tid] ? sdata[tid + 8] : sdata[tid];
  if (blockSize >= 8) sdata[tid] = sdata[tid + 4] < sdata[tid] ? sdata[tid + 4] : sdata[tid];
  if (blockSize >= 4) sdata[tid] = sdata[tid + 2] < sdata[tid] ? sdata[tid + 2] : sdata[tid];
  if (blockSize >= 2) sdata[tid] = sdata[tid + 1] < sdata[tid] ? sdata[tid + 1] : sdata[tid];
}

template <unsigned int blockSize>
__global__ void kernelMin(int *g_idata, int *g_odata){
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;  // 3
  sdata[tid] = g_idata[i] < g_idata[i+blockDim.x] ? g_idata[i] : g_idata[i+blockDim.x]; // 5
  __syncthreads();
  if (blockSize >= 512) { 
    if (tid < 256)
      sdata[tid] = sdata[tid + 256] < sdata[tid] ? sdata[tid + 256] : sdata[tid]; // 7
    __syncthreads(); 
  }
  if (blockSize >= 256) { 
    if (tid < 128) 
      sdata[tid] = sdata[tid + 128] < sdata[tid] ? sdata[tid + 128] : sdata[tid]; 
    __syncthreads();
  }
  if (blockSize >= 128) { 
    if (tid < 64)
      sdata[tid] = sdata[tid + 64] < sdata[tid] ? sdata[tid + 64] : sdata[tid]; 
    __syncthreads(); 
  }
  if (tid < 32) 
    warpReduceMin<blockSize>(sdata, tid);
  if (tid == 0) 
    g_odata[blockIdx.x] = sdata[0];
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
  int allocation = gridsize*THREADxBLOCK;
  int even_size = array_size%2==0?array_size:array_size+1;
  dim3 dimGrid(gridsize,1);
  dim3 dimBlockMaxMin(THREADxBLOCK/2, 1);
  dim3 dimBlockCounting(THREADxBLOCK, 1);
  printf("gridsize:%d\n",gridsize);
  int size = sizeof(int);
  int *dev_min_out, *dev_max_out;
  int *max_out = (int*)malloc(gridsize*size);
  assert(max_out!=NULL);
  int *min_out = (int*)malloc(gridsize*size);
  assert(min_out!=NULL);
  printf("The allocation is %d\n",allocation);
  cudaMalloc(&deviceArray,allocation*size);
  HandleError(cudaMemcpy(deviceArray,array,array_size*size,cudaMemcpyHostToDevice));
  int delta = allocation - array_size;
  if(delta>0){
    int *aux = (int*)malloc(delta*size);
    assert(aux!=NULL);
    for(int i=0;i<delta;i++) aux[i]=array[0];
    HandleError(cudaMemcpy(deviceArray+array_size,aux,delta*size,cudaMemcpyHostToDevice));
    free(aux);
  }
  cudaMalloc(&dev_min_out,gridsize*size);
  cudaMalloc(&dev_max_out,gridsize*size);
  // Create start and stop CUDA events 
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // Create streams
  cudaStream_t stream1, stream2;
  cudaStreamCreate (&stream1);
  cudaStreamCreate (&stream2);
  // Execute kernels
  kernelMin<THREADxBLOCK/2><<<dimGrid,dimBlockMaxMin,THREADxBLOCK/2*size,stream1>>>(deviceArray,dev_min_out);
  kernelMax<THREADxBLOCK/2><<<dimGrid,dimBlockMaxMin,THREADxBLOCK/2*size,stream2>>>(deviceArray,dev_max_out);
  // // Synchronize kernels
  // cudaDeviceSynchronize();
  // event record, synchronization, elapsed time and destruction
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed_maxmin;
  cudaEventElapsedTime(&elapsed_maxmin, start, stop);
  elapsed_maxmin = elapsed_maxmin/1000.f; // convert to seconds
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("kernelMin and kernelMax elapsed time %fs \n", elapsed_maxmin);
  gflops = 1.0e-9*15*THREADxBLOCK/elapsed_maxmin;
  printf("Gflops kernelMin: %.4f\nGflops kernelMax: %.4f\n", gflops,gflops);
  HandleError(cudaMemcpyAsync(min_out,dev_min_out,gridsize*size,cudaMemcpyDeviceToHost,stream1));
  HandleError(cudaMemcpyAsync(max_out,dev_max_out,gridsize*size,cudaMemcpyDeviceToHost,stream2));
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaFree(dev_min_out);
  cudaFree(dev_max_out);
  min = min_out[0];
  max = max_out[0];
  for(int i=0;i<gridsize;i++){
      if(min_out[i]<min)
          min=min_out[i];
      if(max_out[i] > max)
          max = max_out[i];
  }
  free(min_out);
  free(max_out);
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
  kernelCountingArraySum<<<dimGrid, dimBlockCounting, THREADxBLOCK*size>>>(deviceArray, dev_counting_array,even_size,min);
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
  //Calculate Gflops
  printf("kernelCountingArraySum elapsed time %fs \n", elapsed);
  gflops = 1.0e-9*6*THREADxBLOCK/elapsed;
  printf("Gflops CountingArraySum: %.4f\n", gflops);
  // copy back results from device
  cudaMemcpy(counting_array, dev_counting_array, c_len*sizeof(int), cudaMemcpyDeviceToHost);
  // free memory on device
  cudaFree(deviceArray);
  cudaFree(dev_counting_array);
  // if was not even then decrement last element.
  if(even_size>array_size)
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
  unordered_array[0] = 12;
  ordered_array_GPU[0]= 12;
  hostCountingSort(unordered_array,array_size);
  deviceCountingSort(ordered_array_GPU,array_size);
  assert(intArrayEquals(unordered_array,array_size,ordered_array_GPU,array_size));
  printf("Test passed.\n");
  return EXIT_SUCCESS;
}
