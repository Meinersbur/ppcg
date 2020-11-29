#include <assert.h>
#include <stdio.h>
#include "test-reduce_kernel.hu"
#include <stdio.h>
#include <stdlib.h>

#define nn 10000
#define MC(x,y) x*x*x*x*x*x*x+y*y*y*y*y*y*y

void test(int n, int A[nn][nn], int B[nn][nn]){
    // int answer=0;
    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    if (n >= 2) {
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)

      int *dev_A;
      int *dev_B;
      
      cudaCheckReturn(cudaMalloc((void **) &dev_A, (n) * (10000) * sizeof(int)));
      cudaCheckReturn(cudaMalloc((void **) &dev_B, (n) * (10000) * sizeof(int)));
      
      cudaCheckReturn(cudaMemcpy(dev_A, A, (n) * (10000) * sizeof(int), cudaMemcpyHostToDevice));
      cudaCheckReturn(cudaMemcpy(dev_B, B, (n) * (10000) * sizeof(int), cudaMemcpyHostToDevice));
      {
        dim3 k0_dimBlock(16, 32);
        dim3 k0_dimGrid(ppcg_min(256, (n + 31) / 32), ppcg_min(256, (n + 31) / 32));
        kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, dev_B, n);
        cudaCheckKernel();
      }
      
      cudaCheckReturn(cudaMemcpy(B, dev_B, (n) * (10000) * sizeof(int), cudaMemcpyDeviceToHost));
      cudaCheckReturn(cudaFree(dev_A));
      cudaCheckReturn(cudaFree(dev_B));
    }
    // return answer;
}

int main(){
    int** testA = (int**)malloc(sizeof(int*)*nn);
    for(int i=0; i<nn; i++){
        testA[i] = (int*)malloc(sizeof(int)*nn);
    }
    int** testB = (int**)malloc(sizeof(int*)*nn);
    for(int i=0; i<nn; i++){
        testB[i] = (int*)malloc(sizeof(int)*nn);
    }
    for(int i=0; i<nn; i++){
        for(int j=0; j<nn; j++){
           testA[i][j] = 2;
        }
    }
    // int ans = test(nn, testA);
    // for(int tt=0;tt<100; tt++){
        test(nn, (int(*)[nn])testA, (int(*)[nn])testB);
    // }
    printf("%d\n", testB[nn-1][nn-1]);
    return 0;
}