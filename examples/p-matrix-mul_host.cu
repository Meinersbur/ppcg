#include <assert.h>
#include <stdio.h>
#include "p-matrix-mul_kernel.hu"
#include <stdio.h>
#include <stdlib.h>
#define MSIZE 100
#define SIZE MSIZE*MSIZE

#define get(a,x,y) a[x*MSIZE+y]

void mulMartix(int nn, int mat1[SIZE], int mat2[SIZE], int answer[SIZE]){
    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    if (nn >= 1) {
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

      int *dev_answer;
      int *dev_mat1;
      int *dev_mat2;
      
      cudaCheckReturn(cudaMalloc((void **) &dev_answer, (101 * nn - 100) * sizeof(int)));
      cudaCheckReturn(cudaMalloc((void **) &dev_mat1, (101 * nn - 100) * sizeof(int)));
      cudaCheckReturn(cudaMalloc((void **) &dev_mat2, (101 * nn - 100) * sizeof(int)));
      
      cudaCheckReturn(cudaMemcpy(dev_mat1, mat1, (101 * nn - 100) * sizeof(int), cudaMemcpyHostToDevice));
      cudaCheckReturn(cudaMemcpy(dev_mat2, mat2, (101 * nn - 100) * sizeof(int), cudaMemcpyHostToDevice));
      {
        dim3 k0_dimBlock(32);
        dim3 k0_dimGrid(ppcg_min(32768, 3 * nn + (5 * nn + 27) / 32 - 3));
        kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_answer, dev_mat1, dev_mat2, nn);
        cudaCheckKernel();
      }
      
      cudaCheckReturn(cudaMemcpy(answer, dev_answer, (101 * nn - 100) * sizeof(int), cudaMemcpyDeviceToHost));
      cudaCheckReturn(cudaFree(dev_answer));
      cudaCheckReturn(cudaFree(dev_mat1));
      cudaCheckReturn(cudaFree(dev_mat2));
    }
}

int main(){
    int* mat1, *mat2, *result;
    // 用一位数组表示二维矩阵
    mat1 = (int*) malloc(MSIZE * MSIZE * sizeof(int));
    mat2 = (int*) malloc(MSIZE * MSIZE * sizeof(int));
    result = (int*) malloc(MSIZE * MSIZE * sizeof(int));

    // initialize
    for (int i = 0; i < MSIZE * MSIZE; i++) {
        mat1[i] = rand()%10000;
        mat2[i] = rand()%10000;
        result[i] = 0;  
    }   

    mulMartix(MSIZE, mat1, mat2, result);

    printf("final: %d\n", get(result, MSIZE-1, MSIZE-1));
    return 0;
}