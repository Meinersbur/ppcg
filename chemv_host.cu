#include <assert.h>
#include <stdio.h>
#include "chemv_kernel.hu"
/*
 * Copyright 2014      ARM Ltd.
 *
 * Use of this software is governed by the MIT license
 */

#include <stdio.h>
#include <stdlib.h>

struct ComplexFloat
{
	float Re;
	float Im;
};

/* chemv - complex hermitian matrix-vector multiplication
 * The function body was taken from a VOBLA-generated BLAS library.
 */
void chemv(int n, float alpha_re, float alpha_im,
	int ldAT, struct ComplexFloat AT[restrict const static n][ldAT],
	int incX, struct ComplexFloat X[restrict const static n][incX],
	float beta_re, float beta_im,
	int incY, struct ComplexFloat Y[restrict const static n][incY])
{
	#define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
	if (n >= 1) {
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

	  struct ComplexFloat *dev_AT;
	  struct ComplexFloat *dev_X;
	  struct ComplexFloat *dev_Y;
	  
	  cudaCheckReturn(cudaMalloc((void **) &dev_AT, (ppcg_min(n, ldAT)) * (ldAT) * sizeof(struct ComplexFloat)));
	  cudaCheckReturn(cudaMalloc((void **) &dev_X, (n) * (incX) * sizeof(struct ComplexFloat)));
	  cudaCheckReturn(cudaMalloc((void **) &dev_Y, (n) * (incY) * sizeof(struct ComplexFloat)));
	  
	  if (ldAT >= n + 1 || ldAT >= 1)
	    cudaCheckReturn(cudaMemcpy(dev_AT, AT, (ppcg_min(n, ldAT)) * (ldAT) * sizeof(struct ComplexFloat), cudaMemcpyHostToDevice));
	  if (incX >= 1)
	    cudaCheckReturn(cudaMemcpy(dev_X, X, (n) * (incX) * sizeof(struct ComplexFloat), cudaMemcpyHostToDevice));
	  if (incY >= 1)
	    cudaCheckReturn(cudaMemcpy(dev_Y, Y, (n) * (incY) * sizeof(struct ComplexFloat), cudaMemcpyHostToDevice));
	  {
	    dim3 k0_dimBlock(32);
	    dim3 k0_dimGrid(ppcg_min(32768, (n + 31) / 32));
	    kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_AT, dev_X, dev_Y, alpha_im, alpha_re, beta_im, beta_re, incY, n, incX, ldAT);
	    cudaCheckKernel();
	  }
	  
	  if (incY >= 1)
	    cudaCheckReturn(cudaMemcpy(Y, dev_Y, (n) * (incY) * sizeof(struct ComplexFloat), cudaMemcpyDeviceToHost));
	  cudaCheckReturn(cudaFree(dev_AT));
	  cudaCheckReturn(cudaFree(dev_X));
	  cudaCheckReturn(cudaFree(dev_Y));
	}
}

int main()
{
	const int n = 37;
	const int incX = 1;
	const int incY = 1;
	const int ldAT = n;
	struct ComplexFloat AT[n][ldAT];
	struct ComplexFloat X[n][incX];
	struct ComplexFloat Y[n][incY];

	for (int i = 0; i < n; i++) {
		X[i][0] = (struct ComplexFloat){i + 5, i * 2};
		Y[i][0] = (struct ComplexFloat){i * 3, i + 7};
		for (int j = 0; j < ldAT; j++) {
			AT[i][j] = (struct ComplexFloat){i + j, i + 3};
		}
	}

	chemv(n, 3.14f, 1.59f, ldAT, AT, incX, X, 2.71f, 8.28f, incY, Y);

	for (int i = 0; i < n; i++)
		printf("%0.2f %0.2f\n", Y[i][0].Re, Y[i][0].Im);

	return EXIT_SUCCESS;
}
