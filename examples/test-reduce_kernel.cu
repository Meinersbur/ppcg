#include "test-reduce_kernel.hu"
__global__ void kernel0(int *A, int *B, int n)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n; c0 += 8192)
      if (t0 + c0 >= 1 && n >= t0 + c0 + 1)
        for (int c1 = 32 * b1; c1 <= ppcg_min(9999, n - 1); c1 += 8192)
          for (int c3 = ppcg_max(t1, ((t1 + c1 + 15) % 16) - c1 + 1); c3 <= ppcg_min(ppcg_min(31, n - c1 - 1), -c1 + 9999); c3 += 16)
            B[(t0 + c0) * 10000 + (c1 + c3)] = (((((((A[(t0 + c0 - 1) * 10000 + (c1 + c3)] * A[(t0 + c0 - 1) * 10000 + (c1 + c3)]) * A[(t0 + c0 - 1) * 10000 + (c1 + c3)]) * A[(t0 + c0 - 1) * 10000 + (c1 + c3)]) * A[(t0 + c0 - 1) * 10000 + (c1 + c3)]) * A[(t0 + c0 - 1) * 10000 + (c1 + c3)]) * A[(t0 + c0 - 1) * 10000 + (c1 + c3)]) + ((((((A[(t0 + c0) * 10000 + (c1 + c3 - 1)] * A[(t0 + c0) * 10000 + (c1 + c3 - 1)]) * A[(t0 + c0) * 10000 + (c1 + c3 - 1)]) * A[(t0 + c0) * 10000 + (c1 + c3 - 1)]) * A[(t0 + c0) * 10000 + (c1 + c3 - 1)]) * A[(t0 + c0) * 10000 + (c1 + c3 - 1)]) * A[(t0 + c0) * 10000 + (c1 + c3 - 1)]));
}
