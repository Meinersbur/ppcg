#include "p-matrix-mul_kernel.hu"
__global__ void kernel0(int *answer, int *mat1, int *mat2, int nn)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int private_answer[1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    #define ppcg_fdiv_q(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    for (int c0 = 32 * b0; c0 < 101 * nn - 100; c0 += 1048576) {
      for (int c1 = ppcg_max(0, 32 * ppcg_fdiv_q(-nn + c0 + 100, 3200)); c1 <= ppcg_min(nn - 1, c0 / 100); c1 += 32)
        for (int c3 = ppcg_max(0, -c1 + ppcg_fdiv_q(-nn + t0 + c0, 100) + 1); c3 <= ppcg_min(ppcg_min(31, nn - c1 - 1), -c1 + (t0 + c0) / 100); c3 += 1) {
          private_answer[0] = 0;
          for (int c4 = 0; c4 < nn; c4 += 1)
            private_answer[0] += (mat1[100 * c1 + 100 * c3 + c4] + mat2[t0 + c0 - 100 * c1 - 100 * c3 + 100 * c4]);
        }
      if ((101 * nn >= t0 + c0 + 101 && t0 + c0 + 100 >= 100 * nn) || (100 * nn >= t0 + c0 + 101 && nn >= ((t0 + c0) % 100) + 1))
        answer[t0 + c0] = private_answer[0];
      __syncthreads();
    }
}
