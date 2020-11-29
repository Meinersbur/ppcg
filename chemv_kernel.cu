#include "chemv_kernel.hu"
struct ComplexFloat {
    float Re;
    float Im;
};
__global__ void kernel0(struct ComplexFloat *AT, struct ComplexFloat *X, struct ComplexFloat *Y, float alpha_im, float alpha_re, float beta_im, float beta_re, int incY, int n, int incX, int ldAT)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    float private_var5_Re;
    float private_var5_Im;
    float private_var2_Re;
    float private_var3_Im;
    float private_var2_Im;
    float private_var4_Im;
    float private_var4_Re;
    float private_var3_Re;
    float private_var99_Re;
    float private_var96_Re;
    float private_var98_Im;
    float private_var96_Im;
    float private_var94_Im;
    float private_var95_Im;
    float private_var94_Re;
    float private_var95_Re;
    float private_var97_Im;
    float private_var99_Im;
    float private_var97_Re;
    float private_var98_Re;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n; c0 += 1048576) {
      if (n >= t0 + c0 + 1) {
        private_var5_Re = ((Y[(t0 + c0) * incY + 0].Re * beta_re) - (Y[(t0 + c0) * incY + 0].Im * beta_im));
        private_var5_Im = ((Y[(t0 + c0) * incY + 0].Im * beta_re) + (Y[(t0 + c0) * incY + 0].Re * beta_im));
        Y[(t0 + c0) * incY + 0].Re = private_var5_Re;
        Y[(t0 + c0) * incY + 0].Im = private_var5_Im;
        private_var2_Re = (alpha_re * AT[(t0 + c0) * ldAT + (t0 + c0)].Re);
        private_var2_Im = (alpha_im * AT[(t0 + c0) * ldAT + (t0 + c0)].Re);
        private_var3_Re = ((private_var2_Re * X[(t0 + c0) * incX + 0].Re) - (private_var2_Im * X[(t0 + c0) * incX + 0].Im));
        private_var3_Im = ((private_var2_Im * X[(t0 + c0) * incX + 0].Re) + (private_var2_Re * X[(t0 + c0) * incX + 0].Im));
        private_var4_Re = (Y[(t0 + c0) * incY + 0].Re + private_var3_Re);
        private_var4_Im = (Y[(t0 + c0) * incY + 0].Im + private_var3_Im);
        Y[(t0 + c0) * incY + 0].Re = private_var4_Re;
        Y[(t0 + c0) * incY + 0].Im = private_var4_Im;
        for (int c3 = 0; c3 <= ppcg_min(31, t0 + c0 - 1); c3 += 1) {
          private_var97_Re = ((alpha_re * AT[c3 * ldAT + (t0 + c0)].Re) - (alpha_im * AT[c3 * ldAT + (t0 + c0)].Im));
          private_var97_Im = ((alpha_im * AT[c3 * ldAT + (t0 + c0)].Re) + (alpha_re * AT[c3 * ldAT + (t0 + c0)].Im));
          private_var98_Re = ((private_var97_Re * X[c3 * incX + 0].Re) - (private_var97_Im * X[c3 * incX + 0].Im));
          private_var98_Im = ((private_var97_Im * X[c3 * incX + 0].Re) + (private_var97_Re * X[c3 * incX + 0].Im));
          private_var99_Re = (Y[(t0 + c0) * incY + 0].Re + private_var98_Re);
          private_var99_Im = (Y[(t0 + c0) * incY + 0].Im + private_var98_Im);
          Y[(t0 + c0) * incY + 0].Re = private_var99_Re;
          Y[(t0 + c0) * incY + 0].Im = private_var99_Im;
        }
      }
      if (b0 == 0 && c0 == 0)
        for (int c3 = t0; c3 <= ppcg_min(31, n - 2); c3 += 1) {
          private_var94_Re = ((alpha_re * AT[t0 * ldAT + (c3 + 1)].Re) - (alpha_im * (-AT[t0 * ldAT + (c3 + 1)].Im)));
          private_var94_Im = ((alpha_im * AT[t0 * ldAT + (c3 + 1)].Re) + (alpha_re * (-AT[t0 * ldAT + (c3 + 1)].Im)));
          private_var95_Re = ((private_var94_Re * X[(c3 + 1) * incX + 0].Re) - (private_var94_Im * X[(c3 + 1) * incX + 0].Im));
          private_var95_Im = ((private_var94_Im * X[(c3 + 1) * incX + 0].Re) + (private_var94_Re * X[(c3 + 1) * incX + 0].Im));
          private_var96_Re = (Y[t0 * incY + 0].Re + private_var95_Re);
          private_var96_Im = (Y[t0 * incY + 0].Im + private_var95_Im);
          Y[t0 * incY + 0].Re = private_var96_Re;
          Y[t0 * incY + 0].Im = private_var96_Im;
        }
      __syncthreads();
      for (int c1 = 32; c1 < n - 1; c1 += 32) {
        if (n >= t0 + c0 + 1)
          for (int c3 = 0; c3 <= ppcg_min(31, t0 + c0 - c1 - 1); c3 += 1) {
            private_var97_Re = ((alpha_re * AT[(c1 + c3) * ldAT + (t0 + c0)].Re) - (alpha_im * AT[(c1 + c3) * ldAT + (t0 + c0)].Im));
            private_var97_Im = ((alpha_im * AT[(c1 + c3) * ldAT + (t0 + c0)].Re) + (alpha_re * AT[(c1 + c3) * ldAT + (t0 + c0)].Im));
            private_var98_Re = ((private_var97_Re * X[(c1 + c3) * incX + 0].Re) - (private_var97_Im * X[(c1 + c3) * incX + 0].Im));
            private_var98_Im = ((private_var97_Im * X[(c1 + c3) * incX + 0].Re) + (private_var97_Re * X[(c1 + c3) * incX + 0].Im));
            private_var99_Re = (Y[(t0 + c0) * incY + 0].Re + private_var98_Re);
            private_var99_Im = (Y[(t0 + c0) * incY + 0].Im + private_var98_Im);
            Y[(t0 + c0) * incY + 0].Re = private_var99_Re;
            Y[(t0 + c0) * incY + 0].Im = private_var99_Im;
          }
        for (int c3 = ppcg_max(0, t0 + c0 - c1); c3 <= ppcg_min(31, n - c1 - 2); c3 += 1) {
          private_var94_Re = ((alpha_re * AT[(t0 + c0) * ldAT + (c1 + c3 + 1)].Re) - (alpha_im * (-AT[(t0 + c0) * ldAT + (c1 + c3 + 1)].Im)));
          private_var94_Im = ((alpha_im * AT[(t0 + c0) * ldAT + (c1 + c3 + 1)].Re) + (alpha_re * (-AT[(t0 + c0) * ldAT + (c1 + c3 + 1)].Im)));
          private_var95_Re = ((private_var94_Re * X[(c1 + c3 + 1) * incX + 0].Re) - (private_var94_Im * X[(c1 + c3 + 1) * incX + 0].Im));
          private_var95_Im = ((private_var94_Im * X[(c1 + c3 + 1) * incX + 0].Re) + (private_var94_Re * X[(c1 + c3 + 1) * incX + 0].Im));
          private_var96_Re = (Y[(t0 + c0) * incY + 0].Re + private_var95_Re);
          private_var96_Im = (Y[(t0 + c0) * incY + 0].Im + private_var95_Im);
          Y[(t0 + c0) * incY + 0].Re = private_var96_Re;
          Y[(t0 + c0) * incY + 0].Im = private_var96_Im;
        }
        __syncthreads();
      }
    }
}
