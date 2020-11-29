
gcc test-reduce.c -O3
time ./a.out
../ppcg test-reduce.c
nvcc test-reduce_host.cu test-reduce_kernel.cu -o cc.o -O3
time ./cc.o