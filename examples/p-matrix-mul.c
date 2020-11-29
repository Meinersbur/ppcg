#include <stdio.h>
#include <stdlib.h>
#define MSIZE 100
#define SIZE MSIZE*MSIZE

#define get(a,x,y) a[x*MSIZE+y]

void mulMartix(int nn, int mat1[SIZE], int mat2[SIZE], int answer[SIZE]){
    #pragma scop
    for(int i=0; i<nn; i++){
        for(int j=0; j<nn; j++){
            get(answer, i, j) = 0;
            for(int c=0; c<nn; c++){
                get(answer, i, j) += get(mat1, i, c) * get(mat2, c, j);
            }
        }
    }
    #pragma endscop
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