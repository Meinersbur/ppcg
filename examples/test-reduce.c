#include <stdio.h>
#include <stdlib.h>

#define nn 1000
#define MC(x,y) x*x*x*x*x*x*x+y*y*y*y*y*y*y

void test(int n, int A[nn][nn], int B[nn][nn]){
    // int answer=0;
    #pragma scop
    for(int i=1; i<n; i++){
        for(int j=1; j<n; j++){
            B[i][j]=0;
            // for(int c=1; c<n; c++)
            B[i][j] += A[i-1][j] * A[i][j-1];
        }
    }
    #pragma endscop
    // return answer;
}

int main(){
    int** testA = malloc(sizeof(int*)*nn);
    for(int i=0; i<nn; i++){
        testA[i] = malloc(sizeof(int)*nn);
    }
    int** testB = malloc(sizeof(int*)*nn);
    for(int i=0; i<nn; i++){
        testB[i] = malloc(sizeof(int)*nn);
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