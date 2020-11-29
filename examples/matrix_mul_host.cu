#include <assert.h>
#include <stdio.h>
#include "matrix_mul_kernel.hu"
/**
 * @ Author: Minhua Chen
 * @ Create Time: 2019-08-24 09:56:56
 * @ Modified by: Minhua Chen
 * @ Modified time: 2019-08-24 10:43:55
 * @ Description:
 */

#include <stdio.h>
#include <stdlib.h>
#define DEFAULT_SIZE_M 32*32

void print_matrix(int* mat, int row, int col) {
    for (int i = 0; i < row * col; i++) {
        printf("%d\t", mat[i]);
        if ((i+1) % col == 0) {
            printf("\n");
        }
        
    }
     printf("--------c--------------------\n");
}

void mulMatrix(int m_size, int nn, int* mat1, int* mat2, int* result){
    #pragma scop
    for (int r = 0; r < m_size; r++) {
        for (int c = 0; c < m_size; c++) {
            for (int n = 0; n < m_size; n++) {
                result[r*m_size + c] += mat1[r*m_size+n] * mat2[n*m_size+c]; 
            }
        }
    }
    #pragma endscop
}

int main(int argc, char* argv[]) {
    int *mat1, *mat2, *result;
    int m_size;

    if (argc > 1) {
        m_size = atoi(argv[1]);
    } else {
        m_size = DEFAULT_SIZE_M;
    }
    
    // 用一位数组表示二维矩阵
    mat1 = (int*) malloc(m_size * m_size * sizeof(int));
    mat2 = (int*) malloc(m_size * m_size * sizeof(int));
    result = (int*) malloc(m_size * m_size * sizeof(int));

    // initialize
    for (int i = 0; i < m_size * m_size; i++) {
        mat1[i] = rand()%10000;
        mat2[i] = rand()%10000;
        result[i] = 0;
        
    }
    mulMatrix(m_size, m_size*m_size, mat1, mat2, result);
    // #pragma scop
    // for (int r = 0; r < m_size; r++) {
    //     for (int c = 0; c < m_size; c++) {
    //         for (int n = 0; n < m_size; n++) {
    //             result[r*m_size + c] += mat1[r*m_size+n] * mat2[n*m_size+c]; 
    //         }
    //     }
    // }
    // #pragma endscop
    
    if (m_size < 10) {
        printf("-----mat1----\n");
        print_matrix(mat1, m_size, m_size);
        printf("-----mat1----\n");
        print_matrix(mat2, m_size, m_size);
        printf("----mat1 * mat2---\n");
        print_matrix(result, m_size, m_size);
    }

    printf("final element: %d\n", result[m_size*m_size-1]);
}