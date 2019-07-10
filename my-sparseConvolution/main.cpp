#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mat_mul.h"

int main(int argc, char **argv){
  
    int k = 3, C = 3, M = 3, H = 2, W = 2;


    srand(time(NULL));
    float* mat_weight = rand_1_mat(M, C, k);
    float* mat_input = rand_2_mat(C, H, W);
    float* mat_output = rand_3_mat(M, H, W);

    float* com_mat_weight = switch_w_mat(M, C, k, mat_weight);
    sparse_change(com_mat_weight, M * C * k * k, 5);//构造稀疏矩阵，稀疏率为50%

    int*csrRowPtr;
	int*csrColInd;
	float*csrVal;
	dense2csr(com_mat_weight, csrRowPtr, csrColInd, csrVal, k * k * M, C);//将稠密矩阵转换为稀疏矩阵

    print_matrix(csrRowPtr, 1, k * k * M + 1);
	print_matrix(csrColInd, 1, csrRowPtr[k * k * M]);
	print_matrix(csrVal, 1, csrRowPtr[k * k * M]);


    gpu_compute_sparse(csrRowPtr, csrColInd, csrVal, mat_input, mat_output, C, M, k, H, W);
    print2(mat_output, M * H * W);


}
