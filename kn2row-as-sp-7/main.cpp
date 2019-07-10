#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mat_mul.h"

int main(int argc, char **argv){

    int K = 3, C = 3, M = 1, input_H = 5, input_W = 5;
    int padding = 0;
    int stride = 2;
    int dilation = 2;

    srand(time(NULL));
    float* b_mat_weight = rand_1_mat(M, C, K);
    float* mat_input = rand_2_mat(C, input_H, input_W);

    K = dilation * (K - 1) + 1;
    float* mat_weight = dilation_change(M, C, K, dilation, b_mat_weight);

    int output_H = (input_H + 2 * padding - K)/stride + 1;
    int output_W = (input_W + 2 * padding - K)/stride + 1;

    float* dense_output = rand_3_mat(M, output_H, output_W);
    float* sp_output = rand_3_mat(M, output_H, output_W);

    float* com_mat_weight = switch_w_mat(M, C, K, mat_weight);
    sparse_change(com_mat_weight, M * C * K * K, -1);//构造稀疏矩阵，稀疏率为50%


    gpu_compute_dense(com_mat_weight, mat_input, dense_output, C, M, K, input_H, input_W, padding, stride, dilation);
    printf("\n\n\n\n");
    int*csrRowPtr;
	int*csrColInd;
	float*csrVal;
	dense2csr(com_mat_weight, csrRowPtr, csrColInd, csrVal, K * K * M, C);//将稠密矩阵转换为稀疏矩阵

    gpu_compute_sparse(csrRowPtr, csrColInd, csrVal, mat_input, sp_output, C, M, K, input_H, input_W, padding, stride, dilation);

    print2(dense_output, M * output_H * output_W);
    printf("\n\n");
    print2(sp_output, M * output_H * output_W);
}
