#include<cuda.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include "cuda_runtime.h"
#include "cusparse.h"
#include "cublas_v2.h"
#include <helper_cuda.h>
#include <iostream>
using namespace std;
extern "C++" void gpu_compute_sparse(int* h_csrRows, int* h_csrCols, float* h_csrVals, float* h_N, float* h_P, int C, int M, int K, int H, int W);
extern "C++" void gpu_compute_dense(float* h_M, float* h_N, float* h_P, int C, int M, int K, int H, int W);
void printttt(float *p, int len){
    for(int i = 0;i < len; i++)
        printf("%.1f ",p[i]);
    printf("\n");
}
void printttt(int *p, int len){
    for(int i = 0;i < len; i++)
        printf("%d ",p[i]);
    printf("\n");
}
__global__
void add_buffer_Kernel(float* d_P, float* d_temp_P, int M, int H, int W, int mark_row, int mark_col){
    int buffer_row, buffer_col, x, y;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;

    if(Row < M && Col < H*W){
        y = Col / W;
        x = Col % W;

        buffer_row = y - mark_row;
        buffer_col = x - mark_col;

        if(buffer_row >= 0 && buffer_row < H && buffer_col >= 0 && buffer_col < W)
            d_P[H*W*Row + buffer_row * W + buffer_col] += d_temp_P[H*W*Row + y * W + x];
    }
}
__global__
void print(float* P, int len = 12){
    for (int l=0;l<len;l++){
           printf("%.1f ",P[l]);
    }
    printf("\n");
}
__global__
void print(int* P, int len = 12){
    for (int l=0;l<len;l++){
           printf("%d ",P[l]);
    }
    printf("\n");
}
void gpu_compute_sparse(int* h_csrRows, int* h_csrCols, float* h_csrVals, float* h_N, float* h_P, int C, int M, int K, int H, int W){
    for(int i = 0;i < H * W * M; i++)h_P[i] = 0.0; //初始化
    printf("\n");
    printttt(h_N, H*W*C);

    size_t size_N = H * W * C * sizeof(float);
    size_t size_P = H * W * M * sizeof(float);

	float *d_N, *d_P, *d_temp_P;

    checkCudaErrors(cudaMalloc((void**)&d_N, size_N)); // 分配device端的矩阵空间
    checkCudaErrors(cudaMalloc((void**)&d_P, size_P));
    checkCudaErrors(cudaMalloc((void**)&d_temp_P, size_P));//充当buffer，记录中间值

    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);//将数据输入GPU
    checkCudaErrors(cudaMemset((void *)d_P,0, size_P));
    checkCudaErrors(cudaMemset((void *)d_temp_P,0, size_P));

    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    checkCudaErrors(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    checkCudaErrors(cusparseCreate(&handle));

    float alpha=1;
    float beta=0;

    float* temp_csrVals;
	int* temp_csrRows;
	int* temp_csrCols;

	float* d_csrVals;
	int* d_csrRows;
	int* d_csrCols;

    checkCudaErrors(cudaMalloc((void **)&d_csrVals, sizeof(float) * C));
	checkCudaErrors(cudaMalloc((void **)&d_csrRows, sizeof(int) * (M + 1)));// 分配device端的csr空间
	checkCudaErrors(cudaMalloc((void **)&d_csrCols, sizeof(int) * C));
    cudaDeviceSynchronize();

    int temp_count;
    for(int i = 8;i < K*K; i++){
        temp_count = h_csrRows[(i+1) * M] - h_csrRows[i * M];
        printf("\n%d\n",temp_count);

        temp_csrVals = h_csrVals + h_csrRows[i * M];
        temp_csrRows = h_csrRows + i * M;
        temp_csrCols = h_csrCols + h_csrRows[i * M];

	    cudaMemcpy(d_csrVals, temp_csrVals, sizeof(float) * temp_count, cudaMemcpyHostToDevice);
	    cudaMemcpy(d_csrRows, temp_csrRows, sizeof(int)   * (M + 1),  cudaMemcpyHostToDevice);
	    cudaMemcpy(d_csrCols, temp_csrCols, sizeof(int)   * temp_count, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        printttt(temp_csrVals, temp_count);
        print<<<1,1>>>(d_csrVals, temp_count);
        print<<<1,1>>>(d_csrRows, M + 1);
        print<<<1,1>>>(d_csrCols, temp_count);
        cudaDeviceSynchronize();
        checkCudaErrors(cusparseScsrmm2(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE,
            M,  // number of rows of sparse matrix A.
            H*W,  // number of columns of dense matrix op(B) and C.
            C,  // number of columns of sparse matrix A.
            temp_count,   // number of nonzero elements of sparse matrix A.
            &alpha,
            descr,
            d_csrVals,
            d_csrRows,
            d_csrCols,
            d_N,
            H*W,
            &beta,
            d_temp_P,
            M));
        cudaDeviceSynchronize();
        print<<<1,1>>>(d_temp_P);

    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_P, d_P, size_P, cudaMemcpyDeviceToHost);

    cudaFree(d_csrVals);
    cudaFree(d_csrRows);
    cudaFree(d_csrCols);
    // Free device memory for M, N, P
    cudaFree(d_N);
    cudaFree(d_P);
    cudaFree(d_temp_P);

}

void gpu_compute_dense(float* h_M, float* h_N, float* h_P, int C, int M, int K, int H, int W) {

    float *d_M, *d_N, *d_P, *d_temp_P;

    size_t size_M = M * C * sizeof(float);
    size_t size_N = H * W * C * sizeof(float);
    size_t size_P = H * W * M * sizeof(float);


    cudaMalloc((void**)&d_M, size_M);
    cudaMalloc((void**)&d_N, size_N);
    cudaMalloc((void**)&d_P, size_P);
    cudaMalloc((void**)&d_temp_P, size_P);//充当buffer，记录中间值

    for(int i = 0;i < H * W * M; i++)h_P[i] = 0.0; //初始化

    cudaMemcpy(d_P, h_P, size_P, cudaMemcpyHostToDevice);//将数据输入GPU
    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);//将数据输入GPU

    float alpha = 1;
    float beta = 0;
    float *h_temp_M = (float*)malloc(size_M);

    for(int i = 0;i < K*K; i++){
        memcpy(h_temp_M, h_M + i * M * C, M * C * sizeof(float));
        cudaMemcpy(d_M, h_temp_M, size_M, cudaMemcpyHostToDevice);

        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSgemm(handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            H*W,                  //矩阵N的列数
            M,                    //矩阵M的行数
            C,                    //矩阵M的列数
            &alpha,
            d_N,
            H*W,
            d_M,
            C,
            &beta,
            d_temp_P,
            H*W); //中间结果d_temp_P，偏移后加入d_P


        int mark_row, mark_col;
        mark_col = i % K;
        mark_row = i / K;
        mark_row -= K / 2;
        mark_col -= K / 2;

        dim3 dimGrid(ceil((H*W) / 16.0), ceil(M / 16.0), 1);
        dim3 dimBlock(16, 16, 1);

        add_buffer_Kernel<<<dimGrid, dimBlock>>>(d_P, d_temp_P, M, H, W, mark_row, mark_col);

    }


    cudaMemcpy(h_P, d_P, size_P, cudaMemcpyDeviceToHost);

    // Free device memory for M, N, P
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaFree(d_temp_P);

    free(h_temp_M);

}


















/*
for(int L = 0; L < M; L++) //一行一行的进行偏移，每一行的起始地址d_temp_P + H*W*l
            for(int y = 0; y < H; y++)
                for(int x = 0; x < W; x++){
                    row = y - mark_row;
                    col = x - mark_col;

                    if(row >= 0 && row < H && col >= 0 && col < W)
                        d_P[H*W*L + row * W + col] += d_temp_P[H*W*L + y * W + x];

                }
*/