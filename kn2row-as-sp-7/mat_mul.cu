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
extern "C++" void gpu_compute_sparse(int* h_csrRows, int* h_csrCols, float* h_csrVals, float* h_N, float* h_P, int C, int M, int K, int H, int W, int padding, int stride, int dilation);
extern "C++" void gpu_compute_dense(float* h_M, float* h_N, float* h_P, int C, int M, int K, int H, int W, int padding, int stride, int dilation);

void print3(float* P, int len){
    for (int l=0;l<len;l++){
           printf("%.1f ",P[l]);
    }
    printf("\n");
}

void print3(int* P, int len){
    for (int l=0;l<len;l++){
           printf("%d ",P[l]);
    }
    printf("\n");
}
__global__
void printtt(float* P, int len){
    for (int l=0;l<len;l++){
           printf("%.1f ",P[l]);
    }
    printf("----%d\n",len);
}
__global__
void printtt(int* P, int len){
    for (int l=0;l<len;l++){
           printf("%d ",P[l]);
    }
    printf("----%d\n",len);
}


__global__
void add_buffer_Kernel(float* d_P, float* d_temp_P, int M, int K, int H, int W, int mark_row, int mark_col, int padding, int stride, int dilation){
    int buffer_row, buffer_col, x, y;
    int re_row, re_col;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;

    int output_H = (H + 2 * padding - K)/stride + 1;
    int output_W = (W + 2 * padding - K)/stride + 1;

    y = Row / W;
    x = Row % W;
    if(Col < M && Row < H*W){   //此处是转置矩阵的偏移

        buffer_row = y + padding - mark_row;
        buffer_col = x + padding - mark_col;
        re_row = buffer_row/stride;
        re_col = buffer_col/stride;

        if(re_row >= 0 && re_row < output_H && re_col >= 0 && re_col < output_W && (buffer_row % stride == 0) && (buffer_col % stride == 0))
            d_P[re_row * output_W + re_col + Col * output_H * output_W] += d_temp_P[M*(y * H + x) + Col];
    }
}
__global__
void change_row(int *a, int len){
    int start = a[0];
    for(int i = 0;i <= len;i++)
        a[i] -= start;
}

void gpu_compute_sparse(int* h_csrRows, int* h_csrCols, float* h_csrVals, float* h_N, float* h_P, int C, int M, int K, int H, int W, int padding, int stride, int dilation){
    int output_H = (H + 2 * padding - K)/stride + 1;
    int output_W = (W + 2 * padding - K)/stride + 1;
    for(int i = 0;i < output_H * output_W * M; i++)h_P[i] = 0.0; //初始化

    size_t size_N = H * W * C * sizeof(float);
    size_t size_P = H * W * M * sizeof(float);
    size_t size_P_2 = output_H * output_W * M * sizeof(float);

	float *d_N, *d_P, *d_temp_P;
    checkCudaErrors(cudaMalloc((void**)&d_N, size_N)); // 分配device端的矩阵空间
    checkCudaErrors(cudaMalloc((void**)&d_P, size_P_2));
    checkCudaErrors(cudaMalloc((void**)&d_temp_P, size_P));//充当buffer，记录中间值
    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);//将数据输入GPU
    checkCudaErrors(cudaMemset((void *)d_P,0, size_P_2));
    checkCudaErrors(cudaMemset((void *)d_temp_P,0, size_P));

    float* d_csrVals;
	int* d_csrRows;
	int* d_csrCols;
	checkCudaErrors(cudaMalloc((void **)&d_csrVals, sizeof(float) * M * C));
	checkCudaErrors(cudaMalloc((void **)&d_csrRows, sizeof(int) * (M + 1)));// 分配device端的csr空间
	checkCudaErrors(cudaMalloc((void **)&d_csrCols, sizeof(int) * M * C));


    float* temp_csrVals;
	int* temp_csrRows;
	int* temp_csrCols;
	int temp_count;

    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    checkCudaErrors(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    checkCudaErrors(cusparseCreate(&handle));

    float alpha=1.0;
    float beta=0.0;
    for(int i = 0;i < K*K; i++){
        temp_count = h_csrRows[(i+1) * M] - h_csrRows[i * M];
        //printf("%d\n",temp_count);

        temp_csrVals = h_csrVals + h_csrRows[i * M];
        temp_csrRows = h_csrRows + i * M;
        temp_csrCols = h_csrCols + h_csrRows[i * M];

	    cudaMemcpy(d_csrVals, temp_csrVals, sizeof(float) * temp_count, cudaMemcpyHostToDevice);
	    cudaMemcpy(d_csrRows, temp_csrRows, sizeof(int)   * (M + 1),  cudaMemcpyHostToDevice);
	    cudaMemcpy(d_csrCols, temp_csrCols, sizeof(int)   * temp_count, cudaMemcpyHostToDevice);

        change_row<<<1,1>>>(d_csrRows, M);

        dim3 dimGrid(ceil(M / 16.0), ceil((H*W) / 16.0), 1);
        dim3 dimBlock(16, 16, 1);
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
        //printtt<<<1,1>>>(d_temp_P,H * W * M);
        //printtt<<<1,1>>>(d_P,H * W * M);
        //printf("\n");
        int mark_row, mark_col;
        mark_col = i % K;
        mark_row = i / K;


        cudaDeviceSynchronize();
        add_buffer_Kernel<<<dimGrid, dimBlock>>>(d_P, d_temp_P, M, K, H, W, mark_row, mark_col, padding, stride, dilation);

    }
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);

    //printtt<<<1,1>>>(d_P,output_H * output_W * M);

    cudaDeviceSynchronize();

    cudaMemcpy(h_P, d_P, size_P_2, cudaMemcpyDeviceToHost);
    //print3(h_P, H*W*M);//------------------------------------------------print
    cudaFree(d_csrRows);
    cudaFree(d_csrCols);
    cudaFree(d_csrVals);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaFree(d_temp_P);

}

void gpu_compute_dense(float* h_M, float* h_N, float* h_P, int C, int M, int K, int H, int W, int padding, int stride, int dilation) {

    float *d_M, *d_N, *d_P, *d_temp_P;
    int output_H = (H + 2 * padding - K)/stride + 1;
    int output_W = (W + 2 * padding - K)/stride + 1;

    size_t size_M = M * C * sizeof(float);
    size_t size_N = H * W * C * sizeof(float);
    size_t size_P = H * W * M * sizeof(float);
    size_t size_P_2 = output_H * output_W * M * sizeof(float);

    cudaMalloc((void**)&d_M, size_M);
    cudaMalloc((void**)&d_N, size_N);
    cudaMalloc((void**)&d_P, size_P_2);
    cudaMalloc((void**)&d_temp_P, size_P);//充当buffer，记录中间值

    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);//将数据输入GPU
    checkCudaErrors(cudaMemset((void *)d_P,0, size_P_2));
    checkCudaErrors(cudaMemset((void *)d_temp_P,0, size_P));

    float alpha = 1;
    float beta = 0;

    cublasHandle_t handle;
    cublasCreate(&handle);
    for(int i = 0;i < K*K; i++){
        cudaMemcpy(d_M, h_M + i * M * C, size_M, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        cublasSgemm(handle,
            CUBLAS_OP_T,
            CUBLAS_OP_T,
            M,
            H*W,
            C,
            &alpha,
            d_M,
            C,
            d_N,
            H*W,
            &beta,
            d_temp_P,
            M); //中间结果d_temp_P，偏移后加入d_P
        cudaDeviceSynchronize();
        //printtt<<<1,1>>>(d_temp_P,H * W * M);

        int mark_row, mark_col;
        mark_col = i % K;
        mark_row = i / K;


        dim3 dimGrid(ceil(M / 16.0), ceil((H*W) / 16.0), 1);
        dim3 dimBlock(16, 16, 1);

        cudaDeviceSynchronize();

        add_buffer_Kernel<<<dimGrid, dimBlock>>>(d_P, d_temp_P, M, K, H, W, mark_row, mark_col, padding, stride, dilation);

    }
    cublasDestroy(handle);

    cudaMemcpy(h_P, d_P, size_P_2, cudaMemcpyDeviceToHost);
    //print3(h_P, output_H*output_W*M);//------------------------------------------------print
    // Free device memory for M, N, P
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaFree(d_temp_P);

}

//cublasHandle_t handle2;
//cublasCreate(&handle2);
//cudaDeviceSynchronize();
//cublasSgeam(handle2, CUBLAS_OP_T, CUBLAS_OP_N, H*W, M, &alpha, d_P, M, &beta, d_P, H*W, d_temp_P, H*W);//将结果矩阵进行转置
//cublasDestroy(handle2);