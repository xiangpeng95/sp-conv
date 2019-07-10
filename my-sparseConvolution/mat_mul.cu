#include<cuda.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include "cuda_runtime.h"
#include <helper_cuda.h>
#include <iostream>
using namespace std;
extern "C++" void gpu_compute_sparse(int* h_csrRows, int* h_csrCols, float* h_csrVals, float* h_N, float* h_P, int C, int M, int K, int H, int W);
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
__device__
void print1(float* P, int len = 12){
    for (int l=0;l<len;l++){
           printf("%.1f ",P[l]);
    }
    printf("\n");
}
__device__
void print1(int* P, int len = 12){
    for (int l=0;l<len;l++){
           printf("%d ",P[l]);
    }
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
void h_csrmm_Kernel(int* csrRows, int* csrCols, float* csrVals, float* N, float* P, int C, int M, int H, int W){
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;

    float multi_sum = 0.0;
    if(Row < M && Col < H*W){
        int last_count = csrRows[Row] - csrRows[0];//这一行之前有多少数据
        int row_count = csrRows[Row + 1] - csrRows[Row];//该行有多少数据

        for(int i = 0;i < row_count;i++){
            int N_row = csrCols[last_count + i];
            float num1 = csrVals[last_count + i];
            float num2 = N[N_row * H * W + Col];

            multi_sum += num1 * num2;
        }
        P[H * W * Row + Col] = multi_sum;
    }

}

void gpu_compute_sparse(int* h_csrRows, int* h_csrCols, float* h_csrVals, float* h_N, float* h_P, int C, int M, int K, int H, int W){
    for(int i = 0;i < H * W * M; i++)h_P[i] = 0.0; //初始化

    size_t size_N = H * W * C * sizeof(float);
    size_t size_P = H * W * M * sizeof(float);

	float *d_N, *d_P, *d_temp_P;
	float *zHostPtr =new float[H * W * M];
    checkCudaErrors(cudaMalloc((void**)&d_N, size_N)); // 分配device端的矩阵空间
    checkCudaErrors(cudaMalloc((void**)&d_P, size_P));
    checkCudaErrors(cudaMalloc((void**)&d_temp_P, size_P));//充当buffer，记录中间值
    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);//将数据输入GPU


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

    for(int i = 0;i < K*K; i++){
        temp_count = h_csrRows[(i+1) * M] - h_csrRows[i * M];

        temp_csrVals = h_csrVals + h_csrRows[i * M];
        temp_csrRows = h_csrRows + i * M;
        temp_csrCols = h_csrCols + h_csrRows[i * M];

	    cudaMemcpy(d_csrVals, temp_csrVals, sizeof(float) * temp_count, cudaMemcpyHostToDevice);
	    cudaMemcpy(d_csrRows, temp_csrRows, sizeof(int)   * (M + 1),  cudaMemcpyHostToDevice);
	    cudaMemcpy(d_csrCols, temp_csrCols, sizeof(int)   * temp_count, cudaMemcpyHostToDevice);

        dim3 dimGrid(ceil((H*W) / 16.0), ceil(M / 16.0), 1);
        dim3 dimBlock(16, 16, 1);


        cudaDeviceSynchronize();
        h_csrmm_Kernel<<<dimGrid, dimBlock>>>(d_csrRows, d_csrCols, d_csrVals, d_N, d_temp_P, C, M, H, W);
        cudaDeviceSynchronize();
        int mark_row, mark_col;
        mark_col = i % K;
        mark_row = i / K;
        mark_row -= K / 2;
        mark_col -= K / 2;

        add_buffer_Kernel<<<dimGrid, dimBlock>>>(d_P, d_temp_P, M, H, W, mark_row, mark_col);
        cudaDeviceSynchronize();

    }


    cudaMemcpy(h_P, d_P, size_P, cudaMemcpyDeviceToHost);

    cudaFree(d_csrRows);
    cudaFree(d_csrCols);
    cudaFree(d_csrVals);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaFree(d_temp_P);

}

