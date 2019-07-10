#include<cuda.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
extern "C" void gpu_compute(float* h_M, float* h_N, float* h_P, int C, int M, int K, int H, int W);

__global__
void direct_add_multi_Kernel(float* d_P, float* d_M, float* d_N, int M, int H, int W, int mark_row, int mark_col){
    int buffer_row, buffer_col, x, y;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    float result;










    if(Row < M && Col < H*W){
        y = Col / W;
        x = Col % W;

        buffer_row = y - mark_row;
        buffer_col = x - mark_col;

        if(buffer_row >= 0 && buffer_row < H && buffer_col >= 0 && buffer_col < W)
            d_P[H*W*Row + buffer_row * W + buffer_col] += result;
    }
}

void gpu_compute(float* h_M, float* h_N, float* h_P, int C, int M, int K, int H, int W) {

    float *d_M, *d_N, *d_P, *d_temp_P;

    size_t size_M = M * C * sizeof(float);
    size_t size_N = H * W * C * sizeof(float);
    size_t size_P = H * W * M * sizeof(float);


    cudaMalloc((void**)&d_M, size_M);
    cudaMalloc((void**)&d_N, size_N);
    cudaMalloc((void**)&d_P, size_P);


    for(int i = 0;i < H * W * M; i++)h_P[i] = 0.0; //初始化

    cudaMemcpy(d_P, h_P, size_P, cudaMemcpyHostToDevice);//将数据输入GPU
    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);//将数据输入GPU

    float alpha = 1;
    float beta = 0;
    float *h_temp_M = (float*)malloc(size_M);

    for(int i = 0;i < K*K; i++){
        memcpy(h_temp_M, h_M + i * M * C, M * C * sizeof(float));
        cudaMemcpy(d_M, h_temp_M, size_M, cudaMemcpyHostToDevice);

        int mark_row, mark_col;
        mark_col = i % K;
        mark_row = i / K;
        mark_row -= K / 2;
        mark_col -= K / 2;

        dim3 dimGrid(ceil((H*W) / 16.0), ceil(M / 16.0), 1);
        dim3 dimBlock(16, 16, 1);

        direct_add_multi_Kernel<<<dimGrid, dimBlock>>>(d_P, d_M, d_N, M, H, W, mark_row, mark_col);

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