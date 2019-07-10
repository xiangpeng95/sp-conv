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

void gpu_compute_sparse(int* h_csrRows, int* h_csrCols, float* h_csrVals, float* h_N, float* h_P, int C, int M, int K, int H, int W){
    int nnz = h_csrRows[K * K * M];
    int out_col = H * W;

    int *d_csrRows,*d_csrCols;
    float alpha=1;
    float beta=0;
    float *d_csrVals;
    float *d_N, *d_P, *d_temp_P;

    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    checkCudaErrors(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    checkCudaErrors(cusparseCreate(&handle));

    checkCudaErrors(cudaMalloc((void **)&d_csrVals, nnz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_csrRows, (M+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrCols, nnz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_N, out_col*C*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_P, out_col*(M)*sizeof(float)));

    cudaMemcpy(d_csrVals,h_csrVals,nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRows,h_csrRows,(M+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrCols,h_csrCols,nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N,h_N,out_col*C*sizeof(float), cudaMemcpyHostToDevice);

    checkCudaErrors(cudaMemset((void *)d_P,0, out_col*(M)*sizeof(float)));

    for(int i = 0;i < K*K; i++){
        checkCudaErrors(cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,M,out_col,C,nnz,&alpha,descr,d_csrVals,d_csrRows, d_csrCols,d_N,C,&beta,d_P,M));
    }
    checkCudaErrors(cudaMemcpy(h_P,d_P,out_col*(M)*sizeof(float),cudaMemcpyDeviceToHost));

    for (int i=0;i<M*out_col;i++)
    {
        cout<<h_P[i]<<" ";
    }

}