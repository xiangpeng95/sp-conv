#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cusparse.h"

#include<iostream>
using namespace std;

__global__ void Kernel_compute_factor_map(float *factor_map, int height, int weight, float m_R)
{
	  int pos = (gridDim.y * blockIdx.x + blockIdx.y)*(blockDim.x * blockDim.y) + threadIdx.x * blockDim.y + threadIdx.y;

	float tmp = powf((float)(threadIdx.x - blockIdx.x), 2.0f)+ pow((float)(threadIdx.y - blockIdx.y), 2.0f) ;
	tmp = -tmp/(m_R * m_R);
	factor_map[pos] = expf(tmp);
}

int  main()
{
//	cudaSetDevice(0);

	cusparseHandle_t   handle;

   cusparseCreate(&handle);


   cusparseOperation_t opX     =   CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t opB    =   CUSPARSE_OPERATION_TRANSPOSE;
 //cusparseOperation_t opB    =   CUSPARSE_OPERATION_TRANSPOSE;
   cusparseMatDescr_t  desr;//=  (cusparseMatDescr_t *)malloc(sizeof(cusparseMatDescr_t));
   cusparseCreateMatDescr(&desr);

   cusparseSetMatType(desr, CUSPARSE_MATRIX_TYPE_GENERAL);
   cusparseSetMatIndexBase(desr, CUSPARSE_INDEX_BASE_ZERO);



	float h_X[]                           =       {1, 4,  2, 3, 5,  7, 8,  9,  6};
    float h_B[]                           =       {1, 2, 3, 4, 5,6,7,8};
    float h_C[]                           =        {0, 0, 0, 0,0,0,0, 0,0,0};
	float      h_csrX[]                           =              {1, 4, 2, 3, 5, 7, 8, 9, 6};
	int          h_csrRowPtrX[]             =              {0,2,4,7,9};
	int          h_csrColIndX[]               =              {0, 1, 1, 2, 0, 3, 4, 2, 4};

	float * d_csrX;
	int     * d_csrRowPtrX;
	int     * d_csrColIndex;
	float * d_B;
	float *d_C;



	int nnz = sizeof(h_csrX)/sizeof(float);

	cudaMalloc((void **)&d_csrX,  sizeof(h_csrX));
	cudaMalloc((void **)&d_csrRowPtrX, sizeof(h_csrRowPtrX));
	cudaMalloc((void **)&d_csrColIndex, sizeof(h_csrColIndX));
	cudaMalloc((void **)&d_B,  sizeof(h_B));
	cudaMalloc((void **)&d_C,  sizeof(h_C));

	cudaMemcpy(d_csrX, h_csrX,  sizeof(h_csrX), cudaMemcpyDefault );
	cudaMemcpy(d_csrRowPtrX, h_csrRowPtrX, sizeof(h_csrRowPtrX), cudaMemcpyDefault);
	cudaMemcpy(d_csrColIndex, h_csrColIndX, sizeof(h_csrColIndX), cudaMemcpyDefault);
	cudaMemcpy(d_C,  h_C, sizeof(h_C), cudaMemcpyDefault);
	cudaMemcpy(d_B,  h_B, sizeof(h_B), cudaMemcpyDefault);





	float alpha = 1.0f;
	float beta  =   0.0f;
    cusparseScsrmm2(handle, opX,  opB,  4, 2, 5, nnz, &alpha, desr, d_csrX, d_csrRowPtrX, d_csrColIndex, d_B, 2, &beta, d_C, 5 );
   // cusparseScsrmm(handle, opX,   4, 2, 5, nnz, &alpha, desr, d_csrX, d_csrRowPtrX, d_csrColIndex, d_B, 5, &beta, d_C, 4 );
    cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDefault);

#if 0
    float *factor_map;
    int SOM_X = 10;
    int SOM_Y = 8;
    cudaMalloc((void**)&factor_map, sizeof(float)*SOM_X*SOM_Y*SOM_X *SOM_Y);

    float m_R = 2.0f;
    dim3 grid(SOM_Y, SOM_X);
    dim3 block(SOM_Y, SOM_X);
    Kernel_compute_factor_map<<<grid, block>>>(factor_map, SOM_Y, SOM_X, m_R);

    float *h_factor = new float[SOM_X*SOM_Y*SOM_X *SOM_Y];

    cudaMemcpy(h_factor, factor_map, sizeof(float)*SOM_X*SOM_Y*SOM_X *SOM_Y, cudaMemcpyDefault);
#endif

#if 1
    for(int i = 0;  i < sizeof(h_C)/sizeof(float); ++i)
    {
    	cout << h_C[i] << " ";
    }
    cout << endl;
#endif

return 0;
}