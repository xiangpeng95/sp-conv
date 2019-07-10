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
void initialize(float *cooValHostPtr,int *cooColIndexHostPtr,float *yHostPtr,int *csrRowPtr)
{
    cooValHostPtr[0]=1.0;
    cooValHostPtr[1]=2.0;
    cooValHostPtr[2]=3.0;
    cooValHostPtr[3]=4.0;
    cooValHostPtr[4]=5.0;
    cooValHostPtr[5]=6.0;
    cooValHostPtr[6]=7.0;
    cooValHostPtr[7]=8.0;
    cooValHostPtr[8]=9.0;
    cooValHostPtr[9]=10.0;

    cooColIndexHostPtr[0]=0;
    cooColIndexHostPtr[1]=2;
    cooColIndexHostPtr[2]=3;
    cooColIndexHostPtr[3]=1;
    cooColIndexHostPtr[4]=0;
    cooColIndexHostPtr[5]=2;
    cooColIndexHostPtr[6]=3;
    cooColIndexHostPtr[7]=1;
    cooColIndexHostPtr[8]=3;
    cooColIndexHostPtr[9]=0;

    csrRowPtr[0]=0;
    csrRowPtr[1]=3;
    csrRowPtr[2]=4;
    csrRowPtr[3]=7;
    csrRowPtr[4]=9;
    csrRowPtr[5]=10;

    yHostPtr[0] = 10.0;
    yHostPtr[1] = 20.0;
    yHostPtr[2] = 30.0;
    yHostPtr[3] = 40.0;
    yHostPtr[4] = 50.0;
    yHostPtr[5] = 60.0;
    yHostPtr[6] = 70.0;
    yHostPtr[7] = 80.0;
}

void cuda_sparse()
{
    int m=5,n=4,nnz=10;
    float *cooValHostPtr=new float[nnz];
    float *zHostPtr =new float[2*(m)];

    int *cooColIndexHostPtr=new int[nnz];
    int *csrRowPtr=new int[m+1];

    int *crsRow,*cooCol;

    float alpha=1;
    float beta=0;
    float *yHostPtr=new float[2*n];
    float * y,*cooVal,*z;
    initialize(cooValHostPtr,cooColIndexHostPtr,yHostPtr,csrRowPtr);


    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    checkCudaErrors(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    checkCudaErrors(cusparseCreate(&handle));

    checkCudaErrors(cudaMalloc((void **)&cooVal, nnz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&crsRow, (m+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&cooCol, nnz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&y, n*2*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&z, m*2*sizeof(float)));

    cudaMemcpy(cooVal,cooValHostPtr,nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(crsRow,csrRowPtr,(m+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cooCol,cooColIndexHostPtr,nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(y,yHostPtr,n*2*sizeof(float), cudaMemcpyHostToDevice);

    checkCudaErrors(cudaMemset((void *)z,0, m*2*sizeof(float)));

    checkCudaErrors(cusparseScsrmm2(handle,  CUSPARSE_OPERATION_NON_TRANSPOSE ,CUSPARSE_OPERATION_TRANSPOSE ,5,2,4,nnz,&alpha,descr,cooVal,crsRow, cooCol,y,2,&beta,z,m));

    checkCudaErrors(cudaMemcpy(zHostPtr,z,m*2*sizeof(float),cudaMemcpyDeviceToHost));

    //for (int i = 0; i < m; i++)
    //{
    //  //if(i%(2)==0&&i!=0)
    //  //  cout<<endl;
    //  cout<<zHostPtr[i]<<" "<<zHostPtr[i+m]<<endl;
    //}
    for (int i=0;i<m*2;i++)
    {
        cout<<zHostPtr[i]<<" ";
    }
}

int main()
{
    cuda_sparse();
    return 0;
}