#ifndef MAT_MUL_H
#define MAT_MUL_H

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <iostream>
using namespace std;
void gpu_compute_sparse(int* h_csrRows, int* h_csrCols, float* h_csrVals, float* h_N, float* h_P, int C, int M, int K, int H, int W);
void gpu_compute_dense(float* h_M, float* h_N, float* h_P, int C, int M, int K, int H, int W);

static inline float *rand_1_mat(int f, int c, int k) {
  float *mat = (float *) malloc(k * k * c * f *sizeof(float));
  if (mat == NULL) { 
    printf("Error allocating CPU memory");
    exit(1);
  }
  for(int x = 0; x < f; x++)
    for(int y = 0; y < c; y++)
      for(int i = 0; i < k; i++){
        for(int j = 0; j < k; j++){
          mat[x * c * k * k + y * k * k + i * k + j] = (float)(rand() % 100);
        }
      }
  return mat;
}
static inline float *rand_2_mat(int w_channel, int h, int w) {
  float *mat = (float *) malloc(w_channel * h * w * sizeof(float));
  if (mat == NULL) {
    printf("Error allocating CPU memory");
    exit(1);
  }
  for(int k = 0; k < w_channel; k++)
      for(int i = 0; i < h; i++){
         for(int j = 0; j < w; j++){
            mat[k * h * w + i * w + j] = (float)(rand() % 100);
         }
      }
  return mat;
}
static inline float *rand_3_mat(int w_channel, int h, int w) {
  float *mat = (float *) malloc(w_channel * h * w * sizeof(float));
  if (mat == NULL) {
    printf("Error allocating CPU memory");
    exit(1);
  }
  return mat;
}
static inline float *switch_w_mat(int filter, int w_channel, int k, float *mat_weight){
    float *mat = (float *) malloc(k * k * w_channel * filter *sizeof(float));
    if (mat == NULL) {
    printf("Error allocating CPU memory");
    exit(1);
    }
    for(int i = 0; i < k * k; i++)
        for(int j = 0; j < filter; j++)
            for(int x = 0; x < w_channel; x++){
                mat[i * filter * w_channel + j * w_channel + x] = mat_weight[j * w_channel * k * k + x * k * k + i ];
            }
    return mat;
}
void sparse_change(float* mat, int len, int rate){
    float mark;

    for(int i = 0;i < len; i++){
        mark = rand()%10;
        if(mark < rate)mat[i] = 0.0;
    }
}
void dense2csr(float*data, int*&rowPtr, int*&colInd, float*&val, int m, int n) {
	rowPtr = (int*)malloc(sizeof(int)*(m + 1));

	int* tcolInd = (int*)malloc(sizeof(int)*(m *n));
	float* tval = (float*)malloc(sizeof(float)*(m *n));
	int towtal = m * n;
	int nnv = 0;

	for (int i = 0; i < m; i++) {
		rowPtr[i] = nnv;
		for (int j = 0; j < n; j++) {
			int l = i + j * m;
			if (data[l] != 0) {
				tcolInd[nnv] = j;
				tval[nnv] = data[l];
				nnv++;
			}
		}
	}
	rowPtr[m] = nnv;

	colInd = (int*)malloc(sizeof(int)*(nnv));
	val = (float*)malloc(sizeof(float)*(nnv));

	memcpy(colInd, tcolInd, sizeof(float)*nnv);
	memcpy(val, tval, sizeof(float)*nnv);

	free(tcolInd);
	free(tval);
}
void print_matrix(float*data, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			int l = i + j * m;
			cout << data[l] << " ";
		}
		cout << endl;
	}
	cout << endl;
}
void print_matrix(int*data, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			int l = i + j * m;
			cout << data[l] << " ";
		}
		cout << endl;
	}

	cout << endl;
}

#endif
