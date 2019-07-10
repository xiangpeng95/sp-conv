#ifndef MAT_MUL_H
#define MAT_MUL_H

#include <stdio.h>
#include <time.h>


void gpu_compute(float* h_M, float* h_N, float* h_P, int C, int M, int K, int H, int W);


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

#endif
