#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mat_mul.h"

int main(int argc, char **argv){
  
    int k = 3, C = 3, filter = 3;
    int output_H = 3, output_W = 3, M = 3;
    int input_H = 3, input_W = 3;

    srand(time(NULL));
    float* mat_weight = rand_1_mat(filter, C, k);
    float* mat_input = rand_2_mat(C, input_H, input_W);
    float* mat_output = rand_3_mat(M, output_H, output_W);

    float* com_mat_weight = switch_w_mat(filter, C, k, mat_weight);
    printf("hello world!!!!\n");

    /*for(int i = 0;i < M; i++){
        for(int j = 0;j < C;j++){
            for(int l = 0; l < k*k; l++){
                printf("%f ",mat_weight[k*k*C*i + k*k*j + l]);
            }
            printf("\n");
        }
        printf("\n\n\n");
    }
    for(int i = 0;i < C; i++){
        for(int j = 0;j < output_H * output_W;j++){
            printf("%f ",mat_input[output_H * output_W*i + j]);
        }
        printf("\n");
    }
    for(int i = 0;i < k*k; i++){
        for(int j = 0;j < M;j++){
            for(int l = 0; l < C; l++){
                printf("%f ",com_mat_weight[M*C*i + C*j + l]);
            }
            printf("\n");
        }
        printf("\n\n\n");
    }*/

    gpu_compute(com_mat_weight, mat_input, mat_output, C, filter, k, input_H, input_W);

    /*for(int i = 0;i < M; i++){
        for(int j = 0;j < output_H * output_W;j++){
            printf("%f ",mat_output[output_H * output_W*i + j]);
        }
        printf("\n");
    }*/
}
