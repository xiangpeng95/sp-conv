CC = gcc
CFLAGS = -std=c99

NVCC = nvcc
NVCC_FLAGS = --gpu-architecture=sm_50 -std=c++11 -O3 -Wno-deprecated-gpu-targets

LIBRARIES = -L/gpfs/share/software/cuda/cuda-8.0/lib64 -lcudart -lm -lcublas


mat_mul: main.o mat_mul.o
	$(CC) $^ -o $@ $(LIBRARIES)

main.o: main.c
	$(CC) $(CFLAGS) -c $^ -o $@

mat_mul.o: mat_mul.cu
	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@

clean:
	rm -f *.o mat_mul
	
# nvcc -arch=sm_35 -o run run.cpp -lcublas