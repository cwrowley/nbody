all: nbody

% : %.cu
	nvcc $< -O3 -o $@ -g -G -lineinfo -Xcompiler -fopenmp
