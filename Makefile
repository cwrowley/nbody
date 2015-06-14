all: nbody

% : %.cu
	nvcc $< -O3 -o $@ -lineinfo -Xcompiler -fopenmp -arch=sm_35 -lnvToolsExt
