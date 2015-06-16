#include <omp.h>
#include <stdio.h>
#include <nvToolsExt.h>

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                       \
 cudaError_t e=cudaGetLastError();                               \
 if(e!=cudaSuccess) {                                            \
   printf("Cuda failure %s:%d: '%s'\n",                          \
          __FILE__,__LINE__,cudaGetErrorString(e));              \
   exit(0);                                                      \
 }                                                               \
}

__device__ __host__
inline float2 induced_velocity_single(float2 pos, float4 vort) {
  // vortex strength is vort.z
  const float eps = 1.e-6;
  float dx = pos.x - vort.x;
  float dy = pos.y - vort.y;
  float fac = vort.z / (dx * dx + dy * dy + eps);
  float2 vel = {dy * fac, -dx * fac};
  return vel;
}

void induced_vel_reference(const float2 *pos,
                           const float4 *vort,
                           float2 *vel,
                           const int N) {
  memset(vel, 0, N * sizeof(*vel));
  for (int i = 0; i < N; ++i) {   // i indexes position
    for (int j = 0; j < N; ++j) { // j indexes vortices
      float2 v = induced_velocity_single(pos[i], vort[j]);
      vel[i].x += v.x;
      vel[i].y += v.y;
    }
  }
}

void induced_vel_omp(const float2 *pos,
                     const float4 *vort,
                     float2 *vel,
                     const int N) {
  // set number of threads with
  // export OMP_NUM_THREADS=6
  memset(vel, 0, N * sizeof(*vel));
  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {   // i indexes position
    #pragma omp parallel for
    for (int j = 0; j < N; ++j) { // j indexes vortices
      float2 v = induced_velocity_single(pos[i], vort[j]);
      vel[i].x += v.x;
      vel[i].y += v.y;
    }
  }
}

__global__ void induced_vel_kernel(const float2 * __restrict__ pos,
                     const float4 * __restrict__ vort,
                     float2 *vel_out,
                     const int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const float2 p = pos[i];
  float2 vel = {0.,0.};
  for (int j = 0; j < N; ++j) {
    float2 v = induced_velocity_single(p, vort[j]);
    vel.x += v.x;
    vel.y += v.y;
  }
  vel_out[i] = vel;
}

const int TILE_SIZE = 128;

__global__
void induced_vel_kernel_smem(const float2 *pos,
                     const float4 *vort,
                     float2 *vel_out,
                     const int N) {
  __shared__ float4 smem[TILE_SIZE];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = threadIdx.x;

  const float2 p = pos[i];
  float2 vel = {0.,0.};

  for (int j = t; j < N; j += TILE_SIZE) {
    // load a chunk of vortices into shared memory
    smem[t] = vort[j];
    __syncthreads();

    // compute contributions from each vortex in the chunk
    for (int k = 0; k < TILE_SIZE && t + blockIdx.x * TILE_SIZE < N; ++k) {
      float2 v = induced_velocity_single(p, smem[k]);
      vel.x += v.x;
      vel.y += v.y;
    }
    __syncthreads();
  }

  vel_out[i] = vel;
}

void induced_vel_gpu(const float2 *pos,
                     const float4 *vort,
                     float2 *vel,
                     const int N) {
  dim3 threads(TILE_SIZE);
  dim3 blocks((N + threads.x - 1)/threads.x);
  induced_vel_kernel_smem<<<blocks, threads>>>(pos, vort, vel, N);
  cudaCheckError();
}

__global__
void induced_vel_kernel2(const float2 * __restrict__ pos,
                              const float4 * __restrict__ vort,
                              float2 *vel_out,
                              const int N) {
  // __shared__ float2 smem[TILE_SIZE];

  for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < N; i += blockDim.y * gridDim.y) {
    // i indexes position
    // load positions into shared memory
    // smem[threadIdx.x] = pos[blockIdx.y * blockDim.y + threadIdx.x];
    // __syncthreads();

    float2 vel = {0.0, 0.0};
    float2 p = pos[i];
    // float2 p = smem[threadIdx.y];

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < N; j += blockDim.x * gridDim.x) {
      // j indexes vortices
      float2 v = induced_velocity_single(p, vort[j]);
      vel.x += v.x;
      vel.y += v.y;
    }
    atomicAdd(&vel_out[i].x, vel.x);
    atomicAdd(&vel_out[i].y, vel.y);
    // __syncthreads();
  }
}

const int THREADS_PER_POS = 32;
const int NUM_POS = 32;

void induced_vel_gpu2(const float2 *pos,
                      const float4 *vort,
                      float2 *vel,
                      const int N) {
  dim3 threads(THREADS_PER_POS, NUM_POS);
  dim3 blocks(1, (N + threads.y - 1) / threads.y);
  memset(vel, 0, N * sizeof(float2));
  induced_vel_kernel2<<<blocks, threads>>>(pos, vort, vel, N);
  cudaCheckError();
}

const int THREADS_X = 8;
const int NUM_ELEMS_PER_THREAD = 32 / THREADS_X;

__global__
void induced_vel_kernel3(const float2 * __restrict__ pos,
                         const float4 * __restrict__ vort,
                         float2 *vel_out,
                         const int N) {
  const float eps = 1.e-6;

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x * NUM_ELEMS_PER_THREAD + threadIdx.x;

  float2 vel = {0.0, 0.0};
  float2 p = pos[i];

  #pragma unroll
  for (int k = 0; k < NUM_ELEMS_PER_THREAD; ++k) {
    // j indexes vortices
    float4 v = vort[j + k * THREADS_X];
    float dx = p.x - v.x;
    float dy = p.y - v.y;
    float fac = v.z / (dx * dx + dy * dy + eps);
    vel.x +=  dy * fac;
    vel.y += -dx * fac;
  }
  atomicAdd(&vel_out[i].x, vel.x);
  atomicAdd(&vel_out[i].y, vel.y);
}

void induced_vel_gpu3(const float2 *pos,
                      const float4 *vort,
                      float2 *vel,
                      const int N) {
  dim3 threads(THREADS_X, 32);
  dim3 blocks((N + 31 - 1) / 32, (N + threads.y - 1) / threads.y);
  memset(vel, 0, N * sizeof(float2));
  induced_vel_kernel3<<<blocks, threads>>>(pos, vort, vel, N);
  cudaCheckError();
}

__global__
void induced_vel_kernel4(const float2 * __restrict__ pos,
                              const float4 * __restrict__ vort,
                              float2 *vel_out,
                              const int N) {
  // Like version 2, but use privatized memory for sums within a thread block
  // then do a single atomic add to global memory, for each row of the block
  __shared__ float2 smem[NUM_POS];
  const float2 zero = {0., 0.};

  for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < N; i += blockDim.y * gridDim.y) {
    // i indexes position

    // zero shared memory for private sum
    smem[threadIdx.y] = zero;
    __syncthreads();

    float2 vel = {0.0, 0.0};
    float2 p = pos[i];

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < N; j += blockDim.x * gridDim.x) {
      // j indexes vortices
      float2 v = induced_velocity_single(p, vort[j]);
      vel.x += v.x;
      vel.y += v.y;
    }
    atomicAdd(&smem[threadIdx.y].x, vel.x);
    atomicAdd(&smem[threadIdx.y].y, vel.y);
    // __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(&vel_out[i].x, smem[threadIdx.y].x);
      atomicAdd(&vel_out[i].y, smem[threadIdx.y].y);
    }
    // __syncthreads();
  }
}

void induced_vel_gpu4(const float2 *pos,
                      const float4 *vort,
                      float2 *vel,
                      const int N) {
  dim3 threads(THREADS_PER_POS, NUM_POS);
  /* dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y); */
  dim3 blocks(1, (N + threads.y - 1) / threads.y);
  memset(vel, 0, N * sizeof(float2));
  induced_vel_kernel4<<<blocks, threads>>>(pos, vort, vel, N);
  cudaCheckError();
}

__global__
void induced_vel_kernel5(const float2 * __restrict__ pos,
                              const float4 * __restrict__ vort,
                              float2 *vel_out,
                              const int N) {
  // Like version 4, but keep sums in shared memory array and do reduction
  // instead of atomic adds.
  // Then do a single copy into global memory.
  __shared__ float2 svel[NUM_POS][THREADS_PER_POS];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int i = blockIdx.y * blockDim.y + ty;   // i indexes position

  float2 vel = {0.0, 0.0};
  float2 p = pos[i];

  for (int j = blockIdx.x * blockDim.x + tx; j < N; j += blockDim.x * gridDim.x) {
    // j indexes vortices
    float2 v = induced_velocity_single(p, vort[j]);
    vel.x += v.x;
    vel.y += v.y;
  }
  svel[ty][tx] = vel;

  // compute total of each row in svel
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    __syncthreads();
    if (tx < stride) {
      svel[ty][tx].x += svel[ty][tx + stride].x;
      svel[ty][tx].y += svel[ty][tx + stride].y;
    }
  }
  /*
  // unroll manually  - this is a little slower on my workstation!
  // stride = 16
  __syncthreads();
  if (tx < 16) {
    svel[ty][tx].x += svel[ty][tx + 16].x;
    svel[ty][tx].y += svel[ty][tx + 16].y;
  }
  // stride = 8
  __syncthreads();
  if (tx < 8) {
    svel[ty][tx].x += svel[ty][tx + 8].x;
    svel[ty][tx].y += svel[ty][tx + 8].y;
  }
  // stride = 4
  __syncthreads();
  if (tx < 4) {
    svel[ty][tx].x += svel[ty][tx + 4].x;
    svel[ty][tx].y += svel[ty][tx + 4].y;
  }
  // stride = 2
  __syncthreads();
  if (tx < 2) {
    svel[ty][tx].x += svel[ty][tx + 2].x;
    svel[ty][tx].y += svel[ty][tx + 2].y;
  }
  // stride = 1
  __syncthreads();
  if (tx < 1) {
    svel[ty][tx].x += svel[ty][tx + 1].x;
    svel[ty][tx].y += svel[ty][tx + 1].y;
  }
  */

  __syncthreads();
  if (ty == 0) {
    // index using tx so that global write is coalesced
    // NOTE: thread block must be square (eg 32 x 32) for this to work
    vel_out[blockIdx.y * blockDim.y + tx] = svel[tx][0];
  }
}

void induced_vel_gpu5(const float2 *pos,
                      const float4 *vort,
                      float2 *vel,
                      const int N) {
  dim3 threads(THREADS_PER_POS, NUM_POS);
  /* dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y); */
  dim3 blocks(1, (N + threads.y - 1) / threads.y);
  memset(vel, 0, N * sizeof(float2));
  induced_vel_kernel5<<<blocks, threads>>>(pos, vort, vel, N);
  cudaCheckError();
}

typedef void (*func_ptr)(const float2*, const float4*, float2*, const int);

const int NUM_REPS = 1;

inline bool close(float2 a, float2 b) {
  const float rel_tol = 1.e-6;
  return fabs((a.x-b.x) * (a.x-b.x) + (a.y - b.y) * (a.y - b.y)) <
    rel_tol * fabs(a.x * a.x + a.y * a.y);
}

void time_induced_vel(const char *label,
                      func_ptr fptr,
                      const float4 *vort,
                      float2 *vel,
                      const int N,
                      bool cuda) {
  // warm up
  float2 *pos = NULL;
  cudaMallocManaged(&pos, N * sizeof(*pos));
  cudaCheckError();

  for (int i = 0; i < N; ++i) {
    pos[i].x = vort[i].x;
    pos[i].y = vort[i].y;
  }
  fptr(pos, vort, vel, N);

  if (cuda) cudaDeviceSynchronize();
  double start = omp_get_wtime();
  nvtxRangePushA(label);
  for (int i = 0; i < NUM_REPS; ++i) {
    fptr(pos, vort, vel, N);
  }
  if (cuda) cudaDeviceSynchronize();
  nvtxRangePop();
  double end = omp_get_wtime();

  // Check the answer: return time only if answer is correct.
  float2 *validation = (float2 *) malloc(N * sizeof(float2));
  induced_vel_reference(pos, vort, validation, N);
  for (int i = 0; i < N; ++i) {
    if (!close(vel[i], validation[i])) {
      printf("%s: Error: velocity is incorrect at index %d\n", label, i);
      printf("  expected (%f, %f)\n       got (%f, %f)\n",
             validation[i].x, validation[i].y, vel[i].x, vel[i].y);
      return;
    }
  }
  free(validation);
  cudaFree(pos);

  double time = (end - start) / ((double) NUM_REPS);
  double Mflops = (double) (6. * N * N) / (1000 * 1000 * 1000 * time);
  printf("%s: %f GFlops\n", label, Mflops);
}


int main() {
  // const int N = 8192;
  float4 *vort = NULL;
  float2 *vel = NULL;

  cudaSetDevice(1);
  cudaCheckError();

  const int min_exp = 8;
  const int max_exp = 13;

  for (int N = 1<<min_exp; N < (1<<max_exp) + 1; N <<= 1) {
    cudaMallocManaged(&vort, N * sizeof(*vort));
    cudaMallocManaged(&vel, N * sizeof(*vel));
    cudaCheckError();

    // initialize vortex positions and strengths to random values
    for (int i = 0; i < N; ++i) {
      vort[i].x = rand() / (float) RAND_MAX;
      vort[i].y = rand() / (float) RAND_MAX;
      vort[i].z = rand() / (float) RAND_MAX;
      vort[i].w = 0.;
    }
    memset(vel, 0, N * sizeof(float2));

    printf("N = %d\n", N);
    time_induced_vel((char *)"CPU", induced_vel_reference, vort, vel, N, false);
    time_induced_vel((char *)"CPU + OMP", induced_vel_omp, vort, vel, N, false);
    time_induced_vel((char *)"GPU", induced_vel_gpu, vort, vel, N, true);
    time_induced_vel((char *)"GPU v2", induced_vel_gpu2, vort, vel, N, true);
    time_induced_vel((char *)"GPU v3", induced_vel_gpu3, vort, vel, N, true);
    time_induced_vel((char *)"GPU v4", induced_vel_gpu4, vort, vel, N, true);
    time_induced_vel((char *)"GPU v5", induced_vel_gpu5, vort, vel, N, true);
    printf("  =====\n");

    cudaFree(vort);
    cudaFree(vel);
  }
  cudaDeviceReset();
  return 0;
}
