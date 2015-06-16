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

void induced_vel_gpu(const float2 *pos,
                     const float4 *vort,
                     float2 *vel,
                     const int N) {
  dim3 threads(TILE_SIZE);
  dim3 blocks((N + threads.x - 1)/threads.x);
  induced_vel_kernel_smem<<<blocks, threads>>>(pos, vort, vel, N);
  cudaCheckError();
}

void induced_vel_gpu2(const float2 *pos,
                      const float4 *vort,
                      float2 *vel,
                      const int N) {
  dim3 threads(32, 32);
  dim3 blocks(1, (N + threads.y - 1) / threads.y);
  // dim3 threads(N, N);
  // dim3 blocks(1, 1);
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
  const int N = 1<<13;
  // const int N = 1024;
  float4 *vort = NULL;
  float2 *vel = NULL;

  cudaSetDevice(1);
  cudaCheckError();

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

  cudaFree(vort);
  cudaFree(vel);
  cudaDeviceReset();
  return 0;
}
