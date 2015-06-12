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

inline float2 induced_velocity_single(float2 pos, float4 vort) {
  // vortex strength is vort.z
  const float eps = 1.e-6;
  const float2 r = {pos.x - vort.x, pos.y - vort.y};
  float rsq = r.x * r.x + r.y * r.y + eps;
  float2 vel = {vort.z * r.y / rsq, -vort.z * r.x / rsq};
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

__global__ void induced_vel_kernel(const float2 *pos,
                     const float4 *vort,
                     float2 *vel_out,
                     const int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const float2 p = pos[i];
  float2 vel = {0.,0.};
  for (int j = 0; j < N; ++j) {
    const float eps = 1.e-6;
    const float2 r = {p.x - vort[j].x, p.y - vort[j].y};
    float fac = vort[j].z / (r.x * r.x + r.y * r.y + eps);
    vel.x +=  r.y * fac;
    vel.y += -r.x * fac;
  }
  vel_out[i] = vel;
}

void induced_vel_gpu(const float2 *pos,
                     const float4 *vort,
                     float2 *vel,
                     const int N) {
  dim3 threads(128);
  dim3 blocks((N + threads.x - 1)/threads.x);
  induced_vel_kernel<<<blocks, threads>>>(pos, vort, vel, N);
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
  double Mflops = (double) (6 * N * N) / (1000 * 1000 * 1000 * time);
  printf("%s: %f GFlops\n", label, Mflops);
}


int main() {
  const int N = 8192;
  float4 *vort = NULL;
  float2 *vel = NULL;

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

  time_induced_vel((char *)"CPU", induced_vel_reference, vort, vel, N, false);
  time_induced_vel((char *)"CPU + OMP", induced_vel_omp, vort, vel, N, false);
  time_induced_vel((char *)"GPU", induced_vel_gpu, vort, vel, N, true);

  cudaFree(vort);
  cudaFree(vel);
  return 0;
}
