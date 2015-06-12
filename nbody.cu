#include <omp.h>
#include <stdio.h>

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                       \
 cudaError_t e=cudaGetLastError();                               \
 if(e!=cudaSuccess) {                                            \
   printf("Cuda failure %s:%d: '%s'\n",                          \
          __FILE__,__LINE__,cudaGetErrorString(e));              \
   exit(0);                                                      \
 }                                                               \
}

inline float2 induced_velocity_single(float2 pos, float2 vort, float gam) {
  const float eps = 1.e-6;
  const float2 r = {pos.x - vort.x, pos.y - vort.y};
  float rsq = r.x * r.x + r.y * r.y + eps;
  float2 vel = {gam * r.x / rsq, -gam * r.y / rsq};
  return vel;
}

void induced_vel_reference(const float2 *pos,
                           const float2 *vort,
                           const float *gam,
                           float2 *vel,
                           const int N) {
  memset(vel, 0, N * sizeof(float2));
  for (int i = 0; i < N; ++i) {   // i indexes position
    for (int j = 0; j < N; ++j) { // j indexes vortices
      float2 v = induced_velocity_single(pos[i], vort[j], gam[j]);
      vel[i].x += v.x;
      vel[i].y += v.y;
    }
  }
}

void induced_vel_omp(const float2 *pos,
                           const float2 *vort,
                           const float *gam,
                           float2 *vel,
                           const int N) {
  // set number of threads with
  // export OMP_NUM_THREADS=6
  memset(vel, 0, N * sizeof(float2));
  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {   // i indexes position
    #pragma omp parallel for
    for (int j = 0; j < N; ++j) { // j indexes vortices
      float2 v = induced_velocity_single(pos[i], vort[j], gam[j]);
      vel[i].x += v.x;
      vel[i].y += v.y;
    }
  }
}

typedef void (*func_ptr)(const float2*, const float2*, const float*, float2*, const int);

const int NUM_REPS = 1;

void time_induced_vel(const char *label,
                      func_ptr fptr,
                      const float2 *vort,
                      const float *gam,
                      float2 *vel,
                      const int N) {
  // warm up
  fptr(vort, vort, gam, vel, N);

  double start = omp_get_wtime();
  for (int i = 0; i < NUM_REPS; ++i) {
    fptr(vort, vort, gam, vel, N);
  }
  double end = omp_get_wtime();

  // Check the answer: return time only if answer is correct.
  float2 *validation = (float2 *) malloc(N * sizeof(float2));
  induced_vel_reference(vort, vort, gam, validation, N);
  for (int i = 0; i < N; ++i) {
    if (vel[i].x != validation[i].x || vel[i].y != validation[i].y) {
      printf("Error: velocity is incorrect at index %d\n", i);
      return;
    }
  }
  free(validation);

  double time = (end - start) / ((double) NUM_REPS);
  double Mflops = (double) (6 * N * N) / (1000 * 1000 * 1000 * time);
  printf("%s: %f GFlops\n", label, Mflops);
}


int main() {
  const int N = 8192;
  float2 *vort = NULL;
  float2 *vel = NULL;
  float *gam = NULL;

  cudaMallocManaged(&vort, N * sizeof(float2));
  cudaMallocManaged(&vel, N * sizeof(float2));
  cudaMallocManaged(&gam, N * sizeof(float));
  cudaCheckError();

  // initialize vortex positions and strengths to random values
  for (int i = 0; i < N; ++i) {
    vort[i].x = rand() / (float) RAND_MAX;
    vort[i].y = rand() / (float) RAND_MAX;
    gam[i] = rand() / (float) RAND_MAX;
  }
  memset(vel, 0, N * sizeof(float2));

  time_induced_vel((char *)"CPU", induced_vel_reference, vort, gam, vel, N);
  time_induced_vel((char *)"CPU + OMP", induced_vel_omp, vort, gam, vel, N);

  cudaFree(vort);
  cudaFree(vel);
  cudaFree(gam);
  return 0;
}
