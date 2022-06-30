
#ifndef HYDRODYNAMICCORRELATIONS_H
#define HYDRODYNAMICCORRELATIONS_H

#include <cuComplex.h>
#include "rumd/Sample.h"

class HydrodynamicCorrelations {
 public:

  HydrodynamicCorrelations(Sample* sample, unsigned lvecInput,
			   float Dt, unsigned nwave);
  ~HydrodynamicCorrelations();
  
  void Compute(Sample* sample);

 protected:
  const ParticleData* particleData;

 private:

  unsigned length_vector;
  unsigned num_wavevectors;
  unsigned num_particles;
  
  float dt_sample;
  float length_box;
  
  size_t index;
  size_t num_samples;
  
  
  size_t num_loops;
  size_t max_threads;
  
  // Wavevector arrays
  float3 *k_device, *k_host;

  // Density auto-correlations
  cuComplex *rho_k, *rho_mk;
  cuComplex *rho_k_sum, *rho_mk_sum;
  float  *rho_acf_device, *rho_acf_host;

  // Transverse velocity auto-correlations
  cuComplex *vel_k, *vel_mk;
  cuComplex *vel_k_sum, *vel_mk_sum;
  float  *vel_acf_device, *vel_acf_host;

  // Aux functions
  size_t AllocateMemory();
  void CopyArrays();
  
  
};

// Kernels
__global__ void CalcCorrReal(float *corr, cuComplex *a, cuComplex *b, size_t length);
__global__ void CalcRhoK(cuComplex *rhok, cuComplex *rhomk, float3 *k, float4 *r,
			 size_t npart, size_t max_threads, size_t loopindex);
__global__ void CalcVelK(cuComplex *velk, cuComplex *velmk, float3 *k, float4 *r,
			 float4 *v, size_t npart, size_t max_threads, size_t loopindex);

__device__ __forceinline__ cuComplex HC_cexp(cuComplex z);
__global__ void SetArrayElement(float *corr, size_t length, float value);
__global__ void SumComplexArray(cuComplex *a_sum,  cuComplex *a, size_t index, size_t lvec, size_t npart);
__global__ void printArrayComplex(cuComplex *a, int length);
__global__ void printArrayReal(float *a, int length);

#endif



