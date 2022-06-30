#include "rumd/HydrodynamicCorrelations.h"
#include "rumd/RUMD_Error.h"
#include <math.h>
/*
int col = blockIdx.x*blockDim.x+threadIdx.x;
int row = blockIdx.y*blockDim.y+threadIdx.y;
int index = col + row * N;
A[index] = ...
*/

HydrodynamicCorrelations::HydrodynamicCorrelations(Sample* sample, unsigned lvecInput, float Dt, unsigned nwave){
  max_threads = 512;
  
  length_vector   = lvecInput;
  num_wavevectors = nwave;
  dt_sample       = Dt;
  length_box = sample->GetSimulationBox()->GetLength(2); // Wwe use wavevectors in the z-direction 
  index       = 0;
  num_samples = 0;

  num_particles = sample->GetNumberOfParticles();
  particleData = sample->GetParticleData();

  if ( num_particles%max_threads == 0)
    num_loops = num_particles/max_threads;
  else
    num_loops = num_particles/max_threads + 1;

  AllocateMemory();

  FILE *fout = fopen("hc_wavevectors.dat", "w");
  if ( fout == NULL )
    throw RUMD_Error("HydrodynamicCorrelations","HydrodynamicCorrelations","Couldn't open file");

  for ( size_t n=0; n<nwave; n++ ){
    k_host[n].x = 0.0;
    k_host[n].y = 0.0;
    k_host[n].z = 2*M_PI*(n+1)/length_box;
    fprintf(fout, "%f %f %f\n", k_host[n].x, k_host[n].y, k_host[n].z);
  }

  fclose(fout);

  if ( cudaMemcpy(k_device, k_host, sizeof(float3)*num_wavevectors, cudaMemcpyHostToDevice) != cudaSuccess )
    throw RUMD_Error("HydrodynamicCorrelations","HydrodynamicCorrelations","Copy failure");

}

size_t HydrodynamicCorrelations::AllocateMemory(){

  size_t sum_bytes = 0;
  
  // For wavevectors
  size_t num_bytes = num_wavevectors*sizeof(float3); sum_bytes += 2*num_bytes;
  if ( cudaMalloc((void **)&k_device, num_bytes) == cudaErrorMemoryAllocation ||
       cudaMallocHost((void **)&k_host, num_bytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("HydrodynamicCorrelations","AllocateMemory","Memory allocation failure");

  
  // For rho 
  num_bytes = num_wavevectors*num_particles*sizeof(cuComplex);  sum_bytes += 2*num_bytes;
  if ( cudaMalloc((void **)&rho_k, num_bytes) == cudaErrorMemoryAllocation ||
       cudaMalloc((void **)&rho_mk, num_bytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("HydrodynamicCorrelations","AllocateMemory","Memory allocation failure");

 
  num_bytes = num_wavevectors*length_vector*sizeof(cuComplex); sum_bytes += 2*num_bytes;
  if ( cudaMalloc((void **)&rho_k_sum, num_bytes) == cudaErrorMemoryAllocation ||
       cudaMalloc((void **)&rho_mk_sum, num_bytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("HydrodynamicCorrelations","AllocateMemory","Memory allocation failure");

  
  num_bytes = num_wavevectors*length_vector*sizeof(float); sum_bytes += 2*num_bytes;
  if ( cudaMalloc((void **)&rho_acf_device, num_bytes) == cudaErrorMemoryAllocation ||
       cudaMallocHost((void **)&rho_acf_host, num_bytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("HydrodynamicCorrelations","AllocateMemory","Memory allocation failure");

  SetArrayElement<<<num_wavevectors, 1>>>(rho_acf_device, num_wavevectors*length_vector, 0.0);
  
  // For trans. velocity
  num_bytes = num_wavevectors*num_particles*sizeof(cuComplex);  sum_bytes += 2*num_bytes;
  if ( cudaMalloc((void **)&vel_k, num_bytes) == cudaErrorMemoryAllocation ||
       cudaMalloc((void **)&vel_mk, num_bytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("HydrodynamicCorrelations","AllocateMemory","Memory allocation failure");

 
  num_bytes = num_wavevectors*length_vector*sizeof(cuComplex); sum_bytes += 2*num_bytes;
  if ( cudaMalloc((void **)&vel_k_sum, num_bytes) == cudaErrorMemoryAllocation ||
       cudaMalloc((void **)&vel_mk_sum, num_bytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("HydrodynamicCorrelations","AllocateMemory","Memory allocation failure");
  
  num_bytes = num_wavevectors*length_vector*sizeof(float);  sum_bytes += 2*num_bytes;
  if ( cudaMalloc((void **)&vel_acf_device, num_bytes) == cudaErrorMemoryAllocation ||
       cudaMallocHost((void **)&vel_acf_host, num_bytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("HydrodynamicCorrelations","AllocateMemory","Memory allocation failure");


  SetArrayElement<<<num_wavevectors, 1>>>(vel_acf_device, num_wavevectors*length_vector, 0.0);

  return sum_bytes;
}


HydrodynamicCorrelations::~HydrodynamicCorrelations(){

  cudaFree(k_device);   cudaFreeHost(k_host);

  cudaFreeHost(rho_acf_host); cudaFree(rho_acf_device);
  cudaFree(rho_k);   cudaFree(rho_mk);
  cudaFree(rho_k_sum);   cudaFree(rho_mk_sum);

  cudaFreeHost(vel_acf_host); cudaFree(vel_acf_device);
  cudaFree(vel_k);   cudaFree(vel_mk);
  cudaFree(vel_k_sum);   cudaFree(vel_mk_sum);
  
}

  

void HydrodynamicCorrelations::Compute(Sample* __attribute__((unused))sample){
    
  //cudaDeviceSynchronize();
  // This is probably not necessary because kernels or calls to cudaMemcpy
  // won't start until previous ones have finished anyway
  
  for ( size_t n=0; n<num_loops; n++ ){
    
    CalcRhoK<<<num_wavevectors, max_threads>>>(rho_k, rho_mk, k_device,
					       particleData->d_r, num_particles, max_threads, n);
    CalcVelK<<<num_wavevectors, max_threads>>>(vel_k, vel_mk, k_device, particleData->d_r,
					       particleData->d_v, num_particles,  max_threads, n);
  }
    
  //cudaDeviceSynchronize();
  
  SumComplexArray<<<num_wavevectors, 1>>>(rho_k_sum, rho_k, index, length_vector, num_particles);
  SumComplexArray<<<num_wavevectors, 1>>>(rho_mk_sum, rho_mk, index,
					  length_vector, num_particles);
  
  SumComplexArray<<<num_wavevectors, 1>>>(vel_k_sum, vel_k, index, length_vector, num_particles);
  SumComplexArray<<<num_wavevectors, 1>>>(vel_mk_sum, vel_mk, index, length_vector, num_particles);
  
  //cudaDeviceSynchronize();
  index ++;
  
  if ( index==length_vector ){
    index = 0;
    num_samples ++;
    
    
    CalcCorrReal<<<num_wavevectors, 1>>>(rho_acf_device, rho_k_sum, rho_mk_sum, length_vector);
    CalcCorrReal<<<num_wavevectors, 1>>>(vel_acf_device, vel_k_sum, vel_mk_sum, length_vector);
    
    //cudaDeviceSynchronize();
    
    CopyArrays();
    
    // Good old working C-formatted style... aahhh :)
    FILE *fout_rho = fopen("hc_rho_acf.dat", "w");
    FILE *fout_trans_vel = fopen("hc_transvel_acf.dat", "w");
    if ( fout_rho == NULL || fout_trans_vel == NULL )
      throw RUMD_Error("HydrodynamicCorrelations", __func__, "Couldn't open file");
    
    float volume = length_box*length_box*length_box;
    for ( unsigned n=0; n<length_vector; n++ ){
      fprintf(fout_rho, "%f ", dt_sample*n);
      fprintf(fout_trans_vel, "%f ", dt_sample*n);
      double fac = 1.0/(num_samples*(length_vector-n)*volume);
      for ( unsigned k=0; k<num_wavevectors; k++ ){
	fprintf(fout_rho, "%f ", rho_acf_host[k*length_vector+n]*fac);
	fprintf(fout_trans_vel, "%f ", vel_acf_host[k*length_vector+n]*fac);
      }
	fprintf(fout_rho, "\n");
	fprintf(fout_trans_vel, "\n");
    }
    
    fclose(fout_rho); fclose(fout_trans_vel);
  }
  
  
}

__global__ void CalcRhoK(cuComplex *rhok, cuComplex *rhomk, float3 *k, float4 *r,
			 size_t npart, size_t max_threads, size_t loopindex){
  cuComplex exponent;

  int index = threadIdx.x + loopindex*max_threads;

  if ( index < npart ) {
    exponent.x = 0.0;
    exponent.y =  k[blockIdx.x].x*r[index].x + k[blockIdx.x].y*r[index].y +
      k[blockIdx.x].z*r[index].z;

    size_t offset = index + npart*blockIdx.x;

    rhok[offset]  = HC_cexp(exponent);
    exponent.y    = -exponent.y;
    rhomk[offset] = HC_cexp(exponent);
  }

}

__global__ void CalcVelK(cuComplex *velk, cuComplex *velmk, float3 *k, float4 *r,
			 float4 *v, size_t npart, size_t max_threads, size_t loopindex){
  cuComplex exponent;

  int index = threadIdx.x + loopindex*max_threads;
  
  if ( index < npart ) {
    
    float velx = v[index].x;
  
    exponent.x = 0.0;
    exponent.y =   k[blockIdx.x].x*r[index].x + k[blockIdx.x].y*r[index].y +
      k[blockIdx.x].z*r[index].z;

    size_t offset = index + npart*blockIdx.x;
    
    velk[offset]  = HC_cexp(exponent);
    velk[offset].x *= velx; velk[offset].y *=velx;
    
    exponent.y   = -exponent.y;
    velmk[offset] = HC_cexp(exponent);
    velmk[offset].x *= velx; velmk[offset].y *=velx;
    
  }

}



void HydrodynamicCorrelations::CopyArrays(){

  cudaMemcpy(rho_acf_host, rho_acf_device, sizeof(float)*length_vector*num_wavevectors,
	     cudaMemcpyDeviceToHost);
  cudaMemcpy(vel_acf_host, vel_acf_device, sizeof(float)*length_vector*num_wavevectors,
	     cudaMemcpyDeviceToHost);

}


__global__ void CalcCorrReal(float *corr, cuComplex *a, cuComplex *b, size_t lvec){

  size_t offset = lvec*blockIdx.x;

  for ( size_t n=0; n<lvec; n++ ){
    for ( size_t nn=0; nn<lvec-n; nn++ ){
      float real = a[offset+nn].x*b[offset+n+nn].x;
      float imag = a[offset+nn].y*b[offset+n+nn].y;

      corr[offset+n] += real - imag;
    }
  }
  
}

__global__ void SetArrayElement(float *corr, size_t length, float value){

  size_t offset = length*blockIdx.x;

  for ( size_t n=0; n<length; n++ ) corr[offset+n] = value;   
  
}

__global__ void SumComplexArray(cuComplex *a_sum,  cuComplex *a, size_t index, size_t lvec, size_t npart){

  float real = 0.0;
  float imag = 0.0;

  size_t offset = npart*blockIdx.x;

  for ( size_t n=0; n<npart; n++ ){
    real += a[offset + n].x;
    imag += a[offset + n].y;
  }

  a_sum[index+lvec*blockIdx.x].x = real;
  a_sum[index+lvec*blockIdx.x].y = imag;
}
  



__device__ __forceinline__ cuComplex HC_cexp(cuComplex z){ 

  cuComplex res;

  float a = __expf(z.x);
  float b = __cosf(z.y);
  float c = __sinf(z.y);

  res.x = a*b;
  res.y = a*c;
  
  return res;
}

__global__ void printArrayComplex(cuComplex *a, int length){

  printf("-------------------\n");
  for ( int n=0; n<length; n++ )
    printf("%f %f\n", a[n].x, a[n].y);

}

__global__ void printArrayReal(float *a, int length){

  printf("-------------------\n");
  for ( int n=0; n<length; n++ )
    printf("%f\n", a[n]);

}
