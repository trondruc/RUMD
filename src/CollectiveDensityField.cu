
#include "rumd/rumd_technical.h"
#include "rumd/rumd_algorithms.h"

#include "rumd/CollectiveDensityField.h"

#include "rumd/ParticleData.h"
#include "rumd/SimulationBox.h"
#include <iostream>


__global__ void calculate_cos_sin(unsigned int numParticles,  float3 k, float4* position, float2* cos_sin);

__global__ void calculate_rho_k_prefactor(float sqrt_numParticles, float3* rho_k_pref, float2* cos_sin, float kappa, float a);

__global__ void collective_density_field_calcf (unsigned int numParticles, float3 k, float3* rho_k_pref, float4* d_r, float4* d_f);



CollectiveDensityField::CollectiveDensityField() : nx(0), ny(0), nz(0), kappa(0.), a(0.), d_cos_sin(0), d_rho_k_pref(0), testRSB(0), nAllocatedParticles(0) {
  SetID_String("cdf");
  // allocate space for rho_k on device
  if( cudaMalloc( (void**) &d_rho_k_pref , sizeof(float3) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("CollectiveDensityField","CollectiveDensityField","Malloc failed on d_rho_k_pref") );

  std::cerr << "Warning [CollectiveDensityField]: The contribution to the virial has not been implemented yet." << std::endl;
}

CollectiveDensityField::~CollectiveDensityField() {
  if(nAllocatedParticles > 0)
    cudaFree(d_cos_sin);

  cudaFree(d_rho_k_pref);
}


void CollectiveDensityField::Initialize() {

  testRSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  if(!testRSB) throw RUMD_Error("CollectiveDensityField", "SetSampleParameters", "Only RectangularSimulationBox is possible for now");

  unsigned int nParticles = particleData->GetNumberOfParticles();

  if(nParticles != nAllocatedParticles) {
  
  if(nAllocatedParticles > 0)
    cudaFree(d_cos_sin);  
    
    // Allocate space for the summation of cosines and sines on GPU.
    if( cudaMalloc( (void**) &d_cos_sin, nParticles * sizeof(float2) ) == cudaErrorMemoryAllocation )
      throw( RUMD_Error("CollectiveDensityField","SetSampleParameters","Malloc failed on d_cos_sin") );
    nAllocatedParticles = nParticles;
  }
}

void CollectiveDensityField::SetParams(unsigned int nx, unsigned int ny, unsigned int nz, float kappa, float a) {
  this->nx = nx;
  this->ny = ny;
  this->nz = nz;
  this->kappa = kappa;
  this->a = a;
}


void CollectiveDensityField::GetDataInfo(std::map<std::string, bool> &active, std::map<std::string, std::string> &columnIDs) {
  active["rho_k_mag"] = false;
  columnIDs["rho_k_mag"] = "rho_k_mag";
}

void CollectiveDensityField::RemoveDataInfo(std::map<std::string, bool> &active, std::map<std::string, std::string> &columnIDs) {
  active.erase("rho_k_mag");
  columnIDs.erase("rho_k_mag");

}


void CollectiveDensityField::GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active) {
  if(active["rho_k_mag"])
    dataValues["rho_k_mag"] = GetCollectiveDensity();
}


float  CollectiveDensityField::GetCollectiveDensity() {
  float3 h_rho_k_pref;
  cudaMemcpy( &h_rho_k_pref, d_rho_k_pref, sizeof(float3), cudaMemcpyDeviceToHost );
  float rho_k_mag = sqrt(h_rho_k_pref.x*h_rho_k_pref.x + h_rho_k_pref.y*h_rho_k_pref.y);
  return rho_k_mag;
}

float  CollectiveDensityField::GetCollectiveDensity(float* rho_k_re, float* rho_k_im) {
  float3 h_rho_k_pref;
  cudaMemcpy( &h_rho_k_pref, d_rho_k_pref, sizeof(float3), cudaMemcpyDeviceToHost );
  *rho_k_re = h_rho_k_pref.x;
  *rho_k_im = h_rho_k_pref.y;
  float rho_k_mag = sqrt(h_rho_k_pref.x*h_rho_k_pref.x + h_rho_k_pref.y*h_rho_k_pref.y);
  return rho_k_mag;
}

double CollectiveDensityField::GetPotentialEnergy() {
  float rho_k_mag = GetCollectiveDensity();
  double diff = rho_k_mag-a;
  return 0.5*kappa*diff*diff;
}

void CollectiveDensityField::CalcF(bool initialize, bool __attribute__((unused))calc_stresses) {
  static float two_pi = 8.f*atan(1.0);
  float4 box = testRSB->GetSimulationBox();
  float3 k = {two_pi*nx/box.x, two_pi*ny/box.y, two_pi*nz/box.z};
  unsigned int nParticles = particleData->GetNumberOfParticles();
  
  dim3 threads = kp.threads;
  threads.y = 1;
  // call kernel to calculate cosines and sines in an array
  calculate_cos_sin<<<kp.grid, threads >>>(nParticles, k, 
					   particleData->d_r, d_cos_sin);
  
  // sum them up, result is in zero position
  sumIdenticalArrays( d_cos_sin, nParticles, 1, 32 );
  
  calculate_rho_k_prefactor<<<1,1 >>>(sqrt(nParticles), d_rho_k_pref, d_cos_sin, kappa, a);
  
  // call kernel to calculate forces
  if(initialize)
    particleData->SetForcesToZero();

  collective_density_field_calcf<<<kp.grid, threads >>>(nParticles, k, d_rho_k_pref, particleData->d_r, particleData->d_f);

}


__global__ void calculate_cos_sin(unsigned int numParticles,  float3 k, float4* position, float2* cos_sin) {
  if ( MyGP < numParticles ) {
    float4 r = position[MyGP];
    float k_dot_r = k.x*r.x + k.y*r.y + k.z*r.z;
    float2 cos_sin_k_dot_r = {cos(k_dot_r), sin(k_dot_r)};
    cos_sin[MyGP] = cos_sin_k_dot_r;
  }
}

__global__ void calculate_rho_k_prefactor(float sqrt_numParticles, float3* rho_k_pref, float2* cos_sin, float kappa, float a) {

  float2 cos_sin_value = cos_sin[0];
  float rho_k_re = cos_sin_value.x/sqrt_numParticles;
  float rho_k_im = -cos_sin_value.y/sqrt_numParticles;
  float rho_k_mag = sqrt(rho_k_re*rho_k_re + rho_k_im*rho_k_im);

  float prefactor = kappa*(rho_k_mag -a)/(rho_k_mag*sqrt_numParticles);
  float3 result = {rho_k_re, rho_k_im, prefactor};
  rho_k_pref[0] = result;
}



__global__ void collective_density_field_calcf (unsigned int numParticles, float3 k, float3* rho_k_re_pref, float4* position, float4* force) {
  if ( MyGP < numParticles ){
    float4 r = position[MyGP];
    float k_dot_r = k.x*r.x + k.y*r.y + k.z*r.z;
    float cos_k_dot_r = cos(k_dot_r);
    float sin_k_dot_r = sin(k_dot_r);
    float3 my_rho_k_pref = rho_k_re_pref[0];
    float rho_k_re = my_rho_k_pref.x;
    float rho_k_im = my_rho_k_pref.y;
    float prefactor = my_rho_k_pref.z;

    float pref_rho_k_sin_cos = prefactor*(rho_k_re * sin_k_dot_r + 
					  rho_k_im * cos_k_dot_r);
    float4 my_f = {k.x*pref_rho_k_sin_cos, k.y*pref_rho_k_sin_cos, k.z*pref_rho_k_sin_cos, 0.0f};

    atomicFloatAdd(&(force[MyGP].x), my_f.x);
    atomicFloatAdd(&(force[MyGP].y), my_f.y);
    atomicFloatAdd(&(force[MyGP].z), my_f.z);
  }
}
