
#include "rumd_base.h"
#include <thrust/device_vector.h>

#ifndef PARTICLEDATA_H
#define PARTICLEDATA_H

/*
    Copyright (C) 2010  Thomas Schrøder
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/


class ParticleData{
  
 private:
  ParticleData(const ParticleData&);
  
  unsigned int numParticles; // The number of real particles.
  unsigned int numVirtualParticles; // The number of 'particles' assigned threads (>=numParticles).
  unsigned int allocatedNumParticles; // Internal variable for managing dynamic allocation.  
  unsigned int numBlocks; // The number of blocks.

  std::vector<unsigned int> numberOfType;
  std::vector<float> massOfType;

  void AllocateParticles(unsigned int nvp, bool copy_data=false);
  void FreeParticles();
  
 public:
  ParticleData();
  ~ParticleData();
  ParticleData& operator=(const ParticleData& P);
  void CopyHostData(const ParticleData& P);

  // Host.
  float4* h_r;             // [x, y, z, type]
  float4* h_v;             // [vx, vy, vz, 1/mass]
  float4* h_f;             // [fx, fy, fz, u]
  float4* h_w;             // [w_vdw_mol, sxz, sxy, w_vdw_atom]
  float4* h_im;            // [imx, imy, imz, 0]
  float4* h_sts;           // [sxx, syy, szz, syz]
  float4* h_misc;          // [ConfigT]
  unsigned int* h_Type;    // Type (on cpu) 
  
  // Device.
  float4* d_r; 
  float4* d_v; 
  float4* d_f;
  float4* d_w;
  float4* d_im;
  float4* d_sts;
  float4* d_misc; 
  float4* d_temp;            
  unsigned int* d_unsorted_index; // d_unsorted_index[i] = Original (before sorting) index of 
                                  // particle that now has index 'i'.
                                  // By convention particles are allways left unsorted on host
  unsigned int* d_temp_uint; 

  void SetForcesToZero() const;
 
  // Functions for transferring data to the device
  void CopyPosToDevice(bool reset_sorting=true) const;
  void CopyVelToDevice() const;
  void CopyForToDevice() const;
  void CopyImagesToDevice() const;
  void CopyConfToDevice() const; // all of them together

  void UpdateAfterSorting(thrust::device_vector<unsigned int>&  thrust_old_index);

  // Functions for transferring data from the device
  void CopyPosFromDevice(bool sync=true) const;
  void CopyVelFromDevice(bool sync=true) const;
  void CopyForFromDevice(bool sync=true) const;
  void CopyVirFromDevice(bool sync=true) const;
  void CopyImagesFromDevice(bool sync=true) const;
  void CopyStressFromDevice(bool sync=true) const;
  void CopyMiscFromDevice(bool sync=true) const;
  void CopyConfFromDevice(bool sync=true) const; // all of them together

  // this copies positions and images to a other, user-passed device pointers,
  // while unsorting
  void CopyPosImagesDevice(float4* d_r_dest, float4* d_im_dest) const;
  
  // Get methods
  unsigned int GetNumberOfParticles() const { return numParticles; }
  unsigned int GetNumberOfVirtualParticles() const { return numVirtualParticles; }
  unsigned int GetNumberOfBlocks() const { return numBlocks; }
  unsigned int GetNumberOfTypes() const { return numberOfType.size(); }  
  unsigned int GetNumberThisType(unsigned int type) const { if(type >= numberOfType.size()) return 0; else return numberOfType[type]; }
  float GetMass(unsigned int type) const {
    if(type >= numberOfType.size())
      throw RUMD_Error("ParticleData", __func__, "Type index too large");
    return massOfType[type]; }  
  
  // Set methods
  void SetNumberOfParticles(unsigned int num_part, unsigned int pb);
  void SetNumberOfTypes( unsigned n_types ){ 
    numberOfType.resize(n_types);
    massOfType.resize(n_types);
  }
  void SetNumberThisType( unsigned type, unsigned howMany ){ 
    if(type >= numberOfType.size())
      throw RUMD_Error("ParticleData", __func__, "type index too large");
    numberOfType[type] = howMany;
  }
  void SetMass( unsigned type, float mass ){ 
    if(type >= numberOfType.size())
      throw RUMD_Error("ParticleData", __func__, "Type index too large");
    massOfType[type] = mass;
    UpdateParticleMasses();
  }
  void SetAllMasses( double* mass_array, int length);

  void UpdateParticleMasses();

  
  // Functions that work on device data
  void IsotropicScalePositions(float Rscal);
  void AnisotropicScalePositions(float Rscal, unsigned dir);
  void AffinelyShearPositions(float shear_strain);
  void ScaleVelocities(float factor);
  void ApplyLeesEdwardsWrapToImages(float wrap);
};

// Kernels 

__global__ void IsotropicScalePositionsKernel(float4 *r, unsigned numParticles, float Rscal);
__global__ void AnisotropicScalePositionsKernel(float4 *r, unsigned numParticles, float Rscal, unsigned dir);
__global__ void AffinelyShearPositionsKernel(float4 *r, unsigned numParticles, float shear_strain);
__global__ void ScaleVelocitiesKernel(float4 *v, unsigned numParticles, float factor);

__global__ void ApplyLeesEdwardsWrapToImagesKernel( float4 *image, unsigned numParticles, float wrap );

#endif // PARTICLEDATA_H
