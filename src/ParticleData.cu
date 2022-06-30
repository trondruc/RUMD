/*
    Copyright (C) 2010  Thomas Schrøder

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/

#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>

#include "rumd/RUMD_Error.h"
#include "rumd/ParticleData.h"
#include "rumd/Device.h"

//////////////////////////////////////////////////
// CONSTRUCTION/DESTRUCTION 
//////////////////////////////////////////////////

ParticleData::ParticleData() :   numParticles(0),
				 numVirtualParticles(0),
				 allocatedNumParticles(0),
				 numBlocks(0),
				 numberOfType(),
				 massOfType()
{

}

ParticleData::~ParticleData(){
  FreeParticles();
}

// This copies device data only!
ParticleData& ParticleData::operator=(const ParticleData& P){
  if(this != &P){ 
    if(numParticles != P.GetNumberOfParticles())
      SetNumberOfParticles(P.GetNumberOfParticles(), P.GetNumberOfVirtualParticles()/P.GetNumberOfBlocks());

    cudaMemcpy( d_r, P.d_r, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_v, P.d_v, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_f, P.d_f, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_w, P.d_w, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_im, P.d_im, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_sts, P.d_sts, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_misc, P.d_misc, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_unsorted_index, P.d_unsorted_index, numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
  };
  return *this;   
}

// Copy only numParticles items in each array in case the other object
// has a different pb and hence number of virtual particles
void ParticleData::CopyHostData(const ParticleData& P) {
  if(numParticles != P.GetNumberOfParticles())
    SetNumberOfParticles(P.GetNumberOfParticles(), P.GetNumberOfVirtualParticles()/P.GetNumberOfBlocks());
  
  cudaMemcpy( h_r, P.h_r, numParticles * sizeof(float4), cudaMemcpyHostToHost );
  cudaMemcpy( h_v, P.h_v, numParticles * sizeof(float4), cudaMemcpyHostToHost );
  cudaMemcpy( h_f, P.h_f, numParticles * sizeof(float4), cudaMemcpyHostToHost );
  cudaMemcpy( h_w, P.h_w, numParticles * sizeof(float4), cudaMemcpyHostToHost );
  cudaMemcpy( h_im, P.h_im, numParticles * sizeof(float4), cudaMemcpyHostToHost );
  cudaMemcpy( h_sts, P.h_sts, numParticles * sizeof(float4), cudaMemcpyHostToHost );
  cudaMemcpy( h_misc, P.h_misc, numParticles * sizeof(float4), cudaMemcpyHostToHost );
  cudaMemcpy( h_Type, P.h_Type, numParticles * sizeof(unsigned int), cudaMemcpyHostToHost );

  SetNumberOfTypes(P.GetNumberOfTypes());

  for(unsigned int type = 0; type < GetNumberOfTypes(); type++)  {
    SetMass(type, P.GetMass(type));
    SetNumberThisType(type, P.GetNumberThisType(type));
  }
}

//////////////////////////////////////////////////
// MEMORY ALLOCATION/DEALLOCATION 
//////////////////////////////////////////////////

void ParticleData::SetNumberOfParticles(unsigned int set_numParticles, unsigned int pb) {
 
  numBlocks = (set_numParticles + pb - 1) / pb;    
  numVirtualParticles = numBlocks * pb;

  if(numBlocks > Device::GetDevice().GetMaximumGridDimensionX() )
    throw RUMD_Error("ParticleData",__func__, "Too many blocks for this device");
  bool copy_data = false;
  // we reallocate and copy data if the number of particles is unchanged
  // but the number of virtual particles is changed
  if(numParticles > 0 && set_numParticles == numParticles 
     &&  allocatedNumParticles != numVirtualParticles)
    copy_data = true;

  numParticles = set_numParticles;


  if( !( (numParticles <= pb*numBlocks) && (numVirtualParticles >= numParticles) && (numVirtualParticles <  (numParticles + pb)) ) )
    throw(RUMD_Error("ParticleData","SetNumberOfParticles","ParticleData could not validate internal variables. Report this error to RUMD developers"));
  
  // Allocate memory.
  AllocateParticles(numVirtualParticles, copy_data);
}

void ParticleData::AllocateParticles(unsigned int nvp, bool copy_data){
  if( allocatedNumParticles == nvp )
    return;
  
  // for saving the data (ie if only reallocating because pb has changed)
  // we need some extra pointers
  float4* h_r_new = 0; 
  float4* h_v_new = 0; 
  float4* h_f_new = 0;
  float4* h_w_new = 0;
  float4* h_im_new = 0;
  float4* h_sts_new = 0;
  float4* h_misc_new = 0;
  unsigned int* h_Type_new = 0;

  float4* d_r_new = 0; 
  float4* d_v_new = 0; 
  float4* d_f_new = 0;
  float4* d_w_new = 0;
  float4* d_im_new = 0;
  float4* d_sts_new = 0;
  float4* d_misc_new = 0; 
  float4* d_temp_new = 0; 
  unsigned int* d_unsorted_index_new = 0;
  unsigned int* d_temp_uint_new = 0; 

  // if not saving we free the memory first, to cut down on memory use
  if(!copy_data)
    FreeParticles();

  // Page-locked CPU Allocation
  if( cudaMallocHost( (void**) &h_r_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on h_r") );
  
  if( cudaMallocHost( (void**) &h_v_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on h_v") );
  
  if( cudaMallocHost( (void**) &h_f_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on h_f") );
  
  if( cudaMallocHost( (void**) &h_w_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on h_w") );
  
  if( cudaMallocHost( (void**) &h_im_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on h_im") );
  
  if( cudaMallocHost( (void**) &h_sts_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on h_sts") );
  
  if( cudaMallocHost( (void**) &h_misc_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on h_misc") );
  
  if( cudaMallocHost( (void**) &h_Type_new, nvp * sizeof(unsigned int) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on h_Type") );
  
  // GPU Allocation
  if( cudaMalloc( (void**) &d_r_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on d_r") );

  if( cudaMalloc( (void**) &d_v_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on d_v") );
  
  if( cudaMalloc( (void**) &d_f_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on d_f") );
  
  if( cudaMalloc( (void**) &d_w_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on d_w") );
  
  if( cudaMalloc( (void**) &d_im_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on d_im") );
  
  if( cudaMalloc( (void**) &d_sts_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on d_sts") );
  
  if( cudaMalloc( (void**) &d_misc_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on d_misc") );

  if( cudaMalloc( (void**) &d_temp_new, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on d_temp") );

  if( cudaMalloc( (void**) &d_unsorted_index_new, nvp * sizeof(unsigned int) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on d_unsorted_index") );

  if( cudaMalloc( (void**) &d_temp_uint_new, nvp * sizeof(unsigned int) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("ParticleData","AllocateParticles","Malloc failed on d_temp_uint") );

  // Initialize all CPU memory to zero from the start.
  memset( h_r_new,      0, nvp * sizeof(float4) );
  memset( h_v_new,      0, nvp * sizeof(float4) );
  memset( h_f_new,      0, nvp * sizeof(float4) );
  memset( h_w_new,      0, nvp * sizeof(float4) );
  memset( h_im_new,     0, nvp * sizeof(float4) );
  memset( h_sts_new,    0, nvp * sizeof(float4) );
  memset( h_misc_new,   0, nvp * sizeof(float4) );
  memset( h_Type_new,   0, nvp * sizeof(unsigned int) );

  // Initialize all GPU memory to zero from the start (needed+safety).
  cudaMemset( d_r_new,              0, nvp * sizeof(float4) );
  cudaMemset( d_v_new,              0, nvp * sizeof(float4) );
  cudaMemset( d_f_new,              0, nvp * sizeof(float4) );
  cudaMemset( d_w_new,              0, nvp * sizeof(float4) );
  cudaMemset( d_im_new,             0, nvp * sizeof(float4) );
  cudaMemset( d_sts_new,            0, nvp * sizeof(float4) );
  cudaMemset( d_misc_new,           0, nvp * sizeof(float4) );
  cudaMemset( d_temp_new,           0, nvp * sizeof(float4) );
  cudaMemset( d_unsorted_index_new, 0, nvp * sizeof(unsigned int) );
  cudaMemset( d_temp_uint_new,     0, nvp * sizeof(unsigned int) );

  if( cudaDeviceSynchronize() != cudaSuccess )
    throw( RUMD_Error("ParticleData","AllocateParticles","Initialization failed on GPU") );

  if(copy_data) {
    // copy from old arrays
    cudaMemcpy( h_r_new, h_r, numParticles * sizeof(float4), cudaMemcpyHostToHost );
    cudaMemcpy( h_v_new, h_v, numParticles * sizeof(float4), cudaMemcpyHostToHost );
    cudaMemcpy( h_f_new, h_f, numParticles * sizeof(float4), cudaMemcpyHostToHost );
    cudaMemcpy( h_w_new, h_w, numParticles * sizeof(float4), cudaMemcpyHostToHost );
    cudaMemcpy( h_im_new, h_im, numParticles * sizeof(float4), cudaMemcpyHostToHost );
    cudaMemcpy( h_sts_new, h_sts, numParticles * sizeof(float4), cudaMemcpyHostToHost );
    cudaMemcpy( h_misc_new, h_misc, numParticles * sizeof(float4), cudaMemcpyHostToHost );
    cudaMemcpy( h_Type_new, h_Type, numParticles * sizeof(unsigned int), cudaMemcpyHostToHost );
    

    cudaMemcpy( d_r_new, d_r, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_v_new, d_v, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_f_new, d_f, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_w_new, d_w, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_im_new, d_im, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_sts_new, d_sts, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_misc_new, d_misc, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_temp_new, d_temp, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_unsorted_index_new, d_unsorted_index, numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice );
    cudaMemcpy( d_temp_uint_new, d_temp_uint, numParticles * sizeof(unsigned int), cudaMemcpyDeviceToDevice );
 
    // Free old arrays
    FreeParticles();
  }

  // reassign the main pointers to the newly allocated arrays
  h_r = h_r_new;
  h_v = h_v_new;
  h_f = h_f_new;
  h_w = h_w_new;
  h_im = h_im_new;
  h_sts = h_sts_new;
  h_misc = h_misc_new;
  h_Type = h_Type_new;

  d_r = d_r_new;
  d_v = d_v_new;
  d_f = d_f_new;
  d_w = d_w_new;
  d_im = d_im_new;
  d_sts = d_sts_new;
  d_misc = d_misc_new;
  d_temp = d_temp_new;
  d_unsorted_index = d_unsorted_index_new;
  d_temp_uint = d_temp_uint_new;

  allocatedNumParticles = nvp;
}

void ParticleData::FreeParticles() {
  if(allocatedNumParticles == 0)
    return;
  
  cudaFreeHost(h_r);
  cudaFreeHost(h_v);
  cudaFreeHost(h_f);
  cudaFreeHost(h_w);
  cudaFreeHost(h_im);
  cudaFreeHost(h_sts);
  cudaFreeHost(h_misc);
  cudaFreeHost(h_Type);
  cudaFree(d_r);
  cudaFree(d_v);
  cudaFree(d_f);
  cudaFree(d_w);
  cudaFree(d_im);
  cudaFree(d_sts);  
  cudaFree(d_misc);
  cudaFree(d_temp);
  cudaFree(d_unsorted_index);
  cudaFree(d_temp_uint);
}

void ParticleData::SetAllMasses( double* mass_array, int length ) {
  if(length != (int) numberOfType.size())
    throw RUMD_Error("ParticleData", __func__, "Wrong length array passed");
  for(int idx=0;idx< length;idx++)
    massOfType[idx] = mass_array[idx];

  UpdateParticleMasses();
}
  

void ParticleData::UpdateParticleMasses() {
  if(numParticles) {
    for(unsigned idx=0; idx<numParticles;idx++)
      h_v[idx].w = 1.f/massOfType[h_Type[idx]];

    CopyVelToDevice();
  }
}


void ParticleData::SetForcesToZero() const {
  // Also potential energy, virial, stresses.
  // To be used as initialization by some potentials.

  // Note that const-ness is not "deep"; const here means the pointers cannot
  // be changed, while the data in the arrays can be
  cudaMemset( d_f, 0, numVirtualParticles * sizeof(float4) );
  cudaMemset( d_w, 0, numVirtualParticles * sizeof(float4) );
  cudaMemset( d_sts, 0, numVirtualParticles * sizeof(float4) );

}


//////////////////////////////////////////////////
// Copy to the device.
//////////////////////////////////////////////////

void ParticleData::CopyPosToDevice(bool reset_sorting) const{

  if(reset_sorting) {
    cudaMemcpy( d_r, h_r, numVirtualParticles * sizeof(float4), cudaMemcpyHostToDevice );
    
    // reset unsorted_index - happens in default case, for example when reading
    // in a new configuration from a file. It must be assumed that fresh
    // velocities are also being read in from the file.
    thrust::device_ptr<unsigned int> thrust_d_unsorted_index(d_unsorted_index);
    thrust::sequence(thrust_d_unsorted_index, thrust_d_unsorted_index + numParticles);
  }
  else
    {
      // distribute according to existing sorting, for example when using
      // SetPositions from python
      cudaMemcpy( d_temp, h_r, numVirtualParticles * sizeof(float4), cudaMemcpyHostToDevice );
      thrust::device_ptr<float4> thrust_d_r(d_r);
      thrust::device_ptr<float4> thrust_d_temp(d_temp);
      thrust::device_ptr<unsigned int> thrust_d_unsorted_index(d_unsorted_index);
      thrust::gather(thrust_d_unsorted_index, thrust_d_unsorted_index+numParticles, thrust_d_temp, thrust_d_r);      
    }    
}

void ParticleData::CopyVelToDevice() const{
  cudaMemcpy( d_v, h_v, numVirtualParticles * sizeof(float4), cudaMemcpyHostToDevice );
}

void ParticleData::CopyForToDevice() const{
  cudaMemcpy( d_f, h_f, numVirtualParticles * sizeof(float4), cudaMemcpyHostToDevice );
}

void ParticleData::CopyImagesToDevice() const{
  cudaMemcpy( d_im, h_im, numVirtualParticles * sizeof(float4), cudaMemcpyHostToDevice );
}

void ParticleData::CopyConfToDevice() const{
  CopyPosToDevice();
  CopyVelToDevice();
  CopyForToDevice();
  CopyImagesToDevice();
  
  // Check if the copy generated errors.
  if( cudaDeviceSynchronize() != cudaSuccess ) 
    throw( RUMD_Error("ParticleData", "CopyConfToDevice", "cudaMemcpy failed: simulation state => device") );
}

//////////////////////////////////////////////////
// Copy from the device.
//////////////////////////////////////////////////

void ParticleData::CopyPosFromDevice(bool sync) const{
  //cudaMemcpy( h_r, d_r, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );
  thrust::device_ptr<unsigned int> thrust_d_unsorted_index(d_unsorted_index);
  thrust::device_ptr<float4> thrust_d_r(d_r);
  thrust::device_ptr<float4> thrust_d_temp(d_temp);
  thrust::scatter(thrust_d_r, thrust_d_r + numParticles, thrust_d_unsorted_index, thrust_d_temp);
  if(sync)
    cudaMemcpy( h_r, d_temp, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );
  else
    cudaMemcpyAsync( h_r, d_temp, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );
}

void ParticleData::CopyPosImagesDevice(float4* d_r_dest, float4* d_im_dest) const {
  thrust::device_ptr<unsigned int> thrust_d_unsorted_index(d_unsorted_index);
  thrust::device_ptr<float4> thrust_d_r(d_r);
  thrust::device_ptr<float4> thrust_d_r_dest(d_r_dest);
  thrust::scatter(thrust_d_r, thrust_d_r + numParticles, thrust_d_unsorted_index, thrust_d_r_dest);

  thrust::device_ptr<float4> thrust_d_im(d_im);
  thrust::device_ptr<float4> thrust_d_im_dest(d_im_dest);
  thrust::scatter(thrust_d_im, thrust_d_im + numParticles, thrust_d_unsorted_index, thrust_d_im_dest);
}

void ParticleData::CopyVelFromDevice(bool sync) const{
  thrust::device_ptr<unsigned int> thrust_d_unsorted_index(d_unsorted_index);
  thrust::device_ptr<float4> thrust_d_v(d_v);
  thrust::device_ptr<float4> thrust_d_temp(d_temp);
  thrust::scatter(thrust_d_v, thrust_d_v + numParticles, thrust_d_unsorted_index, thrust_d_temp);
  if(sync)
    cudaMemcpy( h_v, d_temp, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );
  else
    cudaMemcpyAsync( h_v, d_temp, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );
}

void ParticleData::CopyForFromDevice(bool sync) const{
  thrust::device_ptr<unsigned int> thrust_d_unsorted_index(d_unsorted_index);
  thrust::device_ptr<float4> thrust_d_f(d_f);
  thrust::device_ptr<float4> thrust_d_temp(d_temp);
  thrust::scatter(thrust_d_f, thrust_d_f + numParticles, thrust_d_unsorted_index, thrust_d_temp);
  if(sync)
    cudaMemcpy( h_f, d_temp, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );
  else
    cudaMemcpyAsync( h_f, d_temp, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );

}

void ParticleData::CopyVirFromDevice(bool sync) const{
  thrust::device_ptr<unsigned int> thrust_d_unsorted_index(d_unsorted_index);
  thrust::device_ptr<float4> thrust_d_w(d_w);
  thrust::device_ptr<float4> thrust_d_temp(d_temp);
  thrust::scatter(thrust_d_w, thrust_d_w + numParticles, thrust_d_unsorted_index, thrust_d_temp);
  if(sync)
    cudaMemcpy( h_w, d_temp, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );
  else
    cudaMemcpyAsync( h_w, d_temp, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );

}

void ParticleData::CopyImagesFromDevice(bool sync) const{
  thrust::device_ptr<unsigned int> thrust_d_unsorted_index(d_unsorted_index);
  thrust::device_ptr<float4> thrust_d_im(d_im);
  thrust::device_ptr<float4> thrust_d_temp(d_temp);
  thrust::scatter(thrust_d_im, thrust_d_im + numParticles, thrust_d_unsorted_index, thrust_d_temp);
  if(sync)
    cudaMemcpy( h_im, d_temp, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );
  else
    cudaMemcpyAsync( h_im, d_temp, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );

}

void ParticleData::CopyStressFromDevice(bool sync) const{
  thrust::device_ptr<unsigned int> thrust_d_unsorted_index(d_unsorted_index);
  thrust::device_ptr<float4> thrust_d_sts(d_sts);
  thrust::device_ptr<float4> thrust_d_temp(d_temp);
  thrust::scatter(thrust_d_sts, thrust_d_sts + numParticles, thrust_d_unsorted_index, thrust_d_temp);
  if(sync)
    cudaMemcpy( h_sts, d_temp, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );
  else
    cudaMemcpyAsync( h_sts, d_temp, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );
}

void ParticleData::CopyMiscFromDevice(bool sync) const{ 
  thrust::device_ptr<unsigned int> thrust_d_unsorted_index(d_unsorted_index);
  thrust::device_ptr<float4> thrust_d_misc(d_misc);
  thrust::device_ptr<float4> thrust_d_temp(d_temp);
  thrust::scatter(thrust_d_misc, thrust_d_misc + numParticles, thrust_d_unsorted_index, thrust_d_temp);
  if(sync)
    cudaMemcpy( h_misc, d_temp, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );
  else
    cudaMemcpyAsync( h_misc, d_temp, numVirtualParticles * sizeof(float4), cudaMemcpyDeviceToHost );
}

void ParticleData::CopyConfFromDevice(bool sync) const{
  CopyPosFromDevice(sync);
  CopyVelFromDevice(sync);
  CopyForFromDevice(sync);
  CopyVirFromDevice(sync);
  CopyImagesFromDevice(sync);
  CopyStressFromDevice(sync);
  CopyMiscFromDevice(sync);

  // Check if the copy generated errors; make sure asynchronous copies finish
  if( cudaDeviceSynchronize() != cudaSuccess ) 
    throw( RUMD_Error("ParticleData","CopyConfFromDevice","cudaMemcpy failed: simulation state => host") );
}

//////////////////////////////////////////////
// Sorting of particles on device.
//////////////////////////////////////////////

void ParticleData::UpdateAfterSorting(thrust::device_vector<unsigned int>&  thrust_old_index) {

  // wrap raw pointers with thrust device_ptrs 
  thrust::device_ptr<float4> thrust_d_r(d_r);
  thrust::device_ptr<float4> thrust_d_im(d_im);
  thrust::device_ptr<float4> thrust_d_v(d_v);
  thrust::device_ptr<float4> thrust_d_f(d_f);
  thrust::device_ptr<float4> thrust_d_w(d_w);
  thrust::device_ptr<float4> thrust_d_misc(d_misc);
  thrust::device_ptr<float4> thrust_d_temp(d_temp);
  thrust::device_ptr<unsigned int> thrust_d_unsorted_index(d_unsorted_index);
  thrust::device_ptr<unsigned int> thrust_d_temp_uint(d_temp_uint);

  // Move all particle data to be consistent with new sorted order (Gather + Swap instead?)
  thrust::copy(thrust_d_r, thrust_d_r + numParticles, thrust_d_temp);
  thrust::gather(thrust_old_index.begin(), thrust_old_index.end(), thrust_d_temp, thrust_d_r);

  thrust::copy(thrust_d_im, thrust_d_im + numParticles, thrust_d_temp);
  thrust::gather(thrust_old_index.begin(), thrust_old_index.end(), thrust_d_temp, thrust_d_im);
  
  thrust::copy(thrust_d_v, thrust_d_v + numParticles, thrust_d_temp);
  thrust::gather(thrust_old_index.begin(), thrust_old_index.end(), thrust_d_temp, thrust_d_v);

  thrust::copy(thrust_d_f, thrust_d_f + numParticles, thrust_d_temp);
  thrust::gather(thrust_old_index.begin(), thrust_old_index.end(), thrust_d_temp, thrust_d_f);

  thrust::copy(thrust_d_w, thrust_d_w + numParticles, thrust_d_temp);
  thrust::gather(thrust_old_index.begin(), thrust_old_index.end(), thrust_d_temp, thrust_d_w);

  thrust::copy(thrust_d_misc, thrust_d_misc + numParticles, thrust_d_temp);
  thrust::gather(thrust_old_index.begin(), thrust_old_index.end(), thrust_d_temp, thrust_d_misc);
  
  thrust::copy(thrust_d_unsorted_index, thrust_d_unsorted_index + numParticles, thrust_d_temp_uint);
  thrust::gather(thrust_old_index.begin(), thrust_old_index.end(), thrust_d_temp_uint, thrust_d_unsorted_index);

}



//////////////////////////////////////////////////
// Work on device data
//////////////////////////////////////////////////

void ParticleData::IsotropicScalePositions( float Rscal ){
  dim3 numBlocks( (numParticles+32-1)/32 );
  IsotropicScalePositionsKernel<<<numBlocks, 32>>>( d_r, numParticles, Rscal );
}


void ParticleData::AnisotropicScalePositions( float Rscal, unsigned dir ){
  dim3 numBlocks( (numParticles+32-1)/32 );
  AnisotropicScalePositionsKernel<<<numBlocks, 32>>>( d_r, numParticles, Rscal, dir );
}

void ParticleData::AffinelyShearPositions(float shear_strain) {
  dim3 numBlocks( (numParticles+32-1)/32 );
  AffinelyShearPositionsKernel<<<numBlocks, 32>>>(d_r, numParticles, shear_strain);
}

void ParticleData::ScaleVelocities(float factor) {
  dim3 numBlocks( (numParticles+32-1)/32 );
  ScaleVelocitiesKernel<<<numBlocks, 32>>>( d_v, numParticles, factor );
}						  


void ParticleData::ApplyLeesEdwardsWrapToImages(float wrap) 
{
  dim3 numBlocks( (numParticles+32-1)/32 );
  ApplyLeesEdwardsWrapToImagesKernel<<<numBlocks, 32>>>( d_im,
							 numParticles,
							 wrap );
}

// Kernels

__global__ void IsotropicScalePositionsKernel( float4 *r, unsigned numParticles,
					       float Rscal ){
  unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
  
  if ( i < numParticles ) {
    float4 my_r = r[i];
    my_r.x *= Rscal;
    my_r.y *= Rscal;
    my_r.z *= Rscal;
    r[i] = my_r;
  }
}


__global__ void AnisotropicScalePositionsKernel( float4 *r, unsigned numParticles, 
						 float Rscal, unsigned dir ){
  unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
  
  if ( i < numParticles ) {
    float4 my_r = r[i];
    if ( dir == 0 )
      my_r.x *= Rscal;
    else if ( dir == 1 )
      my_r.y *= Rscal;
    else if ( dir == 2 )
      my_r.z *= Rscal;
    r[i] = my_r;
  }
}

__global__ void AffinelyShearPositionsKernel(float4 *r, unsigned numParticles, float shear_strain) {
  unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
  if ( i < numParticles ) {
    float4 my_r = r[i];
    my_r.x += shear_strain*my_r.y;  
    r[i] = my_r;
  }
}

__global__ void ScaleVelocitiesKernel( float4 *v, unsigned numParticles, float factor ){
  unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
  
  if ( i < numParticles ) {
    float4 my_v = v[i];
    my_v.x *= factor;
    my_v.y *= factor;
    my_v.z *= factor;
    v[i] = my_v;
  }
}



__global__ void ApplyLeesEdwardsWrapToImagesKernel( float4 *image, unsigned numParticles, float wrap )
{
  unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
  
  if ( i < numParticles ) {
    float4 my_image = image[i];
    my_image.x += wrap * my_image.y;

    image[i] = my_image;
  }
}
