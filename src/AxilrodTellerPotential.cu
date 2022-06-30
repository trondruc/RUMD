
/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/rumd_base.h"
#include "rumd/rumd_utils.h"
#include "rumd/AxilrodTellerPotential.h"
#include "rumd/NeighborList.h"
#include "rumd/rumd_algorithms.h"

template<class S>
__global__ void calcf_AxTe(unsigned int num_part, unsigned int nvp, __restrict__ float4* r, __restrict__ float4* f, float4* w, S* simBox, float* simBoxPointer, float v_AT, unsigned *num_nbrs, unsigned *nbl, float rc2);


AxilrodTellerPotential::AxilrodTellerPotential(float v_AT, float Rcut) :
  neighborList(),
  allocated_size_pe(0),
  testLESB(0),
  testRSB(0) {
  this->v_AT = v_AT;
  this->Rcut = Rcut;
  SetID_String("potAxTe");
}

AxilrodTellerPotential::~AxilrodTellerPotential() {
  if(allocated_size_pe != 0) {
    cudaFreeHost(h_f_w_loc);
    cudaFree(d_f_loc);
    cudaFree(d_w_loc);
  }
}


void AxilrodTellerPotential::AllocatePE_Array(unsigned int nvp) {
  if(allocated_size_pe != 0) {
    cudaFreeHost(h_f_w_loc);
    cudaFree(d_f_loc);
    cudaFree(d_w_loc);
  }
  
  if( cudaMallocHost( (void**) &h_f_w_loc, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("AxilrodTellerPotential", __func__, "Malloc failed on h_f_w_loc") );
  if( cudaMalloc( (void**) &d_f_loc, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("AxilrodTellerPotential", __func__, "Malloc failed on d_f_loc") );
  if( cudaMalloc( (void**) &d_w_loc, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("AxilrodTellerPotential", __func__, "Malloc failed on d_w_loc") );

  memset( h_f_w_loc, 0, nvp * sizeof(float4) );
  cudaMemset( d_f_loc, 0, nvp * sizeof(float4) );
  cudaMemset( d_w_loc, 0, nvp * sizeof(float4) );
  
  allocated_size_pe = nvp;
}


void AxilrodTellerPotential::Initialize() {

 
  neighborList.Initialize(sample);
  
  neighborList.SetEqualCutoffs(particleData->GetNumberOfTypes(), Rcut);

  testRSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  testLESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);
  if (!testRSB && !testLESB) throw RUMD_Error("AxilrodTellerPotential", __func__, "simBox must be either RectangularSimulationBox or LeesEdwardsSimulationBox");

  
}


void AxilrodTellerPotential::CalcF(bool initialize, bool calc_stresses) {
  if(calc_stresses) throw RUMD_Error("AxilrodTellerPotential",__func__,"Calculation of stresses not implemented");

  if(initialize)
    particleData->SetForcesToZero();
  dim3 my_tp = kp.threads;
  //my_tp.y = 1;
  float rc2 = Rcut * Rcut;
  neighborList.UpdateNBlist();


  if(testLESB)
    calcf_AxTe<<<kp.grid, my_tp>>>( particleData->GetNumberOfParticles(), 
				    particleData->GetNumberOfVirtualParticles(),
				    particleData->d_r, particleData->d_f, 
				    particleData->d_w,
				    testLESB, testLESB->GetDevicePointer(),
				    v_AT,
				    neighborList.GetNumNbrsPtr(),
				    neighborList.GetNbListPtr(),
				    rc2
				    );
  else
    calcf_AxTe<<<kp.grid, my_tp>>>( particleData->GetNumberOfParticles(), 
				    particleData->GetNumberOfVirtualParticles(),
				    particleData->d_r, particleData->d_f, 
				    particleData->d_w,
				    testRSB, testRSB->GetDevicePointer(),
				    v_AT,
				    neighborList.GetNumNbrsPtr(),
				    neighborList.GetNbListPtr(),
				    rc2);

}



double AxilrodTellerPotential::GetPotentialEnergy(){

  neighborList.UpdateNBlist();
  unsigned nvp = particleData->GetNumberOfVirtualParticles();
  if(allocated_size_pe != nvp)
    AllocatePE_Array(nvp);

  // set local force, virial arrays to zero
  cudaMemset( d_f_loc, 0, nvp * sizeof(float4) );
  cudaMemset( d_w_loc, 0, nvp * sizeof(float4) ); // currently not used
  dim3 my_tp = kp.threads;
  //my_tp.y = 1;
  float rc2 = Rcut * Rcut;

  if(testLESB)
    calcf_AxTe<<<kp.grid, my_tp>>>( particleData->GetNumberOfParticles(), 
				    nvp,
				    particleData->d_r, d_f_loc, 
				    d_w_loc,
				    testLESB, testLESB->GetDevicePointer(),
				    v_AT,
				    neighborList.GetNumNbrsPtr(),
				    neighborList.GetNbListPtr(),
				    rc2);
  else
    calcf_AxTe<<<kp.grid, my_tp>>>( particleData->GetNumberOfParticles(), 
				    particleData->GetNumberOfVirtualParticles(),
				    particleData->d_r, d_f_loc, 
				    d_w_loc,
				    testRSB, testRSB->GetDevicePointer(),
				    v_AT,
				    neighborList.GetNumNbrsPtr(),
				    neighborList.GetNbListPtr(),
				    rc2);
  


  
  double pe_thisPotential = 0.;

  // copy to host
  cudaMemcpy( h_f_w_loc, d_f_loc, allocated_size_pe * sizeof(float4), cudaMemcpyDeviceToHost );

  // sum over particles
  for(unsigned int i=0; i < particleData->GetNumberOfParticles(); i++)
    pe_thisPotential += h_f_w_loc[i].w;
    
  return pe_thisPotential;
}



template<class S>
__global__ void calcf_AxTe(unsigned int num_part, unsigned int nvp, __restrict__ float4* r, __restrict__ float4* f, float4* w, S* simBox, float* simBoxPointer, float v_AT, unsigned *num_nbrs, unsigned *nbl, float rc2) {
extern __shared__ float4 s_r[];

  float4 my_f = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 my_w = {0.0f, 0.0f, 0.0f, 0.0f};

  float simBoxPtr_local[simulationBoxSize];
  simBox->loadMemory( simBoxPtr_local, simBoxPointer );

 
  int my_num_nbrs = num_nbrs[MyGP];
  int nbj;
  int nbk;
  //int nbj_prefetch = nbl[MyGP];
  
  
  float4 r_i = (r[MyGP]);

  
  for (int j=MyT; j<my_num_nbrs; j+=TPerPart) { 

    //for (unsigned int j=0; j<my_num_nbrs; j++) {
    nbj = nbl[nvp*j + MyGP];
    float4 r_j = LOAD(r[nbj]);
    
    float4 disp_ij = simBox->calculateDistance(r_i, r_j, simBoxPtr_local);
    if(disp_ij.w < rc2)
      for (unsigned int k=j+1; k<my_num_nbrs; k++) {
	nbk = nbl[nvp*k + MyGP];
	float4 r_k = LOAD(r[nbk]);
	float4 disp_ik = simBox->calculateDistance(r_i, r_k, simBoxPtr_local);
	float4 disp_jk = simBox->calculateDistance(r_j, r_k, simBoxPtr_local);

	if(disp_ik.w < rc2 && disp_jk.w < rc2) {
	  // float4 is a unit containing 4 floats
	  // disp_ik.x = r_i.x - r_k.x (with PBC taken into account)
	  // disp_ik.y = r_i.y - r_k.y (with PBC taken into account)
	  // disp_ik.z = r_i.z - r_k.z (with PBC taken into account)
	  // The fourth component, disp_ik.w is the sum of the squares of the others, ie the squared magnitude of the displacement vector (with PBC taken into account)
	  // the following  calculates the force on particle i  here
	  
	  float dot_ik_jk = disp_ik.x*disp_jk.x + disp_ik.y*disp_jk.y + disp_ik.z*disp_jk.z;	
	  float dot_ij_jk = disp_ij.x*disp_jk.x + disp_ij.y*disp_jk.y + disp_ij.z*disp_jk.z;
	  float dot_ij_ik = disp_ij.x*disp_ik.x + disp_ij.y*disp_ik.y + disp_ij.z*disp_ik.z;	
	  
	  float A = disp_ij.w * disp_ik.w * disp_jk.w;
	  float B = 3.f * dot_ik_jk * dot_ij_jk * dot_ij_ik; 
	  float C_inv = powf(A, -2.5f);
	  
	  float4 gradA;
	  gradA.x = 2.f*(disp_ij.x * disp_ik.w * disp_jk.w + disp_ik.x * disp_ij.w * disp_jk.w);
	  gradA.y = 2.f*(disp_ij.y * disp_ik.w * disp_jk.w + disp_ik.y * disp_ij.w * disp_jk.w);
	  gradA.z = 2.f*(disp_ij.z * disp_ik.w * disp_jk.w + disp_ik.z * disp_ij.w * disp_jk.w);
	  
	  float4 gradB;
	  gradB.x = 3.*((disp_ij.x + disp_ik.x) * dot_ij_jk * dot_ik_jk +
			disp_jk.x * dot_ij_ik * (dot_ik_jk + dot_ij_jk) );
	  gradB.y = 3.*((disp_ij.y + disp_ik.y) * dot_ij_jk * dot_ik_jk +
			disp_jk.y * dot_ij_ik * (dot_ik_jk + dot_ij_jk) );
	  gradB.z = 3.*((disp_ij.z + disp_ik.z) * dot_ij_jk * dot_ik_jk +
			disp_jk.z * dot_ij_ik * (dot_ik_jk + dot_ij_jk) );
	  
	  
	  float4 gradC_overC;
	  gradC_overC.x = 5.f * (disp_ij.x / disp_ij.w + disp_ik.x / disp_ik.w);
	  gradC_overC.y = 5.f * (disp_ij.y / disp_ij.w + disp_ik.y / disp_ik.w);
	  gradC_overC.z = 5.f * (disp_ij.z / disp_ij.w + disp_ik.z / disp_ik.w);
	  my_f.x += -v_AT * C_inv * ((gradA.x - gradB.x) -(A-B) * gradC_overC.x);
	  my_f.y += -v_AT * C_inv * ((gradA.y - gradB.y) -(A-B) * gradC_overC.y);
	  my_f.z += -v_AT * C_inv * ((gradA.z - gradB.z) -(A-B) * gradC_overC.z);
	  
	  
	  // contribution to potential energy
	  float pe_over3 = v_AT * (A-B) * C_inv/3.;
	  my_f.w += pe_over3;
	  // contribution to virial : effective exponent is nine, divided by 3 to avoid triple counting. Should also divide by 3 to convert to virial, but that is done later, as is a division by two to avoid double-counting which we don't need, so for now want (9/3)*2 * pe = 18 * pe/3
	  my_w.w += 18.f * pe_over3; // 
	} // if disp ik/jk < rc2 ....
      } // k loop
    
  } // j loop
  
  // add force on particle i to global force array
  atomicFloatAdd(&(f[MyGP].x), my_f.x);
  atomicFloatAdd(&(f[MyGP].y), my_f.y);
  atomicFloatAdd(&(f[MyGP].z), my_f.z);
  atomicFloatAdd(&(f[MyGP].w), my_f.w);
  atomicFloatAdd(&(w[MyGP].w), my_w.w);
  
}
