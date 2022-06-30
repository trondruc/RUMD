
/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/rumd_base.h"
#include "rumd/rumd_utils.h"
#include "rumd/PairPotential.h"
#include "rumd/NeighborList.h"
#include "rumd/rumd_algorithms.h"

#include <fstream>
#include <typeinfo>
#include <sstream>
#include <iostream>

#include <thrust/device_vector.h>




// Forward declaration.


template<int STR, int CUTOFF, bool initialize, class P, class S>
__global__ void Calcf_NBL_tp( P *Pot, unsigned int num_part, unsigned int nvp, __restrict__ float4* r, __restrict__ float4* f, float4* w, float4* sts, S* simBox, float* simBoxPointer, float *params, unsigned int num_types, unsigned *num_nbrs, unsigned *nbl);


template<int STR, int CUTOFF, bool initialize, class P, class S>
__global__ void Calcf_NBL_tp_equal_one( P *Pot, unsigned int num_part, unsigned int nvp, __restrict__ float4* r, __restrict__ float4* f, float4* w, float4* sts, S* simBox, float* simBoxPointer, float *params, unsigned int num_types, unsigned *num_nbrs, unsigned *nbl);


//////////////////////////////////////////////////
// Abstract PairPotential implementations.
//////////////////////////////////////////////////

PairPotential::PairPotential( CutoffMethod cutoff_method ) :
  allocated_num_types(0),
  neighborList(),
  testRSB(0),
  testLESB(0),
  cutoffMethod(cutoff_method),
  params_map(),
  allocated_size_pe(0),
  d_params(0),
  d_f_pe(0),
  d_w_pe(0),
  h_f_pe(0),
  shared_size(0),
  assume_Newton3(true) { 
  
  if(cutoffMethod < NS || cutoffMethod > SF) throw RUMD_Error("PairPotential","PairPotential","Invalid cutoffMethod");

  

  SetID_String("base_pair_pot");
}

PairPotential::~PairPotential() { 
  if(allocated_num_types > 0)
    cudaFree(d_params);

  if(allocated_size_pe != 0) {
    cudaFreeHost(h_f_pe);
    cudaFree(d_f_pe);
    cudaFree(d_w_pe);
  }

}


void PairPotential::SetAllParams(const std::map<std::pair<unsigned, unsigned>, std::vector<float>  > &other_params_map) {
  params_map = other_params_map;

}

void PairPotential::CopyParamsToGPU( unsigned i, unsigned j ){
  
  std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
  unsigned usedParams = params_map[ pair_i_j ].size();
  params_map[ pair_i_j ].resize(NumParam);

  if( (usedParams + 2) > NumParam )
    throw( RUMD_Error(GetID_String(), "CopyParamsToGPU", "RUMD requires more allocated parameters to run. Increase NumParam in rumd_base.h") );
  
  float4 my_f = { 0, 0, 0, 0 };
  
  std::vector<float> these_params = params_map[ pair_i_j ];
  float s = 0.;
  float Rcut = these_params [0];
  if(Rcut > 0.0)
    s = ComputeInteraction_CPU( Rcut*Rcut, &(these_params[0]), &my_f);

  params_map[ pair_i_j ][NumParam-1] = my_f.w;
  params_map[ pair_i_j ][NumParam-2] = s * Rcut;


  // Copy only the relevant pair-interaction parameters to Device
  if(particleData) {
    if(verbose)
      std::cout << "Copying pair-interaction parameters to GPU (label=" << GetID_String() << ")" << std::endl;
    cudaMemcpy( &(d_params[i*particleData->GetNumberOfTypes()*NumParam + j*NumParam]), &(params_map[pair_i_j][0]), NumParam*sizeof(float), cudaMemcpyHostToDevice );
  }


  if(i != j && assume_Newton3) {
    if(verbose)
      std::cout << "Applying Newton's third law to set interaction " << j <<"," << i << " (label=" << GetID_String() << ")" << std::endl;

    std::pair<unsigned, unsigned> pair_j_i = std::make_pair(j, i);
    params_map[ pair_j_i ] = std::vector<float> (params_map[ pair_i_j ]);

    // Copy only the relevant pair interaction parameters
    if(particleData)
      cudaMemcpy( &(d_params[j*particleData->GetNumberOfTypes()*NumParam + i*NumParam]), &(params_map[pair_j_i][0]), NumParam*sizeof(float), cudaMemcpyHostToDevice );
    
  }

  neighborList.SetCutoffs(this);
} 



void PairPotential::CopyAllParamsToGPU(unsigned num_types) {
  if(num_types != allocated_num_types) {
    if(d_params)
      cudaFree(d_params);

    size_t param_array_size = num_types*num_types*NumParam*sizeof(float);
    if( cudaMalloc( (void**) &d_params, param_array_size) == cudaErrorMemoryAllocation )
      throw( RUMD_Error("PairPotential", __func__, "Malloc failed on d_params") );
    cudaMemset( d_params, 0, param_array_size);
    allocated_num_types = num_types;
  }

  float *param0 = 0;
  cudaMallocHost( (void**) & param0, NumParam*sizeof(float));
  memset(param0, 0, NumParam*sizeof(float));
  for (unsigned i = 0; i < num_types; i++)
    for (unsigned j = 0; j < num_types; j++) {
      std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);      
      std::map<std::pair<unsigned, unsigned>, std::vector<float> >::iterator it = params_map.find(pair_i_j);
      unsigned index = i*num_types*NumParam + j*NumParam;
      if(it != params_map.end()) {
	cudaMemcpy( &(d_params[index]), &(it->second[0]), NumParam*sizeof(float), cudaMemcpyHostToDevice );
      }
      else {
        cudaMemcpy( &(d_params[index]), &(param0[0]), NumParam*sizeof(float), cudaMemcpyHostToDevice );
      }
    }
  cudaFreeHost(param0);
}



void PairPotential::UpdateAfterSorting( unsigned* old_index, unsigned* new_index ){ 
  neighborList.UpdateAfterSorting(old_index, new_index);
};



// Writes the actual potential and radial force implementation in PairPotential.h. 

void PairPotential::WritePotentials(SimulationBox* simBox){
  testRSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  if(!testRSB)
    throw RUMD_Error("PairPotential","WritePotentials","Works only with RectangularSimulationBox for now");

  float4 my_r = {0.1f, 0.0f, 0.0f, 0.0f};
  float4    r = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 my_f = {0.0f, 0.0f, 0.0f, 0.0f}; 
  float4 my_w = {0.0f, 0.0f, 0.0f, 0.0f}; 
  float s = 0; float dist; 
  
  std::ofstream pFile( (std::string("potentials_") + GetID_String() + ".dat").c_str() );
  float halfBoxLength = powf( simBox->GetVolume(), 1.f/3.f ) / 2.0;
  
  // could also specify as argument
  unsigned num_types = 1;
  if(particleData)
    num_types = particleData->GetNumberOfTypes();

  pFile.precision(8);
  while ( my_r.x < halfBoxLength ){
    pFile << my_r.x << " ";
    
    for(unsigned int MyType=0; MyType < num_types; MyType++){
      for(unsigned int Type=0; Type < num_types; Type++){
	s = 0; my_f.w = 0.f; 
	
	std::vector<float> param = params_map.at(std::make_pair(MyType, Type));
	float4 distance = testRSB->calculateDistance( my_r, r, testRSB->GetHostPointer() );
	
	if( distance.w <= (param[0]*param[0]) && distance.w >= 0.000001f ){
	  s = ComputeInteraction_CPU( distance.w, &(param[0]), &my_f );
	    
	  // Which cut-off method?
	  switch( cutoffMethod ){
	  case NS:
	    break;
	  case SP:
	    my_f.w -= param[NumParam-1]; 
	    break;
	  case SF:
	    dist = sqrtf(distance.w); 
	    s -= param[NumParam-2] / dist;  
	    my_f.w += param[NumParam-2] * ( dist - param[0] ) - param[NumParam-1]; 
	    my_w.w -= param[NumParam-2] * dist;
	    break;
	  default:
	    break;
	  }
	}
	pFile << my_f.w << " " << my_r.x * s << " ";
      }
    }
    pFile << std::endl;
    // Update only the x component.
    my_r.x += 0.001;
  } 
  pFile.close();
}


void PairPotential::AllocatePE_Array(unsigned int nvp) {
  if(allocated_size_pe != 0) {
    cudaFreeHost(h_f_pe);
    cudaFree(d_f_pe);
    cudaFree(d_w_pe);
  }
  
  if( cudaMallocHost( (void**) &h_f_pe, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("PairPotential","AllocatePE_Array","Malloc failed on h_f_pe") );
  if( cudaMalloc( (void**) &d_f_pe, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("PairPotential","AllocatePE_Array","Malloc failed on d_f_pe") );

  if( cudaMalloc( (void**) &d_w_pe, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("PairPotential","AllocatePE_Array","Malloc failed on d_w_pe") );

  memset( h_f_pe, 0, nvp * sizeof(float4) );
  cudaMemset( d_f_pe, 0, nvp * sizeof(float4) );
  cudaMemset( d_w_pe, 0, nvp * sizeof(float4) );
  
  allocated_size_pe = nvp;
}

double PairPotential::GetPotentialEnergy() {
  // launch kernel with d_f_pe instead of df as the force array; 
  // includes allocation and update of NB list
  CalcF_Local();
  
  double pe_thisPotential = 0.;

  // copy to host
  cudaMemcpy( h_f_pe, d_f_pe, allocated_size_pe * sizeof(float4), cudaMemcpyDeviceToHost );

  // sum over particles
  for(unsigned int i=0; i < particleData->GetNumberOfParticles(); i++)
    pe_thisPotential += h_f_pe[i].w;
  
  return pe_thisPotential;
}



/////////////////////////////////////////////////////////
// Set functions
/////////////////////////////////////////////////////////

void  PairPotential::Initialize() {
  unsigned num_types = particleData->GetNumberOfTypes();
  size_t param_array_size = num_types*num_types*NumParam*sizeof(float);
  shared_size = param_array_size;
  if(kp.shared_size > shared_size) // number of force-contributions per block times sizeof(float4)
    shared_size = kp.shared_size; 

  CopyAllParamsToGPU(num_types);

  neighborList.Initialize(sample);
  neighborList.SetCutoffs(this);

  testRSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  testLESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);
  if (!testRSB && !testLESB) throw RUMD_Error("PairPotential", __func__, "simBox must be either RectangularSimulationBox or LeesEdwardsSimulationBox");
}


void PairPotential::SetExclusionBond(uint1 *h_btlist, uint2* h_blist, unsigned max_num_bonds, unsigned etype ){
  neighborList.SetExclusionBond(h_btlist, h_blist, max_num_bonds, etype);
}

void PairPotential::SetExclusionType( unsigned type0, unsigned type1 ){
  neighborList.SetExclusionType( type0, type1 );
}

void PairPotential::SetExclusionMolecule(int1 *h_mlist, unsigned molindex, unsigned max_num_uau, unsigned num_mol){
  neighborList.SetExclusionMolecule(h_mlist, molindex, max_num_uau, num_mol);
}


void PairPotential::SetExclusionAngle(uint4 *h_alist,  unsigned num_angles){
  neighborList.SetExclusionAngle(h_alist,  num_angles);
}

void PairPotential::SetExclusionDihedral(uint4 *h_dlist,  unsigned num_dihedrals){
  neighborList.SetExclusionDihedral(h_dlist, num_dihedrals);
}

/////////////////////////////////////////////////////////
// Specialized PairPotentials implementations.
// The specialization is needed due to CUDA limitations.
// The file PairPotentialFunctionBodies.inc is generated
// automatically by the python script Generate_PP_FunctionBodies.py
// It should not be edited by hand!!!
/////////////////////////////////////////////////////////

#include "PairPotentialFunctionBodies.inc"

/////////////////////////////////////////////////////////
// Kernel implementation of pair force evaluation.
/////////////////////////////////////////////////////////



template<int STR, int CUTOFF, bool initialize, class P, class S>
__global__ void Calcf_NBL_tp_equal_one( P *Pot, unsigned int num_part, unsigned int nvp, __restrict__ float4* r, __restrict__ float4* f, float4* w, float4* sts, S* simBox, float* simBoxPointer, float *params, unsigned int num_types, unsigned *num_nbrs, unsigned *nbl) {
  extern __shared__ float4 s_r[];
  float* s_params = (float*) &s_r[0]; // NOTE overlaps s_r. Should check size etc 

  float4 my_f = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 my_w = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 my_sts = {0.0f, 0.0f, 0.0f, 0.0f};

  float simBoxPtr_local[simulationBoxSize];
  simBox->loadMemory( simBoxPtr_local, simBoxPointer );

 
  int my_num_nbrs = num_nbrs[MyGP];
  int nb;
  int nb_prefetch = nbl[MyGP];
 

  //float4 my_r = LOAD(r[MyGP]);
  float4 my_r = (r[MyGP]);
  int my_type = __float_as_int(my_r.w);
  
  const unsigned int tid = MyP;
  for (unsigned int index=0; index<num_types*num_types*NumParam; index += PPerBlock) {
    unsigned int myindex = index+tid;
    if (myindex<num_types*num_types*NumParam)
      s_params[myindex] = params[myindex];
  }
  __syncthreads(); // Wait for parameters to be loaded in shared mem.

  for (unsigned int i=0; i<my_num_nbrs; i++) {    
    nb = nb_prefetch;
    //nb = nbl[nvp*(i) + MyGP];
    float4 r_i = LOAD(r[nb]);
    //if(i < my_num_nbrs-1) // have allocated an extra row, no need to check
    nb_prefetch = nbl[nvp*(i+1) + MyGP];
    fij<STR, CUTOFF>( Pot, my_r, r_i, &my_f, &my_w, &my_sts, &(s_params[(my_type * num_types + __float_as_int(r_i.w))*NumParam]), simBox, simBoxPtr_local );
  }
  
  my_f.w *= 0.5f; // Compensate double counting of potential energies
  if(initialize) {
    f[MyGP] = my_f;
    w[MyGP] = my_w;
    if(STR) sts[MyGP] = my_sts;
  }
  else {
      atomicFloatAdd(&(f[MyGP].x), my_f.x);
      atomicFloatAdd(&(f[MyGP].y), my_f.y);
      atomicFloatAdd(&(f[MyGP].z), my_f.z);
      atomicFloatAdd(&(f[MyGP].w), my_f.w);
      atomicFloatAdd(&(w[MyGP].w), my_w.w);
      if(STR) {
	atomicFloatAdd(&(w[MyGP].y), my_w.y);
	atomicFloatAdd(&(w[MyGP].z), my_w.z);
	atomicFloatAdd(&(sts[MyGP].x), my_sts.x);
	atomicFloatAdd(&(sts[MyGP].y), my_sts.y);
	atomicFloatAdd(&(sts[MyGP].z), my_sts.z);
	atomicFloatAdd(&(sts[MyGP].w), my_sts.w);
      }

      
  }

}

template<int STR, int CUTOFF, bool initialize, class P, class S>
__global__ void Calcf_NBL_tp( P *Pot, unsigned int num_part, unsigned int nvp, __restrict__ float4* r, __restrict__ float4* f, float4* w, float4* sts, S* simBox, float* simBoxPointer, float *params, unsigned int num_types, unsigned *num_nbrs, unsigned *nbl) {
  // Introducing tp (Threads per Particle) > 1. Contribution from different threads added in the end, using shared mem
  extern __shared__ float4 s_r[];
  float* s_params = (float*) &s_r[0]; // NOTE overlaps s_r.

  float4 my_f = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 my_w = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 my_sts = {0.0f, 0.0f, 0.0f, 0.0f};

  float simBoxPtr_local[simulationBoxSize];
  simBox->loadMemory( simBoxPtr_local, simBoxPointer );


  float4 my_r = LOAD(r[MyGP]);
  int my_type = __float_as_int(my_r.w);
  
  const unsigned int tid = MyP + MyT*PPerBlock;
  for (unsigned int index=0; index<num_types*num_types*NumParam; index += PPerBlock*TPerPart) {
    unsigned int myindex = index+tid;
    if (myindex<num_types*num_types*NumParam)
      s_params[myindex] = params[myindex];
  }
  __syncthreads();

  if(nbl) { 
    int my_num_nbrs = num_nbrs[MyGP];
    int nb;
    int nb_prefetch = nbl[nvp*MyT + MyGP];
    
    for (int i=MyT; i<my_num_nbrs; i+=TPerPart) {    
      nb = nb_prefetch;
      if(i+TPerPart < my_num_nbrs)
	nb_prefetch = nbl[nvp*(i+TPerPart) + MyGP];  // Last read not used, could decrease loop by one (TPerPart), and do last after loop
      // CAREFUL about num_nbrs<TPerPart case (eg virtual particles)
      float4 r_i = LOAD(r[nb]);
      int type_i =  __float_as_int(r_i.w);
      fij<STR, CUTOFF>( Pot, my_r, r_i, &my_f, &my_w, &my_sts, &(s_params[my_type * num_types*NumParam + type_i*NumParam]), simBox, simBoxPtr_local );    
    }
  } // if(nbl)
  else {
    // not using neighbor-list; loop over all particles
    for (int i=MyT; i<num_part; i+=TPerPart) {
      if(i != MyGP) {
	float4 r_i = LOAD(r[i]);
	int type_i =  __float_as_int(r_i.w);
	fij<STR, CUTOFF>( Pot, my_r, r_i, &my_f, &my_w, &my_sts, &(s_params[my_type * num_types*NumParam + type_i*NumParam]), simBox, simBoxPtr_local );
      }
    }
  }
  
  
  __syncthreads(); // Done with s_params (overlapping s_r)
  s_r[MyP+MyT*PPerBlock] = my_f;
  __syncthreads();
    
  if( MyT == 0 ) {
    for( int i=1; i < TPerPart; i++ ) my_f += s_r[MyP + i*PPerBlock];
    my_f.w *= 0.5f;
    if(initialize)
      f[MyGP] = my_f;
    else {
      atomicFloatAdd(&(f[MyGP].x), my_f.x);
      atomicFloatAdd(&(f[MyGP].y), my_f.y);
      atomicFloatAdd(&(f[MyGP].z), my_f.z);
      atomicFloatAdd(&(f[MyGP].w), my_f.w);
    }
  }
    
  __syncthreads();  
  // Sum the atomic/molecular virial for all local threads.
  s_r[MyP+MyT*PPerBlock] = my_w;
  __syncthreads();  
  

  if( MyT == 0 ){
    for ( int i=1; i < TPerPart; i++ ){
      int j = MyP + __umul24(i,PPerBlock); 
      my_w.w += s_r[j].w;
      
      if(STR){
	my_w.y += s_r[j].y;
	my_w.z += s_r[j].z;
      }
    }
    if(initialize)
      w[MyGP] = my_w;
    else {
      atomicFloatAdd(&(w[MyGP].w), my_w.w);
      if(STR){
	atomicFloatAdd(&(w[MyGP].y), my_w.y);
	atomicFloatAdd(&(w[MyGP].z), my_w.z);
      }
    }
  }
  
  if(STR){
    __syncthreads();  
    
    // Sum the shear stress for all local threads.
    s_r[MyP+MyT*PPerBlock] = my_sts;
    
    __syncthreads();  
    
    if( MyT == 0 ){
      for ( int i=1; i < TPerPart; i++ ){
	int j = MyP + __umul24(i,PPerBlock); 
	my_sts += s_r[j];
      }
      if(initialize)
	sts[MyGP] = my_sts;
      else {
	atomicFloatAdd(&(sts[MyGP].x), my_sts.x);
	atomicFloatAdd(&(sts[MyGP].y), my_sts.y);
	atomicFloatAdd(&(sts[MyGP].z), my_sts.z);
	atomicFloatAdd(&(sts[MyGP].w), my_sts.w);
      }
    } // if(MyT == 0)
  } // END if(STR)

}
