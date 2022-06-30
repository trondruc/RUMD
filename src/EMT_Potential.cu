/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/



#define NumParamEMT 11

#include "rumd/EMT_Potential.h"
#include "rumd/rumd_algorithms.h"

#include <iostream>

template<class S>
__global__ void emt_calc_sigmas_energies( unsigned int num_part, unsigned int nvp, __restrict__ float4* r, float2* dEds_E, S* simBox, float* simBoxPointer, float *params, unsigned int num_types, unsigned int *num_nbrs, unsigned int *nbl);


template<int STR, int INIT, class S>
__global__ void emt_calc_forces_after_energies( unsigned int num_part, unsigned int nvp, __restrict__ float4* r, __restrict__ float4* f, float4* w, float4* sts,float2* dEds_E, S* simBox, float* simBoxPointer, float *params, unsigned int num_types, unsigned int *num_nbrs, unsigned int *nbl);


#define BETA 1.8093997905635548
const float EMT_Potential::Beta = BETA; // pow(16.*pi/3,1./3)/sqrt(2);

const int EMT_Potential::shell0 = 3;
const int EMT_Potential::shell1 = 4;

EMT_Potential::EMT_Potential() : cutoff(0),
				 nb_cut(0),
				 cutslope(0),
				 allocated_num_types(0),
				 allocated_energy(0),
				 d_f_pe(0),
				 h_f_pe(0),
				 d_dEds_E(0),
				 neighborList(),
				 testRSB(0),
				 testLESB(0),
				 d_params_emt(0),
				 shared_size1(0),
				 shared_size2(0),
				 emt_params_map() {
  SetID_String("potEMT");
}


EMT_Potential::~EMT_Potential() {

  if(allocated_num_types > 0)
    cudaFree(d_params_emt);
  
  if(allocated_energy >0) {
    cudaFreeHost(h_f_pe);
    cudaFree(d_f_pe);
    cudaFree(d_dEds_E);
  }
  
}


Potential* EMT_Potential::Copy() {
  EMT_Potential* emt_copy = new EMT_Potential();

  std::map<unsigned, std::vector<float> >::iterator iter;
  for(iter = emt_params_map.begin(); iter != emt_params_map.end(); iter++) {
    unsigned index = iter->first;
    std::vector<float> params = iter->second;
    emt_copy->SetParams(index, params[0], params[1], params[2],
			params[3], params[4], params[5], params[6]);
  }

  
  return emt_copy;
}



std::vector<float> EMT_Potential::GetPotentialParameters(unsigned num_types) const {
  std::vector<float> params_vec;
  for(unsigned i = 0; i < num_types; i++) {
    std::map<unsigned, std::vector<float> >::const_iterator params_i = emt_params_map.find(i);    
    if(params_i != emt_params_map.end()) {
      params_vec.push_back(i);
      std::vector<float>::const_iterator it;
      for(it = params_i->second.begin(); it != params_i->second.end(); it++)
	params_vec.push_back(*it);
    }
  }
  return params_vec;
}

void EMT_Potential::AllocateEnergyArrays(unsigned int nvp) {
  if(allocated_energy != 0) {
    cudaFreeHost(h_f_pe);
    cudaFree(d_f_pe);
    cudaFree(d_dEds_E);
  }
  
  if( cudaMallocHost( (void**) &h_f_pe, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("EMT_Potential", __func__, "Malloc failed on h_f_pe") );
  if( cudaMalloc( (void**) &d_f_pe, nvp * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("EMT_Potential", __func__, "Malloc failed on d_f_pe") );
  if( cudaMalloc( (void**) &d_dEds_E, nvp * sizeof(float2) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("EMT_Potential", __func__, "Malloc failed on d_dEds_E") );

  memset( h_f_pe, 0, nvp * sizeof(float4) );
  cudaMemset( d_f_pe, 0, nvp * sizeof(float4) );
  cudaMemset( d_dEds_E, 0, nvp * sizeof(float2) );
  
  allocated_energy = nvp;
}


void EMT_Potential::SetParams(unsigned type_idx, float E0, float s0, float V0, float eta2, float kappa, float lambda, float n0) {

  emt_params_map[ type_idx ] = std::vector<float>();
  emt_params_map[ type_idx ].push_back(E0);
  emt_params_map[ type_idx ].push_back(s0);
  emt_params_map[ type_idx ].push_back(V0);
  emt_params_map[ type_idx ].push_back(eta2);
  emt_params_map[ type_idx ].push_back(kappa);
  emt_params_map[ type_idx ].push_back(lambda);
  emt_params_map[ type_idx ].push_back(n0);


  // calculate the cutoff, gammas for all elements each time
  // any parameters are changed
  
  CalculateCutoff();

  std::map<unsigned, std::vector<float> >::iterator iter;
  for(iter = emt_params_map.begin(); iter != emt_params_map.end(); iter++)
    CalculateGammas(iter->first);

  if(particleData)
    CopyParamsToGPU(); 
  
}


void EMT_Potential::CopyParamsToGPU() {
  unsigned num_types = particleData->GetNumberOfTypes();

  if(num_types != allocated_num_types) {
    if(d_params_emt)
      cudaFree(d_params_emt);

    size_t param_array_size = num_types*NumParamEMT*sizeof(float);
    if( cudaMalloc( (void**) &d_params_emt, param_array_size) == cudaErrorMemoryAllocation )
      throw( RUMD_Error("EMT_Potential", __func__, "Malloc failed on d_params_emt") );
    cudaMemset( d_params_emt, 0, param_array_size);
    allocated_num_types = num_types;
  }

  float *param0 = 0;
  cudaMallocHost( (void**) & param0, NumParamEMT*sizeof(float));
  memset(param0, 0, NumParamEMT*sizeof(float));
  for (unsigned i = 0; i < num_types; i++) {
      std::map<unsigned, std::vector<float> >::iterator it = emt_params_map.find(i);
      unsigned index = i*NumParamEMT;
      if(it != emt_params_map.end()) {
	cudaMemcpy( &(d_params_emt[index]), &(it->second[0]), NumParamEMT*sizeof(float), cudaMemcpyHostToDevice );
      }
      else {
        cudaMemcpy( &(d_params_emt[index]), &(param0[0]), NumParamEMT*sizeof(float), cudaMemcpyHostToDevice );
      }
    }
  cudaFreeHost(param0);

}



void EMT_Potential::CalculateCutoff() {
  float max_s0 = 0.f;
  std::map<unsigned, std::vector<float> >::iterator iter;
  for(iter = emt_params_map.begin(); iter != emt_params_map.end(); iter++)
    if(iter->second[1] > max_s0)
      max_s0 = iter->second[1];

  // * The cutoff is halfway between the third and fourth shell
  // * Beta converts neutral sphere radius to nearest neighbor distance
  cutoff = 0.5 * max_s0 * Beta * (sqrt((double) shell0) +
                                  sqrt((double) shell0 + 1));

  // following borrowed from Asap code
  // nb_cut is the neighbor-list cutoff
  nb_cut = cutoff * 2.0 * sqrt((double) shell0 + 1) /
    (sqrt((double) shell0) + sqrt((double) shell0 + 1) );
  
  // the following makes the value of the cutoff function
  // 10^-5 at the distance nb_cut
  cutslope = log(9999.0) / (nb_cut - cutoff);

  if(particleData)
    neighborList.SetEqualCutoffs(particleData->GetNumberOfTypes(), nb_cut);  
}



void  EMT_Potential::Initialize() {
  unsigned num_types = particleData->GetNumberOfTypes();
  size_t param_array_size = num_types*NumParamEMT*sizeof(float);
  shared_size1 = param_array_size;
  shared_size2 = param_array_size;
  
  //if(kp.shared_size > shared_size) // number of force-contributions per block times sizeof(float4)
  // for calc_sigmas, unlike with PairPotential, we can't overlap the parameters and the summing since the parameters are needed before and after summing.
  shared_size1 += kp.shared_size/2; // because here it's float2 not float4 [Better way to do this???] 
  shared_size2 += kp.shared_size; // though here maybe we can overlap CHECK

  
  CopyParamsToGPU();

  neighborList.Initialize(sample);
  neighborList.SetEqualCutoffs(particleData->GetNumberOfTypes(), nb_cut);  

  if(allocated_energy != particleData->GetNumberOfVirtualParticles())
    AllocateEnergyArrays(particleData->GetNumberOfVirtualParticles());
    
  testRSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  testLESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);
  if (!testRSB && !testLESB) throw RUMD_Error("PairPotential", __func__, "simBox must be either RectangularSimulationBox or LeesEdwardsSimulationBox");
  
}

void EMT_Potential::CalculateGammas(unsigned type_idx) {

  static int shellpop[5] = {12, 6, 24, 12, 24}; // Population of the first 5 shells (fcc).
  double w, d;	
  float gamma1 = 0.f;
  float gamma2 = 0.f;

  float s0 = emt_params_map[type_idx][1];
  float eta2 = emt_params_map[type_idx][3];
  float kappa = emt_params_map[type_idx][4];
  
  for (unsigned is = 0; is < shell1 - 1 && is < 5; is++)
    {
      d = sqrt((double) (is + 1)) * Beta * s0;
      w = 1. / (1. + exp(cutslope * (d - cutoff)));
      gamma1 += w * shellpop[is] * exp(-d * eta2);
      gamma2 += w * shellpop[is] * exp(-d * kappa / Beta);
    }
  gamma1 /= shellpop[0] * exp(-Beta * s0 * eta2);
  gamma2 /= shellpop[0] * exp(-s0 * kappa);

  // in case this is not the first time calculating gammas for this element,
  // we resize back to the seven primary parameters
  emt_params_map[type_idx].resize(7); 
  emt_params_map[type_idx].push_back(cutoff);
  emt_params_map[type_idx].push_back(cutslope);
  emt_params_map[type_idx].push_back(gamma1);
  emt_params_map[type_idx].push_back(gamma2);
    
}


void EMT_Potential::CalcF(bool initialize, bool calc_stresses) {
  
  neighborList.UpdateNBlist();
  if(neighborList.GetNbListPtr() == 0)
    throw RUMD_Error("EMT_Potential", __func__, "NB method none is not compatible with EMT");

  if(!testRSB && !testLESB)
    throw RUMD_Error("EMT_Potential", __func__, "SimulationBox not recognized");


  //if(calc_stresses)
  // throw RUMD_Error("EMT_Potential", __func__, "Stresses not available for EMT yet");
  
  // first the kernel which does the sums to calculate
  // sigmas and the energy and dE/ds
  if(testRSB)
    emt_calc_sigmas_energies<<<kp.grid, kp.threads, shared_size1>>>(particleData->GetNumberOfParticles(), particleData->GetNumberOfVirtualParticles(), particleData->d_r, d_dEds_E, testRSB, testRSB->GetDevicePointer(), d_params_emt, particleData->GetNumberOfTypes(), neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
  else
     emt_calc_sigmas_energies<<<kp.grid, kp.threads, shared_size1>>>(particleData->GetNumberOfParticles(), particleData->GetNumberOfVirtualParticles(), particleData->d_r, d_dEds_E, testLESB, testLESB->GetDevicePointer(), d_params_emt, particleData->GetNumberOfTypes(), neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());

  // then need a second loop to calculate forces
  if(testRSB) {
    if(!calc_stresses && !initialize)
      emt_calc_forces_after_energies<0, 0><<<kp.grid, kp.threads, shared_size2>>>(particleData->GetNumberOfParticles(), particleData->GetNumberOfVirtualParticles(), particleData->d_r, particleData->d_f, particleData->d_w, particleData->d_sts, d_dEds_E, testRSB, testRSB->GetDevicePointer(), d_params_emt, particleData->GetNumberOfTypes(), neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
    else if(!calc_stresses && initialize)
      emt_calc_forces_after_energies<0, 1><<<kp.grid, kp.threads, shared_size2>>>(particleData->GetNumberOfParticles(), particleData->GetNumberOfVirtualParticles(), particleData->d_r, particleData->d_f, particleData->d_w, particleData->d_sts, d_dEds_E, testRSB, testRSB->GetDevicePointer(), d_params_emt, particleData->GetNumberOfTypes(), neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
    else if(calc_stresses && !initialize)
      emt_calc_forces_after_energies<1, 0><<<kp.grid, kp.threads, shared_size2>>>(particleData->GetNumberOfParticles(), particleData->GetNumberOfVirtualParticles(), particleData->d_r, particleData->d_f, particleData->d_w, particleData->d_sts, d_dEds_E, testRSB, testRSB->GetDevicePointer(), d_params_emt, particleData->GetNumberOfTypes(), neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
    else
      emt_calc_forces_after_energies<1, 1><<<kp.grid, kp.threads, shared_size2>>>(particleData->GetNumberOfParticles(), particleData->GetNumberOfVirtualParticles(), particleData->d_r, particleData->d_f, particleData->d_w, particleData->d_sts, d_dEds_E, testRSB, testRSB->GetDevicePointer(), d_params_emt, particleData->GetNumberOfTypes(), neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
  }
  else if (testLESB) {
       if(!calc_stresses && !initialize)
	 emt_calc_forces_after_energies<0, 0><<<kp.grid, kp.threads, shared_size2>>>(particleData->GetNumberOfParticles(), particleData->GetNumberOfVirtualParticles(), particleData->d_r, particleData->d_f, particleData->d_w, particleData->d_sts, d_dEds_E, testLESB, testLESB->GetDevicePointer(), d_params_emt, particleData->GetNumberOfTypes(), neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
       else if(!calc_stresses && initialize)
	 emt_calc_forces_after_energies<0, 1><<<kp.grid, kp.threads, shared_size2>>>(particleData->GetNumberOfParticles(), particleData->GetNumberOfVirtualParticles(), particleData->d_r, particleData->d_f, particleData->d_w, particleData->d_sts, d_dEds_E, testLESB, testLESB->GetDevicePointer(), d_params_emt, particleData->GetNumberOfTypes(), neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
       else if(calc_stresses && !initialize)
	 emt_calc_forces_after_energies<1, 0><<<kp.grid, kp.threads, shared_size2>>>(particleData->GetNumberOfParticles(), particleData->GetNumberOfVirtualParticles(), particleData->d_r, particleData->d_f, particleData->d_w, particleData->d_sts, d_dEds_E, testLESB, testLESB->GetDevicePointer(), d_params_emt, particleData->GetNumberOfTypes(), neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
       else
	 emt_calc_forces_after_energies<1, 1><<<kp.grid, kp.threads, shared_size2>>>(particleData->GetNumberOfParticles(), particleData->GetNumberOfVirtualParticles(), particleData->d_r, particleData->d_f, particleData->d_w, particleData->d_sts, d_dEds_E, testLESB, testLESB->GetDevicePointer(), d_params_emt, particleData->GetNumberOfTypes(), neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
  }
 
}




template<class S>
__global__ void emt_calc_sigmas_energies( unsigned int num_part, unsigned int nvp, __restrict__ float4* r, float2* dEds_E, S* simBox, float* simBoxPointer, float *params, unsigned int num_types, unsigned *num_nbrs, unsigned *nbl) {

  //extern __shared__ float s_params[];
  //float2*  s_r = (float2*) &s_params[num_types * NumParamEMT];
  // OR (seems to work better, not sure why)
  extern __shared__ float2 s_r[];
  float* s_params = (float*) &s_r[PPerBlock * TPerPart];
  

  float2 my_sigma = {0.f, 0.f};
  float2 my_dEds_E = {0.f, 0.f};
  
  float simBoxPtr_local[simulationBoxSize];
  simBox->loadMemory( simBoxPtr_local, simBoxPointer );


  float4 my_r = LOAD(r[MyGP]);
  int my_type = __float_as_int(my_r.w);
  
  
  const unsigned int tid = MyP + MyT*PPerBlock;
  for (unsigned int index=0; index<num_types*NumParamEMT; index += PPerBlock*TPerPart) {
    unsigned int myindex = index+tid;
    if (myindex<num_types*NumParamEMT)
      s_params[myindex] = params[myindex];
  }
  

  __syncthreads();

  
  int my_num_nbrs = num_nbrs[MyGP];
  int nb;
  int nb_prefetch = nbl[nvp*MyT + MyGP];
  float E0     = s_params[my_type*NumParamEMT];
  float V0     = s_params[my_type*NumParamEMT + 2];
  float eta2   = s_params[my_type*NumParamEMT + 3];
  float kappa  = s_params[my_type*NumParamEMT + 4];
  float lambda = s_params[my_type*NumParamEMT + 5];
  float n0     = s_params[my_type*NumParamEMT + 6];
  float r_cut  = s_params[my_type*NumParamEMT + 7];
  float a_cut  = s_params[my_type*NumParamEMT + 8];
  float gamma1 = s_params[my_type*NumParamEMT + 9];
  float gamma2 = s_params[my_type*NumParamEMT + 10];
  
  for (int i=MyT; i<my_num_nbrs; i+=TPerPart) {    
    nb = nb_prefetch;
    if(i+TPerPart < my_num_nbrs)
      nb_prefetch = nbl[nvp*(i+TPerPart) + MyGP];  // Last read not used, could decrease loop by one (TPerPart), and do last after loop
      // CAREFUL about num_nbrs<TPerPart case (eg virtual particles)
      float4 r_i = LOAD(r[nb]);
      int other_type =  __float_as_int(r_i.w);

      float other_s0     = s_params[other_type*NumParamEMT + 1];
      float other_eta2   = s_params[other_type*NumParamEMT + 3];
      float other_kappa  = s_params[other_type*NumParamEMT + 4];
      float other_n0     = s_params[other_type*NumParamEMT + 6];
      float chi = other_n0 / n0;
      
      float4 disp = simBox->calculateDistance(my_r, r_i, simBoxPtr_local);
      float dist = sqrtf(disp.w);
      float d_m_bs0 = dist - BETA*other_s0;
      float cutoff_factor = 1.f/(1.f+expf(a_cut*(dist-r_cut)));
      my_sigma.x += chi * expf(-other_eta2*d_m_bs0)*cutoff_factor;
      my_sigma.y += chi * expf(-other_kappa*d_m_bs0/BETA)*cutoff_factor;
  
    }

  s_r[MyP+MyT*PPerBlock] = my_sigma;

  __syncthreads();

  // sum contributions to sigmas
  if( MyT == 0 ) {
    for( int i=1; i < TPerPart; i++ ){
      my_sigma.x += s_r[MyP + i*PPerBlock].x;
      my_sigma.y += s_r[MyP + i*PPerBlock].y;
    }
    my_sigma.x /= gamma1;
    my_sigma.y /= gamma2;
    if (my_sigma.x > 0.f) {
      float s_m_s0 = -logf(my_sigma.x/12.f)/(BETA*eta2);
      float x = lambda*(s_m_s0);
      float exp1 = expf(-x);
      float exp2 = expf(-kappa*s_m_s0);

      // this is actually -dE/ds / (beta * eta2 * sigma1)
      // My sigma includes the factor 1/gamma1 (Jacob's doesn't)
      // so should include a gamma1 here

      my_dEds_E.x = (E0*x*lambda * exp1 + 6.f*V0*kappa * exp2) /
	(BETA*eta2*my_sigma.x*gamma1);

      // and energy
      my_dEds_E.y = E0 * (1.f+x) *  exp1  - 0.5f* V0 * ( my_sigma.y -
      12.f*exp2); // CORRECT EXPRESSION

      // store them in global array so available for next kernel
      dEds_E[MyGP] = my_dEds_E;
    }
  } // if (MyT == 0)

}



template<int STR, int INIT, class S>
__global__ void emt_calc_forces_after_energies( unsigned int num_part, unsigned int nvp, __restrict__ float4* r, __restrict__ float4* f, float4* w, float4* sts, float2* dEds_E, S* simBox, float* simBoxPointer, float *params, unsigned int num_types, unsigned int *num_nbrs, unsigned int *nbl) {


  extern __shared__ float4 s_f[];
  float* s_params = (float*) &s_f[PPerBlock * TPerPart];

  float simBoxPtr_local[simulationBoxSize];
  simBox->loadMemory( simBoxPtr_local, simBoxPointer );

  float4 my_r = LOAD(r[MyGP]);
  float2 my_dEds_E = dEds_E[MyGP];
  int my_type = __float_as_int(my_r.w);
  
  float4 my_f = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 my_w = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 my_sts = {0.0f, 0.0f, 0.0f, 0.0f};

  const unsigned int tid = MyP + MyT*PPerBlock;
  for (unsigned int index=0; index<num_types*NumParamEMT; index += PPerBlock*TPerPart) {
    unsigned int myindex = index+tid;
    if (myindex<num_types*NumParamEMT)
      s_params[myindex] = params[myindex];
  }
  
  __syncthreads();

  int my_num_nbrs = num_nbrs[MyGP];
  int nb;
  int nb_prefetch = nbl[nvp*MyT + MyGP];
  float s0     = s_params[my_type*NumParamEMT + 1];
  float V0     = s_params[my_type*NumParamEMT + 2];
  float eta2   = s_params[my_type*NumParamEMT + 3];
  float kappa  = s_params[my_type*NumParamEMT + 4];
  float n0     = s_params[my_type*NumParamEMT + 6];
  float r_cut  = s_params[my_type*NumParamEMT + 7];
  float a_cut  = s_params[my_type*NumParamEMT + 8];
  float gamma2 = s_params[my_type*NumParamEMT + 10];
  
  // to calculate the forces we need a new loop over neighbors
  nb_prefetch = nbl[nvp*MyT + MyGP];
  for (int i=MyT; i<my_num_nbrs; i+=TPerPart) {    
    nb = nb_prefetch;
    if(i+TPerPart < my_num_nbrs)
      nb_prefetch = nbl[nvp*(i+TPerPart) + MyGP];  // Last read not used, could decrease loop by one (TPerPart), and do last after loop
      // CAREFUL about num_nbrs<TPerPart case (eg virtual particles)
      float4 r_i = LOAD(r[nb]);
      float dEds_i = dEds_E[nb].x;
      int other_type =  __float_as_int(r_i.w);
      float4 disp = simBox->calculateDistance(my_r, r_i, simBoxPtr_local);
      float dist = sqrtf(disp.w);
      float cutoff_factor = 1.f/(1.f+expf(a_cut*(dist-r_cut)));
      float d_cut_dr = -a_cut * cutoff_factor * (1.f - cutoff_factor);

      float other_s0     = s_params[other_type*NumParamEMT + 1];
      float other_V0     = s_params[other_type*NumParamEMT + 2];
      float other_eta2   = s_params[other_type*NumParamEMT + 3];
      float other_kappa  = s_params[other_type*NumParamEMT + 4];
      float other_n0     = s_params[other_type*NumParamEMT + 6];
      float other_gamma2 = s_params[other_type*NumParamEMT + 10];
      float d_m_bs0_s = dist - BETA*s0;
      float d_m_bs0_o = dist - BETA*other_s0;

      float chi = other_n0 / n0;
      float dsigma1dr_o = expf(-other_eta2*d_m_bs0_o) * (d_cut_dr - cutoff_factor * other_eta2);
      float dsigma2dr_o = expf(-other_kappa*d_m_bs0_o/BETA) * (d_cut_dr - cutoff_factor * other_kappa/BETA);

      float dsigma1dr_s = expf(-eta2*d_m_bs0_s) * (d_cut_dr - cutoff_factor * eta2);
      float dsigma2dr_s = expf(-kappa*d_m_bs0_s/BETA) * (d_cut_dr - cutoff_factor * kappa/BETA);
      float df = (chi*(-my_dEds_E.x * dsigma1dr_o
		       +0.5f * V0 * dsigma2dr_o / gamma2)
		  + (1.f/chi)*(-dEds_i * dsigma1dr_s
		       +0.5f * other_V0 * dsigma2dr_s / other_gamma2)
		  ) / dist;
      my_f.x += df*disp.x;
      my_f.y += df*disp.y;
      my_f.z += df*disp.z;

      my_w.w += df*disp.w;

       if(STR){
	// stress - diagonal components
	my_sts.x -= disp.x * disp.x * df;  // xx
	my_sts.y -= disp.y * disp.y * df;  // yy
	my_sts.z -= disp.z * disp.z * df;  // zz
	// stress - off-diagonal components
	my_sts.w -=  disp.y * disp.z * df; // yz
	my_w.y   -=  disp.x * disp.z * df; // xz
	my_w.z   -=  disp.x * disp.y * df; // xy      
      }
 
  } // end force loop

  // __syncthreads(); NEED THIS IF OVERLAPPING SHARED MEMORY
  s_f[MyP + MyT*PPerBlock] = my_f;
  __syncthreads();

  if( MyT == 0 ) {
    for( int i=1; i < TPerPart; i++ )
      my_f += s_f[MyP + i*PPerBlock];
    my_f.w = my_dEds_E.y;
 
    if(INIT)
      f[MyGP] = my_f;
    else {
      atomicFloatAdd(&(f[MyGP].x), my_f.x);
      atomicFloatAdd(&(f[MyGP].y), my_f.y);
      atomicFloatAdd(&(f[MyGP].z), my_f.z);
      atomicFloatAdd(&(f[MyGP].w), my_f.w);
    }
  } // end if (MyT == 0)
  
  __syncthreads();  
  // Sum the atomic/molecular virial for all local threads.
  s_f[MyP+MyT*PPerBlock] = my_w;
  __syncthreads();  
  

  if( MyT == 0 ){
    for ( int i=1; i < TPerPart; i++ ){
      int j = MyP + __umul24(i,PPerBlock); 
      my_w.w += s_f[j].w;

      if(STR){
	my_w.y += s_f[j].y;
	my_w.z += s_f[j].z;
      }
    }
    if(INIT)
      w[MyGP] = my_w;
    else {
      atomicFloatAdd(&(w[MyGP].w), my_w.w);
      if(STR){
	atomicFloatAdd(&(w[MyGP].y), my_w.y);
	atomicFloatAdd(&(w[MyGP].z), my_w.z);
      }
    }
    } // if (MyT == 0)
    

  if(STR){
    __syncthreads();  
    
    // Sum the shear stress for all local threads.
    s_f[MyP+MyT*PPerBlock] = my_sts;
    
    __syncthreads();  
    
    if( MyT == 0 ){
      for ( int i=1; i < TPerPart; i++ ){
	int j = MyP + __umul24(i,PPerBlock); 
	my_sts += s_f[j];
      }

      if(INIT)
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
