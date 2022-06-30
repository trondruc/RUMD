
/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/BondPotential.h"
#include "rumd/Sample.h"
#include "rumd/MoleculeData.h"
#include "rumd/SimulationBox.h"
#include "rumd/MoleculeData.h"
#include "rumd/rumd_algorithms.h"



///////////////////////////////////////////////////////
// Harmonic potential
///////////////////////////////////////////////////////


void BondPotential::SetParams(unsigned bond_type, float length_param, float stiffness_param, bool exclude ) {
  std::vector<float> prms;
  prms.push_back(length_param);
  prms.push_back(stiffness_param);
  
  bond_params[bond_type] = prms;
  exclude_bond[bond_type] = exclude;

  if(sample && sample->GetMoleculeData())
    CopyParamsToGPU();
}

void BondPotential::Initialize() {
  MoleculeData* moleculeData = sample->GetMoleculeData();

  if(!moleculeData)
    throw RUMD_Error("BondPotential", __func__, "No molecular data available, call ReadMoleculeData");

  CopyParamsToGPU();
}


void BondPotential::CopyParamsToGPU() {
  MoleculeData* moleculeData = sample->GetMoleculeData();
  
  std::map<unsigned, std::vector<float> >::iterator params_it;
  for(params_it = bond_params.begin(); params_it != bond_params.end(); params_it++) {
    unsigned b_type = params_it->first;
    std::vector<float> b_params = params_it->second;
    moleculeData->SetBondParams(b_type, b_params[0], b_params[1], bond_pot_class);
  }
}


void BondPotential::SetExclusions(PairPotential* non_bond_pot ) {
  MoleculeData* moleculeData = sample->GetMoleculeData();
  std::map<unsigned, bool>::iterator excl_it;
  for(excl_it = exclude_bond.begin(); excl_it != exclude_bond.end(); excl_it++) {
    if(excl_it->second)
      moleculeData->SetExclusionBond(non_bond_pot, excl_it->first);
  }
}


BondHarmonic::BondHarmonic() : BondPotential() {
  ID_String = "bondHarmonic";
  bond_pot_class = 0;
}


void BondHarmonic::CalcF(bool initialize, bool calc_stresses){

  if ( initialize )
    particleData->SetForcesToZero();
  
  RectangularSimulationBox* RSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  LeesEdwardsSimulationBox* LESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);
  
  MoleculeData* M = sample->GetMoleculeData();
  unsigned num_bonds = M->GetNumberOfBonds();
  unsigned num_blocks = num_bonds/num_threads + 1;

  if ( LESB )
    if( calc_stresses )
      Harmonic<2, 1><<<num_blocks,num_threads>>>( particleData->d_r, 
						  particleData->d_f,
						  particleData->d_w,
						  particleData->d_sts,
						  M->d_blist, M->d_btlist, 
						  M->d_bplist, M->d_belist, 
						  M->d_btlist_int, M->d_bonds,
						  num_bonds, LESB, 
						  LESB->GetDevicePointer() );
    else
      Harmonic<1, 1><<<num_blocks,num_threads>>>( particleData->d_r,
						  particleData->d_f,
						  particleData->d_w,
						  particleData->d_sts,
						  M->d_blist, M->d_btlist, 
						  M->d_bplist, M->d_belist, 
						  M->d_btlist_int, M->d_bonds,
						  num_bonds, LESB, 
						  LESB->GetDevicePointer() );


  else if ( RSB )
    if ( calc_stresses )
      Harmonic<2, 1><<<num_blocks,num_threads>>>( particleData->d_r,
						  particleData->d_f,
						  particleData->d_w,
						  particleData->d_sts,
						  M->d_blist, M->d_btlist,
						  M->d_bplist, M->d_belist, 
						  M->d_btlist_int, M->d_bonds,
						  num_bonds, RSB,
						  RSB->GetDevicePointer());
    else
      Harmonic<1, 1><<<num_blocks,num_threads>>>( particleData->d_r,
						  particleData->d_f,
						  particleData->d_w,
						  particleData->d_sts,
						  M->d_blist, M->d_btlist,
						  M->d_bplist, M->d_belist, 
						  M->d_btlist_int, M->d_bonds,
						  num_bonds, RSB,
						  RSB->GetDevicePointer());
  else
    throw RUMD_Error("BondHarmonic","CalcF","unknown simulation box");
}

double BondHarmonic::GetPotentialEnergy(){
  
  RectangularSimulationBox* RSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  LeesEdwardsSimulationBox* LESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);

  MoleculeData* M = sample->GetMoleculeData();
  unsigned num_bonds = M->GetNumberOfBonds();
  unsigned num_blocks = num_bonds/num_threads + 1;
  
  if ( LESB )
    Harmonic< 1, 2 ><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	particleData->d_w, particleData->d_sts,
	M->d_blist, M->d_btlist, M->d_bplist,
	M->d_belist, M->d_btlist_int, M->d_bonds,
	num_bonds, LESB, LESB->GetDevicePointer() );
  else if ( RSB )
    Harmonic<1, 2><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	particleData->d_w, particleData->d_sts,
        M->d_blist, M->d_btlist, M->d_bplist,
        M->d_belist, M->d_btlist_int, M->d_bonds,
        num_bonds, RSB, RSB->GetDevicePointer() );
  else
    throw RUMD_Error("BondHarmonic","GetPotentialEnergy","unknown simulation box");

  // copy to host
  size_t nbytes = num_bonds*sizeof(float);
  cudaMemcpy(M->h_belist, M->d_belist, nbytes, cudaMemcpyDeviceToHost);
  
  // sum over bonds
  double pe = 0.0;
  for ( unsigned i=0; i<num_bonds; i++ ){
    unsigned bond_type = M->h_btlist[i].x;
    if ( M->h_btlist[bond_type].x == 0 )
      pe += (double)M->h_belist[i];
  }  

  return pe;
}

///////////////////////////////////////////////////////
// FENE potential
///////////////////////////////////////////////////////

BondFENE::BondFENE() : BondPotential() {
  ID_String = "bondFene";
  bond_pot_class = 1;
}


void BondFENE::CalcF(bool initialize, bool calc_stresses){

  if ( initialize )
    particleData->SetForcesToZero();


  RectangularSimulationBox* RSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  LeesEdwardsSimulationBox* LESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);

  MoleculeData* M = sample->GetMoleculeData();
  unsigned num_bonds = M->GetNumberOfBonds();
  unsigned num_blocks = num_bonds/num_threads + 1;


  if ( LESB )
    if( calc_stresses )
      FENE<2, 1><<<num_blocks, num_threads>>>
	( particleData->d_r, particleData->d_f,
	  particleData->d_w, particleData->d_sts,
	  M->d_blist, M->d_btlist, M->d_bplist,
	  M->d_belist, M->d_btlist_int, M->d_bonds,
	  num_bonds, LESB, LESB->GetDevicePointer() );
    else
      FENE<1, 1><<<num_blocks, num_threads>>>
	( particleData->d_r, particleData->d_f,
	  particleData->d_w, particleData->d_sts,
	  M->d_blist, M->d_btlist, M->d_bplist,
	  M->d_belist, M->d_btlist_int, M->d_bonds,
	  num_bonds, LESB, LESB->GetDevicePointer() );
  else if ( RSB )
    if( calc_stresses )
      FENE<2, 1><<<num_blocks, num_threads>>>
	( particleData->d_r, particleData->d_f,
	  particleData->d_w, particleData->d_sts,
	  M->d_blist, M->d_btlist, M->d_bplist,
	  M->d_belist, M->d_btlist_int, M->d_bonds,
	  num_bonds, RSB, RSB->GetDevicePointer() );
    else
      FENE<1, 1><<<num_blocks, num_threads>>>
	( particleData->d_r, particleData->d_f,
	  particleData->d_w, particleData->d_sts,
	  M->d_blist, M->d_btlist, M->d_bplist,
	  M->d_belist, M->d_btlist_int, M->d_bonds,
	  num_bonds, RSB, RSB->GetDevicePointer() );
  else
    throw RUMD_Error("BondFENE","CalcF","unknown simulation box");

}

double BondFENE::GetPotentialEnergy(){
  
  RectangularSimulationBox* RSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  LeesEdwardsSimulationBox* LESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);

  MoleculeData* M = sample->GetMoleculeData();
  unsigned num_bonds = M->GetNumberOfBonds();
  unsigned num_blocks = num_bonds/num_threads + 1;

  if ( LESB )
    FENE< 1, 2 ><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	particleData->d_w, particleData->d_sts,
	M->d_blist, M->d_btlist, M->d_bplist,
	M->d_belist, M->d_btlist_int, M->d_bonds,
	num_bonds, LESB, LESB->GetDevicePointer() );
  else if ( RSB )
    FENE<1, 2><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	particleData->d_w, particleData->d_sts,
	M->d_blist, M->d_btlist, M->d_bplist,
	M->d_belist, M->d_btlist_int, M->d_bonds,
	num_bonds, RSB, RSB->GetDevicePointer() );
  else
    throw RUMD_Error("BondFENE","GetPotentialEnergy","unknown simulation box");
  
  // copy to host
  size_t nbytes = num_bonds*sizeof(float);
  cudaMemcpy(M->h_belist, M->d_belist, nbytes, cudaMemcpyDeviceToHost);
  
  // sum over (relevant) bonds
  double pe = 0.0;
  for ( unsigned i=0; i<num_bonds; i++ ){
    unsigned bond_type = M->h_btlist[i].x;
    if ( M->h_btlist_int[bond_type].x == 1 )
      pe +=  (double)M->h_belist[i];
  }  

  return pe;
}

///////////////////////////////////////////////////////
// Kernel implementations of bond potentials
///////////////////////////////////////////////////////

template <int stress, int energy, class Simbox> 
__global__ void Harmonic( float4 *r, float4 *f, float4 *w, float4 *sts,
			  uint2 *blist, uint1 *btlist, float2 *bplist,
			  float *belist, int1 *btlist_int, float *bonds,
			  unsigned num_bonds, Simbox *simbox, float *simboxpointer ){

  unsigned bond_index = blockIdx.x*blockDim.x + threadIdx.x;

  if ( bond_index < num_bonds ){

    unsigned bond_type = btlist[bond_index].x;

    if ( btlist_int[bond_type].x == 0 ){

      // Get bond information
      unsigned a   =  blist[bond_index].x;
      unsigned b   =  blist[bond_index].y;
      float r0 = bplist[bond_type].x; // bond length
      float k  = bplist[bond_type].y; // spring constant
      
      // Load the simulation box in local memory to avoid bank conflicts.
      float array[simulationBoxSize];

      simbox->loadMemory(array, simboxpointer);
      
      // Calculate stuff
      float4 dr = simbox->calculateDistance(r[a], r[b], array);
      float rij = sqrt(dr.w); // squared distance
      bonds[bond_index] = rij;

      float pe = 0.5f*k*( rij-r0 )*( rij-r0 ); // potential energy

      if ( energy == 2 )
	belist[bond_index] = pe;
      else if ( energy == 1 ){
	float ft = -k*(1.0f - r0/rij); // force
	atomicFloatAdd(&(f[a].x), ft*dr.x);
	atomicFloatAdd(&(f[a].y), ft*dr.y);
	atomicFloatAdd(&(f[a].z), ft*dr.z);
	atomicFloatAdd(&(f[a].w), 0.5f*pe);
	
	atomicFloatAdd(&(f[b].x), -ft*dr.x);
	atomicFloatAdd(&(f[b].y), -ft*dr.y);
	atomicFloatAdd(&(f[b].z), -ft*dr.z);
	atomicFloatAdd(&(f[b].w), 0.5f*pe);
	
	if ( stress >= 1 ){
	  float virial = ft*dr.w; // *( dr.x*dr.x + dr.y*dr.y + dr.z+dr.z );
	  atomicFloatAdd(&(w[a].w), virial ); 
	  atomicFloatAdd(&(w[b].w), virial ); 
	  if( stress == 2 ) {
	    float my_stress[6] = {-ft*dr.x*dr.x, -ft*dr.y*dr.y,
				  -ft*dr.z*dr.z, -ft*dr.y*dr.z,
				  -ft*dr.x*dr.z, -ft*dr.x*dr.y};
	    
	    
	    atomicFloatAdd(&(sts[a].x), my_stress[0] );
	    atomicFloatAdd(&(sts[a].y), my_stress[1] );
	    atomicFloatAdd(&(sts[a].z), my_stress[2] );
	    atomicFloatAdd(&(sts[a].w), my_stress[3] );
	    atomicFloatAdd(&(w[a].y), my_stress[4] );
	    atomicFloatAdd(&(w[a].z), my_stress[5] );
	    
	    atomicFloatAdd(&(sts[b].x), my_stress[0] );
	    atomicFloatAdd(&(sts[b].y), my_stress[1] );
	    atomicFloatAdd(&(sts[b].z), my_stress[2] );
	    atomicFloatAdd(&(sts[b].w), my_stress[3] );
	    atomicFloatAdd(&(w[b].y), my_stress[4] );
	    atomicFloatAdd(&(w[b].z), my_stress[5] );
	  }
	  
	}
      }
    }
  }
}


template <int stress, int energy, class Simbox>
__global__ void FENE( float4 *r, float4 *f, float4 *w, float4 *sts,
		      uint2 *blist, uint1 *btlist, float2 *bplist,
		      float *belist, int1 *btlist_int, float *bonds, 
		      unsigned num_bonds, Simbox *simbox, float *simboxpointer ){
  
  unsigned bond_index = blockIdx.x*blockDim.x + threadIdx.x;

  if( bond_index < num_bonds ){

    unsigned bond_type = btlist[bond_index].x;

    if( btlist_int[bond_type].x == 1 ){

      // Get bond information
      unsigned a = blist[bond_index].x; // particle index
      unsigned b = blist[bond_index].y; // particle index
      float r0 = bplist[bond_type].x; // max bond length
      float k0 = bplist[bond_type].y; // prefactor

      // Load the simulation box in local memory to avoid bank conflicts.
      float array[simulationBoxSize];
      simbox->loadMemory(array, simboxpointer);

      // Do calculation
      float4 dr = simbox->calculateDistance(r[a], r[b], array);

      float rij = sqrt(dr.w); // not necesary for force 
      bonds[bond_index] = rij;

      float rr = dr.w / (r0*r0);
      float ft = - k0/(1.0f - rr);
      float pe = - 0.5f* r0*r0 * k0 * logf(1.0f - rr);

      if ( energy == 2 )
	belist[bond_index] = pe;
      else if ( energy == 1 ){
	atomicFloatAdd(&(f[a].x), ft*dr.x);
	atomicFloatAdd(&(f[a].y), ft*dr.y);
	atomicFloatAdd(&(f[a].z), ft*dr.z);
	atomicFloatAdd(&(f[a].w), 0.5f*pe);

	atomicFloatAdd(&(f[b].x), -ft*dr.x);
	atomicFloatAdd(&(f[b].y), -ft*dr.y);
	atomicFloatAdd(&(f[b].z), -ft*dr.z);
	atomicFloatAdd(&(f[b].w), 0.5f*pe);
	if ( stress >= 1 ){
	  float virial = ft*dr.w;// *( dr.x*dr.x + dr.y*dr.y + dr.z+dr.z );

	  atomicFloatAdd(&(w[a].w), virial );
	  atomicFloatAdd(&(w[b].w), virial );
	  
	  if( stress == 2 ) {
	    float my_stress[6] = {-ft*dr.x*dr.x, -ft*dr.y*dr.y,
				  -ft*dr.z*dr.z, -ft*dr.y*dr.z,
				  -ft*dr.x*dr.z, -ft*dr.x*dr.y};

	    atomicFloatAdd(&(sts[a].x), my_stress[0] );
	    atomicFloatAdd(&(sts[a].y), my_stress[1] );
	    atomicFloatAdd(&(sts[a].z), my_stress[2] );
	    atomicFloatAdd(&(sts[a].w), my_stress[3] );
	    atomicFloatAdd(&(w[a].y), my_stress[4] );
	    atomicFloatAdd(&(w[a].z), my_stress[5] );
	    
	    atomicFloatAdd(&(sts[b].x), my_stress[0] );
	    atomicFloatAdd(&(sts[b].y), my_stress[1] );
	    atomicFloatAdd(&(sts[b].z), my_stress[2] );
	    atomicFloatAdd(&(sts[b].w), my_stress[3] );
	    atomicFloatAdd(&(w[b].y), my_stress[4] );
	    atomicFloatAdd(&(w[b].z), my_stress[5] );
	  }


	}
      }
    }
  }
}
