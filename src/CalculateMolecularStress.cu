#include "rumd/Sample.h"
#include "rumd/MoleculeData.h"
#include "rumd/Potential.h"
#include "rumd/PairPotential.h"

#include <algorithm>


#include <cstdio>

template <int CUTOFF, class P, class S>
__global__ void mol_stress_tensor(P* pot, float *stress, 
				  float4 *r, float3 *vcm, float4 *rcm,
				  int1 *d_mlist, unsigned max_num_uau,
				  unsigned num_mol,  S *simbox,
				  float *simboxpointer,
				  unsigned num_types,
				  const float* pot_params);

__global__ void set_stress_to_zero(float *stress, int num_mol);
__global__ void add_kinetic_mol_stress(float *stress, int num_mol, float3 *vcm, float4 *rcm);

void CalculateMolecularStress(Sample* S) {

  MoleculeData* M = S->GetMoleculeData();
  unsigned int num_mol = M->GetNumberOfMolecules();
  unsigned int max_mol_size = M->GetMaximumMoleculeSize();

  M->EvalVelCM();
  M->EvalCM();

  ParticleData* particleData = S->GetParticleData();

  size_t num_threads = 32;
  size_t num_blocks = (9*num_mol)/num_threads + 1;
  set_stress_to_zero<<<num_blocks, num_threads>>>(M->d_stress, num_mol);
  num_blocks = (num_mol)/num_threads + 1;
  add_kinetic_mol_stress<<<num_blocks, num_threads>>>(M->d_stress, num_mol, M->d_vcm, M->d_cm);

  //loop over potentials
  unsigned int threads_per_molecule = 256;
  size_t nbytes_shared = sizeof(float)*9*threads_per_molecule;
  std::vector<Potential*>* potentialList = S->GetPotentials();

  std::vector<Potential*>::iterator potIter;
  LeesEdwardsSimulationBox* test_LESB = dynamic_cast<LeesEdwardsSimulationBox*>(S->GetSimulationBox());
  RectangularSimulationBox* test_RSB = dynamic_cast<RectangularSimulationBox*>(S->GetSimulationBox());
  if ( (!test_LESB) && (!test_RSB) )
      throw RUMD_Error("[global]","CalculateMolecularStress","Require RectangularSimulationBox or LeesEdwardsSimulationBox");
  for(potIter = (*potentialList).begin(); potIter != (*potentialList).end(); potIter++) {

    // the following included file is automatically generated and
    // contains the calls to mol_stress_tensor for every combination of 
    // potential type, simbox type and cutoff method.

    #include "MolecularStress_Instantiation.inc"
  }
  

  cudaMemcpy(M->h_stress, M->d_stress, num_mol*9*sizeof(float), cudaMemcpyDeviceToHost);
  float stress[3][3];
  for ( int k=0; k<3; k++ )
    for ( int kk=0; kk<3; kk++ ) stress[k][kk] = 0.0f;

  for ( unsigned n=0; n<num_mol; n++ )
    for ( int k=0; k<3; k++ )
      for ( int kk=0; kk<3; kk++ )
        stress[k][kk] += M->h_stress[9*n + 3*k + kk];

  float vol = S->GetSimulationBox()->GetVolume();
  //Symmetric part of the stress (order: xx, yy, zz, yz, xz, xy)
  M->symmetricStress[0] = stress[0][0]/vol;
  M->symmetricStress[1] = stress[1][1]/vol;
  M->symmetricStress[2] = stress[2][2]/vol;
  M->symmetricStress[3] = 0.5 * (stress[1][2] + stress[2][1])/vol;
  M->symmetricStress[4] = 0.5 * (stress[0][2] + stress[2][0])/vol;
  M->symmetricStress[5] = 0.5 * (stress[0][1] + stress[1][0])/vol;


}

__global__ void set_stress_to_zero(float *stress, int num_mol) {

  unsigned stress_index = blockIdx.x*blockDim.x + threadIdx.x;

  if (stress_index < 9*num_mol) {
    stress[stress_index] = 0.0;
 }
   
}

__global__ void add_kinetic_mol_stress(float *stress, int num_mol, float3 *vcm, float4 *rcm){

  unsigned i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < num_mol) {
    float P_kin[3][3];
      
    P_kin[0][0] = vcm[i].x*vcm[i].x*rcm[i].w;
    P_kin[0][1] = vcm[i].x*vcm[i].y*rcm[i].w;  
    P_kin[0][2] = vcm[i].x*vcm[i].z*rcm[i].w;  
    P_kin[1][1] = vcm[i].y*vcm[i].y*rcm[i].w;
    P_kin[1][2] = vcm[i].y*vcm[i].z*rcm[i].w;
    P_kin[2][2] = vcm[i].z*vcm[i].z*rcm[i].w;
    P_kin[1][0] = P_kin[0][1];
    P_kin[2][0] = P_kin[0][2];
    P_kin[2][1] = P_kin[1][2];


    // total
    for ( int k=0; k<3; k++ )
      for ( int kk=0; kk<3; kk ++ )
        stress[9*i + k*3 + kk] -= P_kin[k][kk];
  }

}

template <int CUTOFF, class P, class S>
__global__ void mol_stress_tensor(P* Pot, float *stress,
				  float4 *r, float3 *vcm, float4 *rcm,  
				  int1 *mlist, unsigned max_num_uau, 
				  unsigned num_mol,  S *simbox, 
				  float *simboxpointer,
				  unsigned int num_types,
				  const float* pot_params){ 
  extern __shared__ float P_pot_i[];
  const int offset_p = threadIdx.x*9;
  float4 my_w = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 my_sts = {0.0f, 0.0f, 0.0f, 0.0f};
  
  // Load the simulation box in local memory to avoid bank conflicts.
  float simBoxPtr_local[simulationBoxSize];
  simbox->loadMemory(simBoxPtr_local, simboxpointer);
  
  
  // Molecule index indices
  unsigned i = blockIdx.x;
  // Number of atoms in molecule i
  unsigned nuau_i = mlist[i].x;
  // Base offsets in mlist
  size_t offset_i = max_num_uau*i + num_mol;

  // need to loop over molecules j (num_mol). Partly done by parallelization
  // (threads within the block); blockDim.x is the number of threads per 
  // molecule i.
  // In general this is smaller than the number of molecules
  // so have an extra loop
  
  // initialize this thread's assigned part of shared memory to zero
  // no need to synchronize until we're actually adding up contributions
  for ( int n=0; n<9; n++ ) P_pot_i[n+offset_p] = 0.0f;
    
  for (unsigned jdx = 0; jdx*blockDim.x<num_mol; jdx++) {
    
    unsigned j = jdx*blockDim.x + threadIdx.x;

    if ( j != i && j < num_mol ) {
      // Number of atoms in molecules 
      unsigned nuau_j = mlist[j].x;  
      // Base offsets in mlist
      size_t offset_j = max_num_uau*j + num_mol;
      
      // Loop over all interactions
      float4 Fij = {0.0f, 0.0f, 0.0f, 0.0f};
      for ( unsigned ia=offset_i; ia<offset_i+nuau_i; ia++ ){
	
	int index_ia = mlist[ia].x; 
	for ( unsigned jb=offset_j; jb<offset_j+nuau_j; jb++ ){
	  
	  int index_jb = mlist[jb].x;  
	  
	  float4 ria = r[index_ia];
	  float4 rjb = r[index_jb];
	  int type_ia = __float_as_int(ria.w);
	  int type_jb = __float_as_int(rjb.w);
	  const float* params_this_pair = pot_params + type_ia*num_types*NumParam + type_jb*NumParam;
	  fij<0, CUTOFF>( Pot, ria, rjb, &Fij, &my_w, &my_sts, params_this_pair, simbox, simBoxPtr_local );
	}
      }
      // Get the potential part of the pressure tensor
      float4 drcm = simbox->calculateDistance(rcm[i], rcm[j], simBoxPtr_local);
      
      P_pot_i[offset_p]   += drcm.x*Fij.x;
      P_pot_i[offset_p+1] += drcm.x*Fij.y;
      P_pot_i[offset_p+2] += drcm.x*Fij.z;
      P_pot_i[offset_p+3] += drcm.y*Fij.x;
      P_pot_i[offset_p+4] += drcm.y*Fij.y;
      P_pot_i[offset_p+5] += drcm.y*Fij.z;
      P_pot_i[offset_p+6] += drcm.z*Fij.x;
      P_pot_i[offset_p+7] += drcm.z*Fij.y;
      P_pot_i[offset_p+8] += drcm.z*Fij.z;
      
    }
  } // jdx  
  
  __syncthreads();
  
  // Add all contributions to molecule i
  if ( threadIdx.x == 0 ){

    // Potential part
    float P_pot[3][3];
    for ( int k=0; k<3; k++ )
      for ( int kk=0; kk<3; kk ++ ) P_pot[k][kk] = 0.0f;
    
    for ( int n=0; n<blockDim.x; n++ )
      for ( int k=0; k<3; k++ )
	for ( int kk=0; kk<3; kk ++ ) 
	  P_pot[k][kk] += P_pot_i[9*n + 3*k + kk];

    // total
    for ( int k=0; k<3; k++ ) {
      for ( int kk=0; kk<3; kk ++ ) {
	stress[9*i + k*3 + kk] += -0.5*P_pot[k][kk];
      }
    }
  } // if (threadIdx.x == 0)
}

