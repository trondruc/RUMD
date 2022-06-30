#include "rumd/rumd_technical.h"
#include "rumd/rumd_algorithms.h"
#include "rumd/HarmonicUmbrella.h"


__global__ void calculate_umbrella_force(unsigned int numParticles, float4* full_forces, float4* local_forces, float* Q_array, float springConst, float Q0);


__global__ void copy_Q_for_summing(unsigned int numParticles, float4* local_forces, float* Q);

HarmonicUmbrella::HarmonicUmbrella(PairPotential* set_pairPot, float set_springConst, float set_Q0) : pairPot(set_pairPot), springConst(set_springConst), Q0(set_Q0), d_Q(0), allocatedSize(0) {
  SetID_String("potUmbr");
}


HarmonicUmbrella::~HarmonicUmbrella() {
  if(allocatedSize != 0) {
    cudaFree(d_Q);
  }
}

void HarmonicUmbrella::Initialize() {
  // HarmonicUmbrella both *is* a potential and *has* a potential
  // and in both cases SetSample needs to be called
  // (which calls Initialize)

  //pairPot->SetSampleParameters(kPlan,pd, simBox, ss);
  pairPot->SetSample(sample);
  AllocateQ_Array(particleData->GetNumberOfVirtualParticles());
}


void HarmonicUmbrella::AllocateQ_Array(unsigned int np) {
  if(np == allocatedSize)
    return;

  if(allocatedSize != 0) {
    cudaFree(d_Q);
  }
  
  if( cudaMalloc( (void**) &d_Q, np * sizeof(float) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("HarmonicUmbrella",__func__,"cudaMalloc failed on d_Q") );
  cudaMemset( d_Q, 0, np * sizeof(float) );
  
  allocatedSize = np;
}


void HarmonicUmbrella::SetParams(float set_springConst, float set_Q0) {
  springConst = set_springConst;
  Q0 = set_Q0;
}

double HarmonicUmbrella::GetPotentialEnergy() {

  float Qtot = GetOrderParameter();
  double pot_en = 0.5 * springConst * (Qtot-Q0) * (Qtot-Q0);
  
  return pot_en;
}

float HarmonicUmbrella::GetOrderParameter() {
  if(!particleData)
    throw RUMD_Error("HarmonicUmbrella",__func__,"This object has not been assigned to a sample object yet!");


  // copy Q[0] from device. This assumes CalcF has been called so the sum 
  // has been performed
  float Qtot = -1.0;
  cudaMemcpy(&Qtot, d_Q, sizeof(float), cudaMemcpyDeviceToHost);
  Qtot /= particleData->GetNumberOfParticles(); // should really divide by N/2 for the case of two replicas 
  return Qtot;
}


void HarmonicUmbrella::CalcF(bool initialize, bool calc_stresses) {
  if(calc_stresses) throw RUMD_Error("HarmonicUmbrella",__func__,"Calculation of stresses not implemented");

  // could optimize this step via template argument, but initialize is probably
  // going to be false...
  if(initialize)
    particleData->SetForcesToZero();

  pairPot->CalcF_Local(); //puts the forces on the local array
  float4* local_force_array = pairPot->GetLocalForceArray();
  
  unsigned int nParticles = particleData->GetNumberOfParticles();
  dim3 threads = kp.threads;
  threads.y = 1;

  // need to copy to the Q array, and then sum on the device
  copy_Q_for_summing<<<kp.grid, threads>>>(nParticles, local_force_array, d_Q);
  sumIdenticalArrays( d_Q, nParticles, 1, 32 );
  
  // then add to the actual force
  calculate_umbrella_force<<<kp.grid, threads >>>(nParticles, particleData->d_f, local_force_array, d_Q, springConst, Q0);

}
 

__global__ void calculate_umbrella_force(unsigned int numParticles, float4* full_forces, float4* local_forces, float* Q_array, float springConst, float Q0) {
  if ( MyGP < numParticles ) {
    float4 my_full_force = full_forces[MyGP];
    float4 my_local_force = local_forces[MyGP];

    float Q = Q_array[0]/numParticles;
    
    // our definition of Q includes dividing by the number of particles,
    // which is not usually included when calculating forces as gradient
    // of a potential energy. So we need to divide the local forces by
    // numParticles, as well as dividing Q by numParticles.
    float mult_factor = springConst * (Q-Q0)/numParticles;

    my_full_force.x += mult_factor * my_local_force.x;
    my_full_force.y += mult_factor * my_local_force.y;
    my_full_force.z += mult_factor * my_local_force.z;
    // should not contribute to particle energies, so don't change .w component
    full_forces[MyGP] = my_full_force;
  }

}

__global__ void copy_Q_for_summing(unsigned int numParticles, float4* local_forces, float* Q) {
  if ( MyGP < numParticles ) {
    float4 my_local_force = local_forces[MyGP];
    Q[MyGP] = my_local_force.w;
  }
}
