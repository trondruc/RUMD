
/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/IntegratorMMC.h"
#include "rumd/Sample.h"
#include "rumd/PairPotential.h"
#include "rumd/rumd_algorithms.h"
#include "rumd/RUMD_Error.h"
#include "rumd/ParseInfoString.h"

#include <ctime>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

////////////////////////////////////////////////////////////
// Constructors 
////////////////////////////////////////////////////////////

IntegratorMMC::IntegratorMMC(float dispLength, float targetTemperature){
  temperature = targetTemperature;
  timeStep = dispLength;
  numParticlesSample = 0;
  integrationsPerformed = 0;
  previousAcceptanceRate = 0;
}

IntegratorMMC::~IntegratorMMC(){
  FreeArrays();
}

////////////////////////////////////////////////////////////
// Class Methods
////////////////////////////////////////////////////////////

// Integrate Metropolis Monte Carlo with all particle moves.
void IntegratorMMC::Integrate(){
  if(!S || !(S->GetNumberOfParticles())) // Consistency check with itg.
    throw( RUMD_Error("IntegratorMMC","Integrate","There is no integrator associated with Sample or it has no particles" ) );

  RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(S->GetSimulationBox());

  integrateMMCAlgorithm<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_r, d_previous_position, P->d_f, d_previous_force, 
						    P->d_im, d_previous_image, P->d_misc, d_previous_misc, testRSB, testRSB->GetDevicePointer(), 
						    d_hybridTausSeeds, timeStep );
  integrationsPerformed++;

  // Calculate force.
  S->CalcF();
  
  calculatePotentialEnergy<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_f, d_summationArray );
  sumIdenticalArrays( d_summationArray, numParticlesSample, 1, 32 );
  
  // Accept the move?
  acceptMMCmove<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_r, d_previous_position, P->d_f, d_previous_force, 
					    P->d_im, d_previous_image, P->d_misc, d_previous_misc, d_summationArray, 
					    d_previousPotentialEnergy, d_hybridTausSeeds, d_movesAccepted, 1.f/temperature );
  
  // Trigger neighborlist update if the move was not accepted. Alternative is to set NBSkin.
  std::vector<Potential*>* vec = S->GetPotentials();
  
  unsigned long int currentAcceptanceRate = 0;
  cudaMemcpy( &currentAcceptanceRate, d_movesAccepted, sizeof(unsigned long int), cudaMemcpyDeviceToHost );

  // NB The function TriggerNBListUpdate was unnecessary since ResetNeighborList already existed for that purpose. Also this should be handled automatically so no call should be necessary here. Commenting out, but replacing TriggerNBListUpdate with ResetNeighborList in case someone decides it really is necessary. NB 22/11/18
  /*if( previousAcceptanceRate == currentAcceptanceRate ){
    for ( std::vector<Potential*>::iterator itr = vec->begin(); itr < vec->end(); itr++ ){
      PairPotential* testPairPotential = dynamic_cast<PairPotential*>(*itr);
      
      if(testPairPotential)
	testPairPotential->ResetNeighborList();
    }
    }*/
  previousAcceptanceRate = currentAcceptanceRate;
}

// Generate initial seeds using a host side PRNG.
void IntegratorMMC::GenerateHybridTausSeeds( uint4* seeds, unsigned numParticles ){
  boost::mt19937 gen_taus(static_cast<unsigned> (std::time(0)));
  boost::uniform_int<> dist_taus(129, RAND_MAX); // Must be larger than 128!
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > taus(gen_taus, dist_taus);
  
  boost::mt19937 gen_lcg(static_cast<unsigned> (std::time(0)));
  boost::uniform_int<> dist_lcg(0, RAND_MAX); // Any random value.
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > lcg(gen_lcg, dist_lcg);
    
  unsigned taux = taus();
  unsigned tauy = taus();
  unsigned tauz = taus();
  unsigned ccgw = lcg();
  
  // One random number generator pr. particle.
  for( unsigned i=0; i < numParticles; i++ ){
    seeds[i].x = taus();
    seeds[i].y = taus();
    seeds[i].z = taus();
    seeds[i].w = lcg();
    
    // Same state replicated.
    seeds[numParticles+i].x = taux;
    seeds[numParticles+i].y = tauy;
    seeds[numParticles+i].z = tauz;
    seeds[numParticles+i].w = ccgw;
  }
}

////////////////////////////////////////////////////////////
// Allocation
////////////////////////////////////////////////////////////

void IntegratorMMC::AllocateIntegratorState(){
  if(!S || !(S->GetNumberOfParticles())) // Consistency check with itg.
    throw( RUMD_Error("IntegratorMMC","AllocateIntegratorState","There is no integrator associated with Sample or it has no particles" ) );
  
  unsigned int newNumParticlesS = S->GetNumberOfParticles();

  if(newNumParticlesS != numParticlesSample){
    FreeArrays();
    numParticlesSample = newNumParticlesS;
    
    // CPU
    if( cudaMallocHost( (void**) &h_hybridTausSeeds, 2 * numParticlesSample * sizeof(uint4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorMMC","AllocateIntegratorState","Malloc failed on h_hybridTausSeeds") );
    
    if( cudaMallocHost( (void**) &h_previousPotentialEnergy, kp.grid.x * sizeof(float) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorMMC","AllocateIntegratorState","Malloc failed on h_previousPotentialEnergy") );
    
    // GPU
    if( cudaMalloc( (void**) &d_hybridTausSeeds, 2 * numParticlesSample * sizeof(uint4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorMMC","AllocateIntegratorState","Malloc failed on d_hybridTausSeeds") );
    
    if( cudaMalloc( (void**) &d_previous_position, numParticlesSample * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorMMC","AllocateIntegratorState","Malloc failed on d_previous_position") );
    
    if( cudaMalloc( (void**) &d_previous_force, numParticlesSample * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorMMC","AllocateIntegratorState","Malloc failed on d_previous_force") );
    
    if( cudaMalloc( (void**) &d_previous_image, numParticlesSample * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorMMC","AllocateIntegratorState","Malloc failed on d_previous_image") );
    
    if( cudaMalloc( (void**) &d_previous_misc, numParticlesSample * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorMMC","AllocateIntegratorState","Malloc failed on d_previous_misc") );
    
    if( cudaMalloc( (void**) &d_summationArray, numParticlesSample * sizeof(float) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorMMC","AllocateIntegratorState","Malloc failed on d_summationArray") );
    
    if( cudaMalloc( (void**) &d_previousPotentialEnergy, kp.grid.x * sizeof(float) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorMMC","AllocateIntegratorState","Malloc failed on d_previousPotentialEnergy") );
    
    if( cudaMalloc( (void**) &d_movesAccepted, numParticlesSample * sizeof(unsigned long int) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorMMC","AllocateIntegratorState","Malloc failed on d_movedsAccepted") );
    
    // Generate seeds and copy
    GenerateHybridTausSeeds( h_hybridTausSeeds, numParticlesSample );
    cudaMemcpy( d_hybridTausSeeds, h_hybridTausSeeds, 2 * numParticlesSample * sizeof(uint4), cudaMemcpyHostToDevice );
    
    if( cudaDeviceSynchronize() != cudaSuccess ) 
      throw( RUMD_Error("IntegratorMMMC","AllocateIntegratorState","CudaMemcpy failed: h_hybridTausSeeds => d_hybridTausSeeds") );
    
    // Currently disabled in MMC.
    std::cout << "[Info] IntegratorMMC, disabling totalEnergy, temperature, kineticEnergy and pressure in energies file" << std::endl;
    S->SetOutputManagerMetaData("energies","totalEnergy",false);
    S->SetOutputManagerMetaData("energies","temperature",false);
    S->SetOutputManagerMetaData("energies","kineticEnergy",false);
    S->SetOutputManagerMetaData("energies","pressure", false);
    
    // Initial potential energy set to max of float to always accept first move
    std::fill_n( h_previousPotentialEnergy, kp.grid.x, std::numeric_limits<float>::max()); 
    cudaMemcpy( d_previousPotentialEnergy, h_previousPotentialEnergy, kp.grid.x * sizeof(float), cudaMemcpyHostToDevice ); 
    // Changed from: cudaMemset( d_previousPotentialEnergy, 0, kp.grid.x * sizeof(float) ); // Just accept first move.
    cudaMemset( d_movesAccepted, 0, numParticlesSample * sizeof(uint) );  
  }
}

// Frees memory dependent on numParticles
void IntegratorMMC::FreeArrays(){
  if(numParticlesSample){
    cudaFreeHost(h_hybridTausSeeds);
    cudaFree(d_hybridTausSeeds);
    cudaFree(d_previous_position);
    cudaFree(d_previous_force);
    cudaFree(d_previous_image);
    cudaFree(d_previous_misc);
    cudaFree(d_summationArray);
    cudaFree(d_previousPotentialEnergy);
  }
}

////////////////////////////////////////////////////////////
// Set Methods
///////////////////////////////////////////////////////////

// Useful for debugging. Should not be used otherwise.
void IntegratorMMC::SetHybridTausSeeds( float taus1, float taus2, float taus3, float lcg ){
  if(!S || !(S->GetNumberOfParticles()))
    throw( RUMD_Error("IntegratorMMC","AllocateIntegratorState","There is no integrator associated with Sample or it has no particles" ) );
  
  unsigned numParticles = S->GetNumberOfParticles();
  
  // For now just use the same state for everything.
  for( unsigned i=0; i < numParticles; i++ ){
    h_hybridTausSeeds[i].x = taus1;
    h_hybridTausSeeds[i].y = taus2;
    h_hybridTausSeeds[i].z = taus3;
    h_hybridTausSeeds[i].w = lcg;
    
    // Same state replicated.
    h_hybridTausSeeds[numParticles+i].x = taus1;
    h_hybridTausSeeds[numParticles+i].y = taus2;
    h_hybridTausSeeds[numParticles+i].z = taus3;
    h_hybridTausSeeds[numParticles+i].w = lcg;
  }
  
  cudaMemcpy( d_hybridTausSeeds, h_hybridTausSeeds, 2 * numParticles * sizeof(uint4), cudaMemcpyHostToDevice );
}     

std::string IntegratorMMC::GetInfoString(unsigned int precision) const {
    std::ostringstream infoStream;
    infoStream << "IntegratorMMC";
    infoStream << "," << std::setprecision(precision) << GetTimeStep();
    infoStream << "," << std::setprecision(precision) << GetTargetTemperature();
    return infoStream.str();
  }

void IntegratorMMC::InitializeFromInfoString(const std::string& infoStr, bool verbose) {
  std::vector<float> parameterList;
  std::string className = ParseInfoString(infoStr, parameterList);
  // check typename, if no match, raise error
  if(className != "IntegratorMMC")
    throw RUMD_Error("IntegratorMMC","InitializeFromInfoString",std::string("Wrong integrator type: ")+className);
  
  if(parameterList.size() != 2)
    throw RUMD_Error("IntegratorMMC","InitializeFromInfoString","Wrong number of parameters in infoStr");

  if(verbose)
    std::cout << "[Info] Initializing IntegratorMMC. time step:" << parameterList[0] << "; target temperature:" << parameterList[1]  << std::endl;
  
  SetTimeStep(parameterList[0]);
  SetTargetTemperature(parameterList[1]);
} 

////////////////////////////////////////////////////////////
// Get Methods
////////////////////////////////////////////////////////////

float IntegratorMMC::GetAcceptanceRate(){
  unsigned long int movesAccepted = 0;
  cudaMemcpy( &movesAccepted, d_movesAccepted, sizeof(unsigned long int), cudaMemcpyDeviceToHost );
  return float(movesAccepted)/float(integrationsPerformed);
}

////////////////////////////////////////////////////////////
// PRNG. Replication is needed due to no device side linker.
////////////////////////////////////////////////////////////

__device__ unsigned TausStepMMC( unsigned &z, int S1, int S2, int S3, unsigned M ){
  unsigned b = ( ( ( z << S1 ) ^ z ) >> S2 );
  return z = ( ( ( z & M ) << S3 ) ^ b );
}

__device__ unsigned LCGStepMMC( unsigned &z, unsigned A, unsigned C ){
  return z = ( A * z + C );
}

// Different prefactor will produce (0, 1].
__device__ float HybridTausMMC( uint4& state ){
  return 2.3283064365387e-10f* (
				TausStepMMC(state.x, 13, 19, 12, 4294967294UL) ^
				TausStepMMC(state.y,  2, 25,  4, 4294967288UL) ^
				TausStepMMC(state.z,  3, 11, 17, 4294967280UL) ^
				LCGStepMMC( state.w,  1664525,   1013904223UL)
				);
}

////////////////////////////////////////////////////////////////////////
// Monte Carlo: [N. Metropolis et al., J. Chem. Phys. 21, 1087 (1953)]
////////////////////////////////////////////////////////////////////////

// Perform a mc trial move. Store old values for rollback.
template <class S> __global__ void integrateMMCAlgorithm( unsigned numParticles, float4* position, float4* previous_position,
							  float4* force, float4* previous_force, float4* image, float4* previous_image,  
							  float4* misc, float4* previous_misc, S* simBox, float* simBoxPointer, 
							  uint4* hybridTausSeed, float step ){
  if( MyGP < numParticles ){ 
    // Load values.
    float4 my_position = position[MyGP]; float4 my_image = image[MyGP];
    
    // Store old values for rollback. Should be made more general!!
    previous_position[MyGP] = my_position; previous_image[MyGP] = my_image; 
    previous_force[MyGP] = force[MyGP]; previous_misc[MyGP] = misc[MyGP];
    
    // Load the simulation box in local memory to avoid bank conflicts.
    float array[simulationBoxSize];
    simBox->loadMemory( array, simBoxPointer );
    
    // State of the HybridTaus generator.
    uint4 my_hybridTausSeed = hybridTausSeed[MyGP];
    
    // Generate 3 uniform between [0;1].
    float3 uniforms = { HybridTausMMC(my_hybridTausSeed), HybridTausMMC(my_hybridTausSeed), HybridTausMMC(my_hybridTausSeed) };
    
    // The random cubic displacement.
    float3 dr = { uniforms.x - 0.5f,  uniforms.y - 0.5f,  uniforms.z - 0.5f };

    // Update coordinates.
    my_position.x += step * dr.x;
    my_position.y += step * dr.y;
    my_position.z += step * dr.z;
    
    float4 local_image = simBox->applyBoundaryCondition( my_position, array );
    
    // Save the result in global memory
    position[MyGP] = my_position; 
    hybridTausSeed[MyGP] = my_hybridTausSeed;

    my_image.x += local_image.x;
    my_image.y += local_image.y;
    my_image.z += local_image.z;
    my_image.w += local_image.w;
    
    image[MyGP] = my_image;
  }
}

////////////////////////////////////////////////////////////////////////
// Try the move or roll it back
////////////////////////////////////////////////////////////////////////

__global__ void calculatePotentialEnergy( unsigned numParticles, float4* force, float* summationArray ){
  if( MyGP < numParticles )
    summationArray[MyGP] = force[MyGP].w;
}

// Decide if accepted.
__global__ void acceptMMCmove( unsigned numParticles, float4* position, float4* previous_position, 
			       float4* force, float4* previous_force, float4* image, float4* previous_image, 
			       float4* misc, float4* previous_misc, float* summationArrayU, 
			       float* previous_U, uint4* hybridTausSeed, unsigned long int* movesAccepted, float invT ){
  
  if( MyGP < numParticles ){
    float U = previous_U[MyB];

    // All N particles have the same state replicated.
    uint4 my_hybridTausSeed = hybridTausSeed[numParticles+MyGP];
    
    // Generate 1 uniform between [0;1].
    float uniform = HybridTausMMC(my_hybridTausSeed);

    // Update PRNG.
    hybridTausSeed[numParticles+MyGP] = my_hybridTausSeed;
    
    __syncthreads();

    // Decide if accepted. Boltzmann probability.
    if( uniform < expf( - invT * ( summationArrayU[0] - U ) ) ){
      previous_U[MyB] = summationArrayU[0]; 
      movesAccepted[MyGP]++;
    }
    else{
      // Rollback.
      position[MyGP] = previous_position[MyGP]; 
      force[MyGP] = previous_force[MyGP]; 
      image[MyGP] = previous_image[MyGP]; 
      misc[MyGP] = previous_misc[MyGP];
    }
  }
}
