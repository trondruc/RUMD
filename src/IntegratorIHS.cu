/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include <iostream>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>

#include "rumd/Sample.h"
#include "rumd/Potential.h"
#include "rumd/rumd_algorithms.h"
#include "rumd/RUMD_Error.h"
#include "rumd/IntegratorNVT.h"
#include "rumd/IntegratorIHS.h"
#include "rumd/ParseInfoString.h"

////////////////////////////////////////////////////////////
// Constructors 
////////////////////////////////////////////////////////////

IntegratorIHS::IntegratorIHS(float timeStep) :  numTransitions(0), dirCreated(0){
  itg = new IntegratorNVT(timeStep);
  writeDirectory = "./inherentTransitionFiles";
  numParticlesSample = 0;
}

IntegratorIHS::~IntegratorIHS(){
  delete itg;
  FreeArrays();
}

////////////////////////////////////////////////////////////
// Class Methods
////////////////////////////////////////////////////////////

void IntegratorIHS::Integrate(){
  if(!S || !numParticlesSample)
    throw( RUMD_Error("IntegratorIHS","Integrate","Either no sample was set, or it has no particles." ) );
  
  ZeroParticleVelocity(); 
  itg->Integrate();
}

// Private. If (F * v < 0) set v = 0.
void IntegratorIHS::ZeroParticleVelocity(){
  calculateDotProduct<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_v, P->d_f, d_particleDotProduct ); 
  sumIdenticalArrays( d_particleDotProduct, numParticlesSample, 1, 32 );
  zeroParticleVelocity<<< kp.grid, kp.threads>>>( numParticlesSample, P->d_v, d_particleDotProduct );
}

void IntegratorIHS::FreeArrays() {
  if(numParticlesSample){
    cudaFree(d_particleDotProduct);
    cudaFree(d_previousInherentStateConfiguration);
    cudaFree(d_tempInherentStateConfiguration);
  }
}

void IntegratorIHS::AllocateIntegratorState(){
  if(!S || !(S->GetNumberOfParticles())) // Consistency check.
    throw( RUMD_Error("IntegratorIHS","AllocateIntegratorState","Either no sample was set, or it has no particles." ) );
  
  // The dynamics is NVE.
  itg->SetThermostatState(0); 
  
  unsigned int newNumParticlesS = S->GetNumberOfParticles();
  
  if(newNumParticlesS != numParticlesSample){
    FreeArrays();
    numParticlesSample = newNumParticlesS;
    
    // Summation array on GPU.
    if( cudaMalloc( (void**) &d_particleDotProduct, 2 * numParticlesSample * sizeof(float) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorIHS","AllocateIntegratorState","Malloc failed on d_particleDotProduct") );
    
    if( cudaMalloc( (void**) &d_previousInherentStateConfiguration, numParticlesSample * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorIHS","AllocateIntegratorState","Malloc failed on d_previousInherentStateConfiguration") );
    
    if( cudaMalloc( (void**) &d_tempInherentStateConfiguration, numParticlesSample * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorIHS","AllocateIntegratorState","Malloc failed on d_tempInherentStateConfiguration") );
  }
}

void IntegratorIHS::DumpInherentStateTransitionConfigurations(){
  if(!S || !numParticlesSample)
    throw( RUMD_Error("IntegratorIHS","DumpInherentStateTransitionConfigurations","Either no sample was set, or it has no particles." ) );
  
  // Remove and create directory. No backup.
  if(!dirCreated){
    struct stat st;
    int test_dir = stat(writeDirectory.c_str(),&st);
    if(test_dir == 0){
      std::string rmCmd("rm -rf ");
      rmCmd.append(writeDirectory);
      system(rmCmd.c_str());
    }
  
    std::cout << "Creating directory " << writeDirectory << std::endl;
    mode_t dir_mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
    int status = mkdir(writeDirectory.c_str(), dir_mode);

    // Check if mkdir() failed
    if(status == -1)
      throw RUMD_Error("IntegratorIHS","DumpInherentStateTransitionConfigurations", std::string("Error creating directory ") + writeDirectory + ": " + strerror(errno));

    dirCreated = 1;
  }

  std::string fileName = writeDirectory + "/ihs_transition" + toString(numTransitions, std::dec) + ".xyz";
  cudaMemcpy( d_tempInherentStateConfiguration, P->d_r, numParticlesSample * sizeof(float4), cudaMemcpyDeviceToDevice ); // Store.
  cudaMemcpy( P->d_r, d_previousInherentStateConfiguration, numParticlesSample * sizeof(float4), cudaMemcpyDeviceToDevice ); 
  
  S->WriteConf(fileName,"w");
  
  cudaMemcpy( P->d_r, d_tempInherentStateConfiguration, numParticlesSample * sizeof(float4), cudaMemcpyDeviceToDevice ); // Dump previous    
  S->WriteConf(fileName, "a");
  numTransitions++;
}

std::string IntegratorIHS::GetInfoString(unsigned int precision) const {
  std::ostringstream infoStream;
  infoStream << "IntegratorIHS," << std::setprecision(precision) << GetTimeStep();
  return infoStream.str();
}

void IntegratorIHS::InitializeFromInfoString(const std::string& infoStr, bool verbose) {
  std::vector<float> parameterList;
  std::string className = ParseInfoString(infoStr, parameterList);
  // check typename, if no match, raise error
  if(className != "IntegratorIHS")
    throw RUMD_Error("IntegratorIHS","InitializeFromInfoString",std::string("Wrong integrator type: ")+className);
  
  if(parameterList.size() != 1)
    throw RUMD_Error("IntegratorIHS","InitializeFromInfoString","Wrong number of parameters in infoStr");

  if(verbose)
    std::cout << "[Info] Initializing IntegratorIHS. time step:" << parameterList[0] << std::endl;
  SetTimeStep(parameterList[0]);

}

////////////////////////////////////////////////////////////
// Set Methods 
////////////////////////////////////////////////////////////

void IntegratorIHS::SetTimeStep( float dt ){ itg->SetTimeStep(dt); }
void IntegratorIHS::SetPreviousInherentStateConfiguration(){
  if(!S || !numParticlesSample)
    throw( RUMD_Error("IntegratorIHS","SetPreviousInherentStateConfiguration","Either this integrator is not associated with a Sample, or the latter has no particles." ) );
  
  cudaMemcpy( d_previousInherentStateConfiguration, P->d_r, 
	      numParticlesSample * sizeof(float4), cudaMemcpyDeviceToDevice );
}

////////////////////////////////////////////////////////////
// Get Methods 
////////////////////////////////////////////////////////////

float IntegratorIHS::GetTimeStep() const { return itg->GetTimeStep(); }

float IntegratorIHS::GetInherentStateTransitionLength(){
  if(!S || !numParticlesSample)
    throw( RUMD_Error("IntegratorIHS","GetInherentStatePotentialEnergy","Either no sample was set, or it has no particles." ) );
  
  // Virtual functions are not supported in CUDA 2.3
  RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(S->GetSimulationBox());
  
  calculateInherentStateTransitionLength<<< kp.grid, kp.threads >>>( numParticlesSample, d_previousInherentStateConfiguration, 
								     P->d_r, d_particleDotProduct, testRSB, testRSB->GetDevicePointer() ); 
  
  sumIdenticalArrays( &d_particleDotProduct[numParticlesSample], numParticlesSample, 1, 32 );
  
  float* forwardP = &d_particleDotProduct[numParticlesSample];
  float h_inherentStateTransitionLength;
  cudaMemcpy( &h_inherentStateTransitionLength, forwardP, sizeof(float), cudaMemcpyDeviceToHost );
  return sqrtf(h_inherentStateTransitionLength);
}

float IntegratorIHS::GetInherentStatePotentialEnergy(){
  if(!S || !numParticlesSample)
    throw( RUMD_Error("IntegratorIHS","GetInherentStatePotentialEnergy","No sample set, or it has no particles" ) );
  
  return S->GetPotentialEnergy() / double(numParticlesSample);
}

float IntegratorIHS::GetInherentStateForceSquared(){
  if(!S || !numParticlesSample)
    throw( RUMD_Error("IntegratorIHS","GetInherentStateForceSquared","No sample has been set, or it has no particles" ) );

  P->CopyForFromDevice();
  double forcesSquared = 0;
  for(unsigned int i=0; i < numParticlesSample; i++) 
    forcesSquared += P->h_f[i].x * P->h_f[i].x + P->h_f[i].y * P->h_f[i].y + P->h_f[i].z * P->h_f[i].z;
  return forcesSquared;
}

double IntegratorIHS::GetKineticEnergy(bool copy) const { return itg->GetKineticEnergy(copy); }

////////////////////////////////////////////////////////////
// Zero Velocity 
////////////////////////////////////////////////////////////

__global__ void zeroParticleVelocity( unsigned int numParticles, float4* velocity, float* particleDotProduct ){
  if( MyGP < numParticles ){
    if( particleDotProduct[0] < 0.f ){
      float4 my_velocity = velocity[MyGP]; 
      my_velocity.x = 0.f; my_velocity.y = 0.f; my_velocity.z = 0.f;
      velocity[MyGP] = my_velocity;
    }
  }
}

__global__ void calculateDotProduct( unsigned int numParticles, float4* velocity, float4* force, float* particleDotProduct ){
  if( MyGP < numParticles ){
    float4 my_velocity = velocity[MyGP]; float4 my_force = force[MyGP]; 
    particleDotProduct[MyGP] = my_force.x * my_velocity.x + my_force.y * my_velocity.y + my_force.z * my_velocity.z;
  }
  else{
    particleDotProduct[MyGP] = 0.f;
  }
}

////////////////////////////////////////////////////////////
// InherentState Length
////////////////////////////////////////////////////////////

// Needs to be corrected to distance in R space with images.
template <class S> __global__ void calculateInherentStateTransitionLength( unsigned int numParticles, float4* inherentState, float4* previousInherentState, 
									   float* partialInherentStateLength, S* simBox, float* simBoxPointer ){
  
  if( MyGP < numParticles ){
    float4 my_inherentState = inherentState[MyGP]; float4 my_previousInherentState = previousInherentState[MyGP]; 
    partialInherentStateLength[MyGP] = (simBox->calculateDistance( my_inherentState, my_previousInherentState, simBoxPointer )).w;
  }
  else{
    partialInherentStateLength[MyGP] = 0.f;
  }
}
