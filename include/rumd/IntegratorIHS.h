#ifndef INTEGRATORIHS_H
#define INTEGRATORIHS_H

#include "rumd/Integrator.h"
#include "rumd/IntegratorNVT.h"

//class Sample;
class IntegratorNVT;

class IntegratorIHS : public Integrator{
  
 private:
  IntegratorIHS(const IntegratorIHS&);
  IntegratorIHS& operator=(const IntegratorIHS&);
  
  IntegratorNVT* itg;
  
  float* d_particleDotProduct; // Array for summation of dot product.

  float4* d_previousInherentStateConfiguration;
  float4* d_tempInherentStateConfiguration;

  std::string writeDirectory;
  unsigned int numParticlesSample;
  unsigned int dirCreated;
  unsigned long int numTransitions;
  
  void ZeroParticleVelocity();
  void AllocateIntegratorState();
  void FreeArrays();

 public:
  IntegratorIHS(float timeStep);
  ~IntegratorIHS();
  
  // Class methods.
  void Integrate();
  void DumpInherentStateTransitionConfigurations();

  // Set methods.
  void SetTimeStep(float dt);
  void SetPreviousInherentStateConfiguration();

  void SetSample(Sample* S) { 
    Integrator::SetSample(S);
    itg->SetSample(S);
  }
  
  // Get methods.
  float GetTimeStep() const;
  float GetInherentStateTransitionLength();
  float GetInherentStatePotentialEnergy();
  float GetInherentStateForceSquared();

  double GetKineticEnergy(bool copy=true) const;

  std::string GetInfoString(unsigned int precision) const;
  void InitializeFromInfoString(const std::string& infoStr, bool verbose=true);
};

#ifndef SWIG

__global__ void calculateDotProduct( unsigned int numParticles, float4* velocity, float4* force, float* particleDotProduct );

__global__ void zeroParticleVelocity( unsigned int numParticles, float4* velocity, float* particleDotProduct );

__global__ void sumInherentStateTransitionLength( unsigned int numParticles, float* inherentStateResult, float* partialInherentStateLength );

template <class S> __global__ void calculateInherentStateTransitionLength( unsigned int numParticles, float4* inherentState, 
									   float4* previousInherentState, float* partialInherentStateLength, 
									   S* simBox, float* simBoxPointer );

#endif // SWIG
#endif // INTEGRATORIHS_H
