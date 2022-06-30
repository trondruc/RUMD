#ifndef INTEGRATORMMC_H
#define INTEGRATORMMC_H

/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/Integrator.h"

class Sample; class SimulationBox;

class IntegratorMMC : public Integrator{
  
 private:
  IntegratorMMC(const IntegratorMMC&);
  IntegratorMMC& operator=(const IntegratorMMC&);

  // Dynamic allocation
  unsigned int numParticlesSample;

  unsigned long int integrationsPerformed;  
  unsigned long int* d_movesAccepted;

  // PRNG seeeds.
  uint4* h_hybridTausSeeds;
  uint4* d_hybridTausSeeds;

  // Storage of particleData items.
  float4* d_previous_position;  
  float4* d_previous_force;
  float4* d_previous_image;
  float4* d_previous_misc;

  float temperature;
  float previousAcceptanceRate;
  float* d_summationArray;
  float* d_previousPotentialEnergy;
  float* h_previousPotentialEnergy;

  void GenerateHybridTausSeeds( uint4* seeds, unsigned numParticles );
  void AllocateIntegratorState();
  void FreeArrays();

 public:
  IntegratorMMC(float dispLength, float targetTemperature);	
  ~IntegratorMMC();
  
  // Class methods.
  void Integrate();

  // Set methods.
  void SetTargetTemperature(float targetTemperature){ temperature = targetTemperature; };
  void SetHybridTausSeeds( float taus1, float taus2, float taus3, float lcg );

  // Get methods.
  float GetTargetTemperature() const { return temperature; };
  float GetAcceptanceRate();

  std::string GetInfoString(unsigned int precision) const;
  void InitializeFromInfoString(const std::string& infoStr, bool verbose=true);
};

#ifndef SWIG

template <class S> __global__ void integrateMMCAlgorithm( unsigned numParticles, float4* position, float4* previous_position,
							  float4* force, float4* previous_force, float4* image, float4* previous_img,
							  float4* misc, float4* previous_misc, S* simBox, float* simBoxPointer, 
							  uint4* seeds, float step );

__global__ void calculatePotentialEnergy( unsigned numParticles, float4* force, float* summationArray );

__global__ void acceptMMCmove( unsigned numParticles, float4* position, float4* previous_position, 
			       float4* force, float4* previous_force, float4* image, float4* previous_image, 
			       float4* misc, float4* previous_misc, float* summationArrayU, 
			       float* previous_U, uint4* seeds, unsigned long int* movesAccepted, float invT );

#endif // SWIG
#endif // INTEGRATORMMC_H
