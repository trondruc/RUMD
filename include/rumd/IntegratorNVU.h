#ifndef INTEGRATORNVU_H
#define INTEGRATORNVU_H

/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/Integrator.h"

class IntegratorNVU : public Integrator {
  
 private:
  IntegratorNVU(const IntegratorNVU&);
  IntegratorNVU& operator=(const IntegratorNVU&);

  float* d_previousPotentialEnergy; 
  
  // Arrays for summation.
  float*  d_summationArrayForce; 
  float*  d_summationArrayCorrection;
  float4* d_particleVelocity; 

  float displacementLength;
  float targetPotentialEnergy; // Per Particle.

  unsigned int numParticlesSample;

  void CalculateConstraintForce();
  void CalculateNumericalCorrection();
  void FreeArrays();
  void AllocateIntegratorState();
  
 public:
  IntegratorNVU( float dispLength, float potentialEnergyPerParticle );	
  ~IntegratorNVU();
  
  void Integrate();
  void CalculateAfterForce();

  // Set methods.
  void SetDisplacementLength(float step);
  void SetTargetPotentialEnergy(float U);
  void SetMomentumToZero();

  // Get methods.
  float GetSimulationTimeStepSq() const;
  float GetDisplacementLength() const;
  float GetSimulationDisplacementLength() const;
  float GetTargetPotentialEnergy() const;
  float GetPreviousPotentialEnergy() const;

  std::string GetInfoString(unsigned int precision) const;
  void InitializeFromInfoString(const std::string& infoStr, bool verbose=true);

  void GetDataInfo(std::map<std::string, bool> &active,
		   std::map<std::string, std::string> &columnIDs);
  void RemoveDataInfo(std::map<std::string, bool> &active,
		      std::map<std::string, std::string> &columnIDs);
  void GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active);


};

#ifndef SWIG

template <class S> __global__ void integrateNVUAlgorithm( unsigned numParticles, float4* position, float4* velocity, 
							  float4* force, float4* image, S* simulationBox, 
							  float* simulationBoxDevicePointer, float* constraintForceCorrection, 
							  float* diplacementLengthCorrection, float* previousPotentialEnergy, 
							  float targetPotentialEnergy, float displacementLength, float MeanMass );

template <class S> __global__ void calculateConstraintForce( unsigned numParticles, float4* position, float4* image, float4* velocity, float4* force, 
							     float* constraintForceArray, S* simulationBox, 
							     float* simulationBoxDevicePointer, float meanMass );

__global__ void calculateNumericalCorrection( unsigned numParticles, float4* velocity, float4* force, 
					      float* constraintForce, float* numericalCorrection, float* previousPotentialEnergy, 
					      float targetPotentialEnergy, float meanMass );

__global__ void calculateVelocity( unsigned numParticles, float4* velocity, float4* particleMomentum, float meanMass );
__global__ void zeroSystemVelocity( unsigned numParticles, float4* velocity, float4* particleMomentum );

#endif // SWIG
#endif // INTEGRATORNVU_H
