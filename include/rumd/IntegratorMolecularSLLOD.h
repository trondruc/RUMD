#ifndef INTEGRATORMOLECULARSLLOD_H
#define INTEGRATORMOLECULARSLLOD_H

/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/Integrator.h"

class Sample; class SimulationBox; class MoleculeData;

class IntegratorMolecularSLLOD : public Integrator{
  
 private:
  IntegratorMolecularSLLOD(const IntegratorMolecularSLLOD&);
  IntegratorMolecularSLLOD& operator=(const IntegratorMolecularSLLOD&);


  unsigned int numParticlesSample;
  bool initialized;
  double strainRate;
  unsigned int num_molecules;
  unsigned int n_mol_blocks;
  unsigned int n_mol_threads_per_block;
  
  double4* d_thermostatState; // holds g, f1, f2
  double4* d_thermostatParameters; // Array for summation of kinetic energy.
  double4* h_thermostatParameters; // Array for summation of kinetic energy.
  float4* d_particleMomentum; // Array for summation of total momentum.
  
  void AllocateFromConstructor();
  void FreeArrays();
  void AllocateIntegratorState();
  void Update_factor1_factor2();
  void Update_g_factor();
  
 public:
  IntegratorMolecularSLLOD(float timeStep, double strainRate);
  ~IntegratorMolecularSLLOD();
  

  // Class methods.
  void Integrate();

  // Normally should be called automatically
  void Initialize_g_factor();
  
  // Set methods.
  void SetTimeStep(float dt);
  void SetStrainRate(double set_strain_rate);
  void SetKineticEnergy(float set_kinetic_energy);
  void SetMolecularKineticEnergy(float set_kinetic_energy);
  void SetMomentumToZero();

  // Get methods.
  double GetStrainRate() const { return strainRate; }
  double GetMolecularKineticEnergy() const;

  std::string GetInfoString(unsigned int precision) const;
  void InitializeFromInfoString(const std::string& infoStr, bool verbose=true);

};

#ifndef SWIG

__global__ void integrateMolSLLOD_B1(unsigned int nMolecules,
				     const int1* mlist,
				     unsigned int max_size, 
				     float4* velocity, 
				     float4* force,
				     double4* thermostatParameters,
				     float timeStep,
				     float strainRate,
				     double4 *thermostatState);


__global__ void integrateMolSLLOD_B2(unsigned int nMolecules,
				     const int1* mlist,
				     unsigned int max_size, 
				     float4* velocity, 
				     float4* force,
				     float timeStep,
				     double4* thermostatParameters,
				     double4 *thermostatState);


template <class S> __global__ void integrateMolSLLOD_A_B1(unsigned int nMolecules,
							  const int1* mlist,
							  unsigned int max_size, 
							  float4* position,
							  float4* velocity, 
							  float4* image,
							  S* simulationBox, 
							  float* simulationBoxDevicePointer,
							  double4* thermostatParameters,
							  float timeStep, 
							  float strainRate,
							  double4 *thermostatState,
							  float4* cm);



__global__ void initialize_g_factor_molecules(unsigned int nMolecules, 
					      const int1* mlist, 
					      unsigned int max_size, 
					      float4* v, 
					      double4* thermostatParameters);

__global__ void   update_g_kernel(double timeStep,
				      double strainRate, 
				      double4* thermostatParameters,
				      double4* thermostatState);

__global__ void update_factor1_factor2_kernel(double timeStep,
						  double4* thermostatParameters, 
						  double4* thermostatState);


__global__ void particleKineticEnergies(unsigned numParticles, 
					float4* velocity, 
					double4* thermostatParameters);

__global__ void rescale_velocities( unsigned int numParticles,
				    float4* velocity,
				    float rescale);

__global__ void calculateMomentum( unsigned int numParticles, float4* velocity, float4* particleMomentum );
__global__ void zeroTotalMomentum( unsigned int numParticles, float4* velocity, float4* particleMomentum );

#endif // SWIG
#endif // INTEGRATORMOLECULARSLLOD_H
