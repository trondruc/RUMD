#ifndef INTEGRATORNVT_H
#define INTEGRATORNVT_H

/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/Integrator.h"

class Sample; class SimulationBox;

class IntegratorNVT : public Integrator{
  
 private:
  IntegratorNVT(const IntegratorNVT&);
  IntegratorNVT& operator=(const IntegratorNVT&);

  unsigned thermostatOn; // Integer specifying which types to have thermostat on. MAX: 32 types.
  
  // Dynamic allocation.
  unsigned int numParticlesSample;  
  
  float thermostatRelaxationTime; // The user chosen relaxation time.
  double targetTemperature; 
  double temperatureChangePerTimeStep;
  float massFactor;

  double*  d_thermostatState; // GPU.
  float*  d_thermostatKineticEnergy; // Array for summation of kinetic energy.
  float4* d_particleMomentum; // Array for summation of total momentum.
  
  void AllocateFromConstructor();
  void FreeArrays();
  void AllocateIntegratorState();
  void UpdateThermostatState();
  
 public:
  IntegratorNVT(float timeStep);
  IntegratorNVT(float timeStep, double targetTemperature, double thermostatRelaxationTime=0.2f);
  ~IntegratorNVT();
  
  // Class methods.
  void Integrate();
  
  // Set methods.
  void SetRelaxationTime(float tau);
  void SetTargetTemperature(double T, bool reset=true);
  void SetThermostatState(double Ps);
  void SetThermostatOn(bool on = false);
  void SetThermostatOnParticleType(unsigned type, bool on = false);
  void SetMomentumToZero();
  void SetTemperatureChangePerTimeStep(double temperatureChangePerTimeStep);
  // Get methods.
  float GetRelaxationTime() const;
  double GetTargetTemperature() const;
  double GetThermostatState() const;
  double GetKineticEnergy(bool copy=true) const;
  float GetThermostatKineticEnergy() const;
  
  std::string GetInfoString(unsigned int precision) const;
  void InitializeFromInfoString(const std::string& infoStr, bool verbose=true);

  void GetDataInfo(std::map<std::string, bool> &active,
		   std::map<std::string, std::string> &columnIDs);
  void RemoveDataInfo(std::map<std::string, bool> &active,
		      std::map<std::string, std::string> &columnIDs);
  void GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active);


};

#ifndef SWIG

template <class S> __global__ void integrateNVTAlgorithm( unsigned numParticles, float4* position, float4* velocity, 
							  float4* force, float4* image, S* simulationBox, 
							  float* simulationBoxDevicePointer, double* thermostatState, 
							  float* thermostatKineticEnergy, float timeStep, unsigned thermoStatOn );

__global__ void updateNVTThermostatState( float* thermostatKineticEnergy, double* thermostatState, 
					  float thermostatMass, float targetTemperature, 
					  int degreesOfFreedom, float timeStep );

__global__ void calculateMomentum( unsigned int numParticles, float4* velocity, float4* particleMomentum );
__global__ void zeroTotalMomentum( unsigned int numParticles, float4* velocity, float4* particleMomentum );

#endif // SWIG
#endif // INTEGRATORNVT_H
