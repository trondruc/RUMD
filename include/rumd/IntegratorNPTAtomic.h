#ifndef IntegratorNPTAtomic_H
#define IntegratorNPTAtomic_H

/**
  Copyright (C) 2010  Thomas Schrøder

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/Integrator.h"
#include <iostream>
#include <string>

class Sample; class SimulationBox;

/// Integrator for the constant NPT ensemble
/**
 Integrate the equations of motions within the constant pressure ensemble
 using the (alternative) Nose-Hoover equations of motions
 given in Equation 2.9 in [G. J. Martyna, D. J. Tobias and M. L. Klein, J. Chem. Phys.101 (1994), 4177].

 Particle motions:
   \f[  \dot{{r}}_i=\frac{\dot{{p}}_i}{m_i} + \frac{p_\epsilon}{W} {r}_i  \f]
   \f[ \dot{{p}}_i={F}_i - \left(1+\frac{d}{N_f}\right)\frac{p_\epsilon}{W} {p}_i-\frac{p_\eta}{Q} {p}_i \f]

 Motion of barostat state and simulation box volume
   \f[ \dot{p}_\epsilon = V d (P_{\textrm{int}} - P_{\textrm{ext}}) + \frac{d}{N_f} \sum_{i=1}^{N} {\frac{{{p}}_i^2}{m_i}} -\frac{p_\epsilon}{Q} p_\eta \f]
   \f[ \dot V = Vd \frac{p_\epsilon}{W} \f]

 Motion of thermostat state
  \f[ \dot{p}_\eta = \frac{p_\epsilon^2}{W}-(N_f+1) k_B T + \sum_{i=1}^{N} {\frac{{{p}}_i^2}{m_i}}   \f]

 The discrete leap-frog equations are
  \f[ {{p}}_i (t+h/2)=\frac{1}{1+A} \{ h \dot{{F}}_i (t) + {{p}}_i (t-h/2) \left[1-A\right] \} \f]
  \f[ A= \frac{h}{2} \left[\left(1+\frac{d}{N_f}\right) \frac{p_\epsilon (t)}{W} + \frac{p_\eta (t)}{Q} \right] \f]
  \f[ {{r}_i}(t+h) = \frac{h}{m_i} {p}_i (t+h/2) + \left[ 1+ \frac{p_\epsilon (t)}{W} \right] {r}_i (t) \f]
  \f[ p_\eta (t+h) = p_\eta (t) + h \left[2 K (t) + \frac{p_\epsilon^2(t)}{W} - (N_f+1) k_b T\right] \f]
  \f[ p_\epsilon(t+h) = h V d (P_{int}-P_{ext}) + \frac{2 h d}{N_f} K (t) + p_\epsilon \left[1-h \frac{p_\eta(t)}{Q}\right]  \f]

 The function Integrate() make one leap-frog step in the following order
  1. update velocities of particles in integrateNPTAlgorithm() (on GPU)
  2. update positions of particles in integrateNPTAlgorithm() (on GPU)
  3. calculate particle contributions to kinetic energy and virial in integrateNPTAlgorithm() (on GPU)
  4. update the baro- and thermostat states in UpdateState() (on CPU)
  5. scale the box volume in Integrate() (on CPU)

*/
class IntegratorNPTAtomic : public Integrator
{

 private:
  IntegratorNPTAtomic(const IntegratorNPTAtomic&);
  IntegratorNPTAtomic& operator=(const IntegratorNPTAtomic&);

  unsigned thermostatOn; ///< Integer specifying which types to have thermostat on. MAX: 32 types.

  // Dynamic allocation.
  unsigned int numParticlesSample;

  // Thermostat parameters
  double thermostatRelaxationTime;		///< The relaxation time of the thermostat
  double targetTemperature;				///< External temperature
  double temperatureChangePerTimeStep;	///< Change of external temperature per timestep

  double massFactor;					///< a factor = \f$ 1/4 \pi \f$

  // Barostat parameters
  double barostatRelaxationTime;		///< The relaxation time of the barostat
  double targetPressure;				///< The external pressure of the barostat
  double pressureChangePerTimeStep;		///< Change of external pressure per timestep

  // On GPU
  double*  d_thermostatState;			///< p_nu/Q : Thermostatstate devided by thermostat mass (on GPU)
  double*  d_barostatState;				///< p_eps/W : Barostatstate devided by baroostat mass (on GPU)
  float*  d_thermostatKineticEnergy;	///< Array for summation of 2*[kinetic energy] (on GPU)
  float*  d_barostatVirial;			///< Array for summation to get the virial (on GPU)
  float4*  d_particleMomentum;			///< Array for summation of total momentum (on GPU)


  void AllocateFromConstructor();
  void FreeArrays();
  void AllocateIntegratorState();
  void UpdateState();

 public:

  IntegratorNPTAtomic(float timeStep, float targetTemperature, float thermostatRelaxationTime, float targetPressure, float barostatRelaxationTime);
  ~IntegratorNPTAtomic();

  // Class methods.
  void Integrate();

  // Set methods.
  void SetRelaxationTime(double tauTemperature);
  void SetBarostatRelaxationTime(double tauPressure);
  void SetTargetTemperature(double T);
  void SetTargetPressure(double P);
  void SetThermostatState(double Ps);
  void SetBarostatState(double Ps);
  void SetThermostatOn(bool on = false);
  //void SetThermostatOnParticleType(unsigned type, bool on = false);
  void SetMomentumToZero();
  void SetTemperatureChangePerTimeStep(double temperatureChangePerTimeStep);
  void SetPressureChangePerTimeStep(double pressureChangePerTimeStep);

  // Get methods.
  double GetRelaxationTime() const;
  double GetBarostatRelaxationTime() const;
  double GetTargetTemperature() const;
  double GetTargetPressure() const;
  double GetThermostatState() const;
  double GetBarostatState() const;
  double GetKineticEnergy(bool copy=true) const;
  double4 GetTotalMomentum(bool copy=true) const;

  std::string GetInfoString(unsigned int precision) const;
  void InitializeFromInfoString(const std::string& infoStr, bool verbose=true);

};

#ifndef SWIG

template <class S> __global__ void integrateNPTAlgorithm( unsigned numParticles, float4* position, float4* velocity,
							  float4* force, float4* image, float4* virial, S* simulationBox, 
							  float* simulationBoxDevicePointer, 
							  double* thermostatState, double* barostatState,
							  float* thermostatKineticEnergy, float* barostatVirial, 
							  double timeStep, unsigned int degreeOfFreedom );

__global__ void updateNPTState(	float* thermostatKineticEnergy, float* barostatVirial,
				double* thermostatState, double thermostatMass, double targetTemperature,
				double* barostatState, double barostatMass, double targetPressure,
				double volume,
				unsigned int degreesOfFreedom, double timeStep );

__global__ void calculateMomentumNPT( unsigned int numParticles, float4* velocity, float4* particleMomentum );
__global__ void zeroTotalMomentumNPT( unsigned int numParticles, float4* velocity, float4* particleMomentum );

#endif // SWIG
#endif // IntegratorNPTAtomic_H
