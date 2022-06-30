#ifndef IntegratorNPTLangevin_H
#define IntegratorNPTLangevin_H

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
#include <curand.h>

class Sample; class SimulationBox;


/**
   Integrator which samples the constant pressure and temperature ensemble 
   using Langevin equations of motion both for the particles and the volume
   degree of freedom. Based on N. Grønbech-Jensen and Oded Farago, J. Chem.
   Phys. 141, 194108 (2014). It has been reformulated as a leap-frog-type
   algorithm, as outlined in Grønbech-Jensen and Farago, Comput. Phys. Commun. 
   185, 524-527 (2014) [the latter only covers the NVT case though].

   A slight difference from their version is to define the random forces
   as integrals over the offset-timesteps (from t_{n-1/2} to t_{n+1/2})
   instead of taking the mean of two successive non-offset integrals, which is
   what comes out when straightforwardly reformulating to a leap-frog version.
 */

/*
  Remaining tasks

  1. Decide about ResetMomentum - probably remove (and associated kernels, now commented out) DONE
  2. Decide about kinetic energy DONE Need to ADD RANDOM FORCE - DONE
  2.5 Allow turning volume fluctuations off, test KE DONE
  3. Change names of kernels to not clash with NPTAtomic DONE
  4. Check restart DONE
  5. Put in doxygen comments DONE
  6. Test long runs for stability DONE
  7. Test that volume fluctuations are consistent with bulk modulus DONE
  8. Decide whether volume velocity should be reset when changing parameters DONE - the user can do it if they think it's necessary so not automatically
  9. Check performance, use profiler to identify bottlenecks DONE
  10. Check DOFs counted properly DONE kind-of - fixed by hand in user-script
  11. Commit first version to repository DONE
  12. Figure out equations for anisotropic version (one variable direction), and think about what structural changes will be needed. DONE
  13. Implement anisotropic version DONE
  14. Add to documentation: list of integrators in tutorial, and user manual DONE
  15. Remove this list-only after other items are marked DONE
*/


/// Integrator for the constant NPT ensemble using the Langevin equation of motion, based on Grønbech-Jensen and Farago, J. Chem. Phys. 141, 194108 (2014)

class IntegratorNPTLangevin : public Integrator
{
public:
  enum BarostatMode {OFF, ISO, ANISO};
  
  
 private:
  IntegratorNPTLangevin(const IntegratorNPTLangevin&);
  IntegratorNPTLangevin& operator=(const IntegratorNPTLangevin&);

  unsigned boxFlucCoord; // which direction the box fluctuates for anisotropic pressure control
  // Dynamic allocation.
  unsigned int numParticlesSample;

  // Thermostat parameters
  float friction;                          ///< The friction coefficient for particles
  double targetTemperature;				///< External temperature
  double temperatureChangePerTimeStep;	///< Change of external temperature per timestep

  
  // Barostat parameters
  BarostatMode barostatMode;              ///< Can be OFF (no barostatting) or ISO (isotropic barostatting) or ANISO (barostating in one direction given by a coordinate axis)

  double targetPressure;				///< The external pressure of the barostat
  float barostatFriction;                ///< The friction coefficient for the volume degree of freedom
  float barostatMass;                    ///< The inertial parameter for the volume degree of freedom
  
  double pressureChangePerTimeStep;		///< Change of external pressure per timestep

  // For random number generation
  curandGenerator_t curand_generator;
  unsigned num_rand; // number of pseudo-random numbers to be generated
  // float *h_randomForces; // debugging 
  // On GPU
  float *d_randomForces;
  double2 *d_barostatState; // volume ratio and volume velocity  
  float*  d_barostatVirial;			///< Array for summation to get the virial (on GPU)
  float4*  d_particleMomentum;			///< Array for summation of total momentum (on GPU)


  void AllocateFromConstructor();
  void FreeArrays();
  void AllocateIntegratorState();
  double3 UpdateState();
  
public:
  IntegratorNPTLangevin(float timeStep, float targetTemperature, float friction, float targetPressure, float barostatFriction, float barostatMass);
  ~IntegratorNPTLangevin();
  
  bool RequiresStress() {return (barostatMode == ANISO);} // in this case CalcF need to calculate the stress tensor
  
  // Class methods.
  void Integrate();

  // Set methods.
  void SetFriction(double friction) {this->friction = friction;}
  
  void SetTargetTemperature(double T) {targetTemperature = T;}

  void SetBarostatMode(BarostatMode barostatMode) {this->barostatMode = barostatMode;}
  
  void SetTargetPressure(double P) {targetPressure = P;}
  void SetBarostatFriction(float barostatFriction) {this->barostatFriction = barostatFriction;}
  void SetBarostatMass(float barostatMass) {this->barostatMass = barostatMass;}
  void SetBarostatState(double volume_velocity);

  
  void SetTemperatureChangePerTimeStep(double temperatureChangePerTimeStep);
  void SetPressureChangePerTimeStep(double pressureChangePerTimeStep);
  void SetBoxFlucCoord(unsigned boxFlucCoord) {this->boxFlucCoord = boxFlucCoord;}
  
  // Get methods.
  double GetTargetTemperature() const {return targetTemperature;}
  double GetFriction() const {return friction;}

  int GetBarostatMode() const {return barostatMode;}
  //bool GetBarostatOn() const {return barostatOn;}
  double GetTargetPressure() const {return targetPressure;}
  double GetBarostatFriction() const {return barostatFriction;}
  double GetBarostatMass() const {return barostatMass;}
  double2 GetBarostatState() const;
  unsigned GetBoxFlucCoord() const {return boxFlucCoord;}

  double GetKineticEnergy(bool copy=true) const;

  std::string GetInfoString(unsigned int precision) const;
  void InitializeFromInfoString(const std::string& infoStr, bool verbose=true);

};

#ifndef SWIG

template <class S>
__global__ void integrateNPTLangevinAlgorithm( unsigned numParticles,
					       float4* position,
					       float4* velocity,
					       float4* force, float4* image,
					       S* simBox, 
					       float* simBoxPointer, 
					       float* randomForces,
					       double friction,
					       double3 lengthRatio,
					       double timeStep );


__global__ void copyParticleVirial(unsigned numParticles, float4* virial, float* barostatVirial, bool isotropic, unsigned boxFlucCoord);

__global__ void updateNPTLangevinState( float* barostatVirial,
				float barostatFriction, float barostatMass,
				float targetConfPressure, float friction,
				double2* barostatState, double volume,
				float* randomForces, double timeStep );

#endif // SWIG
#endif // IntegratorNPTLangevin_H
