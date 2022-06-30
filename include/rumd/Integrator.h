#ifndef INTEGRATOR_H
#define INTEGRATOR_H

/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

class ParticleData; class KernelPlan;

#include "rumd/Sample.h"

// The abstract Integrator base class. Inherit this class to build your own integrator.
class Integrator{
 
 private:
  Integrator(const Integrator&); 
  Integrator& operator=(const Integrator&); 
  
 protected:
  Sample* S; ParticleData* P; KernelPlan kp;  
  float timeStep;
  
  // Override this function to allocate numParticles dependent memory.
  virtual void AllocateIntegratorState(){}; 

 public:
  Integrator() : S(0), P(0), kp(), timeStep(1.f){}
  virtual ~Integrator(){};
  
  virtual void SetSample(Sample* S){
    this->S = S;
    kp = S->GetKernelPlan();
    kp.threads.y = 1; 
    P = S->GetParticleData(); 
    AllocateIntegratorState();
  }

  virtual std::string GetInfoString(unsigned int precision) const = 0;
  virtual void InitializeFromInfoString(const std::string& infoStr, bool verbose=true) = 0;
  virtual bool RequiresStress() {return false;}
  
  virtual void Integrate() = 0;
  virtual void UpdateAfterSorting( unsigned int * ) {};

  // Override this function to call functions AFTER the force calculation, 
  // but invisible to the user program.
  virtual void CalculateAfterForce(){};

  // Override this function only if it makes sense
  virtual void SetMomentumToZero() {};

  virtual void SetTimeStep(float dt){ timeStep = dt; };
  virtual float GetTimeStep() const { return timeStep; };
  virtual double GetKineticEnergy(bool copy=true) const;
  virtual double4 GetTotalMomentum(bool copy=true) const;

  // for related quantites that could be written to the energies file
  // this interface is modelled after that for external calculators
  virtual void GetDataInfo(std::map<std::string, bool> &,
			     std::map<std::string, std::string> &){}
  virtual void RemoveDataInfo(std::map<std::string, bool> &,
			      std::map<std::string, std::string> &){}
  virtual void GetDataValues(std::map<std::string, float> &, std::map<std::string, bool> &){}

};

#endif // INTEGRATOR_H
