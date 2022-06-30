#ifndef POTENTIAL_H
#define POTENTIAL_H

#include "rumd/rumd_base.h"
#include "rumd/KernelPlan.h"
#include <string>
#include <vector>
#include <map>
#include "rumd/RUMD_Error.h"

class Sample; class ParticleData; class SimulationBox;

/** Potential energy function
This class determine forces on particles.
*/
class Potential{
 private:
  Potential(const Potential&); 
  Potential& operator=(const Potential&); 
  
 protected:
  Sample* sample;
  KernelPlan kp;
  const ParticleData* particleData;
  SimulationBox* simBox;
  bool verbose;
  std::string ID_String;
  
 public:
  Potential();
  virtual ~Potential(){}

  virtual Potential* Copy(){ throw RUMD_Error("Potential","Copy","Not implemented yet"); }  
  virtual void CalcF(bool initialize, bool calc_stresses) = 0;


  /// tells whether the contribution to the total energy is included in the poarticle energies
  virtual bool EnergyIncludedInParticleSum() const = 0;
  virtual bool IsMolecularPotential() {return false;}
  virtual void UpdateAfterSorting( unsigned*, unsigned* ) {};

  // Set methods.
  virtual void SetSample(Sample* sample);
  /// called by SetSample, must be implemented by different potentials
  virtual void Initialize() = 0;
  virtual void ResetInternalData() {}
  virtual void SetVerbose(bool vb) { verbose = vb; }
  void SetID_String(const std::string& set_ID_String){ ID_String = set_ID_String; }
  
  // Get methods.
  /// Return potential energy
  virtual double GetPotentialEnergy() { return 0.; }
  virtual double GetVirial() { return 0.; }
  std::string GetID_String() const { return ID_String; }
  virtual std::vector<float> GetPotentialParameters(unsigned) const {throw RUMD_Error( "Potential.h", __func__, "Not implemented for this potential");}
  // for related quantites that could be written to the energies file
  // this interface is modelled after that for external calculators
  virtual void GetDataInfo(std::map<std::string, bool> &,
			     std::map<std::string, std::string> &){}
  virtual void RemoveDataInfo(std::map<std::string, bool> &,
			      std::map<std::string, std::string> &){}
  virtual void GetDataValues(std::map<std::string, float> &, std::map<std::string, bool> &){}
};

#endif // POTENTIAL_H
