#ifndef BONDPOTENTIAL_H
#define BONDPOTENTIAL_H

#include "rumd/Potential.h"


class PairPotential;

///////////////////////////////////////////////////////
// Abstract BondPotential class
///////////////////////////////////////////////////////

class BondPotential : public Potential{
public:
  BondPotential() : num_threads(64), bond_pot_class(0) {}
  bool EnergyIncludedInParticleSum() const { return true; };
  void SetID_String(const std::string&){ throw RUMD_Error("BondPotential", __func__, "ID string cannot be changed for molecule potentials");}
  void SetParams(unsigned bond_type, float length, float stiffness, bool exclude);
  void Initialize();
  void SetExclusions(PairPotential* non_bond_pot );
  bool IsMolecularPotential() {return true;}
protected:
  unsigned num_threads;
  std::map<unsigned, std::vector<float>> bond_params;
  std::map<unsigned, bool> exclude_bond;
  unsigned bond_pot_class;
  void CopyParamsToGPU();
};

///////////////////////////////////////////////////////
// Specialized bond potentials
///////////////////////////////////////////////////////

// Harmonic
class BondHarmonic : public BondPotential{
private:
  BondHarmonic(const BondHarmonic&);
  BondHarmonic& operator=(const BondHarmonic&);

public:
  BondHarmonic();
  void CalcF(bool initialize, bool calc_stresses);
  double GetPotentialEnergy();
};

// FENE
class BondFENE : public BondPotential{
 private:
  BondFENE(const BondFENE&);
  BondFENE& operator=(const BondFENE&);
  
 public:
  BondFENE();
  void CalcF(bool initialize, bool calc_stresses);
  double GetPotentialEnergy();
};


///////////////////////////////////////////////////////
// Kernel prototypes
///////////////////////////////////////////////////////

template <int stress, int energy, class Simbox>
  __global__ void Harmonic( float4 *r, float4 *f, float4 *w,  float4 *sts,
			    uint2 *blist, uint1 *btlist, float2 *bplist,
			    float *belist, int1 *btlist_int, float *bonds,
			    unsigned num_bonds, Simbox *simbox, float *simboxpointer );

template <int stress, int energy, class Simbox>
  __global__ void FENE( float4 *r, float4 *f, float4 *w,  float4 *sts,
			uint2 *blist, uint1 *btlist, float2 *bplist,
			float *belist, int1 *btlist_int, float *bonds,
			unsigned num_bonds, Simbox *simbox, float *simboxpointer );

/////////////////////////////////////////////////////////

__global__ void BondResetForces( float4 *f, float4 *w );
__device__ inline void BondAtomicFloatAdd(float *address, float val);

#endif // BONDPOTENTIAL_H
