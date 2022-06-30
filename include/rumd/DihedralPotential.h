
#ifndef DIHEDRALPOTENTIAL_H
#define DIHEDRALPOTENTIAL_H

#include "rumd/Potential.h"

class PairPotential;

#define DIHEDRAL_EPS 1.0e-3


///////////////////////////////////////////////////////
// Abstract Potential class
///////////////////////////////////////////////////////

class DihedralPotential : public Potential{
 public:
  DihedralPotential() : num_threads(16), num_dihedral_params(0), exclude_dihedral(true) {}
  bool EnergyIncludedInParticleSum() const { return true; };
  void Initialize();
  void SetID_String(const std::string&){ throw RUMD_Error("DihedralPotential", __func__, "ID string cannot be changed for molecule potentials");}
  void SetParams(unsigned dihedral_type, std::vector<float> coeffs);
  void SetExclude(bool exclude) {exclude_dihedral = exclude;}
  void SetExclusions(PairPotential* non_bond_pot);
  bool IsMolecularPotential() {return true;}
protected:
  unsigned num_threads;
  std::map<unsigned, std::vector<float>> dihedral_params;
  unsigned num_dihedral_params;
  bool exclude_dihedral; // unlike in BondPotential, not separated by dihedral_type
  void CopyParamsToGPU();
};


// Ryckaert-Bellemans potential
class DihedralRyckaert : public DihedralPotential{
 private:
  DihedralRyckaert(const DihedralRyckaert&);
  DihedralRyckaert& operator=(const DihedralRyckaert&);
  
 public:
  DihedralRyckaert();
  void CalcF(bool initialize, bool calc_stresses);
  double GetPotentialEnergy();
};

// Periodic dihedral potential
class PeriodicDihedral : public DihedralPotential{
 private:
  PeriodicDihedral(const PeriodicDihedral&);
  PeriodicDihedral& operator=(const PeriodicDihedral&);

 public:
  PeriodicDihedral();
  void CalcF(bool initialize, bool calc_stresses);
  double GetPotentialEnergy();
};


///////////////////////////////////////////////////////
// Kernel prototypes
///////////////////////////////////////////////////////

template <int energy, class Simbox> 
__global__ void Ryckaert( float4 *r, float4 *f, 
			  uint4 *dlist, uint1 *dtype, float *plist, 
			  float *d_epot_dihedral, float *d_dihedrals,
			  unsigned num_dihedrals, Simbox *simbox, float *simboxpointer  );

template <int energy, class Simbox>
__global__ void PeriodicDih( float4 *r, float4 *f,
                          uint4 *dlist, uint1 *dtype, float *plist,
                          float *d_epot_dihedral, float *d_dihedrals,
                          unsigned num_dihedrals, Simbox *simbox, float *simboxpointer  );

/////////////////////////////////////////////////////////

__global__ void DihedralResetForces( float4 *f, float4 *w );
__device__ inline void DihedralAtomicFloatAdd(float *address, float val);


#endif
