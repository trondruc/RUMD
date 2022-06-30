
#ifndef ANGLEPOTENTIAL_H
#define ANGLEPOTENTIAL_H

#include "rumd/Potential.h"

class PairPotential;

///////////////////////////////////////////////////////
// Abstract BondPotential class
///////////////////////////////////////////////////////

class AnglePotential : public Potential{
 public:
  AnglePotential() : num_threads(32), exclude_angle(true) {}
  bool EnergyIncludedInParticleSum() const { return true; };
  void Initialize();
  void SetID_String(const std::string&){ throw RUMD_Error("AnglePotential", __func__, "ID string cannot be changed for molecule potentials");}
  void SetParams(unsigned angle_type, float theta0, float ktheta);
  void SetExclude(bool exclude) {exclude_angle = exclude;}
  void SetExclusions(PairPotential* non_bond_pot);
  bool IsMolecularPotential() {return true;}
protected:
  unsigned num_threads;
  std::map<unsigned, std::vector<float>> angle_params;
  bool exclude_angle; // unlike in BondPotential, not separated by angle_type
  void CopyParamsToGPU();
};


// Cosine squared potential
class AngleCosSq : public AnglePotential{
 private:
  AngleCosSq(const AngleCosSq&);
  AngleCosSq& operator=(const AngleCosSq&);

 public:
  AngleCosSq();
  void CalcF(bool initialize, bool calc_stresses);
  double GetPotentialEnergy();
  void SetParams(unsigned angle_type, float theta0, float ktheta);
};

// Angle squared potential
class AngleSq : public AnglePotential{
 private:
  AngleSq(const AngleSq&);
  AngleSq& operator=(const AngleSq&);

 public:
  AngleSq();
  void CalcF(bool initialize, bool calc_stresses);
  double GetPotentialEnergy();
};



///////////////////////////////////////////////////////
// Kernel prototypes
///////////////////////////////////////////////////////

template <int energy, class Simbox> 
__global__ void CosSq( float4 *r, float4 *f,
		       uint4 *alist, float2 *parameter,
		       float *d_epot_angle, float *d_angles,
		       unsigned num_angles, Simbox *simbox, float *simboxpointer );

template <int energy, class Simbox>
__global__ void SquaredAngle( float4 *r, float4 *f,
                       uint4 *alist, float2 *parameter,
                       float *d_epot_angle, float *d_angles,
                       unsigned num_angles, Simbox *simbox, float *simboxpointer );


//////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////
__global__ void AngleResetForces( float4 *f, float4 *w );
__device__ inline void AngleAtomicFloatAdd(float *address, float val);

#endif
