
#ifndef WALLPOTENTIAL_H
#define WALLPOTENTIAL_H

#include "rumd/Potential.h"

class Sample;

///////////////////////////////////////////////////////
// Abstract WallPotential class
///////////////////////////////////////////////////////

class WallPotential : public Potential{
 public:
  bool EnergyIncludedInParticleSum() const { return true; }; // Not set correctly yet.
  void Initialize() {}
};

///////////////////////////////////////////////////////
// Specialized wall potentials
///////////////////////////////////////////////////////

// 9-3 Lennard-Jones double wall.
class Wall_LJ_9_3 : public WallPotential{
 private:
  Wall_LJ_9_3(const Wall_LJ_9_3&);
  Wall_LJ_9_3& operator=(const Wall_LJ_9_3&);

  // Location of wall one and two in the z-direction.
  float wallOne;
  float wallTwo;
  float sigma1;
  float epsilon1;
  float sigma2;
  float epsilon2;
  float rhoWall;
  float scale;

 public:
  Wall_LJ_9_3();
  void CalcF(bool initialize, bool calc_stresses);
  void SetParams( float wallOne, float wallTwo, float sigma1, float epsilon1, float sigma2, float epsilon2, float rhoWall, float scale );
  double GetPotentialEnergy();
  void WritePotential();
  void ScaleWalls( float scale );
};

///////////////////////////////////////////////////////
// Kernel prototypes
///////////////////////////////////////////////////////

template <int stress, int energy, bool initialize, class Simbox> 
  __global__ void kernelWallLennardJones( float4* position, float4* force, float4* virial, float4* my_stress, 
					  unsigned numParticles, float wallOne, float wallTwo, float sigma1, float epsilon1,
					  float sigma2, float epsilon2, float rhoWall, float scale, Simbox* simbox, float* simBoxPointer );

#endif // WALLPOTENTIAL_H
