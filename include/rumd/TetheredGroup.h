#ifndef TETHEREDGROUP_H
#define TETHEREDGROUP_H

#include <vector>

#include "rumd/Potential.h"


class TetheredGroup : public Potential {

 private:
  
  std::vector<unsigned> solidAtomTypes;
  unsigned numSolidAtoms;

  float kspring;
  unsigned direction;
 
  // GPU specifics
  int num_blocks;
  int threads_per_block;  

  // Wall atoms' zero force lattice sites
  bool initFlag;
  float4 *h_lattice, *d_lattice;
  unsigned *h_index, *d_index;
  float *h_local_energy;
  float *d_local_energy;

  unsigned CountSolidAtoms();
  void Initialize();

  void CopyLatticeToDevice();
  void CopyLatticeFromDevice();
  
 public:

  TetheredGroup(std::vector<unsigned> solidAtomTypes,
		float springConstant);
  
  ~TetheredGroup();

  void Move(float displacement);
  void SetSpringConstant(float ks);
  void SetDirection(unsigned set_dir);
  void CalcF(bool initialize, bool calc_stresses);
  double GetPotentialEnergy();

 bool EnergyIncludedInParticleSum() const { return true; };

 void UpdateAfterSorting(unsigned* old_index,  unsigned int * new_index );

};


#endif // TETHEREDGROUP_H
