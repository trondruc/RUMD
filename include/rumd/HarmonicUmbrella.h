#ifndef HARMONICUMBRELLA_H
#define HARMONICUMBRELLA_H

#include "rumd/PairPotential.h"

class HarmonicUmbrella : public Potential {
public:
  HarmonicUmbrella(PairPotential* set_pairPot, float set_springConst, float set_Q0 );
  virtual ~HarmonicUmbrella();
  
  void Initialize();

  void CalcF(bool initialize, bool calc_stresses);
  
  double GetPotentialEnergy();
  float GetOrderParameter();
  void SetParams(float set_springConst, float set_Q0); 
  bool EnergyIncludedInParticleSum() const { return false; };
private:
  void AllocateQ_Array(unsigned int np);

  PairPotential* pairPot;
  float springConst;
  float Q0;
  float* d_Q;
  unsigned int allocatedSize;
};


#endif // HARMONICUMBRELLA_H
