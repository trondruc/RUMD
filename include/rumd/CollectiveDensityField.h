#ifndef COLLECTIVEDENSITYFIELD_H
#define COLLECTIVEDENSITYFIELD_H

#include "rumd/Potential.h"

class RectangularSimulationBox;

class CollectiveDensityField : public Potential {
 public:
  CollectiveDensityField();
  ~CollectiveDensityField();

  void CalcF(bool initialize, bool calc_stresses);
  void SetParams(unsigned int nx, unsigned int ny, unsigned int nz, float kappa, float a);
  void Initialize();
  float  GetCollectiveDensity();
  float  GetCollectiveDensity(float* rho_k_re, float* rho_k_im);
  double GetPotentialEnergy();
  bool EnergyIncludedInParticleSum() const { return false; }
  void GetDataInfo(std::map<std::string, bool> &active, std::map<std::string, std::string> &columnIDs);
  void RemoveDataInfo(std::map<std::string, bool> &active, std::map<std::string, std::string> &columnIDs);
  void GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active);

private:
  unsigned int nx;
  unsigned int ny;
  unsigned int nz;
  float kappa;
  float a;
  float2* d_cos_sin;
  float3* d_rho_k_pref;
  RectangularSimulationBox* testRSB;
  unsigned int nAllocatedParticles;
};


#endif // COLLECTIVEDENSITYFIELD_H
