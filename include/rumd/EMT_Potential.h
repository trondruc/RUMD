#ifndef EMT_POTENTIAL_H
#define EMT_POTENTIAL_H

/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/RUMD_Error.h"
#include "rumd/Potential.h"
#include "rumd/NeighborList.h"
#include "rumd/SimulationBox.h"


class EMT_Potential : public Potential {
public:
  EMT_Potential();
  ~EMT_Potential();
  void SetParams(unsigned type_idx, float E0, float s0, float V0, float eta2, float kappa, float lambda, float n0);
  void CalcF(bool initialize, bool calc_stresses);
  bool EnergyIncludedInParticleSum() const { return true; }

  std::vector<float> GetPotentialParameters(unsigned num_types) const;
  float GetNbListSkin() const { return neighborList.GetSkin(); }
  float GetMaxCutoff() const { return neighborList.GetMaxCutoff(); }
  void SetNbListSkin( float skin ){ neighborList.SetSkin(skin); }
  void SetNB_Method(const std::string& method) {neighborList.SetNB_Method(method);}
  std::string GetNB_Method() { return neighborList.GetNB_Method(); }

  void ResetNeighborList(){ neighborList.ResetNeighborList(); }
  Potential* Copy();
private:
  EMT_Potential(const EMT_Potential&);
  EMT_Potential& operator=(const EMT_Potential&);
  void AllocatePE_Array(unsigned int nvp);

  static const float Beta;
  static const int shell0;
  static const int shell1;

  float cutoff;
  float nb_cut;
  float cutslope;
  unsigned int allocated_num_types;
  unsigned int allocated_energy;
  
  float4* d_f_pe;
  float4* h_f_pe;
  float2* d_dEds_E;

protected:
  NeighborList neighborList;
  RectangularSimulationBox* testRSB;
  LeesEdwardsSimulationBox* testLESB;

  float *d_params_emt;
  size_t shared_size1;
  size_t shared_size2;
  std::map< unsigned, std::vector<float> > emt_params_map;
 
  void CopyParamsToGPU();
  void CalculateCutoff();
  void CalculateGammas(unsigned type_idx);
  void Initialize();
  void AllocateEnergyArrays(unsigned int nvp);
};


#endif // EMT_POTENTIAL_H
