#ifndef AXILRODTELLERPOTENTIAL_H
#define AXILRODTELLERPOTENTIAL_H

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


class AxilrodTellerPotential : public Potential {

private:
  AxilrodTellerPotential(const AxilrodTellerPotential&);
  AxilrodTellerPotential& operator=(const AxilrodTellerPotential&);
  void AllocatePE_Array(unsigned int nvp);
  
  NeighborList neighborList;
  RectangularSimulationBox* testRSB;
  LeesEdwardsSimulationBox* testLESB;
  float v_AT;
  float Rcut;

  unsigned int allocated_size_pe;    
  float4* d_f_loc;
  float4* d_w_loc;
  float4* h_f_w_loc;
public:
  AxilrodTellerPotential(float v_AT, float Rcut);
  virtual ~AxilrodTellerPotential();
  void Initialize();
  void CalcF(bool initialize, bool calc_stresses);

  void SetVAT(float vat) {this->v_AT = vat;}
  bool EnergyIncludedInParticleSum() const { return true; }
  void UpdateAfterSorting( unsigned* old_index, unsigned* new_index) { neighborList.UpdateAfterSorting(old_index, new_index); }
  void SetNbListSkin( float skin ){ neighborList.SetSkin(skin); }
  void SetNB_Method(const std::string& method) {neighborList.SetNB_Method(method);}
  std::string GetNB_Method() { return neighborList.GetNB_Method(); }
  void SetNbListAllocateMax(bool set_allocate_max) {neighborList.SetAllocateMax(set_allocate_max);} 
  void SetNbMaxNumNbrs(unsigned maxNumNbrs) {neighborList.SetMaxNumNbrs(maxNumNbrs); }
  
  double GetPotentialEnergy();
};


#endif // AXILRODTELLERPOTENTIAL_H
