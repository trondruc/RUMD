#include "rumd_trajectory.h"
#include <cassert>

class rumd_vhc : public rumd_trajectory {
public:
  rumd_vhc();
  ~rumd_vhc() {
    FreeArrays();
  }
  void ComputeVHC(unsigned int nBins, int min_dt, unsigned int first_block=0, int last_block=-1);
  unsigned int GetNumBins() const { return num_bins; }
  void WriteVHC(const std::string& filename);
  unsigned int GetMaxDataIndex() const { return MaxDataIndex; }

  const double* GetVHC(int type1, int type2) const
  {
    if(type1 < 0 || (unsigned int) type1 >= num_types || type2 < 0 || (unsigned int) type2 >= num_types)
      throw RUMD_Error("rumd_vhc","Getvhc","Invalid type index");

    return vhc[type1][type2];
  }
  const double* GetRVals() const { return rVals; }

private:
  rumd_vhc(const rumd_vhc&);
  rumd_vhc& operator=(const rumd_vhc&);
  void CalcSingleVHC(const Conf &C0, const Conf &Ct, unsigned int num_types);
  void NormalizeVHC();
  void AllocateArrays(unsigned int nTypes, unsigned int nBins);
  void FreeArrays();
  void ResetArrays();
  
  unsigned long int *Time;
  unsigned int Count;
  double **R;
  int *numParticlesOfType;
  long ***vhcNonNorm;
  double ***vhc;
  unsigned int MaxDataIndex;

  float dt;
  double *rVals;
  unsigned int num_bins;
  float L;
  float dx;
};
