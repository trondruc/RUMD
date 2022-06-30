#include "rumd_trajectory.h"

#include <cassert>
#include <vector>

struct float4 { float x; float y; float z; float w; };
struct bond { unsigned type; unsigned a; unsigned b;};

class rumd_bonds : public rumd_trajectory {

 public:
  rumd_bonds();
  ~rumd_bonds(){ FreeArrays(); }

  void SetWriteEachConfiguration(bool wec) {writeEachConfiguration = wec;}  
  void ReadTopology(std::string topFilename);
  void ComputeAll(int nBins, float min_dt, unsigned first_block=0, int last_block=-1);
  void WriteBonds(const std::string& filename);

  int GetNumBondTypes() const {return (int)numBondTypes;}
  int GetNumBins()      const {return numBins;}
  const double* GetRVals() const { return rVals; }
  const double* GetBondDistribution(unsigned bondType) const {
    if(bondType >= num_types)
      throw RUMD_Error("rumd_bonds","GetBondDistribution","Invalid type index");
    return hist[bondType];
  }

 private:

  void ReadTopology();
  void AllocateArrays(unsigned nBins);
  void FreeArrays();
  void ResetArrays();
  void CalcSingleHistogram(Conf &C0);
  void Normalize();
  unsigned long GetLastIndex(Conf &C0);

  rumd_bonds(const rumd_bonds&);
  rumd_bonds& operator=(const rumd_bonds&);

  bool writeEachConfiguration;
  unsigned Count;
  unsigned numBins;
  unsigned numBondTypes;
  float L;
  float dx;
  float dt;

  std::vector<unsigned> numBondsOfType;
  std::vector<bond> bondList;

  long   **histNonNorm;
  double **hist;
  double *rVals;
};
