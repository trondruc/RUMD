#include "rumd_trajectory.h"

#include <cassert>


struct float4 { float x; float y; float z; float w; };  

class rumd_rdf : public rumd_trajectory {

 public:
  rumd_rdf();
  ~rumd_rdf(){ FreeArrays(); }


  void ComputeAll(int nBins, float min_dt, unsigned int first_block=0, int last_block=-1, unsigned particlesPerMol=1);
  void WriteRDF(const std::string& filename);
  void WriteRDFinter(const std::string& filename);
  void WriteRDFintra(const std::string& filename);
  void WriteRDF_CM(const std::string& filename);
  void WriteSDF(const std::string& filename);
  int GetNumTypes() const {return (int)num_types;}
  int GetNumBins() const {return (int)num_bins;}
  const double* GetRDF(int type1, int type2) const 
  {
    if(type1 < 0 || (unsigned int) type1 >= num_types || type2 < 0 || (unsigned int) type2 >= num_types)
      throw RUMD_Error("rumd_rdf","GetRDF","Invalid type index");

    return rdf[type1][type2];
  }
  const double* GetSDF(int type1, int coord) const {return sdf[type1][coord];}
  const double* GetRVals() const { return rVals; }
  
  void SetWriteEachConfiguration(bool wec) {writeEachConfiguration = wec;}

 private:
  void AllocateArrays(unsigned int nTypes, unsigned int nBins);
  void AllocateArraysCM(unsigned int particlesPerMol);
  void FreeArrays();
  void FreeArraysMolecule();
  void ResetArrays();
  void CalcSingleRDF(Conf &C0);
  void CalcSingleRDFCM(Conf &C0);
  void Normalize();
  float4 CalculateCM( Conf &C0, int index );
  //unsigned long int GetLastIndex(Conf &C0);

  rumd_rdf(const rumd_rdf&);
  rumd_rdf& operator=(const rumd_rdf&);

  unsigned int Count;
  bool writeEachConfiguration;

  int *numParticlesOfType;
  int *numMoleculesOfType;

  long ***rdfNonNorm;
  long ***rdfInterNonNorm;
  long ***rdfIntraNonNorm;
  long ***rdfCMNonNorm;
  long ***sdfNonNorm;
  
  double ***rdf;
  double ***rdfInter;
  double ***rdfIntra;
  double ***rdfCM;
  double ***sdf;
  double *rVals;

  unsigned int num_bins;
  
  float Lx;
  float Ly;
  float Lz;
  float b_width;
  float dt;
};
