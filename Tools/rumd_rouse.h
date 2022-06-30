#include "rumd_trajectory.h"
#include <cassert>

class rumd_rouse : public rumd_trajectory {
public:
  rumd_rouse();
  ~rumd_rouse() { 
    FreeArrays();
  }
  void ComputeAll(unsigned int first_block=0, int last_block=-1, unsigned int particlesPerMol=1);
  void Copy_X0Xt_To_Array(double (*gp_array)[2], unsigned int p);
  void WriteR0Rt(const std::string& filename);
  void WriteR0R0(const std::string& filename);
  void WriteX0Xt(const std::string& filename);
  void WriteX0X0(const std::string& filename);
  void WriteXp0Xq0(const std::string& filename);
  void NormalizeR0Rt();
  void NormalizeX0X0();
  void NormalizeX0Xt();
  unsigned int GetMaxDataIndex() const { return MaxDataIndex; }

  private:
  rumd_rouse(const rumd_rouse&);
  rumd_rouse& operator=(const rumd_rouse&);

  void AllocateArrays(unsigned int set_MaxDataIndex, unsigned int particlesPerMol);
  void FreeArrays();
  void ResetArrays();
  void CalcR0Rt(Conf &C0, Conf &Ct, double *Rt );
  void CalcR0R0(Conf &C0 );
  void CalcX0Xt(Conf &C0, Conf &Ct, double *Xt );
  void CalcX0X0(Conf &C0 );

  // Setup arrays to collect statistics
  unsigned long int *Time;
  unsigned long int *Count;
  double   RgRg;
  double   R0R0;
  double  *R0Rt;
  double **X0Xt;
  double **X0X0;

  unsigned int MaxDataIndex;
  unsigned int num_mol;
  float dt;
};
