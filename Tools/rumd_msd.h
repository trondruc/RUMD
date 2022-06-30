#include "rumd_trajectory.h"
#include <cassert>
#include <iostream>

class rumd_msd : public rumd_trajectory {
public:
  rumd_msd();
  ~rumd_msd() {}
  void ComputeAll(unsigned int first_block=0, int last_block=-1, unsigned int particlesPerMol = 1);

  void SetExtraTimesWithinBlock(bool set_etwb) {extraTimesWithinBlock = set_etwb; }
  void SetNumS4_kValues(unsigned nS4_kValues) {this->nS4_kValues = nS4_kValues;}

  void SetSubtractCM_Drift(bool set_subtract_cm_drift) {this->subtract_cm_drift = set_subtract_cm_drift;}

  void SetAllowTypeChanges(bool set_allow_type_changes) {this->allow_type_changes = set_allow_type_changes;}
  
  void SetQValues(std::vector<double> qvalues) {
    if(verbose) {
      std::cout << "Setting q-values:";
      for(unsigned idx =0; idx< qvalues.size(); idx++)
	std::cout << " " << qvalues[idx];
      std::cout << std::endl;
    }
    this->qvalues = qvalues;
  }

  void Copy_MSD_To_Array(double (*msd_array)[2], unsigned int type);
  void Copy_ISF_To_Array(double (*isf_array)[2], unsigned int type);
  void Copy_VAF_To_Array(double (*msd_array)[2], unsigned int type);
  void Copy_Alpha2_To_Array(double (*alpha2_array)[2], unsigned int type);
  void Fill_Chi4_Array( double (*chi4_array)[2], unsigned int type);
  void Fill_S4_Array( double (*S4_array)[2], unsigned int type, unsigned int k_index);


  void WriteMSD(const std::string& filename);
  void WriteMSD_CM(const std::string& filename);
  void WriteAlpha2(const std::string& filename);
  void WriteISF(const std::string& filename);
  void WriteISF_CM(const std::string& filename);
  void WriteVAF(const std::string& filename);
  void WriteISF_SQ(const std::string& filename);
  void WriteChi4(const std::string& filename);
  void WriteS4(const std::string& filename);


  unsigned int GetNumberOfTimes() const { return Count.size(); }
  unsigned int GetNumberOfS4_kValues() const { return nS4_kValues; }

  private:
  rumd_msd(const rumd_msd&);
  rumd_msd& operator=(const rumd_msd&);

  void ReadQValues();
  void ResetArrays();
  void EnsureMapKeyPresent(unsigned long rel_index);
  void CalcR2(Conf &C0, Conf &C1);
  void CalcCM_displacement(Conf &C0, Conf &C1, double* CM_disp);
  void CalcR2cm(Conf &C0, Conf &C1, unsigned long rel_time_index);
  
  // Setup arrays to collect statistics
  std::map<unsigned long, unsigned long> Count;
  std::map<unsigned long, std::vector<double> > R2;
  std::map<unsigned long, std::vector<double> > R4;
  std::map<unsigned long, std::vector<double> > Fs;
  std::map<unsigned long, std::vector<double> > Fs2;
  std::map<unsigned long, std::vector<double> > VAF;
  std::map<unsigned long, std::vector<std::vector<double> > > S4;
  std::map<unsigned long, std::vector<double> > R2cm;
  std::map<unsigned long, std::vector<double> > R4cm;
  std::map<unsigned long, std::vector<double> > Fscm;

  std::vector<double>Fs_;
  std::vector<double> qvalues;
  std::vector<int> numParticlesOfType;
  unsigned nS4_kValues;

  bool extraTimesWithinBlock;
  bool subtract_cm_drift;
  bool allow_type_changes;
  float dt;
};
