#ifndef  SQUAREDDISPLACEMENTCALCULATOR_H
#define SQUAREDDISPLACEMENTCALCULATOR_H

#include "rumd/ExternalCalculator.h"

class Sample;
class Potential;
#include <vector>

class SquaredDisplacementCalculator : public ExternalCalculator {
public:
  SquaredDisplacementCalculator(Sample* S, Sample* S_ref=0);
  virtual ~SquaredDisplacementCalculator();

  void GetDataInfo(std::map<std::string, bool> &active,
		   std::map<std::string, std::string> &columnIDs);
  void RemoveDataInfo(std::map<std::string, bool> &active,
			      std::map<std::string, std::string> &columnIDs);

  void GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active);  
private:
  void Allocate(unsigned nParticles);
  void Free();
  Sample* mainSample;


  float4* d_r_unsrt;
  float4* d_r_ref_unsrt;
  float4* d_im_unsrt;
  float4* d_im_ref_unsrt;
  
  
  int3* d_sum_sq_images;
  double4* d_sum_dbl_comp;
  unsigned num_allocated;
};

__global__ void squared_displacement_kernel(unsigned nParticles, float4* im1, float4* pos1, float4* im2, float4* pos2, int3* sum_sq_images, double4* sum_dbl_comp);

#endif // SQUAREDDISPLACEMENTCALCULATOR_H
