#ifndef  HYPERVIRIALCALCULATOR_H
#define HYPERVIRIALCALCULATOR_H

#include "rumd/ExternalCalculator.h"

class Sample;
class Potential;
#include <vector>

class HypervirialCalculator : public ExternalCalculator {
public:
  HypervirialCalculator(Sample* S, double set_delta_ln_rho);
  virtual ~HypervirialCalculator();

  void GetDataInfo(std::map<std::string, bool> &active,
		   std::map<std::string, std::string> &columnIDs);
  void RemoveDataInfo(std::map<std::string, bool> &active,
			      std::map<std::string, std::string> &columnIDs);

  void GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active);  
  void SetDeltaLogRho(double set_delta_ln_rho) { delta_ln_rho = set_delta_ln_rho; }
private:
  Sample* mainSample;
  Sample* mySample;
  double delta_ln_rho;
  std::vector<Potential*> my_potentialList;
};

#endif // HYPERVIRIALCALCULATOR_H
