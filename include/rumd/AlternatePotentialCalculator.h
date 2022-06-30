#ifndef  ALTERNATEPOTENTIALCALCULATOR_H
#define ALTERNATEPOTENTIALCALCULATOR_H

#include "rumd/ExternalCalculator.h"

class Sample;
class Potential;

class AlternatePotentialCalculator : public ExternalCalculator {
public:
  AlternatePotentialCalculator(Sample* S, Potential* alt_pot);
  virtual ~AlternatePotentialCalculator();

  void GetDataInfo(std::map<std::string, bool> &active,
		   std::map<std::string, std::string> &columnIDs);
  void RemoveDataInfo(std::map<std::string, bool> &active,
			      std::map<std::string, std::string> &columnIDs);

  void GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active);  

private:
  Sample* mainSample;
  Sample* mySample;
  Potential* myPotential;
};

#endif // ALTERNATEPOTENTIALCALCULATOR_H
