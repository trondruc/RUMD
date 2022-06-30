#ifndef EXTERNALCALCULATOR_H
#define EXTERNALCALCULATOR_H

#include <string>
#include <map>

class ExternalCalculator {
public:
  ExternalCalculator() {}
  virtual ~ExternalCalculator() {}
  virtual void GetDataInfo(std::map<std::string, bool> &active,
			   std::map<std::string, std::string> &columnIDs) = 0;
  virtual void RemoveDataInfo(std::map<std::string, bool> &active,
			      std::map<std::string, std::string> &columnIDs) = 0;
  virtual void GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active) = 0;
private:
  ExternalCalculator(const ExternalCalculator&); 
  ExternalCalculator& operator=(const ExternalCalculator&); 
};

#endif // EXTERNALCALCULATOR_H
