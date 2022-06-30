#ifndef  IHS_OUTPUTMANAGER_H
#define IHS_OUTPUTMANAGER_H

#include "rumd/LogLinOutputManager.h"

#include <map>
#include <zlib.h>
#include <vector>

class IntegratorIHS;
class ConfigurationWriterReader;
class Potential;

class IHS_OutputManager : public LogLinOutputManager {
public:
  IHS_OutputManager(Sample* S, float timeStep);
  virtual ~IHS_OutputManager();
  void SetMaximumNumberIterations(unsigned int set_max_num_iter) {max_num_iter = set_max_num_iter;}
  void SetForceTolerance(float max_force) {stoppingCriterion = max_force;}
  void SetMetaData(const std::string& key, bool on);
  void SetMetaData(const std::string& key, int value);

  void Write();
   void Terminate() {
     if(gz_ihs_energyFile) gzclose(gz_ihs_energyFile);
     gz_ihs_energyFile = 0;
  }
private:
  Sample* ihsSample;
  IntegratorIHS* itg;
  unsigned int max_num_iter;
  unsigned int num_iter_inner;
  float stoppingCriterion;
  ConfigurationWriterReader* ihsWriter;
  gzFile gz_ihs_energyFile;
  std::map<std::string, bool> options;
  std::vector<Potential*> ihs_potentialList;
};

#endif // IHS_OUTPUTMANAGER_H
