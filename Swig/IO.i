%{
#include "rumd/ConfigurationMetaData.h"
#include "rumd/ConfigurationWriterReader.h"
#include "rumd/IHS_OutputManager.h"
  %}


%rename(assign) *::operator=; // to remove warnings about operator=
%include "rumd/LogLin.h"


// turn off keyword arguments for overloaded functions to avoid warnings
%feature("kwargs", "0") ConfigurationMetaData::Set;
%include "rumd/ConfigurationMetaData.h"

%nodefaultctor LogLinOutputManager;

class LogLinOutputManager {};

class IHS_OutputManager : public LogLinOutputManager {
 public:
  IHS_OutputManager(Sample* sample, float timeStep);
  void SetMaximumNumberIterations(unsigned int set_max_num_iter);
  void SetForceTolerance(float max_force);
};
