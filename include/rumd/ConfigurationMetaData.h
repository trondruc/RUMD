#ifndef CONFIGURATIONMETADATA_H
#define CONFIGURATIONMETADATA_H

#include <string>
#include <map>
#include <vector>

#include "rumd/rumd_base.h"

#include "rumd/LogLin.h"
#include "rumd/RUMD_Error.h"

// this class handles two kinds of metaData for configuration files
// (1) specific data in the comment line, such as the box-length (this is
// specfically needed for reading, not so much for writing since it is available
// to the sample class anyway)
// (2) which per-particle data is in the file (specified as the "columns" entry 
// in the comment line)

class ConfigurationMetaData
{
  friend class ConfigurationWriterReader;
public:
  ConfigurationMetaData():
    rumd_conf_ioformat(0), // refers to a file, so could be different from the current rumd_conf_ioformat
    timeStepIndex(0),
    Ps(0.),
    Pv(0.),
    dt(0.),
    massOfType(),
    blockSize(0),
    logLin(),
    bool_options(),
    int_options(),
    simulationBoxInfoStr(""),
    integratorInfoStr(""),
    found_integrator(false)
  {
    bool_options["images"] = false;
    bool_options["velocities"] = false;
    bool_options["forces"] = false;
    bool_options["pot_energies"] = false;
    bool_options["virials"] = false;
    bool_options["logLin"] = false;
    int_options["precision"] = 6;
  }
  // the function which parses the comment line for configuration files
  void ReadMetaData(char *commentLine, bool verbose);
  void SetLogLin(const LogLin& set_logLin) {this->logLin = set_logLin;}

  void Set(const std::string& key, bool val) {
    if(bool_options.find(key) == bool_options.end())
      throw RUMD_Error("ConfigurationMetaData","Set [bool]",std::string("Unrecognized key:") + key);
    bool_options[key] = val;
  }
  void Set(const std::string& key, int value) {
    if(int_options.find(key) == int_options.end())
      throw RUMD_Error("ConfigurationMetaData","Set [int]",std::string("Unrecognized key:") + key);
    int_options[key] = value;
  }
  int Get(const std::string& key) {
    if(bool_options.find(key) != bool_options.end()) {
      return bool_options.find(key)->second;
    }
    else if(int_options.find(key) != int_options.end()) {
      return int_options.find(key)->second;
    }
      throw RUMD_Error("EnergiesMetaData","Set",std::string("Unrecognized key:") + key);
  }
  
  bool IntegratorFound() const { return found_integrator; }    
  std::string GetIntegratorInfoStr() const { return integratorInfoStr; }
  std::string GetSimulationBoxInfoStr() const { return simulationBoxInfoStr; }

  unsigned int GetIO_Format() const { return rumd_conf_ioformat; }
  
  float GetThermostatPs() const { return Ps; }
  unsigned int GetNumTypes() const {return massOfType.size(); }
  unsigned long int GetTimeStepIndex() const { return timeStepIndex; }
  float GetDt() const { return dt; }
  const LogLin& GetLogLin() const { return logLin; }
  float GetMassOfType(unsigned typeIdx) const {
    if(typeIdx >= massOfType.size())
      throw RUMD_Error("ConfigurationMetaData","GetMassOfType","type index too large");
    return massOfType[typeIdx];
}

private:
  unsigned int rumd_conf_ioformat;
  unsigned long int timeStepIndex;
  float Ps;
  float Pv;
  float dt;
  std::vector<float> massOfType;
  unsigned int blockSize;
  LogLin logLin;

  std::map<std::string, bool> bool_options;
  std::map<std::string, int> int_options;

  // info strings for simulation box and integrator (ioformat 2)
  std::string simulationBoxInfoStr;
  std::string integratorInfoStr;
  bool found_integrator;
};


#endif // CONFIGURATIONMETADATA_H
