#ifndef ENERGIESMETADATA_H
#define ENERGIESMETADATA_H

#include <string>
#include <map>
#include <vector>
#include "rumd/LogLin.h"
#include "rumd/RUMD_Error.h"

class EnergiesMetaData 
{
public:
  EnergiesMetaData():
    rumd_energies_ioformat(0),
    usingAtomicVirPress(true),
    timeStepIndex(0),
    logLin(),
    bool_options(),
    int_options(),
    fileStr()
  {
    // these quantities are more or less independent of integrator and potential
    bool_options["potentialEnergy"] = true;
    fileStr["potentialEnergy"] = "pe";
    bool_options["kineticEnergy"] = true;
    fileStr["kineticEnergy"] = "ke";
    bool_options["virial"] = true;
    fileStr["virial"] = "W";
    bool_options["totalEnergy"] = true;
    fileStr["totalEnergy"] = "Etot";
    bool_options["temperature"] = true;
    fileStr["temperature"] = "T";
    bool_options["pressure"] = true;
    fileStr["pressure"] = "p";	
    bool_options["volume"] = false;
    fileStr["volume"] = "V";
    bool_options["density"] = false;
    fileStr["density"] = "rho";
    bool_options["enthalpy"] = false;
    fileStr["enthalpy"] = "H";
    bool_options["potentialVirial"] = false;
    fileStr["potentialVirial"] = "pot_W";
    bool_options["stress_xx"] = false;
    fileStr["stress_xx"] = "sxx";
    bool_options["stress_yy"] = false;
    fileStr["stress_yy"] = "syy";
    bool_options["stress_zz"] = false;
    fileStr["stress_zz"] = "szz";
    bool_options["stress_yz"] = false;
    fileStr["stress_yz"] = "syz";
    bool_options["stress_xz"] = false;
    fileStr["stress_xz"] = "sxz";
    bool_options["stress_xy"] = false;
    fileStr["stress_xy"] = "sxy";
    int_options["precision"] = 6;
  }
  void Set(const std::string& key, bool on) {
    if(bool_options.find(key) == bool_options.end())
      throw RUMD_Error("EnergiesMetaData","Set [bool]",std::string("Unrecognized key:") + key);
    bool_options[key] = on;
  }
  void Set(const std::string& key, int value) {
    if(int_options.find(key) == int_options.end())
      throw RUMD_Error("EnergiesMetaData","Set [int]",std::string("Unrecognized key:") + key);
    int_options[key] = value;
  }

  int Get(const std::string& key) {
    if(bool_options.find(key) != bool_options.end())
      return bool_options[key];
    else if(int_options.find(key) != int_options.end())
      return int_options[key];
    throw RUMD_Error("EnergiesMetaData","Get",std::string("Unrecognized key:") + key);
  }
  void ShowOptions() const;
  void ShowFileStrings() const;
  

  std::map<std::string, bool>::const_iterator Start() const {return bool_options.begin();}
  std::map<std::string, bool>::const_iterator End() const {return bool_options.end();}
  const std::string& GetFileString(const std::string& key) {return fileStr[key];}

  unsigned int rumd_energies_ioformat;
  bool usingAtomicVirPress;
  unsigned int timeStepIndex;
  LogLin logLin;
  
  // the function which parses the comment line for configuration files
  void ReadMetaData(char *commentLine, bool verbose, std::vector<std::string> &column_labels);

  std::map<std::string, bool> bool_options;
  std::map<std::string, int> int_options;
  std::map<std::string, std::string> fileStr;
};

#endif // ENERGIESMETADATA_H
