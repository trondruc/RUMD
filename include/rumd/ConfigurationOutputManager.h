#ifndef CONFIGURATIONOUTPUTMANAGER_H
#define CONFIGURATIONOUTPUTMANAGER_H

#include "rumd/LogLinOutputManager.h"

class ConfigurationWriterReader; 

class ConfigurationOutputManager : public LogLinOutputManager
{
public:
  ConfigurationOutputManager(Sample *S, const std::string& outputDirectory, 
		       const std::string& baseFilename,
		       ConfigurationWriterReader* writer) : 
    LogLinOutputManager(S, outputDirectory, baseFilename), writer(writer) {}
  
  void SetMetaData(const std::string &key, bool on);
  void SetMetaData(const std::string &key, int value);

 private:
  ConfigurationOutputManager(const  ConfigurationOutputManager&); 
  ConfigurationOutputManager& operator=(const  ConfigurationOutputManager&);

protected:
  void Write();
  ConfigurationWriterReader* writer;  
};


#endif // CONFIGURATIONOUTPUTMANAGER_H
