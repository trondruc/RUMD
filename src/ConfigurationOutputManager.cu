#include "rumd/ConfigurationOutputManager.h"
#include "rumd/ConfigurationWriterReader.h"

#include <iostream>


void ConfigurationOutputManager::SetMetaData(const std::string &key, bool on) {
  writer->metaData.Set(key, on);
}

void ConfigurationOutputManager::SetMetaData(const std::string &key, int value) {
  writer->metaData.Set(key, value);
}


void ConfigurationOutputManager::Write() {
  
  std::ostringstream conf_filename;
  conf_filename << outputDirectory << "/" << baseFilename << std::setfill('0') << std::setw(numFilenameDigits) << std::setprecision(numFilenameDigits) << logLin.block << ".xyz";
  
  // set the logLin details in writer->metaData (this would not necessarily make sense for all Configuration writers)  
  
  writer->metaData.SetLogLin(logLin);
  writer->SetVerbose(verbose);

  if(logLin.index > 0) 
    writer->Write(sample, conf_filename.str(), "a");
  else
    writer->Write(sample, conf_filename.str(), "w");
  
}
