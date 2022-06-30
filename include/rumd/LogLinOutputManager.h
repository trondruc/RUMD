#ifndef LOGLINOUTPUTMANAGER_H
#define LOGLINOUTPUTMANAGER_H


#include "rumd/LogLin.h"

#include <string>
#include <sstream>
#include <iomanip>

class Sample;

class LogLinOutputManager
{
public:

  LogLinOutputManager(Sample *S, const std::string &outputDirectory, const std::string& baseFilename);
  
  virtual ~LogLinOutputManager() {}

  void SetVerbose(bool vb) { verbose = vb; }
  void SetBlockSize(unsigned long int blockSize) {this->blockSize = blockSize;}
  void SetUserMaxIndex(long int user_maxIndex) {this->user_maxIndex = user_maxIndex;}
  void SetOutputDirectory(const std::string& outputDirectory) {this->outputDirectory = outputDirectory;}
  void SetActive(bool set_active) { this->active = set_active; }
  void SetDuplicateBlockEnds(bool dbe) { duplicateBlockEnds = dbe; }
  void EnableBackup(bool make_backup) { this->make_backup = make_backup; }

  virtual void SetLogLinParams(unsigned long int base, unsigned long int maxInterval);
  virtual void SetMetaData(const std::string& key, bool on)=0;
  virtual void SetMetaData(const std::string& key, int value)=0;
  virtual void Initialize(unsigned long int step, bool create_directory=false);


 
  virtual void Update(unsigned long int timeStep);

  virtual void Terminate() {}

private:
  LogLinOutputManager(const LogLinOutputManager&); // no default copyconstructor
  LogLinOutputManager& operator=(const LogLinOutputManager&); // no default assignment

protected:
  virtual void Write()=0;
  void MakeBackupDir();
  void CreateDirectory();
  void CheckAndInitializeLogLin();
  Sample *sample;
  unsigned long int blockSize;
  long int user_maxIndex;
  std::string outputDirectory;
  std::string baseFilename;
  std::string backup_ext;
  unsigned int maxNumIoBlocks;
  unsigned int numFilenameDigits;
  LogLin logLin;
  bool active;
  bool duplicateBlockEnds;
  bool verbose;
  bool make_backup;
};

#endif // LOGLINOUTPUTMANAGER_H
