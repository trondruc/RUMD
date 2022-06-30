#include "rumd/LogLinOutputManager.h"

#include <sys/stat.h>
#include <cerrno>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cassert>
#include "rumd/RUMD_Error.h"


LogLinOutputManager::LogLinOutputManager(Sample *S, const std::string &outputDirectory, const std::string& baseFilename) : 
  sample(S),
  blockSize(0),
  user_maxIndex(-1),
  outputDirectory(outputDirectory),  
  baseFilename(baseFilename), 
  backup_ext(".rumd_bak"), 
  maxNumIoBlocks(9999),
  numFilenameDigits(4),
  logLin(),
  active(true),
  duplicateBlockEnds(false),
  verbose(true),
  make_backup(true)
{
}

void LogLinOutputManager::SetLogLinParams(unsigned long int base, unsigned long int maxInterval) {
  logLin.base = base;
  logLin.maxInterval = maxInterval;
}

void LogLinOutputManager::CheckAndInitializeLogLin() {
  if(blockSize == 0)
    throw RUMD_Error("LogLinOutputManager","CheckAndInitializeLogLin","blockSize has not been set");

  if(logLin.base == 0)
    throw RUMD_Error("LogLinOutputManager","CheckAndInitializeLogLin",std::string("Attempt to set base to zero in log-lin with baseFilename ") + baseFilename);

  if(logLin.base > blockSize)
    throw RUMD_Error("LogLinOutputManager","CheckAndInitializeLogLin",std::string("logLin.base is greater than the blockSize in log-lin with baseFilename ") + baseFilename);

  if(logLin.maxInterval % logLin.base != 0) {
    std::ostringstream errorStr;
    errorStr << "maxInterval  " << logLin.maxInterval << " is not a multiple of base " << logLin.base << "for log-lin with baseFilename " << baseFilename;
    throw RUMD_Error("LogLinOutputManager","CheckAndInitializeLogLin", errorStr.str()); 
  }
  if(logLin.maxInterval != 0 && logLin.maxInterval != logLin.base && blockSize % logLin.maxInterval != 0) {
    std::ostringstream errorStr;
    errorStr <<  "maxInterval " << logLin.maxInterval << " does not divide into blockSize " << blockSize << " for log-lin with baseFilename " << baseFilename << ", with non-linear saving";
    throw RUMD_Error("LogLinOutputManager","CheckAndInitializeLogLin",errorStr.str());
  }

  if(blockSize % logLin.base != 0 && logLin.maxInterval != logLin.base) {
    std::ostringstream errorStr;
    errorStr << "blockSize " << blockSize << " is not a multiple of base " << logLin.base << " for log-lin with baseFilename " << baseFilename << ", with non-linear saving";
    throw RUMD_Error("LogLinOutputManager","CheckAndInitializeLogLin", errorStr.str());
  }

  // unless specified by the user,
  // set maxIndex based on blockSize, base and maxInterval
  unsigned int log2_block_size = (unsigned int)(log2(float(blockSize/logLin.base)));
  // In the next line, 1 needs to be a long integer type otherwise have problems when block size exceeds 2**32 times the base 
  unsigned long int rounded_block_size = (1UL << log2_block_size)*logLin.base;
    
  if( (blockSize != rounded_block_size) && 
      (logLin.maxInterval == 0) ) {
    // this applies when you try to do pure logarithmic saving
    std::cerr << "WARNING (LogLinOutputManager::CheckAndInitializeLogLin, baseFilename " << baseFilename 
      << "): Nominal block size " << blockSize 
      << " is not a power of 2 times the base. This means effectively that the base of LogLin will be set to this value, i.e. no logarithmic saving"
      << std::endl;
    logLin.base = blockSize;
    logLin.maxIndex = 0;
    duplicateBlockEnds = false;
  }
  else {

    logLin.maxIndex = log2_block_size+1;

    if(logLin.maxInterval > 0)
      logLin.maxIndex = blockSize/logLin.maxInterval + (unsigned int)(log2(float(logLin.maxInterval/logLin.base)));

    bool unevenLinear = (logLin.base == logLin.maxInterval && blockSize%logLin.base != 0);
    if(unevenLinear && duplicateBlockEnds)
      throw RUMD_Error("LogLinOutputManager","CheckAndInitializeLogLin",std::string("duplicateBlockEnds cannot be true when doing linear saving with an interval which does not divide the block-size (baseFilename:") + baseFilename + ")");

    if(!duplicateBlockEnds && !unevenLinear)
      logLin.maxIndex -= 1;
    // in the case of unevenLinear being true, maxIndex is correctly set
    // only for the first block (case of restarts is handled in Initialize())
  }

  // user-specified max index is applied if non-zero and less than the computed
  // maxIndex
  if(user_maxIndex >= 0 && (unsigned long) user_maxIndex < logLin.maxIndex)
    logLin.maxIndex = user_maxIndex;

  logLin.nextTimeStep = 0;
  logLin.index = 0;
}



void LogLinOutputManager::Initialize(unsigned long int timeStepIndex, bool create_directory) {

  if(active)
    CheckAndInitializeLogLin();

  // Can be responsible for creating the directory etc. even if not active
  
  bool restart = (timeStepIndex > 0);
  // test existence of outputDirectory
  struct stat st;
  int test_dir = stat(outputDirectory.c_str(),&st);
  bool outDirExists = (test_dir == 0);
  bool mayCreateDirectory = create_directory && !restart;
  
  if(outDirExists && mayCreateDirectory) {
    MakeBackupDir();
    CreateDirectory();
  }
  else if(!outDirExists && mayCreateDirectory)
    CreateDirectory();
  else if (!outDirExists && !mayCreateDirectory) 
    throw RUMD_Error("LogLinOutputManager","Initialize",std::string("Output directory does not exist and either this manager does not have the right to create it or this is a restart job"));
  
  // if not active there's no need to do anything else
  if(!active)
    return

  assert(timeStepIndex % blockSize == 0); // don't want to deal with anything else...
  logLin.block = timeStepIndex/blockSize;


  if(logLin.base == logLin.maxInterval) {
    // Need to set nextTimeStep to a non-zero value (in general) when
    // restarting, and have non-evenly dividing linear interval
    logLin.nextTimeStep = (logLin.maxInterval-(timeStepIndex%logLin.maxInterval))%logLin.maxInterval;

    // need to set maxIndex differently when doing uneven linear saving
    if(blockSize % logLin.maxInterval != 0)
      logLin.maxIndex = (blockSize-logLin.nextTimeStep)/logLin.maxInterval;
  }
  
  if(user_maxIndex >= 0 && (unsigned long) user_maxIndex < logLin.maxIndex) {
    // limited linear saving: always starts at beginning of block
    logLin.nextTimeStep = 0;
    logLin.maxIndex = user_maxIndex;
  }
}


void LogLinOutputManager::Update(unsigned long int timeStepIndex) {
  if (active && timeStepIndex%blockSize==logLin.nextTimeStep%blockSize) {

    assert(logLin.index <= logLin.maxIndex);
    if(logLin.block == maxNumIoBlocks)
      throw RUMD_Error("LogLinOutputManager","Update", std::string("Maximum number of blocks reached."));
    Write();

    unsigned long int currentIndex = logLin.index; // want to be able to change index while still checking its previous value
    unsigned long int currentMaxIndex = logLin.maxIndex; // sometimes need to change maxIndex while still checking its previous value
    bool initialize_increments = false;
    const unsigned int log_base = 2; // could imagine this being variable


    if (currentIndex == currentMaxIndex) //  (which may be zero) 
      {
	std::ofstream((outputDirectory+"/LastComplete_" + baseFilename + ".txt").c_str()) << logLin.block << " " << blockSize << std::endl;
	
	if (verbose && logLin.block%10 == 0)
	  std::cout << "Finished writing " << baseFilename << " nr. "  << std::setw(numFilenameDigits) << std::setprecision(numFilenameDigits) << logLin.block << std::endl;
	
	logLin.block++;
	logLin.index = 0;
	// the following gives zero when base and maxInterval evenly divide
	// blockSize
	

	if(blockSize % logLin.base != 0) // (uneven linear saving)
	  {
	    assert(logLin.base == logLin.maxInterval);
	    assert(logLin.increment == logLin.base);
	    logLin.nextTimeStep = (logLin.nextTimeStep+logLin.increment)%blockSize;
	    logLin.maxIndex = (blockSize-logLin.nextTimeStep-1)/logLin.maxInterval; // note subtract one to avoid including the start of the next block
	    if(user_maxIndex >= 0 && (unsigned long) user_maxIndex < logLin.maxIndex) {
	      // limited linear saving: start at beginning of each block
	      logLin.nextTimeStep = 0;
	      logLin.maxIndex = user_maxIndex;
	    }
	  }
	else
	  logLin.nextTimeStep = 0.;

	if(duplicateBlockEnds)  {
	  assert(logLin.nextTimeStep == 0);
	  // we are already at the beginning of the next block
	  Write();
	  initialize_increments = true;
	}
      } // [if currentIndex == currentMaxIndex]
    else if(currentIndex > 0) { 
      logLin.nextTimeStep += logLin.increment;
      // restrict to an increment of maxInterval if the latter is not zero
      logLin.increment *= log_base;
      if(logLin.maxInterval > 0 && logLin.increment > logLin.maxInterval)
	logLin.increment = logLin.maxInterval;
      
      assert(logLin.nextTimeStep <= blockSize);
      // it's allowed to equal blockSize (last configuration in the block)
      logLin.index++;
    } // [if currentIndex > 0 and not equal to currentMaxIndex]
    else // (currentIndex == 0)
      initialize_increments = true; // index was zero but was currentMaxIndex not, so we are not at the end of the block

    if(initialize_increments) {
      logLin.increment = logLin.base;
      logLin.nextTimeStep += logLin.increment;
      assert(logLin.nextTimeStep <= blockSize);
      logLin.index = 1;
    }
  
  } // [if(timeStepIndex%blockSize==logLin.nextTimeStep%blockSize)]
}


void LogLinOutputManager::MakeBackupDir() {
  
  // directory exists, mv it to backup (except if step > 0, ie a restart)
  std::string backupDir = outputDirectory + backup_ext;
  
  // test if back-up exists
  struct stat st;
  int test_dir = stat(backupDir.c_str(),&st);
  if(test_dir == 0) {
    // Remove back-up directory if it exists (use system rather than remove
    // because need recursion)
    std::string rmCmd("rm -rf ");
    rmCmd.append(backupDir);
    system(rmCmd.c_str());
  }

  if(make_backup) {
    std::cout << outputDirectory << " exists; renaming as " << backupDir << std::endl;
    int status = rename(outputDirectory.c_str(), backupDir.c_str());
    if(status == -1)
      throw RUMD_Error("LogLinOutputManager","MakeBackupDir", std::string("Error renaming directory: ") + strerror(errno));
  }
  else {
    // No backup: just remove the output directory
    std::cout << "Removing existing output directory " << outputDirectory << " (back-up has been disabled)  " << std::endl;
    std::string rmCmd("rm -rf ");
    rmCmd.append(outputDirectory);
    system(rmCmd.c_str());
  }
    

}

void LogLinOutputManager::CreateDirectory() {
  if(verbose)
    std::cout << "Creating directory " << outputDirectory << std::endl;
  
  mode_t dir_mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
  int status = mkdir(outputDirectory.c_str(), dir_mode);
  // check if mkdir() failed
  if(status == -1)
    throw RUMD_Error("LogLinOutputManager","CreateDirectory", std::string("Error opening directory ") + outputDirectory + ": " + strerror(errno));
  
}

