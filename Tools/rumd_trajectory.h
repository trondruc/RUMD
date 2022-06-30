#ifndef RUMD_TRAJECTORY_H
#define RUMD_TRAJECTORY_H

#include "rumd_data.h"
#include "rumd/RUMD_Error.h"

#include <fstream>
#include <iomanip>
#include <sstream>

class rumd_trajectory {
 public:
  rumd_trajectory() : directory("TrajectoryFiles"),
		      basename("trajectory"),
		      verbose(false),
		      numFilenameDigits(4),
		      last_saved_block(0),
		      blockSize(0),
		      numParticles(0),
		      num_types(0),
		      MaxIndex(0),		   
		      particlesPerMol(1),
		      num_moleculetypes(0) {}
  virtual ~rumd_trajectory() {}
  void SetVerbose(bool v) { verbose = v; }
  void SetDirectory(const std::string& directory) {this->directory = directory;}
  unsigned int GetNumTypes() const { return num_types; }
  
protected:
  void ReadLastCompleteFile(std::string test_base) {
    std::string lastComp_filename = directory+"/LastComplete_" + test_base + ".txt";
    std::ifstream lastCompFile(lastComp_filename.c_str());
    if(!lastCompFile.is_open())
      throw RUMD_Error("rumd_rdf","ReadLastCompleteFile",std::string("Could not open " + lastComp_filename));
    
    basename = test_base;
    lastCompFile >> last_saved_block;
    lastCompFile >> blockSize;
  }
  std::string GetConfFilename(unsigned int i) {
     std::ostringstream ConfName;
     ConfName << directory << "/" << basename << std::setfill('0') << std::setw(numFilenameDigits) << std::setprecision(numFilenameDigits) << i << ".xyz.gz";
    return ConfName.str();
  }

  std::string directory;
  std::string basename;
  bool verbose;
  unsigned int numFilenameDigits;
  unsigned int last_saved_block;
  unsigned long blockSize;

  unsigned int numParticles;
  unsigned int num_types;
  unsigned int MaxIndex;
  unsigned int particlesPerMol;
  unsigned int num_moleculetypes;

};

#endif // RUMD_TRAJECTORY_H
