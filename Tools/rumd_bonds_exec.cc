#include "rumd_bonds.h"
#include "rumd/RUMD_Error.h"
#include <iostream>


int main(int argc, char *argv[])
{  
  unsigned int num_bins = 5000;   // Number of bins inhistogram

  unsigned int first_block = 0;
  unsigned int last_block = -1;

  float min_dt = 0.0; // Only consider configurations spaced by at least this in time

  bool verbose = false;
  bool writeEachConfiguration = false;

  std::string directory("TrajectoryFiles");
  std::string topologyFile;

  static const char *usage = "Calculate time averaged distribution probabilities.\nUsage: %s [-h] [-n <number of bins>] [-m <minimum time between configurations>] [-f <first_block>] [-l <last_block>] [-d directory] [-t <topology file>] [-e <writeEachConfig>] [-v <verbose>] \n";
  // FIXME fold extended argument description in here
  int opt;
  while ((opt = getopt(argc, argv, "hn:m:f:l:d:t:e:v:")) != -1) {
    switch (opt) {
    case 'n':
      num_bins = atoi(optarg); 
      break;
    case 'm':
      min_dt = strtof(optarg, NULL);
      break;
    case 'f':
      first_block = atoi(optarg);
      break;
    case 'l':
      last_block = atoi(optarg);
      break;
    case 'd':
      directory = std::string(optarg);
      break;
    case 't':
      topologyFile = std::string(optarg);
      break;
    case 'e':
      writeEachConfiguration = (bool) atoi(optarg);
      break;
    case 'v':
      verbose = (bool) atoi(optarg);
      break;
    case 'h':
      fprintf(stdout, usage, argv[0]);
      exit(EXIT_SUCCESS);
    case '?':
      fprintf(stderr, "%s: unknown option: -%c\n", argv[0], optopt);
      fprintf(stderr, usage, argv[0]);
      exit(EXIT_FAILURE);
      break;
    case ':':
      fprintf(stderr, "%s: missing option argument for option: -%c\n", argv[0], optopt);
      fprintf(stderr, usage, argv[0]);
      exit(EXIT_FAILURE);
      break;
    default:
      fprintf(stderr, usage, argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  try{
    rumd_bonds rb;
    rb.SetVerbose(verbose);
    rb.SetDirectory(directory);
    rb.ReadTopology(topologyFile);
    rb.SetWriteEachConfiguration(writeEachConfiguration);
    
    rb.ComputeAll(num_bins, min_dt, first_block, last_block);

    if(!writeEachConfiguration)
      rb.WriteBonds("bonds.dat");
  }

  catch (const RUMD_Error &e) {
    std::cerr << "RUMD_Error thrown from function " 
	      << e.className << "::" << e.methodName << ": " << std::endl
	      << e.errorStr << std::endl;
    return 1;
  }
  return 0;
}
