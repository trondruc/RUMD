#include "rumd_vhc.h"
#include "rumd/RUMD_Error.h"
#include <iostream>

int main(int argc, char *argv[])
{
  unsigned int first_block = 0;
  unsigned int last_block = -1;
  bool verbose = false;
  unsigned int num_bins = 1000;   // Number of bins for the correlation function
  int min_dt = 0; // Only consider configurations spaced by at least this in time
  std::string directory("TrajectoryFiles");

  static const char *usage = "Usage: rumd_vhc [-n <number of bins>] [-m <time between configurations>] [-f <first_block>] [-l <last_block>] [-d <directory>] [-v <verbose>] [-h]\n";
  int opt;
  while ((opt = getopt(argc, argv, "hn:m:f:l:vd:e")) != -1) {
    switch (opt) {
    case 'n':
      num_bins = atoi(optarg);
      break;
    case 'm':
      min_dt = atoi(optarg);
      break;
    case 'f':
      first_block = atoi(optarg);
      break;
    case 'l':
      last_block = atoi(optarg);
      break;
    case 'v':
      verbose = true;
      break;
    case 'd':
      directory = std::string(optarg);
      break;
    case 'h':
      std::cout << usage << std::endl;
      exit(EXIT_SUCCESS);
    case '?':
      std::cerr << "rumd_rdf: unknown option: -" << optopt << std::endl;
      std::cerr << usage << std::endl;
      exit(EXIT_FAILURE);
      break;
    case ':':
      std::cerr << "rumd_rdf: missing option argument for option: -" << optopt << std::endl;
      std::cerr << usage << std::endl;
      exit(EXIT_FAILURE);
      break;
    default:
      std::cerr << usage << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  try {
    rumd_vhc vhc;
    vhc.SetVerbose(verbose);
    vhc.SetDirectory(directory);
    vhc.ComputeVHC(num_bins, min_dt, first_block,last_block);
    vhc.WriteVHC("vhc.dat");
  }
  catch (const RUMD_Error &e) {
    std::cerr << "RUMD_Error thrown from function " << e.className << "::" << e.methodName << ": " << std::endl << e.errorStr << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;

}
