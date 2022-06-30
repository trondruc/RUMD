#include "rumd_stats.h"
#include "rumd/RUMD_Error.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[])
{  
  bool verbose = false;
  std::string directory("TrajectoryFiles");
  std::string base_filename("energies");
  unsigned int first_block = 0;
  int last_block = -1;

  static const char *usage = "Calculate averages, variances, standard deviations and covariances, and drifts  of quantities in the energies files.\nUsage: %s [-h] [-f<first_block>] [-l<last_block>] [-v <verbose> ] [-d <directory>] [-b base_filename]\n";
  int opt;
  while ((opt = getopt(argc, argv, "hf:l:v:d:b:")) != -1) {
    switch (opt) {
    case 'h':
      fprintf(stdout, usage, argv[0]);
      exit(EXIT_SUCCESS);
    case 'f':
      first_block = atoi(optarg);
      break;
    case 'l':
      last_block = atoi(optarg);
      break;
    case 'v':
      verbose = (bool) atoi(optarg);
      break;
    case 'd':
      directory = std::string(optarg);
      break;
    case 'b':
      base_filename = std::string(optarg);
      break;
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
  
  try {
    rumd_stats rs;
    rs.SetVerbose(verbose);
    rs.SetDirectory(directory);
    rs.SetBaseFilename(base_filename);
    rs.ComputeStats(first_block, last_block);
    rs.PrintStats();
    rs.WriteStats();
  }
  catch (const RUMD_Error &e) {
    std::cerr << "RUMD_Error thrown from function " << e.className << "::" << e.methodName << ": " << std::endl << e.errorStr << std::endl;
    return 1;
  }
  return 0;
}
