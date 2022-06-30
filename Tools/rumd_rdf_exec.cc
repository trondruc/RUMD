#include "rumd_rdf.h"
#include "rumd/RUMD_Error.h"
#include <iostream>


int main(int argc, char *argv[])
{  
  int num_bins = 1000;   // Number of bins in radial distribution function
  int last_block = -1;

  unsigned int first_block = 0;
  unsigned int particlesPerMol = 1;

  float min_dt = 0.0; // Only consider configurations spaced by at least this in time

  bool verbose = false;
  bool writeEachConfiguration = false;

  std::string directory("TrajectoryFiles");

  static const char *usage = "Calculate time averaged radial distribution function.\nUsage: %s [-h] [-n <number of bins>] [-m <minimum time between configurations>] [-f <first_block>] [-l <last_block>] [-v <verbose>] [-d directory] [-e <writeEachConfig>] [-p <particles per molecule>]\n";
  // FIXME fold extended argument description in here
  int opt;
  while ((opt = getopt(argc, argv, "hn:m:f:l:v:d:e:p:")) != -1) {
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
    case 'v':
      verbose = (bool) atoi(optarg);
      break;
    case 'd':
      directory = std::string(optarg);
      break;
    case 'e':
      writeEachConfiguration = (bool) atoi(optarg);
      break;
    case 'p':
      particlesPerMol = atoi(optarg);
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
    rumd_rdf rr;
    rr.SetVerbose(verbose);
    rr.SetDirectory(directory);
    rr.SetWriteEachConfiguration(writeEachConfiguration);
    
    rr.ComputeAll(num_bins, min_dt, first_block,last_block, particlesPerMol);

    if(!writeEachConfiguration){
      rr.WriteRDF("rdf.dat");
      
      if(particlesPerMol>1){
	rr.WriteRDF_CM("rdf_cm.dat");
	rr.WriteRDFinter("rdf_inter.dat");
	rr.WriteRDFintra("rdf_intra.dat");
      }
    }

    rr.WriteSDF("sdf.dat");

  }

  catch (const RUMD_Error &e){
    std::cerr << "RUMD_Error thrown from function " << e.className << "::" << e.methodName << ": " << std::endl << e.errorStr << std::endl;
    return 1;
  }
  return 0;
}
