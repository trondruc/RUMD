#include "rumd_msd.h"
#include "rumd/RUMD_Error.h"
#include <iostream>


int main(int argc, char *argv[])
{  
  unsigned int first_block=0;
  unsigned int last_block = -1;   
  unsigned int particlesPerMol = 0;
  bool extra_times = false;
  bool subtract_cm_drift = false;
  bool verbose = false;
  std::string directory("TrajectoryFiles");
  bool allow_type_changes = false;
  
  static const char *usage = "Calculate dynamical correlation functions: intermediate scattering function, mean-squared displacement, non-Gaussian parameter.\nUsage: %s [-h] [-p <particles per molecule>] [-f <first_block>] [-l <last_block>] [-d directory] [-e <extra_times>] [-s <subtract_cm>] [-v <verbose>] [-t <allow_type_changes>]\n";
  int opt;
  while ((opt = getopt(argc, argv, "hm:p:f:l:d:e:s:v:t:")) != -1) {
    switch (opt) {
    case 'm':
      std::cout << "-m for particles per molecule is deprecated; use -p instead" << std::endl;
      particlesPerMol = atoi(optarg);
      break;
    case 'p':
      particlesPerMol = atoi(optarg);
      break;
    case 'h':
      fprintf(stdout, usage, argv[0]);
      exit(EXIT_SUCCESS);
    case 'f':
      first_block = atoi(optarg);
      break;
    case 'l':
      last_block = atoi(optarg);
      break;
    case 'd':
      directory = std::string(optarg);
      break;
    case 'v':
      verbose = (bool) atoi(optarg);
      break;
    case 'e':
      extra_times = true;
      break;
    case 's':
      subtract_cm_drift = true;
      break;
    case 't':
      allow_type_changes = true;
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
    rumd_msd msd;
    msd.SetVerbose(verbose);
    msd.SetDirectory(directory);
    msd.SetExtraTimesWithinBlock(extra_times);
    msd.SetSubtractCM_Drift(subtract_cm_drift);
    msd.SetAllowTypeChanges(allow_type_changes);
    msd.ComputeAll(first_block, last_block, particlesPerMol);
    
    msd.WriteMSD("msd.dat");
    msd.WriteAlpha2("alpha2.dat");
    msd.WriteISF("Fs.dat");
    msd.WriteISF_SQ("FsSq.dat");

    msd.WriteVAF("vaf.dat");
    
    msd.WriteChi4("chi4.dat");
    msd.WriteS4("S4.dat");

    if (particlesPerMol>1) {
      msd.WriteMSD_CM("msd_cm.dat");
      msd.WriteISF_CM("Fs_cm.dat");
    }
    
  }
  catch (const RUMD_Error &e) {
    std::cerr << "RUMD_Error thrown from function " << e.className << "::" << e.methodName << ": " << std::endl << e.errorStr << std::endl;
    return 1;
  }
  return 0;

}
