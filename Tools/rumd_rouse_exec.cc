#include "rumd_rouse.h"
#include "rumd/RUMD_Error.h"
#include <iostream>


int main(int argc, char *argv[])
{
  unsigned int first_block=0;
  unsigned int last_block = -1;
  unsigned int particlesPerMol = 0;
  bool verbose = false;
  std::string directory("TrajectoryFiles");

  static const char *usage = "Calculate Rouse modes autocorrelation functions and the orientational autocorrelation of the end-to-end vector of linear molecules. \nUsage: %s [-h] [-p <particles per molecule>] [-f <first_block>] [-l <last_block>] [-d directory] [-v <verbose>]\n";
  int opt;
  while ((opt = getopt(argc, argv, "hm:p:f:l:d:v:")) != -1) {
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

  if(particlesPerMol == 0) {
    fprintf(stderr, "%s: Must supply a value for particles per mol [-p]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  try {
    rumd_rouse rouse;
    rouse.SetVerbose(verbose);
    rouse.SetDirectory(directory);


    rouse.ComputeAll(first_block, last_block, particlesPerMol);

    rouse.NormalizeR0Rt();
    rouse.NormalizeX0Xt();
    rouse.WriteX0Xt("rouse_X0Xt.dat");
    rouse.WriteX0X0("rouse_X0X0.dat");
    rouse.WriteR0Rt("rouse_R0Rt.dat");
    rouse.WriteR0R0("rouse_R0R0.dat");

    // Dont call NormalizeX0X0() before WriteX0X0(), output will be ones
    //rouse.NormalizeX0X0();
    //rouse.WriteXp0Xq0("rouse_Xp0Xq0.dat");
  }
  catch (const RUMD_Error &e) {
    std::cerr << "RUMD_Error thrown from function " << e.className << "::" << e.methodName << ": " << std::endl << e.errorStr << std::endl;
    return 1;
  }
  return 0;

}
