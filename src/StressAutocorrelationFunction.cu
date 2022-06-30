#include "rumd/StressAutocorrelationFunction.h"
#include "rumd/UserFunctions.h"
#include "rumd/PairPotential.h"
#include "rumd/Sample.h"
#include "rumd/MoleculeData.h"
#include "rumd/Integrator.h"

#include <cstdio>

StressAutocorrelationFunction::StressAutocorrelationFunction(int lvec, float kBT, int isample, int printingScheme, int sacfPrintingFrequency) : lvec(lvec), kBT(kBT), isample(isample), sacfPrintingFrequency(sacfPrintingFrequency), printingScheme(printingScheme), index(0), nsample(0) {

  stressOS = new double[9*lvec];
  sacf10 = new double[lvec];
  sacf3 = new double[lvec];
  

  for ( int n=0; n<lvec; n++ ) sacf10[n] = 0.0;
  for ( int n=0; n<lvec; n++ ) sacf3[n] = 0.0;
  
  system("rm -f viscosity.dat");
}


StressAutocorrelationFunction::~StressAutocorrelationFunction() {
  delete[] stressOS;
  delete[] sacf10;
  delete[] sacf3;
}


void StressAutocorrelationFunction::Update( Sample* S ){

  MoleculeData* M = S->GetMoleculeData();
  Integrator* itg = S->GetIntegrator();
  double timeStep = itg->GetTimeStep();
  double isampleUnit = isample*timeStep;


  CalculateMolecularStress(S);

  //double pressure;
  //FILE* pressureFile = fopen("mol_pressure.dat", "a");
  //pressure = -1.0*(symmetricStress[0] + symmetricStress[1] + symmetricStress[2])/3.0;
  //fprintf(pressureFile, "%e\n", pressure);
  //fclose(pressureFile);

  //symmetric traceless part of the stress tensor (order: xx, yy, zz, xy, xz, yz, yx, zx, zy)
  stressOS[9*index    ] = M->symmetricStress[0] -(M->symmetricStress[0] + M->symmetricStress[1] + M->symmetricStress[2])/3.0;
  stressOS[9*index + 1] = M->symmetricStress[1] -(M->symmetricStress[0] + M->symmetricStress[1] + M->symmetricStress[2])/3.0;
  stressOS[9*index + 2] = M->symmetricStress[2] -(M->symmetricStress[0] + M->symmetricStress[1] + M->symmetricStress[2])/3.0;
  stressOS[9*index + 3] = M->symmetricStress[5];
  stressOS[9*index + 4] = M->symmetricStress[4];
  stressOS[9*index + 5] = M->symmetricStress[3];
  stressOS[9*index + 6] = stressOS[9*index + 3];
  stressOS[9*index + 7] = stressOS[9*index + 4];
  stressOS[9*index + 8] = stressOS[9*index + 5];
  
  index ++;

  if ( index == lvec ){

    nsample++;

    for ( int n=0; n<lvec; n++ ){
      for ( int nn=0; nn<lvec-n; nn++ ){
        sacf3[n] += stressOS[9*(n+nn)+3] * stressOS[9*nn+3] + stressOS[9*(n+nn)+4] * stressOS[9*nn+4] + stressOS[9*(n+nn)+5] * stressOS[9*nn+5];
	sacf10[n] += stressOS[9*(n+nn)] * stressOS[9*nn]
                 + stressOS[9*(n+nn)+1] * stressOS[9*nn+1]
                 + stressOS[9*(n+nn)+2] * stressOS[9*nn+2]
                 + stressOS[9*(n+nn)+3] * stressOS[9*nn+6]
                 + stressOS[9*(n+nn)+4] * stressOS[9*nn+7]
                 + stressOS[9*(n+nn)+5] * stressOS[9*nn+8]
                 + stressOS[9*(n+nn)+6] * stressOS[9*nn+3]
                 + stressOS[9*(n+nn)+7] * stressOS[9*nn+4]
                 + stressOS[9*(n+nn)+8] * stressOS[9*nn+5];
      }
    }

    char sacffilename[100];
    FILE *sacffile  = NULL;

    if (nsample%sacfPrintingFrequency == 0) {
      sprintf(sacffilename, "mol_sacf.%05d.dat", nsample);
      sacffile = fopen(sacffilename, "w");
      if ( sacffile == NULL ){
        fprintf(stderr, "%s Couldn't open file", __func__);
        exit(EXIT_FAILURE);
      }
      fprintf(sacffile, "#time(ru) sacf3 sacf10\n");

      if (printingScheme == 0) {
        for ( int n=0; n<lvec; n++ ){
          float fac = S->GetSimulationBox()->GetVolume()/(nsample*(lvec-n)*kBT);
          if (nsample%sacfPrintingFrequency == 0) {
            fprintf(sacffile, "%e %e %e\n", isampleUnit*n, sacf3[n]*fac/3.0, sacf10[n]*fac/10.0);
          } 
        }
      } else { //print sacf every isample until 10*printingScheme and then every isample*printingScheme
        int lastHighFrequency = printingScheme*10;
        for ( int n=0; n<lastHighFrequency; n++ ){
          float fac = S->GetSimulationBox()->GetVolume()/(nsample*(lvec-n)*kBT);
          fprintf(sacffile, "%e %e %e\n", isampleUnit*n, sacf3[n]*fac/3.0, sacf10[n]*fac/10.0);
        }
        for ( int n=lastHighFrequency; n<lvec; n++ ) {
          float fac = S->GetSimulationBox()->GetVolume()/(nsample*(lvec-n)*kBT);
          if (n%printingScheme == 0)
            fprintf(sacffile, "%e %e %e\n", isampleUnit*n, sacf3[n]*fac/3.0, sacf10[n]*fac/10.0);
        }
      }

      fclose(sacffile);
    }
    index = 0;

  }

}
