#include "rumd_vhc.h"
#include "rumd/ParseInfoString.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>

rumd_vhc::rumd_vhc(): Time(0),
		      Count(0),
		      R(0),
		      numParticlesOfType(0),
		      vhcNonNorm(0),
		      vhc(0),
		      MaxDataIndex(0),
		      dt(0.0),
		      rVals(0),
		      num_bins(0),
		      L(0.),
		      dx(0.) {}

void rumd_vhc::AllocateArrays(unsigned int nTypes, unsigned int nBins) {
  assert(nTypes > 0);
  assert(nBins > 0);

  if(nTypes != num_types || nBins != num_bins) {
    FreeArrays();

    num_types = nTypes;
    num_bins = nBins;

    numParticlesOfType = new int[num_types];

    vhcNonNorm = new long**[num_types];
    vhc = new double**[num_types];
    for (unsigned int i=0; i<num_types; i++) {
      vhcNonNorm[i] = new long*[num_types];
      vhc[i] = new double*[num_types];
      for (unsigned int j=0; j<num_types; j++) {
        vhcNonNorm[i][j] = new long[num_bins];
        vhc[i][j] = new double[num_bins];
      }
    }

    rVals = new double[num_bins];
  }
}

void rumd_vhc::FreeArrays() {
  if(!num_types) return;

  delete [] numParticlesOfType;

  for (unsigned int i=0; i<num_types; i++) {
    for (unsigned int j=0; j<num_types; j++) {
      delete [] vhcNonNorm[i][j];
      delete [] vhc[i][j];
    }
    delete [] vhcNonNorm[i];
    delete [] vhc[i];
  }
  delete [] vhcNonNorm;
  delete [] vhc;
  delete [] rVals;
}

void rumd_vhc::ResetArrays() {
  for (unsigned int i=0; i<num_types; i++)
    for (unsigned int j=0; j<num_types; j++)
      for (unsigned int k=0; k<num_bins; k++)
	vhcNonNorm[i][j][k] = 0;
}


void rumd_vhc::CalcSingleVHC(const Conf &C0, const Conf &Ct, unsigned int num_types) {
  unsigned int num_part = C0.num_part;
  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClass = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);

  double invDx = 1./dx;
  double    Lx = sim_box_params_C0[0];
  double    Ly = sim_box_params_C0[1];
  double    Lz = sim_box_params_C0[2];
  unsigned int dr_index;
  for (unsigned int i=0; i<num_part; i++) {
    unsigned int TypeI =  C0.P[i].MyType;
    assert(TypeI<num_types);
    assert(TypeI==Ct.P[i].MyType);
    for (unsigned int j=0; j<i; j++) {
      int TypeJ =  C0.P[j].MyType;
      double Rsq = C0.P[i].Rsq(&Ct.P[i], Lx, Ly, Lz );
      double R = sqrt( Rsq );
      dr_index = (int)floor(R*invDx);
      if (dr_index < num_bins) {
	vhcNonNorm[TypeI][TypeJ][dr_index]++;
	vhcNonNorm[TypeJ][TypeI][dr_index]++;
      }
    }
  }
}

void rumd_vhc::ComputeVHC(unsigned int nBins, int min_dt, unsigned int first_block, int last_block) {
  unsigned int last_block_to_read;
  gzFile gzfp;
  gzFile gzfp2;
  Conf C0, Ct;

  try {
    ReadLastCompleteFile("trajectory");
  }
  catch (const RUMD_Error &e) {
    ReadLastCompleteFile("block");
  }

  if (last_block < 0) {
    last_block_to_read = last_saved_block;
  } else {
    last_block_to_read = (unsigned int) last_block;
  }

  // Read metadata from first configuration in first block
  std::string C0_name = GetConfFilename(first_block);

  if(verbose)
    std::cout << "Reading block " << first_block << ": " << C0_name << std::endl;

  gzfp=gzopen(C0_name.c_str(),"r");
  if (!gzfp)
    throw RUMD_Error("rumd_vhc","Computevhc",std::string("Error opening file") + C0_name + ": " + strerror(errno) );

  C0.Read(gzfp, C0_name, verbose);
  gzclose(gzfp);

  // Get meta-data and allocate arrays
  dt = C0.metaData.GetDt();
  numParticles = C0.num_part;
  unsigned int MaxIndex = C0.metaData.GetLogLin().maxIndex;
  unsigned int MaxInterval = C0.metaData.GetLogLin().maxInterval;
  if(MaxInterval)
    throw RUMD_Error("rumd_vhc","ComputeAll","Only logarithmically saved data allowed");

  // int set_MaxDataIndex = MaxIndex + int(log2(last_block_to_read-first_block+1)+1);

  AllocateArrays(C0.metaData.GetNumTypes(), nBins); // sets num_types and num_bins
  
  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClassC0 = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);
  L  = sim_box_params_C0[0];     // Assuming Lx=Ly=Lz for now...
  for(unsigned int cdx =1; cdx < DIM; cdx++)
    if(sim_box_params_C0[0] != sim_box_params_C0[cdx])
      throw RUMD_Error("rumd_rdf","ComputeAll","Require equal box lengths in all three directions");

  dx = L/float(num_bins);

  ResetArrays();

  Count = 0;

  for (unsigned int tdx=0; tdx<num_types; tdx++)
    numParticlesOfType[tdx] = 0;
  for (unsigned int i=0; i<numParticles; i++) {
    unsigned int Type = C0.P[i].MyType;
    assert(Type<num_types);
    numParticlesOfType[Type]++;
  }

  //unsigned long int last_step_used;
  //unsigned long int last_block_used;

  if(verbose) {
    std::cout << "last_block_to_read: " << last_block_to_read << std::endl;
    for (unsigned int i=0; i<num_types; i++)
    std::cout << "Particles of type " << i << ": " << numParticlesOfType[i] << std::endl;
    std::cout << "\nReading block: ";
  }
  // start looping over blocks.
  for (unsigned int i=first_block; i<=last_block_to_read; i++) {

    if (verbose) {
	std::cout << "." ;
    }
    // Read first configuration of block 'i' and the next configuration with min_dt distance from the first
    std::string C0_name = GetConfFilename(i);

    unsigned int Ct_i = i;
    std::string Ct_name = GetConfFilename(Ct_i);

    if (verbose && (i-first_block)%10==0) std::cout << i << " ";
    fflush(stdout);
    gzfp=gzopen(C0_name.c_str(),"r");
    if (!gzfp)
      throw RUMD_Error("rumd_vhc","Computevhc",std::string("Error opening file ")+C0_name + ": " + strerror(errno));

    gzfp2=gzopen(Ct_name.c_str(),"r");
    if (!gzfp2)
      throw RUMD_Error("rumd_vhc","Computevhc",std::string("Error opening file ")+Ct_name + ": " + strerror(errno));

    C0.Read(gzfp,C0_name,false);
    Ct.Read(gzfp2,Ct_name,false);
    // Check that relevant metaData didn't change
    assert(C0.metaData.GetDt()       == dt);
    assert(C0.num_part               == numParticles);
    assert(C0.metaData.GetNumTypes() == num_types);
    assert(MaxIndex == C0.metaData.GetLogLin().maxIndex);

    assert(Ct.num_part               == numParticles);
    assert(Ct.metaData.GetNumTypes() == num_types);
    assert(MaxIndex == Ct.metaData.GetLogLin().maxIndex);

    unsigned int C0_configurations_left = MaxIndex-1;

    while (true) {
      unsigned long int C0time = ((i-first_block)*blockSize + C0.metaData.GetTimeStepIndex());
      unsigned long Cttime = ((i-first_block)*blockSize + Ct.metaData.GetTimeStepIndex());
      unsigned long Ctgoal = C0time + min_dt;
      unsigned int Ct_configurations_left = MaxIndex-1;
      while ((Cttime < Ctgoal)) {
        if (Ct_configurations_left==0) {
          if (Ct_i<last_block_to_read) {
            ++Ct_i;
          } else {
            // std::cout << "at last block Ct_i: " <<  Ct_i << std::endl;
            break;
          }
          gzclose(gzfp2);
          std::ostringstream CtName;
          CtName << directory << "/" << basename << std::setfill('0') << std::setw(numFilenameDigits) << std::setprecision(numFilenameDigits) << Ct_i << ".xyz.gz";
          // std::cout << "next block Ct_i: " <<  Ct_i << " " << CtName.str() << std::endl;
          gzfp2=gzopen(CtName.str().c_str(),"r");
          if (!gzfp2)
            throw RUMD_Error("rumd_vhc","Computevhc",std::string("Error opening file ")+CtName.str() + ": " + strerror(errno));
          Ct_configurations_left = MaxIndex; // not ... -1
          // no files left ?
        }
        // std::cout << "search: " << C0time << " " << Cttime << " " << Ctgoal << std::endl;
        Ct.Read(gzfp2,Ct_name,false);
        Cttime = ((Ct_i-first_block)*blockSize + Ct.metaData.GetTimeStepIndex());
        --Ct_configurations_left;
      }
      if (fabs(Cttime - Ctgoal) < 1.e-6) {
        // std::cout << "hit:    " << C0time << " " << Cttime << " " << Ctgoal << std::endl;
        CalcSingleVHC(C0,Ct,num_types);
        Count++;
      } else {
        // std::cout << "miss:   " <<  C0time << " " << Cttime << " " << Ctgoal << std::endl;
      }
      if (C0_configurations_left>0) {
        C0.Read(gzfp, C0_name, false);
        --C0_configurations_left;
      } else {
        // std::cout << "no C0 configurations left in block" << std::endl;
        break;
      }
    }
    //last_step_used = C0.metaData.GetTimeStepIndex();
    //last_block_used = i;
    gzclose(gzfp);
    gzclose(gzfp2);
  }
  NormalizeVHC();
  if(verbose) {
    std::cout << std::endl;
  }
}

void rumd_vhc::NormalizeVHC() {
  if(!Count)
    throw RUMD_Error("rumd_vhc","NormalizeVHC","No pairs of configurations with correct time-separation found");
  for (unsigned int k=0; k<num_bins; k++) {
    double r_start = k*dx;
    double r_last = (k+1)*dx;
    double rhoVol = 1./pow(L,3.) * 4./3.*M_PI*(pow(r_last,3.) - pow(r_start,3.));
    rVals[k] = (r_start + r_last)/2.;
    for (unsigned int i=0; i<num_types; i++ ) {
      for (unsigned int j=0; j<num_types; j++ ) {
	double factor = 1./float(Count)/rhoVol;
	if (i==j) {
	  // the following doesn't make sense, and could cause a seg fault
	  // when num_types is 1 ! (NB 11/7/2013)
factor /= float((numParticlesOfType[0]+numParticlesOfType[1])*numParticlesOfType[i]);
        }
	vhc[i][j][k] = vhcNonNorm[i][j][k] *factor/numParticlesOfType[i];
      }
    }
  }
}

void rumd_vhc::WriteVHC(const std::string& filename) {

  FILE* fp = fopen(filename.c_str(),"w");

  fprintf(fp, "# concentrations:");
  for(unsigned int tdx=0; tdx < num_types; ++tdx)
    fprintf(fp, " %f", float(numParticlesOfType[tdx])/numParticles );
  fprintf(fp, "\n");

  for (unsigned int k=0; k<num_bins; k++) {
    fprintf(fp, "%f ", rVals[k]);
     for (unsigned int i=0; i<num_types; i++ ) {
       for (unsigned int j=0; j<num_types; j++ )
	 fprintf(fp, "%f ", vhc[i][j][k]*(4.*M_PI*rVals[k]*rVals[k]));
     }
     fprintf(fp, "\n");
  }
  fclose(fp);
}
