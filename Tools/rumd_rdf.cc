#include "rumd_rdf.h"
#include "rumd/ParseInfoString.h"
#include "rumd/RUMD_Error.h"
#include <iostream>
#include <fstream>

rumd_rdf::rumd_rdf() : Count(0),
		       writeEachConfiguration(false),
		       numParticlesOfType(0),
		       numMoleculesOfType(0),
		       rdfNonNorm(0),
		       rdfInterNonNorm(0),
		       rdfIntraNonNorm(0),
		       rdfCMNonNorm(0),
		       sdfNonNorm(0),
		       rdf(0),
		       rdfInter(0),
		       rdfIntra(0),
		       rdfCM(0),
		       sdf(0),
		       rVals(0),
		       num_bins(0), 
		       Lx(0.),
		       Ly(0.),
		       Lz(0.),
		       b_width(0.),
		       dt(0.) {}

void rumd_rdf::AllocateArrays(unsigned int nTypes, unsigned int nBins) {
  assert(nTypes > 0 && nBins > 0);

  if(nTypes != num_types || nBins != num_bins) {
    FreeArrays();

    num_types = nTypes;
    num_bins = nBins;
   
    numParticlesOfType = new int[num_types];

    rdfNonNorm = new long**[num_types];
    rdf = new double**[num_types];
    for (unsigned int i=0; i<num_types; i++) {
      rdfNonNorm[i] = new long*[num_types];
      rdf[i] = new double*[num_types];
      for (unsigned int j=0; j<num_types; j++) {
        rdfNonNorm[i][j] = new long[num_bins];
        rdf[i][j] = new double[num_bins];
      }
    }
    
    sdfNonNorm = new long**[num_types];
    sdf = new double**[num_types];
    for (unsigned int i=0; i<num_types; i++) {
      sdfNonNorm[i] = new long*[DIM];
      sdf[i] = new double*[DIM];
      for (unsigned int j=0; j<DIM; j++) {
        sdfNonNorm[i][j] = new long[num_bins];
	sdf[i][j] = new double[num_bins];
      }
    }

    rVals = new double[num_bins];
  } // if(nTypes != ... )
}

// WARNING: Assuming identical molecules.
void rumd_rdf::AllocateArraysCM(unsigned ppm){
  particlesPerMol = ppm;
  num_moleculetypes = 1;

  numMoleculesOfType = new int[num_moleculetypes];
    
  if ( particlesPerMol <= 1 )
    return;
    
  rdfInterNonNorm = new long**[num_types];
  rdfIntraNonNorm = new long**[num_types];
  rdfCMNonNorm = new long**[num_moleculetypes];
  rdfInter = new double**[num_types];
  rdfIntra = new double**[num_types];
  rdfCM = new double**[num_moleculetypes];
  
  for (unsigned int i=0; i<num_types; i++) {
    rdfInterNonNorm[i] = new long*[num_types];
    rdfIntraNonNorm[i] = new long*[num_types];
    rdfInter[i] = new double*[num_types];
    rdfIntra[i] = new double*[num_types];
    
    for (unsigned int j=0; j<num_types; j++) {
      rdfInterNonNorm[i][j] = new long[num_bins];
      rdfIntraNonNorm[i][j] = new long[num_bins];
      rdfInter[i][j] = new double[num_bins];
      rdfIntra[i][j] = new double[num_bins];
    }
  }
  
  for (unsigned int i=0; i<num_moleculetypes; i++) {
    rdfCMNonNorm[i] = new long*[num_moleculetypes];
    rdfCM[i] = new double*[num_moleculetypes];
    
    for (unsigned int j=0; j<num_moleculetypes; j++) {
      rdfCMNonNorm[i][j] = new long[num_bins];
      rdfCM[i][j] = new double[num_bins];
    }
  }
    
}

void rumd_rdf::FreeArrays() {
  if(!num_types) return;

  delete [] numParticlesOfType; 
  delete [] numMoleculesOfType;

  for (unsigned int i=0; i<num_types; i++) {
    for (unsigned int j=0; j<num_types; j++) {
      delete [] rdfNonNorm[i][j];
      delete [] rdf[i][j];
    }
    delete [] rdfNonNorm[i];
    delete [] rdf[i];
  }
  delete [] rdfNonNorm;
  delete [] rdf;
  
  // molecule stuff
  if ( particlesPerMol > 1 ){
    for (unsigned int i=0; i<num_types; i++) {
      for (unsigned int j=0; j<num_types; j++) {
	delete [] rdfInterNonNorm[i][j];
	delete [] rdfIntraNonNorm[i][j];
	delete [] rdfInter[i][j];
	delete [] rdfIntra[i][j];
      }
      delete [] rdfInterNonNorm[i];
      delete [] rdfIntraNonNorm[i];
      delete [] rdfInter[i];
      delete [] rdfIntra[i];
    }
    delete [] rdfInterNonNorm;
    delete [] rdfIntraNonNorm;
    delete [] rdfInter;
    delete [] rdfIntra;
    
    // center of mass 
    for (unsigned int i=0; i<num_moleculetypes; i++) {
      for (unsigned int j=0; j<num_moleculetypes; j++) {
	delete [] rdfCMNonNorm[i][j];
	delete [] rdfCM[i][j];
      }
      delete [] rdfCMNonNorm[i];
      delete [] rdfCM[i];
    }
    delete [] rdfCMNonNorm;
    delete [] rdfCM;
  } // end molecule stuff
  
  for (unsigned int i=0; i<num_types; i++) {
    for (unsigned int j=0; j<DIM; j++) {
      delete [] sdfNonNorm[i][j];
      delete [] sdf[i][j];
    }
    delete [] sdfNonNorm[i];
    delete [] sdf[i];
  }
  delete [] sdfNonNorm;
  delete [] sdf;

  delete [] rVals;
}


void rumd_rdf::ResetArrays() {
  for (unsigned int i=0; i<num_types; i++)
    for (unsigned int j=0; j<num_types; j++)
      for (unsigned int k=0; k<num_bins; k++){
	rdfNonNorm[i][j][k] = 0;
	if ( particlesPerMol > 1 ){
	  rdfInterNonNorm[i][j][k] = 0;
	  rdfIntraNonNorm[i][j][k] = 0;
	}
      }

  // center of mass
  if ( particlesPerMol > 1 )
    for (unsigned int i=0; i<num_moleculetypes; i++)
      for (unsigned int j=0; j<num_moleculetypes; j++)
	for (unsigned int k=0; k<num_bins; k++)
	  rdfCMNonNorm[i][j][k] = 0;
  
  // sdf
  for (unsigned int i=0; i<num_types; i++)
    for (unsigned int j=0; j<DIM; j++)
      for (unsigned int k=0; k<num_bins; k++)
	sdfNonNorm[i][j][k] = 0;
}

/*
unsigned long int rumd_rdf::GetLastIndex(Conf &C0) {
  gzFile gzfp;

  std::ostringstream LC0Name;
  LC0Name << directory << "/" << basename << "0000.xyz.gz";

  gzfp=gzopen(LC0Name.str().c_str(),"r");
  unsigned long int lastIndex = 0;
  for(unsigned int j=0; j<=MaxIndex; j++)
    C0.Read(gzfp, LC0Name.str().c_str(), false);

  lastIndex = C0.metaData.GetTimeStepIndex();
  gzclose(gzfp);

  return lastIndex;
}
*/

void rumd_rdf::CalcSingleRDF(Conf &C0) {
  unsigned int num_part = C0.num_part;
  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClass = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);

  double inv_bw = 1./b_width;
  double    Lx = sim_box_params_C0[0];
  double invLx = 1./Lx;
  double    Ly = sim_box_params_C0[1];
  double invLy = 1./Ly;
  double    Lz = sim_box_params_C0[2];
  double invLz = 1./Lz;
  double boxShift = 0.;


  if(simulationBoxClass == "LeesEdwardsSimulationBox") 
    boxShift = sim_box_params_C0[3];
  unsigned int dr_index;

  for (unsigned int i=0; i<num_part; i++) {
    unsigned int TypeI =  C0.P[i].MyType;
    assert(TypeI<num_types);
    dr_index = (int)floor(( C0.P[i].x + Lx/2.)*inv_bw);
    if (dr_index < num_bins) sdfNonNorm[TypeI][0][dr_index]++;
    dr_index = (int)floor(( C0.P[i].y + Ly/2.)*inv_bw);
    if (dr_index < num_bins) sdfNonNorm[TypeI][1][dr_index]++;
    dr_index = (int)floor(( C0.P[i].z + Lz/2.)*inv_bw);
    if (dr_index < num_bins) sdfNonNorm[TypeI][2][dr_index]++;

    for (unsigned int j=0; j<i; j++) {
      int TypeJ =  C0.P[j].MyType;
      double R = sqrt( C0.P[i].RsqMinImage(&C0.P[j], Lx, Ly, Lz, invLx, invLy, invLz, boxShift) );

      dr_index = (int)floor(R*inv_bw);
      if (dr_index < num_bins) {
	rdfNonNorm[TypeI][TypeJ][dr_index]++;
	rdfNonNorm[TypeJ][TypeI][dr_index]++;
	if ( particlesPerMol > 1 ){
	  if (i/particlesPerMol != j/particlesPerMol){
	    rdfInterNonNorm[TypeI][TypeJ][dr_index]++;
	    rdfInterNonNorm[TypeJ][TypeI][dr_index]++;
	  }
	  else {
	    double R = sqrt( C0.P[i].Rsq(&C0.P[j], Lx, Ly, Lz) );
	    dr_index = (int)floor(R*inv_bw);
	    if ( dr_index < num_bins ){
	      rdfIntraNonNorm[TypeI][TypeJ][dr_index]++;
	      rdfIntraNonNorm[TypeJ][TypeI][dr_index]++;
	    }
	  }
	}
      }
    }
  }
}

float4 rumd_rdf::CalculateCM( Conf &C0, int index){
  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClass = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);
  double Lx = sim_box_params_C0[0], invLx = 1./Lx; 
  double Ly = sim_box_params_C0[1], invLy = 1./Ly; 
  double Lz = sim_box_params_C0[2], invLz = 1./Lz;
  double boxShift = 0.;
  if(simulationBoxClass == "LeesEdwardsSimulationBox") 
    boxShift = sim_box_params_C0[3];

  double first_x = C0.P[index].x;
  double first_y = C0.P[index].y;
  double first_z = C0.P[index].z;
  
  double M = C0.metaData.GetMassOfType(C0.P[index].MyType);
  double CM_x = M*first_x;
  double CM_y = M*first_y;
  double CM_z = M*first_z;


  for(unsigned i=index+1; i < index + particlesPerMol; i++){ 
    float mass = C0.metaData.GetMassOfType(C0.P[i].MyType);
    
    // Vector from the first particle to the next particle.
    double dx = C0.P[i].x - first_x;
    double dy = C0.P[i].y - first_y;
    double dz = C0.P[i].z - first_z;

    double y_wrap = rintf( dy * invLy );    
    dx -= y_wrap * boxShift;
    
    dx -= Lx*rint( dx*invLx );
    dy -= Ly*y_wrap ;
    dz -= Lz*rint( dz*invLz );

    
    // The correct position of the next particle.
    double new_pos_x =  first_x + dx;
    double new_pos_y =  first_y + dy;
    double new_pos_z =  first_z + dz;
    
    CM_x += mass * new_pos_x;
    CM_y += mass * new_pos_y;
    CM_z += mass * new_pos_z;
    
    M += mass; 
  }
  
  CM_x /= M;
  CM_y /= M;
  CM_z /= M;
  
  float4 CMf4 = {float(CM_x), float(CM_y), float(CM_z), float(1.0/M)};
  return CMf4;
}

void rumd_rdf::CalcSingleRDFCM(Conf &C0) {
  unsigned int num_part = C0.num_part;

  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClass = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);

  
  double inv_bw = 1./b_width; double Lx = sim_box_params_C0[0];
  double invLx = 1./Lx; double Ly = sim_box_params_C0[1];
  double invLy = 1./Ly; double Lz = sim_box_params_C0[2];
  double invLz = 1./Lz; double boxShift = 0.;
  
  if(simulationBoxClass == "LeesEdwardsSimulationBox") 
    boxShift = sim_box_params_C0[3];
  
  for (unsigned int i=0; i<num_part; i+=particlesPerMol) {
    float4 molOne = CalculateCM( C0, i );
    
    // Loop over remaining molecules.
    for (unsigned int j=0; j<i; j+=particlesPerMol){
      float4 molTwo = CalculateCM( C0, j );
      
      double dx = molOne.x - molTwo.x;
      double dy = molOne.y - molTwo.y;
      double dz = molOne.z - molTwo.z;
      

      double y_wrap = rintf( dy * invLy );
      dx -= y_wrap * boxShift;

      // Assuming Rectangular box.
      dx -= Lx*rintf(dx*invLx);
      dy -= Ly * y_wrap;
      dz -= Lz*rintf(dz*invLz);

      double R = sqrt(dx*dx + dy*dy + dz*dz);

      unsigned int dr_index = (unsigned int) floor(R*inv_bw);
      
      if (dr_index < num_bins) {
	rdfCMNonNorm[0][0][dr_index]++;
	rdfCMNonNorm[0][0][dr_index]++;
      }
    }
  }
}


void rumd_rdf::ComputeAll(int nBins, float min_dt, unsigned int first_block, int last_block, unsigned particlePerMol) {
  unsigned int last_block_to_read;
  gzFile gzfp;
  Conf C0;

  try {
    ReadLastCompleteFile("trajectory");
  }
  catch (const RUMD_Error &e) {
    ReadLastCompleteFile("block");
  }

  if(first_block > last_saved_block) {
    std::cerr << "Warning: user-specified first_block out of range; setting to zero" << std::endl;
    first_block = 0;
  }
  if(last_block < 0)
    last_block_to_read = last_saved_block;
  else
    if((unsigned int)last_block > last_saved_block) {
      last_block_to_read = last_saved_block;
      std::cerr << "Warning, specified last_block out of range; using last saved block" << std::endl;
    }
    else
      last_block_to_read = (unsigned int) last_block;

  if(first_block > last_block_to_read)
    std::cerr << "Warning: last_block smaller than first_block: no rdfs will be computed" << std::endl;

  // Read metadata from first configuration in first block
  std::string C0_name = GetConfFilename(first_block);
  
  if(verbose)
    std::cout << "Reading block " << first_block << ": " << C0_name << std::endl << std::endl;

  gzfp=gzopen(C0_name.c_str(),"r");
  if (!gzfp)
    throw RUMD_Error("rumd_rdf","ComputeAll",std::string("Error opening file") + C0_name + ": " + strerror(errno) );

  C0.Read(gzfp, C0_name, verbose);
  gzclose(gzfp);

  // Get meta-data and allocate arrays
  AllocateArrays(C0.metaData.GetNumTypes(), nBins); // sets num_types and num_bins
  AllocateArraysCM(particlePerMol);

  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClassC0 = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);


  Lx  = sim_box_params_C0[0];
  Ly  = sim_box_params_C0[1];
  Lz  = sim_box_params_C0[2];
  //for(unsigned int cdx =1; cdx < DIM; cdx++)
  //  if(sim_box_params_C0[0] != sim_box_params_C0[cdx])
  //    throw RUMD_Error("rumd_rdf","ComputeAll","Require equal box lengths in all three directions");

  float maxL = Lx;
  if(Ly > maxL || Lz > maxL) maxL = (Lz > Ly ? Lz : Ly);
  b_width = maxL/float(num_bins);
  dt  = C0.metaData.GetDt();
  numParticles = C0.num_part;

  ResetArrays();
  Count = 0;

  for (unsigned int tdx=0; tdx<num_types; tdx++)
    numParticlesOfType[tdx] = 0;
  
  for (unsigned int i=0; i<numParticles; i++) {
    unsigned int Type = C0.P[i].MyType;
    assert(Type<num_types);
    numParticlesOfType[Type]++;
  }

  numMoleculesOfType[0] = numParticles / particlesPerMol;

  unsigned long int last_step_used;
  unsigned long int last_block_used;
  
  if(verbose) {
    for (unsigned int i=0; i<num_types; i++)
      std::cout << "Particles of type " << i << ": " << numParticlesOfType[i] << std::endl;
    std::cout << "\nReading block: ";
  }

  for (unsigned int i=first_block; i<=last_block_to_read; i++) {
    // Read first configuration of block 'i'
    std::string C0_name = GetConfFilename(i);

    if (verbose && (i-first_block)%10==0) std::cout << i << " ";
    fflush(stdout);
    gzfp=gzopen(C0_name.c_str(),"r");
    if (!gzfp)
      throw RUMD_Error("rumd_rdf","ComputeAll",std::string("Error opening file ")+C0_name + ": " + strerror(errno));

    C0.Read(gzfp,C0_name,false);

    // Check that relevant metaData didn't change
    assert(C0.metaData.GetDt()       == dt);
    assert(C0.num_part          == numParticles);
    assert(C0.metaData.GetNumTypes() == num_types);

    MaxIndex = C0.metaData.GetLogLin().maxIndex;

    unsigned int nDigits = 5;

    Count++;
    CalcSingleRDF(C0);
    if ( particlePerMol > 1 ) CalcSingleRDFCM(C0);

    if(writeEachConfiguration){
      std::ostringstream rdf_filename;
      rdf_filename << "rdf_" << std::setfill('0') << std::setw(nDigits) << std::setprecision(nDigits) << i << "_" << std::setfill('0') << std::setw(nDigits) << std::setprecision(nDigits) << 0 << ".dat";
      Normalize();
      WriteRDF(rdf_filename.str());
      //WriteRDFCM(rdf_filename.str());
      ResetArrays();
      Count = 0;
    }
    
    last_step_used = C0.metaData.GetTimeStepIndex();
    last_block_used = i;

    // Analyze configurations in the same block-file
    for (unsigned int j=1; j<=MaxIndex; j++) {
      C0.Read(gzfp, C0_name.c_str(), false);
      
      if ( ( (i - last_block_used) * blockSize +  C0.metaData.GetTimeStepIndex() - last_step_used)*dt > min_dt){
       	Count++;
	CalcSingleRDF(C0);
	if ( particlePerMol > 1 ) CalcSingleRDFCM(C0);
	
	if(writeEachConfiguration) {
	  std::ostringstream rdf_filename;
	   rdf_filename << "rdf_" << std::setfill('0') << std::setw(nDigits) << std::setprecision(nDigits) << i << "_" << std::setfill('0') << std::setw(nDigits) << std::setprecision(nDigits) << j << ".dat";
	   Normalize();
	   WriteRDF(rdf_filename.str());
	   //WriteRDFCM(rdf_filename.str());
	   ResetArrays();
	   Count = 0;
	}

	last_step_used = C0.metaData.GetTimeStepIndex();
	last_block_used = i;
      }
    }
    gzclose(gzfp);
  } // end loop over blocks
  if(verbose)
    std::cout << std::endl;

  if(!writeEachConfiguration)
    Normalize();
}

void rumd_rdf::Normalize() {
  for (unsigned int k=0; k<num_bins; k++) {
    double r_start = k*b_width;
    double r_last = (k+1)*b_width;
    double rhoVol = 4./3.*M_PI*(pow(r_last,3.) - pow(r_start,3.)) / (Lx*Ly*Lz);
    rVals[k] = (r_start + r_last)/2.;

    for (unsigned int i=0; i<num_types; i++ ) {
      for (unsigned int j=0; j<num_types; j++ ) {
	double factor = 1./float(Count)/rhoVol;
	if (i==j) factor /= (double (numParticlesOfType[i])*double (numParticlesOfType[i]-1));
	else factor /= (double (numParticlesOfType[i])*double (numParticlesOfType[j]));
	rdf[i][j][k] = rdfNonNorm[i][j][k] * factor;
	if ( particlesPerMol > 1 ) {
	  rdfInter[i][j][k] = rdfInterNonNorm[i][j][k] * factor;
	  rdfIntra[i][j][k] = rdfIntraNonNorm[i][j][k] * factor;
	}
      } // j loop rdf
      
      for (unsigned int j=0; j<3; j++ ) {
	double factor = Lx/b_width/float(Count)/float(numParticles);
	sdf[i][j][k] = sdfNonNorm[i][j][k] * factor;
      } // j loop sdf
    } // i loop

    // center of mass
    if ( particlesPerMol > 1 ){
      for (unsigned int i=0; i<1; i++ ){
	for (unsigned int j=0; j<1; j++ ){
	  double factor = 1./float(Count)/rhoVol;
	  if (i==j) factor /= (double (numMoleculesOfType[i])*double (numMoleculesOfType[i]-1));
	  else factor /= (double (numMoleculesOfType[i])*double (numMoleculesOfType[j]));
	  rdfCM[i][j][k] = rdfCMNonNorm[i][j][k] * factor;
	} // j loop rdf
      } // i loop rdf
    } // center of mass 

  } // k loop
}

void rumd_rdf::WriteRDF(const std::string& filename) {
  FILE* fp = fopen(filename.c_str(),"w");

  fprintf(fp, "# concentrations:");
  for(unsigned int tdx=0; tdx < num_types; ++tdx)
    fprintf(fp, " %f", float(numParticlesOfType[tdx])/numParticles );
  fprintf(fp, "\n");

  for (unsigned int k=0; k<num_bins; k++) {
    fprintf(fp, "%f ", rVals[k]);
     for (unsigned int i=0; i<num_types; i++ ) {
       for (unsigned int j=0; j<num_types; j++ )
	 fprintf(fp, "%f ", rdf[i][j][k]);
     }
     fprintf(fp, "\n");
  }
  fclose(fp);
}
void rumd_rdf::WriteRDFintra(const std::string& filename) {
  FILE* fp = fopen(filename.c_str(),"w");

  fprintf(fp, "# concentrations:");
  for(unsigned int tdx=0; tdx < num_types; ++tdx)
    fprintf(fp, " %f", float(numParticlesOfType[tdx])/numParticles );
  fprintf(fp, "\n");

  for (unsigned int k=0; k<num_bins; k++) {
    fprintf(fp, "%f ", rVals[k]);
     for (unsigned int i=0; i<num_types; i++ ) {
       for (unsigned int j=0; j<num_types; j++ )
	 fprintf(fp, "%f ", rdfIntra[i][j][k]);
     }
     fprintf(fp, "\n");
  }
  fclose(fp);
}

void rumd_rdf::WriteRDFinter(const std::string& filename) {
  FILE* fp = fopen(filename.c_str(),"w");

  fprintf(fp, "# concentrations:");
  for(unsigned int tdx=0; tdx < num_types; ++tdx)
    fprintf(fp, " %f", float(numParticlesOfType[tdx])/numParticles );
  fprintf(fp, "\n");

  for (unsigned int k=0; k<num_bins; k++) {
    fprintf(fp, "%f ", rVals[k]);
     for (unsigned int i=0; i<num_types; i++ ) {
       for (unsigned int j=0; j<num_types; j++ )
	 fprintf(fp, "%f ", rdfInter[i][j][k]);
     }
     fprintf(fp, "\n");
  }
  fclose(fp);
}

void rumd_rdf::WriteRDF_CM(const std::string& filename) {
  FILE* fp = fopen(filename.c_str(),"w");
  
  fprintf(fp, "# concentrations:");
  for(unsigned int tdx=0; tdx < 1; ++tdx)
    fprintf(fp, " %f", float(numMoleculesOfType[tdx])/(numParticles/particlesPerMol) );
  fprintf(fp, "\n");

  for (unsigned int k=0; k<num_bins; k++) {
    fprintf(fp, "%f ", rVals[k]);
     for (unsigned int i=0; i<1; i++ ) {
       for (unsigned int j=0; j<1; j++ )
	 fprintf(fp, "%f ", rdfCM[i][j][k]);
     }
     fprintf(fp, "\n");
  }
  fclose(fp);
}

void rumd_rdf::WriteSDF(const std::string& filename) {
  FILE* fp = fopen(filename.c_str(),"w");
  for (unsigned int k=0; k<num_bins; k++) {
     fprintf(fp, "%f ", rVals[k]);
     for (unsigned int i=0; i<num_types; i++ ) {
        for (int j=0; j<3; j++ ) {
	  fprintf(fp, "%f ", sdf[i][j][k]);
        }
     }
     fprintf(fp, "\n");
  }
  fclose(fp);
}
