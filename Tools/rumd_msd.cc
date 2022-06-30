
#include "rumd_msd.h"
#include "rumd/ParseInfoString.h"
#include <iostream>

#include <fstream>

rumd_msd::rumd_msd() : Count(),
		       R2(),
		       R4(),
		       Fs(),
		       Fs2(),
		       VAF(),
		       S4(),
		       R2cm(),
		       R4cm(),
		       Fscm(),
		       Fs_(),
		       qvalues(),
		       numParticlesOfType(),
		       nS4_kValues(4),
		       extraTimesWithinBlock(false),
		       subtract_cm_drift(false),
		       allow_type_changes(false),
		       dt(0.0) {}

void rumd_msd::ResetArrays() {
  Count.clear();
  R2.clear();
  R4.clear();
  Fs.clear();
  Fs2.clear();
  VAF.clear();
  S4.clear();
  R2cm.clear();
  R4cm.clear();
  Fscm.clear();

  
  numParticlesOfType.assign(num_types,0);
  Fs_.assign(num_types,0.0);
}

void rumd_msd::CalcR2(Conf &C0, Conf &C1) {

  unsigned long rel_time_index = (C1.metaData.GetLogLin().block - C0.metaData.GetLogLin().block)*blockSize + C1.metaData.GetTimeStepIndex() - C0.metaData.GetTimeStepIndex();
  EnsureMapKeyPresent(rel_time_index);
  Count[rel_time_index]++;

  std::vector<double>& R2_vec = R2[rel_time_index];
  std::vector<double>& R4_vec = R4[rel_time_index];
  std::vector<double>& Fs_vec = Fs[rel_time_index];
  std::vector<double>& Fs2_vec = Fs2[rel_time_index];
  std::vector<double>& VAF_vec = VAF[rel_time_index];
  std::vector<std::vector<double> >& S4_vec = S4[rel_time_index];


  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClassC0 = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);
  
  float Lx = sim_box_params_C0[0];
  float Ly = sim_box_params_C0[1];
  float Lz = sim_box_params_C0[2];
  
  bool LeesEdwards = (simulationBoxClassC0 == "LeesEdwardsSimulationBox") ;

  // for taking the Fourier transform with respect to initial position,
  // which is needed for S_4(k)
  std::vector<std::vector<double> > Fs_k_Cos(num_types, std::vector<double>(nS4_kValues));
  std::vector<std::vector<double> > Fs_k_Sin(num_types, std::vector<double>(nS4_kValues));

  double cm_disp[3] = {0., 0., 0.};
  if(subtract_cm_drift) {
    CalcCM_displacement(C0, C1, &cm_disp[0]);
  }
  

  for (unsigned int T=0; T<num_types; T++) Fs_[T] = 0.0;
  for (unsigned int i=0; i<numParticles; i++) {
    unsigned int Type = C0.P[i].MyType;
    if (Type>=num_types)
      throw RUMD_Error("rumd_msd", __func__, "Type index too high.");
    if(!allow_type_changes && Type!=C1.P[i].MyType)
      throw RUMD_Error("rumd_msd", __func__, std::string("Particle type has changed"));
    double Rsq;
    double this_Fs;
    if(LeesEdwards) {
      // take only the transverse (y and z) contributions
      Rsq = C0.P[i].RsqYZ(&C1.P[i], Ly, Lz );
      this_Fs = C0.P[i].FsqYZ(&C1.P[i], Ly, Lz, qvalues[Type]);
    }
    else {
      if(subtract_cm_drift)
	Rsq = C0.P[i].Rsq_wrt_cm(&C1.P[i], Lx, Ly, Lz, cm_disp );
      else
	Rsq = C0.P[i].Rsq(&C1.P[i], Lx, Ly, Lz );

      this_Fs = C0.P[i].Fsq(&C1.P[i], Lx, Ly, Lz, qvalues[Type]);
    }
    Fs_[Type] += this_Fs;
    float x0 = C0.P[i].x; // initial x position
    for(unsigned k_idx = 0; k_idx < nS4_kValues; k_idx++) {
      float S4_k = 2.*M_PI*(k_idx+1)/Lx; // don't need k=0
      Fs_k_Cos[Type][k_idx] += this_Fs * cos(S4_k*x0);
      Fs_k_Sin[Type][k_idx] += this_Fs * sin(S4_k*x0);
    }

    VAF_vec[Type] += C0.P[i].VelocityDotProduct(&C1.P[i]);

    
    R2_vec[Type] += Rsq;
    R4_vec[Type] += Rsq*Rsq;


  } // for(i==0 ...)

  for (unsigned int T=0; T<num_types; T++) {
    Fs_vec[T] += Fs_[T];
    Fs2_vec[T] += Fs_[T]*Fs_[T];
    // the following gives the real part of S_4(k)
    for(unsigned k_idx = 0; k_idx < nS4_kValues; k_idx++)
      S4_vec[T][k_idx] +=  Fs_k_Cos[T][k_idx] * Fs_k_Cos[T][k_idx] + Fs_k_Sin[T][k_idx] * Fs_k_Sin[T][k_idx];
  }

  // Now call the center of mass version of CalcR2
  if (num_moleculetypes>0)
    CalcR2cm(C0, C1, rel_time_index);
  
}

void rumd_msd::CalcCM_displacement(Conf &C0, Conf &C1, double* CM_disp) {

  for(unsigned coord=0;coord<3;coord++)
    CM_disp[coord] = 0.;

  double my_disp[3];
  double m_tot = 0.;

  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClassC0 = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);
  
  float Lx = sim_box_params_C0[0];
  float Ly = sim_box_params_C0[1];
  float Lz = sim_box_params_C0[2];


  
  for (unsigned int i=0; i<numParticles; i++) {
    unsigned int Type = C0.P[i].MyType;
    if (Type>=num_types)
      throw RUMD_Error("rumd_msd", __func__, "Type index too high.");
    if(!allow_type_changes && Type!=C1.P[i].MyType)
      throw RUMD_Error("rumd_msd", __func__, std::string("Particle type has changed"));

    double myMass = C0.metaData.GetMassOfType(Type);
    m_tot += myMass;

    C0.P[i].Disp(&C1.P[i], Lx, Ly, Lz, &my_disp[0] );

    for(unsigned coord=0;coord<3;coord++)
      CM_disp[coord] += myMass*my_disp[coord];
      
  }

  for(unsigned coord=0;coord<3;coord++)
    CM_disp[coord] /= m_tot;
  
  
}


void rumd_msd::CalcR2cm(Conf &C0, Conf &C1, unsigned long rel_time_index) {
  // Calculate dynamical properties of center of mass of molecules, assuming they all have 'particlesPerMol' atoms

  std::vector<double>& R2_vec = R2cm[rel_time_index];
  std::vector<double>& R4_vec = R4cm[rel_time_index];
  std::vector<double>& Fs_vec = Fscm[rel_time_index];

  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClassC0 = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);
  if(simulationBoxClassC0 == "LeesEdwardsSimulationBox")
    throw RUMD_Error("rumd_msd", "CalcR2cm", "Handling of LeesEdwards boundaries not implmemented yet");


  double Lx = sim_box_params_C0[0];
  double Ly = sim_box_params_C0[1];
  double Lz = sim_box_params_C0[2];

  for (unsigned int i=0; i<numParticles; i+=particlesPerMol) {
    double dXtot   = 0.0;
    double dYtot   = 0.0;
    double dZtot   = 0.0;
    double totMass = 0.0;
   
    for (unsigned int j=0; j<particlesPerMol; j++) {
      unsigned int myType = C0.P[i+j].MyType;
      if (myType>=num_types)
	throw RUMD_Error("rumd_msd", __func__, "Type index too high.");
      if(!allow_type_changes && myType!=C1.P[i+j].MyType)
	throw RUMD_Error("rumd_msd", __func__, std::string("Particle type has changed"));

      double myMass = C0.metaData.GetMassOfType(myType);
      totMass += myMass;
 
      // Calculate the displacement of the i+j'th particle
      double dx = C1.P[i+j].x - C0.P[i+j].x + Lx*(C1.P[i+j].Imx - C0.P[i+j].Imx);
      double dy = C1.P[i+j].y - C0.P[i+j].y + Ly*(C1.P[i+j].Imy - C0.P[i+j].Imy);
      double dz = C1.P[i+j].z - C0.P[i+j].z + Lz*(C1.P[i+j].Imz - C0.P[i+j].Imz);
           
      // Add to the center of mass discplacement
      dXtot += myMass*dx;
      dYtot += myMass*dy;
      dZtot += myMass*dz;
    }                         
    
    dXtot /= totMass;
    dYtot /= totMass;
    dZtot /= totMass;
    
    double Rsq = dXtot*dXtot + dYtot*dYtot + dZtot*dZtot;
                    
    // We are not (yet) differentiating between different types of molecules
    R2_vec[0] += Rsq;
    R4_vec[0] += Rsq*Rsq;
    Fs_vec[0] += (cos(dXtot*qvalues[0]) + cos(dYtot*qvalues[0]) + cos(dZtot*qvalues[0]) )/3.0;
  }
}


void rumd_msd::EnsureMapKeyPresent(unsigned long rel_index) {
  if (Count.find(rel_index) == Count.end()) {
    Count[rel_index] = 0;
    R2[rel_index] = std::vector<double>(num_types, 0.0);
    R4[rel_index] = std::vector<double>(num_types, 0.0);
    Fs[rel_index] = std::vector<double>(num_types, 0.0);
    Fs2[rel_index] = std::vector<double>(num_types, 0.0);
    VAF[rel_index] = std::vector<double>(num_types, 0.0);
    S4[rel_index] = std::vector<std::vector<double> > (num_types, std::vector<double>(nS4_kValues));
    R2cm[rel_index] = std::vector<double>(num_types, 0.0);
    R4cm[rel_index] = std::vector<double>(num_types, 0.0);
    Fscm[rel_index] = std::vector<double>(num_types, 0.0);
  }
  
}

void rumd_msd::ReadQValues() {
  qvalues.resize(num_types);  
  std::string qfilename("qvalues.dat");
  std::ifstream qvaluesFile(qfilename.c_str());
  if(!qvaluesFile.is_open())
    throw RUMD_Error("rumd_msd",__func__,std::string("Could not open " + qfilename));
  std::cout << "\nReading q-values: ";  
  for (unsigned int index=0;index<num_types;index++) {
    qvaluesFile >> qvalues[index];
    std::cout << qvalues[index] << " ";
    assert(qvalues[index]>0);
  }
  std::cout << std::endl << std::endl;
  
}



void rumd_msd::ComputeAll(unsigned int first_block, int last_block,  unsigned int particlesPerMol) {
  unsigned int last_block_to_read; 
  gzFile gzfp;
  Conf C0, CtA, CtB, CtC;

  this->particlesPerMol = particlesPerMol;
  num_moleculetypes = particlesPerMol  > 1 ? 1 : 0;

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
    std::cerr << "Warning: last_block smaller than first_block: no results will be computed" << std::endl;

  // Read metaData from first configuration in first block
  std::string C0_name = GetConfFilename(first_block);
  
  if(verbose)
    std::cout << "Reading block " << first_block << ": " << C0_name << std::endl << std::endl;

  gzfp=gzopen(C0_name.c_str(),"r");
  if (!gzfp)
    throw RUMD_Error("rumd_msd","ComputeAll",std::string("Error opening file") + C0_name + ": " + strerror(errno) );

  C0.Read(gzfp,C0_name,verbose);
  gzclose(gzfp);

  // Store the metaData we need
  dt = C0.metaData.GetDt();
  numParticles = C0.num_part;
  num_types = C0.metaData.GetNumTypes();

  ResetArrays();

  if(qvalues.size() != num_types) {
    if(verbose)
      std::cout << "No q-values have been set or wrong number of types; reading from file" << std::endl;
    ReadQValues();
  }

  // Read and print the number of particles of each type
  for (unsigned int i=0; i<numParticles; i++) {
    unsigned int Type = C0.P[i].MyType;
    assert(Type<num_types);
    numParticlesOfType[Type]++;
  }

  for (unsigned int i=0; i<num_types; i++) {
    std::cout << "Particles of type " << i << ": " << numParticlesOfType[i] << std::endl;
  }


  std::cout << "\nReading block: ";
  for (unsigned int i=first_block; i<=last_block_to_read; i++) {

    // Read first configuration of block 'i'
    std::string C0_name = GetConfFilename(i);


    if ((i-first_block)%10==0) std::cout << i << " ";
    fflush(stdout);
    gzfp=gzopen(C0_name.c_str(),"r");
    C0.Read(gzfp,C0_name,0);

    // Check that relevant metaData didn't change
    assert(C0.metaData.GetDt()       == dt);
    assert(C0.num_part          == numParticles);
    assert(C0.metaData.GetNumTypes() == num_types);
    unsigned int MaxIndex = C0.metaData.GetLogLin().maxIndex;
    unsigned int MaxInterval = C0.metaData.GetLogLin().maxInterval;
    
    // Analyze configurations in the same block-file
    if(!extraTimesWithinBlock)
      for (unsigned int j=1; j<=MaxIndex; j++) {
	CtA.Read(gzfp,C0_name,0);
	CalcR2(C0, CtA);
      }
    else {
      // need a total of four configuration objects to get the extra times
      Conf* pCt1 = &CtA;
      Conf* pCt2 = &CtB;
      Conf* pCt3 = &CtC;
      
      pCt1->Read(gzfp,C0_name,false);
      pCt2->Read(gzfp,C0_name,false);
      pCt3->Read(gzfp,C0_name,false);
      unsigned index = 3; // (index of last read configuration from this block)
      
      CalcR2(C0, *pCt1);
      CalcR2(C0, *pCt2);
      CalcR2(C0, *pCt3);
      CalcR2(*pCt1, *pCt2);
      CalcR2(*pCt1, *pCt3);
      CalcR2(*pCt2, *pCt3);
      
      while(index <MaxIndex) {
	// swap the pointers and read the next configuration
	Conf* pC_temp = pCt1;
	pCt1 = pCt2;
	pCt2 = pCt3;
	pCt3 = pC_temp;
	pCt3->Read(gzfp,C0_name,false);
	index++;
	CalcR2(C0,    *pCt3);
	CalcR2(*pCt1, *pCt3);
	CalcR2(*pCt2, *pCt3);
      };
    } // end if(!extraTimesWithinBlock) ...
    gzclose(gzfp);
    
    // Analyze configurations in the later block-files
    unsigned jstart = 1;
    if(MaxInterval) jstart = 0;
    for (unsigned int j=jstart; i+(1UL<<j)<=last_block_to_read; j++) {
      std::string C0_name = GetConfFilename(i+(1<<j));
      gzfp=gzopen(C0_name.c_str(),"r");
      CtA.Read(gzfp,C0_name,0);
      CalcR2(C0, CtA);
      gzclose(gzfp);
    }
  }
  std::cout << std::endl;
}


///////////////// HELPER FUNCTIONS FOR PYTHON INTERFACE /////////////////

void rumd_msd::Copy_MSD_To_Array(double (*msd_array)[2], unsigned int type) {
  if (Count.size() == 0)
    throw RUMD_Error("rumd_msd","Copy_MSD_To_Array","No data! Call ComputeAll first");
  if(type >= num_types)
    throw RUMD_Error("rumd_msd","Copy_MSD_To_Array","Invalid type index");
  
  std::map<unsigned long, unsigned long>::iterator it;
  unsigned index;
  for(it=Count.begin(), index=0; it != Count.end(); it++, index++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;
    
    msd_array[index][0] = rel_idx*dt;
    msd_array[index][1] = R2[rel_idx][type]/float(count*numParticlesOfType[type]);
  } 
}


void rumd_msd::Copy_ISF_To_Array(double (*isf_array)[2], unsigned int type) {
  if (Count.size() == 0)
    throw RUMD_Error("rumd_msd","Copy_ISF_To_Array","No data! Call ComputeAll first");
  if(type >= num_types)
    throw RUMD_Error("rumd_msd","Copy_ISF_To_Array","Invalid type index");

  std::map<unsigned long, unsigned long>::iterator it;
  unsigned index;
  for(it=Count.begin(), index=0; it != Count.end(); it++, index++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;
    isf_array[index][0] = rel_idx*dt;
    isf_array[index][1] = Fs[rel_idx][type]/float(count*numParticlesOfType[type]);
  } 
}


void rumd_msd::Copy_VAF_To_Array(double (*vaf_array)[2], unsigned int type) {
  if (Count.size() == 0)
    throw RUMD_Error("rumd_msd", __func__, "No data! Call ComputeAll first");
  if(type >= num_types)
    throw RUMD_Error("rumd_msd", __func__, "Invalid type index");

  std::map<unsigned long, unsigned long>::iterator it;
  unsigned index;
  for(it=Count.begin(), index=0; it != Count.end(); it++, index++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;
    vaf_array[index][0] = rel_idx*dt;
    vaf_array[index][1] = VAF[rel_idx][type]/float(count*numParticlesOfType[type]);
  } 
}




void rumd_msd::Copy_Alpha2_To_Array(double (*alpha2_array)[2], unsigned int type) {
  if (Count.size() == 0)
    throw RUMD_Error("rumd_msd","Copy_Alpha2_To_Array","No data! Call ComputeAll first");
  if(type >= num_types)
    throw RUMD_Error("rumd_msd","Copy_Alpha2_To_Array","Invalid type index");

  std::map<unsigned long, unsigned long>::iterator it;
  unsigned index;
  for(it=Count.begin(), index=0; it != Count.end(); it++, index++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;

    alpha2_array[index][0] = rel_idx*dt;
    alpha2_array[index][1] = 3.f/5.f*R4[rel_idx][type]/powf(R2[rel_idx][type],2.f)*float(count*numParticlesOfType[type])-1.f;

  } 
}


void rumd_msd::Fill_Chi4_Array( double (*chi4_array)[2], unsigned int type) {
  if (Count.size() == 0)
    throw RUMD_Error("rumd_msd",__func__,"No data! Call ComputeAll first");
  if(type >= num_types)
    throw RUMD_Error("rumd_msd",__func__,"Invalid type index");
  
  std::map<unsigned long, unsigned long>::iterator it;
  unsigned index;
  for(it=Count.begin(), index=0; it != Count.end(); it++, index++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;

    double fs = Fs[rel_idx][type]/float(count*numParticlesOfType[type]);
    double chi4 = numParticles*(Fs2[rel_idx][type]/float(count*numParticlesOfType[type]*numParticlesOfType[type]) - fs*fs);

    chi4_array[index][0] = rel_idx*dt;
    chi4_array[index][1] = chi4;

  } 

}



void rumd_msd::Fill_S4_Array( double (*S4_array)[2], unsigned int type, unsigned int k_index) {
  if (Count.size() == 0)
    throw RUMD_Error("rumd_msd",__func__,"No data! Call ComputeAll first");
  if(type >= num_types)
    throw RUMD_Error("rumd_msd",__func__,"Invalid type index");
  
  std::map<unsigned long, unsigned long>::iterator it;
  unsigned index;
  for(it=Count.begin(), index=0; it != Count.end(); it++, index++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;

    S4_array[index][0] = rel_idx*dt;
    S4_array[index][1] = numParticles*(S4[rel_idx][type][k_index]/float(count*numParticlesOfType[type]*numParticlesOfType[type]));

  } 

}




////////////// FILE OUTPUT FUNCTIONS /////////////////////////////////////

void rumd_msd::WriteMSD(const std::string& filename) {
  std::ofstream outfile(filename.c_str());
  std::map<unsigned long, unsigned long>::iterator it;

  for(it=Count.begin(); it != Count.end(); it++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;

    outfile << rel_idx*dt;
    for (unsigned int j=0; j<num_types; j++)
      outfile << " " << R2[rel_idx][j]/float(count*numParticlesOfType[j]);
    outfile << std::endl;
  }
  outfile.close();
}


void rumd_msd::WriteMSD_CM(const std::string& filename) {
  std::ofstream outfile(filename.c_str());
  std::map<unsigned long, unsigned long>::iterator it;

  for(it=Count.begin(); it != Count.end(); it++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;

    outfile << rel_idx*dt;
    for (unsigned int j=0; j<num_moleculetypes; j++)
      outfile << " " << R2cm[rel_idx][j]/float(count*numParticles)*float(particlesPerMol);
    outfile << std::endl;
  }
  outfile.close();
}


void rumd_msd::WriteAlpha2(const std::string& filename) {
  std::ofstream outfile(filename.c_str());
  std::map<unsigned long, unsigned long>::iterator it;

  for(it=Count.begin(); it != Count.end(); it++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;

    outfile << rel_idx*dt;
    for (unsigned int j=0; j<num_types; j++)
      outfile << " " << 3.f/5.f*R4[rel_idx][j]/powf(R2[rel_idx][j],2.f)*float(count*numParticlesOfType[j])-1.f;
    outfile << std::endl;
  }
}

void rumd_msd::WriteISF(const std::string& filename) {
  std::ofstream outfile(filename.c_str());
  std::map<unsigned long, unsigned long>::iterator it;
  
  outfile << "# qvalues: ";
  for(unsigned idx=0;idx < qvalues.size();idx++)
    outfile << " " << qvalues[idx];
  outfile << std::endl;

  for(it=Count.begin(); it != Count.end(); it++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;

    outfile << rel_idx*dt;
    for (unsigned int j=0; j<num_types; j++)
      outfile << " " << Fs[rel_idx][j]/float(count*numParticlesOfType[j]);
    outfile << std::endl;
  }
  outfile.close();
}


void rumd_msd::WriteVAF(const std::string& filename) {
  std::ofstream outfile(filename.c_str());
  std::map<unsigned long, unsigned long>::iterator it;

  outfile << "# time";
  for(unsigned i=0; i< num_types; i++)
    outfile << " " << i;
  outfile << std::endl;
  
  for(it=Count.begin(); it != Count.end(); it++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;

    outfile << rel_idx*dt;
    for (unsigned int j=0; j<num_types; j++)
      outfile << " " << VAF[rel_idx][j]/float(count*numParticlesOfType[j]);
    outfile << std::endl;
  }
  outfile.close();
}


void rumd_msd::WriteISF_CM(const std::string& filename) {
  std::ofstream outfile(filename.c_str());
  std::map<unsigned long, unsigned long>::iterator it;

  for(it=Count.begin(); it != Count.end(); it++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;

    outfile << rel_idx*dt;
    for (unsigned int j=0; j<num_moleculetypes; j++)
      outfile << " " << Fscm[rel_idx][j]/float(count*numParticles)*float(particlesPerMol);
    outfile << std::endl;
  }
  outfile.close();
}


void rumd_msd::WriteChi4(const std::string& filename) {
  std::ofstream outfile(filename.c_str());
  std::map<unsigned long, unsigned long>::iterator it;
  
  for(it=Count.begin(); it != Count.end(); it++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;
    
    outfile << rel_idx*dt;
    for (unsigned int T=0; T<num_types; T++) {
      double fs = Fs[rel_idx][T]/float(count*numParticlesOfType[T]);
      double chi4 = numParticles*(Fs2[rel_idx][T]/float(count*numParticlesOfType[T]*numParticlesOfType[T]) - fs*fs);
      outfile << " " << chi4;
    
    }

    outfile << std::endl;
  }
  outfile.close();
}


void rumd_msd::WriteS4(const std::string& filename) {
  std::ofstream outfile(filename.c_str());
  std::map<unsigned long, unsigned long>::iterator it;

  outfile << "# ntypes x n_kvalues = " << num_types << " x " << nS4_kValues  << "; columns are t0,k0 t0,k1 ... t1,k0, t1,k1, ... " << std::endl;
  
  for(it=Count.begin(); it != Count.end(); it++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;
    
    outfile << rel_idx*dt;
    for (unsigned int T=0; T<num_types; T++)
      for(unsigned k_idx = 0; k_idx < nS4_kValues; k_idx++)
	outfile << " " << numParticles*(S4[rel_idx][T][k_idx]/float(count*numParticlesOfType[T]*numParticlesOfType[T]));

    outfile << std::endl;
  }
  outfile.close();
}


void rumd_msd::WriteISF_SQ(const std::string& filename) {
  std::ofstream outfile(filename.c_str());
  std::map<unsigned long, unsigned long>::iterator it;

  for(it=Count.begin(); it != Count.end(); it++) {
    unsigned long rel_idx = (*it).first;
    unsigned long count = (*it).second;

    outfile << rel_idx*dt;
    for (unsigned int j=0; j<num_types; j++)
      outfile << " " << Fs2[rel_idx][j]/float(count*numParticlesOfType[j]);
    outfile << std::endl;
  }
  outfile.close();
}
