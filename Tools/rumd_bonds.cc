#include "rumd_bonds.h"
#include "rumd/ParseInfoString.h"
#include "rumd/RUMD_Error.h"
#include <iostream>
#include <fstream>

rumd_bonds::rumd_bonds() : writeEachConfiguration(false),
			   Count(0),
			   numBins(0),
			   numBondTypes(0),
			   L(0.),
			   dx(0.),
			   dt(0.),
			   numBondsOfType(0),
			   bondList(0),
			   histNonNorm(0),
			   hist(0),
  			   rVals(0) {}


void rumd_bonds::ReadTopology(std::string topFilename){
  // Erase previous stuff
  numBondTypes = 0;
  numBondsOfType.clear();
  bondList.clear();
  
  // Open topology file
  std::ifstream topFile(topFilename.c_str());
  if(!topFile.good())
    throw RUMD_Error("rumd_bonds","ReadTopology()","No topology file found");

  // Find the start of the bonds section
  std::string line;
  while(topFile.peek() != EOF){
    std::getline(topFile, line);
    if(line=="[ bonds ]"){
      std::getline(topFile, line); // Skip comment line
      break;
    }
  }

  // Loop over bonds
  unsigned mol, a, b, type;
  while(topFile >> mol >> a >> b >> type){
    
    // Check if type exists
    if(numBondTypes < type)
      throw RUMD_Error("rumd_bonds","ReadTopology()","Wrong bond type order");
    if(numBondTypes==type){
      numBondTypes = type+1;
      numBondsOfType.push_back(0);
    }
    
    // Add bond to list
    bond bond = {type, a, b};
    numBondsOfType[type]++;
    bondList.push_back(bond);
  } 

  topFile.close();
  std::cout << "Read " << bondList.size() 
	    << " bonds with " << numBondTypes 
	    << " types from topology file" << std::endl;
}


void rumd_bonds::AllocateArrays(unsigned nBins) {
  assert(numBondTypes > 0 && nBins > 0);

  if(nBins != numBins) {
    FreeArrays();
    numBins = nBins;
   
    hist        = new double*[numBondTypes];
    histNonNorm = new   long*[numBondTypes];
    for(unsigned t=0; t<numBondTypes; t++){
      hist[t]        = new double[numBins];
      histNonNorm[t] = new   long[numBins];
    }
    rVals = new double[numBins];
  }
}


void rumd_bonds::FreeArrays(){
  if(!hist) return;

  for (unsigned t=0; t<numBondTypes; t++){
    delete [] hist[t];
    delete [] histNonNorm[t];
  }
  delete [] hist;
  delete [] histNonNorm;
  delete [] rVals;
}


void rumd_bonds::ResetArrays() { 
  for (unsigned t=0; t<numBondTypes; t++)
    for (unsigned b=0; b<numBins; b++){
      hist[t][b] = 0;
      histNonNorm[t][b] = 0;
    }
}


unsigned long rumd_bonds::GetLastIndex(Conf &C0) {
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


void rumd_bonds::CalcSingleHistogram(Conf &C0) {

  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClass = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);

  double invDx = 1./dx;
  double    Lx = sim_box_params_C0[0];
  double invLx = 1./Lx;
  double    Ly = sim_box_params_C0[1];
  double invLy = 1./Ly;
  double    Lz = sim_box_params_C0[2];
  double invLz = 1./Lz;
  double boxShift = 0.;
  if(simulationBoxClass == "LeesEdwardsSimulationBox") 
    boxShift = sim_box_params_C0[3];

  unsigned dr_index;

  for (unsigned int b=0; b<bondList.size(); b++) {
    bond bond = bondList[b];
    assert(bond.type < numBondTypes);

    double R = sqrt( C0.P[bond.a].RsqMinImage(&C0.P[bond.b], Lx, Ly, Lz, invLx, invLy, invLz, boxShift) );
    dr_index = (int)floor(R*invDx);

    if (dr_index < numBins)
	histNonNorm[bond.type][dr_index]++;
  }
}


void rumd_bonds::ComputeAll(int nBins, float min_dt, unsigned int first_block, int last_block) {
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
    std::cerr << "Warning: user-specified first_block out of range; "
	      << "setting to zero" << std::endl;
    first_block = 0;
  }
  if(last_block < 0)
    last_block_to_read = last_saved_block;
  else
    if((unsigned int)last_block > last_saved_block) {
      last_block_to_read = last_saved_block;
      std::cerr << "Warning, specified last_block out of range; "
		<< "using last saved block" << std::endl;
    }
    else
      last_block_to_read = (unsigned int) last_block;

  if(first_block > last_block_to_read)
    std::cerr << "Warning: last_block smaller than first_block: "
	      << "nothing will be computed" << std::endl;

  // Read first configuration in first block
  std::string C0_name = GetConfFilename(first_block);
  if(verbose)
    std::cout << "Reading block " << first_block << ": "
	      << C0_name << std::endl;
  gzfp=gzopen(C0_name.c_str(),"r");
  if (!gzfp)
    throw RUMD_Error("rumd_bonds", "ComputeAll",
		     std::string("Error opening file")+C0_name+": "+strerror(errno) );
  C0.Read(gzfp, C0_name, verbose);
  gzclose(gzfp);

  // Allocate arrays
  AllocateArrays(nBins); // sets numBins
  ResetArrays();

  // Get meta data
  std::vector<float> sim_box_params_C0;
  std::string simulationBoxClassC0 = ParseInfoString(C0.metaData.GetSimulationBoxInfoStr(), sim_box_params_C0);
  L  = sim_box_params_C0[0];     // Assuming Lx=Ly=Lz for now...
  for(unsigned int cdx =1; cdx < DIM; cdx++)
    if(sim_box_params_C0[0] != sim_box_params_C0[cdx])
      throw RUMD_Error("rumd_bonds", "ComputeAll",
		       "Require equal box lengths in all three directions");
  dx = L/float(numBins);
  dt  = C0.metaData.GetDt();
  MaxIndex = C0.metaData.GetLogLin().maxIndex;

  unsigned long int lastIndex = GetLastIndex(C0);
  unsigned long int last_step_used;
  unsigned long int last_block_used;
  if(verbose) {
    std::cout << "LastIndex: " << lastIndex << std::endl;
    std::cout << "\nReading block: ";
  }

  // Loop over configurations
  Count = 0;
  for (unsigned i=first_block; i<=last_block_to_read; i++) {
    // Read first configuration of block 'i'
    std::string C0_name = GetConfFilename(i);

    if (verbose && (i-first_block)%10==0) std::cout << i << " ";
    fflush(stdout);
    gzfp=gzopen(C0_name.c_str(),"r");
    if (!gzfp)
      throw RUMD_Error("rumd_bonds","ComputeAll",
		       std::string("Error opening file ")+C0_name +": "+strerror(errno));
    C0.Read(gzfp,C0_name,false);

    // Check that relevant metaData didn't change
    assert(C0.metaData.GetDt()       == dt);
    assert(MaxIndex == C0.metaData.GetLogLin().maxIndex);
    unsigned int nDigits = 5;

    // Calculate historgram for this configuration
    Count++;
    CalcSingleHistogram(C0);

    // Maybe save this
    if(writeEachConfiguration){
      std::ostringstream bonds_filename;
      bonds_filename << "bonds_" << std::setfill('0') << std::setw(nDigits) << std::setprecision(nDigits) << i << "_" << std::setfill('0') << std::setw(nDigits) << std::setprecision(nDigits) << 0 << ".dat";
      Normalize();
      WriteBonds(bonds_filename.str());
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
	CalcSingleHistogram(C0);
	
	if(writeEachConfiguration) {
	  std::ostringstream bonds_filename;
	  bonds_filename << "bonds_" << std::setfill('0') << std::setw(nDigits) << std::setprecision(nDigits) << i << "_" << std::setfill('0') << std::setw(nDigits) << std::setprecision(nDigits) << j << ".dat";
	  Normalize();
	  WriteBonds(bonds_filename.str());
	  ResetArrays();
	  Count = 0;
	}

	last_step_used = C0.metaData.GetTimeStepIndex();
	last_block_used = i;
      }
    }
    gzclose(gzfp);
  } // end loop over blocks

  if(verbose) std::cout << std::endl;

  if(!writeEachConfiguration) Normalize();
}


void rumd_bonds::Normalize() {
  double factor = 1./float(Count)/dx;

  for(unsigned bin=0; bin<numBins; bin++) {
    double r_start = bin*dx;
    double r_last = (bin+1)*dx;
    rVals[bin] = 0.5*(r_start + r_last);

    for (unsigned t=0; t<numBondTypes; t++ )
      hist[t][bin] = histNonNorm[t][bin]*factor/numBondsOfType[t];
  }
}


void rumd_bonds::WriteBonds(const std::string& filename) {

  // Find first bin with non-zero data (to save space)
  bool allZero = true;
  unsigned firstBin = 0;
  for(unsigned b=0; b+1<numBins && allZero; b++){
    for(unsigned t=0; t<numBondTypes; t++)
      if(histNonNorm[t][b+1]>0)
	allZero = false;
    firstBin = b;
  }

  // Find last bin with non-zero data (to save space)
  allZero = true;
  unsigned lastBin = 0;
  for(unsigned b=numBins; b-2>0 && allZero; b--){
    for(unsigned t=0; t<numBondTypes; t++)
      if(histNonNorm[t][b-2] > 0)
	allZero = false;
    lastBin = b;
  }

  FILE* fp = fopen(filename.c_str(),"w");

  fprintf(fp, "#bondlength, probability\n");
  for(unsigned b=firstBin; b<lastBin; b++) {
    fprintf(fp, "%f ", rVals[b]);
    for(unsigned t=0; t<numBondTypes; t++)
      fprintf(fp, "%f ", hist[t][b]);
    fprintf(fp, "\n");
  }
  fclose(fp);
}
