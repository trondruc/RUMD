
/*
    Copyright (C) 2010  Thomas Schrøder

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/

#include "rumd/Sample.h"
#include "rumd/Integrator.h"
#include "rumd/IntegratorNVT.h"
#include "rumd/Potential.h"
#include "rumd/PairPotential.h"
#include "rumd/AnglePotential.h"
#include "rumd/DihedralPotential.h"
#include "rumd/ConstraintPotential.h"
#include "rumd/ConfigurationOutputManager.h"
#include "rumd/ConfigurationWriterReader.h"
#include "rumd/RUMD_Error.h"
#include "rumd/MoleculeData.h"
#include "rumd/Device.h"
#include "rumd/EnergiesOutputManager.h"




#include <algorithm>

/////////////////////////////////////////////
// CONSTRUCTION/DESTRUCTION 
/////////////////////////////////////////////


// for N larger than this choose pb=64 as default, otherwise 32
const unsigned int Sample::default_pb_size_threshold = 10000;
// for N larger than this choose tp=1 as default, otherwise 4
const unsigned int Sample::default_tp_size_threshold = 10000;

//const unsigned int Sample::max_num_particles_sort_XY = 600000;

Sample::Sample( unsigned int pb, unsigned int tp) 
  : particleData(),
    moleculeData(0),
    trajectoryDir("./TrajectoryFiles"),
    degreesOfFreedom(0),
    userDegreesOfFreedom(0),
    blockSize(0),
    simulationBox(0),
    own_sim_box(true),
    include_kinetic_stress(true),
    itg(0),
    verbose(true),
    check_cuda_errors(false),
    sortingScheme(SORT_XY)
{
  cudaDeviceSetCacheConfig( cudaFuncCachePreferShared );
  SetPB_TP(pb, tp);

  CreateDefaultOutputManagers();
}

Sample::~Sample(){
  delete restartWriter;
  delete restartManager;
  
  delete trajectoryWriter;
  delete trajectoryManager;

  delete energiesOutputManager;
  
  if(simulationBox && own_sim_box)
    delete simulationBox;
}

// Copy-constructor
Sample::Sample(const Sample& S) : particleData(),
				  moleculeData(0),
				  trajectoryDir("./TrajectoryFiles"),
				  degreesOfFreedom(0),
				  userDegreesOfFreedom(0),
				  blockSize(0),
				  own_sim_box(true),
				  itg(0),
				  verbose(true),
				  sortingScheme(SORT_XY) {
  SetPB_TP(S.GetParticlesPerBlock(), S.GetThreadsPerParticle());

  simulationBox = S.GetSimulationBox()->MakeCopy();
  include_kinetic_stress = S.include_kinetic_stress;

  SetNumberOfParticles(S.GetNumberOfParticles());
  // we assume the device data is always the most up to date
  S.CopySimulationDataFromDevice();
  particleData.CopyHostData(*S.GetParticleDataConst()); // copies only on CPU
  CopySimulationDataToDevice();

  // may want to make it optional to do the following, or have a different
  // trajectoryDir
  CreateDefaultOutputManagers();
}

// here would like to be able to check a "DataStatus" flag to see if
// need to copy data from the device
Sample* Sample::Copy() {
  Sample* new_sample = new Sample( (*this) );
  return new_sample;
}

Sample& Sample::operator=(const Sample& S){
  if(this != &S){ 
    if(GetNumberOfParticles() != S.GetNumberOfParticles())
      SetNumberOfParticles(S.GetNumberOfParticles());

    // Note: the following only copies on the device.
    particleData = *S.GetParticleDataConst();
    
    if(!simulationBox) {
      simulationBox = S.GetSimulationBox()->MakeCopy();
      own_sim_box = true;
    }
    else
      simulationBox->CopyAnotherBox(S.GetSimulationBox());

    // Trigger neighborList update in any pair potential(s).
    std::vector<Potential*>::iterator potIter;
    for(potIter = potentialList.begin(); potIter != potentialList.end(); potIter++)
      (*potIter)->ResetInternalData();

  }
  return *this; 
}

/////////////////////////////////////////////
// Initialization of Sample 
/////////////////////////////////////////////

void Sample::SetPB_TP( unsigned int pb, unsigned int tp ) {
  
  particles_per_block = pb;
  threads_per_particle = tp;
  
  unsigned current_nParticles = GetNumberOfParticles();
  if(current_nParticles)
    SetNumberOfParticles(current_nParticles); // reallocation may be needed
}



void Sample::CreateDefaultOutputManagers() {
  managers_initialized = false;
  
  restartWriter = new ConfigurationWriterReader();
  // Here we directly access the metaData object belonging to the writer object
  // We could alternatively call SetMetaData on the manager object (below)
  restartWriter->metaData.Set("precision", 9);
  restartWriter->metaData.Set("images", true);
  restartWriter->metaData.Set("velocities", true);
  restartWriter->metaData.Set("forces", true);
  restartWriter->metaData.Set("pot_energies", true);
  restartWriter->metaData.Set("virials", true);
  
  restartManager = new ConfigurationOutputManager(this, trajectoryDir, "restart", restartWriter);
  restartManager->SetVerbose(verbose);

  outputManagers["restart"] = restartManager;

  trajectoryWriter = new ConfigurationWriterReader();
  trajectoryWriter->metaData.Set("precision", 4);
  trajectoryWriter->metaData.Set("images", true);
  trajectoryWriter->metaData.Set("velocities", false);
  trajectoryWriter->metaData.Set("forces", false);
  trajectoryWriter->metaData.Set("pot_energies", true);
  trajectoryWriter->metaData.Set("virials", true);
  trajectoryWriter->metaData.Set("logLin", true);  

  trajectoryManager = new ConfigurationOutputManager(this, trajectoryDir, "trajectory", trajectoryWriter);
  trajectoryManager->SetDuplicateBlockEnds(true);
  trajectoryManager->SetVerbose(verbose);
  // make the default behaviour pure logarithmic saving with base 1:  
  trajectoryManager->SetLogLinParams(1,0);
  outputManagers["trajectory"] = trajectoryManager;

  energiesOutputManager = new EnergiesOutputManager(this, trajectoryDir, "energies");
  energiesOutputManager->SetMetaData("precision", 6);
  energiesOutputManager->SetVerbose(verbose);
  outputManagers["energies"] = energiesOutputManager;
}

void Sample::SetSimulationBox(SimulationBox* simBox, bool own_this_box){ 
  if(own_sim_box) delete simulationBox;
  simulationBox = simBox;

  for(std::vector<Potential*>::iterator potIter = potentialList.begin(); potIter != potentialList.end(); potIter++)
    (*potIter)->SetSample(this);

  own_sim_box = own_this_box;
}



void Sample::SetSortingScheme(SortingScheme ss) {
  sortingScheme = ss;
}


void Sample::SetNumberOfParticles(unsigned int set_num_part){
  if(set_num_part == 0)
    return;
  
  if(particles_per_block == 0) particles_per_block  = set_num_part > 
				 default_pb_size_threshold ? 64 : 32;
  if(threads_per_particle == 0) threads_per_particle = set_num_part >
				  default_tp_size_threshold ? 1 : 4;


 if (particles_per_block*threads_per_particle > Device::GetDevice().GetMaxThreadsPerBlock())
    throw(RUMD_cudaResourceError("Sample", __func__, "Too many threads per block (pb*tp>maxthreads)"));
 
  particleData.SetNumberOfParticles(set_num_part, particles_per_block);
  // Print useful info.
  if(verbose) {
    std::cout << std::endl << "Setting number of particles and allocating on host and device:" << std::endl;
    std::cout << "Number of particles (N): " << std::setw( 9 ) << particleData.GetNumberOfParticles() << std::endl;
    std::cout << "Particles per block (pb): " << std::setw( 6 ) <<  particles_per_block << std::endl;
    std::cout << "Threads per particle (tp): " << std::setw( 4 )<<  threads_per_particle << std::endl;
    std::cout << "Number of blocks: " <<  std::setw( 14 ) << particleData.GetNumberOfBlocks() << std::endl;
    std::cout << "Virtual particles: " <<  std::setw( 15 ) << particleData.GetNumberOfVirtualParticles() << std::endl;
    std::cout << "Unused virtual particles: " << std::setw( 6 ) << (particleData.GetNumberOfVirtualParticles() - particleData.GetNumberOfParticles()) << std::endl;
    std::cout << "Threads per block: " << std::setw( 14 ) << (threads_per_particle*particles_per_block) << std::endl;
    std::cout << "Number of threads: " <<  std::setw( 15 ) << (threads_per_particle*particles_per_block*particleData.GetNumberOfBlocks()) << std::endl << std::endl;
  }

  UpdateDegreesOfFreedom();

  kPlan.grid.x = particleData.GetNumberOfBlocks();
  kPlan.threads.x = particles_per_block;
  kPlan.threads.y = threads_per_particle;
  kPlan.shared_size = particles_per_block * threads_per_particle * sizeof(float4);
  kPlan.num_blocks = particleData.GetNumberOfBlocks();
  kPlan.num_virt_part = particleData.GetNumberOfVirtualParticles();
  
  for(std::vector<Potential*>::iterator potIter = potentialList.begin(); potIter != potentialList.end(); potIter++)
    (*potIter)->SetSample(this);
  
  if(itg)
    itg->SetSample(this);
}

void Sample::SetPotential(Potential* potential){ 
  // remove output items associated with potentials from energiesOutputManager
  std::vector<Potential*>::iterator potIter;
  for(potIter = potentialList.begin(); potIter != potentialList.end(); potIter++)
    energiesOutputManager->UnregisterPotential(*potIter);

  potentialList.clear();
  AddPotential(potential);
}

void Sample::AddPotential(Potential* potential){
  potential->SetVerbose(verbose);
  energiesOutputManager->RegisterPotential(potential);
  
  if(GetNumberOfParticles())
    potential->SetSample(this);

  if(dynamic_cast<ConstraintPotential*>(potential)){
    potentialList.push_back(potential);
  }
  else{
    // Will induce reallocation, but this is a small list.
    potentialList.insert(potentialList.begin(), potential);
  }
  energiesOutputManager->metaData.bool_options[potential->GetID_String()] = false;
  energiesOutputManager->metaData.fileStr[potential->GetID_String()] = potential->GetID_String();

}

// Set the current integrator.
void Sample::SetIntegrator(Integrator* newItg){
  if(itg && newItg != itg )
    energiesOutputManager->UnregisterIntegrator(itg);

  itg = newItg;
  energiesOutputManager->RegisterIntegrator(itg);
  // SetNumberOfParticles has been called earlier and did not allocate.
  if(GetNumberOfParticles())
    itg->SetSample(this);

}

// Update the number of degrees of freedom
void Sample::UpdateDegreesOfFreedom(unsigned numberOfConstraints){
  if(userDegreesOfFreedom > 0){
    std::cout << "WARNING: Updating the degrees of freedom has no effect, since it has been set manually." << std::endl;
    std::cout << "Currently using " << GetNumberOfDOFs() << " degrees of freedom." << std::endl;
  }
  
  else{
    degreesOfFreedom = 3*particleData.GetNumberOfParticles() - 3;
    if(moleculeData){
      degreesOfFreedom -= numberOfConstraints;
      if(verbose)
	std::cout << "Subtracting " << numberOfConstraints << " degrees of freedom in total due to constraints. " << std::endl;
    }
    if(verbose)
      std::cout << "Updated to " << degreesOfFreedom << " degrees of freedom." << std::endl;
  }
}

void Sample::SetNumberOfDOFs(unsigned int DOFs){
  userDegreesOfFreedom = DOFs;
  std::cout << "Manually setting the degrees of freedom to " << userDegreesOfFreedom << "." << std::endl;
  std::cout << "WARNING: this will prevent any automatic changes in the degrees of freedom due to for instance changes in the number of particles or bond constraints." << std::endl;
}

unsigned int Sample::GetNumberOfDOFs() const {
  if(userDegreesOfFreedom > 0)
    return userDegreesOfFreedom;
  else
    return degreesOfFreedom;
}

///////////////////////////////////////////////////////////////
// Force and potential
///////////////////////////////////////////////////////////////

void Sample::CalcF(bool calc_stresses){
  if(!simulationBox)
    throw RUMD_Error("Sample","CalcF","No simulation box has been set");
  if(!potentialList.size())
    std::cerr << "Warning [Sample::CalcF]: No potential has been set" << std::endl;

  for(unsigned int pdx=0; pdx < potentialList.size(); ++pdx)
    potentialList[pdx]->CalcF(pdx==0, calc_stresses);
  if(itg)
    itg->CalculateAfterForce();
  
  if(check_cuda_errors) {
    cudaDeviceSynchronize();
    cudaError err = cudaGetLastError();
    if(err != cudaSuccess) {
      if(err == cudaErrorLaunchOutOfResources ||
	 err == cudaErrorInvalidConfiguration)
	throw( RUMD_cudaResourceError("Sample","CalcF",std::string("tp too high or too many thread blocks: ") + cudaGetErrorString(err)) );
      else   
	throw( RUMD_Error("Sample","CalcF",std::string("cuda error: ") + cudaGetErrorString(err) ) );
    }
  }
}

double Sample::GetPotentialEnergy(bool copy) const {

  if(copy)
    particleData.CopyForFromDevice();

  unsigned int nParticles = GetNumberOfParticles();

  double potentialEnergy = 0.;
  for(unsigned int i=0; i < nParticles; i++)
    potentialEnergy += particleData.h_f[i].w;

  // loop over potentials
  for(unsigned int pdx=0; pdx < potentialList.size(); ++pdx) {
    if(!potentialList[pdx]->EnergyIncludedInParticleSum())
      potentialEnergy += potentialList[pdx]->GetPotentialEnergy();
  }

  return potentialEnergy;
}

double Sample::GetVirial(bool copy) const {
  if(copy)
    particleData.CopyVirFromDevice();
  double virial = 0.;
  unsigned int nParticles = GetNumberOfParticles();

  for(unsigned int i=0; i < nParticles; i++) { 
    virial += particleData.h_w[i].w;
  }
  virial /= 6.0;
  
  // loop over potentials to get contributions not included in particle sum
  for(unsigned int pdx=0; pdx < potentialList.size(); ++pdx) {
    if(!potentialList[pdx]->EnergyIncludedInParticleSum())
      virial += potentialList[pdx]->GetVirial();
  }

  return virial;
}

std::vector<double> Sample::GetStress(bool copy) const {

  if(copy) {
    particleData.CopyVirFromDevice();
    particleData.CopyStressFromDevice();
    if(include_kinetic_stress)
      particleData.CopyVelFromDevice();
  }

  double stress[6] = {0., 0., 0., 0., 0., 0.};
  for(unsigned int i=0; i < GetNumberOfParticles(); i++) {
	stress[0] += particleData.h_sts[i].x;
	stress[1] += particleData.h_sts[i].y;
	stress[2] += particleData.h_sts[i].z;
	stress[3] += particleData.h_sts[i].w;
	stress[4] += particleData.h_w[i].y;
	stress[5] += particleData.h_w[i].z;
	if( include_kinetic_stress ) {
	  float mass2 = 2./particleData.h_v[i].w; // divide by 2 below
	  float vx = particleData.h_v[i].x;
	  float vy = particleData.h_v[i].y;
	  float vz = particleData.h_v[i].z;
	 
	  stress[0] -= mass2 * vx * vx;
	  stress[1] -= mass2 * vy * vy;
	  stress[2] -= mass2 * vz * vz;
	  stress[3] -= mass2 * vy * vz;
	  stress[4] -= mass2 * vx * vz;
	  stress[5] -= mass2 * vx * vy;
	}
  }

  // Have commented this out because ConstraintPotential is now in this class
  // and a test is failing. It's not clear anything else should be done.
  // A warning message is now written by ConstraintPotential to point out that
  // it doesn't support stress calculation.
  // TBD Have a GetStress for these potentials?

  // check if any potentials are present which do not contribute to particle stresses
  /*for(unsigned int pdx=0; pdx < potentialList.size(); ++pdx) {
   if(!potentialList[pdx]->EnergyIncludedInParticleSum())
     throw RUMD_Error("Sample",__func__, std::string("Stress not available for  potential ")+potentialList[pdx]->GetID_String());
  }
  */

  float V = simulationBox->GetVolume();
  std::vector<double> stress_vec;
  // need to include 0.5 here to account for double counting
  for(int idx=0;idx<6;++idx)
    stress_vec.push_back(0.5*stress[idx]/V);
  return stress_vec;
}


////////////////////////////////////////////////////////
// IO   
////////////////////////////////////////////////////////

void Sample::SetVerbose(bool vb) {
  verbose = vb;
  std::map<std::string, LogLinOutputManager*>::iterator OM_Iter;
  for(OM_Iter = outputManagers.begin(); OM_Iter != outputManagers.end(); OM_Iter++)
    OM_Iter->second->SetVerbose(verbose);

  for(unsigned int pdx=0; pdx < potentialList.size(); ++pdx)
    potentialList[pdx]->SetVerbose(verbose);

}

void Sample::SetOutputDirectory(const std::string& trajectoryDir) {
  this->trajectoryDir = trajectoryDir;
  std::map<std::string, LogLinOutputManager*>::iterator OM_Iter;
  for(OM_Iter = outputManagers.begin(); OM_Iter != outputManagers.end(); OM_Iter++)  
    OM_Iter->second->SetOutputDirectory(trajectoryDir);
  
  managers_initialized = false;
}


void Sample::EnableBackup(bool make_backup) {

  std::map<std::string, LogLinOutputManager*>::iterator OM_Iter;
  for(OM_Iter = outputManagers.begin(); OM_Iter != outputManagers.end(); OM_Iter++)  
    OM_Iter->second->EnableBackup(make_backup);
  
  managers_initialized = false;
}


void Sample::SetOutputBlockSize(unsigned long int blockSize) {
  if(!blockSize) throw RUMD_Error("Sample","SetOutputBlockSize","blockSize must be greater than zero");
  this->blockSize = blockSize;
  
  std::map<std::string, LogLinOutputManager*>::iterator OM_Iter;
  for(OM_Iter = outputManagers.begin(); OM_Iter != outputManagers.end(); OM_Iter++)  
    OM_Iter->second->SetBlockSize(blockSize);

  restartManager->SetLogLinParams(blockSize,0);

  managers_initialized = false;
}

void Sample::SetOutputManagerActive(const std::string &manager_name, bool active) {
  std::map<std::string, LogLinOutputManager*>::iterator OM_It = outputManagers.find(manager_name);
  if(OM_It == outputManagers.end())
    std::cerr << "Warning [Sample::SetOutputManagerActive]: Manager " << manager_name << " " << "not present" << std::endl;
  else
    OM_It->second->SetActive(active);
}

void Sample::SetOutputManagerMetaData(const std::string& manager_name, const std::string& key, bool on) {
  std::map<std::string, LogLinOutputManager*>::iterator OM_It = outputManagers.find(manager_name);
  if(OM_It == outputManagers.end())
    std::cerr << "Warning [Sample::SetOutputManagerMetaData]: Manager " << manager_name << " " << "not present" << std::endl;
  else
    OM_It->second->SetMetaData(key, on);
}

void Sample::SetOutputManagerMetaData(const std::string& manager_name, const std::string& key, int value) {
  std::map<std::string, LogLinOutputManager*>::iterator OM_It = outputManagers.find(manager_name);
  if(OM_It == outputManagers.end())
    std::cerr << "Warning [Sample::SetOutputManagerMetaData]: Manager " << manager_name << " " << "not present" << std::endl;
  else
    OM_It->second->SetMetaData(key, value);
}

void Sample::AddExternalCalculator(ExternalCalculator* calc) {
  energiesOutputManager->AddExternalCalculator(calc);
  managers_initialized = false;
}

void Sample::RemoveExternalCalculator(ExternalCalculator* calc) {
  energiesOutputManager->RemoveExternalCalculator(calc);
  managers_initialized = false;
}

void Sample::AddOutputManager(const std::string& manager_name, LogLinOutputManager* OM) {
  OM->SetVerbose(verbose);
  OM->SetBlockSize(blockSize);
  outputManagers[manager_name] = OM;
}


void Sample::SetLogLinParameters(const std::string& manager_name, unsigned long int base, unsigned long int maxInterval, long int user_maxIndex) {
  std::map<std::string, LogLinOutputManager*>::iterator OM = outputManagers.find(manager_name);
  OM->second->SetLogLinParams(base, maxInterval);
  OM->second->SetUserMaxIndex(user_maxIndex);
  if(base == maxInterval || user_maxIndex >= 0)
    // linear, or logarithmic and restricting to first few...
    OM->second->SetDuplicateBlockEnds(false);
  else
    OM->second->SetDuplicateBlockEnds(true);
  managers_initialized = false;
}

void Sample::InitializeOutputManagers(unsigned long int timeStepIndex) {
  std::map<std::string, LogLinOutputManager*>::iterator OM_Iter;
  bool create_dir = true;
  for(OM_Iter = outputManagers.begin(); OM_Iter != outputManagers.end(); OM_Iter++) {
    OM_Iter->second->Initialize(timeStepIndex, create_dir);
    create_dir = false; // only the first one creates the directory
  }
  managers_initialized = true;
}

void Sample::NotifyOutputManagers(unsigned long int timeStepIndex) {
  if(!managers_initialized)
    InitializeOutputManagers(timeStepIndex);
  
  std::map<std::string, LogLinOutputManager*>::iterator OM_Iter;
  for(OM_Iter = outputManagers.begin(); OM_Iter != outputManagers.end(); OM_Iter++)
    OM_Iter->second->Update(timeStepIndex);
}

void Sample::TerminateOutputManagers() {
  if(!managers_initialized)
    std::cerr << "Warning: TerminateOutputManagers called although they have not been initialized" << std::endl;
  
  std::map<std::string, LogLinOutputManager*>::iterator OM_Iter;
  for(OM_Iter = outputManagers.begin(); OM_Iter != outputManagers.end(); OM_Iter++)
    OM_Iter->second->Terminate();
}

void Sample::ReadRestartConf( int restartBlock, unsigned int numDigitsBlkIdx ) {
  std::ostringstream restartFilePath;
  restartFilePath << trajectoryDir << "/restart" << std::setfill('0') 
		  << std::setw(numDigitsBlkIdx) << std::setprecision(numDigitsBlkIdx)  
		  << restartBlock << ".xyz.gz";
  
  ReadConf(restartFilePath.str(), true);
}

void Sample::ReadConf( const std::string& filename, bool init_itg ) {
  ConfigurationWriterReader reader;
  reader.Read(this, filename);
  if(itg && init_itg && reader.metaData.IntegratorFound()) {
    if(reader.metaData.GetIO_Format() == 1) {
       IntegratorNVT* testItg = 0;
       testItg = dynamic_cast<IntegratorNVT*>(itg);
       if(testItg) {
	 if(verbose)
	   std::cout << "[Info] (Sample::ReadConf): Initializing IntegratorNVT thermostat state from parameter in file " << filename << "; thermostat state:" <<reader.metaData.GetThermostatPs() << std::endl;
	 testItg->SetThermostatState(reader.metaData.GetThermostatPs());
       }
    }
    else
      itg->InitializeFromInfoString(reader.metaData.GetIntegratorInfoStr(), verbose);
     
  } // if(itg ...)
}

void Sample::WriteConf(const std::string& filename, const std::string& mode) {
  restartWriter->Write(this, filename, mode);
}

//////////////////////////////////////////////
// Simulation modification. !! Adjust for topology file. !!
//////////////////////////////////////////////

// Calculate the CM of a molecule. The first particle defines the molecule.
float4 Sample::CalculateCenterOfMass( float4* position, float4* velocity, int moleculeIndex ){
  RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(simulationBox);
  
  float moleculeMass = 1.f / velocity[moleculeIndex].w;
  float4 firstParticle = position[moleculeIndex];
  float4 CM = { moleculeMass * firstParticle.x, moleculeMass * firstParticle.y, moleculeMass * firstParticle.z, 0 }; 
  
  for(int i = moleculeIndex+1; i < moleculeIndex + 2; i++){ 
    float particleMass = 1.f / velocity[i].w;
    
    float4 difference = testRSB->calculateDistance( position[i], firstParticle, testRSB->GetHostPointer() );
    float3 nextParticle = { firstParticle.x + difference.x, firstParticle.y + difference.y, firstParticle.z + difference.z };
    
    CM.x += particleMass * nextParticle.x;
    CM.y += particleMass * nextParticle.y;
    CM.z += particleMass * nextParticle.z;
    moleculeMass += particleMass; 
  }
  
  CM.x /= moleculeMass;
  CM.y /= moleculeMass;
  CM.z /= moleculeMass;
  CM.w = ( 1.0 / moleculeMass );
  
  return CM;
}


// Simple particle scaling
void Sample::IsotropicScaleSystem( float Rscal ){

  particleData.IsotropicScalePositions( Rscal );
  simulationBox->ScaleBox( Rscal );  
}

void Sample::IsotropicScaleSystemCM( float Rscal ){
  if(!moleculeData)
    throw RUMD_Error("Sample", __func__, "No molecule data present");
  moleculeData->IsotropicScaleCM( Rscal );
}

void Sample::AnisotropicScaleSystem( float Rscal, unsigned dir ){

  particleData.AnisotropicScalePositions( Rscal, dir );
  simulationBox->ScaleBoxDirection( Rscal, dir );  
}

void Sample::AffinelyShearSystem( float shear_strain ) {
  LeesEdwardsSimulationBox* testLESB = dynamic_cast<LeesEdwardsSimulationBox*>(simulationBox);
  if(!testLESB)
    throw RUMD_Error("Sample",__func__,"Require a LeesEdwardsSimulationBox");
  particleData.AffinelyShearPositions( shear_strain);
  testLESB->IncrementBoxStrain(shear_strain);
}

void Sample::ScaleVelocities( float factor ) {
  
  particleData.ScaleVelocities( factor);
}
					     

//////////////////////////////////////////////
// Copy Function
//////////////////////////////////////////////

void Sample::CopySimulationDataToDevice() const {
  particleData.CopyConfToDevice();
}

void Sample::CopySimulationDataFromDevice() const{
  particleData.CopyConfFromDevice();
}

//////////////////////////////////////////////
// Sorting of particles on device.
//////////////////////////////////////////////

__global__  void calc_sort_key( unsigned int num_part, float4* r, float *L, unsigned int Nx,  unsigned int Ny,  unsigned int Nz,  float* key){
  
  if( MyGP < num_part ){
    float4 My_r = r[MyGP];
   
    // Calculate key to be sorted after
    float iy = floor((My_r.y/L[2]/(1.+1.e-6)+0.5f)*Ny);
    float iz = floor((My_r.z/L[3]/(1.+1.e-6)+0.5f)*Nz);
    
    key[MyGP] = (iy + iz*Ny)*L[1] + My_r.x;
  }
}


__global__ void invert_index_mapping( unsigned* old_index, unsigned* new_index, unsigned numParticles ) {
  if( MyGP < numParticles ){
    unsigned oldIndex = old_index[MyGP];
    new_index[oldIndex] = MyGP;
  }
}


void Sample::SortParticles(){
  // Do spatial sorting of particles
  unsigned int numParticles = particleData.GetNumberOfParticles();

  BaseRectangularSimulationBox* testBRSB = dynamic_cast<BaseRectangularSimulationBox*>(simulationBox);
  if(!testBRSB) 
    throw RUMD_Error("Sample", "SortParticles", "Can only do spatial sorting with derivatives of BaseRectangularSimulationBox");

  float4 simBox = testBRSB->GetSimulationBox();

  unsigned int Ny = 1, Nz = 1;
  float Lx = simBox.x, Ly = simBox.y, Lz = simBox.z, V = simBox.w;

  if(sortingScheme == SORT_XY) {
    float alpha = pow(GetNumberOfParticles()/particles_per_block/Lx/Ly,0.5);
    Ny = (unsigned int) ( ceil(alpha*Ly) -1);
  }
  else if (sortingScheme == SORT_XYZ) {
    float alpha = pow(GetNumberOfParticles()/particles_per_block/V,1./3);
    Ny = (unsigned int) ( ceil(alpha*Ly) - 1);
    Nz = (unsigned int) ( ceil(alpha*Lz) - 1);
  }

  // generate key vector 
  thrust::device_vector<float> thrust_key(numParticles);
  float* raw_key_ptr = raw_pointer_cast(&thrust_key[0]);
  
  // Fill key vector.
  int block_size = 256;
  int grid_size = (numParticles + block_size - 1) / block_size;
  
  calc_sort_key<<< grid_size, block_size >>>(numParticles, particleData.d_r, testBRSB->GetDevicePointer(), 1, Ny, Nz, raw_key_ptr);

  thrust::device_ptr<float> thrust_key_ptr = thrust_key.data();
  SortParticlesByKey(thrust_key_ptr);

}


void Sample::UpdateAfterSorting(thrust::device_vector<unsigned int> &thrust_old_index) {
  
  // Now to re-order the various arrays. This is done by functions called
  // UpdateAfterSorting. Right now the interface is not uniform:
  // Some of the UpdateAfterSorting functions need an array which gives the 
  // old indices from the new, some the other way around
  // For potentials we provide both, since some need one, others need the other
  // also it varies whether the argument is a raw pointer, a thrust device
  // pointer, or a reference to a thrust device vector...

  unsigned int* d_raw_ptr_old_index = raw_pointer_cast(&thrust_old_index[0]);
  thrust::device_vector<unsigned int> thrust_new_index(GetNumberOfParticles());
  unsigned int* d_raw_ptr_new_index = raw_pointer_cast(&thrust_new_index[0]);
  invert_index_mapping<<< kPlan.grid, kPlan.threads.x >>>( d_raw_ptr_old_index, d_raw_ptr_new_index, GetNumberOfParticles() );

  // Update data arrays in ParticleData
  particleData.UpdateAfterSorting(thrust_old_index);

  // Update data structure in MoleculeData if present
  if(moleculeData)
    moleculeData->UpdateAfterSorting(d_raw_ptr_old_index, d_raw_ptr_new_index);

  // Trigger update in potentials
  std::vector<Potential*>::iterator potIter;
  for(potIter = potentialList.begin(); potIter != potentialList.end(); potIter++)
    (*potIter)->UpdateAfterSorting(d_raw_ptr_old_index, d_raw_ptr_new_index);

  // Trigger update in any (currently only one!) integrator.
  if(GetIntegrator())
    GetIntegrator()->UpdateAfterSorting(d_raw_ptr_old_index);
}
