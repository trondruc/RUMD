#include "rumd/EnergiesOutputManager.h"
#include "rumd/Sample.h"
#include "rumd/Potential.h"
#include "rumd/ConstraintPotential.h"
#include "rumd/Integrator.h"

#include <algorithm>
#include <iostream>
#include <vector>

void EnergiesOutputManager::Write() {

  if(logLin.index == 0) // (start a new file)
    {
      if(gz_energyFile) gzclose(gz_energyFile);
      std::ostringstream energyFilename;
      energyFilename << outputDirectory << "/" << baseFilename << std::setfill('0') << std::setw(numFilenameDigits) << std::setprecision(numFilenameDigits) << logLin.block << ".dat.gz";
      gz_energyFile = gzopen(energyFilename.str().c_str(),"w");
      
      // comment line at top of new file
      std::map<std::string, bool>::const_iterator it;
      float dt = 0.;
      if(sample->GetIntegrator())
	dt = sample->GetIntegrator()->GetTimeStep();
      
      
      gzprintf(gz_energyFile, "# ioformat=%d N=%d",energies_ioformat,sample->GetNumberOfParticles());

      if(logLin.base != logLin.maxInterval)
	gzprintf(gz_energyFile," timeStepIndex=%u logLin=%u,%u,%u,%u,%u", logLin.nextTimeStep, logLin.block, logLin.base, logLin.index, logLin.maxIndex, logLin.maxInterval);
      else
	gzprintf(gz_energyFile," Dt=%f", dt*logLin.base);
      
      gzprintf(gz_energyFile," columns=");
      bool haveFirst = false;
      // list structure to iterate through and then print only the items which
      // are "true", for the comment line (changes the order)
      
      for(it = metaData.Start(); it != metaData.End(); it++ )
	if(it->second) {
	  if(haveFirst)
	    gzprintf(gz_energyFile, ",");
	  
	  gzprintf(gz_energyFile, "%s",metaData.GetFileString(it->first).c_str());
	  haveFirst = true;
	} // if(it->second)
	gzprintf(gz_energyFile, "\n");
	
	} // end if(logLin.index == 0)  
  WriteOneLine();
}

void EnergiesOutputManager::AddExternalCalculator(ExternalCalculator* calc) {
  externalCalculators.push_back(calc);
  calc->GetDataInfo(metaData.bool_options, metaData.fileStr);
}


void EnergiesOutputManager::RemoveExternalCalculator(ExternalCalculator* calc) {
 calc->RemoveDataInfo(metaData.bool_options, metaData.fileStr); 

 std::vector<ExternalCalculator*>::iterator found_calc = std::find(externalCalculators.begin(), externalCalculators.end(), calc);
 if(found_calc == externalCalculators.end())
   throw RUMD_Error("EnergiesOutputManager","RemoveExternalCalculator","Calculator not found");
 externalCalculators.erase(found_calc);
}

void EnergiesOutputManager::RegisterPotential(Potential* pot) {
  pot->GetDataInfo(metaData.bool_options, metaData.fileStr);
}

void EnergiesOutputManager::UnregisterPotential(Potential* pot) {
  pot->RemoveDataInfo(metaData.bool_options, metaData.fileStr);
}


void EnergiesOutputManager::RegisterIntegrator(Integrator* itg) {
  itg->GetDataInfo(metaData.bool_options, metaData.fileStr);
}

void EnergiesOutputManager::UnregisterIntegrator(Integrator* itg) {
  itg->RemoveDataInfo(metaData.bool_options, metaData.fileStr);
}



void EnergiesOutputManager::WriteOneLine(){

  std::map<std::string, float> Values;
  // calculate all standard energy-like quantities that are generally available
  CalcEnergies(Values);

  // individual potential energy contributions
  std::vector<Potential*> *potentials = sample->GetPotentials();
  unsigned int numPotentials = potentials->size();
  if(numPotentials)
    for(unsigned int pdx = 0; pdx < numPotentials; pdx++) {
      std::string potID = (*potentials)[pdx]->GetID_String();
      if(metaData.Get(potID))
	Values[potID] = (*potentials)[pdx]->GetPotentialEnergy()/sample->GetNumberOfParticles();
      // add extra data items that a particular potential may supply
      (*potentials)[pdx]->GetDataValues(Values, metaData.bool_options);
    }

  // data associated with this specific integrator
  sample->GetIntegrator()->GetDataValues(Values, metaData.bool_options);
  
  // iterate over externalCalculators to get their data-values
  for(unsigned int edx = 0; edx < externalCalculators.size(); edx++) 
    externalCalculators[edx]->GetDataValues(Values, metaData.bool_options);

  int precision = metaData.Get("precision");
  bool haveFirst = false;
  // iterate over all items and write them to the current line
  std::map<std::string, bool>::const_iterator it;
  unsigned int nItems = 0;
  for(it = metaData.Start(); it != metaData.End(); it++ )
    if(it->second) {
      if(haveFirst)
	gzprintf(gz_energyFile, " ");
      
      gzprintf(gz_energyFile, "%.*g", precision, Values[it->first]);
      haveFirst = true;
      nItems++;
    }
  if(nItems)
    // final newline
    gzprintf(gz_energyFile, "\n");
}


void EnergiesOutputManager::CalcEnergies(std::map<std::string, float>& dataValues)
{
  unsigned int degreesOfFreedom = sample->GetNumberOfDOFs();
  const Integrator* itg = sample->GetIntegrator();
  const ParticleData* particleData = sample->GetParticleData();
  unsigned int N = sample->GetNumberOfParticles();  
  float V = sample->GetSimulationBox()->GetVolume();

  bool need_stress = metaData.Get("stress_xx") || metaData.Get("stress_yy") || metaData.Get("stress_zz") || metaData.Get("stress_xy") || metaData.Get("stress_yz") || metaData.Get("stress_xz");
  bool need_ke = itg &&  (metaData.Get("kineticEnergy") || metaData.Get("temperature") || metaData.Get("pressure") || metaData.Get("totalEnergy"));  
  bool need_pe = metaData.Get("potentialEnergy") || metaData.Get("pressure") || metaData.Get("totalEnergy");
  bool need_vir = metaData.Get("virial") || metaData.Get("pressure");

  
  if(need_stress)
    sample->CalcF(true); // need to calculate before we copy
  
  if(need_ke) {
    particleData->CopyVelFromDevice(false); // async
    particleData->CopyForFromDevice(true); // sync-- now we know we have vel,for
  }

  if(need_pe && !need_ke)
    particleData->CopyForFromDevice(false);
  if(need_stress && !need_ke)
    particleData->CopyVelFromDevice(false);

  if(need_vir)
    particleData->CopyVirFromDevice(false);
  
  if(need_stress) {
    if(!need_vir)
      particleData->CopyVirFromDevice(false);
    particleData->CopyStressFromDevice(false);
  }

  double ke = 0.0, pe = 0.0, vir = 0.;
  if(need_ke) ke = itg->GetKineticEnergy(false);
  float T = ( 2.0 * ke ) / (double) degreesOfFreedom;
  
  if(need_pe && need_ke)
    // have the force arrays including pe
    pe = sample->GetPotentialEnergy(false);

  // synchronize now if any of the relevant arrays were copied
  if( (need_pe && !need_ke) || need_vir || need_stress) {
      cudaError err = cudaDeviceSynchronize();
      if( err != cudaSuccess ) 
	throw( RUMD_Error("EnergiesOutputManager","CalcEnergies",std::string("Error detected after synchronization followed device -> to host data transfer: ")+ cudaGetErrorString(err) ) );  
  }
	
  if(need_pe && !need_ke)
    // in this case only have force array after synchronization
    pe = sample->GetPotentialEnergy(false); 

  if(need_vir)
    vir = sample->GetVirial(false);

  float E = pe + ke;
  float P = ( N * T + vir ) / V;
  float H = E + P * V; 

  dataValues["kineticEnergy"] = ke / N;
  dataValues["potentialEnergy"] = pe / N;
  dataValues["totalEnergy"] = E / N;
  dataValues["virial"] = vir / N;
  dataValues["potentialVirial"] = vir / N;
  dataValues["temperature"] = T;
  dataValues["pressure"] = P; 
  dataValues["volume"] = V;
  dataValues["density"] = N/V;
  dataValues["enthalpy"] = H / N;
  
  if(need_stress) {
    std::vector<double> stress_vector(6,0);
    stress_vector = sample->GetStress(false);
    dataValues["stress_xx"] = stress_vector[0];
    dataValues["stress_yy"] = stress_vector[1];
    dataValues["stress_zz"] = stress_vector[2];
    dataValues["stress_yz"] = stress_vector[3];
    dataValues["stress_xz"] = stress_vector[4];
    dataValues["stress_xy"] = stress_vector[5];
  }
}
