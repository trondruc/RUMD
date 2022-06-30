#include "rumd/IHS_OutputManager.h"
#include "rumd/Sample.h"
#include "rumd/Potential.h"
#include "rumd/IntegratorIHS.h"
#include "rumd/ConfigurationWriterReader.h"

IHS_OutputManager::IHS_OutputManager(Sample* S, float timeStep) : LogLinOutputManager(S, S->GetOutputDirectory(), "ihs"), max_num_iter(1000), num_iter_inner(50), stoppingCriterion(1.e-4f), gz_ihs_energyFile(0) {
  ihsSample = new Sample(*sample);
  ihsSample->SetVerbose(false);
  itg = new IntegratorIHS(timeStep);
  options["writeIHS_Config"] = true; 

  std::vector<Potential*>* potentialList = sample->GetPotentials();
  std::vector<Potential*>::iterator potIter;
  for(potIter = potentialList->begin(); potIter != potentialList->end(); potIter++) {
    Potential* pot_copy = (*potIter)->Copy();
    ihs_potentialList.push_back( pot_copy ); // so can delete them later
    ihsSample->AddPotential( pot_copy );
  }
  
  
  ihsSample->SetIntegrator(itg);
  ihsSample->SetOutputManagerActive("energies",true);
  ihsSample->SetOutputManagerMetaData("energies","potentialEnergy", true);
  ihsSample->SetOutputManagerMetaData("energies","virial", true);
  ihsSample->SetOutputManagerMetaData("energies","kineticEnergy", false);
  ihsSample->SetOutputManagerMetaData("energies","totalEnergy", false);
  ihsSample->SetOutputManagerMetaData("energies","pressure", false);

  ihsWriter = new ConfigurationWriterReader();
  ihsWriter->metaData.Set("precision", 4);
  ihsWriter->metaData.Set("images", true);
  ihsWriter->metaData.Set("velocities", false);
  ihsWriter->metaData.Set("forces", false);
  ihsWriter->metaData.Set("pot_energies", false);
  ihsWriter->metaData.Set("virials", false);
  ihsWriter->metaData.Set("logLin", true);  

  SetLogLinParams(1,0 );
  SetDuplicateBlockEnds(true);
}


IHS_OutputManager::~IHS_OutputManager() {
  delete ihsSample;
  delete itg;
  delete ihsWriter;
  
  std::vector<Potential*>::iterator potIter;
  for(potIter = ihs_potentialList.begin(); potIter != ihs_potentialList.end(); potIter++)
    delete (*potIter);
  
}

void IHS_OutputManager::SetMetaData(const std::string& key, bool on) {
    if(options.find(key) == options.end())
      throw RUMD_Error("IHS_OutputManager","SetMetaData",std::string("Unrecognized key:") + key);
    options[key] = on;
}

void IHS_OutputManager::SetMetaData(const std::string& key, int value) {
  // right now precision is the only possibility, and we force it to be
  // the same for the configurations and the energies
  // probably want to allow different precisions at some point 


  if(key != "precision")
    throw RUMD_Error("IHS_OutputManager","SetMetaData [int]",std::string("Unrecognized key:") + key);

  ihsWriter->metaData.Set("precision", value);
  ihsSample->SetOutputManagerMetaData("energies","precision", value);
}


void IHS_OutputManager::Write() {
  // need to avoid doing the minimization on the same configuration
  // twice when duplicateBlockEnds is true

  // also should calculate the max f2 on the device if possible

  (*ihsSample) = (*sample); // copies the data
  unsigned int num_total_iter = 0;

  // calc max force
  float max_force2 = 0.0f;
  float sc2 = stoppingCriterion*stoppingCriterion;
  ihsSample->CalcF();
  ihsSample->GetParticleData()->CopyForFromDevice();
  
  for(unsigned int atIdx = 0;atIdx < ihsSample->GetNumberOfParticles(); atIdx++)
    {
      float4& f = ihsSample->GetParticleData()->h_f[atIdx];
      float f2 = f.x*f.x + f.y*f.y + f.z*f.z;
      if( f2 > max_force2 )
	max_force2 = f2;
    }
  while (max_force2 > sc2 && num_total_iter < max_num_iter) {
    for(unsigned  int j=0; j < num_iter_inner; j++ ){
      itg->Integrate();
      ihsSample->CalcF();
    }
    ihsSample->GetParticleData()->CopyForFromDevice();
    max_force2 = 0.0f;
    for(unsigned int atIdx = 0; atIdx < ihsSample->GetNumberOfParticles(); atIdx++)
      {
	float4& f = ihsSample->GetParticleData()->h_f[atIdx];
	float f2 = f.x*f.x + f.y*f.y + f.z*f.z;
	if( f2 > max_force2 )
	  max_force2 = f2;
      }    
    num_total_iter += num_iter_inner;
  }
  
  if(options["writeIHS_Config"]) {
    std::ostringstream conf_filename;
    conf_filename << outputDirectory << "/" << baseFilename << std::setfill('0') << std::setw(numFilenameDigits) << std::setprecision(numFilenameDigits) << logLin.block << ".xyz";
    
    // set the logLin details in writer->metaData (this would not necessarily make sense for all Configuration writers)  
    
    ihsWriter->metaData.SetLogLin(logLin);
    ihsWriter->SetVerbose(verbose);
    
    if(logLin.index > 0) 
      ihsWriter->Write(ihsSample, conf_filename.str(), "a");
    else
      ihsWriter->Write(ihsSample, conf_filename.str(), "w");
  }
  

 if(logLin.index == 0) // (start a new file)
    {
      if(gz_ihs_energyFile) gzclose(gz_ihs_energyFile);
      std::ostringstream ihs_energyFilename;
      ihs_energyFilename << outputDirectory << "/ihsEnergies" << std::setfill('0') << std::setw(numFilenameDigits) << std::setprecision(numFilenameDigits) << logLin.block << ".dat.gz";
      gz_ihs_energyFile = gzopen(ihs_energyFilename.str().c_str(),"w");
      
      // comment line at top of new file
      std::map<std::string, bool>::const_iterator it;
      float dt = 0.;
      if(sample->GetIntegrator())
	dt = sample->GetIntegrator()->GetTimeStep();
      
      
      gzprintf(gz_ihs_energyFile, "# N=%d",ihsSample->GetNumberOfParticles());

      if(logLin.base != logLin.maxInterval)
	gzprintf(gz_ihs_energyFile," timeStepIndex=%u logLin=%u,%u,%u,%u,%u", logLin.nextTimeStep, logLin.block, logLin.base, logLin.index, logLin.maxIndex, logLin.maxInterval);
      else
	gzprintf(gz_ihs_energyFile," Dt=%f", dt*logLin.base);

      gzprintf(gz_ihs_energyFile," columns=pe,virial,max_force,num_iter\n");
    } // end if(logLin.index == 0)  
 
  int precision = 6;
  gzprintf(gz_ihs_energyFile, "%.*g ", precision, ihsSample->GetPotentialEnergy()/ihsSample->GetNumberOfParticles());
  gzprintf(gz_ihs_energyFile, "%.*g ", precision, ihsSample->GetVirial()/ihsSample->GetNumberOfParticles());
  gzprintf(gz_ihs_energyFile, "%.*g ", precision, sqrt(max_force2));
  gzprintf(gz_ihs_energyFile, "%u ", num_total_iter);

  gzprintf(gz_ihs_energyFile, "\n");  

}

