#include "rumd/HypervirialCalculator.h"
#include "rumd/Sample.h"
#include "rumd/Potential.h"
#include "rumd/PairPotential.h"


HypervirialCalculator::HypervirialCalculator(Sample* S, double set_delta_ln_rho) : mainSample(S), delta_ln_rho(set_delta_ln_rho) {

  if(mainSample->GetMoleculeData())
    throw RUMD_Error("HypervirialCalculator","HypervirialCalculator","HypervirialCalculator not implemented for molecular systems yet");
  
  // The  copy doesn't take account of molecule data, so the calculation
  // of contributions from bonding terms will fail. There needs to be a copy
  // of the molecule data structure. Constraints will be harder!

  mySample = new Sample(*mainSample);
  mySample->SetVerbose(false);

  // Need to copy the potential(s) from the given sample rather than pass them
  // in 

  std::vector<Potential*>* potentialList = mainSample->GetPotentials();
  std::vector<Potential*>::iterator potIter;
  for(potIter = potentialList->begin(); potIter != potentialList->end(); potIter++) {
    Potential* pot_copy = (*potIter)->Copy();
    PairPotential* test_pair = dynamic_cast<PairPotential*>(pot_copy);
    if(test_pair)
      test_pair->SetNbListSkin(0.01);
    my_potentialList.push_back( pot_copy ); // so can delete them later
    mySample->AddPotential( pot_copy );
  }
}


HypervirialCalculator::~HypervirialCalculator() {
  delete mySample;

  std::vector<Potential*>::iterator potIter;
  for(potIter = my_potentialList.begin(); potIter != my_potentialList.end(); potIter++)
    delete (*potIter);
  
}

void HypervirialCalculator::GetDataInfo(std::map<std::string, bool> &active, std::map<std::string, std::string> &columnIDs) {
  active["approx_vir"] = true;
  columnIDs["approx_vir"] = "approx_vir";

  active["approx_h_vir"] = true;
  columnIDs["approx_h_vir"] = "approx_h_vir";
}

void HypervirialCalculator::RemoveDataInfo(std::map<std::string, bool> &active, std::map<std::string, std::string> &columnIDs) {
  active.erase("approx_vir");
  columnIDs.erase("approx_vir");

  active.erase("approx_h_vir");
  columnIDs.erase("approx_h_vir");
}

void HypervirialCalculator::GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active) {

  if(active["approx_vir"] || active["approx_h_vir"]) {
    (*mySample) = (*mainSample); // copies the data

    mySample->IsotropicScaleSystem( exp(-0.5* delta_ln_rho/3.));
    mySample->CalcF();
    double my_pe_plus = mySample->GetPotentialEnergy();
    double my_vir_plus = mySample->GetVirial();
    
    mySample->IsotropicScaleSystem( exp(delta_ln_rho/3.) );
    mySample->CalcF();
    double my_pe_minus = mySample->GetPotentialEnergy();
    double my_vir_minus = mySample->GetVirial();
    
    mySample->IsotropicScaleSystem( -0.5*exp(delta_ln_rho/3.) );
    
    unsigned int nParticles = mySample->GetNumberOfParticles();


    if(active["approx_vir"])
      dataValues["approx_vir"] = ( my_pe_plus - my_pe_minus )/delta_ln_rho/nParticles;
    if(active["approx_h_vir"])
      dataValues["approx_h_vir"] = ( my_vir_plus - my_vir_minus )/delta_ln_rho/nParticles;
  }
} 
