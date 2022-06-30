#include "rumd/AlternatePotentialCalculator.h"
#include "rumd/Sample.h"
#include "rumd/Potential.h"

AlternatePotentialCalculator::AlternatePotentialCalculator(Sample* S, Potential* alt_pot) : mainSample(S), myPotential(alt_pot) {

  mySample = new Sample(*mainSample);
  mySample->SetVerbose(false);
  mySample->SetPotential(myPotential);
}


AlternatePotentialCalculator::~AlternatePotentialCalculator() {
  delete mySample;
}

void AlternatePotentialCalculator::GetDataInfo(std::map<std::string, bool> &active, std::map<std::string, std::string> &columnIDs) {
  active[myPotential->GetID_String()] = true;
  columnIDs[myPotential->GetID_String()] = myPotential->GetID_String();

  active[myPotential->GetID_String() + "_W"] = true;
  columnIDs[myPotential->GetID_String() + "_W"] = myPotential->GetID_String() + "_W";
}

void AlternatePotentialCalculator::RemoveDataInfo(std::map<std::string, bool> &active, std::map<std::string, std::string> &columnIDs) {
  active.erase(myPotential->GetID_String());
  columnIDs.erase(myPotential->GetID_String());

  active.erase(myPotential->GetID_String() + "_W");
  columnIDs.erase(myPotential->GetID_String() + "_W");
}

void AlternatePotentialCalculator::GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active) {

  if(active[myPotential->GetID_String()] ||
     active[myPotential->GetID_String() + "_W"]) {
    (*mySample) = (*mainSample); // copies the data
    mySample->CalcF();
  }

  if(active[myPotential->GetID_String()])
    dataValues[myPotential->GetID_String()] = mySample->GetPotentialEnergy()/mySample->GetNumberOfParticles();

  if(active[myPotential->GetID_String() + "_W"])
    dataValues[myPotential->GetID_String() + "_W"] = mySample->GetVirial()/mySample->GetNumberOfParticles();
  
}
