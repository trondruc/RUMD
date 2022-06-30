#include "rumd/Potential.h"
#include "rumd/Sample.h"
#include "rumd/ParticleData.h"


Potential::Potential() :
  sample(0),
  particleData(0),
  simBox(0),
  verbose(true),
  ID_String("pot")
{
}

void Potential::SetSample(Sample* sample)
{
  this->sample = sample;
  simBox = sample->GetSimulationBox();
  particleData = sample->GetParticleData();
  kp = sample->GetKernelPlan();  
  Initialize();
}
