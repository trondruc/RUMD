
%{
#include "rumd/ExternalCalculator.h"
#include "rumd/AlternatePotentialCalculator.h"
#include "rumd/HypervirialCalculator.h"
#include "rumd/SquaredDisplacementCalculator.h"
  %}


%nodefaultctor ExternalCalculator; // disable generation of wrapper for default constructor

class ExternalCalculator
{};

class AlternatePotentialCalculator : public ExternalCalculator {
 public:
  AlternatePotentialCalculator(Sample* sample, Potential* alt_pot);
};

class HypervirialCalculator : public ExternalCalculator {
 public:
  HypervirialCalculator(Sample* sample, float delta_ln_rho);
  void SetDeltaLogRho(float delta_ln_rho);
};

class SquaredDisplacementCalculator : public ExternalCalculator {
 public:
  SquaredDisplacementCalculator(Sample* sample, Sample* sample_ref=0);
};

// The following is not derived from ExternalCalculator but serves a similar puupose

%include "HydrodynamicCorrelations.i"
