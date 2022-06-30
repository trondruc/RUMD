%{
#include "rumd/HydrodynamicCorrelations.h"
  %}

class HydrodynamicCorrelations {
 public:

  HydrodynamicCorrelations(Sample* sample, unsigned lvecInput, float Dt, unsigned nwave);

  ~HydrodynamicCorrelations();
  
  void Compute(Sample* );
  
};

