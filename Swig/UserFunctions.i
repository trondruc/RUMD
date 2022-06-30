%{
#include "rumd/UserFunctions.h"
#include "rumd/StressAutocorrelationFunction.h"
  %}

void CalculateMolecularStress(Sample* M);

%include "rumd/StressAutocorrelationFunction.h"
