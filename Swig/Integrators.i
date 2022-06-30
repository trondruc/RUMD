%{
#include "rumd/Integrator.h"
#include "rumd/IntegratorNVT.h"
#include "rumd/IntegratorNPTAtomic.h"
#include "rumd/IntegratorMMC.h"
#include "rumd/IntegratorNVU.h"
#include "rumd/IntegratorIHS.h"
#include "rumd/IntegratorSLLOD.h"
#include "rumd/IntegratorMolecularSLLOD.h"
#include "rumd/IntegratorNPTLangevin.h"
  %}


%include "rumd/Integrator.h"

%rename (IntegratorNVE) IntegratorNVT(float timeStep);

%include "rumd/IntegratorNVT.h"

%include "rumd/IntegratorNPTAtomic.h"

%include "rumd/IntegratorMMC.h"

%include "rumd/IntegratorNVU.h"

%include "rumd/IntegratorIHS.h"

%feature("autodoc","Integrates the SLLOD equations of motion for Couette-type shearing in the x-direction proportional to y-coordinate. Uses the operator-splitting algorithm of Pan et al. Requires LeesEdwardsSimulationBox to be attached to sample. Strain rate should be at least of order 10^-5, depending on time step and box size.\n") IntegratorSLLOD; 

%include "rumd/IntegratorSLLOD.h"

%include "rumd/IntegratorMolecularSLLOD.h"


%include "rumd/IntegratorNPTLangevin.h"

%pythoncode %{
  Off = IntegratorNPTLangevin.OFF
  Iso = IntegratorNPTLangevin.ISO
  Aniso = IntegratorNPTLangevin.ANISO
%}


