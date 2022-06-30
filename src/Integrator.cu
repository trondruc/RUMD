#include "rumd/Integrator.h"


double Integrator::GetKineticEnergy(bool copy) const {
  if(!S || !P->GetNumberOfParticles())
    throw( RUMD_Error("Integrator", __func__,
		      "Sample has no particles" ) );

  if(copy)
    P->CopyVelFromDevice();
  double sum = 0.;
  for(unsigned int i=0; i < P->GetNumberOfParticles(); i++){
    double mass = 1.f / P->h_v[i].w;
    double v2 = P->h_v[i].x * P->h_v[i].x + P->h_v[i].y * P->h_v[i].y + P->h_v[i].z * P->h_v[i].z;
    sum += mass * v2;
  }
  return 0.5*sum;
}




// Return the system total momentum, P.
double4 Integrator::GetTotalMomentum(bool copy) const {
  if(!S || !P->GetNumberOfParticles())
    throw( RUMD_Error("IntegratorNVT", "GetTotalMomentum",
		      "Sample has no particles" ) );
  
  if(copy)
    P->CopyVelFromDevice();
  double4 totalMomentum = { 0, 0, 0, 0 };
  for(unsigned int i=0; i < P->GetNumberOfParticles(); i++){
    double mass = 1.f / P->h_v[i].w;
    totalMomentum.x += mass * P->h_v[i].x;
    totalMomentum.y += mass * P->h_v[i].y;
    totalMomentum.z += mass * P->h_v[i].z;
    totalMomentum.w += mass; // Total mass.
  }
  return totalMomentum;
}
