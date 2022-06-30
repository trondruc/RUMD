

/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/AnglePotential.h"
#include "rumd/Sample.h"
#include "rumd/MoleculeData.h"
#include "rumd/rumd_algorithms.h"
#include "rumd/PairPotential.h"

__global__ void AngleResetForces( float4 *f, float4 *w ){
  float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
  f[blockIdx.x] = zero;
  w[blockIdx.x] = zero;
}

AngleCosSq::AngleCosSq() : AnglePotential() {
  ID_String ="AngleCosSq";
}


void AnglePotential::SetParams(unsigned angle_type, float theta0, float ktheta) {
  std::vector<float> prms;
  prms.push_back(theta0);
  prms.push_back(ktheta);
  
  angle_params[angle_type] = prms;

  if(sample && sample->GetMoleculeData())
    CopyParamsToGPU();
}


void AnglePotential::Initialize() {
  MoleculeData* moleculeData = sample->GetMoleculeData();

  if(!moleculeData)
    throw RUMD_Error("AngleCosSq", __func__, "No molecular data available, call ReadMoleculeData");

  CopyParamsToGPU();
}


void AnglePotential::CopyParamsToGPU() {
  MoleculeData* moleculeData = sample->GetMoleculeData();
  
  std::map<unsigned, std::vector<float> >::iterator params_it;
  for(params_it = angle_params.begin(); params_it != angle_params.end(); params_it++) {
    unsigned a_type = params_it->first;
    std::vector<float> a_params = params_it->second;
    moleculeData->SetAngleParams(a_type, a_params[0], a_params[1]);
  }
}


void AnglePotential::SetExclusions(PairPotential* non_bond_pot) {
  if(exclude_angle)
    sample->GetMoleculeData()->SetExclusionAngle(non_bond_pot);
}


void AngleCosSq::CalcF(bool initialize, bool __attribute__((unused))calc_stresses){

  unsigned int nParticles = particleData->GetNumberOfParticles();

  if ( initialize )
    AngleResetForces<<<nParticles,1>>>(particleData->d_f, particleData->d_w);
  
  RectangularSimulationBox* RSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  LeesEdwardsSimulationBox* LESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);

  MoleculeData* M = sample->GetMoleculeData();
  unsigned num_angles = M->GetNumberOfAngles();
  unsigned num_blocks = num_angles/num_threads + 1;
  
  if ( LESB )
    CosSq<1><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_alist, M->d_aplist,
	M->d_epot_angle, M->d_angles,
	num_angles, LESB, LESB->GetDevicePointer() );
  else if ( RSB )
    CosSq<1><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_alist, M->d_aplist,
	M->d_epot_angle, M->d_angles,
	num_angles, RSB, RSB->GetDevicePointer() );
  else
    throw RUMD_Error("AngleCosSq","CalcF","unknown simulation box");
  
}

double AngleCosSq::GetPotentialEnergy(){

  RectangularSimulationBox* RSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  LeesEdwardsSimulationBox* LESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);

  MoleculeData* M = sample->GetMoleculeData();
  unsigned num_angles = M->GetNumberOfAngles();
  unsigned num_blocks = num_angles/num_threads + 1;
  
  if ( LESB )
    CosSq<2><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_alist, M->d_aplist,
	M->d_epot_angle, M->d_angles,
	num_angles, LESB, LESB->GetDevicePointer() );
  else if ( RSB )
    CosSq<2><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_alist, M->d_aplist,
	M->d_epot_angle, M->d_angles,
	num_angles, RSB, RSB->GetDevicePointer() );
  else
    throw RUMD_Error("angleCosSq","GetPotentialEnergy","unknown simulation box");
  
  // copy to host
  size_t nbytes = num_angles*sizeof(float);
  cudaMemcpy(M->h_epot_angle, M->d_epot_angle, nbytes, cudaMemcpyDeviceToHost);
  
  // sum over angles
  double epot = 0.0;
  for ( unsigned n=0; n<num_angles; n++ )
    epot += (double)M->h_epot_angle[n];
    
  return epot;
}

template <int energy, class Simbox> 
__global__ void CosSq( float4 *r, float4 *f,
		       uint4 *alist, float2 *parameter,
		       float *d_epot_angle, float *d_angles,
		       unsigned num_angles, Simbox *simbox, float *simboxpointer ){

  // First approach is to let one thread work on an angle
  float4 dr1, dr2;
  float f1, f2;

  unsigned angle_index = blockIdx.x*blockDim.x + threadIdx.x;

  if ( angle_index < num_angles ){

    unsigned a = alist[angle_index].x;
    unsigned b = alist[angle_index].y;
    unsigned c = alist[angle_index].z;
    unsigned angle_type = alist[angle_index].w;
    
    float theta0 = parameter[angle_type].x;
    float ktheta = parameter[angle_type].y;

    const float cCon = cos((float) M_PI - theta0);

    // Load the simulation box in local memory to avoid bank conflicts. 
    float array[simulationBoxSize];
    simbox->loadMemory(array, simboxpointer);
  
    // Calculate stuff
    dr1 = simbox->calculateDistance(r[b], r[a], array);
    dr2 = simbox->calculateDistance(r[c], r[b], array);
  
    float c11 = dr1.w;
    float c12 = dr1.x*dr2.x + dr1.y*dr2.y + dr1.z*dr2.z;
    float c22 = dr2.w;

    float icD = 1.f/sqrtf(c11*c22);
    float cc = c12*icD;  

    d_angles[angle_index] = (float) M_PI - acos(cc); 
    float epot = 0.5f*ktheta*(cc-cCon)*(cc-cCon);

    if ( energy == 2 ){
      d_epot_angle[angle_index] = epot;
    }
    else if ( energy == 1) {

      atomicFloatAdd(&(f[a].w), epot);
 
      float ff = -ktheta*(cc - cCon);
      float c1 = c12/c11;
      float c2 = c12/c22;
    
      f1 = ff*(c1*dr1.x - dr2.x)*icD;
      f2 = ff*(dr1.x - c2*dr2.x)*icD;


      atomicFloatAdd(&(f[a].x), f1);
      atomicFloatAdd(&(f[b].x), -f1-f2);
      atomicFloatAdd(&(f[c].x), f2);
    
    
      f1 = ff*(c1*dr1.y - dr2.y)*icD;
      f2 = ff*(dr1.y - c2*dr2.y)*icD;

      atomicFloatAdd(&(f[a].y), f1);
      atomicFloatAdd(&(f[b].y), -f1-f2);
      atomicFloatAdd(&(f[c].y), f2);
    
      f1 = ff*(c1*dr1.z - dr2.z)*icD;
      f2 = ff*(dr1.z - c2*dr2.z)*icD;
      
      atomicFloatAdd(&(f[a].z), f1);
      atomicFloatAdd(&(f[b].z), -f1-f2);
      atomicFloatAdd(&(f[c].z), f2);
    }
  }
}


AngleSq::AngleSq() : AnglePotential() {
  ID_String = "AngleSq";
}

void AngleSq::CalcF(bool initialize, bool __attribute__((unused))calc_stresses){

  unsigned int nParticles = particleData->GetNumberOfParticles();

  if ( initialize )
    AngleResetForces<<<nParticles,1>>>(particleData->d_f, particleData->d_w);
  
  RectangularSimulationBox* RSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  LeesEdwardsSimulationBox* LESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);

  MoleculeData* M = sample->GetMoleculeData();
  unsigned num_angles = M->GetNumberOfAngles();
  unsigned num_blocks = num_angles/num_threads + 1;
  

  if ( LESB )
    SquaredAngle<1><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_alist, M->d_aplist,
	M->d_epot_angle, M->d_angles,
	num_angles, LESB, LESB->GetDevicePointer() );
  else if ( RSB )
    SquaredAngle<1><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_alist, M->d_aplist,
	M->d_epot_angle, M->d_angles,
	num_angles, RSB, RSB->GetDevicePointer() );
  else
    throw RUMD_Error("AngleSq","CalcF","unknown simulation box");
  
}

double AngleSq::GetPotentialEnergy(){

  RectangularSimulationBox* RSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  LeesEdwardsSimulationBox* LESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);

  MoleculeData* M = sample->GetMoleculeData();
  unsigned num_angles = M->GetNumberOfAngles();
  unsigned num_blocks = num_angles/num_threads + 1;
  
  if ( LESB )
    SquaredAngle<2><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_alist, M->d_aplist,
	M->d_epot_angle, M->d_angles,
	num_angles, LESB, LESB->GetDevicePointer() );
  else if ( RSB )
    SquaredAngle<2><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_alist, M->d_aplist,
	M->d_epot_angle, M->d_angles,
	num_angles, RSB, RSB->GetDevicePointer() );
  else
    throw RUMD_Error("angleCosSq","GetPotentialEnergy","unknown simulation box");
  
  // copy to host
  size_t nbytes = num_angles*sizeof(float);
  cudaMemcpy(M->h_epot_angle, M->d_epot_angle, nbytes, cudaMemcpyDeviceToHost);
  
  // sum over angles
  double epot = 0.0;
  for ( unsigned n=0; n<num_angles; n++ )
    epot += (double)M->h_epot_angle[n];
    
  return epot;
}

template <int energy, class Simbox> 
__global__ void SquaredAngle( float4 *r, float4 *f,
		       uint4 *alist, float2 *parameter,
		       float *d_epot_angle, float *d_angles,
		       unsigned num_angles, Simbox *simbox, float *simboxpointer ){

  // First approach is to let one thread work on an angle
  float4 dr1, dr2;
  float f1, f2;

  unsigned angle_index = blockIdx.x*blockDim.x + threadIdx.x;

  if ( angle_index < num_angles ){

    unsigned a = alist[angle_index].x;
    unsigned b = alist[angle_index].y;
    unsigned c = alist[angle_index].z;
    unsigned angle_type = alist[angle_index].w;
    
    float theta0 = parameter[angle_type].x;
    float ktheta = parameter[angle_type].y;

    //const float cCon = cos((float) M_PI - theta0);

    // Load the simulation box in local memory to avoid bank conflicts. 
    float array[simulationBoxSize];
    simbox->loadMemory(array, simboxpointer);
  
    // Calculate stuff
    dr1 = simbox->calculateDistance(r[b], r[a], array);
    dr2 = simbox->calculateDistance(r[c], r[b], array);
  
    float c11 = dr1.w;
    float c12 = dr1.x*dr2.x + dr1.y*dr2.y + dr1.z*dr2.z;
    float c22 = dr2.w;

    float icD = 1.f/sqrtf(c11*c22);
    float cc = c12*icD;
    if (cc > 1.0) cc = 1.0;
    if (cc < -1.0) cc = -1.0;
    float theta =(float) M_PI - acos(cc);

    //d_angles[angle_index] = (float) M_PI - acos(cc); 
    //float epot = 0.5f*ktheta*(cc-cCon)*(cc-cCon);
    d_angles[angle_index] = theta;
    float epot = 0.5f*ktheta*(theta-theta0)*(theta-theta0);

    if ( energy == 2 ){
      d_epot_angle[angle_index] = epot; 
    }
    else if ( energy == 1) {

      atomicFloatAdd(&(f[a].w), epot);

      //float ff = -ktheta*(theta-theta0) / (sin(theta));
      // better put in some regularization to avoid dividing by zero
      float ff = -ktheta*(theta-theta0) / (1.0e-6+sin(theta));

      float c1 = c12/c11;
      float c2 = c12/c22;
      //printf("%e %e %e %e %e %e\n", cc, c11, c22, c12, theta, ff);
    
      f1 = ff*(c1*dr1.x - dr2.x)*icD;
      f2 = ff*(dr1.x - c2*dr2.x)*icD;

      atomicFloatAdd(&(f[a].x), f1);
      atomicFloatAdd(&(f[b].x), -f1-f2);
      atomicFloatAdd(&(f[c].x), f2);
    
    
      f1 = ff*(c1*dr1.y - dr2.y)*icD;
      f2 = ff*(dr1.y - c2*dr2.y)*icD;

      atomicFloatAdd(&(f[a].y), f1);
      atomicFloatAdd(&(f[b].y), -f1-f2);
      atomicFloatAdd(&(f[c].y), f2);
    
      f1 = ff*(c1*dr1.z - dr2.z)*icD;
      f2 = ff*(dr1.z - c2*dr2.z)*icD;

      atomicFloatAdd(&(f[a].z), f1);
      atomicFloatAdd(&(f[b].z), -f1-f2);
      atomicFloatAdd(&(f[c].z), f2);
    }
  }
}
