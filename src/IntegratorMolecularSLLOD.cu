
/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/Sample.h"
#include "rumd/rumd_algorithms.h"
#include "rumd/RUMD_Error.h"
#include "rumd/IntegratorMolecularSLLOD.h"
#include "rumd/ParseInfoString.h"
#include "rumd/MoleculeData.h"

const std::string SLLOD_Error_Code1("There is no integrator associated with Sample or the latter has no particles");

////////////////////////////////////////////////////////////
// Constructors 
////////////////////////////////////////////////////////////


IntegratorMolecularSLLOD::IntegratorMolecularSLLOD(float timeStep, double strainRate) {
  this->timeStep = timeStep;
  this->strainRate = strainRate;
  num_molecules = 0;
  n_mol_blocks = 0;
  n_mol_threads_per_block = 32;

  numParticlesSample = 0;
  initialized = false;
  AllocateFromConstructor();
}

IntegratorMolecularSLLOD::~IntegratorMolecularSLLOD(){
  cudaFree(d_thermostatState);
  FreeArrays();
}

////////////////////////////////////////////////////////////
// Class Methods
////////////////////////////////////////////////////////////


// Integrate SLLOD dynamics using the operator-splitting method of 
// Pan et al. JCP 122, 094114 (2005)
void IntegratorMolecularSLLOD::Integrate(){
  if(!S || !(numParticlesSample))
    throw( RUMD_Error("IntegratorMolecularSLLOD","Integrate", SLLOD_Error_Code1 ) );
  
  LeesEdwardsSimulationBox* testLESB = dynamic_cast<LeesEdwardsSimulationBox*>(S->GetSimulationBox());
  if(!testLESB) throw RUMD_Error("IntegratorMolecularSLLOD","Integrate","LeesEdwardsSimulationBox is needed for this integrator");

  MoleculeData* moleculeData = S->GetMoleculeData();
  if(!moleculeData)
    throw RUMD_Error("IntegratorMolecularSLLOD",__func__,"Must set molecule data before calling integrate");
  num_molecules = moleculeData->GetNumberOfMolecules();
  n_mol_blocks = (num_molecules + n_mol_threads_per_block
		  - 1)/ n_mol_threads_per_block;

  
  if(!initialized)
    Initialize_g_factor();


  // B1 (half-step on velocities from shear)
  integrateMolSLLOD_B1<<< n_mol_blocks, n_mol_threads_per_block >>>( num_molecules, moleculeData->GetMoleculeListDevice(), moleculeData->GetMaximumMoleculeSize(), P->d_v, P->d_f, d_thermostatParameters, timeStep, strainRate, d_thermostatState);


  Update_factor1_factor2();
  // B2 (full-step on velocities from forces)
  integrateMolSLLOD_B2<<< n_mol_blocks, n_mol_threads_per_block >>>( num_molecules, moleculeData->GetMoleculeListDevice(), moleculeData->GetMaximumMoleculeSize(), P->d_v, P->d_f, timeStep, d_thermostatParameters, d_thermostatState);


  Update_g_factor();
  float wrap = testLESB->IncrementBoxStrain(strainRate*timeStep);
  P->ApplyLeesEdwardsWrapToImages(wrap);

  S->GetMoleculeData()->EvalCM(false);
  // B1 (half-step on velocities from shear)  + A (full-step on positions)
  integrateMolSLLOD_A_B1<<< n_mol_blocks, n_mol_threads_per_block >>>(num_molecules, moleculeData->GetMoleculeListDevice(), moleculeData->GetMaximumMoleculeSize(), P->d_r, P->d_v, P->d_im, testLESB, testLESB->GetDevicePointer(), d_thermostatParameters, timeStep, strainRate, d_thermostatState, moleculeData->d_cm);
  
  Update_g_factor();
}

// Private. Update the thermostat state.
void IntegratorMolecularSLLOD::Update_factor1_factor2(){
  
    sumIdenticalArrays( d_thermostatParameters, num_molecules, 1, 32 );    

    update_factor1_factor2_kernel<<<1, 1 >>>( timeStep, d_thermostatParameters, d_thermostatState);
    //update_factor1_factor2_Mol_kernel<<<1, 1 >>>( timeStep, d_thermostatParameters, d_thermostatState);
}


void IntegratorMolecularSLLOD::Update_g_factor(){

  sumIdenticalArrays( d_thermostatParameters, num_molecules, 1, 32 );
  
  update_g_kernel<<<1, 1>>>(timeStep, strainRate, d_thermostatParameters, d_thermostatState);
  //update_g_Mol_kernel<<<1, 1>>>(timeStep, strainRate, d_thermostatParameters, d_thermostatState);
}



void IntegratorMolecularSLLOD::Initialize_g_factor() {
  if(!S || !(numParticlesSample))
    throw( RUMD_Error("IntegratorMolecularSLLOD","Integrate", SLLOD_Error_Code1 ) );

  // launch a kernel to set the quantities to be added for the g_factor

  initialize_g_factor_molecules<<<  n_mol_blocks, n_mol_threads_per_block >>>(num_molecules, S->GetMoleculeData()->GetMoleculeListDevice(), S->GetMoleculeData()->GetMaximumMoleculeSize(), P->d_v, d_thermostatParameters);

  // do the sums and calculate g_factor
  Update_g_factor();
  initialized = true;
}



void IntegratorMolecularSLLOD::AllocateFromConstructor(){
  // Allocate space on the GPU for the thermostat state.
  if( cudaMalloc( (void**) &d_thermostatState, sizeof(double4) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("IntegratorMolecularSLLOD","IntegratorMolecularSLLOD","Malloc failed on d_thermostatState") );
  
  // Initialize the state on the GPU.
  double4 thermostatState = {0., 0., 0., 0.};
  cudaMemcpy( d_thermostatState, &thermostatState, sizeof(double4), cudaMemcpyHostToDevice );
  
  // Blocks until the device has completed all preceding requested tasks. 
  // Host => Device is async.
  if( cudaDeviceSynchronize() != cudaSuccess ) 
    throw( RUMD_Error("IntegratorMolecularSLLOD","IntegratorMolecularSLLOD","CudaMemcpy failed: d_thermostatState") ); 
}



// Allocates memory dependent on the N particles.
void IntegratorMolecularSLLOD::AllocateIntegratorState(){
  if(!S || !(S->GetNumberOfParticles())) // Consistency check.
    throw( RUMD_Error("IntegratorMolecularSLLOD","AllocateIntegratorState", SLLOD_Error_Code1 ) );


  unsigned int newNumParticlesS = S->GetNumberOfParticles();

  if(newNumParticlesS != numParticlesSample){
    FreeArrays();
    numParticlesSample = newNumParticlesS;
    
    // Allocate space for the summation of kinetic energy on GPU.
    if( cudaMalloc( (void**) &d_thermostatParameters, numParticlesSample * sizeof(double4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorMolecularSLLOD","AllocateIntegratorState","Malloc failed on d_thermostatParameters") );

    if( cudaMallocHost( (void**) &h_thermostatParameters, numParticlesSample * sizeof(double4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorMolecularSLLOD","AllocateIntegratorState","Malloc failed on h_thermostatParameters") );
    
    // Allocate space for the summation of momentum on GPU.
    if( cudaMalloc( (void**) &d_particleMomentum, numParticlesSample * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorMolecularSLLOD","AllocateIntegratorState","Malloc failed on d_particleMomentum") );
  }
}

// Frees memory dependent on numParticles
void IntegratorMolecularSLLOD::FreeArrays(){
  if(numParticlesSample){
    cudaFree(d_thermostatParameters);
    cudaFreeHost(h_thermostatParameters); 
    cudaFree(d_particleMomentum);
  }
}

////////////////////////////////////////////////////////////
// Set Methods 
////////////////////////////////////////////////////////////

void IntegratorMolecularSLLOD::SetTimeStep(float dt) { 
  timeStep = dt;
  initialized = false;
}

void IntegratorMolecularSLLOD::SetStrainRate(double set_strain_rate) {
  strainRate = set_strain_rate;
  initialized = false;
}

void IntegratorMolecularSLLOD::SetKineticEnergy( float set_kinetic_energy ){  
  float current_KE = GetKineticEnergy();
  if(current_KE > 0.0) {
    float scale_factor = sqrt( set_kinetic_energy / current_KE );
    rescale_velocities<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_v, scale_factor); 
      }
  else
    throw RUMD_Error("IntegratorMolecularSLLOD", "SetKineticEnergy", "Cannot rescale; KE is zero!");

  initialized = false;
}


void IntegratorMolecularSLLOD::SetMolecularKineticEnergy( float set_mol_kinetic_energy ){  
  float current_mol_KE = GetMolecularKineticEnergy();
  if(current_mol_KE > 0.0) {
    float scale_factor = sqrt( set_mol_kinetic_energy / current_mol_KE );
    rescale_velocities<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_v, scale_factor); 
      }
  else
    throw RUMD_Error("IntegratorMolecularSLLOD", __func__, "Cannot rescale; KE is zero!");

  initialized = false;
}


void IntegratorMolecularSLLOD::SetMomentumToZero(){
  if(!S || !numParticlesSample)
    throw( RUMD_Error("IntegratorMolecularSLLOD","SetMomentumToZero", SLLOD_Error_Code1 ) );
  
  calculateMomentum<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_v, d_particleMomentum );
  sumIdenticalArrays( d_particleMomentum, numParticlesSample, 1, 32 );
  zeroTotalMomentum<<<kp.grid, kp.threads>>>( numParticlesSample, P->d_v, d_particleMomentum );

}


////////////////////////////////////////////////////////////
// Get Methods 
////////////////////////////////////////////////////////////

std::string IntegratorMolecularSLLOD::GetInfoString(unsigned int precision) const{
  std::ostringstream infoStream;
  infoStream << "IntegratorMolecularSLLOD";
  infoStream << "," << std::setprecision(precision) << GetTimeStep();
  infoStream << "," << std::setprecision(precision) << GetStrainRate();
  return infoStream.str();
}

void IntegratorMolecularSLLOD::InitializeFromInfoString(const std::string& infoStr, bool verbose) {
  std::vector<float> parameterList;
  std::string className = ParseInfoString(infoStr, parameterList);
  // check typename, if no match, raise error
  if(className != "IntegratorMolecularSLLOD")
    throw RUMD_Error("IntegratorMolecularSLLOD","InitializeFromInfoString",std::string("Wrong integrator type: ")+className);
  
  if(parameterList.size() != 2)
    throw RUMD_Error("IntegratorMolecularSLLOD","InitializeFromInfoString","Wrong number of parameters in infoStr");

  if(verbose)
    std::cout << "[Info] Initializing IntegratorMolecularSLLOD. time step:" << parameterList[0] << "; strain rate:" << parameterList[1]  << std::endl;
  SetTimeStep(parameterList[0]);
  SetStrainRate(parameterList[1]);
}



double IntegratorMolecularSLLOD::GetMolecularKineticEnergy() const {
  if(!numParticlesSample)
    throw( RUMD_Error("IntegratorMolecularSLLOD",__func__,"Sample has no particles" ) );

  // call the B1 part with zero time step, to get the current KE values
  integrateMolSLLOD_B1<<<  n_mol_blocks, n_mol_threads_per_block >>>(num_molecules,  S->GetMoleculeData()->GetMoleculeListDevice(), S->GetMoleculeData()->GetMaximumMoleculeSize(), P->d_v, P->d_f, d_thermostatParameters, 0., 0., 0);
  

  //sumIdenticalArrays( d_thermostatParameters, num_molecules, 1, 32 );

  // copy results of the sum to host
  double molecularKE = 0.;
  cudaMemcpy( h_thermostatParameters, d_thermostatParameters, num_molecules * sizeof(double4), cudaMemcpyDeviceToHost );
  for(unsigned idx = 0; idx < num_molecules; idx++)
    molecularKE += h_thermostatParameters[idx].x;

      return 0.5 * molecularKE;
}

__global__ void integrateMolSLLOD_B1(unsigned int nMolecules,
				     const int1* mlist,
				     unsigned int max_size, 
				     float4* velocity, 
				     float4* force, 
				     double4* thermostatParameters,
				     float timeStep,
				     float strainRate,
				     double4* thermostatState){
  unsigned int my_mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( my_mol_idx < nMolecules ){ 
    double3 my_velocity_CM = {0., 0., 0.};
    double3 my_mol_force = {0., 0., 0.};
    double g_factor = 1.;
    double4 localThermostatParameters; // = thermostatParameters[0];
    double strainStepHalf = strainRate*timeStep*0.5f;
    if(thermostatState)
      g_factor = thermostatState[0].x;
    /*{
      // g_factor for 0.5*timeStep
      double c1 = strainRate*localThermostatParameters.y / localThermostatParameters.x ;
      double c2 = strainRate*strainRate * localThermostatParameters.z / localThermostatParameters.x;
      g_factor = 1./sqrt(1. - c1*timeStep  + 0.25 * c2 * timeStep*timeStep);
      //g_factor = thermostatState[0].x;
      }*/
    
    unsigned int size = mlist[my_mol_idx].x;
    unsigned offset = max_size*my_mol_idx + nMolecules;
    unsigned int i;
    float my_mol_mass = 0.f;

    for ( unsigned m=0; m<size; m++ ){
      i = mlist[offset+m].x;
      float4 this_v = velocity[i];
      float4 this_f = force[i];
      float mi = 1.0/this_v.w;
      my_velocity_CM.x += mi * this_v.x;
      my_velocity_CM.y += mi * this_v.y;
      my_velocity_CM.z += mi * this_v.z;
      my_mol_force.x += this_f.x;
      my_mol_force.y += this_f.y;
      my_mol_force.z += this_f.z;
      my_mol_mass += mi;
    }
    my_velocity_CM.x /= my_mol_mass;
    my_velocity_CM.y /= my_mol_mass;
    my_velocity_CM.z /= my_mol_mass;

    double3 delta_velocity_CM;
    delta_velocity_CM.x = ( (g_factor-1.) * my_velocity_CM.x - g_factor * strainStepHalf*my_velocity_CM.y );
    delta_velocity_CM.y = (g_factor-1.) * my_velocity_CM.y;
    delta_velocity_CM.z = (g_factor-1.) * my_velocity_CM.z;

    
    for ( unsigned m=0; m<size; m++ ){
      i = mlist[offset+m].x;
      float4 this_v = velocity[i];

      this_v.x += delta_velocity_CM.x;
      this_v.y += delta_velocity_CM.y;
      this_v.z += delta_velocity_CM.z;

      // Save the integration in global memory.
      velocity[i] = this_v;
    }
    // now update the total momentum before computing thermostatParameters
    //my_velocity_CM.x = g_factor*( my_velocity_CM.x - strainStepHalf*my_velocity_CM.y );
    //my_velocity_CM.y *= g_factor;
    //my_velocity_CM.z *= g_factor;
    my_velocity_CM.x += delta_velocity_CM.x;
    my_velocity_CM.y += delta_velocity_CM.y;
    my_velocity_CM.z += delta_velocity_CM.z;

    localThermostatParameters.x =  ( my_velocity_CM.x * my_velocity_CM.x +
				      my_velocity_CM.y * my_velocity_CM.y +
				      my_velocity_CM.z * my_velocity_CM.z) * my_mol_mass; // p^2/m 
    localThermostatParameters.y = ( my_mol_force.x * my_velocity_CM.x +
				     my_mol_force.y * my_velocity_CM.y + 
				     my_mol_force.z * my_velocity_CM.z ); // f.p/m 
    localThermostatParameters.z = ( my_mol_force.x * my_mol_force.x +
				     my_mol_force.y * my_mol_force.y + 
				     my_mol_force.z * my_mol_force.z ) / my_mol_mass; // f^2/m
    thermostatParameters[my_mol_idx] = localThermostatParameters;
 
  } // if ( my_mol_idx ... )
}



__global__ void integrateMolSLLOD_B2(unsigned int nMolecules,
				     const int1* mlist,
				     unsigned int max_size, 
				     float4* velocity, 
				     float4* force,
				     float timeStep,
				     double4* thermostatParameters,
				     double4* thermostatState){
  unsigned int my_mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( my_mol_idx < nMolecules ){ 
    double4 localThermostatParameters; // = thermostatParameters[0];
    /*double alpha = localThermostatParameters.y / localThermostatParameters.x;
    double beta = sqrt(localThermostatParameters.z / localThermostatParameters.x);
    double integrate_factor1, integrate_factor2;
    if (beta*timeStep < 1.e-8) {
      integrate_factor1 = 1.f;
      integrate_factor2 = timeStep;
    }
    else {
      double h = (alpha + beta) / (alpha - beta) ;
      double e = exp(-beta*timeStep);
      integrate_factor1 = (1.f - h) / (e - h/e);
      integrate_factor2 = (1.f + h - e - h/e)/((1.f-h)*beta);
      }*/
    double integrate_factor1 = thermostatState[0].y;
    double integrate_factor2 = thermostatState[0].z;

    double3 my_velocity_CM = {0., 0., 0.};
    double3 my_accel_CM = {0., 0., 0.};

    unsigned int size = mlist[my_mol_idx].x;
    unsigned offset = max_size*my_mol_idx + nMolecules;
    unsigned int i;
    float my_mol_mass = 0.f;

    for ( unsigned m=0; m<size; m++ ){
      i = mlist[offset+m].x;
      float4 this_v = velocity[i];
      float4 this_f = force[i];
      float mi = 1.0/this_v.w;
      my_velocity_CM.x += mi * this_v.x;
      my_velocity_CM.y += mi * this_v.y;
      my_velocity_CM.z += mi * this_v.z;
      my_accel_CM.x += this_f.x;
      my_accel_CM.y += this_f.y;
      my_accel_CM.z += this_f.z;
      my_mol_mass += mi;
    }
    my_velocity_CM.x /= my_mol_mass;
    my_velocity_CM.y /= my_mol_mass;
    my_velocity_CM.z /= my_mol_mass;
    my_accel_CM.x /= my_mol_mass;
    my_accel_CM.y /= my_mol_mass;
    my_accel_CM.z /= my_mol_mass;
    
    double3 delta_velocity_CM;
    delta_velocity_CM.x = integrate_factor1*(my_velocity_CM.x + integrate_factor2*my_accel_CM.x) - my_velocity_CM.x;
    delta_velocity_CM.y = integrate_factor1*(my_velocity_CM.y + integrate_factor2*my_accel_CM.y) - my_velocity_CM.y;
    delta_velocity_CM.z = integrate_factor1*(my_velocity_CM.z + integrate_factor2*my_accel_CM.z) - my_velocity_CM.z;
    
    my_velocity_CM.x += delta_velocity_CM.x;
    my_velocity_CM.y += delta_velocity_CM.y;
    my_velocity_CM.z += delta_velocity_CM.z;


    for ( unsigned m=0; m<size; m++ ){
      i = mlist[offset+m].x;
      float4 this_v = velocity[i];
      float4 this_f = force[i];
    
      this_v.x += delta_velocity_CM.x + (this_f.x*this_v.w - my_accel_CM.x)*timeStep;
      this_v.y += delta_velocity_CM.y + (this_f.y*this_v.w - my_accel_CM.y)*timeStep;
      this_v.z += delta_velocity_CM.z + (this_f.z*this_v.w - my_accel_CM.z)*timeStep;

    // Save the integration in global memory.
    velocity[i] = this_v;
    }


    localThermostatParameters.x = ( my_velocity_CM.x * my_velocity_CM.x +
				     my_velocity_CM.y * my_velocity_CM.y +
    				     my_velocity_CM.z * my_velocity_CM.z) * my_mol_mass; // p^2/m 
    localThermostatParameters.y = my_velocity_CM.x * my_velocity_CM.y * my_mol_mass; // px py / m (for c1)

    localThermostatParameters.z = my_velocity_CM.y * my_velocity_CM.y * my_mol_mass; // py^2 / m (for c2)
    thermostatParameters[my_mol_idx] = localThermostatParameters;


  } // if(my_mol_idx ...)

}



template <class S> __global__ void integrateMolSLLOD_A_B1(unsigned int nMolecules,
							  const int1* mlist,
							  unsigned int max_size, 
							  float4* position, 
							  float4* velocity, 
							  float4* image, 
							  S* simBox, 
							  float* simBoxPointer, 
							  double4* thermostatParameters,
							  float timeStep,
							  float strainRate,
							  double4* thermostatState,
							  float4* cm){
  //  A after B1
  unsigned int my_mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( my_mol_idx < nMolecules ){
    double4 localThermostatParameters; // = thermostatParameters[0];
    double3 my_velocity_CM = {0., 0., 0.};
    unsigned int size = mlist[my_mol_idx].x;
    unsigned offset = max_size*my_mol_idx + nMolecules;
    unsigned int i;
    double my_mol_mass = 0.;
    double strainStep = timeStep * strainRate;
    /*double c1 = strainRate*localThermostatParameters.y / localThermostatParameters.x ;
    double c2 = strainRate*strainRate * localThermostatParameters.z / localThermostatParameters.x;
    double g_factor = 1./sqrt(1. - c1*timeStep  + 0.25 * c2 * timeStep*timeStep);*/
    double g_factor = thermostatState[0].x;
    float4 my_cm = cm[my_mol_idx];
    for ( unsigned m=0; m<size; m++ ){
      i = mlist[offset+m].x;
      float4 this_v = velocity[i];
      float mi = 1.0/this_v.w;
      my_velocity_CM.x += mi * this_v.x;
      my_velocity_CM.y += mi * this_v.y;
      my_velocity_CM.z += mi * this_v.z;
      my_mol_mass += mi;
    }
    my_velocity_CM.x /= my_mol_mass;
    my_velocity_CM.y /= my_mol_mass;
    my_velocity_CM.z /= my_mol_mass;

    // need the change in CM velocity for the particle velocity update
    double3 delta_velocity_CM;
    delta_velocity_CM.x = ( (g_factor-1.)*my_velocity_CM.x -g_factor*0.5*strainStep*my_velocity_CM.y );
    delta_velocity_CM.y = (g_factor-1.)*my_velocity_CM.y;
    delta_velocity_CM.z = (g_factor-1.)*my_velocity_CM.z;

    // update the CM velocity (needs to be updated now for the A part)
    my_velocity_CM.x = g_factor* ( my_velocity_CM.x - 0.5*strainStep*my_velocity_CM.y );
    my_velocity_CM.y *= g_factor;
    my_velocity_CM.z *= g_factor;

    

    // Load the simulation box in local memory to avoid bank conflicts.
    float array[simulationBoxSize];
    simBox->loadMemory( array, simBoxPointer );

    for ( unsigned m=0; m<size; m++ ){
      i = mlist[offset+m].x;
      float4 this_v = velocity[i];
      float4 this_p = position[i];
      float4 this_im = image[i];

      // The B1 part of the operator: update the velocities (half-step)
      this_v.x += delta_velocity_CM.x;
      this_v.y += delta_velocity_CM.y;
      this_v.z += delta_velocity_CM.z;
      
      velocity[i] = this_v;

      // The A part: update the positions (full-step) at fixed velocities
      this_p.x += (this_v.x + 0.5*strainStep*my_velocity_CM.y)*timeStep
	+ strainStep * my_cm.y;
      this_p.y += this_v.y * timeStep; 
      this_p.z += this_v.z * timeStep; 
      
    float4 local_image = simBox->applyBoundaryCondition( this_p, array );
    
    // Save the integration in global memory.
    position[i] = this_p; 
    this_im.x += local_image.x;
    this_im.y += local_image.y;
    this_im.z += local_image.z;
    this_im.w += local_image.w;
    
    image[i] = this_im;


    }

    localThermostatParameters.x =  ( my_velocity_CM.x * my_velocity_CM.x +
				      my_velocity_CM.y * my_velocity_CM.y +
				      my_velocity_CM.z * my_velocity_CM.z) * my_mol_mass; // p^2/m 
    localThermostatParameters.y = my_velocity_CM.x * my_velocity_CM.y * my_mol_mass; // px py / m (for c1)
    localThermostatParameters.z = my_velocity_CM.y * my_velocity_CM.y * my_mol_mass; // py^2 / m (for c2)
    thermostatParameters[my_mol_idx] = localThermostatParameters;

  }
  
}



// set the quantities which need to be summed for the g_factor, as at the end
// of integrateSLLOD_A_B1
__global__ void initialize_g_factor_molecules(unsigned int nMolecules, const int1* mlist, unsigned int max_size, float4* v, double4* thermostatParameters) {
  unsigned int my_mol_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(my_mol_idx < nMolecules) {
    double3 my_mol_momentum = {0., 0., 0.};
    // need size of this molecule
    unsigned int size = mlist[my_mol_idx].x;
    unsigned offset = max_size*my_mol_idx + nMolecules;
    unsigned int i;
    float my_mol_mass = 0.f;

    for ( unsigned m=0; m<size; m++ ){
      i = mlist[offset+m].x;
      float4 this_v = v[i];
      float mi = 1.0/this_v.w;
      my_mol_momentum.x += mi * this_v.x;
      my_mol_momentum.y += mi * this_v.y;
      my_mol_momentum.z += mi * this_v.z;
      my_mol_mass += mi;
    }

    double4 localThermostatParameters;
    
    localThermostatParameters.x =  (my_mol_momentum.x * my_mol_momentum.x +
				     my_mol_momentum.y * my_mol_momentum.y +
				     my_mol_momentum.z * my_mol_momentum.z) / my_mol_mass; // p^2/m
    localThermostatParameters.y = my_mol_momentum.x * my_mol_momentum.y / my_mol_mass; //  px py / m (for c1)
    localThermostatParameters.z = my_mol_momentum.y * my_mol_momentum.y / my_mol_mass; //  py py / m (for c2)
    thermostatParameters[my_mol_idx] = localThermostatParameters;
  }
}


__global__ void particleKineticEnergies(unsigned numParticles, 
					float4* velocity, 
					double4* thermostatParameters) {
					  
  if ( MyGP < numParticles ){ 
    float4 my_velocity = velocity[MyGP]; 
    double4 local_thermostatParameters;
    local_thermostatParameters.x =  ( my_velocity.x * my_velocity.x +
				      my_velocity.y * my_velocity.y +
				      my_velocity.z * my_velocity.z) / my_velocity.w; // p^2/m 
    thermostatParameters[MyGP] = local_thermostatParameters;
  } // if (MyGP ... )
}
