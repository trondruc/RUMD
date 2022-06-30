
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
#include "rumd/IntegratorSLLOD.h"
#include "rumd/ParseInfoString.h"

const std::string SLLOD_Error_Code1("There is no integrator associated with Sample or the latter has no particles");

////////////////////////////////////////////////////////////
// Constructors 
////////////////////////////////////////////////////////////

IntegratorSLLOD::IntegratorSLLOD(float timeStep, double strainRate) {
  this->timeStep = timeStep;
  this->strainRate = strainRate;

  numParticlesSample = 0;
  initialized = false;
  AllocateFromConstructor();
}

IntegratorSLLOD::~IntegratorSLLOD(){
  cudaFree(d_thermostatState);
  FreeArrays();
}

////////////////////////////////////////////////////////////
// Class Methods
////////////////////////////////////////////////////////////

// Integrate SLLOD dynamics using the operator-splitting method of 
// Pan et al. JCP 122, 094114 (2005)
void IntegratorSLLOD::Integrate(){
  if(!S || !(numParticlesSample))
    throw( RUMD_Error("IntegratorSLLOD","Integrate", SLLOD_Error_Code1 ) );
  
  LeesEdwardsSimulationBox* testLESB = dynamic_cast<LeesEdwardsSimulationBox*>(S->GetSimulationBox());
  if(!testLESB) throw RUMD_Error("IntegratorSLLOD","Integrate","LeesEdwardsSimulationBox is needed for this integrator");

  if(!initialized)
    Initialize_g_factor();


  // B1 (half-step on velocities from shear)
  integrateSLLOD_B1<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_v, P->d_f, d_thermostatParameters, timeStep, strainRate, d_thermostatState);

  Update_factor1_factor2();
  // B2 (full-step on velocities from forces)
  integrateSLLOD_B2<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_v, P->d_f, timeStep, d_thermostatParameters, d_thermostatState);

  
  Update_g_factor();
  float wrap = testLESB->IncrementBoxStrain(strainRate*timeStep);
  P->ApplyLeesEdwardsWrapToImages(wrap);
  // B1 (half-step on velocities from shear)  + A (full-step on positions)
  integrateSLLOD_A_B1<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_r, P->d_v, P->d_im, testLESB, testLESB->GetDevicePointer(), d_thermostatParameters, timeStep, strainRate, d_thermostatState);

  Update_g_factor();
}

// Private. Update the thermostat state.
void IntegratorSLLOD::Update_factor1_factor2(){
  
  sumIdenticalArrays( d_thermostatParameters, numParticlesSample, 1, 32 );

  update_factor1_factor2_kernel<<<1, 1 >>>( timeStep, d_thermostatParameters, d_thermostatState);
}


void IntegratorSLLOD::Update_g_factor(){

  sumIdenticalArrays( d_thermostatParameters, numParticlesSample, 1, 32 );

  update_g_kernel<<<1, 1>>>(timeStep, strainRate, d_thermostatParameters, d_thermostatState);
}

void IntegratorSLLOD::Initialize_g_factor() {
  if(!S || !(numParticlesSample))
    throw( RUMD_Error("IntegratorSLLOD","Integrate", SLLOD_Error_Code1 ) );

  // launch a kernel to set the quantities to be added for the g_factor
  initialize_g_factor<<< kp.grid, kp.threads >>>(numParticlesSample, P->d_v, d_thermostatParameters);
  // do the sums and calculate g_factor
  Update_g_factor();
  initialized = true;
}

void IntegratorSLLOD::AllocateFromConstructor(){
  // Allocate space on the GPU for the thermostat state.
  if( cudaMalloc( (void**) &d_thermostatState, sizeof(double4) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("IntegratorSLLOD","IntegratorSLLOD","Malloc failed on d_thermostatState") );
  
  // Initialize the state on the GPU.
  double4 thermostatState = {0., 0., 0., 0.};
  cudaMemcpy( d_thermostatState, &thermostatState, sizeof(double4), cudaMemcpyHostToDevice );
  
  // Blocks until the device has completed all preceding requested tasks. 
  // Host => Device is async.
  if( cudaDeviceSynchronize() != cudaSuccess ) 
    throw( RUMD_Error("IntegratorSLLOD","IntegratorSLLOD","CudaMemcpy failed: d_thermostatState") ); 
}



// Allocates memory dependent on the N particles.
void IntegratorSLLOD::AllocateIntegratorState(){
  if(!S || !(S->GetNumberOfParticles())) // Consistency check.
    throw( RUMD_Error("IntegratorSLLOD","AllocateIntegratorState", SLLOD_Error_Code1 ) );
  
  unsigned int newNumParticlesS = S->GetNumberOfParticles();
  
  if(newNumParticlesS != numParticlesSample){
    FreeArrays();
    numParticlesSample = newNumParticlesS;
    
    // Allocate space for the summation of kinetic energy on GPU.
    if( cudaMalloc( (void**) &d_thermostatParameters, numParticlesSample * sizeof(double4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorSLLOD","AllocateIntegratorState","Malloc failed on d_thermostatParameters") );
    
    // Allocate space for the summation of momentum on GPU.
    if( cudaMalloc( (void**) &d_particleMomentum, numParticlesSample * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorSLLOD","AllocateIntegratorState","Malloc failed on d_particleMomentum") );
  }
}

// Frees memory dependent on numParticles
void IntegratorSLLOD::FreeArrays(){
  if(numParticlesSample){
    cudaFree(d_thermostatParameters);
    cudaFree(d_particleMomentum);
  }
}

////////////////////////////////////////////////////////////
// Set Methods 
////////////////////////////////////////////////////////////

void IntegratorSLLOD::SetTimeStep(float dt) { 
  timeStep = dt;
  initialized = false;
}

void IntegratorSLLOD::SetStrainRate(double set_strain_rate) {
  strainRate = set_strain_rate;
  initialized = false;
}

void IntegratorSLLOD::SetKineticEnergy( float set_kinetic_energy ){  
  float current_KE = GetKineticEnergy(true);
  if(current_KE > 0.0) {
    float scale_factor = sqrt( set_kinetic_energy / current_KE );
    rescale_velocities<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_v, scale_factor); 
      }
  else
    throw RUMD_Error("IntegratorSLLOD", "SetKineticEnergy", "Cannot rescale; KE is zero!");

  initialized = false;
}



void IntegratorSLLOD::SetMomentumToZero(){
  if(!S || !numParticlesSample)
    throw( RUMD_Error("IntegratorSLLOD","SetMomentumToZero", SLLOD_Error_Code1 ) );
  
  calculateMomentum<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_v, d_particleMomentum );
  sumIdenticalArrays( d_particleMomentum, numParticlesSample, 1, 32 );
  zeroTotalMomentum<<<kp.grid, kp.threads>>>( numParticlesSample, P->d_v, d_particleMomentum );

  //initialized = false; // not sure if this is necessary
}


////////////////////////////////////////////////////////////
// Get Methods 
////////////////////////////////////////////////////////////

std::string IntegratorSLLOD::GetInfoString(unsigned int precision) const{
  std::ostringstream infoStream;
  infoStream << "IntegratorSLLOD";
  infoStream << "," << std::setprecision(precision) << GetTimeStep();
  infoStream << "," << std::setprecision(precision) << GetStrainRate();
  return infoStream.str();
}

void IntegratorSLLOD::InitializeFromInfoString(const std::string& infoStr, bool verbose) {
  std::vector<float> parameterList;
  std::string className = ParseInfoString(infoStr, parameterList);
  // check typename, if no match, raise error
  if(className != "IntegratorSLLOD")
    throw RUMD_Error("IntegratorSLLOD","InitializeFromInfoString",std::string("Wrong integrator type: ")+className);
  
  if(parameterList.size() != 2)
    throw RUMD_Error("IntegratorSLLOD","InitializeFromInfoString","Wrong number of parameters in infoStr");

  if(verbose)
    std::cout << "[Info] Initializing IntegratorSLLOD. time step:" << parameterList[0] << "; strain rate:" << parameterList[1]  << std::endl;
  SetTimeStep(parameterList[0]);
  SetStrainRate(parameterList[1]);
}


__global__ void integrateSLLOD_B1(unsigned numParticles, 
				  float4* velocity, 
				  float4* force, 
				  double4* thermostatParameters,
				  float timeStep,
				  float strainRate,
				  double4* thermostatState){
  
  if ( MyGP < numParticles ){ 
    float4 my_velocity = velocity[MyGP]; 
    float4 my_force = force[MyGP];
    double g_factor = 1.;
    double4 localThermostatParameters;// = thermostatParameters[0];
    double strainStepHalf = strainRate*timeStep*0.5;
    if(thermostatState)
      g_factor = thermostatState[0].x;
    /*{
    // g_factor for 0.5*timeStep
    double c1 = strainRate*localThermostatParameters.y / localThermostatParameters.x ;
    double c2 = strainRate*strainRate * localThermostatParameters.z / localThermostatParameters.x;
    g_factor = 1./sqrt(1. - c1*timeStep  + 0.25 * c2 * timeStep*timeStep);
    }*/

    my_velocity.x = g_factor*(my_velocity.x - strainStepHalf*my_velocity.y);
    my_velocity.y *= g_factor;
    my_velocity.z *= g_factor;
    

    // Save the integration in global memory.
    velocity[MyGP] = my_velocity;    

    // Parameters needed for Gaussian thermostat in step B2

    double4 local_force = {my_force.x, my_force.y, my_force.z, my_force.w};
    localThermostatParameters.x =  ( my_velocity.x * my_velocity.x +
				     my_velocity.y * my_velocity.y +
				     my_velocity.z * my_velocity.z) / my_velocity.w; // p^2/m 
    localThermostatParameters.y = local_force.x * my_velocity.x + local_force.y * my_velocity.y + local_force.z * my_velocity.z; // f.p/m 
    localThermostatParameters.z = ( local_force.x * local_force.x +
				     local_force.y * local_force.y + 
				     local_force.z * local_force.z ) * my_velocity.w; // f^2/m

    
    thermostatParameters[MyGP] = localThermostatParameters;
  } // if (MyGP ... )
}

__global__ void integrateSLLOD_B2(unsigned numParticles, 
				  float4* velocity, 
				  float4* force,
				  float timeStep,
				  double4* thermostatParameters,
				  double4* thermostatState){
  if ( MyGP < numParticles ){ 
    double4 localThermostatParameters;// = thermostatParameters[0];
    double integrate_factor1 = thermostatState[0].y;
    double integrate_factor2 = thermostatState[0].z;
    /*double alpha = localThermostatParameters.y / localThermostatParameters.x;
    double beta = sqrt(localThermostatParameters.z / localThermostatParameters.x);
    double h = (alpha + beta) / (alpha - beta) ;
    double e = exp(-beta*timeStep);
    double integrate_factor1 =  (1.f - h) / (e - h/e);
    double integrate_factor2 = (1.f + h - e - h/e)/((1.f-h)*beta);*/

    float4 my_velocity = velocity[MyGP]; 
    float4 my_force = force[MyGP];
 

    // Note there was erroneously a minus sign in front of the second
    // term in the Pan et al article
    my_velocity.x = integrate_factor1*(my_velocity.x + integrate_factor2*my_force.x*my_velocity.w);
    my_velocity.y = integrate_factor1*(my_velocity.y + integrate_factor2*my_force.y*my_velocity.w);
    my_velocity.z = integrate_factor1*(my_velocity.z + integrate_factor2*my_force.z*my_velocity.w);
    

    // Save the integration in global memory.
    velocity[MyGP] = my_velocity;
    // Parameters needed for Gaussian thermostat in step A_B1
    //double4 local_thermostatParameters;
    localThermostatParameters.x =  (my_velocity.x * my_velocity.x +
				    my_velocity.y * my_velocity.y +
				    my_velocity.z * my_velocity.z) / my_velocity.w; // p^2/m
    localThermostatParameters.y = my_velocity.x *my_velocity.y / my_velocity.w; //  px py / m (for c1)
    localThermostatParameters.z = my_velocity.y *my_velocity.y / my_velocity.w; //  py py / m (for c2)

    thermostatParameters[MyGP] = localThermostatParameters;
  }

}



template <class S> __global__ void integrateSLLOD_A_B1(unsigned numParticles, 
						       float4* position, 
						       float4* velocity, 
						       float4* image, 
						       S* simBox, 
						       float* simBoxPointer, 
						       double4* thermostatParameters,
						       float timeStep,
						       float strainRate,
						       double4* thermostatState){
  //  A after B1
  if ( MyGP < numParticles ){ 
    double4 localThermostatParameters;// = thermostatParameters[0];
    float4 my_position = position[MyGP]; float4 my_velocity = velocity[MyGP]; 
    float4 my_image = image[MyGP];
    double strainStep = timeStep * strainRate;
    /*// g_factor for 0.5*timeStep
    double c1 = strainRate*localThermostatParameters.y / localThermostatParameters.x ;
    double c2 = strainRate*strainRate * localThermostatParameters.z / localThermostatParameters.x;
    double g_factor = 1./sqrt(1. - c1*timeStep  + 0.25 * c2 * timeStep*timeStep);
    */
    double g_factor = thermostatState[0].x;

    // Load the simulation box in local memory to avoid bank conflicts.
    float array[simulationBoxSize];
    simBox->loadMemory( array, simBoxPointer );
 

    // The B1 part of the operator: update the velocities (half-step)
    my_velocity.x = g_factor*(my_velocity.x - 0.5*strainStep*my_velocity.y);
    my_velocity.y *= g_factor;
    my_velocity.z *= g_factor;

    // The A part: update the positions (full-step) at fixed velocities
    my_position.x += (my_velocity.x + 0.5*strainStep*my_velocity.y)*timeStep
      + strainStep * my_position.y;
    my_position.y += my_velocity.y * timeStep; 
    my_position.z += my_velocity.z * timeStep; 
    
    float4 local_image = simBox->applyBoundaryCondition( my_position, array );
    
    // Save the integration in global memory.
    position[MyGP] = my_position; 
    velocity[MyGP] = my_velocity;    

    my_image.x += local_image.x;
    my_image.y += local_image.y;
    my_image.z += local_image.z;
    my_image.w += local_image.w;
    
    image[MyGP] = my_image;
 
    // Parameters needed for Gaussian thermostat in first B1 step
    //double4 local_thermostatParameters;
    localThermostatParameters.x =  (my_velocity.x * my_velocity.x +
				     my_velocity.y * my_velocity.y +
				     my_velocity.z * my_velocity.z) / my_velocity.w; // p^2/m
    localThermostatParameters.y = my_velocity.x *my_velocity.y / my_velocity.w; //  px py / m (for c1)
    localThermostatParameters.z = my_velocity.y *my_velocity.y / my_velocity.w; //  py py / m (for c2)

    thermostatParameters[MyGP] = localThermostatParameters;
 }
  
}

// set the quantities which need to be summed for the g_factor, as at the end
// of integrateSLLOD_A_B1
__global__ void initialize_g_factor(unsigned int numParticles,
			 float4* velocity,
			 double4* thermostatParameters) {
  if ( MyGP < numParticles ){ 
    float4 my_velocity = velocity[MyGP];
    double4 local_thermostatParameters;
    
    local_thermostatParameters.x =  (my_velocity.x * my_velocity.x +
				     my_velocity.y * my_velocity.y +
				     my_velocity.z * my_velocity.z) / my_velocity.w; // p^2/m
    local_thermostatParameters.y = my_velocity.x * my_velocity.y / my_velocity.w; //  px py / m (for c1)
    local_thermostatParameters.z = my_velocity.y * my_velocity.y / my_velocity.w; //  py py / m (for c2)

    thermostatParameters[MyGP] = local_thermostatParameters;
   
  }
}


__global__ void   update_g_kernel(double timeStep, 
				  double strainRate,
				  double4* thermostatParameters,
				  double4* thermostatState) {
  
  double4 localThermostatParameters = thermostatParameters[0];
  double c1 = strainRate*localThermostatParameters.y / localThermostatParameters.x ;
  double c2 = strainRate*strainRate * localThermostatParameters.z / localThermostatParameters.x;
  
  // this is the g_factor for 0.5*timeStep
  double4 localThermostatState = {1./sqrt(1. - c1*timeStep  + 0.25 * c2 * timeStep*timeStep), 0., 0., 0.};
  thermostatState[0] = localThermostatState;
}

__global__ void update_factor1_factor2_kernel(double timeStep,
					      double4* thermostatParameters, 
					      double4* thermostatState) {
  double4 local_thermostatState = {0., 0., 0., 0.}; // = thermostatState[0];
  // x-component was g_factor, but won't keep it (requires extra read)
  double4 localThermostatParameters = thermostatParameters[0];

  double alpha = localThermostatParameters.y / localThermostatParameters.x;
  double beta = sqrt(localThermostatParameters.z / localThermostatParameters.x);
  if(beta * timeStep < 1.e-7) {
    local_thermostatState.y = 1.;
    local_thermostatState.z = timeStep;
  }
  else {
    double h = (alpha + beta) / (alpha - beta) ;
    double e = exp(-beta*timeStep);
    local_thermostatState.y = (1. - h) / (e - h/e);
    local_thermostatState.z = (1. + h - e - h/e)/((1.-h)*beta);
  }
  thermostatState[0] = local_thermostatState;
}


__global__ void rescale_velocities( unsigned int numParticles,
				    float4* velocity,
				    float rescale) {
  if(MyGP < numParticles) {
    float4 my_velocity = velocity[MyGP];
    my_velocity.x *= rescale;
    my_velocity.y *= rescale;
    my_velocity.z *= rescale;
    velocity[MyGP] = my_velocity;
  }

}
