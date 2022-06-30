/**
  Copyright (C) 2010  Thomas Schrøder

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/Sample.h"
#include "rumd/rumd_algorithms.h"
#include "rumd/RUMD_Error.h"
#include "rumd/IntegratorNPTLangevin.h"
#include "rumd/ParseInfoString.h"

#include <iostream>
#include <string>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>


#include <curand_kernel.h>

const std::string NPTL_Error_Code1("There is no integrator associated with Sample or the latter has no particles");

////////////////////////////////////////////////////////////
// Constructors
////////////////////////////////////////////////////////////

///  Constructor for the NPT Integrator
/**
*	@param timeStep						Timestep for the integrator
*	@param targetTemperature			External temperature of the NPT ensemble
        @param friction                                 Friction coefficient for particles
*	@param targetPressure				External pressure of the NPT ensemble
        @param barostatFriction                         Friction coefficient for box degree of freedom
	@param barostatMass                             Inertial parameter for box degree of freedom

*/
IntegratorNPTLangevin::IntegratorNPTLangevin(float timeStep, float targetTemperature, float friction, float targetPressure, float barostatFriction, float barostatMass) : boxFlucCoord(2)
{

  this->timeStep = timeStep;
  numParticlesSample = 0;

  // Thermostat parameters
  this->friction = friction;
  this->targetTemperature = targetTemperature;
  temperatureChangePerTimeStep = 0.;

  // Barostat parameters
  this->targetPressure = targetPressure;
  this->barostatFriction = barostatFriction;
  this->barostatMass = barostatMass;
  pressureChangePerTimeStep = 0.;

  num_rand = 0;
  this->barostatMode = ISO; // default is isotropic barostatting
  
  AllocateFromConstructor();

}

/// Destructor (free up memory)
IntegratorNPTLangevin::~IntegratorNPTLangevin()
{
  cudaFree(d_barostatState);
  curandDestroyGenerator(curand_generator);
  FreeArrays();
}


////////////////////////////////////////////////////////////
// Class Methods
////////////////////////////////////////////////////////////


/// Called by the main loop to make one integration step.
void IntegratorNPTLangevin::Integrate()
{

  if(!S || !(numParticlesSample))
    throw( RUMD_Error("IntegratorNPTLangevin", __func__, NPTL_Error_Code1) );

  if(targetTemperature < 0.)
    throw( RUMD_Error("IntegratorNPTLangevin", __func__, "Negative target temperature") );
  float stddev = sqrtf(2.f * friction * targetTemperature*timeStep);


  curandGenerateNormal(curand_generator, d_randomForces, num_rand,
  		       0.f, stddev);

  double3 length_ratio = {1.0, 1.0, 1.0};
  if(barostatMode == ISO || barostatMode == ANISO)
    length_ratio = UpdateState();

  // Update positions on GPU. Exit and print error if Lees-Edwards Simulation Box is used.  
  RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(S->GetSimulationBox());
  if(testRSB)
    integrateNPTLangevinAlgorithm<<< kp.grid, kp.threads >>>(	numParticlesSample, P->d_r, P->d_v, P->d_f, P->d_im,
								testRSB , testRSB->GetDevicePointer(),
								d_randomForces,  friction, length_ratio,
								timeStep);
  else
    throw( RUMD_Error("IntegratorNPTLangevin", __func__, "Cannot use Lees-Edwards simulation box with NPT integrator") );

  // Update external pressure and temperature
  targetTemperature += temperatureChangePerTimeStep;
  targetPressure += pressureChangePerTimeStep;

}

/// Calculate the total virial and update the barostat state (box geometry) 
double3 IntegratorNPTLangevin::UpdateState()
{

  // Copy virial into barostatVirial array. For anisotropic mode it's not
  // the virial, but the diagonal element corresponding to the fluctuating
  // direction (the virial is the mean of these)
  if(barostatMode==ISO)
    copyParticleVirial<<< kp.grid, kp.threads >>>(numParticlesSample, P->d_w, d_barostatVirial, true, 0);
  else
    copyParticleVirial<<< kp.grid, kp.threads >>>(numParticlesSample, P->d_sts, d_barostatVirial, false, boxFlucCoord);
  
  // Compute virial by summing particle contributions
  sumIdenticalArrays( d_barostatVirial, numParticlesSample, 1, 32 );

  // Update the barostat state on GPU
  double volume = S->GetSimulationBox()->GetVolume();
  double targetConfPressure = targetPressure - targetTemperature * S->GetNumberOfParticles() / volume;
  updateNPTLangevinState<<<1, 1>>>( 	d_barostatVirial,
					barostatFriction, barostatMass,
					targetConfPressure, friction,
					d_barostatState, volume,
					d_randomForces, timeStep );
  
  // Change Volume of box
  double2 barostatState = GetBarostatState();
  double vol_scale_factor = barostatState.x;


  if(barostatMode == ISO) {
    double lr_iso = pow(vol_scale_factor, 1.0/3.0);
    // ScaleBoxFraction does addition in double precision
    S->GetSimulationBox()->ScaleBoxFraction(lr_iso - 1.);
    double3 length_ratio = {lr_iso, lr_iso, lr_iso};
    return length_ratio;
  }
  else {
    S->GetSimulationBox()->ScaleBoxDirection(vol_scale_factor, boxFlucCoord);
    double3 length_ratio = {1.0, 1.0, 1.0};
    if(boxFlucCoord == 0)
      length_ratio.x = vol_scale_factor;
    else if (boxFlucCoord == 1)
      length_ratio.y = vol_scale_factor;
    else
      length_ratio.z = vol_scale_factor;

    return length_ratio;
  }


}


/// Allocate space on the GPU for the barostat state
void IntegratorNPTLangevin::AllocateFromConstructor()
{

  // Allocate space on the GPU for the barostat state.
  if( cudaMalloc( (void**) &d_barostatState, sizeof(double2) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("IntegratorNPTLangevin", __func__,"Malloc failed on d_barostatState") );

  // Initialize the state on the GPU.
  cudaMemset( d_barostatState, 0, sizeof(double2));


  curandStatus_t err1 = curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT);
  if(err1 != CURAND_STATUS_SUCCESS)
    std::cerr << "Error in curandCreateGenerator" << std::endl;
  curandSetPseudoRandomGeneratorSeed(curand_generator, 1234ULL);
    
  
  // Blocks until the device has completed all preceding requested tasks.
  // Host => Device is async.
  if( cudaDeviceSynchronize() != cudaSuccess )
    throw( RUMD_Error("IntegratorNPTLangevin", __func__,"cuda error") );


  
}


/// Allocates memory dependent on the N particles.
void IntegratorNPTLangevin::AllocateIntegratorState()
{

  // Consistency check.
  if(!S || !(S->GetNumberOfParticles()))
    throw( RUMD_Error("IntegratorNPTLangevin", __func__, NPTL_Error_Code1 ) );

  unsigned int newNumParticlesS = S->GetNumberOfParticles();

  if(newNumParticlesS != numParticlesSample){
    FreeArrays();
    numParticlesSample = newNumParticlesS;

    // Allocate space for random numbers on GPU.
    num_rand = numParticlesSample * 3 + 1; // +1 for box dynamics
    // for technical reasons (Box-Muller transform) the
    // number of random samples must be a multiple of 2
    if(num_rand % 2)
      num_rand += 1;
    
    if( cudaMalloc( (void**) &d_randomForces, num_rand * sizeof(float) ) == cudaErrorMemoryAllocation )
      throw( RUMD_Error("IntegratorNPTLangevin",  __func__, "Malloc failed on d_randomForces") );

    cudaMemset( d_randomForces, 1, num_rand * sizeof(float));

    
  // Allocate space for the summation of momentum on GPU.
  if( cudaMalloc( (void**) &d_particleMomentum, numParticlesSample * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("IntegratorNPTLangevin", __func__,"Malloc failed on d_particleMomentum") );

  // Allocate space for the summation of pressure on GPU.
  if( cudaMalloc( (void**) &d_barostatVirial, numParticlesSample * sizeof(double) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("IntegratorNPTLangevin",  __func__,"Malloc failed on d_barostatVirial") );
  } // newNumParticlesS != numParticlesSample
  
  
}


/// Frees memory dependent on numParticles
void IntegratorNPTLangevin::FreeArrays()
{
  if(numParticlesSample){
    cudaFree(d_randomForces);
    cudaFree(d_particleMomentum);
    cudaFree(d_barostatVirial);
  }
}



////////////////////////////////////////////////////////////
// Set Methods
////////////////////////////////////////////////////////////





void IntegratorNPTLangevin::SetBarostatState( double volume_velocity )
{
  double2 barostatState = {1.0, volume_velocity};
  cudaMemcpy( d_barostatState, &barostatState, sizeof(double2), cudaMemcpyHostToDevice );
}


void IntegratorNPTLangevin::SetTemperatureChangePerTimeStep(double temperatureChangePerTimeStep)
{
  this->temperatureChangePerTimeStep = temperatureChangePerTimeStep;
}



void IntegratorNPTLangevin::SetPressureChangePerTimeStep(double pressureChangePerTimeStep)
{
  this->pressureChangePerTimeStep = pressureChangePerTimeStep;
}



////////////////////////////////////////////////////////////
// Get Methods
////////////////////////////////////////////////////////////


/// Used to save the integrator parameters and state to header of restart file, appearing after "integrator=".
std::string IntegratorNPTLangevin::GetInfoString(unsigned int precision) const
{
  // Note a true restart cannot be implemented without saving the state(s)
  // of the random number generators
  std::ostringstream infoStream;
    infoStream << "IntegratorNPTLangevin";
    infoStream << "," << std::setprecision(precision) << GetTimeStep();
    infoStream << "," << std::setprecision(precision) << GetTargetTemperature();
    infoStream << "," << std::setprecision(precision) << GetFriction();
    infoStream << "," << std::setprecision(precision) << GetTargetPressure();
    infoStream << "," << std::setprecision(precision) << GetBarostatFriction();
    infoStream << "," << std::setprecision(precision) << GetBarostatMass();
    double2 barostatState = GetBarostatState();
    infoStream << "," << std::setprecision(precision) << barostatState.y;

    return infoStream.str();
}



/// Used when restarting a simulation to set the integrator state to old values, appearing after "integrator=".
void IntegratorNPTLangevin::InitializeFromInfoString(const std::string& infoStr, bool verbose)
{
  std::vector<float> parameterList;
  std::string className = ParseInfoString(infoStr, parameterList);
  // check typename, if no match, raise error
  bool npt = (className == "IntegratorNPTLangevin");
  if(!npt)
    throw RUMD_Error("IntegratorNPTLangevin", __func__,"Expected IntegratorNPTLangevin.");
  unsigned int requiredNumParams = 7;
  if(parameterList.size() != requiredNumParams)
    throw RUMD_Error("IntegratorNPTLangevin",__func__,"Wrong number of parameters in infoStr");

  if(verbose)
    std::cout << "[Info] Initializing IntegratorNPTLangevin. time step:" << parameterList[0] << "; target temperature:" << parameterList[1]  << "; friction:" << parameterList[2] << std::endl;
  
  SetTimeStep(parameterList[0]);
  SetTargetTemperature(parameterList[1]);
  SetFriction(parameterList[2]);
  if(npt) {
    if(verbose)
      std::cout << "target pressure:" << parameterList[3] << "; barostat friction: " << parameterList[4] << "; barostat mass:" << parameterList[5] << "; barostat state:" << parameterList[6] << std::endl;
    SetTargetPressure(parameterList[3]);
    SetBarostatFriction(parameterList[4]);
    SetBarostatMass(parameterList[5]);
    SetBarostatState(parameterList[6]);
  }

}



/// Returns the barostat state, a pair of doubles giving the ratio of new volume to old, and the volume velocity
double2 IntegratorNPTLangevin::GetBarostatState() const
{
  double2 barostatState = {0., 0.};
  cudaMemcpy( &barostatState, d_barostatState, sizeof(double2), cudaMemcpyDeviceToHost );
  return barostatState;
}



/// Return kinetic energy \f$ \frac{1}{2}\sum m v^2 \f$ of the LFL algorithm. @param copy copy from device if true.
double IntegratorNPTLangevin::GetKineticEnergy(bool copy) const
{
  if(!numParticlesSample)
    throw( RUMD_Error("IntegratorNPTLangevin", __func__,"Sample has no particles" ) );

  if(copy) {
    P->CopyVelFromDevice(); P->CopyForFromDevice();
  }

  double stddev = sqrt(2. * friction * targetTemperature * timeStep);

  boost::mt19937 rng;
  boost::normal_distribution<> nd(0.0, stddev);

  boost::variate_generator<boost::mt19937, 
                           boost::normal_distribution<> > generator(rng, nd);

  
  double totalKineticEnergy = 0.0;
 
  for(unsigned int i=0; i < numParticlesSample; i++){
    double mass = 1.0 / P->h_v[i].w;
    double scaled_dt = 0.5*friction*timeStep * P->h_v[i].w;
    double prm_b = 1./(1. + scaled_dt);
    double prm_a = (1. - scaled_dt) * prm_b;

    // random forces
    double rfx = generator();
    double rfy = generator();
    double rfz = generator();

    double velocityFx = prm_a * P->h_v[i].x + (prm_b/mass)*(timeStep * P->h_f[i].x + rfx);
    double velocityFy = prm_a * P->h_v[i].y + (prm_b/mass)*(timeStep * P->h_f[i].y + rfy);
    double velocityFz = prm_a * P->h_v[i].z + (prm_b/mass)*(timeStep * P->h_f[i].z + rfz);
   
    // Velocity sq before: v^2(t-h/2)
    double velocity2B = P->h_v[i].x * P->h_v[i].x + P->h_v[i].y * P->h_v[i].y + P->h_v[i].z * P->h_v[i].z;

    // Velocity sq after: v^2(t+h/2)
    double velocity2A = velocityFx * velocityFx + velocityFy * velocityFy + velocityFz * velocityFz;

    // The kinetic energy v^2(t) should be calculated like this in a LFA.
    totalKineticEnergy += mass * (velocity2B + velocity2A);
  }
  
  return 0.25 * totalKineticEnergy;
}

__global__ void copyParticleVirial(unsigned numParticles, float4* conf_press, float* barostatVirial, bool isotropic, unsigned boxFlucCoord) {
 if ( MyGP < numParticles ) {
   float4 my_conf_press = conf_press[MyGP];
   if(isotropic)
     // in this case conf_press is the virial array, but haven't normalized
     // by 3 yet or accounted for double-counting
     barostatVirial[MyGP] = my_conf_press.w/6.f;
   // in the following cases conf_press is the stress array, need minus sign
   else if (boxFlucCoord == 0)
     barostatVirial[MyGP] = - my_conf_press.x*0.5f;
   else if (boxFlucCoord == 1)
     barostatVirial[MyGP] = - my_conf_press.y*0.5f;
   else if(boxFlucCoord == 2)
     barostatVirial[MyGP] = - my_conf_press.z*0.5f;
 } // MyGP < numParticles

}


/// Move particle positions and velocities on GPU kernel
template <class S>
__global__ void integrateNPTLangevinAlgorithm( unsigned numParticles,
					       float4* position,
					       float4* velocity,
					       float4* force, float4* image,
					       S* simBox,
					       float* simBoxPointer,
					       float *randomForces,
					       double friction,
					       double3 lengthRatio,
					       double timeStep)
{

  if ( MyGP < numParticles ){
    float4 my_position = position[MyGP]; float4 my_velocity = velocity[MyGP];
    float4 my_force = force[MyGP];  float4 my_image = image[MyGP];
    
    // Load the simulation box in local memory to avoid bank conflicts.
    float array[simulationBoxSize];
    simBox->loadMemory( array, simBoxPointer );
    double scaled_dt = 0.5*friction*timeStep * my_velocity.w;
    double prm_b = 1./(1. + scaled_dt);
    double prm_a = (1. - scaled_dt) * prm_b;

    // indexing the random forces takes into account that the first (index 0)
    // is used for the volume DOF

    my_velocity.x = prm_a * my_velocity.x + prm_b * my_velocity.w * (my_force.x * timeStep + randomForces[3*MyGP+1]);
    my_velocity.y = prm_a * my_velocity.y + prm_b * my_velocity.w * (my_force.y * timeStep + randomForces[3*MyGP+2]);
    my_velocity.z = prm_a * my_velocity.z + prm_b * my_velocity.w * (my_force.z * timeStep + randomForces[3*MyGP+3]);
								     
    // Update to r(t+h).

    double3 L_factor = {2.* lengthRatio.x / (1. + lengthRatio.x),
			2.* lengthRatio.y / (1. + lengthRatio.y),
			2.* lengthRatio.z / (1. + lengthRatio.z)};
    my_position.x = lengthRatio.x * my_position.x + L_factor.x * my_velocity.x * timeStep;
    my_position.y = lengthRatio.y * my_position.y + L_factor.y * my_velocity.y * timeStep;
    my_position.z = lengthRatio.z * my_position.z + L_factor.z * my_velocity.z * timeStep;

    // Apply boundary condition
    float4 local_image = simBox->applyBoundaryCondition( my_position, array );
    my_image.x += local_image.x;
    my_image.y += local_image.y;
    my_image.z += local_image.z;
    my_image.w += local_image.w;

    // Save the integration in global memory.
    position[MyGP] = my_position;
    velocity[MyGP] = my_velocity;
    image[MyGP] = my_image; // This is done after the box is scaled (?????? NB)


  }
}



/// Updates the barostat state (new volume and volume velocity) on GPU kernel
__global__ void updateNPTLangevinState(float* barostatVirial,
				       float barostatFriction,
				       float barostatMass,
				       float targetConfPressure, float friction,
				       double2* barostatState, double volume,
				       float* randomForces, double timeStep)
{
  double2 current_barostatState = barostatState[0];
  double barostatForce = barostatVirial[0]/volume - targetConfPressure;
  float barostatRandomForce = randomForces[0] * sqrtf(barostatFriction / friction);


  double current_volume_velocity = current_barostatState.y;
  float inv_baro_mass = 1.f/barostatMass;
  float scaled_dt = 0.5f*timeStep*barostatFriction* inv_baro_mass;
  double b_tilde = 1./(1.+scaled_dt);
  double a_tilde = b_tilde*(1.-scaled_dt);

  
  // Leap-frog version of the barostat update equations
  double new_volume_vel = a_tilde * current_volume_velocity +  b_tilde*inv_baro_mass * (barostatForce*timeStep + barostatRandomForce);
  double new_volume = volume + timeStep * new_volume_vel;
  current_barostatState.x = new_volume/volume; // or just new_volume ???
  // or time*new_volume_vel/volume ???

  current_barostatState.y = new_volume_vel;
  barostatState[0] = current_barostatState;
}
