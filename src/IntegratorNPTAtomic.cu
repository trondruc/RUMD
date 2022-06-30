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
#include "rumd/IntegratorNPTAtomic.h"
#include "rumd/ParseInfoString.h"

#include <iostream>
#include <string>



const std::string NPT_Error_Code1("There is no integrator associated with Sample or the latter has no particles");
const double max_barostat_state = 100.0;

////////////////////////////////////////////////////////////
// Constructors
////////////////////////////////////////////////////////////

///  Constructor for the NPT Integrator
/**
*	@param timeStep						Timestep for the integrator
*	@param targetTemperature			External temperature of the NPT ensemble
*	@param targetPressure				External pressure of the NPT ensemble
*	@param thermostatRelaxationTime 	Relaxation time of thermostat	(0.4 is a std value for LJ)
*	@param barostatRelaxationTime		Relaxation time of barostat		(25.0 is a possible choice for LJ)
*/
IntegratorNPTAtomic::IntegratorNPTAtomic(float timeStep, float targetTemperature, float thermostatRelaxationTime, float targetPressure, float barostatRelaxationTime)
{

  this->timeStep = timeStep;
  numParticlesSample = 0;
  massFactor = 1.0 / ( 4.0 * M_PI * M_PI );	//Is the conversion constant from period of oscillation to frequency ( T = 1/2*Pi*frequency )

  // Thermostat parameters
  thermostatOn = ~( 0x00000000 );
  this->thermostatRelaxationTime = thermostatRelaxationTime;
  this->targetTemperature = targetTemperature;
  temperatureChangePerTimeStep = 0.;

  // Barostat parameters
  this->barostatRelaxationTime = barostatRelaxationTime;
  this->targetPressure = targetPressure;
  pressureChangePerTimeStep = 0.;

  AllocateFromConstructor();

}

/// Destructor (free up memory)
IntegratorNPTAtomic::~IntegratorNPTAtomic()
{
  cudaFree(d_thermostatState);
  cudaFree(d_barostatState);
  FreeArrays();
}


////////////////////////////////////////////////////////////
// Class Methods
////////////////////////////////////////////////////////////


/// Called by the main loop to make one integration step.
void IntegratorNPTAtomic::Integrate()
{

  if(!S || !(numParticlesSample))
    throw( RUMD_Error("IntegratorNPTAtomic","Integrate", NPT_Error_Code1) );

  // Update positions on GPU. Exit and print error if Lees-Edwards Simulation Box is used.
  RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(S->GetSimulationBox());
  if(testRSB)
    integrateNPTAlgorithm<<< kp.grid, kp.threads >>>(	numParticlesSample, P->d_r, P->d_v, P->d_f, P->d_im, P->d_w,
							testRSB , testRSB->GetDevicePointer(),
							d_thermostatState,  d_barostatState,
							d_thermostatKineticEnergy, d_barostatVirial, timeStep, S->GetNumberOfDOFs() );
  else
    throw( RUMD_Error("IntegratorNPTAtomic","Integrate", "Cannot use Lees-Edwards simulation box with NPT integrator") );

  // Update barostat and thermostat state
  UpdateState();

}


/// Update the barostat and thermostat states and calculate total kin energy and total virial.
void IntegratorNPTAtomic::UpdateState()
{

  // Compute kinetic energy and virial by summing particle contributions
  sumIdenticalArrays( d_thermostatKineticEnergy, numParticlesSample, 1, 32 );
  sumIdenticalArrays( d_barostatVirial         , numParticlesSample, 1, 32 );

  // Assert that temperature is positive.
  if(targetTemperature < 0.)
    throw RUMD_Error("IntegratorNPT","UpdateThermostatState","targetTemperature is negative");

  // Avoid small thermostat mass by resetting a temperature close to zero
  double temp = targetTemperature;
  if(temp < 0.001)
    temp = 0.001;

  // Compute masses. These quantities are calculated at every timestep because you could need to change temp if you use a ramp.
  // Note: massFactor = 1/4pi*pi
  double Nf = S->GetNumberOfDOFs();
  double thermostatMass	= (Nf-1) * 3.0 * temp * thermostatRelaxationTime * thermostatRelaxationTime * massFactor;
  double barostatMass	= (Nf-1) * 3.0 * temp * barostatRelaxationTime * barostatRelaxationTime * massFactor;
  // Update the baro- and thermostat state on GPU
  updateNPTState<<<1, 1>>>( 	d_thermostatKineticEnergy, d_barostatVirial,
				d_thermostatState, thermostatMass, targetTemperature,
				d_barostatState, barostatMass, targetPressure,
				S->GetSimulationBox()->GetVolume(),
				S->GetNumberOfDOFs(), timeStep );

  // Change Volume of box
  double barostatState = GetBarostatState();
  double timestep = GetTimeStep();
  double frac_length_change = pow(1. + timestep * barostatState, 1.0/3.0)-1.;
  
  // barostat equations: Equation 2.9 in [G. J. Martyna, D. J. Tobias and M. L. Klein, J. Chem. Phys.101 (1994), 4177] 
  S->GetSimulationBox()->ScaleBoxFraction(frac_length_change);

  // Print Stuff for debugging (and exit if exploding)
  if(barostatState > max_barostat_state)
    throw RUMD_Error("IntegratorNPTAtomic", __func__, std::string("Barostat unstable; try longer barostat relaxation time."));


  // Update external pressure and temperature
  targetTemperature += temperatureChangePerTimeStep;
  targetPressure += pressureChangePerTimeStep;

}


/// Allocate space on the GPU for the thermostat state and barostat state
void IntegratorNPTAtomic::AllocateFromConstructor()
{

  // Allocate space on the GPU for the thermostat state.
  if( cudaMalloc( (void**) &d_thermostatState, sizeof(double) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("IntegratorNPTAtomic","IntegratorNPTAtomic","Malloc failed on d_thermostatState") );

  // Allocate space on the GPU for the barostat state.
  if( cudaMalloc( (void**) &d_barostatState, sizeof(double) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("IntegratorNPTAtomic","IntegratorNPTAtomic","Malloc failed on d_barostatState") );

  // Initialize the state on the GPU.
  SetThermostatState(0.);
  SetBarostatState(0.);

  // Blocks until the device has completed all preceding requested tasks.
  // Host => Device is async.
  if( cudaDeviceSynchronize() != cudaSuccess )
    throw( RUMD_Error("IntegratorNPTAtomic","IntegratorNPTAtomic","CudaMemcpy failed: d_thermostatState and/or d_barostatState") );

}


/// Allocates memory dependent on the N particles.
void IntegratorNPTAtomic::AllocateIntegratorState()
{

  // Consistency check.
  if(!S || !(S->GetNumberOfParticles()))
    throw( RUMD_Error("IntegratorNPTAtomic","AllocateIntegratorState", NPT_Error_Code1 ) );

  unsigned int newNumParticlesS = S->GetNumberOfParticles();

  if(newNumParticlesS != numParticlesSample){
    FreeArrays();
    numParticlesSample = newNumParticlesS;

  // Allocate space for the summation of kinetic energy on GPU.
  if( cudaMalloc( (void**) &d_thermostatKineticEnergy, numParticlesSample * sizeof(double) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("IntegratorNPTAtomic","AllocateIntegratorState","Malloc failed on d_thermostatKineticEnergy") );

  // Allocate space for the summation of momentum on GPU.
  if( cudaMalloc( (void**) &d_particleMomentum, numParticlesSample * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("IntegratorNPTAtomic","AllocateIntegratorState","Malloc failed on d_particleMomentum") );

  // Allocate space for the summation of pressure on GPU.
  if( cudaMalloc( (void**) &d_barostatVirial, numParticlesSample * sizeof(double) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("IntegratorNPTAtomic","AllocateIntegratorState","Malloc failed on d_barostatVirial") );
  }

}


/// Frees memory dependent on numParticles
void IntegratorNPTAtomic::FreeArrays()
{
  if(numParticlesSample){
    cudaFree(d_thermostatKineticEnergy);
    cudaFree(d_particleMomentum);
    cudaFree(d_barostatVirial);
  }
}



////////////////////////////////////////////////////////////
// Set Methods
////////////////////////////////////////////////////////////


void IntegratorNPTAtomic::SetRelaxationTime( double tauTemperature )
{
  thermostatRelaxationTime = tauTemperature;
  SetThermostatState(0.0);
}



void IntegratorNPTAtomic::SetBarostatRelaxationTime( double tauPressure )
{
  barostatRelaxationTime = tauPressure;
  SetBarostatState(0.0);
}



void IntegratorNPTAtomic::SetTargetTemperature( double T )
{
  targetTemperature = T;
  SetThermostatState(0.0);
}



void IntegratorNPTAtomic::SetTargetPressure( double P )
{
  targetPressure = P;
  SetBarostatState(0.0);
}



void IntegratorNPTAtomic::SetThermostatState( double Ps )
{
  cudaMemcpy( d_thermostatState, &Ps, sizeof(double), cudaMemcpyHostToDevice );
}



void IntegratorNPTAtomic::SetBarostatState( double Ps )
{
  cudaMemcpy( d_barostatState, &Ps, sizeof(double), cudaMemcpyHostToDevice );
}



void IntegratorNPTAtomic::SetThermostatOn( bool on )
{
  if(on)
    thermostatOn = ~( 0x00000000 );
  else
    //thermostatOn = ( 0x00000000 );  //TODO thermostat off could be the switch NPT/NPH (in analogy with NVT/NVE integrator)
    throw( RUMD_Error("IntegratorNPTAtomic","SetThermostatOn", "Can not yet turn baro/thermostat off" ) );
}



/*
// TODO: Decide if this makes sense (ie is consistent with NPT algorithm)
/// Make a bit flip in unsigned thermostatOn
void IntegratorNPTAtomic::SetThermostatOnParticleType( unsigned type, bool on )
{
  if( type > ( ( sizeof(thermostatOn) * 8 ) - 1 ) )
    throw( RUMD_Error("IntegratorNPTAtomic","SetThermostatOnParticleType", "The particle type number is larger than 31" ) );

  bool typeOnOff = thermostatOn & ( 1U << type );

  if( typeOnOff != unsigned(on) ){ // Flip the bit.
    thermostatOn = ( thermostatOn ^ ( 1U << type ) );
    std::cout << "Thermostat on particle type " << type << " was changed from " << !on << " to " << on << std::endl;
  }
}
*/


/// Set Momentum To Zero
void IntegratorNPTAtomic::SetMomentumToZero()
{
  if(!S || !numParticlesSample)
    throw( RUMD_Error("IntegratorNPTAtomic","SetMomentumToZero", NPT_Error_Code1 ) );

  calculateMomentumNPT<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_v, d_particleMomentum );
  sumIdenticalArrays( d_particleMomentum, numParticlesSample, 1, 32 );
  zeroTotalMomentumNPT<<<kp.grid, kp.threads>>>( numParticlesSample, P->d_v, d_particleMomentum );
}



void IntegratorNPTAtomic::SetTemperatureChangePerTimeStep(double temperatureChangePerTimeStep)
{
  this->temperatureChangePerTimeStep = temperatureChangePerTimeStep;
}



void IntegratorNPTAtomic::SetPressureChangePerTimeStep(double pressureChangePerTimeStep)
{
  this->pressureChangePerTimeStep = pressureChangePerTimeStep;
}



////////////////////////////////////////////////////////////
// Get Methods
////////////////////////////////////////////////////////////


/// Used to save thermo- and barostatstate to header of restart file, appearing after "integrator=".
std::string IntegratorNPTAtomic::GetInfoString(unsigned int precision) const
{
  std::ostringstream infoStream;
  //if(thermostatOn) { TODO: the algorithm should work for NPH also but is not tested yet.
  //                         it could be interesting to check that in future.
    infoStream << "IntegratorNPTAtomic";
    infoStream << "," << std::setprecision(precision) << GetTimeStep();
    infoStream << "," << std::setprecision(precision) << GetTargetTemperature();
    infoStream << "," << std::setprecision(precision) << GetRelaxationTime();
    infoStream << "," << std::setprecision(precision) << GetThermostatState();
    infoStream << "," << std::setprecision(precision) << GetTargetPressure();
    infoStream << "," << std::setprecision(precision) << GetBarostatRelaxationTime();
    infoStream << "," << std::setprecision(precision) << GetBarostatState();
  //}
  //else infoStream << "IntegratorNPH" << "," << std::setprecision(precision) << timeStep;
  return infoStream.str();
}



/// Used when restarting a simulation to set to thermo- and barostat state to old values, appearing after "integrator=".
void IntegratorNPTAtomic::InitializeFromInfoString(const std::string& infoStr, bool verbose)
{
  std::vector<float> parameterList;
  std::string className = ParseInfoString(infoStr, parameterList);
  // check typename, if no match, raise error
  bool npt = (className == "IntegratorNPTAtomic");
  if(!npt)
    throw RUMD_Error("IntegratorNPTAtomic","InitializeFromInfoString","Expected IntegratorNPTAtomic.");
  unsigned int requiredNumParams = 7;
  if(parameterList.size() != requiredNumParams)
    throw RUMD_Error("IntegratorNPTAtomic","InitializeFromInfoString","Wrong number of parameters in infoStr");

  if(verbose)
    std::cout << "[Info] Initializing IntegratorNPTAtomic. time step:" << parameterList[0] << "target temperature:" << parameterList[1] << "; relaxation time:" << parameterList[2] << "thermostat state:" << parameterList[3]  << "; target pressure:" << parameterList[4]  << "; barostat relaxation time:" << parameterList[5]  << "; barostat state:" << parameterList[6] << std::endl;

  SetTimeStep(parameterList[0]);
  SetTargetTemperature(parameterList[1]);
  SetRelaxationTime(parameterList[2]);
  SetThermostatState(parameterList[3]);
  SetTargetPressure(parameterList[4]);
  SetBarostatRelaxationTime(parameterList[5]);
  SetBarostatState(parameterList[6]);

}



double IntegratorNPTAtomic::GetRelaxationTime() const { return thermostatRelaxationTime; }



double IntegratorNPTAtomic::GetBarostatRelaxationTime() const { return barostatRelaxationTime; }



double IntegratorNPTAtomic::GetTargetTemperature() const { return targetTemperature; }



double IntegratorNPTAtomic::GetTargetPressure() const { return targetPressure; }



double IntegratorNPTAtomic::GetThermostatState() const {
  double thermostatState = 0.0;

  if( thermostatOn )
    cudaMemcpy( &thermostatState, d_thermostatState, sizeof(double), cudaMemcpyDeviceToHost );

  return thermostatState;
}




double IntegratorNPTAtomic::GetBarostatState() const
{
	double barostatState = 0.0;
	cudaMemcpy( &barostatState, d_barostatState, sizeof(double), cudaMemcpyDeviceToHost );
	return barostatState ;
}



/// Return kinetic energy \f$ \frac{1}{2}\sum m v^2 \f$ of the LFL algorithm. @param copy copy from device if true.
double IntegratorNPTAtomic::GetKineticEnergy(bool copy) const
{
  if(!numParticlesSample)
    throw( RUMD_Error("IntegratorNPTAtomic","GetKineticEnergy","Sample has no particles" ) );

  if(copy) {
    P->CopyVelFromDevice(); P->CopyForFromDevice();
  }

  double thermostatState = 0.0;
  if(thermostatOn) thermostatState = GetThermostatState();
  double totalKineticEnergy = 0.0;
  double barostatState=GetBarostatState();

  unsigned int degreesOfFreedom = S->GetNumberOfDOFs();

  for(unsigned int i=0; i < numParticlesSample; i++){
    double mass = 1.0 / P->h_v[i].w;
    double timeStepTimesinvMass = P->h_v[i].w * timeStep;

    double factor = 0.;
    factor = 0.5 * timeStep * ( thermostatState + barostatState * ( 1.0 + 3.0/degreesOfFreedom) ) ;

    double plus = 1.0 / ( 1.0 + factor );
    double minus = 1.0 - factor;

    double velocityFx = plus * ( minus * P->h_v[i].x + timeStepTimesinvMass * P->h_f[i].x );
    double velocityFy = plus * ( minus * P->h_v[i].y + timeStepTimesinvMass * P->h_f[i].y );
    double velocityFz = plus * ( minus * P->h_v[i].z + timeStepTimesinvMass * P->h_f[i].z );

    // Velocity: v^2(t-h/2)
    double velocityB = P->h_v[i].x * P->h_v[i].x + P->h_v[i].y * P->h_v[i].y + P->h_v[i].z * P->h_v[i].z;

    // Velocity: v^2(t+h/2)
    double velocityF = velocityFx * velocityFx + velocityFy * velocityFy + velocityFz * velocityFz;

    // The kinetic energy v^2(t) should be calculated like this in a LFA.
    totalKineticEnergy += mass * ( ( velocityB + velocityF ) / 2.0 );
  }
  return 0.5 * totalKineticEnergy;
}







/// Return the system total momentum. @param copy from devise if true
double4 IntegratorNPTAtomic::GetTotalMomentum(bool copy) const
{
  if(!S || !numParticlesSample)
    throw( RUMD_Error("IntegratorNPTAtomic","GetTotalMomentum","Sample has no particles" ) );

  if(copy)
    P->CopyVelFromDevice();
  double4 totalMomentum = { 0.0, 0.0, 0.0, 0.0 };
  for(unsigned int i=0; i < numParticlesSample; i++){
    double particleMass = 1.0 / P->h_v[i].w;
    totalMomentum.x += particleMass * P->h_v[i].x;
    totalMomentum.y += particleMass * P->h_v[i].y;
    totalMomentum.z += particleMass * P->h_v[i].z;
    totalMomentum.w += particleMass; // Total mass.
  }
  return totalMomentum;
}





/// Move particle positions and velocities on GPU kernel and compute particle contribution for kinetic energy and viral
template <class S> __global__ void integrateNPTAlgorithm( unsigned numParticles, float4* position, float4* velocity,
							  float4* force, float4* image, float4* virial, S* simBox, float* simBoxPointer,
							  double* thermostatState, double* barostatState,
							  float* thermostatKineticEnergy, float* barostatVirial,
							  double timeStep, unsigned int degreeOfFreedom )
{

  if ( MyGP < numParticles ){
    float4 my_position = position[MyGP]; float4 my_velocity = velocity[MyGP];
    float4 my_force = force[MyGP];  float4 my_image = image[MyGP];
    float4 my_virial = virial[MyGP];
    double localThermostatState = (double) thermostatState[0];
    double localBarostatState = (double) barostatState[0];

    // Load the simulation box in local memory to avoid bank conflicts.
    float array[simulationBoxSize];
    simBox->loadMemory( array, simBoxPointer );

	// barostat equations: Equation 2.9 in [G. J. Martyna, D. J. Tobias and M. L. Klein, J. Chem. Phys.101 (1994), 4177] 
    // Update to v(t+h/2).
    double factor = 0.5 * timeStep * (localThermostatState + localBarostatState * (1.0 + 3.0/degreeOfFreedom));
    double plus = 1.0 / ( 1.0 + factor );	// Possibly change to exp(...)
    double minus = 1.0 - factor; 			// Possibly change to exp(...)
    my_velocity.x = plus * ( minus * my_velocity.x + my_velocity.w * my_force.x * timeStep );
    my_velocity.y = plus * ( minus * my_velocity.y + my_velocity.w * my_force.y * timeStep );
    my_velocity.z = plus * ( minus * my_velocity.z + my_velocity.w * my_force.z * timeStep );

    // Update to r(t+h).
    double rFactor= 1.0 + localBarostatState * timeStep;
    my_position.x = rFactor * my_position.x + my_velocity.x * timeStep;
    my_position.y = rFactor * my_position.y + my_velocity.y * timeStep;
    my_position.z = rFactor * my_position.z + my_velocity.z * timeStep;

    // Apply boundary condition
    float4 local_image = simBox->applyBoundaryCondition( my_position, array );
    my_image.x += local_image.x;
    my_image.y += local_image.y;
    my_image.z += local_image.z;
    my_image.w += local_image.w;

    // Save the integration in global memory.
    position[MyGP] = my_position;
    velocity[MyGP] = my_velocity;
    image[MyGP] = my_image; // This is done after the box is scaled

    // Save kinetic energy for a possible NVT calculation. 
    double kineticEnergy = ( my_velocity.x * my_velocity.x + my_velocity.y * my_velocity.y + my_velocity.z * my_velocity.z ) / my_velocity.w;
    thermostatKineticEnergy[MyGP] = kineticEnergy; // This value is the particle square velocity, therefore when summed over particles will give 2*ekin

    // Compute contribution to pressure
    barostatVirial[MyGP] = my_virial.w;

  }
}



/// Updates the baro- and thermostat state on GPU kernel
__global__ void updateNPTState( float* thermostatKineticEnergy, float* barostatVirial,
				double* thermostatState, double thermostatMass, double targetTemperature,
				double* barostatState, double barostatMass, double targetPressure,
				double volume,
				unsigned int degreesOfFreedom, double timeStep )
{

  // thermostatKineticEnergy[0] is actually twice the kinetic energy
  double oldThermostatState  = thermostatState[0];
  double oldBarostatState = barostatState[0];
  //double volume = (double) boxLengthX * (double) boxLengthY * (double) boxLengthZ;
  double ekin = 0.5*thermostatKineticEnergy[0];                      // the variable thermostatKineticEnergy[0] is 2*ekin

  double pressure = (barostatVirial[0]/6.0 + 2.0/3.0*ekin)/volume;   // the factor 6 is because barostatVirial[0] is 2 times the total virial
  
  // Thermostatstate update
  thermostatState[0] = oldThermostatState + timeStep/thermostatMass * ( 2.0*ekin - ((double)degreesOfFreedom+1.0) * targetTemperature + oldBarostatState*oldBarostatState*barostatMass );


  
  //barostatState[0] *= ( 1.0 - timeStep * oldThermostatState );
  //barostatState[0] += 3.0 * timeStep * ( volume * ( pressure - targetPressure ) + 2.0 * ekin / degreesOfFreedom ) / barostatMass;

  barostatState[0] = oldBarostatState * ( 1.0 - timeStep * oldThermostatState ) + 3.0 * timeStep * ( volume * ( pressure - targetPressure ) + 2.0 * ekin / degreesOfFreedom ) / barostatMass;
  

}





/// Sets the total momentum of the system to zero on GPU kernel
__global__ void zeroTotalMomentumNPT( unsigned int numParticles, float4* velocity, float4* particleMomentum )
{
  if( MyGP < numParticles ){
    float4 my_velocity = velocity[MyGP];
    float4 totalMomentum = particleMomentum[0];
    float invTotalMass = 1.0 / totalMomentum.w;

    // Subtract off total momentum (convert to velocity).
    my_velocity.x -= invTotalMass * totalMomentum.x;
    my_velocity.y -= invTotalMass * totalMomentum.y;
    my_velocity.z -= invTotalMass * totalMomentum.z;

    velocity[MyGP] = my_velocity;
  }
}





/// Compute particle momentum on GPU kernel
__global__ void calculateMomentumNPT( unsigned int numParticles, float4* velocity, float4* particleMomentum )
{
  if( MyGP < numParticles ){
    float4 my_momentum = velocity[MyGP];
    float mass = 1.0 / my_momentum.w;

    // Convert to momenta and mass.
    my_momentum.x *= mass;
    my_momentum.y *= mass;
    my_momentum.z *= mass;
    my_momentum.w = mass;

    particleMomentum[MyGP] = my_momentum;
  }
}
