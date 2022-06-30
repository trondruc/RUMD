
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
#include "rumd/IntegratorNVT.h"
#include "rumd/ParseInfoString.h"

#include <iostream>

const std::string NVT_Error_Code1("There is no integrator associated with Sample or the latter has no particles");


////////////////////////////////////////////////////////////
// Constructors 
////////////////////////////////////////////////////////////

// This constructor gives NVE behavior
IntegratorNVT::IntegratorNVT(float timeStep) {
  thermostatOn = ( 0x00000000 );
  thermostatRelaxationTime = 0.2f;  

  this->timeStep = timeStep;
  targetTemperature = 0.;
  temperatureChangePerTimeStep = 0.;

  numParticlesSample = 0;
  AllocateFromConstructor();
}

// This constructor gives NVT behavior
IntegratorNVT::IntegratorNVT(float timeStep, double targetTemperature,
			     double thermostatRelaxationTime) {
  thermostatOn = ~( 0x00000000 );

  this->timeStep = timeStep;
  this->targetTemperature = targetTemperature; 
  this->thermostatRelaxationTime = thermostatRelaxationTime;

  temperatureChangePerTimeStep = 0.;

  numParticlesSample = 0;
  AllocateFromConstructor();
}

IntegratorNVT::~IntegratorNVT(){
  cudaFree(d_thermostatState);
  FreeArrays();
}

////////////////////////////////////////////////////////////
// Class Methods
////////////////////////////////////////////////////////////

// Integrate Nose-Hoover dynamics.
void IntegratorNVT::Integrate(){
  if(!S || !(numParticlesSample))
    throw( RUMD_Error("IntegratorNVT","Integrate", NVT_Error_Code1 ) );
  
  RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(S->GetSimulationBox());
  LeesEdwardsSimulationBox* testLESB = dynamic_cast<LeesEdwardsSimulationBox*>(S->GetSimulationBox());
  if (testLESB)
    integrateNVTAlgorithm<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_r, P->d_v, P->d_f, P->d_im, testLESB, 
						      testLESB->GetDevicePointer(), d_thermostatState, d_thermostatKineticEnergy, timeStep, thermostatOn );
  else
    integrateNVTAlgorithm<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_r, P->d_v, P->d_f, P->d_im, testRSB, 
						      testRSB->GetDevicePointer(), d_thermostatState, d_thermostatKineticEnergy, timeStep, thermostatOn );

  if(thermostatOn)
    UpdateThermostatState();
}

// Private. Update the thermostat state.
void IntegratorNVT::UpdateThermostatState(){
  sumIdenticalArrays( d_thermostatKineticEnergy, numParticlesSample, 1, 32 );

  if(targetTemperature < 0.)
    throw RUMD_Error("IntegratoNVT","UpdateThermostatState","targetTemperature is negative");

  float omega2 = 4.*M_PI*M_PI/(thermostatRelaxationTime * thermostatRelaxationTime);
  // Needs update S->GetNumberOfDOFs() to subtract 3*particle type.
  updateNVTThermostatState<<<1, 1>>>( d_thermostatKineticEnergy, d_thermostatState, omega2, targetTemperature, S->GetNumberOfDOFs(), timeStep );
  targetTemperature += temperatureChangePerTimeStep;
}

void IntegratorNVT::AllocateFromConstructor(){
  // Allocate space on the GPU for the thermostat state.
  if( cudaMalloc( (void**) &d_thermostatState, sizeof(double) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("IntegratorNVT","IntegratorNVT","Malloc failed on d_thermostatState") );
  
  // Initialize the state on the GPU.
  SetThermostatState(0);

  // Blocks until the device has completed all preceding requested tasks. 
  // Host => Device is async.
  if( cudaDeviceSynchronize() != cudaSuccess ) 
    throw( RUMD_Error("IntegratorNVT","IntegratorNVT","CudaMemcpy failed: d_thermostatState") ); 
}

// Allocates memory dependent on the N particles.
void IntegratorNVT::AllocateIntegratorState(){
  if(!S || !(S->GetNumberOfParticles())) // Consistency check.
    throw( RUMD_Error("IntegratorNVT","AllocateIntegratorState", NVT_Error_Code1 ) );
  
  unsigned int newNumParticlesS = S->GetNumberOfParticles();
  
  if(newNumParticlesS != numParticlesSample){
    FreeArrays();
    numParticlesSample = newNumParticlesS;
    
    // Allocate space for the summation of kinetic energy on GPU.
    if( cudaMalloc( (void**) &d_thermostatKineticEnergy, numParticlesSample * sizeof(float) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorNVT","AllocateIntegratorState","Malloc failed on d_thermostatKineticEnergy") );
    
    // Allocate space for the summation of momentum on GPU.
    if( cudaMalloc( (void**) &d_particleMomentum, numParticlesSample * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorNVT","AllocateIntegratorState","Malloc failed on d_particleMomentum") );
  }
}

// Frees memory dependent on numParticles
void IntegratorNVT::FreeArrays(){
  if(numParticlesSample){
    cudaFree(d_thermostatKineticEnergy);
    cudaFree(d_particleMomentum);
  }
}

////////////////////////////////////////////////////////////
// Set Methods 
////////////////////////////////////////////////////////////

void IntegratorNVT::SetRelaxationTime( float tau ){ 
  thermostatRelaxationTime = tau; 
  SetThermostatState(0.);
} 

void IntegratorNVT::SetTargetTemperature( double T, bool reset ){  
  targetTemperature = T;
  if(reset)
    SetThermostatState(0.);
}

void IntegratorNVT::SetThermostatState( double Ps ){
  //cudaMemcpy( d_thermostatState, &Ps, sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_thermostatState, &Ps, sizeof(double), cudaMemcpyHostToDevice );
}

void IntegratorNVT::SetThermostatOn( bool on ){ 
  if(on)
    thermostatOn = ~( 0x00000000 );
  else
    thermostatOn = ( 0x00000000 );
}

void IntegratorNVT::SetThermostatOnParticleType( unsigned type, bool on ){
  if( type > ( ( sizeof(thermostatOn) * 8 ) - 1 ) ) 
    throw( RUMD_Error("IntegratorNVT","SetThermostatOnParticleType", "The particle type number is larger than 31" ) );
  
  bool typeOnOff = thermostatOn & ( 1U << type );
 
  if( typeOnOff != unsigned(on) ){ // Flip the bit.
    thermostatOn = ( thermostatOn ^ ( 1U << type ) );
    std::cout << "Thermostat on particle type " << type << " was changed from " << !on << " to " << on << std::endl;
  }
}

void IntegratorNVT::SetMomentumToZero(){
  if(!S || !numParticlesSample)
    throw( RUMD_Error("IntegratorNVT","SetMomentumToZero", NVT_Error_Code1 ) );
  
  calculateMomentum<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_v, d_particleMomentum );
  sumIdenticalArrays( d_particleMomentum, numParticlesSample, 1, 32 );
  zeroTotalMomentum<<<kp.grid, kp.threads>>>( numParticlesSample, P->d_v, d_particleMomentum );
}

void IntegratorNVT::SetTemperatureChangePerTimeStep(double temperatureChangePerTimeStep) {
  this->temperatureChangePerTimeStep = temperatureChangePerTimeStep;
}


////////////////////////////////////////////////////////////
// Get Methods 
////////////////////////////////////////////////////////////

void IntegratorNVT::GetDataInfo(std::map<std::string, bool> &active,
				  std::map<std::string, std::string> &columnIDs) {
    active["thermostat_Ps"] = false;
    columnIDs["thermostat_Ps"] = "Ps";
    active["thermostat_KE"] = false;
    columnIDs["thermostat_KE"] = "th_KE";  

    active["v_com_x"] = false;
    columnIDs["v_com_x"] = "v_com_x";
    active["v_com_y"] = false;
    columnIDs["v_com_y"] = "v_com_y";
    active["v_com_z"] = false;
    columnIDs["v_com_z"] = "v_com_z";
}

void IntegratorNVT::RemoveDataInfo(std::map<std::string, bool> &active,
				     std::map<std::string, std::string> &columnIDs) {
  active.erase("thermostat_Ps");
  columnIDs.erase("thermostat_Ps");
  active.erase("thermostat_KE");
  columnIDs.erase("thermostat_KE");

  active.erase("v_com_x");
  columnIDs.erase("v_com_x");
  active.erase("v_com_y");
  columnIDs.erase("v_com_y");
  active.erase("v_com_z");
  columnIDs.erase("v_com_z");
}

void IntegratorNVT::GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active) {
  if(!S || !(S->GetNumberOfParticles()))
    throw( RUMD_Error("IntegratorNVU", __func__, NVT_Error_Code1 ) );
  
  if(active["thermostat_Ps"])
    dataValues["thermostat_Ps"] = GetThermostatState();
  if(active["thermostat_KE"])
    dataValues["thermostat_KE"] = GetThermostatKineticEnergy()/S->GetNumberOfParticles();

  bool need_v_com = active["v_com_x"] || active["v_com_y"] || active["v_com_z"];
  
  double4 totalMomentum = { 0, 0, 0, 0 };
  bool need_ke = (active["kineticEnergy"] || active["temperature"] || active["pressure"] || active["totalEnergy"]);  
  if(need_v_com) {      
    // if need_ke is not true then velocities have not been copied
    totalMomentum = GetTotalMomentum(!need_ke);
    if(active["v_com_x"])
      dataValues["v_com_x"] = totalMomentum.x / totalMomentum.w;
    if(active["v_com_y"])
      dataValues["v_com_y"] = totalMomentum.y / totalMomentum.w;
    if(active["v_com_z"])
      dataValues["v_com_z"] = totalMomentum.z / totalMomentum.w;
  }
  
}



std::string IntegratorNVT::GetInfoString(unsigned int precision) const {
  std::ostringstream infoStream;
  if(thermostatOn) { 
    infoStream << "IntegratorNVT";
    infoStream << "," << std::setprecision(precision) << GetTimeStep();
    infoStream << "," << std::setprecision(precision) << GetTargetTemperature();
    infoStream << "," << std::setprecision(precision) << GetRelaxationTime();
    infoStream << "," << std::setprecision(precision) << GetThermostatState();
  }
  else infoStream << "IntegratorNVE" << "," << std::setprecision(precision) << timeStep;
  return infoStream.str();
}

void IntegratorNVT::InitializeFromInfoString(const std::string& infoStr, bool verbose) {
  std::vector<float> parameterList;
  std::string className = ParseInfoString(infoStr, parameterList);
  // check typename, if no match, raise error
  bool nvt = (className == "IntegratorNVT");
  if(!nvt && className != "IntegratorNVE")
    throw RUMD_Error("IntegratorNVT","InitializeFromInfoString","Expected IntegratorNVT or IntegratorNVE");
  unsigned int requiredNumParams = 1;
  if(nvt) requiredNumParams = 4;
  if(parameterList.size() != requiredNumParams)
    throw RUMD_Error("IntegratorNVT","InitializeFromInfoString","Wrong number of parameters in infoStr");

  if(verbose)
    std::cout << "[Info] Initializing " << className << ". time step:" << parameterList[0];
  SetTimeStep(parameterList[0]);

  if(nvt) {
    if(verbose)
      std::cout << "; target temperature:" << parameterList[1] << "; relaxation time:" << parameterList[2] << "; thermostat state:" << parameterList[3];
    SetTargetTemperature(parameterList[1]);
    SetRelaxationTime(parameterList[2]);
    SetThermostatState(parameterList[3]);
  }
  if(verbose)
    std::cout << std::endl;
}

float IntegratorNVT::GetRelaxationTime() const { return thermostatRelaxationTime; }

double IntegratorNVT::GetTargetTemperature() const { return targetTemperature; }

double IntegratorNVT::GetThermostatState() const {
  double thermostatState = 0;
  
  if( thermostatOn )
    cudaMemcpy( &thermostatState, d_thermostatState, sizeof(double), cudaMemcpyDeviceToHost );
  
  return thermostatState;
}

float IntegratorNVT::GetThermostatKineticEnergy() const {

  float thermostatKineticEnergy = 0.;
  cudaMemcpy(&thermostatKineticEnergy, d_thermostatKineticEnergy, sizeof(float), cudaMemcpyDeviceToHost);
  thermostatKineticEnergy /=2.;
      
  return thermostatKineticEnergy;

}

// State: r(t) and v(t-h/2). Return sum( 0.5 m v^2(t) )
double IntegratorNVT::GetKineticEnergy(bool copy) const{
  if(!numParticlesSample)
    throw( RUMD_Error("IntegratorNVT","GetKineticEnergy","Sample has no particles" ) );

  if(copy) {
    P->CopyVelFromDevice(); P->CopyForFromDevice();
  }

  double thermostatState = 0.f;
  if(thermostatOn) thermostatState = GetThermostatState();  
  double totalKineticEnergy = 0;

  for(unsigned int i=0; i < numParticlesSample; i++){
    double mass = 1.f / P->h_v[i].w;
    float timeStepTimesinvMass = P->h_v[i].w * timeStep;
    
    //bool applyThermostat = thermostatOn & ( 1U << int(P->h_r[i].w) );
    // use host type array ( note: assumes type does not change )
    bool applyThermostat = thermostatOn & ( 1U << (P->h_Type[i]) );

    float factor = 0.;
    if( applyThermostat )
      factor = 0.5f * timeStep * thermostatState;
    float plus = 1.f / ( 1.f + factor );
    float minus = 1.f - factor;
    
    float velocityFx = plus * ( minus * P->h_v[i].x + timeStepTimesinvMass * P->h_f[i].x );
    float velocityFy = plus * ( minus * P->h_v[i].y + timeStepTimesinvMass * P->h_f[i].y );
    float velocityFz = plus * ( minus * P->h_v[i].z + timeStepTimesinvMass * P->h_f[i].z );
    
    // Velocity: v^2(t-h/2)
    double velocityB = P->h_v[i].x * P->h_v[i].x + P->h_v[i].y * P->h_v[i].y + P->h_v[i].z * P->h_v[i].z;
    
    // Velocity: v^2(t+h/2)
    double velocityF = velocityFx * velocityFx + velocityFy * velocityFy + velocityFz * velocityFz;
    
    // The kinetic energy v^2(t) should be calculated like this in a LFA.
    totalKineticEnergy += mass * ( ( velocityB + velocityF ) / 2.0 );
  }
  return 0.5 * totalKineticEnergy;
}


////////////////////////////////////////////////////////////
// NVE algorithm: Leap frog 
// NVT algorithm: [S. Toxvaerd, Mol. Phys. 72, 159 (1991)]
////////////////////////////////////////////////////////////

template <class S> __global__ void integrateNVTAlgorithm( unsigned numParticles, float4* position, float4* velocity, 
							  float4* force, float4* image, S* simBox, float* simBoxPointer, 
							  double* thermostatState, float* thermostatKineticEnergy, 
							  float timeStep, unsigned thermostatOn ){
  
  if ( MyGP < numParticles ){ 
    float4 my_position = position[MyGP]; float4 my_velocity = velocity[MyGP]; 
    float4 my_force = force[MyGP]; float4 my_image = image[MyGP];
    float localThermostatState = (float) thermostatState[0];

    // Load the simulation box in local memory to avoid bank conflicts.
    float array[simulationBoxSize];
    simBox->loadMemory( array, simBoxPointer );

    // Check if myType should apply the thermostat.
    bool applyThermostat = thermostatOn & ( 1U << __float_as_int(my_position.w) );
    
    // Thermostat variable. NVE: = 0.    
    if( !applyThermostat )
      localThermostatState = 0.f;
    
    float factor = 0.5f * localThermostatState * timeStep;
    float plus = 1.f / ( 1.f + factor ); // Possibly change to exp(...)
    float minus = 1.f - factor; // Possibly change to exp(...)

    // Update to v(t+h/2).
    my_velocity.x = plus * ( minus * my_velocity.x + my_velocity.w * my_force.x * timeStep ); 
    my_velocity.y = plus * ( minus * my_velocity.y + my_velocity.w * my_force.y * timeStep );
    my_velocity.z = plus * ( minus * my_velocity.z + my_velocity.w * my_force.z * timeStep );
    
    // Update to r(t+h).
    my_position.x += my_velocity.x * timeStep; 
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
    
    // Save kinetic energy for a possible NVT calculation.
    if(applyThermostat)
      thermostatKineticEnergy[MyGP] = ( my_velocity.x * my_velocity.x + my_velocity.y * my_velocity.y + my_velocity.z * my_velocity.z ) / my_velocity.w;
    else
      thermostatKineticEnergy[MyGP] = 0;
  }
}

////////////////////////////////////////////////////////////
// Update Nose Hoover State 
////////////////////////////////////////////////////////////

// Updates the thermostat state on GPU.

__global__ void updateNVTThermostatState( float* thermostatKineticEnergy, double* thermostatState, float omega2, float targetTemperature, int degreesOfFreedom, float timeStep ) {

// thermostatKineticEnergy[0] is actually twice the kinetic energy
  float ke_deviation = thermostatKineticEnergy[0] / (degreesOfFreedom * targetTemperature) - 1.f;
  // if KE exceeds set temperature by more than a factor of ten, we limit the
  // relaxation rate to avoid unstable behavior
  const float ke_dev_max = 10.f;
  if(ke_deviation > ke_dev_max)
    ke_deviation = ke_dev_max;
  thermostatState[0] += timeStep * omega2 * ke_deviation;
    
}

////////////////////////////////////////////////////////////
// Set Momentum to Zero 
////////////////////////////////////////////////////////////

// Sets the total momentum of the system to zero.
__global__ void zeroTotalMomentum( unsigned int numParticles, float4* velocity, float4* particleMomentum ){
  if( MyGP < numParticles ){
    float4 my_velocity = velocity[MyGP]; 
    float4 totalMomentum = particleMomentum[0];
    float invTotalMass = 1.f / totalMomentum.w;
    
    // Subtract off total momentum (convert to velocity).
    my_velocity.x -= invTotalMass * totalMomentum.x;
    my_velocity.y -= invTotalMass * totalMomentum.y;
    my_velocity.z -= invTotalMass * totalMomentum.z;
    
    velocity[MyGP] = my_velocity;
  }
}

__global__ void calculateMomentum( unsigned int numParticles, float4* velocity, float4* particleMomentum ){
  if( MyGP < numParticles ){
    float4 my_momentum = velocity[MyGP]; 
    float mass = 1.f / my_momentum.w;
    
    // Convert to momenta and mass.
    my_momentum.x *= mass;
    my_momentum.y *= mass;
    my_momentum.z *= mass;
    my_momentum.w = mass;
    
    particleMomentum[MyGP] = my_momentum;
  }
}
