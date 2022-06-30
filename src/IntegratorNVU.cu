
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
#include "rumd/IntegratorNVU.h"
#include "rumd/Potential.h"
#include "rumd/ParseInfoString.h"

const std::string NVU_Error_Code1("This integrator is not associated with a Sample or the latter has no particles");
const std::string NVU_Error_Code2("Failed to dynamic_cast simulation box to RectangularSimulationBox");

////////////////////////////////////////////////////////////
// Constructors 
////////////////////////////////////////////////////////////

IntegratorNVU::IntegratorNVU( float dispLength, float potentialEnergyPerParticle ){
  timeStep = 1.f; // Not used
  displacementLength = dispLength;
  targetPotentialEnergy = potentialEnergyPerParticle;
  numParticlesSample = 0;
  
  if( cudaMalloc( (void**) &d_previousPotentialEnergy, sizeof(float) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("IntegratorNVU","IntegratorNVU","Malloc failed on d_previousPotentialEnergy") );
}

IntegratorNVU::~IntegratorNVU(){
  cudaFree(d_previousPotentialEnergy);
  FreeArrays();
}

////////////////////////////////////////////////////////////
// Class Methods
////////////////////////////////////////////////////////////

void IntegratorNVU::Integrate(){
  if(!S || !(S->GetNumberOfParticles()) || numParticlesSample != S->GetNumberOfParticles())
    throw( RUMD_Error("IntegratorNVU","Integrate", NVU_Error_Code1 ) );
  
  RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(S->GetSimulationBox());
  if(!testRSB) throw RUMD_Error("IntegratorNVU","Integrate", NVU_Error_Code2);
  
  integrateNVUAlgorithm<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_r, P->d_v, P->d_f, 
						    P->d_im, testRSB, testRSB->GetDevicePointer(), d_summationArrayForce, 
						    d_summationArrayCorrection, d_previousPotentialEnergy, 
						    numParticlesSample * targetPotentialEnergy, displacementLength, 
						    S->GetMeanMass() );
  
  float* forwardP = &d_summationArrayCorrection[numParticlesSample];
  cudaMemcpy( d_previousPotentialEnergy, forwardP, sizeof(float), cudaMemcpyDeviceToDevice );
}

// Constrains the motion on U.
void IntegratorNVU::CalculateConstraintForce() {
  RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(S->GetSimulationBox());
  if(!testRSB) throw RUMD_Error("IntegratorNVU","Integrate", NVU_Error_Code2);
  
  calculateConstraintForce<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_r, P->d_im, P->d_v, P->d_f, d_summationArrayForce, testRSB, 
						       testRSB->GetDevicePointer(), S->GetMeanMass() );
  
  sumIdenticalArrays( d_summationArrayForce, numParticlesSample, 3, 32 );
}

// Private. Step length and potential energy correction.
void IntegratorNVU::CalculateNumericalCorrection(){
  calculateNumericalCorrection<<< kp.grid, kp.threads >>>( numParticlesSample, P->d_v, P->d_f, d_summationArrayForce, 
							   d_summationArrayCorrection, d_previousPotentialEnergy, 
							   numParticlesSample * targetPotentialEnergy, S->GetMeanMass() );
  
  sumIdenticalArrays( d_summationArrayCorrection, numParticlesSample, 2, 32 );
}

void IntegratorNVU::FreeArrays(){
  if(numParticlesSample){
    cudaFree(d_particleVelocity);
    cudaFree(d_summationArrayForce);
    cudaFree(d_summationArrayCorrection);
  }
}

void IntegratorNVU::AllocateIntegratorState(){
  if(!S || !(S->GetNumberOfParticles())) // Consistency check with itg.
    throw( RUMD_Error("IntegratorNVU","AllocateIntegratorState", NVU_Error_Code1 ) );
  
  unsigned int newNumParticlesS = S->GetNumberOfParticles();
  
  if(newNumParticlesS != numParticlesSample){
    FreeArrays();
    numParticlesSample = newNumParticlesS;
    
    float h_previousPotentialEnergy = numParticlesSample * targetPotentialEnergy;
    cudaMemcpy( d_previousPotentialEnergy, &h_previousPotentialEnergy, sizeof(float), cudaMemcpyHostToDevice );
    
    // Blocks until the device has completed all preceding requested tasks. Host => Device is async.
    if( cudaDeviceSynchronize() != cudaSuccess ) 
      throw( RUMD_Error("IntegratorNVU","AllocateIntegratorState","CudaMemcpy failed: h_previousPotentialEnergy => d_previousPotentialEnergy") );
    
    // Allocate space on the GPU for the summation of the constraint force.
    if( cudaMalloc( (void**) &d_summationArrayForce, 3 * numParticlesSample * sizeof(float) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorNVU","AllocateIntegratorState","Malloc failed on d_summationArrayForce") );
    
    // Allocate space on the GPU for the summation of the numerical correction.
    if( cudaMalloc( (void**) &d_summationArrayCorrection, 2 * numParticlesSample * sizeof(float) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorNVU","AllocateIntegratorState","Malloc failed on d_summationArrayCorrection") );
    
    // Allocate space for the summation of the system velocity on GPU.
    if( cudaMalloc( (void**) &d_particleVelocity, numParticlesSample * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
      throw( RUMD_Error("IntegratorNVU","AllocateIntegratorState","Malloc failed on d_particleVelocity") );
    
    // Currently disabled in NVU (not sure that the integrator should be
    // controlling the output though)
    std::cout << "[Info] IntegratorNVU, disabling totalEnergy, temperature, kineticEnergy and pressure in energies file" << std::endl;
    S->SetOutputManagerMetaData("energies","totalEnergy",false);
    S->SetOutputManagerMetaData("energies","temperature",false);
    S->SetOutputManagerMetaData("energies","kineticEnergy",false);
    S->SetOutputManagerMetaData("energies","pressure", false);
  }
}

void IntegratorNVU::CalculateAfterForce(){
  if(!S || !(S->GetNumberOfParticles()) || numParticlesSample != S->GetNumberOfParticles())
    throw( RUMD_Error("IntegratorNVU","CalculateConstraintForce", NVU_Error_Code1 ));
  
  CalculateConstraintForce();
  CalculateNumericalCorrection(); 
}

void IntegratorNVU::InitializeFromInfoString(const std::string& infoStr, bool verbose) {
  std::vector<float> parameterList;
  std::string className = ParseInfoString(infoStr, parameterList);
  // check typename, if no match, raise error
  if(className != "IntegratorNVU")
    throw RUMD_Error("IntegratorNVU","InitializeFromInfoString",std::string("Wrong integrator type: ")+className);
  
  if(parameterList.size() != 2)
    throw RUMD_Error("IntegratorNVU","InitializeFromInfoString","Wrong number of parameters in infoStr");
  
  if(verbose)
    std::cout << "[Info] Initializing IntegratorNVU. displacement length:" << parameterList[0] << "; target potential energy:" << parameterList[1]  << std::endl;
  
  SetDisplacementLength(parameterList[0]);
  SetTargetPotentialEnergy(parameterList[1]);
}

////////////////////////////////////////////////////////////
// Set Methods
////////////////////////////////////////////////////////////

void IntegratorNVU::SetDisplacementLength( float step ){ displacementLength = step; }
void IntegratorNVU::SetTargetPotentialEnergy( float U ){ targetPotentialEnergy = U; };

void IntegratorNVU::SetMomentumToZero(){
  unsigned int nParticlesS;
  if(!S || !(nParticlesS = S->GetNumberOfParticles()))
    throw( RUMD_Error("IntegratorNVU","SetMomentumToZero", NVU_Error_Code1 ) );
  
  calculateVelocity<<< kp.grid, kp.threads >>>( nParticlesS, P->d_v, d_particleVelocity, S->GetMeanMass() );
  sumIdenticalArrays( d_particleVelocity, numParticlesSample, 1, 32 );
  zeroSystemVelocity<<<kp.grid, kp.threads>>>( nParticlesS, P->d_v, d_particleVelocity );
}

////////////////////////////////////////////////////////////
// Get Methods 
////////////////////////////////////////////////////////////

float IntegratorNVU::GetSimulationTimeStepSq() const { 
  if(!S || !(S->GetNumberOfParticles()))
    throw( RUMD_Error("IntegratorNVU","GetSimulationTimeStepSq", NVU_Error_Code1 ) );
  
  float* forwardP = &d_summationArrayForce[numParticlesSample];
  float constraintForceNumerator = 0; float constraintForceDenominator = 1.f;
  cudaMemcpy( &constraintForceNumerator, d_summationArrayForce, sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( &constraintForceDenominator, forwardP, sizeof(float), cudaMemcpyDeviceToHost );
  return (constraintForceNumerator/constraintForceDenominator) * S->GetMeanMass(); 
}

float IntegratorNVU::GetDisplacementLength() const { return displacementLength; }

float IntegratorNVU::GetSimulationDisplacementLength() const { 
  unsigned int nParticlesS = S->GetNumberOfParticles();
  if(!nParticlesS)
    throw( RUMD_Error("IntegratorNVU","GetSimulationDisplacementLength","Sample has no particles" ) );
  
  P->CopyVelFromDevice();
  float dispLength = 0;
  for(unsigned int i=0; i < S->GetNumberOfParticles(); i++){
    // <m_i> / m_i
    float reducedMass = 1.f / (S->GetMeanMass() * P->h_v[i].w); 
    dispLength += reducedMass * (P->h_v[i].x * P->h_v[i].x + P->h_v[i].y * P->h_v[i].y + P->h_v[i].z * P->h_v[i].z);
  }
  return sqrtf(dispLength);
} 

// Divided by numParticles.
float IntegratorNVU::GetTargetPotentialEnergy() const { return targetPotentialEnergy; };

// NOT divided by numParticles.
float IntegratorNVU::GetPreviousPotentialEnergy() const { 
  float previousU = 0;
  cudaMemcpy( &previousU, d_previousPotentialEnergy, sizeof(float), cudaMemcpyDeviceToHost );
  return previousU; 
};

std::string IntegratorNVU::GetInfoString(unsigned int precision) const{
  std::ostringstream infoStream;
  infoStream << "IntegratorNVU";
  infoStream << "," << std::setprecision(precision) << GetDisplacementLength();
  infoStream << "," << std::setprecision(precision) << GetTargetPotentialEnergy();
  return infoStream.str();
}


void IntegratorNVU::GetDataInfo(std::map<std::string, bool> &active,
				  std::map<std::string, std::string> &columnIDs) {

  active["simulationDisplacementLength"] = false;
  columnIDs["simulationDisplacementLength"] = "dispLength";

  active["instantaneousTimeStepSq"] = false;
  columnIDs["instantaneousTimeStepSq"] = "dt^2";
  
}

void IntegratorNVU::RemoveDataInfo(std::map<std::string, bool> &active,
				     std::map<std::string, std::string> &columnIDs) {

  // WHO ENSURES THAT THIS GETS CALLED????? WHEN A NEW INTEGRATOR IS SET I SUPPOSE
  
  active.erase("simulationDisplacementLength");
  columnIDs.erase("simulationDisplacementLength");

  active.erase("instantaneousTimeStepSq");
  columnIDs.erase("instantaneousTimeStepSq");
  }

void IntegratorNVU::GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active) {
  if(!S || !(S->GetNumberOfParticles())) // Consistency check with itg.
    throw( RUMD_Error("IntegratorNVU", __func__, NVU_Error_Code1 ) );

    if(active["simulationDisplacementLength"])
      dataValues["simulationDisplacementLength"] = GetSimulationDisplacementLength();
    if(active["instantaneousTimeStepSq"])
    dataValues["instantaneousTimeStepSq"] = GetSimulationTimeStepSq();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// NVU algorithm:  [ T. S. Ingebrigtsen, S. Toxvaerd, O. J. Heilmann, T. B. Schroeder and J. C. Dyre,
//                   J. Chem. Phys. 135, 104101 (2011) ] 
// NVU validation: [ T. S. Ingebrigtsen, S. Toxvaerd, T. B. Schroeder and J. C. Dyre,
//                   J. Chem. Phys. 135, 104102 (2011) ]
//////////////////////////////////////////////////////////////////////////////////////////////////////

template <class S> __global__ void integrateNVUAlgorithm( unsigned numParticles, float4* position, float4* velocity, 
							  float4* force, float4* image, S* simBox, float* simBoxPointer, 
							  float* constraintForce, float* displacementLengthCorrection, 
							  float* previousPotentialEnergy, float targetPotentialEnergy, 
							  float displacementLength, float meanMass ){
  
  if ( MyGP < numParticles ){ 
    float4 my_position = position[MyGP]; float4 my_velocity = velocity[MyGP]; 
    float4 my_force = force[MyGP]; float4 my_image = image[MyGP];

    // <m_i> / m_i
    float invReducedMass = meanMass * my_velocity.w; 
    
    float cForce = (constraintForce[0] + (previousPotentialEnergy[0] - targetPotentialEnergy)) / constraintForce[numParticles];
    float dispLengthCorrection = displacementLength / sqrtf(displacementLengthCorrection[0]);

    // Load the simulation box in 'local' memory to avoid bank conflicts.
    float array[simulationBoxSize];
    simBox->loadMemory( array, simBoxPointer );

    // Delta R(i+1/2).
    my_velocity.x = dispLengthCorrection * ( my_velocity.x + cForce * invReducedMass * my_force.x );
    my_velocity.y = dispLengthCorrection * ( my_velocity.y + cForce * invReducedMass * my_force.y ); 
    my_velocity.z = dispLengthCorrection * ( my_velocity.z + cForce * invReducedMass * my_force.z );
    
    // R(i+1).
    my_position.x += my_velocity.x;
    my_position.y += my_velocity.y;
    my_position.z += my_velocity.z;
    
    float4 local_image = simBox->applyBoundaryCondition( my_position, array );
    
    // Save the integration in global memory.
    position[MyGP] = my_position; 
    velocity[MyGP] = my_velocity; 

    my_image.x += local_image.x;
    my_image.y += local_image.y;
    my_image.z += local_image.z;
    my_image.w += local_image.w;

    image[MyGP] = my_image;
  }
}

////////////////////////////////////////////////////////////
// Constraint Force/Numerical Correction 
////////////////////////////////////////////////////////////

template <class S> __global__ void calculateConstraintForce( unsigned numParticles, float4* position, float4* image, float4* velocity, float4* force, 
							     float* constraintForceArray, S* simBox, float* simBoxPointer, float meanMass ){
  if( MyGP < numParticles ){
    float4 my_velocity = velocity[MyGP]; float4 my_force = force[MyGP]; 
    
    // Load the simulation box in 'local' memory to avoid bank conflicts.
    float array[simulationBoxSize];
    simBox->loadMemory( array, simBoxPointer );

    float invReducedMass = meanMass * my_velocity.w; 

    constraintForceArray[MyGP] = -2.f * (my_velocity.x * my_force.x + my_velocity.y * my_force.y + my_velocity.z * my_force.z);
    constraintForceArray[numParticles+MyGP] = invReducedMass * (my_force.x * my_force.x + my_force.y * my_force.y + my_force.z * my_force.z);
  }
}

__global__ void calculateNumericalCorrection( unsigned numParticles, float4* velocity, float4* force, 
					      float* constraintForce, float* numericalCorrection, float* previousPotentialEnergy, 
					      float targetPotentialEnergy, float meanMass ){
  if( MyGP < numParticles ){
    float4 my_velocity = velocity[MyGP]; float4 my_force = force[MyGP]; 

    float invReducedMass = meanMass * my_velocity.w; 
    float reducedMass = 1.f / invReducedMass;
    
    float constraintF = 1.f;

    constraintF = ( constraintForce[0] + (previousPotentialEnergy[0] - targetPotentialEnergy) ) / constraintForce[numParticles];
    
    float3 normVector = { (my_velocity.x + constraintF * invReducedMass * my_force.x), 
			  (my_velocity.y + constraintF * invReducedMass * my_force.y),  
			  (my_velocity.z + constraintF * invReducedMass * my_force.z) };
    
    numericalCorrection[MyGP] = reducedMass * (normVector.x * normVector.x + normVector.y * normVector.y + normVector.z * normVector.z);
    numericalCorrection[numParticles+MyGP] = my_force.w;
  }
}

////////////////////////////////////////////////////////////
// Zero the mean system velocity 
////////////////////////////////////////////////////////////

__global__ void zeroSystemVelocity( unsigned numParticles, float4* velocity, float4* particleMomentum ){
  if( MyGP < numParticles ){
    float4 my_velocity = velocity[MyGP]; 
    float4 totalMomentum = particleMomentum[0];
    float invTotalReducedMass = 1.f / totalMomentum.w;
    
    // Subtract off total momentum (convert to velocity).
    my_velocity.x -= invTotalReducedMass * totalMomentum.x;
    my_velocity.y -= invTotalReducedMass * totalMomentum.y;
    my_velocity.z -= invTotalReducedMass * totalMomentum.z;
    
    velocity[MyGP] = my_velocity;
  }
}

__global__ void calculateVelocity( unsigned numParticles, float4* velocity, float4* particleMomentum, float meanMass ){
  if( MyGP < numParticles ){
    float4 my_momentum = velocity[MyGP]; 
    // m_i / <m_i>
    float reducedMass = 1.f / (my_momentum.w * meanMass);
    
    // Convert to momenta and mass.
    my_momentum.x *= reducedMass;
    my_momentum.y *= reducedMass;
    my_momentum.z *= reducedMass;
    my_momentum.w = reducedMass;
    
    // Store momenta in shared mem.
    particleMomentum[MyGP] = my_momentum;
  }
}
