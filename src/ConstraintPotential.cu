
/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/ConstraintPotential.h"
#include "rumd/ConstraintPotentialHelper.h"
#include "rumd/IntegratorNVT.h"
#include "rumd/MoleculeData.h"
#include "rumd/Sample.h"
#include "rumd/rumd_algorithms.h"

#include <iostream>
#include <cstdio>

using namespace std;

//////////////////////////////////////////////////
// Constructors 
//////////////////////////////////////////////////

ConstraintPotential::ConstraintPotential(){
  ID_String = "PotConstraint";
  bond_pot_class = 2;
  matrixAllocated = 0;
  numIterations = 10;
  sizeMem = 32;
  linearMolecules = false;
  std::cout << "[Info] Calculation of (atomic stresses) is not supported for constraints." << std::endl;
}

ConstraintPotential::~ConstraintPotential(){
  Free();
}

// Free allocated memory.
void ConstraintPotential::Free(){
  if(matrixAllocated){
    cudaFreeHost(h_constraintInfo);
    
    cudaFree(d_dimLinearSystems); 
    cudaFree(d_dimConsecLinearSystems);
    cudaFree(d_dimConsecSqLinearSystems);
    cudaFree(d_constraintInfo); 
    cudaFree(d_constraintInfoPerParticle);
    cudaFree(d_tempForce); 
    cudaFree(d_sumStandardDeviation);
    cudaFree(d_A); 
    cudaFree(d_b);
    cudaFree(d_x);

    matrixAllocated = 0;
  }
}

//////////////////////////////////////////////////
// Class methods 
//////////////////////////////////////////////////
void ConstraintPotential::SetParams(unsigned bond_type, float length_param) {
  BondPotential::SetParams(bond_type, length_param, 0., true);
}

// Calculates the constraint force.
void ConstraintPotential::CalcF(bool __attribute__((unused))initialize, bool __attribute__((unused))calc_stresses){

  RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(sample->GetSimulationBox());
  IntegratorNVT* testNVT = dynamic_cast<IntegratorNVT*>(sample->GetIntegrator());
  
  if(!testNVT || !testRSB)
    throw( RUMD_Error("ConstraintPotential","CalcF", "The chosen constraint algorithm only supports NVE/NVT in combination with a rectangular simulation box." ) );
  
  float timeStep = testNVT->GetTimeStep(); 
  float Ps = testNVT->GetThermostatState();

  buildConstraintsOnParticle<<< gridBLS, threadBLS >>>( d_dimLinearSystems, d_dimConsecLinearSystems, d_constraintInfo, d_constraintInfoPerParticle, 
						  particleData->d_r, particleData->d_v, testRSB, testRSB->GetDevicePointer(), 
						  totalNumberOfConstraints, numberOfLinearSystems );

  buildLinearSystem<<< gridBLS, threadBLS >>>( d_A, d_b, d_dimLinearSystems, d_dimConsecLinearSystems, d_dimConsecSqLinearSystems, d_constraintInfo, 
					 d_constraintInfoPerParticle, particleData->d_r, particleData->d_v, particleData->d_f, testRSB, 
					 testRSB->GetDevicePointer(), totalNumberOfConstraints, Ps, timeStep, numberOfLinearSystems );
  
  dim3 threadUCF = thread;
  threadUCF.y = maxConstraintsPerMolecule;
  
  
  // set misc array to zero
  cudaMemset(particleData->d_misc, 0, particleData->GetNumberOfParticles() * sizeof(float4));

  // Some number of iterations.
  for(unsigned i=0; i < numIterations; i++){

    // Gauss elimination.
    if(linearMolecules)
      solveTridiagonalLinearSystems( d_x, d_b, d_A, d_dimLinearSystems, d_dimConsecLinearSystems, d_dimConsecSqLinearSystems, numberOfLinearSystems, maxConstraintsPerMolecule );
    else
      solveLinearSystems( d_x, d_b, d_A, d_dimLinearSystems, d_dimConsecLinearSystems, d_dimConsecSqLinearSystems, numberOfLinearSystems, maxConstraintsPerMolecule );

    
    // Update the RHS to iteration.
    cudaMemcpy( d_tempForce, particleData->d_f, sample->GetNumberOfParticles() * sizeof(float4), cudaMemcpyDeviceToDevice );

    
    updateConstraintForce<0><<< grid, threadUCF >>>( particleData->d_r, d_tempForce, particleData->d_misc, d_x, d_dimLinearSystems, 
    						  d_dimConsecLinearSystems, d_constraintInfo, testRSB, testRSB->GetDevicePointer(), 
    						  numberOfLinearSystems );

    // update b
    updateRHS<<< grid1, thread1 >>>( d_x, d_b, particleData->d_r, particleData->d_v, d_tempForce, d_constraintInfo, testRSB, 
    				     testRSB->GetDevicePointer(), Ps, timeStep, totalNumberOfConstraints );
  }
  
  // Update the total force with the final constraint force + virial.
  updateConstraintForce<1><<< grid, threadUCF >>>( particleData->d_r, particleData->d_f, particleData->d_misc, d_x, d_dimLinearSystems, 
						d_dimConsecLinearSystems, d_constraintInfo, testRSB, testRSB->GetDevicePointer(), numberOfLinearSystems );
  
}


void ConstraintPotential::GetDataInfo(std::map<std::string, bool> &active, std::map<std::string, std::string> &columnIDs) {
    active["constraintVirial"] = false;
    columnIDs["constraintVirial"] = "con_W";

    active["bondLengthStdDev"] = false;
    columnIDs["bondLengthStdDev"] = "bond_dev";


}

void ConstraintPotential::RemoveDataInfo(std::map<std::string, bool> &active, std::map<std::string, std::string> &columnIDs) {
  active.erase("constraintVirial");
  columnIDs.erase("constraintVirial");

  active.erase("bondLengthStdDev");
  columnIDs.erase("bondLengthStdDev");
}


void ConstraintPotential::GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active) {
  // TBD As it is now, GetVirial gets called twice, each time causing a device to host copy and then summing on the host. Want to include the particle constraint virial contributions in the total particle virial 
  if(active["constraintVirial"])
    dataValues["constraintVirial"] = GetVirial();


  if(active["bondLengthStdDev"]) {
    RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(sample->GetSimulationBox());
    
    calculateStandardDeviation<<< grid1, thread1 >>>( testRSB, testRSB->GetDevicePointer(), particleData->d_r, d_constraintInfo, totalNumberOfConstraints, d_sumStandardDeviation );
    sumIdenticalArrays( d_sumStandardDeviation, totalNumberOfConstraints, 1, 32 );
    
    float std = 0;
    cudaMemcpy( &std, d_sumStandardDeviation, sizeof(float), cudaMemcpyDeviceToHost );
    
    
    dataValues["bondLengthStdDev"] = sqrt(std/double(totalNumberOfConstraints));
    //cout << setprecision(6) << "ConstraintPotential: Bond-length standard deviation " << sqrt(std/double(totalNumberOfConstraints)) << endl;
  }
}


// Returns the contribution to the virial from this potential
double ConstraintPotential::GetVirial() {

  double constraintW = 0;   
  particleData->CopyMiscFromDevice();
  for ( unsigned int i=0; i < sample->GetNumberOfParticles(); i++ )
    constraintW += particleData->h_misc[i].y;
  
    return constraintW / 3.0;
}



// Updates the constraints to be consistent with the sorting algorithm.
void ConstraintPotential::UpdateAfterSorting( unsigned int* d_old_index, unsigned int* __attribute__((unused))d_new_index ){
  cudaMemset( d_constraintInfoPerParticle, 0, sample->GetNumberOfParticles()*totalNumberOfConstraints*sizeof(float3) );
  UpdateConstraintsAfterSorting<<< grid1, thread1, sizeMem*sizeof(unsigned) >>>( d_old_index, sample->GetNumberOfParticles(), d_constraintInfo, totalNumberOfConstraints );
}


void ConstraintPotential::Initialize() {
  BondPotential::Initialize();
  UpdateConstraintLayout();
  sample->UpdateDegreesOfFreedom(GetNumberOfConstraints());
}

void ConstraintPotential::UpdateConstraintLayout(){
  // Free previously allocated memory.
  Free(); matrixAllocated = 1;
  
  std::list<float4> cList; numberOfLinearSystems = 1;
  BuildConstraintGraph( sample->GetMoleculeData(), cList );
  
  // Count the number of linear systems.
  float4 previous = cList.front();
  for( list<float4>::iterator it = cList.begin(); it != cList.end(); it++ ){
    if( (*it).w != previous.w )
      numberOfLinearSystems++;
    
    previous.w = (*it).w;
  }
  
  // Initialize the size of linear systems.
  unsigned constraintList[numberOfLinearSystems]; 
  unsigned constraintConsecList[numberOfLinearSystems];
  unsigned constraintConsecSqList[numberOfLinearSystems];
  
  maxConstraintsPerMolecule = 0;
  for(unsigned i = 0; i < numberOfLinearSystems; i++){ 
    constraintList[i] = 0;
    constraintConsecList[i] = 0;
    constraintConsecSqList[i] = 0;
  }
  
  // Update size of linear systems.
  for (std::list<float4>::iterator it = cList.begin(); it != cList.end(); it++)
    constraintList[(unsigned) (*it).w]++;
  
  // Consecutive sums of index purposes.
  unsigned consecSum = 0; unsigned consecSumSq = 0;
  for( unsigned i=0; i < numberOfLinearSystems; i++){
    constraintConsecList[i] = consecSum;
    constraintConsecSqList[i] = consecSumSq;
    
    consecSum += constraintList[i];
    consecSumSq += ( constraintList[i] * constraintList[i] );
  }
  
  // Total and maximum number of constraints.
  totalNumberOfConstraints = 0;
  totalSquaredNumberOfConstraints = 0;
  maxConstraintsPerMolecule = 0;
  
  for(unsigned i = 0; i < numberOfLinearSystems; i++){
    if ( maxConstraintsPerMolecule < constraintList[i] )
      maxConstraintsPerMolecule = constraintList[i];
    totalNumberOfConstraints += constraintList[i];
    totalSquaredNumberOfConstraints += ( constraintList[i] * constraintList[i] );
  }
  
  // Allocate host memory.
  
  if( cudaMallocHost( (void**) &h_constraintInfo, totalNumberOfConstraints * sizeof(float3) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ConstraintPotential","ConstraintPotential","Malloc failed on h_constraintInfo") );
  
  // Allocate device memory.
  
  if( cudaMalloc( (void**) &d_dimLinearSystems, numberOfLinearSystems * sizeof(unsigned) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ConstraintPotential","ConstraintPotential","Malloc failed on d_dimLinearSystems") );
  
  if( cudaMalloc( (void**) &d_dimConsecLinearSystems, numberOfLinearSystems * sizeof(unsigned) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ConstraintPotential","ConstraintPotential","Malloc failed on d_dimConsecLinearSystems") );
  
  if( cudaMalloc( (void**) &d_dimConsecSqLinearSystems, numberOfLinearSystems * sizeof(unsigned) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ConstraintPotential","ConstraintPotential","Malloc failed on d_dimConsecSqLinearSystems") );
  
  if( cudaMalloc( (void**) &d_constraintInfo, totalNumberOfConstraints * sizeof(float3) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ConstraintPotential","ConstraintPotential","Malloc failed on d_constraintInfo") );
  
  if( cudaMalloc( (void**) &d_sumStandardDeviation, totalNumberOfConstraints * sizeof(float) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ConstraintPotential","ConstraintPotential","Malloc failed on d_sumStandardDeviation") );

  if( cudaMalloc( (void**) &d_tempForce, sample->GetNumberOfParticles() * sizeof(float4) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ConstraintPotential","ConstraintPotential","Malloc failed on d_tempForce") );

  if( cudaMalloc( (void**) &d_constraintInfoPerParticle, sample->GetNumberOfParticles()*totalNumberOfConstraints*sizeof(float3) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ConstraintPotential","ConstraintPotential","Malloc failed on d_constraintInfoPerParticle") );
  
  if( cudaMalloc( (void**) &d_A, totalSquaredNumberOfConstraints * sizeof(float) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ConstraintPotential","ConstraintPotential","Malloc failed on d_A") );
  
  if( cudaMalloc( (void**) &d_b, totalNumberOfConstraints * sizeof(float) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ConstraintPotential","ConstraintPotential","Malloc failed on d_b") );

  if( cudaMalloc( (void**) &d_x, totalNumberOfConstraints * sizeof(float) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("ConstraintPotential","ConstraintPotential","Malloc failed on d_x") );

  unsigned counter = 0;
  for(std::list<float4>::iterator it = cList.begin(); it != cList.end(); it++){
    h_constraintInfo[counter].x = (*it).x;
    h_constraintInfo[counter].y = (*it).y;
    h_constraintInfo[counter].z = (*it).z;
    counter++;
  }
  
  // Copy constraint pair information and the dimension of the linear systems.
  cudaMemcpy( d_dimLinearSystems, constraintList, numberOfLinearSystems * sizeof(unsigned), cudaMemcpyHostToDevice );  
  cudaMemcpy( d_dimConsecLinearSystems, constraintConsecList, numberOfLinearSystems * sizeof(unsigned), cudaMemcpyHostToDevice );  
  cudaMemcpy( d_dimConsecSqLinearSystems, constraintConsecSqList, numberOfLinearSystems * sizeof(unsigned), cudaMemcpyHostToDevice );  
  cudaMemcpy( d_constraintInfo, h_constraintInfo, totalNumberOfConstraints * sizeof(float3), cudaMemcpyHostToDevice );

  // Clear arrays.
  cudaMemset( d_constraintInfoPerParticle, 0, sample->GetNumberOfParticles()*totalNumberOfConstraints*sizeof(float3) );
  cudaMemset( d_A, 0, totalSquaredNumberOfConstraints * sizeof(float) );
  cudaMemset( d_b, 0, totalNumberOfConstraints * sizeof(float) );
  cudaMemset( d_x, 0, totalNumberOfConstraints * sizeof(float) );
  
  grid.x = ((numberOfLinearSystems + sizeMem - 1) / sizeMem); grid.y = 1; 
  thread.x = sizeMem; thread.y = 1;
  
  grid1.x = ((totalNumberOfConstraints + sizeMem - 1) / sizeMem); grid1.y = 1; 
  thread1.x = sizeMem; thread1.y = 1;

  gridBLS = grid;
  threadBLS = thread;
  
  while(threadBLS.y < maxConstraintsPerMolecule && threadBLS.y < 32)
    threadBLS.y *= 2;

  threadBLS.x = 32 / threadBLS.y;
  gridBLS.x = ((numberOfLinearSystems + threadBLS.x - 1) / threadBLS.x);


}  

// Writes if the constraints are satisfied. 
void ConstraintPotential::WritePotential(SimulationBox* __attribute__((unused))simBox){

  std::cout << "[Info] ConstraintPotential::WritePotential is deprecated because this information can now be written to the energies file via the key bondLengthStdDev." << std::endl;
  RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(sample->GetSimulationBox());

  calculateStandardDeviation<<< grid1, thread1 >>>( testRSB, testRSB->GetDevicePointer(), particleData->d_r, d_constraintInfo, totalNumberOfConstraints, d_sumStandardDeviation );
  sumIdenticalArrays( d_sumStandardDeviation, totalNumberOfConstraints, 1, 32 );
  
  float std = 0;
  cudaMemcpy( &std, d_sumStandardDeviation, sizeof(float), cudaMemcpyDeviceToHost );
  
  cout << setprecision(6) << "ConstraintPotential: Bond-length standard deviation " << sqrt(std/double(totalNumberOfConstraints)) << endl;
}

//////////////////////////////////////////////////
// Build Linear system for NVE/NVT
//////////////////////////////////////////////////

__device__ float dot_product( float4* a, float4* b ){ return (a->x * b->x + a->y * b->y + a->z * b->z); }


__device__ float4 construct_B_ij( float4 r_ij, float4 v_ij, float4 s_ij, float xi, float xi_plus_inv, float xi_minus, float dt ){
  float4 B_ij_vector = { 0.f, 0.f, 0.f, 0.f };
  
  float xi_plus_inv_sq = xi_plus_inv * xi_plus_inv;
  float factor = 1.f - 0.5f * xi * dt * xi_plus_inv;
  float dt_sq = dt * dt;
  
  B_ij_vector.x = r_ij.x * factor + xi_plus_inv_sq * ( dt_sq * s_ij.x + dt * xi_minus * v_ij.x );
  B_ij_vector.y = r_ij.y * factor + xi_plus_inv_sq * ( dt_sq * s_ij.y + dt * xi_minus * v_ij.y );
  B_ij_vector.z = r_ij.z * factor + xi_plus_inv_sq * ( dt_sq * s_ij.z + dt * xi_minus * v_ij.z );
  return B_ij_vector;
}

// Construct the C_ij scalar.
__device__ float construct_C_ij( float4 r_ij, float4 v_ij, float4 s_ij, float xi, 
				 float xi_plus_inv, float xi_minus, float xi_sq, float dt, float l_ij ){
  float C_ij = 0.f;
  float dt2 = dt*dt;
  float xi_plus_inv_sq = xi_plus_inv * xi_plus_inv;
  
  C_ij = xi_plus_inv_sq * ( xi_sq * dot_product(&v_ij,&v_ij) 
			    + 0.5f * dt2 * dot_product(&s_ij,&s_ij) + xi_minus * dt * dot_product(&s_ij,&v_ij) )
    + (1.f - 0.5f * xi * dt * xi_plus_inv) * dot_product(&r_ij,&s_ij) - (xi * xi_plus_inv) * dot_product(&r_ij,&v_ij);
  C_ij *= 2.f*dt2;
  // Numerical correction
  C_ij += (dot_product(&r_ij,&v_ij))* 2.f*dt - dt2 * dot_product(&v_ij,&v_ij)
  // Perfect
  + ( dot_product(&r_ij,&r_ij) - l_ij * l_ij );// / ( 2.f * dt * dt );    
  
  return C_ij;
}

// Build p_ij see NVE article. The Matrix is N x G.
template <class S> __global__ void buildConstraintsOnParticle( unsigned* dimLinearSystems, unsigned* dimConsecLinearSystems, float3* d_constraintInfo, 
							       float3* constraintInfoPerParticle, float4* position, float4* velocity, 
							       S* simBox, float* simBoxPointer, unsigned totalNumConstraints, 
							       float totalLinearConstraints ){
  
  if( MyGP < totalLinearConstraints ){
    float array[simulationBoxSize];
    simBox->loadMemory( array, simBoxPointer );
  
    unsigned myDimension = dimLinearSystems[MyGP];
    unsigned myConsecDimension = dimConsecLinearSystems[MyGP];

    int i = threadIdx.y;
    if (i < myDimension) {
    //for(int i = 0; i < myDimension; i++){
	float3 info = d_constraintInfo[myConsecDimension+i];
	unsigned index1 = (unsigned) info.x;
	unsigned index2 = (unsigned) info.y;
	
	float4 position1 = position[index1];
	float4 position2 = position[index2];
	
	float4 velocity1 = velocity[index1];
	float4 velocity2 = velocity[index2];
	
	float4 before = simBox->calculateDistance( position1, position2, array );
	
	// Particle i.
	constraintInfoPerParticle[index1*totalNumConstraints+i].x = before.x * velocity1.w;
	constraintInfoPerParticle[index1*totalNumConstraints+i].y = before.y * velocity1.w;
	constraintInfoPerParticle[index1*totalNumConstraints+i].z = before.z * velocity1.w;
	
	// Particle j.
	constraintInfoPerParticle[index2*totalNumConstraints+i].x = - before.x * velocity2.w;
	constraintInfoPerParticle[index2*totalNumConstraints+i].y = - before.y * velocity2.w;
	constraintInfoPerParticle[index2*totalNumConstraints+i].z = - before.z * velocity2.w;
    }
  }
}

template < class S > __global__ void buildLinearSystem( float* A, float* b, unsigned* dimLinearSystems, unsigned* dimConsecLinearSystems, 
							unsigned* dimConsecSqLinearSystems, float3* d_constraintInfo, 
							float3* constraintInfoPerParticle, float4* position, float4* velocity, 
							float4* force, S* simBox, float* simBoxPointer, unsigned totalNumConstraints,
							float xi, float timeStep, float totalLinearConstraints ){
  
  if( MyGP < totalLinearConstraints ){
    // Load the simulation box in local memory to avoid bank conflicts.
    float array[simulationBoxSize];
    simBox->loadMemory( array, simBoxPointer );
    
    unsigned myDimension = dimLinearSystems[MyGP];
    unsigned myConsecDimension = dimConsecLinearSystems[MyGP];
    unsigned myConsecSqDimension = dimConsecSqLinearSystems[MyGP];
    // Calculate the abbreviations.
    float xi_factor = 0.5f * xi * timeStep;
    float xi_plus_inv = 1.f / (1.f + xi_factor); 
    float xi_minus = 1.f - xi_factor;
    float xi_sq = 1.f + xi_factor * xi_factor;

    for(int i = threadIdx.y; i < myDimension; i += blockDim.y) {
      float3 info = d_constraintInfo[myConsecDimension+i];
      unsigned index1 = (unsigned) info.x;
      unsigned index2 = (unsigned) info.y;
      float length = info.z;
      

      
      float4 position1 = position[index1];
      float4 position2 = position[index2];
      
      float4 velocity1 = velocity[index1];
      float4 velocity2 = velocity[index2];
      
      float4 force1 = force[index1];
      float4 force2 = force[index2];
      
      float4 before = simBox->calculateDistance( position1, position2, array );
      
      float4 v_dimer = { velocity1.x - velocity2.x, velocity1.y - velocity2.y, velocity1.z - velocity2.z, 0 };
      
      float4 s_dimer = { force1.x * velocity1.w - force2.x * velocity2.w,
			 force1.y * velocity1.w - force2.y * velocity2.w,
			 force1.z * velocity1.w - force2.z * velocity2.w, 0 };
      
      float4 Bij = construct_B_ij( before, v_dimer, s_dimer, xi, xi_plus_inv, xi_minus, timeStep );
      
      for(int j=0; j < myDimension; j++){
	float4 difference = { constraintInfoPerParticle[index1*totalNumConstraints+j].x - constraintInfoPerParticle[index2*totalNumConstraints+j].x,
			      constraintInfoPerParticle[index1*totalNumConstraints+j].y - constraintInfoPerParticle[index2*totalNumConstraints+j].y,
			      constraintInfoPerParticle[index1*totalNumConstraints+j].z - constraintInfoPerParticle[index2*totalNumConstraints+j].z,
			      0 };
	
	// Update A. Some numerical error here when removing 2*dt*dt. Find out why...
	A[myConsecSqDimension+i*myDimension+j] = (2.f*timeStep*timeStep)*dot_product(&Bij, &difference);
      }
      
      // Update b.
      b[myConsecDimension+i] = - construct_C_ij( before, v_dimer, s_dimer, xi, xi_plus_inv, xi_minus, xi_sq, timeStep, length );//*(2.f*timeStep*timeStep);
    }
  }
}

//////////////////////////////////////////////////
// Update force with constraint force: NVE/NVT
//////////////////////////////////////////////////

// Iterate RHS by doing one integration step with temporary constraint force.
template < class S > __global__ void updateRHS( float* lagrangeMultiplier, float* b, float4* position, float4* velocity, 
						float4* force, float3* d_constraintInfo, S* simBox, float* simBoxPointer, 
						float localThermostatState, float timeStep, float totalConstraints ){
  if( MyGP < totalConstraints ){
    float array[simulationBoxSize];
    simBox->loadMemory( array, simBoxPointer );
    float my_b = b[MyGP];
        
    float3 info = d_constraintInfo[MyGP];
    unsigned index1 = (unsigned) info.x;
    unsigned index2 = (unsigned) info.y;
    float length = info.z;
        
    float4 position1 = position[index1]; float4 position2 = position[index2];
    float4 velocity1 = velocity[index1]; float4 velocity2 = velocity[index2];
    float4 force1 = force[index1]; float4 force2 = force[index2];
    

    float factor = 0.5f * localThermostatState * timeStep;
    float plus = 1.f / ( 1.f + factor ); 
    float minus = 1.f - factor; 
    
    // Update to v(t+h/2).
    velocity1.x = plus * ( minus * velocity1.x + velocity1.w * force1.x * timeStep ); 
    velocity1.y = plus * ( minus * velocity1.y + velocity1.w * force1.y * timeStep );
    velocity1.z = plus * ( minus * velocity1.z + velocity1.w * force1.z * timeStep );
    
    // Update to r(t+h).
    position1.x += velocity1.x * timeStep; 
    position1.y += velocity1.y * timeStep; 
    position1.z += velocity1.z * timeStep; 
    
    // Update to v(t+h/2).
    velocity2.x = plus * ( minus * velocity2.x + velocity2.w * force2.x * timeStep ); 
    velocity2.y = plus * ( minus * velocity2.y + velocity2.w * force2.y * timeStep );
    velocity2.z = plus * ( minus * velocity2.z + velocity2.w * force2.z * timeStep );
    
    // Update to r(t+h).
    position2.x += velocity2.x * timeStep; 
    position2.y += velocity2.y * timeStep; 
    position2.z += velocity2.z * timeStep; 

    // Update RHS.
    my_b += (length * length) - (double) simBox->calculateDistance( position1, position2, array ).w;
    b[MyGP] = my_b;
  }
}

// Updates the constraint force and virial for each linear system.
template <int updateVirial, class S> __global__ void updateConstraintForce( float4* position, float4* force, float4* virial, float* lagrangeMultiplier, 
									    unsigned* dimLinearSystems, unsigned* dimConsecLinearSystems, 
									    float3* d_constraintInfo, S* simBox, float* simBoxPointer, 
									    float totalLinearConstraints ){


  
  //if( MyGP < totalLinearConstraints ){
  if( threadIdx.x  + blockIdx.x  * blockDim.x < totalLinearConstraints ){
    float array[simulationBoxSize];
    simBox->loadMemory( array, simBoxPointer );
    
    unsigned myDimension = dimLinearSystems[MyGP];
    unsigned myConsecDimension = dimConsecLinearSystems[MyGP];

    //for(int i=myConsecDimension; i < myConsecDimension + myDimension; i++) {
    if(threadIdx.y < myDimension) {
      int i = myConsecDimension + threadIdx.y;
      float3 info = d_constraintInfo[i]; // Slow.
      unsigned index1 = (unsigned) info.x;
      unsigned index2 = (unsigned) info.y;
      
      float4 position1 = position[index1];
      float4 position2 = position[index2];

      float myLagrangeMultiplier = lagrangeMultiplier[i];
      float4 position_difference = simBox->calculateDistance( position1, position2, array );
      
      atomicFloatAdd(&force[index1].x, myLagrangeMultiplier * position_difference.x);
      atomicFloatAdd(&force[index1].y, myLagrangeMultiplier * position_difference.y);
      atomicFloatAdd(&force[index1].z, myLagrangeMultiplier * position_difference.z);


      atomicFloatAdd(&force[index2].x, -myLagrangeMultiplier * position_difference.x);
      atomicFloatAdd(&force[index2].y, -myLagrangeMultiplier * position_difference.y);
      atomicFloatAdd(&force[index2].z, -myLagrangeMultiplier * position_difference.z);

      
      if(updateVirial){
	float my_half_virial = 0.5f * myLagrangeMultiplier * position_difference.w;
	atomicFloatAdd(&virial[index1].y, my_half_virial);
	atomicFloatAdd(&virial[index2].y, my_half_virial);
      }
    }
  }
}

//////////////////////////////////////////////////
// Output.
//////////////////////////////////////////////////

template <class S> __global__ void calculateStandardDeviation( S* simBox, float* simBoxPointer, float4* position, float3* constraintInfo, float totalConstraints, float* sumArray ){
  
  if( MyGP < totalConstraints ){
    float array[simulationBoxSize];
    simBox->loadMemory( array, simBoxPointer );
    
    float3 myConstraint = constraintInfo[MyGP];
    float4 particleOne = position[(unsigned) myConstraint.x];
    float4 particleTwo = position[(unsigned) myConstraint.y];
    
    float4 distanceSquared = simBox->calculateDistance( particleOne, particleTwo, array );
    
    double distance = sqrt( distanceSquared.w );
    sumArray[MyGP] = (distance - myConstraint.z) * (distance - myConstraint.z);
  }
}

//////////////////////////////////////////////////
// Sorting.
//////////////////////////////////////////////////

__global__ void UpdateConstraintsAfterSorting( unsigned int* new_index, unsigned int nParticles, float3* d_constraintInfo, float totalNumberOfConstraints ){

  extern __shared__ unsigned s_r[];
  
  float3 old_bond, new_bond;
  
  if( MyGP < totalNumberOfConstraints ){
    old_bond = d_constraintInfo[MyGP];
    new_bond = old_bond;
  }

  unsigned particleI = (unsigned) old_bond.x;
  unsigned particleJ = (unsigned) old_bond.y;
  
  for ( unsigned i = 0; i < nParticles; i+=PPerBlock ){
    
    if( (i + MyP) < nParticles )
      s_r[MyP] = new_index[i+MyP];

    __syncthreads();
    
    for( unsigned j = 0; j < PPerBlock && (i + j) < nParticles; j++){
      if ( s_r[j] == particleI ){ 
	new_bond.x = i + j;
      }
      else if (s_r[j] == particleJ ){ 
	new_bond.y = i + j;
      }
    }
    __syncthreads();
  }
  
  if( MyGP < totalNumberOfConstraints )
    d_constraintInfo[MyGP] = new_bond;
}
