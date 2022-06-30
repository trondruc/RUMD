#include "rumd/TetheredGroup.h"

#include "rumd/SimulationBox.h"
#include "rumd/ParticleData.h"


// Forward declaration of kernels

template<class S>
__global__ void forceSolid(float4 *d_lattice, unsigned *d_index,
			   float4 *r, float4 *f, float4 *w,
			   S *simbox, float *simboxpointer, 
			   unsigned numSolidAtoms, float ks);

template<class S>
__global__ void moveSolid(float4 *d_lattice, float dx,  
			  unsigned dir,
			  S *simbox, 
			  float *simboxpointer, unsigned numSolidAtoms);


template<class S>
__global__ void energySolid(float4 *d_lattice, unsigned *d_index,
			    float4 *r, float *energy, 
			    S *simbox, float *simboxpointer, 
			    unsigned numSolidAtoms, float ks);

__global__ void set_new_indices(unsigned *solid_index, unsigned *new_particle_index, unsigned numSolidAtoms);

///////////////////////////////////////////////////////
// Constructor / Destructor
///////////////////////////////////////////////////////

TetheredGroup::TetheredGroup(std::vector<unsigned> solidAtomTypes, float springConstant) : solidAtomTypes(solidAtomTypes), numSolidAtoms(0), kspring(springConstant), direction(0), initFlag(false), h_lattice(0), d_lattice(0), h_index(0), d_index(0), h_local_energy(0), d_local_energy(0) {
  SetID_String("solid");  
}


TetheredGroup::~TetheredGroup(){

  if ( initFlag ){
    cudaFreeHost(h_lattice);
    cudaFree(d_lattice);

    cudaFreeHost(h_index);
    cudaFree(d_index);

    cudaFreeHost(h_local_energy);
    cudaFree(d_local_energy);
  }

}


unsigned TetheredGroup::CountSolidAtoms(){
  
  unsigned numParticles = particleData->GetNumberOfParticles();
  unsigned numSolid = 0;

  std::vector<unsigned>::iterator it;
  for ( unsigned n=0; n<numParticles; n++ ){
    for(it=solidAtomTypes.begin(); it != solidAtomTypes.end(); it++)
      if ( particleData->h_Type[n] == (*it) ) {
	numSolid ++;
	break; // in case same type appears twice (better to use set?)
      }
  }
  
  return numSolid;
}

void TetheredGroup::Initialize() {

  numSolidAtoms = CountSolidAtoms();

  std::cout << "TetheredGroup, name " << GetID_String() << "; found " << numSolidAtoms << " wall atoms\n";

  particleData->CopyPosFromDevice();
  
  size_t nbytes = sizeof(float4)*numSolidAtoms;
  if ( cudaMallocHost((void **)&h_lattice, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("Solid",__func__,"Memory allocation failure");
  if ( cudaMalloc((void **)&d_lattice, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("Solid",__func__,"Memory allocation failure");

  nbytes = sizeof(float)*numSolidAtoms;
  if ( cudaMallocHost((void **)&h_local_energy, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("Solid",__func__,"Memory allocation failure");
  if ( cudaMalloc((void **)&d_local_energy, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("Solid",__func__,"Memory allocation failure");

  memset(h_local_energy, 0, nbytes);
  cudaMemset(d_local_energy, 0, nbytes);
  
  nbytes =  sizeof(unsigned)*numSolidAtoms;
  if ( cudaMallocHost((void **)&h_index, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("Solid",__func__,"Memory allocation failure");
  if ( cudaMalloc((void **)&d_index, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("Solid",__func__,"Memory allocation failure");


  unsigned numParticles = particleData->GetNumberOfParticles();
  std::vector<unsigned>::iterator it;
  unsigned index = 0;
  for ( unsigned n=0; n<numParticles; n++ ){
    for(it=solidAtomTypes.begin(); it != solidAtomTypes.end(); it++)
      if ( particleData->h_Type[n] == (*it) ) {
	h_lattice[index].x = particleData->h_r[n].x;
	h_lattice[index].y = particleData->h_r[n].y;
	h_lattice[index].z = particleData->h_r[n].z;
	
	h_index[index] = n;
	
	index++;
	break;
      }
  }
  
  nbytes = sizeof(float4)*numSolidAtoms;
  cudaMemcpy(d_lattice, h_lattice, nbytes, cudaMemcpyHostToDevice);

  nbytes =  sizeof(unsigned)*numSolidAtoms;
  cudaMemcpy(d_index, h_index, nbytes, cudaMemcpyHostToDevice);


  threads_per_block = 32;
  num_blocks = (numSolidAtoms + threads_per_block - 1)/threads_per_block + 1;

  initFlag = true;
 
}

void TetheredGroup::SetSpringConstant(float ks){
  
  kspring = ks;

}


void TetheredGroup::SetDirection(unsigned set_dir){
  if(set_dir > 2)
    throw RUMD_Error("TetheredGroup",__func__,"Invalid direction, must be 0, 1 or 2");

  direction = set_dir;
}

void TetheredGroup::CopyLatticeToDevice(){

  size_t nbytes = sizeof(float4)*numSolidAtoms;
  cudaMemcpy(d_lattice, h_lattice, nbytes, cudaMemcpyHostToDevice);


}

void TetheredGroup::CopyLatticeFromDevice(){

  size_t nbytes = sizeof(float4)*numSolidAtoms;
  cudaMemcpy(h_lattice, d_lattice, nbytes, cudaMemcpyDeviceToHost);
}



void TetheredGroup::Move(float displacement){
  if(!particleData || !simBox)
    throw RUMD_Error("Solid",__func__,"No particleData/simBox available");


  // Alternatively, could supply direction as an argument to Move
  // or even supply a vector displacement to Move, and avoid having direction
  // as a separate variable

  LeesEdwardsSimulationBox* testLESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);
  RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(simBox);

  if(testLESB) {
    moveSolid<<<num_blocks, threads_per_block>>>
      (d_lattice, displacement, direction, 
       testLESB, testLESB->GetDevicePointer(), numSolidAtoms);
  }
  else if(testRSB) {
    moveSolid<<<num_blocks, threads_per_block>>>
      (d_lattice, displacement, direction,
       testRSB, testRSB->GetDevicePointer(), numSolidAtoms);
  }
  else
    throw RUMD_Error("Solid",__func__,"Could not cast simBox to recognized type");
}

void TetheredGroup::CalcF(bool initialize, bool __attribute__((unused))calc_stresses) {
  if(!particleData || !simBox)
    throw RUMD_Error("Solid",__func__,"No particleData/simBox available");
  
  // if initialize is true, we must set the force on *all* particles to zero,
  // not just the wall particles. So we call the ParticleData function for this.
  if(initialize)
    particleData->SetForcesToZero();


	
  LeesEdwardsSimulationBox* testLESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);
  RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  
  if(testLESB)
    forceSolid<<<num_blocks, threads_per_block>>>
      (d_lattice, d_index, particleData->d_r,  
       particleData->d_f, particleData->d_w,
       testLESB, testLESB->GetDevicePointer(), numSolidAtoms, kspring);
  else if(testRSB)
    forceSolid<<<num_blocks, threads_per_block>>>
      (d_lattice, d_index, particleData->d_r,  
       particleData->d_f, particleData->d_w,
       testRSB, testRSB->GetDevicePointer(), numSolidAtoms, kspring);
  else
    throw RUMD_Error("Solid",__func__,"Could not cast simBox to recognized type");
  
}


double TetheredGroup::GetPotentialEnergy() {
  if(!particleData || !simBox)
    throw RUMD_Error("Solid",__func__,"No particleData/simBox available");
  
  LeesEdwardsSimulationBox* testLESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);
  RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(simBox);

  if(testLESB)
    energySolid<<<num_blocks, threads_per_block>>>(d_lattice, d_index,
						   particleData->d_r,
						   d_local_energy, 
						   testLESB,
						   testLESB->GetDevicePointer(),
						   numSolidAtoms, kspring);
  else if(testRSB)
    energySolid<<<num_blocks, threads_per_block>>>(d_lattice, d_index,
						   particleData->d_r,
						   d_local_energy, 
						   testRSB,
						   testRSB->GetDevicePointer(), 
						   numSolidAtoms, kspring);
  else
    throw RUMD_Error("Solid",__func__,"Could not cast simBox to recognized type");

  double pe_thisPotential = 0.;

  // copy to host
  cudaMemcpy( h_local_energy, d_local_energy, numSolidAtoms * sizeof(float), cudaMemcpyDeviceToHost );
  
  // sum over particles
  for(unsigned int i=0; i < numSolidAtoms; i++)
    pe_thisPotential += h_local_energy[i];
  
  return pe_thisPotential;

}



void TetheredGroup::UpdateAfterSorting(unsigned* __attribute__((unused))old_index, unsigned* new_index) {
  
  set_new_indices<<<num_blocks, threads_per_block  >>>( d_index, new_index, numSolidAtoms);
  
}




/* Kernels */


__global__ void set_new_indices(unsigned *solid_index, unsigned *new_particle_index, unsigned numSolidAtoms) {
  int global_thread_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(global_thread_idx < numSolidAtoms) {
    unsigned old_solid_index = solid_index[global_thread_idx];
    solid_index[global_thread_idx] = new_particle_index[old_solid_index];
  }
}


template<class S>
__global__ void forceSolid(float4 *d_lattice, unsigned *d_index,
			   float4 *r, float4 *f, float4 *w, 
			   S *simbox, float *simboxpointer, 
			   unsigned numSolidAtoms, float ks){

  unsigned global_thread_idx = blockIdx.x*blockDim.x + threadIdx.x;

  if( global_thread_idx <  numSolidAtoms ) {
    
    // Global atom index;
    unsigned i = d_index[global_thread_idx];
    
    float simBoxPtr_local[simulationBoxSize];
    simbox->loadMemory(simBoxPtr_local, simboxpointer);

    float4 dr = simbox->calculateDistance(d_lattice[global_thread_idx], r[i], simBoxPtr_local);

    f[i].x += ks*dr.x;
    f[i].y += ks*dr.y;
    f[i].z += ks*dr.z;
    f[i].w += 0.5*ks*dr.w;
    w[i].w -= ks*dr.w; 

  }

} 


template<class S>
__global__ void energySolid(float4 *d_lattice, unsigned *d_index,
			   float4 *r, float *energy, 
			   S *simbox, float *simboxpointer, 
			   unsigned numSolidAtoms, float ks){

  unsigned global_thread_idx = blockIdx.x*blockDim.x + threadIdx.x;

  if( global_thread_idx <  numSolidAtoms ) {
    
    // Global atom index;
    unsigned i = d_index[global_thread_idx];
    
    float simBoxPtr_local[simulationBoxSize];
    simbox->loadMemory(simBoxPtr_local, simboxpointer);

    float4 dr = simbox->calculateDistance(d_lattice[global_thread_idx], r[i], simBoxPtr_local);
    
    energy[global_thread_idx] =  0.5*ks*dr.w;
  }

} 


template<class S>
__global__ void moveSolid(float4 *d_lattice, float dx, unsigned dir,
			 S *simbox, float *simboxpointer, 
			  unsigned numSolidAtoms){


  unsigned global_thread_idx  = blockIdx.x*blockDim.x + threadIdx.x;

  if( global_thread_idx  < numSolidAtoms ) {
    
    float4 my_lattice = d_lattice[global_thread_idx ];
    
    if ( dir==0 )
      my_lattice.x += dx;
    else if ( dir==1 )
      my_lattice.y += dx;
    else if (dir==2 )
      my_lattice.z += dx;

    float simBoxPtr_local[simulationBoxSize];
    simbox->loadMemory( simBoxPtr_local, simboxpointer );
    
    simbox->applyBoundaryCondition(my_lattice, simBoxPtr_local);
    
    d_lattice[global_thread_idx ] = my_lattice;
    
  }

} 
