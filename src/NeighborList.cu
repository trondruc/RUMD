/*
    Copyright (C) 2010  Thomas Schrøder

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/

#include <iostream>

#include "rumd/rumd_utils.h"
#include "rumd/Device.h"
#include "rumd/PairPotential.h"
#include "rumd/SimulationBox.h"
#include "rumd/NeighborList.h"
#include "rumd/Sample.h"

#include <cmath>
#include <cstdio>

#include <thrust/sort.h>
#include <thrust/gather.h>

#define n_Cells_def 2

const unsigned int NeighborList::min_size_memory_opt = 10000;
const unsigned int NeighborList::default_maxNumNbrs = 50;
const unsigned int NeighborList::block_size_cells = 96;
const int NeighborList::n_Cells = n_Cells_def;
const unsigned int NeighborList::default_NB_method_size_threshold = 8000;
const unsigned int NeighborList::block_size_excl = 32;

const size_t NeighborList::sharedMemPerBlock = Device::GetDevice().GetSharedMemPerBlock();


///////////////////////////////////////////
// CONSTRUCTION/DESTRUCTION 
///////////////////////////////////////////

NeighborList::NeighborList() : particleData(0),
			       kp(),
			       simBox(0),
			       testLESB(0),
			       testRSB(0),
			       numBlocks(0),
			       numParticles(0),
			       particlesPerBlock(0),
			       numVirtualParticles(0),
			       skin(0.5f),
			       maxCutoff(0.f),
			       maxNumNbrs(0),
			       maxNumExcluded(100),
			       numTypes(0),
			       nb_method(DEFAULT),
			       allocate_max(false),
			       allow_update_after_sorting(false),
			       have_exclusion_type(false)
{
  // for accepting strings representations of NB method
  method_str_map["default"] = DEFAULT;
  method_str_map["none"] = NONE;
  method_str_map["n2"] = N2;
  method_str_map["sort"] = SORT;
  

  // for keeping track of allocation
  numberAllocatedUnits["neighborList"] = 0;
  numberAllocatedUnits["savedNeighborList"] = 0;  
  numberAllocatedUnits["numNbrs"] = 0;
  numberAllocatedUnits["exclusionType"] = 0;
  numberAllocatedUnits["exclusionList"] = 0;  
  numberAllocatedUnits["numExcluded"] = 0;

  numberAllocatedUnits["cutoffArray"] = 0;
  numberAllocatedUnits["last_r"] = 0;
  
  numberAllocatedUnits["cellIndex"] = 0;
  numberAllocatedUnits["cellStartEnd"] = 0;
  numberAllocatedUnits["tempArrays"] = 0;

  d_neighborList = 0;
  d_savedNeighborList = 0;
  d_numNbrs = 0;

  h_exclusionType = 0;
  d_exclusionType = 0;

  h_exclusionList = 0;
  d_exclusionList = 0;
  h_numExcluded = 0;
  d_numExcluded = 0;

  h_cutoffArray = 0;
  d_cutoffArray = 0;
  d_last_r = 0;
  
  d_cellIndex = 0;
  d_cellStart = 0;
  d_cellEnd = 0;


  d_temp_float4 = 0;
  d_temp_int = 0;

  // Here we allocate memory that doesn't depend on the system size
  if ( cudaMalloc( (void**) &d_lastBoxShift, sizeof(float) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("NeighborList",__func__,"Malloc failed on d_lastBoxShift") );
  
  float h_lastBoxShift = 0.f;
  cudaMemcpy(d_lastBoxShift, &h_lastBoxShift, sizeof(float), cudaMemcpyHostToDevice);
  
  if ( cudaMalloc( (void**) &d_dStrain, sizeof(float2) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("NeighborList",__func__,"Malloc failed on d_dStrain") );
  
  float2 h_dStrain = {0.f,0.f};
  cudaMemcpy(d_dStrain, &h_dStrain, sizeof(float2), cudaMemcpyHostToDevice);
  
  if( cudaMalloc( (void**) &d_updateRequired, 3*sizeof(unsigned int) ) == cudaErrorMemoryAllocation )
    // the second element indicates whether the maximum number of neighbors was reached
    throw( RUMD_Error("NeighborList",__func__,"Malloc failed on d_updateRequired") );

  ResetNeighborList();
  rebuild_was_required = 0;
}

NeighborList::~NeighborList() {

  if(numberAllocatedUnits["neighborList"] > 0)
    cudaFree(d_neighborList); 

  if(numberAllocatedUnits["savedNeighborList"] > 0)
    cudaFree(d_savedNeighborList);

  if(numberAllocatedUnits["numNbrs"] > 0)
    cudaFree(d_numNbrs);
  
  if( numberAllocatedUnits["exclusionType"] > 0) {
    cudaFreeHost(h_exclusionType);
    cudaFree(d_exclusionType);
  }
  
  if(numberAllocatedUnits["exclusionList"] > 0) {
    cudaFreeHost(h_exclusionList);
    cudaFree(d_exclusionList);
  }

  if(numberAllocatedUnits["numExcluded"] > 0) {
    cudaFreeHost(h_numExcluded);
    cudaFree(d_numExcluded);
  }

  if(numberAllocatedUnits["cutoffArray"] > 0) {
    cudaFreeHost(h_cutoffArray);
    cudaFree(d_cutoffArray);
  }
  
  if(numberAllocatedUnits["last_r"] > 0)
    cudaFree(d_last_r); 

  if(numberAllocatedUnits["cellIndex"] > 0) {
    cudaFree(d_cellIndex);
  }
  
  if(numberAllocatedUnits["tempArrays"] > 0) {
    cudaFree(d_temp_float4);
    cudaFree(d_temp_int);
  }
  
  if(numberAllocatedUnits["cellStartEnd"] > 0) {
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);
  }
  cudaFree(d_lastBoxShift);
  cudaFree(d_dStrain);
  cudaFree(d_updateRequired);
}

///////////////////////////////////////////
// MEMORY ALLOCATION/DEALLOCATION 
///////////////////////////////////////////


void NeighborList::Initialize(Sample* sample) {
  this->sample  = sample;
  particleData = sample->GetParticleData();
  numParticles = particleData->GetNumberOfParticles();
  kp = sample->GetKernelPlan();

  if(kp.threads.y == 1 && nb_method == NONE)
    // the code for ignoring NB list is only in the tp>1 calcf kernel
    throw RUMD_Error("NeighborList", __func__, "Cannot choose tp=1 and no neighborlist");

  this->simBox = sample->GetSimulationBox();
  testRSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  testLESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);


  numBlocks = kp.num_blocks;
  numVirtualParticles = kp.num_virt_part;
  particlesPerBlock = kp.threads.x;

  AllocateLastPositions();
  AllocateNumNbrs();
  AllocateExclusionType( particleData->GetNumberOfTypes() );
}

std::string NeighborList::GetNB_Method() {
  std::map<std::string, NB_method>::iterator it;
  for(it = method_str_map.begin(); it != method_str_map.end(); it++) {
    if(it->second == nb_method)
      return it->first;
  }
  throw RUMD_Error("NeighborList", __func__, "Failure in NB_method string map");
}


void NeighborList::SetNB_Method(const std::string& method) {
  std::map<std::string, NB_method>::iterator it = method_str_map.find(method);
  if( it == method_str_map.end() )
    throw RUMD_Error("NeighborList", __func__, std::string("Unrecognized NB_method: ") + method);
  else
    SetNB_Method(it->second);

}

void NeighborList::SetNB_Method(NB_method method) {


  if(method == NONE && numParticles && kp.threads.y == 1)
    throw RUMD_Error("NeighborList", __func__, "Cannot use NB method NONE (\"none\") together with tp=1");
  if(method == NONE && (d_exclusionList || have_exclusion_type))
    throw RUMD_Error("NeighborList", __func__, "Cannot use NB method NONE (\"none\") when exclusions are present");
  
  this->nb_method = method;
}

void NeighborList::SetEqualCutoffs(unsigned int num_types, float cutoff) {
  numTypes = num_types;
  AllocateCutoffArray();
  for (unsigned int i=0; i<numTypes; i++){
    for (unsigned int j=0; j<numTypes; j++){ 
      unsigned int k = i * numTypes + j;
      h_cutoffArray[k] = cutoff;
    }
  }
  maxCutoff = cutoff;
  cudaMemcpy( d_cutoffArray, h_cutoffArray, sizeof(float)*numTypes*numTypes, cudaMemcpyHostToDevice);
}


void NeighborList::SetCutoffs(PairPotential* potential){
  if(!particleData) return;
  numTypes = particleData->GetNumberOfTypes();
  
  AllocateCutoffArray();

  maxCutoff = 0.f;
  for (unsigned int i=0; i<numTypes; i++){
    for (unsigned int j=0; j<numTypes; j++){ 
      unsigned int k = i * numTypes + j;
      float Rcut = potential->GetPotentialParameter(i,j,0);
      if(Rcut > maxCutoff) maxCutoff = Rcut;
      h_cutoffArray[k] = Rcut;
    }
  }  
  
  cudaMemcpy( d_cutoffArray, h_cutoffArray, sizeof(float)*numTypes*numTypes, cudaMemcpyHostToDevice);


}



void NeighborList::AllocateNeighborList() {
  if(numVirtualParticles * maxNumNbrs == 0)
    throw RUMD_Error("NeighborList", __func__, "Allocating a zero-sized neighbor-list; something is wrong");

  if(numberAllocatedUnits["neighborList"] != numVirtualParticles * maxNumNbrs) {
    
    if(numberAllocatedUnits["neighborList"] > 0)
      cudaFree(d_neighborList);


    if( cudaMalloc( (void**) &d_neighborList, numVirtualParticles*maxNumNbrs * sizeof(int) ) == cudaErrorMemoryAllocation )
      {
	cudaError err = cudaGetLastError();
	throw RUMD_Error("NeighborList", __func__, std::string("Memory allocation failed on d_neighborList. cuda error is: ") + cudaGetErrorString(err) );

      }
    cudaMemset(d_neighborList, 0, numVirtualParticles*maxNumNbrs * sizeof(int) );
    numberAllocatedUnits["neighborList"]  = numVirtualParticles * maxNumNbrs;
    ResetNeighborList();
  }
}


void NeighborList::AllocateSavedNeighborList() {

  unsigned size_saved_nbr_list = numVirtualParticles * (maxNumExcluded > maxNumNbrs ? maxNumExcluded : maxNumNbrs);

  if( numberAllocatedUnits["savedNeighborList"]  != size_saved_nbr_list) {
    if (numberAllocatedUnits["savedNeighborList"] > 0)
      cudaFree(d_savedNeighborList);
    
    if( cudaMalloc( (void**) &d_savedNeighborList, size_saved_nbr_list * sizeof(int)) == cudaErrorMemoryAllocation )
      throw( RUMD_Error("NeighborList", __func__, "cudaMalloc failed on d_savedNeighborList") );
    
    numberAllocatedUnits["savedNeighborList"]  = size_saved_nbr_list;
  }
}

void NeighborList::AllocateExclusionType( unsigned int numberOfTypes ) {
  if( numberAllocatedUnits["exclusionType"] != numberOfTypes*numberOfTypes) {

    if( numberAllocatedUnits["exclusionType"] > 0) {
      cudaFreeHost(h_exclusionType);
      cudaFree(d_exclusionType);
    }
    if( cudaMallocHost( (void**) &h_exclusionType, sizeof(unsigned) * numberOfTypes * numberOfTypes ) == cudaErrorMemoryAllocation )
      throw( RUMD_Error("NeighborList",__func__,"cudaMalloc failed on h_exclusionType") );
    if( cudaMalloc( (void**) &d_exclusionType, sizeof(unsigned) * numberOfTypes * numberOfTypes ) == cudaErrorMemoryAllocation )
      throw( RUMD_Error("NeighborList",__func__,"cudaMalloc failed on d_exclusionType") );

    numberAllocatedUnits["exclusionType"] = numberOfTypes*numberOfTypes;

    memset( h_exclusionType, 0, numberOfTypes*numberOfTypes * sizeof(unsigned) );
    CopyExclusionTypeToDevice();
  }
}

void NeighborList::AllocateExclusionList() {
  if(numberAllocatedUnits["exclusionList"] != numVirtualParticles * maxNumExcluded) {

    if(nb_method == NONE) {
      std::cout << "[Info] Cannot use NB method NONE (\"none\") when exclusions are present. Setting to N2 (\"n2\")" << std::endl;
      nb_method = N2;
    }
    
    
    if(numberAllocatedUnits["exclusionList"] > 0) {
      cudaFreeHost(h_exclusionList);
      cudaFree(d_exclusionList);
    }

    if( cudaMallocHost( (void**) &h_exclusionList, numVirtualParticles*maxNumExcluded * sizeof(int) ) == cudaErrorMemoryAllocation )
      {
	cudaError err = cudaGetLastError();
	throw RUMD_Error("NeighborList", __func__, std::string("Memory allocation failed on h_exclusionList. cuda error is: ") + cudaGetErrorString(err) );

      }
    if( cudaMalloc( (void**) &d_exclusionList, numVirtualParticles*maxNumExcluded * sizeof(int) ) == cudaErrorMemoryAllocation )
      {
	cudaError err = cudaGetLastError();
	throw RUMD_Error("NeighborList", __func__, std::string("Memory allocation failed on d_exclusionList. cuda error is: ") + cudaGetErrorString(err) );

      }


    memset(h_exclusionList, 0, numVirtualParticles*maxNumExcluded * sizeof(int) );
    cudaMemset(d_exclusionList, 0, numVirtualParticles*maxNumExcluded * sizeof(int) );
    numberAllocatedUnits["exclusionList"]  = numVirtualParticles * maxNumExcluded;
  }
}



void NeighborList::AllocateNumNbrs() {
  if(numberAllocatedUnits["numNbrs"] != numVirtualParticles) {

    if(numberAllocatedUnits["numNbrs"] > 0)
      cudaFree(d_numNbrs);

    if( cudaMalloc( (void**) &d_numNbrs, numVirtualParticles * sizeof(int) ) == cudaErrorMemoryAllocation )
      throw RUMD_Error("NeighborList",__func__,"Memory allocation failed on d_numNbrs");
    cudaMemset(d_numNbrs, 0, numVirtualParticles * sizeof(int));
    numberAllocatedUnits["numNbrs"]  = numVirtualParticles;
    ResetNeighborList();
  }
}

void NeighborList::AllocateNumExcluded() {
  if(numberAllocatedUnits["numExcluded"] != numVirtualParticles) {

    if(numberAllocatedUnits["numExcluded"] > 0) {
      cudaFreeHost(h_numExcluded);
      cudaFree(d_numExcluded);
    }

    if( cudaMallocHost( (void**) &h_numExcluded, numVirtualParticles * sizeof(int) ) == cudaErrorMemoryAllocation )
      throw RUMD_Error("NeighborList",__func__,"Memory allocation failed on h_numExcluded");
    
    if( cudaMalloc( (void**) &d_numExcluded, numVirtualParticles * sizeof(int) ) == cudaErrorMemoryAllocation )
      throw RUMD_Error("NeighborList",__func__,"Memory allocation failed on d_numExcluded");
    

    memset(h_numExcluded, 0, numVirtualParticles * sizeof(int) );
    cudaMemset(d_numExcluded, 0, numVirtualParticles * sizeof(int) );

    numberAllocatedUnits["numExcluded"]  = numVirtualParticles;
  }
}



void NeighborList::AllocateCutoffArray() {
  
  unsigned int num_types2 = numTypes * numTypes;
  if(numberAllocatedUnits["cutoffArray"] != num_types2) {

    if( numberAllocatedUnits["cutoffArray"] > 0 ) {
      cudaFree(d_cutoffArray);
      cudaFreeHost(h_cutoffArray);
    }

    if( cudaMalloc( (void**) &d_cutoffArray, num_types2 * sizeof(float) ) == cudaErrorMemoryAllocation )
      throw( RUMD_Error("NeighborList",__func__,"cudaMalloc failed on d_cutoffArray") );
    if( cudaMallocHost( (void**) &h_cutoffArray, num_types2*sizeof(float) ) == cudaErrorMemoryAllocation )
      throw( RUMD_Error("NeighborList",__func__,"cudaMalloc failed on h_cutoffArray") );

    // initialize to zeros 
    memset( h_cutoffArray, 0, num_types2 * sizeof(float) );
    cudaMemset( d_cutoffArray, 0, num_types2 * sizeof(float) );
  
    numberAllocatedUnits["cutoffArray"] = num_types2;
  }
  
}


void NeighborList::AllocateLastPositions() {

  if( numberAllocatedUnits["last_r"] != numVirtualParticles ) {

    if( numberAllocatedUnits["last_r"] > 0 )
      cudaFree(d_last_r);

    if( cudaMalloc( (void**) &d_last_r, numVirtualParticles * sizeof(float4) ) == cudaErrorMemoryAllocation )
      throw( RUMD_Error("NeighborList",__func__,"cudaMalloc failed on d_last_r") );
    // initialize to zeros
    cudaMemset( d_last_r, 0, numVirtualParticles * sizeof(float4) );
  
    numberAllocatedUnits["last_r"] = numVirtualParticles;
  }
}

void NeighborList::AllocateCellArrays(unsigned int num_cells) {

  unsigned int num_blocks_cells = (numParticles + block_size_cells - 1) / block_size_cells;
  unsigned int numVirtualCellIndices =  num_blocks_cells * block_size_cells;
  
  if( numberAllocatedUnits["cellIndex"] != numVirtualCellIndices ) {

    if( numberAllocatedUnits["cellIndex"] > 0 ) {
      cudaFree(d_cellIndex);
    }
    if( cudaMalloc( (void**) &d_cellIndex, numVirtualCellIndices * sizeof(int) ) == cudaErrorMemoryAllocation ) throw RUMD_Error("NeighborList", __func__ , "cudaMalloc failed on d_cellIndex\n");
    

    numberAllocatedUnits["cellIndex"] = numVirtualCellIndices;
  }
  
  if( numberAllocatedUnits["cellStartEnd"] != num_cells ) {
    if( numberAllocatedUnits["cellStartEnd"] > 0 ) {
      cudaFree(d_cellStart);
      cudaFree(d_cellEnd);
    }
    
    if( cudaMalloc( (void**) &d_cellStart, num_cells * sizeof(int) ) == cudaErrorMemoryAllocation ) throw RUMD_Error("NeighborList", __func__ , "cudaMalloc failed on d_cellStart\n");
    if( cudaMalloc( (void**) &d_cellEnd, num_cells * sizeof(int) ) == cudaErrorMemoryAllocation ) throw RUMD_Error("NeighborList", __func__ , "cudaMalloc failed on d_cellEnd\n");
    
    numberAllocatedUnits["cellStartEnd"] = num_cells;
  }

}



void NeighborList::AllocateTempArrays() {


  if(numberAllocatedUnits["tempArrays"] != numVirtualParticles) {
    if(numberAllocatedUnits["tempArrays"] > 0) {
      cudaFree(d_temp_float4);
      cudaFree(d_temp_int);
    }
    
    if( cudaMalloc( (void**) &d_temp_float4, numVirtualParticles * sizeof(float4) ) == cudaErrorMemoryAllocation )
      throw RUMD_Error("NeighborList", __func__ , "cudaMalloc failed on d_temp_float4\n");
    
    if( cudaMalloc( (void**) &d_temp_int, numVirtualParticles * sizeof(int) ) == cudaErrorMemoryAllocation )
      throw RUMD_Error("NeighborList",__func__,"cudaMalloc failed on d_temp_int");

    numberAllocatedUnits["tempArrays"] = numVirtualParticles;
  }
}


///////////////////////////////////////////
// Main methods.
///////////////////////////////////////////


void NeighborList::ResetNeighborList() {
  unsigned num_blocks_excl = (numVirtualParticles+block_size_excl-1)/block_size_excl;
  unsigned int h_updateRequired[3] = {kp.grid.x, 0, num_blocks_excl};
  cudaMemcpy(d_updateRequired, h_updateRequired, 3*sizeof(unsigned int), cudaMemcpyHostToDevice);
  rebuild_required = 1; // so host can check also
}

void NeighborList::UpdateAfterSorting( unsigned* old_index, unsigned* new_index) {
  AllocateTempArrays();
  unsigned threads_per_block = 64;
  unsigned num_blocks = (numVirtualParticles+threads_per_block-1)/threads_per_block;

  thrust::device_ptr<unsigned> thrust_old_index(old_index);
  thrust::device_ptr<int> thrust_d_temp_int(d_temp_int);

  if(numberAllocatedUnits["exclusionList"] > 0) {
    // update exclusion list if present 
    // NOTE: this needs to be done even if NB-list doesn't ie also 
    // if allow_update_after_sorting is false. So maybe should move it up 
    
    SaveNeighborList(true);
    update_nblist_after_sorting<<<num_blocks, threads_per_block >>>(numParticles, numVirtualParticles, d_exclusionList, d_savedNeighborList, maxNumExcluded, d_numExcluded, new_index);
    
    // then rearrange d_numExcluded
    thrust::device_ptr<unsigned> thrust_d_numExcluded(d_numExcluded);
    
    thrust::copy(thrust_d_numExcluded, thrust_d_numExcluded + numParticles, thrust_d_temp_int);
    thrust::gather(thrust_old_index, thrust_old_index + numParticles, thrust_d_temp_int, thrust_d_numExcluded);
  }
  
  if(!allow_update_after_sorting)
    return;

  // first re-arrange d_last_r in order to check whether a rebuild is necessary
  thrust::device_ptr<float4> thrust_d_last_r(d_last_r);
  thrust::device_ptr<float4> thrust_d_temp_float4(d_temp_float4);
  thrust::device_ptr<unsigned> thrust_d_numNbrs(d_numNbrs);


  thrust::copy(thrust_d_last_r, thrust_d_last_r + numParticles, thrust_d_temp_float4);
  thrust::gather(thrust_old_index, thrust_old_index + numParticles, thrust_d_temp_float4, thrust_d_last_r);


  CheckRebuildRequired();
  cudaMemcpy(&rebuild_required, d_updateRequired, sizeof(int), cudaMemcpyDeviceToHost);
  if(rebuild_required)
    return;
  
  SaveNeighborList();

  update_nblist_after_sorting<<<num_blocks, threads_per_block >>>(numParticles, numVirtualParticles, d_neighborList, d_savedNeighborList, maxNumNbrs, d_numNbrs, new_index);

  // then rearrange d_numNbrs
  
  thrust::copy(thrust_d_numNbrs, thrust_d_numNbrs + numParticles, thrust_d_temp_int);
  thrust::gather(thrust_old_index, thrust_old_index + numParticles, thrust_d_temp_int, thrust_d_numNbrs);

}


unsigned NeighborList::GetActualMaxNumNbrs() const {
  thrust::device_ptr<unsigned> num_nbrs_dev_ptr(d_numNbrs);
  return thrust::reduce(num_nbrs_dev_ptr, num_nbrs_dev_ptr + numParticles, (unsigned) 0, thrust::maximum<unsigned>());
}

void NeighborList::CheckRebuildRequired() {
  float4* d_r = particleData->d_r;
  float4* d_im = particleData->d_im;
  unsigned num_blocks_excl = (numVirtualParticles+block_size_excl-1)/block_size_excl;
  
  if(testLESB) {
    update_deltaStrain<<<1, 1>>>(d_lastBoxShift, d_dStrain, testLESB->GetDevicePointer(), skin,  maxCutoff);
    displacement_since_nb_build<<< kp.grid, particlesPerBlock >>>( numParticles, d_r, d_last_r, d_im, testLESB, testLESB->GetDevicePointer(), 0.25*skin*skin, d_updateRequired, d_dStrain, num_blocks_excl );
  }
  else if(testRSB)
    displacement_since_nb_build<<< kp.grid, particlesPerBlock >>>( numParticles, d_r, d_last_r, d_im, testRSB, testRSB->GetDevicePointer(), 0.25*skin*skin, d_updateRequired, 0, num_blocks_excl );
  
}

void NeighborList::UpdateNBlist() {
  if(nb_method == NONE)
    return;
  else if(nb_method == DEFAULT)
    nb_method = numParticles > default_NB_method_size_threshold ? SORT : N2;
  

  // First check whther a rebuild is necessary, unless we (on the host side)
  // know already that it is (rebuild_required is only useful if true)
  if(!rebuild_required)
    CheckRebuildRequired();

  if(maxNumNbrs == 0) {
    if(numParticles < min_size_memory_opt || allocate_max)
      maxNumNbrs = numParticles;
    else {
      float rm_sk = maxCutoff + skin;
      unsigned int est_maxNumNbrs = unsigned (1.5*(4./3.)*M_PI*rm_sk*rm_sk*rm_sk*numParticles/simBox->GetVolume());
      if(est_maxNumNbrs < default_maxNumNbrs)
	est_maxNumNbrs = default_maxNumNbrs;
      if(maxNumNbrs < est_maxNumNbrs)
	maxNumNbrs = est_maxNumNbrs;
    }
  }
  else if(maxNumNbrs > numParticles) {
    std::cout << "[Info] NeighborList: maxNumNbrs has been set to a value larger than the number of particles. Resetting to latter value." << std::endl;
    maxNumNbrs = numParticles;
  }
  
  
  if( !rebuild_required && (nb_method == SORT) ) {
    // so  whether a rebuild is required is known on the host if not already
    // (if it was true then we know a rebuild is required, but if false
    // we don't know)
    cudaMemcpy(&rebuild_required, d_updateRequired, sizeof(int), cudaMemcpyDeviceToHost);
  }
  
  bool reallocation_required;
  do {
    AllocateNeighborList(); // just-in-time allocation; only does something if not (correctly) allocated  
    reallocation_required = false;
    
    if(nb_method == SORT) {
      if(rebuild_required)
	CalculateNBL_Sorting();
    }
    else {
      int tp_NBL = kp.threads.y;
      dim3 block_tp_NBL(particlesPerBlock,tp_NBL);
      int numTypes = particleData->GetNumberOfTypes();
      float4* d_r = particleData->d_r;
      
      size_t shared_size = particlesPerBlock*tp_NBL*sizeof(float4)+particlesPerBlock*sizeof(unsigned int) + numTypes*numTypes*sizeof(float) + sizeof(int);
      // add space for exclusionType
      shared_size += numTypes*numTypes * sizeof(unsigned);

      if(shared_size > sharedMemPerBlock) {
	std::ostringstream errorStr;
	errorStr << "Requested more shared memory than is available on the device. shared_size=" << shared_size << "; sharedMemPerBlock=" << sharedMemPerBlock;
	throw RUMD_Error("NeighborList", __func__, errorStr.str());
      }
      
      
      if(testLESB)
	calculateNBL_N2<<<numBlocks, block_tp_NBL, shared_size>>>(numParticles, numVirtualParticles, d_r, testLESB, testLESB->GetDevicePointer(), d_cutoffArray, numTypes, skin, d_numNbrs, d_neighborList, maxNumNbrs, d_updateRequired, d_last_r, d_lastBoxShift, d_exclusionType);
      else if(testRSB)
	calculateNBL_N2<<<numBlocks, block_tp_NBL, shared_size>>>(numParticles, numVirtualParticles, d_r, testRSB, testRSB->GetDevicePointer(), d_cutoffArray, numTypes, skin, d_numNbrs, d_neighborList, maxNumNbrs, d_updateRequired, d_last_r, d_lastBoxShift, d_exclusionType);
      else
	throw RUMD_Error("NeighborList", __func__, 
			 "Unrecognized SimulationBox");
      
    } // end if(nb_method == SORT) ... else ...
    
    
    if(maxNumNbrs < numParticles && !(nb_method == SORT && !rebuild_required)) {
      // do not need to check if we know the rebuild didn't take place,
      // which we know when using sort_cells
      

      int h_updateRequired[2];
      cudaMemcpy(h_updateRequired, d_updateRequired, 2*sizeof(int), cudaMemcpyDeviceToHost);
      if(h_updateRequired[1])
	reallocation_required = true;

	
      if(reallocation_required) {
	maxNumNbrs *= 2;
	if(maxNumNbrs > numParticles)
	  maxNumNbrs = numParticles;
	std::cout << "Too many neighbors; will re-allocate and re-build the neighbor-list; new maxNumNbrs: " << maxNumNbrs << " (in NB with maxCutoff=" << maxCutoff << ")" << std::endl;
      }
    }

  } while (reallocation_required);
    
  // apply the exclusion list
  if(numberAllocatedUnits["exclusionList"] && !(nb_method == SORT && !rebuild_required)) {

    unsigned num_blocks_excl = (numVirtualParticles+block_size_excl-1)/block_size_excl;

    apply_exclusion_list<<<num_blocks_excl, block_size_excl >>>(numParticles, numVirtualParticles, d_neighborList, d_numNbrs, d_exclusionList, d_numExcluded, d_updateRequired);
  }

  allow_update_after_sorting = true;
  rebuild_was_required = rebuild_required;
  rebuild_required = 0; // reset the host-side flag; the kernels did it for the device
  
}


void NeighborList::SaveNeighborList(bool save_exclusionList){
  AllocateSavedNeighborList(); // only allocates if not already done

  unsigned threads_per_block = 128;
  unsigned num_blocks = (numVirtualParticles+threads_per_block-1)/threads_per_block;
  if(!save_exclusionList)
    save_nblist_kernel<<<num_blocks, threads_per_block >>>(numParticles, numVirtualParticles, d_neighborList, d_savedNeighborList, d_numNbrs);
  else
    save_nblist_kernel<<<num_blocks, threads_per_block >>>(numParticles, numVirtualParticles, d_exclusionList, d_savedNeighborList, d_numExcluded);

}


void NeighborList::RestoreNeighborList(){
  if(nb_method == SORT)
    throw RUMD_Error("NeighborList", __func__, "Saving and restoring the neighbor list is not compatible with the sort-based method");

  if(numberAllocatedUnits["d_savedNeighborList"] != numberAllocatedUnits["NbList"])
    throw RUMD_Error("NeighborList",__func__,"Improperly allocated d_savedNeighborList; perhaps SaveNeighborList has not been called");
    cudaMemcpy( d_neighborList, d_savedNeighborList, numVirtualParticles * maxNumNbrs * sizeof(int), cudaMemcpyDeviceToDevice );
}



void NeighborList::CalculateNBL_Sorting() {

  float4 sim_box_lengths = {simBox->GetLength(0), simBox->GetLength(1), simBox->GetLength(2), 0.f};

  int3 num_cells_vec = {(int)floor(n_Cells*sim_box_lengths.x/(maxCutoff+skin)),
			(int)floor(n_Cells*sim_box_lengths.y/(maxCutoff+skin)),
			(int)floor(n_Cells*sim_box_lengths.z/(maxCutoff+skin))};
  if(num_cells_vec.x < 2*n_Cells+1) num_cells_vec.x = 2*n_Cells+1;
  if(num_cells_vec.y < 2*n_Cells+1) num_cells_vec.y = 2*n_Cells+1;
  if(num_cells_vec.z < 2*n_Cells+1) num_cells_vec.z = 2*n_Cells+1;
  
  float4 inv_cell_lengths = {num_cells_vec.x/sim_box_lengths.x,
			    num_cells_vec.y/sim_box_lengths.y,
			    num_cells_vec.z/sim_box_lengths.z, 0.f};
  int num_cells = num_cells_vec.x*num_cells_vec.y*num_cells_vec.z;


  AllocateCellArrays(num_cells);

  float4* d_r = particleData->d_r;
  int numTypes = particleData->GetNumberOfTypes();


  int grid_size = (numParticles + block_size_cells - 1) / block_size_cells;


  if(testLESB)
    calculateCellIndices<<< grid_size, block_size_cells, 0>>>(d_r, testLESB, testLESB->GetDevicePointer(), inv_cell_lengths, num_cells_vec, d_cellIndex, d_cellStart, d_cellEnd);
  else
    calculateCellIndices<<< grid_size, block_size_cells, 0>>>(d_r, testRSB, testRSB->GetDevicePointer(), inv_cell_lengths, num_cells_vec, d_cellIndex, d_cellStart, d_cellEnd);
 
  // wrap raw pointers with thrust device_ptrs 
  

  thrust::device_ptr<int> thrust_d_cellIndex(d_cellIndex);
  // Get sample to do the sorting, which will trigger various 
  // UpdateAfterSorting functions, but we don't want that function to be called
  // on *this* NeighborList because we're about the overwrite the data
  // anyway
  allow_update_after_sorting = false;
  sample->SortParticlesByKey(thrust_d_cellIndex);
  allow_update_after_sorting = true;

  // re-calculate indices
  /*if(testLESB)
    calculateCellIndices<<< grid_size, block_size_cells, 0>>>(d_r, testLESB, testLESB->GetDevicePointer(), inv_cell_lengths, num_cells_vec, d_cellIndex, d_cellStart, d_cellEnd);
  else
    calculateCellIndices<<< grid_size, block_size_cells, 0>>>(d_r, testRSB, testRSB->GetDevicePointer(), inv_cell_lengths, num_cells_vec, d_cellIndex, d_cellStart, d_cellEnd);
  */


  calculateCellsSorted<<< grid_size, block_size_cells+2, (block_size_cells+2)*sizeof(int) >>>(numParticles, d_cellIndex, d_cellStart, d_cellEnd);

  size_t shared_ctr_size = sizeof(unsigned int);
  shared_ctr_size += numTypes * numTypes * sizeof(unsigned);
  shared_ctr_size += numTypes * numTypes * sizeof(float);

  if(testLESB) {
    // the box shift is available to the kernel anyway, but it's not clear how
    // the kernel can know which type of simulation box is present
    calculateNBL_CellsSorted<<< grid_size, block_size_cells, shared_ctr_size>>>(numParticles, numVirtualParticles, d_r, d_cellStart, d_cellEnd, testLESB, testLESB->GetDevicePointer(), d_cutoffArray, numTypes, skin, inv_cell_lengths, num_cells_vec, d_numNbrs, d_neighborList, maxNumNbrs, d_updateRequired, d_last_r, d_lastBoxShift, testLESB->GetBoxShift(), d_exclusionType);
  }
  else
    calculateNBL_CellsSorted<<< grid_size, block_size_cells, shared_ctr_size>>>(numParticles, numVirtualParticles, d_r, d_cellStart, d_cellEnd, testRSB, testRSB->GetDevicePointer(), d_cutoffArray, numTypes, skin, inv_cell_lengths, num_cells_vec, d_numNbrs, d_neighborList, maxNumNbrs, d_updateRequired, d_last_r, d_lastBoxShift, 0.f, d_exclusionType);
}




///////////////////////////////////////////
// Set methods
///////////////////////////////////////////

// Excludes two particles.
void NeighborList::SetExclusion( unsigned particleI, unsigned particleJ ){
  AllocateExclusionList(); // if not already done
  AllocateNumExcluded();


  if(particleI >= numParticles || particleJ >= numParticles)
    throw RUMD_Error("NeighborList", __func__, "Indicated indices too high.");
  
  unsigned num_ex_I = h_numExcluded[particleI];
  unsigned num_ex_J = h_numExcluded[particleJ];

  bool found_J = false;
  for(unsigned idx = 0; idx < num_ex_I; idx++) {
    unsigned excl = h_exclusionList[particleI + numVirtualParticles*idx];
    if (excl == particleJ) {
      found_J = true;
      break;
    }
  } // for(unsigned idx ... )
  
  if(!found_J) {
    if(h_numExcluded[particleI] == maxNumExcluded-1)
      throw RUMD_Error("NeighborList", __func__, "Have reached the maximum number of excluded particles" );
    h_exclusionList[particleI + numVirtualParticles * num_ex_I] = particleJ;
    h_numExcluded[particleI]++;
  }
  
  bool found_I = false;
  for(unsigned jdx = 0; jdx < num_ex_J; jdx++) {
    unsigned excl = h_exclusionList[particleJ + numVirtualParticles*jdx];
    if (excl == particleI) {
      found_I = true;
      break;
    }
  } // for(unsigned jdx ... )
  if(!found_I) {
    if(h_numExcluded[particleJ] == maxNumExcluded-1)
      throw RUMD_Error("NeighborList", __func__, "Have reached the maximum number of excluded particles" );
    h_exclusionList[particleJ + numVirtualParticles * num_ex_J] = particleI;
    h_numExcluded[particleJ]++;
  }
  
}


// Exclude two given particle pair interactions based on type.
void NeighborList::SetExclusionType( unsigned type0, unsigned type1 ){
  have_exclusion_type = true;

  if(numberAllocatedUnits["exclusionType"] == 0)
    throw RUMD_Error("NeighborList",__func__,"Object is not yet initialized, probably because the potential object has not been associated with a sample object yet.");
  unsigned numberOfTypes = particleData->GetNumberOfTypes();
  
  if(nb_method == NONE) {
    std::cout << "[Info] Cannot use NB method NONE (\"none\") when exclusions are present. Setting to N2 (\"n2\")" << std::endl;
    nb_method = N2;
  }


  h_exclusionType[type0*numberOfTypes+type1] = 1;
  // For now symmetric.
  h_exclusionType[type1*numberOfTypes+type0] = 1;
  
  CopyExclusionTypeToDevice();
}


// Exclude two given particle pair interactions based on bond and/or constraint.
void NeighborList::SetExclusionBond(uint1* h_btlist, uint2* h_blist, unsigned num_bonds, unsigned etype){
  if( num_bonds == 0 )
    throw( RUMD_Error("NeighborList",__func__,"No bonds have been read") );

  for( unsigned i = 0; i < num_bonds; i++){
    unsigned userBondType = h_btlist[i].x;
    
    if( userBondType == etype )
      SetExclusion(h_blist[i].x, h_blist[i].y);
  }
  CopyExclusionListToDevice();
  
}

// Exclude two given particle pair interactions based on the angle.
void NeighborList::SetExclusionAngle(uint4 *h_alist, unsigned num_angles){
  if( num_angles == 0 )
    throw( RUMD_Error("NeighborList",__func__,"No angles have been read") );
  

  // The last two particles in an angle.
  for ( unsigned i=0; i < num_angles; i++ )
    SetExclusion(h_alist[i].x, h_alist[i].z);
  
  CopyExclusionListToDevice();
}

// Exclude two given particle pair interactions based on dihedral.
void NeighborList::SetExclusionDihedral(uint4* h_dlist, unsigned num_dihedrals){
  if( num_dihedrals == 0 )
    throw( RUMD_Error("NeighborList",__func__,"No dihedrals have been read") );
  
  // The last two particles in a dihedral.    
  for ( unsigned i=0; i < num_dihedrals; i++ )
    SetExclusion(h_dlist[i].x, h_dlist[i].w);
  
  CopyExclusionListToDevice();
}

// Exclude all pair interactions in the same molecule.
void NeighborList::SetExclusionMolecule(int1* h_mlist, unsigned molindex, unsigned max_num_uau, unsigned num_mol){

  unsigned i = molindex*max_num_uau + num_mol;

  for ( unsigned n=i; n<i+max_num_uau-1; n++ ){
    int ai = h_mlist[n].x;

    if ( ai == - 1) break;

    for ( unsigned m=n+1; m<i+max_num_uau; m++ ){

      int bi = h_mlist[m].x;

      if ( bi == -1) break;

      SetExclusion(ai, bi);
    }
  }

  CopyExclusionListToDevice();

}

///////////////////////////////////////////
// Copy functions
///////////////////////////////////////////

void NeighborList::CopyExclusionListToDevice(){
  unsigned int num_units = numberAllocatedUnits["exclusionList"];

  if(num_units) {
    cudaMemcpy( d_exclusionList, h_exclusionList, sizeof(int) * num_units, cudaMemcpyHostToDevice );
    cudaMemcpy( d_numExcluded, h_numExcluded, sizeof(int) * numberAllocatedUnits["numExcluded"], cudaMemcpyHostToDevice );
  }

}
 

void NeighborList::CopyExclusionTypeToDevice(){
  unsigned int num_units = numberAllocatedUnits["exclusionType"];
  if(num_units)
    cudaMemcpy( d_exclusionType, h_exclusionType, num_units * sizeof(unsigned), cudaMemcpyHostToDevice );
}



///////////////////////////////////////////////
// Kernel Implementations
///////////////////////////////////////////////


// the following kernel is for when using LeesEdwardsSimulationBox
__global__ void update_deltaStrain(float* lastBoxShift, float2* dStrain, float* simBoxPointer, float skin, float maxCutoff) {
  float Lx = simBoxPointer[1];
  float Lx_inv = simBoxPointer[4];
  float Ly_inv = simBoxPointer[5];

  // change in boxShift
  float deltaStrain = simBoxPointer[7] - lastBoxShift[0];
  deltaStrain -= Lx * rintf( deltaStrain * Lx_inv );
  deltaStrain *= Ly_inv; // convert to strain
  
  float2 tmp = {0.f, 0.f};
  tmp.x = deltaStrain;
  float skin_adj = skin - fabs(deltaStrain) * maxCutoff; 
  if (skin_adj < 0.) skin_adj = 0.;
  tmp.y = 0.25f*(skin_adj*skin_adj);
  dStrain[0] = tmp;
}

// When to update the neighborlist.
template <class S> __global__ void displacement_since_nb_build( unsigned numParticles, float4* r, float4* last_r, float4* image, 
								S* simBox, float* simBoxPointer, float skin2_over_4, unsigned int* updateRequired, float2* dStrain, unsigned num_blocks_excl ){
  if( MyGP < numParticles ){
    float simBoxPtr_local[simulationBoxSize];
    simBox->loadMemory( simBoxPtr_local, simBoxPointer );
    
    // Distance moved since last update.
    float distanceMoved2 = (simBox->calculateDistanceMoved(r[MyGP], last_r[MyGP], simBoxPtr_local, dStrain)).w;

    // Need update?
    float s2_4 = skin2_over_4;

    if(dStrain != 0)
      s2_4 = dStrain[0].y;
    
    if( distanceMoved2 > s2_4 ){
      // The number of blocks MUST be the same as for calls to calculateNBL_N2!
      updateRequired[0] = NumBlocks;
      // Here the number of blocks must be that for apply_exclusion_list
      updateRequired[2] = num_blocks_excl;
      image[MyGP].w = 0.f;
    }
    else{
      // Consistency check performed when configuration IO. Doesn't take account of pair-potentials with different cut-offs
      image[MyGP].w = distanceMoved2;
    }
  }
}



template <class S> __global__ void calculateNBL_N2(int numParticles, int nvp , __restrict__ float4* r,  S* simBox, float* simBoxPointer, float *params, int numTypes, float skin, unsigned* numNbrs, unsigned* nb_list, unsigned maxNumNbrs, unsigned* updateRequired, float4* last_r, float* lastBoxShift, unsigned* exclusionType) {
  const unsigned int tid = MyP + MyT*PPerBlock;
  extern __shared__ float4 s_r[];
  unsigned* s_Count = (unsigned*) &s_r[PPerBlock*TPerPart]; 
  float* s_cut_skin2 = (float*) &s_Count[PPerBlock];
  unsigned* localExclusionTypes = (unsigned*) &s_cut_skin2[numTypes*numTypes];

  if (updateRequired[0]) {

    // Load the simulation box in local memory
    float simBoxPtr_local[simulationBoxSize];
    simBox->loadMemory( simBoxPtr_local, simBoxPointer );


    if (MyT==0) s_Count[MyP]=0;

    // Copy cut-offs plus skin to shared memory, squaring now rather than later
    for (unsigned index=0; index<numTypes*numTypes; index += PPerBlock*TPerPart) {
      unsigned myindex = index+tid;
      if (myindex < numTypes*numTypes) {
	float cut_skin = params[myindex] + skin;
	s_cut_skin2[myindex] = cut_skin * cut_skin;
      }
    }
    
    // Read exclusion types into shared memory.
    for (unsigned index=0; index<numTypes*numTypes; index += PPerBlock*TPerPart) {
      unsigned myindex = index+tid;
      if (myindex < numTypes*numTypes) {
	localExclusionTypes[myindex] = exclusionType[myindex];
      }
    }
    float4 my_r = r[MyGP];  
    int my_type = __float_as_int(my_r.w);
  

    for (int FirstGP=0; FirstGP<numParticles; FirstGP+=TPerPart*PPerBlock) {

      int ReadPart = FirstGP + tid;
      if (ReadPart<numParticles) {
	s_r[tid] = r[ReadPart];
      }
      
      __syncthreads();  // Shared data in s_r ready
    
#pragma unroll 
      for (int i=0; i<PPerBlock*TPerPart; i+=TPerPart) {
	int OtherP = i + MyT;
	int OtherGP = FirstGP + OtherP;
	
	if (MyGP<numParticles && MyGP!=OtherGP && OtherGP < numParticles) {
	  float4 r_i = s_r[OtherP]; 
	  int type_i =  __float_as_int(r_i.w);
	  int interaction_index = my_type*numTypes+type_i;
	
	  float RcutSk2 = s_cut_skin2[interaction_index]; // Only 2% faster overall (N=2k)

	  float dist2 = (simBox->calculateDistance(my_r, r_i, simBoxPtr_local)).w;

	  if (dist2 < RcutSk2 && !(localExclusionTypes[my_type*numTypes + type_i])) {
	    // second argument is where the counter resets to zero
	    unsigned int nextNbrIdx= atomicInc(&s_Count[MyP], numParticles);
	    if (nextNbrIdx<maxNumNbrs)
	      nb_list[nvp*nextNbrIdx + MyGP] = OtherGP;
	    else
	      break;
	  } // if(dist2 ... )
	} // if (MyGP ... )
      } // for(int i ... ) 
      __syncthreads();  // Done with shared data before proceeding to next block


    } // for (int firstGP ... )

    __syncthreads();


    // then a reduction over blocks.

    if (MyT == 0) {
      last_r[MyGP] = my_r;
      numNbrs[MyGP] = s_Count[MyP];
      // require one unused space in the neighborlist (per particle)
      // in order to avoid the if statement in CalcF (tp=1). So the test is for
      // >= maxNumNbrs instead of > maxNumNbrs
      if(s_Count[MyP] >= maxNumNbrs)
	updateRequired[1] = 1;
      // the second argument is the maximum value that will be decremented 
      // (if the existing value is greater nothing happens) or the value to 
      // reset to if 0 is decremented. We just want it to be at least NumBlocks
      if (MyP == 0 )  {
	atomicDec(&(updateRequired[0]), NumBlocks);
	
	if(lastBoxShift != 0)
          lastBoxShift[0] = simBoxPtr_local[7];
      }
    } // if (MyT == 0 ... )
  } // if(udpate_required[0] .. ))
}




// Create NB-list by sorting (like "particles" demo in SDK)

__host__ __device__ int3 calculateCellCoordinates( float4 r, float4 inv_sim_box_lengths, int3 num_cells_vec ) {
  float3 scaled_r;
  int3 Coordinates;
  
  scaled_r.x = r.x*inv_sim_box_lengths.x + 0.5f;
  scaled_r.y = r.y*inv_sim_box_lengths.y + 0.5f;
  scaled_r.z = r.z*inv_sim_box_lengths.z + 0.5f;
  
  Coordinates.x = (int) (num_cells_vec.x*(scaled_r.x     ));//- floor(scaled_r.x)) );
  if(Coordinates.x == num_cells_vec.x)
    Coordinates.x -= 1;
  else if (Coordinates.x < 0)
    Coordinates.x = 0;
  
  Coordinates.y = (int) (num_cells_vec.y*(scaled_r.y     ));//   - floor(scaled_r.y)) );
  if(Coordinates.y == num_cells_vec.y)
    Coordinates.y -= 1;
  else if (Coordinates.y < 0)
    Coordinates.y = 0;

  Coordinates.z = (int) (num_cells_vec.z*(scaled_r.z    ));// - floor(scaled_r.z)) );  
  if(Coordinates.z == num_cells_vec.z)
    Coordinates.z -= 1;
  else if (Coordinates.z < 0)
    Coordinates.z = 0;

  return Coordinates;
}



__host__ __device__ int calculateCellIndex( int3 CellCoordinates, int3 num_cells_vec ) {
  // Try something more clever?
  return CellCoordinates.x + CellCoordinates.y * num_cells_vec.x + CellCoordinates.z * num_cells_vec.x * num_cells_vec.y;
}

template <class S>
__global__ void calculateCellIndices( float4 *r, 
				      S* simBox, float* simBoxPointer,
				      float4 inv_cell_lengths,
				      int3 num_cells_vec,
				      int *cellIndex,
				      int *cellStart,  int *cellEnd) {
  unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x; // global thd
  unsigned int num_cells3D = num_cells_vec.x * num_cells_vec.y * num_cells_vec.z;


  // This should handle the case of num_cells3D > numParticles
  for (unsigned int index1 = 0; index1 < num_cells3D; index1 += gridDim.x*blockDim.x) {
    unsigned index2 = index1 + gtid;
    if (index2<num_cells3D) {
      cellStart[index2] = 0;
      cellEnd[index2] = -1;//0;
    }   
  }

  float simBoxPtr_local[simulationBoxSize];
  simBox->loadMemory( simBoxPtr_local, simBoxPointer );


  float4 my_r = r[gtid];
  float4 inv_sim_box_lengths = {1.f/simBoxPtr_local[1], 1.f/simBoxPtr_local[2], 1.f/simBoxPtr_local[3], 0.f};
  int3 my_CellCoordinates = calculateCellCoordinates(my_r, inv_sim_box_lengths, num_cells_vec);
  int my_cellIndex = calculateCellIndex(my_CellCoordinates, num_cells_vec);

  cellIndex[gtid] = my_cellIndex;
}


__global__ void calculateCellsSorted( int numParticles, int *cellIndex, int *cellStart,  int *cellEnd) {
  // First and last thread in each thread-block used to load halo

  extern __shared__ int s_index[];

  int gtid = blockIdx.x*(blockDim.x-2) + threadIdx.x - 1;
  int my_cellIndex = 0;

  if(gtid >= 0 && gtid<numParticles)
    my_cellIndex = cellIndex[gtid];
  s_index[threadIdx.x] = my_cellIndex;
  __syncthreads();
  
  if ( (gtid>=0) && (gtid<numParticles) ) {                       // is it a real particle?
    if ( (threadIdx.x>0) && (threadIdx.x<blockDim.x-1) ) {  // and not halo

      if ( (s_index[threadIdx.x-1]!=my_cellIndex) || (gtid==0) ) {
        cellStart[my_cellIndex] = gtid; // My left neighbor has different cellIndex 
      }	                             // (or I'm first particle), so I'm the first with my cellIndex
      
      if ( (s_index[threadIdx.x+1]!=my_cellIndex) || (gtid==numParticles-1)) {
	cellEnd[my_cellIndex] = gtid; // My right neighbor has different cellIndex 
      }	                             // (or I'm last particle), so I'm the last with my cellIndex
      
    }
  }

}

template<class S>
__global__ void calculateNBL_CellsSorted( int numParticles, int nvp, 
					  float4 *r, int *cellStart,
					  int *cellEnd, S* simBox,
					  float* simBoxPointer,
					  float *params, int numTypes, 
					  float skin, 
					  float4 inv_cell_lengths,
					  int3 num_cells_vec,
					  unsigned* numNbrs, 
					  unsigned* nbl, unsigned maxNumNbrs, 
					  unsigned int *updateRequired, 
					  float4* last_r,
					  float* lastBoxShift,
					  float box_shift,
					  unsigned* exclusionType) {
  
  int gtid = blockIdx.x*blockDim.x + threadIdx.x;
  int Count = 0;
  int scaled_box_shift = int(box_shift*inv_cell_lengths.x);
  extern __shared__ float s_cut_skin2[];
  unsigned* localExclusionTypes = (unsigned*) &s_cut_skin2[numTypes*numTypes];
  // Read exclusion types into shared memory.
  
  for (unsigned int index=0; index < numTypes*numTypes; index += blockDim.x) {
    unsigned int myindex = index + threadIdx.x;
    if (myindex < numTypes*numTypes) {
      localExclusionTypes[myindex] = exclusionType[myindex];
    }
  }

  // Copy cut-offs plus skin to shared memory, squaring now rather than later
  for (unsigned index=0; index<numTypes*numTypes; index += blockDim.x) {
    unsigned myindex = index + threadIdx.x;
    if (myindex < numTypes*numTypes) {
      float cut_skin = params[myindex] + skin;
      s_cut_skin2[myindex] = cut_skin * cut_skin;
    }
  }
  __syncthreads();

  
  if (gtid<numParticles) {
    
    float simBoxPtr_local[simulationBoxSize];
    simBox->loadMemory( simBoxPtr_local, simBoxPointer );
    
    float4 my_r = r[gtid];
    int my_type = __float_as_int(my_r.w);
    

    float4 inv_sim_box_lengths = {1.f/simBoxPtr_local[1], 1.f/simBoxPtr_local[2], 1.f/simBoxPtr_local[3], 0.f};
    int3 my_CellCoordinates = calculateCellCoordinates(my_r, inv_sim_box_lengths, num_cells_vec);

    int3 OtherCellCoordinates;

    // Should try to avoid load-imbalancing below
    for (int dZ=-n_Cells_def; dZ<=n_Cells_def; dZ++) {
      OtherCellCoordinates.z = (my_CellCoordinates.z + dZ + num_cells_vec.z)%num_cells_vec.z;

      for (int dY=-n_Cells_def; dY<=n_Cells_def; dY++) {
	OtherCellCoordinates.y = my_CellCoordinates.y + dY;
	int y_wrap = ( (OtherCellCoordinates.y >= num_cells_vec.y) -
		       (OtherCellCoordinates.y < 0) );
	// following is to take care of Lees-Edwards BC
	int correction_lower = -y_wrap * scaled_box_shift - (y_wrap*box_shift > 0);

	int correction_upper = -y_wrap * scaled_box_shift + (y_wrap*box_shift < 0);
	OtherCellCoordinates.y -= y_wrap * num_cells_vec.y;

#pragma unroll
	for (int dX=-n_Cells_def + correction_lower;
	     dX<=n_Cells_def + correction_upper; dX++) {
	  OtherCellCoordinates.x = my_CellCoordinates.x + dX;
	  OtherCellCoordinates.x += ((OtherCellCoordinates.x<0)-(OtherCellCoordinates.x>=num_cells_vec.x))*num_cells_vec.x;
	  
	  int otherCellIndex = calculateCellIndex(OtherCellCoordinates, num_cells_vec);
	  int Start = (cellStart[otherCellIndex]);
	  int End =   (cellEnd[otherCellIndex]);


	  for (int OtherP=Start; OtherP<=End; OtherP++) { 
	    if (gtid != OtherP) {
	      float4 r_i = LOAD(r[OtherP]); 
	      int type_i =  __float_as_int(r_i.w);
	      int interaction_index = my_type*numTypes+type_i;
	
	      float dist2 = (simBox->calculateDistance(my_r, r_i, simBoxPtr_local)).w;  
	      if (dist2 < s_cut_skin2[interaction_index] && !(localExclusionTypes[interaction_index])) 
	      {
		if (Count<maxNumNbrs) {
		  nbl[nvp*Count + gtid] = OtherP;
		  Count++;
		}
		else
		  break; // breaks out of for(int OtherP ... )
	      }
	    }
	  } // end for (int OtherP....)
	}
      }
    } // end for (int dZ ... )
    
    last_r[gtid] = my_r;
    numNbrs[gtid] = Count;
    if(Count >= maxNumNbrs)
      updateRequired[1] = 1;

    if ( gtid==0 ) {
      updateRequired[0] = 0;
      if(lastBoxShift != 0)
	lastBoxShift[0] = simBoxPtr_local[7];
    }
  } // if(gtid < numParticles)

}



__global__ void apply_exclusion_list(unsigned numParticles,
				     unsigned int nvp,
				     unsigned* neighborList,
				     unsigned* numNbrs,
				     unsigned* exclusionList,
				     unsigned* numExcluded,
				     unsigned* updateRequired) {
  if( updateRequired[2] && MyGP < numParticles ) {
    unsigned my_num_nbrs = numNbrs[MyGP];
    //unsigned my_num_nbrs_start = my_num_nbrs;
    unsigned num_excl_left = numExcluded[MyGP];

    const unsigned batch_size = 8;
    // need to read exclusions into local memory to avoid all those reads
    unsigned my_exclusions[batch_size];
    unsigned num_batches_done = 0;
    unsigned next_nbr;

    while (num_excl_left > 0) {
      unsigned num_excl_read = (num_excl_left > batch_size ? batch_size : num_excl_left);

      my_exclusions[0] = exclusionList[MyGP+nvp*(batch_size *  num_batches_done + 0)];
      if(num_excl_read > 1)
	my_exclusions[1] = exclusionList[MyGP+nvp*(batch_size *  num_batches_done + 1)];      
      if(num_excl_read > 2)
	my_exclusions[2] = exclusionList[MyGP+nvp*(batch_size *  num_batches_done + 2)];


      if(num_excl_read > 3)
	my_exclusions[3] = exclusionList[MyGP+nvp*(batch_size *  num_batches_done + 3)];
      if(num_excl_read > 4)
	my_exclusions[4] = exclusionList[MyGP+nvp*(batch_size *  num_batches_done + 4)];
      
      if(num_excl_read > 5)
	my_exclusions[5] = exclusionList[MyGP+nvp*(batch_size *  num_batches_done + 5)];
      if(num_excl_read > 6)
	my_exclusions[6] = exclusionList[MyGP+nvp*(batch_size *  num_batches_done + 6)];
      if(num_excl_read > 7)
      my_exclusions[7] = exclusionList[MyGP+nvp*(batch_size *  num_batches_done + 7)];
      
      /*if(num_excl_read > 8)
	my_exclusions[8] = exclusionList[MyGP+nvp*(batch_size *  num_batches_done + 8)];
      if(num_excl_read > 9)
      my_exclusions[9] = exclusionList[MyGP+nvp*(batch_size *  num_batches_done + 9)];*/

      
      // next_excl = batch_size *  num_batches_done + rdx;
      /*for(unsigned rdx=0; rdx < num_excl_read; rdx++) {
	unsigned next_excl = batch_size *  num_batches_done + rdx;
	my_exclusions[rdx] = exclusionList[MyGP+nvp*next_excl];
	}*/
      unsigned write_idx = 0;
      for (unsigned read_idx = 0; read_idx < my_num_nbrs; read_idx++) {
	next_nbr = neighborList[nvp*read_idx + MyGP];
	
	bool have_exclusion = false;
	if (next_nbr == my_exclusions[0])
	  have_exclusion = true;

       
	if (num_excl_read > 1 && next_nbr == my_exclusions[1])
	  have_exclusion = true;
	
	
	if (num_excl_read > 2 && next_nbr == my_exclusions[2])
	  have_exclusion = true;
	if (num_excl_read > 3 && next_nbr == my_exclusions[3])
	  have_exclusion = true;
	if (num_excl_read > 4 && next_nbr == my_exclusions[4])
	  have_exclusion = true;
	if (num_excl_read > 5 && next_nbr == my_exclusions[5])
	  have_exclusion = true;
	if (num_excl_read > 6 && next_nbr == my_exclusions[6])
	  have_exclusion = true;
	if (num_excl_read > 7 && next_nbr == my_exclusions[7])
	  have_exclusion = true;	
	/*if (num_excl_read > 8 && next_nbr == my_exclusions[8])
	  have_exclusion = true;
	if (num_excl_read > 9 && next_nbr == my_exclusions[9])
	have_exclusion = true; */

	

	/*for(unsigned edx=0; edx < num_excl_read; edx++)
	  if (next_nbr == my_exclusions[edx]) {
	    have_exclusion = true;
	    break;
	    }*/

	if(!have_exclusion) {
	  neighborList[nvp*write_idx + MyGP] = next_nbr;
	  write_idx++;
	}
      
      } // for (read_idx=0 ...)

      num_excl_left -= num_excl_read;
      num_batches_done++;
      my_num_nbrs = write_idx;
    } // while(num_excl_left...)
    
    numNbrs[MyGP] = my_num_nbrs;
    
    if (threadIdx.x == 0 )  {
      atomicDec(&(updateRequired[2]), NumBlocks);
    }
    
  } // if(MyGP ...)
  
}



__global__ void save_nblist_kernel(unsigned numParticles, unsigned int nvp, unsigned* neighborList, unsigned* savedNeighborList, unsigned* numNbrs) {
  if( MyGP < numParticles) {
    unsigned my_num_nbrs = numNbrs[MyGP];
    for(unsigned idx = 0; idx < my_num_nbrs; idx++)
      savedNeighborList[nvp*idx + MyGP] = neighborList[nvp*idx + MyGP];
  }
}

__global__ void update_nblist_after_sorting(unsigned numParticles, unsigned int nvp, unsigned* neighborList, unsigned* savedNeighborList, unsigned max_num_nbrs, unsigned* numNbrs, unsigned* new_index) {
  unsigned my_old_index = blockIdx.x * blockDim.x + threadIdx.x;
  if( my_old_index < numParticles) {

    // get my new_index (coalesced access)
    unsigned my_new_index = new_index[my_old_index]; // where to write my neighbors
    
    // find number of my neighbors
    unsigned my_num_nbrs = numNbrs[my_old_index];

    // loop over number of my neighbors
    for(unsigned idx = 0; idx < my_num_nbrs; idx++) {
      unsigned nbr_old_index = LOAD(savedNeighborList[nvp*idx + my_old_index]);
      unsigned nbr_new_index = new_index[nbr_old_index];
      
      neighborList[nvp*idx + my_new_index] = nbr_new_index;
    } // for(int idx=0; ... )
    
  } // if (my_old_index < numParticles)
}
