#ifndef NEIGHBORLIST_H
#define NEIGHBORLIST_H

/*
    Copyright (C) 2010  Thomas Schrøder

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/

#include "rumd/rumd_technical.h"
#include "rumd/rumd_base.h"
#include "rumd/ParticleData.h"
#include "rumd/KernelPlan.h"
#include "rumd/SimulationBox.h"
#include <map>


class Sample; class PairPotential;

class NeighborList {
 
 private:
  NeighborList(const NeighborList&); 
  NeighborList& operator=(const NeighborList& N);
  Sample* sample;

  ParticleData* particleData;
  KernelPlan kp;
  SimulationBox* simBox;
  RectangularSimulationBox* testRSB;
  LeesEdwardsSimulationBox* testLESB;

  unsigned int numBlocks;
  unsigned int numParticles;
  unsigned int particlesPerBlock;
  unsigned int numVirtualParticles; 

  std::map<std::string, unsigned int> numberAllocatedUnits;



  float skin;
  float maxCutoff;
  unsigned int maxNumNbrs;
  unsigned int maxNumExcluded;
  unsigned int numTypes;

  enum NB_method {DEFAULT, NONE, N2, SORT} nb_method;
  std::map<std::string, NB_method> method_str_map;

  bool allocate_max;
  bool allow_update_after_sorting;
  int rebuild_required;
  int rebuild_was_required;
  bool have_exclusion_type;

  float* d_lastBoxShift; // for LE sim box, also includes deltaStrain and adjusted skin
  float2* d_dStrain;     // deltaStrain and 0.25 times the square of the strain-adjusted skin
  unsigned int* d_updateRequired;

  unsigned * d_neighborList;             // Neighbor list
  unsigned * d_savedNeighborList;   // For storing a neighbor list
  unsigned * d_numNbrs;            // number of neighbors for each particle

  unsigned* h_exclusionType;
  unsigned* d_exclusionType;
  
  unsigned* h_exclusionList;
  unsigned* d_exclusionList;
  unsigned* d_numExcluded;
  unsigned* h_numExcluded;


  float* h_cutoffArray;
  float* d_cutoffArray;
  float4* d_last_r;

  float4* d_temp_float4; 
  int* d_temp_int;

  int* d_cellIndex;
  int* d_cellStart;
  int* d_cellEnd;


  void AllocateNeighborList();
  void AllocateSavedNeighborList();
  void AllocateNumNbrs();

  void AllocateExclusionType( unsigned int numberOfTypes );
  void AllocateExclusionList();
  void AllocateNumExcluded();

  void AllocateCutoffArray();
  void AllocateLastPositions();
  void AllocateCellArrays(unsigned int num_cells);
  void AllocateTempArrays();

  void CheckRebuildRequired();

  static const unsigned int min_size_memory_opt;
  static const unsigned int default_maxNumNbrs;
  static const unsigned int block_size_cells;
  static const int n_Cells;
  static const size_t sharedMemPerBlock;
  static const unsigned int default_NB_method_size_threshold;
  static const unsigned int block_size_excl;

 public:
  NeighborList();
  ~NeighborList();

  void Initialize(Sample* sample);
  void UpdateNBlist();
  void UpdateAfterSorting( unsigned* old_index, unsigned* new_index);

  void CalculateNBL_Sorting();
  void CopyExclusionListToDevice();
  void CopyExclusionTypeToDevice();
  void SaveNeighborList(bool save_exclusion_list=false);
  void RestoreNeighborList();
  void ResetNeighborList();

  // Set methods.
  void SetNB_Method(const std::string& method);
  void SetNB_Method(NB_method method);
  void SetCutoffs(PairPotential* potential);
  void SetEqualCutoffs(unsigned int num_types, float cutoff);
  void SetSkin(float skin) { this->skin = skin; }
  void SetExclusion( unsigned particleI, unsigned particleJ );

  void SetExclusionBond(uint1* h_btlist, uint2* h_blist, unsigned max_num_bonds, unsigned etype); 
  void SetExclusionAngle(uint4* h_alist, unsigned num_angles);
  void SetExclusionDihedral(uint4* h_dlist, unsigned num_dihedrals);
  void SetExclusionMolecule(int1* h_mlist, unsigned molindex, unsigned max_num_uau, unsigned num_mol);
  void SetExclusionType( unsigned type0, unsigned type1 );
  // the following two are mostly for testing/debugging
  void SetAllocateMax( bool set_allocate_max) {this->allocate_max = set_allocate_max;}
  void SetMaxNumNbrs(unsigned set_maxNumNbrs) { this->maxNumNbrs = set_maxNumNbrs; }
  unsigned GetActualMaxNumNbrs() const;


  // Get methods.
  std::string GetNB_Method();  
  unsigned* GetNbListPtr() const  // not const in deep sense
  {
    if(nb_method == NONE)
      return 0;
    else
      return d_neighborList;
  }
  
  unsigned* GetNumNbrsPtr() const  // not const in deep sense
  {
    if(nb_method == NONE)
      return 0;
    else
      return d_numNbrs;
  }
  float GetSkin() const { return skin; }
  float GetMaxCutoff() const { return maxCutoff; }
  int GetRebuildRequired() const { return rebuild_was_required; }
};


__global__ void update_deltaStrain(float* lastBoxShift, float2* dStrain, 
				   float* simBoxPointer, float skin, 
				   float maxCutoff);

template <class S>
__global__ void displacement_since_nb_build( unsigned numParticles, float4* r,
					     float4* last_r, float4* image,
					     S* simBox, float* simBoxPointer, 
					     float skin2over_4, 
					     unsigned* updateRequired, 
					     float2* dStrain,
					     unsigned num_blocks_excl);

template <class S>
__global__ void calculateNBL_N2( int numParticles, 
				 int nvp, 
				 __restrict__ float4* r,  
				 S* simBox, 
				 float* simBoxPointer,
				 float *params,
				 int numTypes,
				 float skin, 
				 unsigned* numNbrs,
				 unsigned* neighborList, 
				 unsigned maxNumNbrs, 
				 unsigned *updateRequired,
				 float4* last_r, 
				 float* lastBoxShift,
				 unsigned* exclusionType );

template <class S>
__global__ void calculateCellIndices( float4 *r, 
				     S* simBox,
				     float* simBoxPointer, 
				     float4 inv_cell_lengths,
				     int3 num_cells_vec,
				     int *CellIndex, 
				     int *CellStart,  int *CellEnd );

__global__ void calculateCellsSorted( int num_part, int *CellIndex,
				      int *CellStart,  int *CellEnd );

template <class S>
__global__ void calculateNBL_CellsSorted( int numParticles, int nvp, 
					 float4 *r, int *CellStart,
					 int *CellEnd, S* simBox, 
					 float* simBoxPointer,
					 float *params, int numTypes,
					 float skin, 
					 float4 inv_cell_lengths,
					 int3 num_cells_vec,
					 unsigned* numNbrs,
					 unsigned* nbl,
					 unsigned maxNumNbrs, 
					 unsigned* updateRequired,
					 float4* last_r,
					 float* lastBoxShift,
					 float box_shift,
					 unsigned* exclusionType );

__global__ void update_nblist_after_sorting(unsigned numParticles, 
					    unsigned nvp, 
					    unsigned* neighborList, 
					    unsigned* savedNeighborList, 
					    unsigned max_numNbrs, 
					    unsigned* numNbrs, 
					    unsigned* new_index);

__global__ void save_nblist_kernel(unsigned numParticles, 
				   unsigned int nvp, 
				   unsigned* neighborList, 
				   unsigned* savedNeighborList, 
				   unsigned* num_nbr);


__global__ void apply_exclusion_list(unsigned numParticles, 
				     unsigned int nvp, 
				     unsigned* neighborList, 
				     unsigned* numNbrs, 
				     unsigned* exclusion_list, 
				     unsigned* num_excluded,
				     unsigned* updateRequired);

#endif // NEIGHBORLIST_H
