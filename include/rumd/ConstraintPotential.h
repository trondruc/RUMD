#ifndef CONSTRAINTPOTENTIAL_H
#define CONSTRAINTPOTENTIAL_H

/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/RUMD_Error.h"
#include "rumd/BondPotential.h"


// Implements holonomic bond constraints into the dynamics.
class ConstraintPotential : public BondPotential{
  
 private:
  ConstraintPotential(const ConstraintPotential&);
  ConstraintPotential& operator=(const ConstraintPotential&); 

  
  unsigned numberOfMolecules;
  unsigned numIterations;
  unsigned matrixAllocated;
  unsigned numberOfLinearSystems;
  unsigned maxConstraintsPerMolecule;
  unsigned totalNumberOfConstraints;
  unsigned totalSquaredNumberOfConstraints;
  unsigned sizeMem;
  bool     linearMolecules;
  
  unsigned* d_dimLinearSystems;
  unsigned* d_dimConsecLinearSystems;
  unsigned* d_dimConsecSqLinearSystems;
  
  float* d_A;
  float* d_b;
  float* d_x;
  float* d_sumStandardDeviation;

  float3* h_constraintInfo;
  float3* d_constraintInfo;
  float3* d_constraintInfoPerParticle;

  float4* d_tempForce;
  
  dim3 grid, grid1, gridBLS;
  dim3 thread, thread1, threadBLS;
  
  void Free();
  
 public: 
  ConstraintPotential();
  ~ConstraintPotential();
  
  void Initialize();
  void CalcF(bool initialize, bool calc_stresses);
  
  void ResetInternalData() {UpdateConstraintLayout();}
  void UpdateConstraintLayout();
  void UpdateAfterSorting( unsigned int *old_index, unsigned int *new_index );
  
  bool EnergyIncludedInParticleSum() const { return false; }
  void WritePotential(SimulationBox* simBox);

  void SetParams(unsigned bond_type, float length_parameter);
  // Get methods
  unsigned GetNumberOfConstraints() const { return totalNumberOfConstraints; }
  double GetVirial();
  // Set methods.
  void SetNumberOfConstraintIterations( unsigned numIte ){ numIterations = numIte; }
  void SetLinearConstraintMolecules( bool linMol ){ linearMolecules = linMol;}
  
  void GetDataInfo(std::map<std::string, bool> &active, std::map<std::string, std::string> &columnIDs);
  void RemoveDataInfo(std::map<std::string, bool> &active, std::map<std::string, std::string> &columnIDs);
  void GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active);
};


template <class S> __global__ void buildConstraintsOnParticle( unsigned* dimLinearSystems, unsigned* dimConsecLinearSystems, float3* d_constraintInfo, 
							       float3* constraintInfoPerParticle, float4* position, float4* velocity, 
							       S* simBox, float* simBoxPointer, unsigned totalNumConstraints, 
							       float totalLinearConstraints );

template <class S> __global__ void buildLinearSystem( float* A, float* b, unsigned* dimLinearSystems, unsigned* dimConsecLinearSystems, 
						      unsigned* dimConsecSqLinearSystems, float3* d_constraintInfo, float3* constraintInfoPerParticle, 
						      float4* position, float4* velocity, float4* force, S* simBox, float* simBoxPointer, 
						      unsigned totalNumConstraints, float localThermostatState, float timeStep, 
						      float totalLinearConstraints );

template <class S> __global__ void updateRHS( float* lagrangeMultiplier, float* b, float4* position, float4* velocity, float4* force, 
					      float3* d_constraintInfo, S* simBox, float* simBoxPointer, float localThermostatState, 
					      float timeStep, float totalLinearConstraints );

template <int updateVirial, class S> __global__ void updateConstraintForce( float4* position, float4* force, float4* virial, float* lagrangeMultiplier, 
									    unsigned* dimLinearSystems, unsigned* dimConsecLinearSystems, 
									    float3* d_constraintInfo, S* simBox, float* simBoxPointer, 
									    float totalLinearConstraints );

__global__ void UpdateConstraintsAfterSorting( unsigned int *new_index, unsigned int nParticles, float3* d_constraintInfo, float totalNumberOfConstraints );

template <class S> __global__ void calculateStandardDeviation( S* simBox, float* simBoxPointer, float4* postion, float3* constraintInfo, float totalConstraints, float* sumArray );

#endif // CONSTRAINTPOTENTIAL_H
