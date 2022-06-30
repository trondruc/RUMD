#ifndef SAMPLE_H
#define SAMPLE_H

/*
    Copyright (C) 2010  Thomas Schrøder

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/

#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

#include "rumd/rumd_base.h"
#include "rumd/rumd_technical.h"
#include "rumd/LogLinOutputManager.h"

#include "rumd/SimulationBox.h"
#include "rumd/ParticleData.h"
#include "rumd/NeighborList.h"
#include "rumd/KernelPlan.h"

class Integrator; class Potential;
class MoleculeData;
class EnergiesOutputManager;  class ExternalCalculator;
class ConfigurationWriterReader;

/// Sample class
class Sample{
  friend class ConfigurationWriterReader;

private:
  unsigned int particles_per_block;
  unsigned int threads_per_particle;
  unsigned int degreesOfFreedom;
  unsigned int userDegreesOfFreedom;
  unsigned long int blockSize;

  // Input/Output
  std::string trajectoryDir;
  bool managers_initialized;
  std::map<std::string, LogLinOutputManager*> outputManagers;

  LogLinOutputManager* restartManager;
  LogLinOutputManager* trajectoryManager;
  EnergiesOutputManager* energiesOutputManager; 
  std::vector<Potential*> potentialList;


  bool verbose;
  bool check_cuda_errors;
  float4 CalculateCenterOfMass( float4* position, float4* velocity, int moleculeIndex );

  ConfigurationWriterReader* restartWriter;
  ConfigurationWriterReader* trajectoryWriter;
  void CreateDefaultOutputManagers();

  // size thresholds for choosing default pb and tp
  static const unsigned int default_pb_size_threshold;
  static const unsigned int default_tp_size_threshold;

 protected:
  SimulationBox* simulationBox;
  bool own_sim_box;
  bool include_kinetic_stress;
  Integrator* itg; 
  ParticleData particleData;
  MoleculeData* moleculeData;
  KernelPlan kPlan;
  SortingScheme sortingScheme;
  void UpdateAfterSorting(thrust::device_vector<unsigned int> &thrust_old_index);
  
 public:
  Sample( unsigned int pb, unsigned int tp );
  Sample(const Sample& S);
  Sample* Copy();
  ~Sample();
  Sample& operator=(const Sample& S);

  void InitializeOutputManagers(unsigned long int timeStepIndex);
  void NotifyOutputManagers(unsigned long int timeStepIndex);
  void TerminateOutputManagers(); // close open files etc
  void ReadConf( const std::string& filename, bool init_itg=false );
  void ReadRestartConf(int restartBlock, unsigned int numDigitsBlkIdx=4 );
  void WriteConf(const std::string& filename, const std::string& mode="w");
  void AddOutputManager(const std::string& manager_name, LogLinOutputManager* om);
 
  void CalcF(bool calc_stresses = false);
  void IsotropicScaleSystem( float Rscal );
  void IsotropicScaleSystemCM( float Rscal );
  void AnisotropicScaleSystem( float Rscal, unsigned dir );
  void AffinelyShearSystem( float shear_strain );
  void ScaleVelocities( float factor );
  void CopySimulationDataToDevice() const;
  void CopySimulationDataFromDevice() const;
  void AddPotential(Potential* potential);
  void UpdateDegreesOfFreedom(unsigned numberOfConstraints=0);
  void SetNumberOfDOFs(unsigned int DOFs);
  void SetMoleculeData(MoleculeData* md) { moleculeData = md; }
  void SetMass( unsigned type, float mass ) {particleData.SetMass(type, mass);}
  void SetAllMasses( double* mass_array, int length ) {particleData.SetAllMasses(mass_array, length);}

  // Spatial sorting
  void SortParticles();
  template<typename T>
  void SortParticlesByKey(T thrust_ptr_sort_key);
  //  void SortParticlesByKey(thrust::device_vector<float>& thrust_sort_key);
  // Set methods
  void EnableBackup(bool make_backup);
  void SetOutputBlockSize(unsigned long int blockSize);
  void SetOutputManagerActive(const std::string &manager_name, bool active);
  void SetLogLinParameters(const std::string& manager_name, unsigned long int base, unsigned long int maxInterval, long int user_maxIndex=-1);
  void SetOutputManagerMetaData(const std::string& manager_name, const std::string& key, bool on);
  void SetOutputManagerMetaData(const std::string& manager_name, const std::string& key, int value);

  void AddExternalCalculator(ExternalCalculator* calc);
  void RemoveExternalCalculator(ExternalCalculator* calc); 

  void SetNumberOfParticles(unsigned int num_part);
  void SetSimulationBox(SimulationBox* simBox, bool own_this_box=false);
  void SetIntegrator(Integrator* newItg);
  void SetPotential(Potential* potential);
  void SetVerbose(bool vb);
  void SetCheckCudaErrors(bool set_check_cuda_errors) {check_cuda_errors = set_check_cuda_errors; }
  void SetOutputDirectory(const std::string& trajectoryDir);

  //void SetConfIO_Format(unsigned int ioformat);

  void SetPB_TP( unsigned int pb, unsigned int tp );
  void SetSortingScheme(SortingScheme ss);
  void SetIncludeKineticStress(bool include_kinetic_stress) {this->include_kinetic_stress = include_kinetic_stress; } 

  // Get methods.
  SimulationBox* GetSimulationBox() const { return simulationBox; }  
  ParticleData* GetParticleData() { return &particleData; }
  const ParticleData* GetParticleDataConst() const { return &particleData; }
  MoleculeData* GetMoleculeData() { return moleculeData; }
  Integrator* GetIntegrator() const { return itg; }
  std::vector<Potential*>* GetPotentials() { return &potentialList; }  
  const KernelPlan& GetKernelPlan() const { return kPlan; }
  std::string GetOutputDirectory() const { return trajectoryDir; }

  unsigned int GetParticlesPerBlock() const { return particles_per_block; }
  unsigned int GetThreadsPerParticle() const { return threads_per_particle; }
  unsigned int GetNumberOfDOFs() const;
  unsigned int GetNumberOfParticles() const { return particleData.GetNumberOfParticles(); }
  unsigned int GetNumberOfVirtualParticles() const { return particleData.GetNumberOfVirtualParticles(); }
  unsigned int GetNumberOfTypes() const { return particleData.GetNumberOfTypes(); }
  unsigned int GetNumberThisType(unsigned int type) const { return particleData.GetNumberThisType(type); }
  unsigned long int GetOutputBlockSize() const { return blockSize; }

  float GetMass(unsigned int type) const { return particleData.GetMass(type); }
  float GetMeanMass() const { float meanMass = 0; for(unsigned i=0; i < GetNumberOfTypes(); i++){ meanMass += GetNumberThisType(i) * GetMass(i); } meanMass /= GetNumberOfParticles(); return meanMass; }


  double GetPotentialEnergy(bool copy=true) const;  
  double GetVirial(bool copy=true) const;
  std::vector<double> GetStress(bool copy=true) const;
  SortingScheme GetSortingScheme() const { return sortingScheme; }
};



__global__  void calc_sort_key( unsigned int num_part, float4* r, float *L, unsigned int Nx,  unsigned int Ny,  unsigned int Nz,  float* key);

__global__ void invert_index_mapping( unsigned* old_index, unsigned* new_index, unsigned numParticles );

// I wanted T to be the template argument of a thrust::dev_ptr but the compiler couldn't handle that
template<typename T>
void Sample::SortParticlesByKey(T thrust_ptr_sort_key) {
  
  unsigned int numParticles = particleData.GetNumberOfParticles();
  // generate index vector to be used as value in sorting - used for updating data structures after sorting 
  thrust::device_vector<unsigned int> thrust_old_index(numParticles);
  thrust::sequence(thrust_old_index.begin(), thrust_old_index.end());

  // Sort key and values.
  // original version where sort_key was a vector
  thrust::sort_by_key(thrust_ptr_sort_key, thrust_ptr_sort_key+numParticles, thrust_old_index.begin());
  

  UpdateAfterSorting(thrust_old_index);
}




#endif // SAMPLE_H
