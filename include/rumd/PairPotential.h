#ifndef PAIRPOTENTIAL_H
#define PAIRPOTENTIAL_H

/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/RUMD_Error.h"
#include "rumd/Potential.h"
#include "rumd/NeighborList.h"
#include "rumd/SimulationBox.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <sstream>
#include <map>

////////////////////////////////////////////////////////////////////////////////////////////////////////
// This header implements specialized Pair Potentials. Implement your potential here.
// DISCLAIMER: params_map[pair_i_j] is a vector of floats containing parameters. Rcut_ij should 
// always be the 0'th parameter, i.e., params_map[pair_i_j][0].
////////////////////////////////////////////////////////////////////////////////////////////////////////




class PairPotential : public Potential{
 public:
  enum CutoffMethod {NS, SP, SF};


  virtual void CalcF_Local() = 0;

 private:
  PairPotential(const PairPotential&);
  PairPotential& operator=(const PairPotential&); 
  unsigned int allocated_num_types;

 protected:
  NeighborList neighborList;
  RectangularSimulationBox* testRSB;
  LeesEdwardsSimulationBox* testLESB;
  

  CutoffMethod cutoffMethod;

  std::map< std::pair<unsigned, unsigned>, std::vector<float> > params_map;
  unsigned int allocated_size_pe;    
  float *d_params;
  float4* d_f_pe;
  float4* h_f_pe;
  float4* d_w_pe;
  size_t shared_size;
  bool assume_Newton3;
  
  void AllocatePE_Array(unsigned int nvp);
  
 public: 
  PairPotential( CutoffMethod cutoff_method ); 
  virtual ~PairPotential();

  void SetAllParams(const std::map<std::pair<unsigned, unsigned>, std::vector<float>  > &other_params_map);
  
  // Needed due to no virtual function on device.
  virtual float ComputeInteraction_CPU(float /* dist2 */ , const float* /* param */, float4* /* my_f */) { return 0.f; };  

  bool EnergyIncludedInParticleSum() const { return true; }
  void WritePotentials(SimulationBox* simBox);
  void CopyParamsToGPU( unsigned i, unsigned j ); 
  void CopyAllParamsToGPU(unsigned num_types);
  void SaveNeighborList(){ neighborList.SaveNeighborList(); }
  void RestoreNeighborList(){ neighborList.RestoreNeighborList(); }
  void ResetInternalData(){ neighborList.ResetNeighborList(); }
  void UpdateAfterSorting( unsigned* old_index, unsigned* new_index); 
  // Set methods.
  void SetNbListSkin( float skin ){ neighborList.SetSkin(skin); }
  void SetNB_Method(const std::string& method) {neighborList.SetNB_Method(method);}
  std::string GetNB_Method() { return neighborList.GetNB_Method(); }
  void SetNbListAllocateMax(bool set_allocate_max) {neighborList.SetAllocateMax(set_allocate_max);} 
  void SetNbMaxNumNbrs(unsigned maxNumNbrs) {neighborList.SetMaxNumNbrs(maxNumNbrs); }
  void CopyExclusionListToDevice() {neighborList.CopyExclusionListToDevice();}
  
  void Initialize();

  void SetExclusion(unsigned particleI, unsigned particleJ) {neighborList.SetExclusion(particleI, particleJ);}
  void SetExclusionBond( uint1* h_btlist, uint2* h_blist, unsigned max_num_bonds, unsigned etype );
  void SetExclusionType( unsigned type0, unsigned type1 );
  void SetExclusionMolecule( int1 *h_mlist, unsigned molindex, unsigned max_num_uau, unsigned num_mol );
  void SetExclusionDihedral(uint4 *h_dlist,  unsigned num_dihedrals);
  void SetExclusionAngle(uint4 *h_alist,  unsigned num_angles);
  void SetAssumeNewton3(bool set_N3) { assume_Newton3 = set_N3; }
  
  // Get methods.
  std::vector<float> GetPotentialParameters(unsigned num_types) const {
    std::vector<float> params_vec;
    for(unsigned i = 0; i < num_types; i++)
      for(unsigned j = 0; j < num_types; j++)
	for(unsigned p = 0; p < NumParam-2; p++)
	params_vec.push_back(GetPotentialParameter(i, j, p));
    return params_vec;
  }


  float GetPotentialParameter(unsigned int type1, unsigned int type2, unsigned int paramIdx) const {
    const std::pair<unsigned, unsigned> pair_1_2 = std::make_pair(type1, type2);
    std::map<std::pair<unsigned, unsigned>, std::vector<float> >::const_iterator it = params_map.find(pair_1_2);
    if(it != params_map.end())
      return it->second[paramIdx];
    else
      return 0.0;
  }
  float GetNbListSkin() const { return neighborList.GetSkin(); }
  float GetMaxCutoff() const { return neighborList.GetMaxCutoff(); }
  int GetCutoffMethod() const { return cutoffMethod; }
  int GetNbListRebuildRequired() const { return neighborList.GetRebuildRequired(); }
  unsigned GetActualMaxNumNbrs() const { return neighborList.GetActualMaxNumNbrs(); }

  const float* GetDeviceParamsPtr() const { return d_params; }
  float4* GetLocalForceArray() const {return d_f_pe; }
  double GetPotentialEnergy();
  double GetVirial() { throw RUMD_Error("PairPotential", __func__, "Not implemented yet");}
};

////////////////////////////////////////////////////////////////////////////////////////////////
// General 12-6 Lennard-Jones
// v(r) = A12/r^12 + A6/r^6 
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_IPL_12_6 : public PairPotential {
 public: 
  Pot_IPL_12_6( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("potIPL_12_6");}
  
  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };  
  void CalcF_Local();  
  
  Potential* Copy() {
    Pot_IPL_12_6* new_pot = new Pot_IPL_12_6(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams( unsigned int i, unsigned int j, float A12, float A6, float Rcut ){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut);
    params_map[ pair_i_j ].push_back(A12);
    params_map[ pair_i_j ].push_back(A6);
    
    if(verbose)
      std::cout << GetID_String() + ": " << "A12 = " << A12 << ", A6 = " << A6 << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }                                    
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f){
    float invDist2 = 1.0f / dist2;
    float invDist6 = invDist2 * invDist2 * invDist2;
    float temp = param[1]* invDist6;
    float s = invDist6 * invDist2 * ( 12.f*temp + 6.f*param[2] ); // F_ij = s * r_ij.
    (*my_f).w += invDist6 * ( temp + param[2] );
    return s;
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////
// Standard 12-6 Lennard-Jones
// v(r) = 4 Epsilon ( (Sigma/r)^12 - (Sigma/r)^6 )
////////////////////////////////////////////////////////////////////////////////////////////////

/** Standard 12-6 Lennard-Jones
 This PairPotential implements Lennard-Jones pair forces:
 \f[ 
    v(r) = 4 \varepsilon ((\sigma/r)^{12}-(\sigma/r)^6) 
 \f] 
  Interactions are trunctated at \f$r_c\f$.
 # Usage Example
 \include lennardJones/run.py
*/
class Pot_LJ_12_6 : public Pot_IPL_12_6 {
 public: 
  Pot_LJ_12_6 ( CutoffMethod cutoff_method ) : Pot_IPL_12_6 (cutoff_method) { SetID_String("potLJ_12_6");}
  
  Potential* Copy() {
    Pot_LJ_12_6* new_pot = new Pot_LJ_12_6(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }

  /** Set parameters for potential
  * \param i,j Particle types
  * \param Sigma Pair diameter, \f$ \sigma \f$.
  * \param Epsilon Minimum pair energy,  \f$ \varepsilon \f$.
  * \param Rcut Distance for truncation of pair interactions, \f$ r_c \f$.
  */
  void SetParams( unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut ){
    if(verbose) // this maybe should be replaced by a virtual function to avoid
      // the base class also printing its message, but not necessarily
      std::cout << GetID_String() + ": " << "Sigma = " << Sigma << ", Epsilon = " << Epsilon << ", Rcut = " << Rcut << "." << std::endl;
    Pot_IPL_12_6::SetParams(i, j, 4.*Epsilon*pow(Sigma,12.), -4.*Epsilon*pow(Sigma, 6.), Rcut*Sigma);
  }                                    
};

////////////////////////////////////////////////////////////////////////////////////////////////
// 12-6 Lennard-Jones smoothed 
// v(r) = 4 Epsilon ( (Sigma/r)^12 - (Sigma/r)^6 + c0 + c2*(r/Sigma)^2 + c4*(r/Sigma)^4)
// c0 = 10*xc^-6 -  28*xc^-12, xc = Rcut/Sigma.  We will let cutoffMethod=ShiftedPotential handle this one
// c2 = 48*xc^-14 - 15*xc^-8
// c4 =  6*xc^-10 - 21*xc^-16
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_LJ_12_6_smooth_2_4 : public PairPotential {
 public: 
  Pot_LJ_12_6_smooth_2_4( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("potLJ_12_6_smooth_2_4");}
  
  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };  
  void CalcF_Local();  
  
  Potential* Copy() {
    Pot_LJ_12_6_smooth_2_4* new_pot = new Pot_LJ_12_6_smooth_2_4(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams( unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut ){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    float c2 = 48.*pow(Rcut, -14.0) - 15.*pow(Rcut, -8.);   // Rcut is in units of Sigma here
    float c4 =  6.*pow(Rcut, -10.0) - 21.*pow(Rcut, -16.);  
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut*Sigma); // Cut-off in absolute units // param[0]
    params_map[ pair_i_j ].push_back(1.0f/(Sigma*Sigma));                      // param[1]
    params_map[ pair_i_j ].push_back(4.0f*Epsilon);                            // param[2]
    params_map[ pair_i_j ].push_back(c2);                                      // param[3]
    params_map[ pair_i_j ].push_back(c4);                                      // param[4]
    
    if(verbose)
      std::cout << GetID_String() + ": " << "Sigma = " << Sigma << ", Epsilon = " << Epsilon << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }                                    
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f){
    float Dist2 = dist2*param[1];
    float invDist2 = 1.0f / Dist2;
    float invDist6 = invDist2 * invDist2 * invDist2;
    float s =  param[2]*param[1]*( invDist2*invDist6*(12.f*invDist6 - 6.f) - 2.0f*param[3] - 4.0f*param[4]*Dist2 ) ; // F_ij = s * r_ij.
    (*my_f).w += param[2]*( invDist6*(invDist6 - 1.0f) + param[3]*Dist2 + param[4]*Dist2*Dist2 );
    return s;
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////
// Generalized Lennard-Jones
// v(r) = Epsilon / (m -n) ( n*(Sigma/r)^m -  m*(Sigma/r)^n )
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_gLJ_m_n : public PairPotential {
 public: 
  float m, n; // exponents 
  
  Pot_gLJ_m_n(float set_m, float set_n, CutoffMethod cutoff_method ): PairPotential(cutoff_method), m(set_m), n(set_n){ std::stringstream ss; ss << "potgLJ_" << m << "_" << n; SetID_String(ss.str()); }
  
  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };  
  void CalcF_Local();
  
  Potential* Copy() {
    Pot_gLJ_m_n* new_pot = new Pot_gLJ_m_n(m, n, cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut * Sigma);            // Rcut_ij
    params_map[ pair_i_j ].push_back(1.f / ( Sigma * Sigma ) );// (1/Sigma_ij)^2
    params_map[ pair_i_j ].push_back(m*n / (m-n) * Epsilon);   // Prefactor for forces
    params_map[ pair_i_j ].push_back(m*0.5f);                  // m/2.0
    params_map[ pair_i_j ].push_back(n*0.5f);                  // n/2.0
    params_map[ pair_i_j ].push_back( Epsilon / (m-n));
   
     
    if(verbose)
      std::cout << GetID_String() << ": " << "Sigma = " << Sigma << ", Epsilon = " << Epsilon << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f) {
    float invDist2 = 1.0f / ( dist2 * param[1] );                             
    float invDist_m = exp(log(invDist2)*param[3]); 
    float invDist_n = exp(log(invDist2)*param[4]);
    float s =   param[1] * param[2] * (invDist_m - invDist_n) * invDist2; 
    (*my_f).w += param[5] * ( 2.0f*param[4]*invDist_m - 2.0f*param[3]*invDist_n );
    return s;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////
// Standard 12-6 Lennard-Jones formulated for consistency with Generalized 
// Lennard-Jones (so the minimum is at Sigma)
// v(r) = Epsilon ( (Sigma/r)^12 -  2 *  (Sigma/r)^6 )
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_gLJ_m12_n6 : public Pot_IPL_12_6 {
 public: 
  Pot_gLJ_m12_n6( CutoffMethod cutoff_method ) : Pot_IPL_12_6 (cutoff_method) { SetID_String("potLJA"); }

  Potential* Copy() {
    Pot_gLJ_m12_n6 * new_pot = new Pot_gLJ_m12_n6 (cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut){
    
    if(verbose)
      std::cout << GetID_String() << ": " << "Sigma = " << Sigma << ", Epsilon = " << Epsilon << ", Rcut = " << Rcut << "." << std::endl;
    Pot_IPL_12_6 ::SetParams(i, j, Epsilon*pow(Sigma,12.), -2.*Epsilon*pow(Sigma, 6.), Rcut*Sigma);
  }
  
};

////////////////////////////////////////////////////////////////////////////////////////////////
// 12-6 Lennard-Jones plus Gaussian potential
// v(r) = Epsilon ( (Sigma/r)^12 -  2(Sigma/r)^6 - Epsilon_0 (exp[-(r - r_0)^2 / (2 Sigma_0^2)]) ) 
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_LJ_12_6_Gauss : public PairPotential{
 public: 
  Pot_LJ_12_6_Gauss( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("potLJG"); }

  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };  
  void CalcF_Local();
  
  Potential* Copy() {
    Pot_LJ_12_6_Gauss* new_pot = new Pot_LJ_12_6_Gauss(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;  
  }
  
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut, float Sigma0, float Epsilon0, float r0) {
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut * Sigma);
    params_map[ pair_i_j ].push_back(1.f / ( Sigma * Sigma ));
    params_map[ pair_i_j ].push_back(12.f * Epsilon);
    params_map[ pair_i_j ].push_back(2.0f * Sigma0 * Sigma0);
    params_map[ pair_i_j ].push_back(r0);
    params_map[ pair_i_j ].push_back(Epsilon0);
    params_map[ pair_i_j ].push_back(Epsilon);

    if(verbose)
      std::cout << GetID_String() << ": " << "Sigma = " << Sigma << ", Epsilon = " << Epsilon << ", Sigma0 = " << Sigma0 
		<< ", Epsilon0 = " << Epsilon0 << ", r0 = " << r0 << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }               
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f) {
    float invDist2 = 1.0f / ( dist2 * param[1] );                             
    float invDist6 = invDist2 * invDist2 * invDist2;                          
    float r = sqrtf(dist2);
    float exp_factor = exp (- ((r - param[4]) * (r - param[4])) / param[3]);
    float s = param[1] * param[2] * invDist6 * invDist2 * ( invDist6 - 1.f ) - (param[5] * param[2] * ( r - param[4] ) / ( 6.0f * r *  param[3])) * exp_factor; 
    (*my_f).w += param[6] * (invDist6 * ( invDist6 - 2.0f ) - param[5] * exp_factor );                                                 
    return s;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////
// Gaussian core potential.
// v(r) = Epsilon * exp[-(r/Sigma)^2]
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_Gauss : public PairPotential{
 public: 
  Pot_Gauss( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("potGau"); }

  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };  
  void CalcF_Local();

  Potential* Copy() {
    Pot_Gauss* new_pot = new Pot_Gauss(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }

  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut * Sigma);
    params_map[ pair_i_j ].push_back(1.f / ( Sigma * Sigma ));
    params_map[ pair_i_j ].push_back(Epsilon);

    if(verbose)
      std::cout << GetID_String() << ": " << "Sigma = " << Sigma << ", Epsilon = " << Epsilon << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f){
    float Dist2 = dist2 * param[1];                    // (r/Sigma)^2 
    float s = 2.f * param[1] * param[2] * exp(-Dist2); // F_ij = s * r_ij
    (*my_f).w += param[2] * exp(- Dist2);              // U_ij 
    return s;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////
// Buckingham potential.
// v(r) = Epsilon{(6/(Alpha - 6))*exp[Alpha(1-r/rm)] - (Alpha/(Alpha -6))*(rm/r)**6}
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_Buckingham : public PairPotential {
 public: 
  Pot_Buckingham( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("potBuc"); }
  
  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };  
  void CalcF_Local();

  Potential* Copy() {
    Pot_Buckingham* new_pot = new Pot_Buckingham(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams(unsigned int i, unsigned int j, float rm, float Alpha, float Epsilon, float Rcut){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut);
    params_map[ pair_i_j ].push_back(1.f / (rm * rm));
    params_map[ pair_i_j ].push_back(Alpha);
    params_map[ pair_i_j ].push_back(6.f * Epsilon / (Alpha - 6.f));
    params_map[ pair_i_j ].push_back(Alpha * Epsilon / (Alpha - 6.f));
    params_map[ pair_i_j ].push_back(6.f * Epsilon * Alpha / (Alpha -6));

    
    if(verbose)
      std::cout << GetID_String() << ": " << "Alpha = " << Alpha << ", Epsilon = " << Epsilon << ", rm = " << rm << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f){
    float invDist2 = 1.0f / ( dist2 * param[1] );           // (rm/r)^2
    float invDist6 = invDist2 * invDist2 * invDist2;        // (rm/r)^6
    float invDist  = sqrtf(invDist2);                        //  rm/r
    float expDist  = exp(param[2] * (1.f - 1.f/invDist));   // exponential term 

    float s = param[1]*param[5] * ( invDist*expDist - invDist6*invDist2 );
    (*my_f).w +=  param[3] * expDist - param[4] * invDist6;
    return s;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////
// IPL potential, n=12
// v(r) = Epsilon ( (Sigma/r)^12 )
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_IPL_12 : public PairPotential{
 public: 
  Pot_IPL_12( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("potIPL12"); }
  
  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };  
  void CalcF_Local();
  
  Potential* Copy() {
    Pot_IPL_12* new_pot = new Pot_IPL_12(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut) {
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut * Sigma);             // Rcut_ij
    params_map[ pair_i_j ].push_back(1.f / ( Sigma * Sigma )); // (1/Sigma_ij)^2
    params_map[ pair_i_j ].push_back(12.f * Epsilon);  // Prefactor for forces
    params_map[ pair_i_j ].push_back(Epsilon);
    

    if(verbose)
      std::cout << GetID_String() << ": " << "Sigma = " << Sigma << ", Epsilon = " << Epsilon << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f) {
    float invDist2  = 1.0f / ( dist2 * param[1] );        // (sigma/r)^2 
    float invDist6  = invDist2 * invDist2 * invDist2;     // (sigma/r)^6
    float invDist12 = invDist6 * invDist6;                // (sigma/r)^12    
    float s = param[1] * param[2] * invDist12 * invDist2; // F_ij = s * r_ij
    (*my_f).w += param[3] * invDist12;                  
    return s;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////
// IPL potential, n=9
// v(r) = Epsilon ( (Sigma/r)^9 )
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_IPL_9 : public PairPotential{
 public: 
  Pot_IPL_9( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("potIPL9"); }
  
  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };  
  void CalcF_Local();
  
  Potential* Copy() {
    Pot_IPL_9* new_pot = new Pot_IPL_9(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut) {
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut * Sigma);             // Rcut_ij
    params_map[ pair_i_j ].push_back(1.f / ( Sigma * Sigma )); // (1/Sigma_ij)^2
    params_map[ pair_i_j ].push_back(9.f * Epsilon);  // Prefactor for forces
    params_map[ pair_i_j ].push_back(Epsilon);
    

    if(verbose)
      std::cout << GetID_String() << ": " << "Sigma = " << Sigma << ", Epsilon = " << Epsilon << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f) {
    float invDist2  = 1.0f / ( dist2 * param[1] );        // (sigma/r)^2 
    float invDist = sqrtf(invDist2);
    float invDist4  = invDist2 * invDist2;     // (sigma/r)^4
    float invDist9 = invDist4 * invDist4 * invDist;       // (sigma/r)^9 
    float s = param[1] * param[2] * invDist9 * invDist2; // F_ij = s * r_ij
    (*my_f).w += param[3] * invDist9;                  
    return s;
  }
};



////////////////////////////////////////////////////////////////////////////////////////////////
// IPL potential, n=18
// v(r) = Epsilon ( (Sigma/r)^18 ) 
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_IPL_18 : public PairPotential{
 public: 
  Pot_IPL_18( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("potIPL18"); }

  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };  
  void CalcF_Local();  

  Potential* Copy() {
    Pot_IPL_18* new_pot = new Pot_IPL_18(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }

  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut * Sigma);            // Rcut_ij
    params_map[ pair_i_j ].push_back(1.f / ( Sigma * Sigma )); // (1/Sigma_ij)^2);
    params_map[ pair_i_j ].push_back(18.f * Epsilon); // Prefactor for forces);
    params_map[ pair_i_j ].push_back(Epsilon);

    if(verbose)
      std::cout << GetID_String() << ": " << "Sigma = " << Sigma << ", Epsilon = " << Epsilon << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f) {
    float invDist2  = 1.0f / ( dist2 * param[1] );        // (sigma/r)^2 
    float invDist6  = invDist2 * invDist2 * invDist2;     // (sigma/r)^6
    float invDist18 = invDist6 * invDist6 * invDist6;     // (sigma/r)^18    
    float s = param[1] * param[2] * invDist18 * invDist2; // F_ij = s * r_ij
    (*my_f).w += param[3] * invDist18;                   
    return s;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////
// IPL potential, abritrary exponent n
// v(r) =   Epsilon ( (Sigma/r)^n )
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_IPL_n : public PairPotential{
 public: 
  float n;   // exponent
  
  Pot_IPL_n( float set_n, CutoffMethod cutoff_method ) : PairPotential(cutoff_method), n(set_n) { std::stringstream ss; ss << "potIPL_" << n; SetID_String(ss.str()); }

  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };  
  void CalcF_Local();

  Potential* Copy() {
    Pot_IPL_n* new_pot = new Pot_IPL_n(n, cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }

  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut * Sigma);             // Rcut_ij
    params_map[ pair_i_j ].push_back(1.f / ( Sigma * Sigma )); // (1/Sigma_ij)^2
    params_map[ pair_i_j ].push_back(Epsilon);      // Prefactor for forces
    params_map[ pair_i_j ].push_back(n);


    if(verbose)
      std::cout << GetID_String() << ": " << "Sigma = " << Sigma << ", Epsilon = " << Epsilon << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }

  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f) {
    float invDist2   = 1.0f / ( dist2 * param[1] );                  // (sigma/r)^2 
    float invDist_n = expf(logf(invDist2)*param[3]*0.5f);            // (sigma/r)^n (faster)
    float s = param[1] * param[2] * param[3] * invDist_n * invDist2; // F_ij = s * r_ij
    (*my_f).w += param[2] * invDist_n;                              
    return s;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Integral of IPLs r^(-n) for n from 6 to inf
// v(r) = Epsilon ( (Sigma/r)^6 / ln(r/Sigma) )
////////////////////////////////////////////////////////////////////////////////

class Pot_IPL6_LN : public PairPotential{
 public: 
  Pot_IPL6_LN( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("potIPL6LN");}
  
  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };  
  void CalcF_Local();  
  
  Potential* Copy() {
    Pot_IPL6_LN* new_pot = new Pot_IPL6_LN(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams( unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut ){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut * Sigma);       // Rcut_ij);
    params_map[ pair_i_j ].push_back(Epsilon);            // Prefactor 
    params_map[ pair_i_j ].push_back(1.f / ( Sigma * Sigma )); // (1/Sigma_ij)^2
    params_map[ pair_i_j ].push_back(Epsilon / ( Sigma * Sigma )) ; // Prefactor for force

    if(verbose)
      std::cout << GetID_String() + ": " << "Sigma = " << Sigma << ", Epsilon = " << Epsilon << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f){
    float  lnDist  = 0.5f * log( dist2 * param[2] );  // ln(sigma/r).
    float invDist2 =    1.0f / ( dist2 * param[2] );  // (sigma/r)^2.
    float invDist6 = invDist2 * invDist2 * invDist2;  // (sigma/r)^6.
    
    float s = param[3] * invDist6 * invDist2 * ( 6.0f * lnDist + 1 ) / ( lnDist * lnDist );
    (*my_f).w += param[1] * invDist6 / lnDist;
    return s;
  }
};
////////////////////////////////////////////////////////////////////////////////
// 12-6 LJ / ln(r/R0) : LJ-like potential with hard sphere of radius R0
// v(r) = Epsilon ( a (Rm/r)^12 - b (Rm/r)^6 ) / ln(r/R0) )
// with a =   ln(Rm/R0) + 1/6
// and  b = 2 ln(Rm/R0) + 1/6
// potential minimum is at (Rm,-Epsilon)
////////////////////////////////////////////////////////////////////////////////

class Pot_LJ_LN : public PairPotential{
 public: 
  Pot_LJ_LN( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("pot_LJ_LN");}
  
  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };  
  void CalcF_Local();  
  
  Potential* Copy() {
    Pot_LJ_LN* new_pot = new Pot_LJ_LN(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams( unsigned int i, unsigned int j, float Rm, float R0, float Epsilon, float Rcut ){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut * Rm);          // Rcut_ij
    params_map[ pair_i_j ].push_back(1.f / ( Rm * Rm ) );   // (1/Rm_ij)^2
    params_map[ pair_i_j ].push_back(1.f / ( R0 * R0 ) ); //  1/(Rm_ij*R0_ij)^2
    params_map[ pair_i_j ].push_back(Epsilon * ( log(Rm/R0) + 1.f/6.f )); // Prefactor for potential (Epsilon*a)
    float next_param = params_map[ pair_i_j ][1] * params_map[ pair_i_j ][3];
    params_map[ pair_i_j ].push_back( next_param );
    params_map[ pair_i_j ].push_back(( 12.f*log(Rm/R0)+1 ) / ( 6.f*log(Rm/R0)+1 )); // factor for attractive term (b/a)


    if(verbose)
      std::cout << GetID_String() + ": " << ", Epsilon = " << Epsilon 
                << "Rm = " << Rm  << ", R0 = " << R0 << ", Rcut = " << Rcut << "." 
                << std::endl;
    CopyParamsToGPU(i,j);
  }
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f){
    float  lnDist   = 0.5f * log( dist2 * param[2] );  // ln(r/r0)
    float invlnDist = 1.0f / lnDist;                   // 1/ln(r/r0)
    float invDist2  = 1.0f /    ( dist2 * param[1] );  // (sigma/r)^2
    float invDist6  = invDist2 * invDist2 * invDist2;  // (sigma/r)^6
    
    float s = param[4] * invDist6 * invDist2 * invlnDist * invlnDist *
              ( invDist6 * ( 12.f*lnDist + 1.f ) - param[5] * ( 6.f*lnDist + 1.f ) );
    (*my_f).w += param[3] * invDist6 * ( invDist6 - param[5] ) * invlnDist;
    return s;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Girifalco potential (pair interaction for buckyballs)
// v(r) = - alpha (  s(s-1)^-3 + s(s+1)^-3 - 2s^-4  )
//        + beta  (  s(s-1)^-9 + s(s+1)^-9 - 2s^-10  )
// where s = r/R0 and R0 = buckyball diameter
// This potential is defined in terms of R0 instead of Sigma or Rm!
////////////////////////////////////////////////////////////////////////////////

class Pot_Girifalco : public PairPotential{
 public: 
  Pot_Girifalco( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("pot_Girifalco");}
  
  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };  
  void CalcF_Local();  
  
  Potential* Copy() {
    Pot_Girifalco* new_pot = new Pot_Girifalco(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams( unsigned int i, unsigned int j, float alpha, float beta, float R0, float Rcut ){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut * R0);     // Rcut in terms of R0    
    params_map[ pair_i_j ].push_back(1.f / (R0*R0) ); // prefactor for forces
    params_map[ pair_i_j ].push_back(alpha);         // Prefactor for attractive terms (alpha)
    params_map[ pair_i_j ].push_back(beta);   // Prefactor for repulsive terms (beta)

    if(verbose)
      std::cout << GetID_String() + ": " << "alpha " << alpha << ", beta = " << beta 
		<< ", R0 = " << R0 << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f){
    float r    = sqrtf( dist2*param[1] );                         // r/R0 (=s)
    float m4   = 1.f / ( (r-1.f) * (r-1.f) * (r-1.f) * (r-1.f) ); // (s-1)^-4
    float p4   = 1.f / ( (r+1.f) * (r+1.f) * (r+1.f) * (r+1.f) ); // (s+1)^-4
    float m3   = m4 * (r-1.f);                                    // (s-1)^-3
    float p3   = p4 * (r+1.f);                                    // (s+1)^-3
    float invr = 1.f / r;                                         // (r/R0)^-1
    float r3   = invr*invr*invr;                                  // (r/R0)^-3

    (*my_f).w += invr*( - param[2] * (    m3    +    p3    - 2.f*r3       )  
			+ param[3] * ( m3*m3*m3 + p3*p3*p3 - 2.f*r3*r3*r3 ) );

    float s = r3*param[1]*
                ( -param[2] * ( ( 4.f*r-1.f)*m4       + ( 4.f*r+1.f)*p4       -  8.f*r3       ) 
		  +param[3] * ( (10.f*r-1.f)*m4*m3*m3 + (10.f*r+1.f)*p4*p3*p3 - 20.f*r3*r3*r3 ) );
    return s;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Yukawa potential.
// v(r) = epsilon * exp[-r/sigma] * sigma / r
////////////////////////////////////////////////////////////////////////////////

class Pot_Yukawa : public PairPotential{
 public: 
  Pot_Yukawa( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) {
    SetID_String("potYukawa");
  }
  
  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) {
    return ComputeInteraction(dist2, param, my_f);
  };  
  void CalcF_Local();

  Potential* Copy() {
    Pot_Yukawa* new_pot = new Pot_Yukawa(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams(unsigned int i, unsigned int j,
		 float sigma, float epsilon, float Rcut){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut*sigma);
    params_map[ pair_i_j ].push_back(sigma);
    params_map[ pair_i_j ].push_back(1.f / sigma);
    params_map[ pair_i_j ].push_back(epsilon);


    if(verbose)
      std::cout << GetID_String() << ": "
		<< "Sigma = " << sigma << ", Epsilon = " << epsilon 
		<< ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }
  
  __host__ __device__ float ComputeInteraction(float dist2,
					       const float* param, float4* my_f){
    float dist      = sqrtf( dist2 );
    float invDist   = 1.f / dist;
    float potential = param[3] * param[1] * invDist * exp( -param[2] * dist );
    (*my_f).w +=  potential;
    float s = invDist * ( invDist + param[2] ) * potential;
    return s;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////
// Dzugutov potential.
// v(r) = V1 + V2 = [ A ( r^(-n) - B ) * exp(c/(r-a)) ] + [ B * exp(d/(r-b) ], V1 = 0 for r >= a 
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_Dzugutov : public PairPotential{
 public: 
 Pot_Dzugutov() : PairPotential(NS) { SetID_String("potDzu"); }

  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };
  void CalcF_Local();

  Potential* Copy() {
    Pot_Dzugutov* new_pot = new Pot_Dzugutov();
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams(unsigned int i, unsigned int j, float a, float b, float c, float d, float A, float B){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back((b - 0.000001f)); // To avoid division by zero.
    params_map[ pair_i_j ].push_back(a);
    params_map[ pair_i_j ].push_back(b);
    params_map[ pair_i_j ].push_back(c);
    params_map[ pair_i_j ].push_back(d);
    params_map[ pair_i_j ].push_back(A);
    params_map[ pair_i_j ].push_back(B);

    
    if(verbose)
      std::cout << GetID_String() << ": " << "a = " << a << ", b = " << b << ", c = " << c << ", d = " << d << ", A = " << A 
		<< ", B = " << B << "." << std::endl;
    CopyParamsToGPU(i,j);
  }
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f){
    float r = sqrtf(dist2); 
    
    float inv_r = 1.f / r;
    float inv_b = 1.f / ( r - param[2] );
    float exp_b = expf( param[4] * inv_b );
    float inv_b_sq_exp = inv_b * inv_b * exp_b;
    
    float s = inv_r * param[4] * param[6] * inv_b_sq_exp; 
    (*my_f).w += param[6] * exp_b; 
    
    if( r < param[1] ){
      float invDist2 = 1.f / dist2; 
      float invDist4 = invDist2 * invDist2;
      float invDist8 = invDist4 * invDist4;
      
      float inv_a = 1.f / ( r - param[1] );
      float exp_a = expf( param[3] * inv_a );
      
      float potential_energy = param[5] * exp_a * ( invDist8 * invDist8 - param[6] );
      s += 16 * param[5] * (exp_a * invDist8 * invDist8) * invDist2 + inv_r * param[3] * inv_a * inv_a * potential_energy;
      (*my_f).w += potential_energy;
    }
    return s;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////
// Fermi-Jagla potential [J. Phys. Chem. B 115, 14229 (2011)].
// v(r) = epsilon_0 [ (a/r)^n + A0 / [1+exp[A1/A0(r/a - A2)]] - B0 / [1+exp[B1/B0(r/a - B2)]]]
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_Fermi_Jagla : public PairPotential{
 public: 
  Pot_Fermi_Jagla( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("potFermiJagla"); }
  
  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };
  void CalcF_Local();
  
  Potential* Copy() {
    Pot_Fermi_Jagla* new_pot = new Pot_Fermi_Jagla(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams(unsigned int i, unsigned int j, float a, float epsilon0, float A0, float A1, float A2, float B0, float B1, float B2, float n, float Rcut){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut * a);
    params_map[ pair_i_j ].push_back(1.f / ( a * a ));
    params_map[ pair_i_j ].push_back(A0);
    params_map[ pair_i_j ].push_back(A1 / A0);
    params_map[ pair_i_j ].push_back(A2);
    params_map[ pair_i_j ].push_back(B0);
    params_map[ pair_i_j ].push_back(B1 / B0);
    params_map[ pair_i_j ].push_back(B2);
    params_map[ pair_i_j ].push_back(epsilon0);
    params_map[ pair_i_j ].push_back(n);
    
    if(verbose)
      std::cout << GetID_String() << ": " << "a = " << a << ", epsilon0 = " << epsilon0 << ", A0 = " << A0 << ", A1 = " << A1 << ", A2 = " << A2 
		<< ", B0 = " << B0 << ", B1 = " << B1 << ", B2 = " << B2 << ", n = " << n << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f){
    float Dist = sqrtf( dist2 * param[1] );                          // (r/a)
    float invDist2 = 1.0f / ( dist2 * param[1] );                    // (a/r)^2 
    float invDist_n = expf(logf(invDist2)*param[9]*0.5f);            // (a/r)^n 
    
    float A_factor = 1 + expf(param[3]*(Dist - param[4]));
    float B_factor = 1 + expf(param[6]*(Dist - param[7]));
    
    float inv_A_factor = 1.f / A_factor;
    float inv_B_factor = 1.f / B_factor;
    
    float s = param[8] * ( param[9] * param[1] * invDist2 * invDist_n +
			   param[1] * ( ( param[2] * param[3] * inv_A_factor * inv_A_factor ) * (A_factor - 1) - 
					( param[5] * param[6] * inv_B_factor * inv_B_factor ) * (B_factor - 1) ) / Dist );
    
    (*my_f).w = param[8] * ( invDist_n + param[2] * inv_A_factor - param[5] * inv_B_factor );
    return s;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////
// Repulsive Shoulder
// v(r) = (Sigma/r)^(14) + 0.5 Epsilon ( 1 - tanh( k0 ( r - Sigma1 ) ) )
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_Repulsive_Shoulder : public PairPotential{
 public: 
  Pot_Repulsive_Shoulder( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("potRepulsiveShoulder"); }
  
  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };
  void CalcF_Local();
  
  Potential* Copy() {
    Pot_Repulsive_Shoulder* new_pot = new Pot_Repulsive_Shoulder(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float k0, float Sigma1, float Rcut){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut * Sigma);
    params_map[ pair_i_j ].push_back(1.f / ( Sigma * Sigma ) );
    params_map[ pair_i_j ].push_back(0.5 * Epsilon);
    params_map[ pair_i_j ].push_back(k0);
    params_map[ pair_i_j ].push_back(Sigma1);

    if(verbose)
      std::cout << GetID_String() << ": " << "Sigma = " << Sigma << ", Epsilon = " << Epsilon << ", k0 = " << k0 << ", Sigma1 = " << Sigma1 << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f){
    float Dist = sqrtf( dist2 );                          
    float invDist2 = 1.0f / ( dist2 * param[1] );                    
    float invDist8 = invDist2 * invDist2 * invDist2 * invDist2;
    float invDist16 = invDist8 * invDist8;
    
    float factor = param[3] * ( Dist - param[4] );
    float tanH = tanhf(factor);

    float s = 14.f * param[1] * invDist16 + param[2] * param[3] * ( 1 - tanH * tanH ) / Dist;
    
    (*my_f).w = invDist8 * invDist2 * invDist2 * invDist2 + param[2] * ( 1 - tanH );
    return s;
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////
// Exponential potential.
// v(r) = Epsilon * exp[-(r/Sigma)]
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_Exp : public PairPotential{
 public:
  Pot_Exp( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("potExp"); }

  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };
  void LaunchPE_Kernel();
  void CalcF_Local();

  Potential* Copy() {
    Pot_Exp* new_pot = new Pot_Exp(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }

  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float Rcut){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut * Sigma);
    params_map[ pair_i_j ].push_back(1.f / ( Sigma ));
    params_map[ pair_i_j ].push_back(Epsilon);
    
    if(verbose)
      std::cout << GetID_String() << ": " << "Sigma = " << Sigma << ", Epsilon = " << Epsilon << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }

  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f){
    float Dist = sqrtf(dist2) * param[1];                      // (r/Sigma)
    float invDist  = param[1]/Dist;                           //  1/r
    float s = param[1] * param[2] * invDist * exp(-Dist);     // F_ij = s * r_ij
    (*my_f).w += param[2] * exp(- Dist);                      // U_ij 
    return s;
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////
// The SAAP for nobel elements, see:
// J. Chem. Phys. 150, 134504 (2019); https://doi.org/10.1063/1.5085420
////////////////////////////////////////////////////////////////////////////////////////////////
class Pot_SAAP : public PairPotential{
 public: 
  Pot_SAAP( CutoffMethod cutoff_method ) : PairPotential(cutoff_method) { SetID_String("SAAP"); }
  
  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) { return ComputeInteraction(dist2, param, my_f); };  
  void CalcF_Local();
  
  Potential* Copy() {
    Pot_SAAP* new_pot = new Pot_SAAP(cutoffMethod);
    new_pot->SetAllParams(params_map);
    return new_pot;
  }
  
  void SetParams(unsigned int i, unsigned int j, float Sigma, float Epsilon, float a0, float a1, float a2, float a3, float a4, float a5, float a6, float Rcut) {
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut * Sigma);             // Rcut_ij
    params_map[ pair_i_j ].push_back(a0); // a0
    params_map[ pair_i_j ].push_back(a1); // a1
    params_map[ pair_i_j ].push_back(a2); // a2
    params_map[ pair_i_j ].push_back(a3); // a3
    params_map[ pair_i_j ].push_back(a4); // a4
    params_map[ pair_i_j ].push_back(a5); // a5    
    params_map[ pair_i_j ].push_back(a6); // a6
    params_map[ pair_i_j ].push_back(1.f / (Sigma * Sigma));	// Sigma
    params_map[ pair_i_j ].push_back(Epsilon);			// Epsilon

    if(verbose)
      std::cout << GetID_String() << ": " << "Sigma = " << Sigma << ", Epsilon = " << Epsilon << ", a0 = " << a0 << ", a1 = " << a1 << ", a2 = " << a2 << ", a3 = " << a3 << ", a4 = " << a4 << ", a5 = " << a5  << ", a6 = " << a6 << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }
  
  __host__ __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f) {
    
    float dist = sqrtf (dist2);    
    float Dist2 = dist2 * param[8];
    float Dist = sqrtf ( Dist2 );
    float Dist6 = Dist2 * Dist2 * Dist2;

    float a0 = param[1];
    float a1 = param[2];
    float a2 = param[3];
    float a3 = param[4];
    float a4 = param[5];
    float a5 = param[6];
    float a6 = param[7];
    float i_sigma2 = param[8];
    float Epsilon = param[9];

    float pot_eq1 = ( a0 / Dist ) * exp ( a1 * Dist + a6 * Dist2);
    float pot_eq2 = a2 * exp( a3 * Dist );
    float pot_eq3 = 1.f / ( 1 + a5 * Dist6 );

    float s_eq1 = a0 * exp (a6 * Dist2 + a1 * Dist);

    float s_11 = a2  * a3 * exp (a3 * Dist);
    float s_12 = s_eq1 * (1.f / Dist2);
    float s_13 = s_eq1 * (a1 + 2 * a6 * Dist) * (1.f / Dist);
    float s_21 = 6 * a5 * Dist2 * Dist2 * Dist * (a4 + a2 * exp(a3 * Dist) + (s_eq1 / Dist));
    
    float s = Epsilon * i_sigma2 * (1.f / dist) * ((s_21 * pot_eq3 * pot_eq3) - (s_11 - s_12 + s_13) * pot_eq3); 
    (*my_f).w += Epsilon * (pot_eq1 + pot_eq2 + a4) * pot_eq3;
    return s;

    /*
    float s_11 = a2 * a3 * exp( a3 * Dist);
    float s_12 = (a0 * a1 * exp( a1 * Dist )) / Dist;
    float s_13 = -(a0 * exp( a1 * Dist)) / Dist2;
    float s_21 = 6 * a5 * Dist2 * Dist2 * Dist2;
    float s_22 = (a2 * exp ( a3 * Dist )) + (a0 * exp( a1 * Dist) / Dist) + a4;
    float s = Epsilon * i_sigma2 * (-1.f / dist ) *( (s_11 + s_12 - s_13) * pot_eq3 - (s_21 * s_22 * pot_eq3 * pot_eq3) );
    */

  }
};


////////////////////////////////////////////////////////////////////////////////////////////////
// Coulomb potential truncated using shifted-force method.
// v(r) = Epsilon/r
////////////////////////////////////////////////////////////////////////////////////////////////

class Pot_ShiftedForceCoulomb : public PairPotential{
 public:
  Pot_ShiftedForceCoulomb( ) : PairPotential(NS) { SetID_String("potCoul"); }

  void CalcF(bool initialize, bool calc_stresses);
  float ComputeInteraction_CPU(float dist2, const float* param, float4* my_f) {
    float inv_dist = 1./sqrtf(dist2);
    float r = 1./inv_dist;
    float s = ( param[1] / dist2 - param[3] ) * inv_dist;
    (*my_f).w += param[1] * inv_dist + param[3] * ( r - param[0] ) - param[2]; 

    return s;
  }
  void LaunchPE_Kernel();
  void CalcF_Local();

  Potential* Copy() {
    Pot_ShiftedForceCoulomb* new_pot = new Pot_ShiftedForceCoulomb();
    new_pot->SetAllParams(params_map);
    return new_pot;
  }

  void SetParams(unsigned int i, unsigned int j, float Epsilon, float Rcut){
    std::pair<unsigned, unsigned> pair_i_j = std::make_pair(i, j);
    params_map[ pair_i_j ] = std::vector<float>();
    params_map[ pair_i_j ].push_back(Rcut);
    params_map[ pair_i_j ].push_back(Epsilon);
    params_map[ pair_i_j ].push_back(Epsilon/Rcut);
    params_map[ pair_i_j ].push_back(Epsilon/(Rcut*Rcut));
    
    if(verbose)
      std::cout << GetID_String() << ": Epsilon = " << Epsilon << ", Rcut = " << Rcut << "." << std::endl;
    CopyParamsToGPU(i,j);
  }

  __device__ float ComputeInteraction(float dist2, const float* param, float4* my_f){
    float inv_dist2 = 1.f/dist2;
    float r = sqrtf(dist2);
    float inv_dist = 1.f/r;
    float s = ( param[1] * inv_dist2 - param[3] ) * inv_dist;
    (*my_f).w += param[1] * inv_dist + param[3] * ( r - param[0] ) - param[2];
    return s;
    }
};





/////////////////////////////////////////////////////////////////////////////
/// Device function for calculating a pair interaction for a given 
/// interparticle distance and pair potential (the latter specified as a 
/// template argument)
/////////////////////////////////////////////////////////////////////////////

template<int STR, int CUTOFF, class P, class S> 
__device__ __host__ float fij( P* Pot, float4 my_r, float4 rj, float4* my_f,
			       float4* my_w, float4* my_sts,  const float* param,
			       S* simBox, float* simBoxPtr ){
  
  float4 dist = simBox->calculateDistance(my_r, rj, simBoxPtr);
  
  // Inside cut-off for interaction? (param[0]=Rcut)
  if ( dist.w < (param[0]*param[0]) ) {
    float s = Pot->ComputeInteraction(dist.w, param, my_f); float r;
    
    // Which cut-off method?
    switch( CUTOFF ){
    case PairPotential::NS:
      break;
    case PairPotential::SP:
      (*my_f).w -= param[NumParam-1]; 
      break;
    case PairPotential::SF:
      r = sqrtf(dist.w); 
      s -= param[NumParam-2] / r; 
      (*my_f).w += param[NumParam-2] * ( r - param[0] ) - param[NumParam-1]; 
      break;
    default:
      break;
    }
    // F_ij.
    (*my_f).x += dist.x * s;
    (*my_f).y += dist.y * s;
    (*my_f).z += dist.z * s;
    
    // W_ij.
    (*my_w).w += dist.w * s;
    
    if(STR){
      // stress - diagonal components
      (*my_sts).x -= dist.x * dist.x * s;  // xx
      (*my_sts).y -= dist.y * dist.y * s;  // yy
      (*my_sts).z -= dist.z * dist.z * s;  // zz
      // stress - off-diagonal components
      (*my_sts).w -=  dist.y * dist.z * s; // yz
      (*my_w).y   -=  dist.x * dist.z * s; // xz
      (*my_w).z   -=  dist.x * dist.y * s; // xy      
    }
  } 
  return dist.w;                                                 
}


#endif // PAIRPOTENTIAL_H
