#ifndef MOLECULEDATA_H
#define MOLECULEDATA_H


#include <cuda.h>
#include <map>
#include <vector>
#include <string>

class Sample; class ParticleData; class PairPotential;

class MoleculeData {
private:
  Sample* S;

  unsigned num_mol;         // Number of molecules 
  unsigned max_num_uau;     // Maximum nuau per molecule

  unsigned max_num_bonds;   // Only used by constraints because this uses the old stuff.
  unsigned num_bonds;       // Maximum bonds per particle
  unsigned num_btypes;      // Number of bonding types 

  unsigned num_angles;      // Number of angles
  unsigned num_atypes;      // Number of angle types
  
  unsigned num_dihedrals;   // Number of dihedrals
  unsigned num_dtypes;      // Number of dihedral types

  std::map<std::string, unsigned int> allocatedValue;


  void AllocateMolecules();
  void AllocateBonds();
  void AllocateAngles();
  void AllocateDihedrals();

  void FreeMolecules();
  void FreeBonds();
  void FreeAngles();
  void FreeDihedrals();


 public:
  MoleculeData(Sample *set_sample, const std::string& top_filename);
  MoleculeData(const MoleculeData& M);
  ~MoleculeData();
  MoleculeData* Copy();
  MoleculeData& operator=(const MoleculeData& M);


  // Bond Lists/arrays
  std::vector<uint2> h_blist;  // Bonding list
                               // x, y: indices of bonded particles 
  uint2 *d_blist;
  std::vector<uint1> h_btlist; // Bond type list
  uint1  *d_btlist;
  float2 *h_bplist, *d_bplist; // Bond type parameter list (r0, k)
  float  *h_belist, *d_belist; // Potential energy for each bond
  int1   *h_btlist_int,        // Internal mapping from user specified type 
         *d_btlist_int;        //   to internal type

  // Angle lists/arrays
  std::vector<uint4> h_alist;  // Angle list:
  uint4  *d_alist;             // x, y, z: indices of bonded particles 
                               // w: Angle type
  float2 *h_aplist, *d_aplist; // Angle type parameter list 
                               // x: force constant
                               // y: zero force angle
  float  *h_epot_angle,        // Holds the potential energy for
         *d_epot_angle;        //   each angle

  // Dihedral lists/arrays
  std::vector<uint4> h_dlist; // Dihedral list:
  uint4 *d_dlist;                // x, y, z, w: indices of bonded particles
  std::vector<uint1> h_dtype; // Dihedral type list
  uint1 *d_dtype;
  float *h_dplist, *d_dplist; // Parameter lists            
  float *h_epot_dihedral,     // Holds the potential energy for
        *d_epot_dihedral;     //   each dihedral

  // Molecule lists (Convenient for data analysis)
  int1 *h_mlist;            
  int1 *d_mlist;            
  unsigned lmlist;          
  
  // Data arrays
  float *h_bonds, *d_bonds;              // The bonds 
  float *h_angles, *d_angles;            // The angles
  float *h_dihedrals, *d_dihedrals;      // The dihedrals

  float4 *h_cm, *d_cm;                   // Centre of mass + cm.w mass 
  float4 *h_cm_im, *d_cm_im;             // Center of mass images
  float3 *h_vcm, *d_vcm;                 // Velocity centre of mass
  float3 *h_s, *d_s;                     // (Intrinsic) angular momentum
  float3 *h_omega, *d_omega;             // (Intrinsic) angular velocity
  float *h_inertia, *d_inertia;          // Moment of intertia (length=9*num_mols)
                                         // Is symmetric, but keep all element due to 
                                         // calculation of ang. velocity - row-wise matrix

  float *h_stress, *d_stress;            // Molecular stress tensor per molecule
  float symmetricStress[6];              // Molecule symmetric stress for the whole box

  // GPU specifics
  dim3 threads_per_block;  
  //dim3 num_blocks;

  // Helper functions 
  void AddmlistEntry(unsigned *, unsigned, unsigned, unsigned);
  void ReadTopology(const std::string& top_filename);
  void UpdateAfterSorting( unsigned int *old_index, unsigned int* new_index );
  

  // Set methods.
  void SetBondParams(unsigned bond_type, float length_param, float stiffness_param, unsigned bond_class);
  void SetAngleParams(unsigned angle_type, float theta0, float ktheta);
  void SetDihedralParams(unsigned dihedral_type, float prm0, float prm1, float prm2, float prm3, float prm4, float prm5);

  void SetExclusionBond(PairPotential *potential, unsigned etype);
  void SetExclusionMolecule(PairPotential* potential);  
  void SetExclusionMolecule(PairPotential* potential, unsigned molindex);
  void SetExclusionAngle(PairPotential* potential);
  void SetExclusionDihedral(PairPotential* potential);

  void SetAngleConstant(unsigned type, float ktheta);
  void SetDihedralRyckaertConstants(unsigned type, float c0, float c1, float c2, 
				    float c3, float c4, float c5);
  void SetPeriodicDihedralConstants(unsigned type, float phiEq, float v1, float v2, float v3);
  void SetRigidBodyVelocities();
  
  // Get methods.
  unsigned GetNumberOfMolecules() const { return num_mol; };
  unsigned GetMaximumNumberOfBonds() const { return max_num_bonds; }
  unsigned GetNumberOfBonds() const { return num_bonds; }
  unsigned GetNumberOfAngles() const { return num_angles; }
  unsigned GetNumberOfDihedrals() const { return num_dihedrals; }
  const int1*  GetMoleculeListDevice() const { return d_mlist; }
  unsigned GetMaximumMoleculeSize() const { return max_num_uau; }

  void GetBondsFromTopology(const std::string& top_filename);
  void GetAnglesFromTopology(const std::string& top_filename);
  void GetDihedralsFromTopology(const std::string& top_filename);
  
  void GetBonds();
  void GetAngles();
  void GetDihedrals();
  
  // Write or output methods
  void WriteMolConf(const std::string& fstr);
  void WriteMolxyz(const std::string& fstr, unsigned mol_id, const std::string& mode);
  void WriteMolxyz_filtered(const std::string& fstr, unsigned filter_uau, const std::string& mode);
  
  // Functions for changing boxsize
  void IsotropicScaleCM( double scaleFactor );

  // Run time data evaluations
  void EvalMolMasses();
  void EvalCM(bool copy_to_host=true); 
  void EvalVelCM(bool copy_to_host=true);
  void EvalSpin();

  // Debugging.
  void PrintLists();
  void PrintLists(unsigned i);
  void PrintMlist();

};

// Kernels for updating after sorting
__global__ void UpdateMlistAfterSorting( unsigned int *new_index,
					 unsigned int nParticles,
					 int1 *mlist_offset,
					 unsigned length_mlist_offset );

__global__ void updateBondsAfterSorting(  unsigned int *new_index, 
					  unsigned int nParticles, 
					  uint2 *blist, 
					  unsigned num_bonds );

__global__ void updateAnglesAfterSorting( unsigned int *new_index, 
					  unsigned int nParticles, 
					  uint4 *alist, 
					  unsigned num_angles );

__global__ void updateDihedralsAfterSorting( unsigned int *new_index,
					     unsigned int nParticles,
					     uint4 *dlist,
					     unsigned num_dihedrals);

// Should of course be moved to a generic place
void GaussPivotBackCPU(float *x, float *b, float *A, unsigned nrc);
double Determinant3(float *A);


template <class Simbox>
__global__ void get_positions_relative_to_CM_kernel(unsigned int num_mol,
						    unsigned int max_num_uau,
						    float4 *r_intra,
						    int1   *mlist,
						    float4 *rcm,
						    float4 *imcm,
						    float4 *atom_positions,
						    float4 *atom_images,
						    Simbox *simbox,
						    float  *simBoxPointer);

template <class Simbox>
__global__ void scale_CM_kernel(unsigned int num_mol,
				unsigned int max_num_uau,
				float  factor,
				float4 *r_intra,
				int1   *mlist,
				float4 *rcm,
				float4 *imcm,
				float4 *atom_positions,
				float4 *atom_images,
				Simbox *simbox,
				float  *simBoxPointer);

template <class Simbox>
__global__ void eval_CM_kernel(unsigned int num_mol, 
			       unsigned int max_num_uau,
			       int1* mlist,
			       float4 *atom_positions,
			       float4 *atom_velocities,
			       float4 *atom_images,
			       float4 *rcm,
			       float4 *imcm,
			       Simbox *simbox, 
			       float *simBoxPointer);

template <class Simbox>
__global__ void eval_vel_CM_kernel(unsigned int num_mol, 
				   unsigned int max_num_uau,
				   int1* mlist,
				   float4 *atom_velocities,
				   float3 *vcm, 
				   Simbox *simbox, 
				   float *simBoxPointer);


template <class Simbox>
__global__ void setrigidbodyvelocities_kernel(unsigned int num_mol, 
					      unsigned int max_num_uau,
					      int1* mlist,
					      float4 *atom_positions,
					      float4 *atom_velocities,
					      float4 *rcm,
					      float3 *vcm,
					      float3 *omega,
					      Simbox *simbox, 
					      float *simBoxPointer);

#endif // MOLECULEDATA_H
