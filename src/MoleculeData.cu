#include "rumd/MoleculeData.h"

#include <iostream>
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include "rumd/rumd_algorithms_CPU.h"

#include "rumd/RUMD_Error.h"
#include "rumd/Sample.h"
#include "rumd/SimulationBox.h"
#include "rumd/PairPotential.h"

///////////////////////////////////////////////////////
// Constructor / Destructor
///////////////////////////////////////////////////////

MoleculeData::MoleculeData(Sample *set_sample, const std::string& top_filename) : 
  num_mol(0),
  max_num_uau(0),
  max_num_bonds(0),
  num_bonds(0),
  num_angles(0),
  num_dihedrals(0),
  num_btypes(0),
  num_atypes(0),
  num_dtypes(0) {

  // for keeping track of allocation.
  allocatedValue["num_mol"] = 0;
  allocatedValue["max_num_uau"] = 0;
  allocatedValue["num_bonds"] = 0;
  allocatedValue["num_btypes"] = 0;
  allocatedValue["num_angles"] = 0;
  allocatedValue["num_atypes"] = 0;
  allocatedValue["num_dihedrals"] = 0;
  allocatedValue["num_dtypes"] = 0;
  
  // The Sample class
  S = set_sample;
  
  // Get number of molecules and max nuau from top file
  // and allocate arrays that depend on num_mol
  ReadTopology(top_filename);
  // Read data and allocate memory for bonds, angles, dihedrals
  GetBondsFromTopology(top_filename);     // Memory is allocated here
  GetAnglesFromTopology(top_filename);    // Memory is allocated here
  GetDihedralsFromTopology(top_filename); // Memory is allocated here

  // Calculate the molecular masses (not sure necessary any more NB)
  EvalMolMasses();

  //Allocation
  AllocateMolecules();
  AllocateBonds();
  AllocateAngles();
  AllocateDihedrals();
  
  // Print warning message about the off-diagonal components in the stress tensor
  std::cout << "[Info]: RUMD does not support (atomic) stress tensor calculations of molecular systems involving constraints or 3- and 4- body forces [contributions from bond-forces are correctly computed, likewise the molecular stress]" << std::endl; 
}

MoleculeData::~MoleculeData(){

  if (allocatedValue["num_mol"] > 0 )
    FreeMolecules();
  
  if (allocatedValue["num_bonds"] > 0 )
    FreeBonds();
    
  if (allocatedValue["num_angles"] > 0 )
    FreeAngles();

  if ( allocatedValue["num_dihedrals"] > 0 )
    FreeDihedrals();

}



MoleculeData* MoleculeData::Copy() {
  MoleculeData* new_moleculeData = new MoleculeData(*this);
  return new_moleculeData;
}

MoleculeData::MoleculeData(const MoleculeData& M) {
  // We copy the host data then copy that to this object's device arrays,
  // so that this object starts in a "pristine" state with host and device
  // data in the same order

  allocatedValue["num_mol"] = 0;
  allocatedValue["max_num_uau"] = 0;
  allocatedValue["num_bonds"] = 0;
  allocatedValue["num_btypes"] = 0;
  allocatedValue["num_angles"] = 0;
  allocatedValue["num_atypes"] = 0;
  allocatedValue["num_dihedrals"] = 0;
  allocatedValue["num_dtypes"] = 0;

  
  num_mol       = M.num_mol;
  max_num_uau   = M.max_num_uau;
  max_num_bonds = M.max_num_bonds;
  num_bonds     = M.num_bonds;
  num_angles    = M.num_angles;
  num_dihedrals = M.num_dihedrals;
  num_btypes    = M.num_btypes;
  num_atypes    = M.num_atypes;
  num_dtypes    = M.num_dtypes;

  // this is not associated with a sample
  S = 0;
  
  // next, all the allocation
  AllocateMolecules();
  AllocateBonds();
  AllocateAngles();
  AllocateDihedrals();

  // Copy the arrays needed to run the simulation

  // molecule list
  cudaMemcpy(h_mlist, M.h_mlist, sizeof(int1)*lmlist, cudaMemcpyHostToHost);
  cudaMemcpy(d_mlist, h_mlist, sizeof(int1)*lmlist, cudaMemcpyHostToDevice);
  // bond list
  h_blist = M.h_blist; // vector (operator=)
  cudaMemcpy(d_blist, &(h_blist[0]), sizeof(uint2)*num_bonds, cudaMemcpyHostToDevice);
  // bond-type list
  h_btlist = M.h_btlist; // vector
  cudaMemcpy(d_btlist, &(h_btlist[0]), sizeof(uint1)*num_bonds, cudaMemcpyHostToDevice);

  cudaMemcpy(h_btlist_int, M.h_btlist_int, sizeof(int1)*num_btypes, cudaMemcpyHostToHost);
  cudaMemcpy(d_btlist_int, h_btlist_int, sizeof(int1)*num_btypes, cudaMemcpyHostToDevice);
  // Bond parameters
  cudaMemcpy( h_bplist, M.h_bplist, num_btypes * sizeof(float2), cudaMemcpyHostToHost );
  cudaMemcpy( d_bplist, h_bplist, num_btypes * sizeof(float2), cudaMemcpyHostToDevice );

  // Angles and types
  h_alist = M.h_alist; // vector
  cudaMemcpy(d_alist, &(h_alist[0]), sizeof(uint4)*num_angles, cudaMemcpyHostToDevice);
  // Angle parameters
  cudaMemcpy(h_aplist, M.h_aplist, sizeof(float2)*num_atypes, cudaMemcpyHostToHost);
  cudaMemcpy(d_aplist, h_aplist, sizeof(float2)*num_atypes, cudaMemcpyHostToDevice);

  // Dihedral list
  h_dlist = M.h_dlist; // vector
  cudaMemcpy(d_dlist, &(h_dlist[0]), sizeof(uint4)*num_dihedrals, cudaMemcpyHostToDevice);
  // Dihedral types
  h_dtype = M.h_dtype;
  cudaMemcpy(d_dtype, &(h_dtype[0]), sizeof(uint1)*num_dihedrals, cudaMemcpyHostToDevice);
  // Dihedral parameters
  cudaMemcpy(h_dplist, M.h_dplist, sizeof(float)*6*num_dtypes, cudaMemcpyHostToHost);
  cudaMemcpy(d_dplist, h_dplist, sizeof(float)*6*num_dtypes, cudaMemcpyHostToDevice);

}

MoleculeData& MoleculeData::operator=(const MoleculeData& M) {
  if(this != &M){ 

    num_mol       = M.num_mol;
    max_num_uau   = M.max_num_uau;
    max_num_bonds = M.max_num_bonds;
    num_bonds     = M.num_bonds;
    num_angles    = M.num_angles;
    num_dihedrals = M.num_dihedrals;
    num_btypes    = M.num_btypes;
    num_atypes    = M.num_atypes;
    num_dtypes    = M.num_dtypes;

    AllocateMolecules();
    AllocateBonds();
    AllocateAngles();
    AllocateDihedrals();

    // molecule list
    cudaMemcpy(h_mlist, M.h_mlist, sizeof(int1)*lmlist, cudaMemcpyHostToHost);
    cudaMemcpy(d_mlist, M.d_mlist, sizeof(int1)*lmlist, cudaMemcpyDeviceToDevice);
    // bond list
    h_blist = M.h_blist; // vector operator=
    cudaMemcpy(d_blist, M.d_blist, sizeof(uint2)*num_bonds, cudaMemcpyDeviceToDevice);
    // bond-type list
    h_btlist = M.h_btlist; // vector operator=
    cudaMemcpy(d_btlist, M.d_btlist, sizeof(uint1)*num_bonds, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_btlist_int, M.d_btlist_int, sizeof(int1)*num_btypes, cudaMemcpyDeviceToDevice);
    // Bond parameters
    cudaMemcpy( h_bplist, M.h_bplist, num_btypes * sizeof(float2), cudaMemcpyHostToHost );
    cudaMemcpy( d_bplist, M.d_bplist, num_btypes * sizeof(float2), cudaMemcpyDeviceToDevice );
    
    // Angles and types
    h_alist = M.h_alist; // vector operator=
    cudaMemcpy(d_alist, M.d_alist, sizeof(uint4)*num_angles, cudaMemcpyDeviceToDevice);
    // Angle parameters
    cudaMemcpy(h_aplist, M.h_aplist, sizeof(float2)*num_atypes, cudaMemcpyHostToHost);
    cudaMemcpy(d_aplist, M.d_aplist, sizeof(float2)*num_atypes, cudaMemcpyDeviceToDevice);
    
    // Dihedral list
    h_dlist = M.h_dlist; // vector
    cudaMemcpy(d_dlist, M.d_dlist, sizeof(uint4)*num_dihedrals, cudaMemcpyDeviceToDevice);
    // Dihedral types
    h_dtype = M.h_dtype;
    cudaMemcpy(d_dtype, M.d_dtype, sizeof(uint1)*num_dihedrals, cudaMemcpyDeviceToDevice);
    // Dihedral parameters
    cudaMemcpy(h_dplist, M.h_dplist, sizeof(float)*6*num_dtypes, cudaMemcpyHostToHost);
    cudaMemcpy(d_dplist, M.d_dplist, sizeof(float)*6*num_dtypes, cudaMemcpyDeviceToDevice);
  }
  return *this;
}


///////////////////////////////////////////////////////
// Allocation and deallocation
///////////////////////////////////////////////////////

void MoleculeData::AllocateMolecules() {

  if( num_mol == allocatedValue["num_mol"] && max_num_uau == allocatedValue["max_num_uau"] )
    // this will include cases where the number to be allocated is zero, since
    // allocatedValue starts at zero
    return;
 
  // Allocation here needs num_mol, max_num_uau

  if(allocatedValue["num_mol"] > 0)
    FreeMolecules();

  // Length of lists
  lmlist = (max_num_uau + 1)*num_mol;

  // mlist: Allocate memory on host and initialize
  size_t nbytes = sizeof(int1)*lmlist;
  if ( cudaMallocHost((void **)&h_mlist, nbytes) == cudaErrorMemoryAllocation)
    throw RUMD_Error("MoleculeData","AllocateMolecules","Memory allocation failure");
  if ( cudaMalloc((void **)&d_mlist, nbytes) == cudaErrorMemoryAllocation)
    throw RUMD_Error("MoleculeData","AllocateMolecules","Memory allocation failure");
  for ( unsigned n=0; n<lmlist; n++ ) h_mlist[n].x = -1;


  // Data arrays 
  nbytes = sizeof(float4)*num_mol;
  if ( cudaMalloc((void **)&d_cm, nbytes) == cudaErrorMemoryAllocation || 
       cudaMallocHost((void **)&h_cm, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateMolecules","Memory allocation failure");

 if ( cudaMalloc((void **)&d_cm_im, nbytes) == cudaErrorMemoryAllocation || 
       cudaMallocHost((void **)&h_cm_im, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateMolecules","Memory allocation failure");

  nbytes = sizeof(float3)*num_mol;
  if ( cudaMalloc((void **)&d_vcm, nbytes) == cudaErrorMemoryAllocation || 
       cudaMallocHost((void **)&h_vcm, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateMolecules","Memory allocation failure");
  
  nbytes = sizeof(float3)*num_mol;
  if ( cudaMalloc((void **)&d_s, nbytes) == cudaErrorMemoryAllocation || 
       cudaMallocHost((void **)&h_s, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateMolecules","Memory allocation failure");
  
  nbytes = sizeof(float3)*num_mol;
  if ( cudaMalloc((void **)&d_omega, nbytes) == cudaErrorMemoryAllocation || 
       cudaMallocHost((void **)&h_omega, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateMolecules","Memory allocation failure");

  nbytes = sizeof(float)*num_mol*9;
  if ( cudaMalloc((void **)&d_inertia, nbytes) == cudaErrorMemoryAllocation || 
       cudaMallocHost((void **)&h_inertia, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateMolecules","Memory allocation failure");

  nbytes = sizeof(float)*num_mol*9;
  if ( cudaMalloc((void **)&d_stress, nbytes) == cudaErrorMemoryAllocation ||
       cudaMallocHost((void **)&h_stress, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","MoleculeData","Memory allocation failure");

  allocatedValue["num_mol"] = num_mol;
  allocatedValue["max_num_uau"] = max_num_uau;
}

void MoleculeData::AllocateBonds() {

  if( num_bonds == allocatedValue["num_bonds"] && num_btypes == allocatedValue["num_btypes"] )
    return;

  if(allocatedValue["num_bonds"] > 0)
    FreeBonds();

  // Allocating  bond list
  size_t nbytes = sizeof(uint2)*num_bonds;
  if ( cudaMalloc((void **)&d_blist, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateBonds","Memory allocation failure");

  // Allocating bond type list
  nbytes = sizeof(uint1)*num_bonds;
  if ( cudaMalloc((void **)&d_btlist, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateBonds","Memory allocation failure");


  // btlist_int: Allocate on host and device and initialize
  nbytes = sizeof(int1)*num_btypes;
  if ( cudaMallocHost((void **)&h_btlist_int, nbytes) == cudaErrorMemoryAllocation)
    throw RUMD_Error("MoleculeData",__func__,"Memory allocation failure");
  if ( cudaMalloc((void **)&d_btlist_int, nbytes) == cudaErrorMemoryAllocation)
    throw RUMD_Error("MoleculeData",__func__,"Memory allocation failure");
  for ( unsigned n=0; n<num_btypes; n++ ) h_btlist_int[n].x = -1;


  // Allocate memory for parameter lists
  nbytes = sizeof(float2)*num_btypes;
  if ( ( h_bplist = (float2 *)malloc(nbytes) ) == NULL ||  
       cudaMalloc((void **)&d_bplist, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateBonds","Memory allocation failure");
  
  // Allocate for potential energy arrays
  nbytes = sizeof(float)*num_bonds;
  if ( ( h_belist = (float *)malloc(nbytes) ) == NULL ||  
       cudaMalloc((void **)&d_belist, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateBonds","Memory allocation failure");

  // Allocate for storing the bond lengths
  if ( ( h_bonds = (float *)malloc(nbytes) ) == NULL ||  
       cudaMalloc((void **)&d_bonds, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateBonds","Memory allocation failure");

  allocatedValue["num_bonds"] = num_bonds;
  allocatedValue["num_btypes"] = num_btypes;
}

void MoleculeData::AllocateAngles() {

  if( num_angles == allocatedValue["num_angles"] && num_atypes == allocatedValue["num_atypes"] )
    return;

  if(allocatedValue["num_angles"] > 0)
    FreeAngles();


  // Allocating angles and types
  size_t nbytes = sizeof(uint4)*num_angles;
  if ( cudaMalloc((void **)&d_alist, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateAngles","Memory allocation failure");

  // Allocate memory for parameters
  nbytes = sizeof(float2)*num_atypes;
  if ( ( h_aplist = (float2 *)malloc(nbytes) ) == NULL ||  
       cudaMalloc((void **)&d_aplist, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateAngles","Memory allocation failure");
  
  // Allocate for potential energy arrays
  nbytes = sizeof(float)*num_angles;
  if ( ( h_epot_angle = (float *)malloc(nbytes) ) == NULL ||  
       cudaMalloc((void **)&d_epot_angle, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateAngles","Memory allocation failure");

  // Allocate for storing the actual/computed angles
  if ( ( h_angles = (float *)malloc(nbytes) ) == NULL ||  
       cudaMalloc((void **)&d_angles, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateAngles","Memory allocation failure");
  
  allocatedValue["num_angles"] = num_angles;
  allocatedValue["num_atypes"] = num_atypes;
}

void MoleculeData::AllocateDihedrals() {

 if( num_dihedrals == allocatedValue["num_dihedrals"] && num_dtypes == allocatedValue["num_dtypes"] )
    return;

  if(allocatedValue["num_dihedrals"] > 0)
    FreeDihedrals();

  // Allocate dlist
  size_t nbytes = sizeof(uint4)*num_dihedrals;
  if ( cudaMalloc((void **)&d_dlist, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateDihedrals","Memory allocation failure");

  // Allocate and dtype
  nbytes = sizeof(uint1)*num_dihedrals;
  if ( cudaMalloc((void **)&d_dtype, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateDihedrals","Memory allocation failure");


  // Allocate memory for parameter lists (both host and device) - 6 because 
  // of the Ryckaert-Bellemans potential
  nbytes = sizeof(float)*6*num_dtypes;
  if ( ( h_dplist = (float *)malloc(nbytes) ) == NULL ||  
       cudaMalloc((void **)&d_dplist, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateDihedrals","Memory allocation failure");
  
  // Allocate for potential energy arrays
  nbytes = sizeof(float)*num_dihedrals;
  if ( ( h_epot_dihedral = (float *)malloc(nbytes) ) == NULL ||  
       cudaMalloc((void **)&d_epot_dihedral, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateDihedrals","Memory allocation failure");

  // Allocate for storing the actual/computed dihedrals
  if ( ( h_dihedrals = (float *)malloc(nbytes) ) == NULL ||  
       cudaMalloc((void **)&d_dihedrals, nbytes) == cudaErrorMemoryAllocation )
    throw RUMD_Error("MoleculeData","AllocateDihedrals","Memory allocation failure");

  allocatedValue["num_dihedrals"] = num_dihedrals;
  allocatedValue["num_dtypes"] = num_dtypes;
}


void MoleculeData::FreeMolecules() {

  if(num_mol == 0)
    throw RUMD_Error("MoleculeData","FreeMolecules","Attempt to free molecule arrays when none have been allocated (num_mol == 0)");

  cudaFreeHost(h_mlist);
  cudaFree(d_mlist);

  cudaFreeHost(h_cm);
  cudaFree(d_cm);
  cudaFreeHost(h_cm_im);
  cudaFree(d_cm_im);
  cudaFreeHost(h_vcm);
  cudaFree(d_vcm);
  cudaFreeHost(h_s);
  cudaFree(d_s);
  cudaFreeHost(h_omega);
  cudaFree(d_omega);
  cudaFreeHost(h_inertia);
  cudaFree(d_inertia);
}

void MoleculeData::FreeBonds() {

  if(num_bonds == 0)
    throw RUMD_Error("MoleculeData","FreeBonds","Attempt to free bond arrays when none have been allocated (num_bonds == 0)");

  cudaFree(d_blist);
  cudaFree(d_btlist);
  free(h_bplist);
  cudaFree(d_bplist);
  free(h_belist);
  cudaFree(d_belist);
  free(h_bonds);
  cudaFree(d_bonds);
  
  cudaFreeHost(h_btlist_int);
  cudaFree(d_btlist_int);

}

void MoleculeData::FreeAngles() {

  if(num_angles == 0)
    throw RUMD_Error("MoleculeData","FreeAngles","Attempt to free angles arrays when none have been allocated (num_angles == 0)");

  cudaFree(d_alist);
  free(h_aplist);
  cudaFree(d_aplist);
  free(h_epot_angle);
  cudaFree(d_epot_angle);
  free(h_angles);
  cudaFree(d_angles);
}

void MoleculeData::FreeDihedrals() {

  if(num_dihedrals == 0)
    throw RUMD_Error("MoleculeData","FreeDihedrals","Attempt to free dihedral arrays when none have been allocated (num_dihedrals == 0)");

  cudaFree(d_dlist);
  cudaFree(d_dtype);
  free(h_dplist);
  cudaFree(d_dplist);
  free(h_epot_dihedral);
  cudaFree(d_epot_dihedral);
  free(h_dihedrals);
  cudaFree(d_dihedrals);
}


///////////////////////////////////////////////////////
// Methods.
///////////////////////////////////////////////////////

// Read the top file - get number of molecules and number of maximum united atomic units
void MoleculeData::ReadTopology(const std::string& top_filename){
  char line[256];
  const char section[] = {'[', ' ', 'b', 'o', 'n', 'd', 's', ' ' , ']', '\n', '\0'};
  unsigned moli, ai, aj, btype;

  FILE *top = fopen(top_filename.c_str(), "r");
  if ( top == NULL )
    throw RUMD_Error("MoleculeData","ReadTopology",
		     "Couldn't open topology file.");

  // Set the file pointer to the line after 
  // the [ bond ] section string
  do { 
    fgets(line, 256, top);
  } while ( strcmp(line, section) != 0 ); 

  fgets(line, 256, top); // Skip next line (comments)

  std::vector<unsigned> marray;

  // Go through the pair list
  do {
    // Check that we have an entry
    fpos_t fpos;
    fgetpos(top, &fpos);    
    fgets(line, 256, top);
    if ( line[0] == '[' ) break; 

    // If so, reset pointer to line start and read again but with format
    fsetpos(top, &fpos);
    if ( fscanf(top, "%u%u%u%u\n", &moli, &ai, &aj, &btype) != 4 )
      throw RUMD_Error("MoleculeData","ReadTopology",
		       "Format not correct");
    
    if (moli >= marray.size())
      marray.push_back(0);
    marray[moli]++;

  } while ( !feof(top) );

  fclose(top);

  num_mol = marray.size();
  std::vector<unsigned>::iterator max_it = std::max_element(marray.begin(), marray.end());
  max_num_uau = (*max_it)+1;

  AllocateMolecules();

  std::cout << "Topology file " << top_filename << " found." << std::endl;  
  std::cout << "Read " << num_mol << " molecules "
	    << "with " << max_num_uau << " maximum united atomic units." << std::endl;
}


// Read the top file - get bonds
void MoleculeData::GetBondsFromTopology(const std::string& top_filename){
  char line[256];
  const char section[] = {'[', ' ', 'b', 'o', 'n', 'd', 's' , ' ', ']', '\n', '\0'};
  unsigned a, b, btype, moli;
  unsigned num_part = S->GetNumberOfParticles();

  unsigned *mlist_index = new unsigned [num_mol];
  for ( unsigned n=0; n<num_mol; n++ ) mlist_index[n] = 0;
  
  h_blist.clear();
  h_btlist.clear();
  num_btypes = 0;

  // Open file
  FILE *top = fopen(top_filename.c_str(), "r");
  if ( top == NULL )
    throw RUMD_Error("MoleculeData","GetBondsFromTopology",
		     "Couldn't open topology file.");

  // Set the file pointer to the line after the [ bond ] section string
  do { 
    fgets(line, 256, top);
  } while ( strcmp(line, section) != 0 ); 
  fgets(line, 256, top); // Skip next line (comments)

  // Go through the pair list
  do {
    // Check that we have an entry
    fpos_t fpos;
    fgetpos(top, &fpos);    
    fgets(line, 256, top);
    if ( line[0] == '[' ) break; 

    // If so, reset pointer to line start and read again but with format
    fsetpos(top, &fpos);
    if ( fscanf(top, "%u%u%u%u\n", &moli, &a, &b, &btype) !=4 )
      throw RUMD_Error("MoleculeData","GetBondsFromTopology",
		       "Format not correct");
    
    if ( moli>num_mol-1 || a>num_part-1 || b>num_part-1  )
      throw RUMD_Error("MoleculeData","GetBondsFromTopology",
		       "Indices are too large");

    h_blist.push_back(make_uint2(a,b));
    h_btlist.push_back(make_uint1(btype));

    // Add entries to mlist
    AddmlistEntry(mlist_index, moli, a, b);

    if ( btype > num_btypes ) num_btypes++;

  } while ( !feof(top) );
  
  fclose(top);

  num_bonds = h_blist.size();
  num_btypes++;

  AllocateBonds();
  
  // Copy molecule list
  cudaMemcpy(d_mlist, h_mlist, sizeof(int1)*lmlist, cudaMemcpyHostToDevice);
  // Copying bond list
  cudaMemcpy(d_blist, &(h_blist[0]), sizeof(uint2)*num_bonds, cudaMemcpyHostToDevice);
  // Copying bond type list
  cudaMemcpy(d_btlist, &(h_btlist[0]), sizeof(uint1)*num_bonds, cudaMemcpyHostToDevice);

  delete[] mlist_index;

  std::cout << "Read "<< num_bonds << " bond(s). " ;
  std::cout << "Read " << num_btypes << " bond type(s)." << std::endl;
}


// Read the top file - get number of angles
void MoleculeData::GetAnglesFromTopology(const std::string& top_filename){
  char line[256];
  const char section[] = {'[', ' ', 'a', 'n', 'g', 'l', 'e', 's' , ' ', ']', '\n', '\0'};
  unsigned a, b, c, atype, moli;
  unsigned num_part = S->GetNumberOfParticles();

  num_angles = 0;
  h_alist.clear();

  FILE *top = fopen(top_filename.c_str(), "r");
  if ( top == NULL )
    throw RUMD_Error("MoleculeData","GetAnglesFromTopology",
		     "Couldn't open topology file.");

  // Set the file pointer to the line after the [ angles ] section string
  do { 
    fgets(line, 256, top);
    if ( feof(top) ){
      fclose(top);
      return;
    }
  } while ( strcmp(line, section) != 0 ); 
  fgets(line, 256, top); // Skip next line (comments)

  // Go through the pair list
  do {
    // Check that we have an entry
    fpos_t fpos;
    fgetpos(top, &fpos);    
    fgets(line, 256, top);
    if ( line[0] == '[' ) break; 

    // If so, reset pointer to line start and read again but with format
    fsetpos(top, &fpos);
    if ( fscanf(top, "%u%u%u%u%u\n", &moli, &a, &b, &c, &atype) !=5 )
      throw RUMD_Error("MoleculeData","GetAnglesFromTopology",
		       "Format not correct");
    
    if ( moli>num_mol-1 || a>num_part-1 || b>num_part-1 || c>num_part-1 )
      throw RUMD_Error("MoleculeData","GetAnglesFromTopology",
		       "Indices are too large"); 
    
    h_alist.push_back(make_uint4(a, b, c, atype));

    if ( atype > num_atypes ) num_atypes++;
     
  } while ( !feof(top) );
  
  fclose(top);
  
  num_angles = h_alist.size();
  num_atypes ++;

  AllocateAngles();
  // Copying angles and types
  cudaMemcpy(d_alist, &(h_alist[0]), sizeof(uint4)*num_angles, cudaMemcpyHostToDevice);

  std::cout << "Read "<< num_angles << " angle(s). " ;
  std::cout << "Read " << num_atypes << " angle type(s)." << std::endl;
}


// Read the top file - get number of dihedrals
void MoleculeData::GetDihedralsFromTopology(const std::string& top_filename){
  char line[256];
  const char section[] = {'[', ' ', 'd', 'i', 'h', 'e', 'd', 'r' , 'a', 'l', 's', ' ', ']', '\n', '\0'};
  unsigned a, b, c, d, dtype, moli;
  unsigned num_part = S->GetNumberOfParticles();

  num_dihedrals = 0;
  h_dlist.clear();
  h_dtype.clear();

  FILE *top = fopen(top_filename.c_str(), "r");
  if ( top == NULL )
    throw RUMD_Error("MoleculeData","GetDihedralsFromTopology",
		     "Couldn't open topology file.");

  // Set the file pointer to the line after the [ dihedrals ] section string
  do { 
    fgets(line, 256, top);
    if ( feof(top) ){
      fclose(top);
      return;
    }
  } while ( strcmp(line, section) != 0 );
  fgets(line, 256, top);  // Skip next line (comments)

  // Go through the pair list
  do {
    // Check that we have an entry
    fpos_t fpos;
    fgetpos(top, &fpos);    
    fgets(line, 256, top);
    if ( line[0] == '[' ) break;

    // If so, reset pointer to line start and read again but with format
    fsetpos(top, &fpos);
    if ( fscanf(top, "%u%u%u%u%u%u\n", &moli, &a, &b, &c, &d, &dtype) != 6 )
      throw RUMD_Error("MoleculeData","GetDihedralsFromTopology",
		       "Format not correct");
    
    if ( moli>num_mol-1 ||
	 a>num_part-1 || b>num_part-1 || c>num_part-1 || d>num_part-1)
      throw RUMD_Error("MoleculeData","GetDihedralsFromTopology",
		       "Indices are too large");
 
    h_dlist.push_back(make_uint4(a,b,c,d));
    h_dtype.push_back(make_uint1(dtype));

    if ( dtype > num_dtypes ) num_dtypes++;
     
  } while ( !feof(top) );
  
  fclose(top);
  
  num_dihedrals = h_dlist.size();
  num_dtypes ++;

  AllocateDihedrals();

  // Copy dlist
  cudaMemcpy(d_dlist, &(h_dlist[0]), sizeof(uint4)*num_dihedrals, cudaMemcpyHostToDevice);

  // Copy dtype
  cudaMemcpy(d_dtype, &(h_dtype[0]), sizeof(uint1)*num_dihedrals, cudaMemcpyHostToDevice);
    
  std::cout << "Read "<< num_dihedrals<< " dihedrals(s). " ;
  std::cout << "Read " << num_dtypes << " dihedral type(s)."<< std::endl;
  
}

///////////////////////////////////////////////////////
// Add from topology
///////////////////////////////////////////////////////

// Helper function for top-reader
void MoleculeData::AddmlistEntry(unsigned *index, unsigned i, 
				 unsigned ai, unsigned aj){
  int flag[2] = {0};
  unsigned si = i*max_num_uau;
  
  for ( unsigned n=0; n<index[i]; n++ ){
    if ( h_mlist[si+n+num_mol].x == (signed)ai ) flag[0] = 1;
    if ( h_mlist[si+n+num_mol].x == (signed)aj ) flag[1] = 1;
  }

  if ( flag[0] == 0 ){
    h_mlist[si+index[i]+num_mol].x = ai;
    index[i]++;
  }
  
  if ( flag[1] == 0 ){
    h_mlist[si+index[i]+num_mol].x = aj;
    index[i]++;
  }
   
  h_mlist[i].x = (signed)index[i];
}

///////////////////////////////////////////////////////
// Set methods
///////////////////////////////////////////////////////


void  MoleculeData::SetBondParams( unsigned bond_type, float length_param, float stiffness_param, unsigned bond_class ){

  if( bond_type > num_btypes || length_param < 0)
    throw RUMD_Error("MoleculeData", __func__, "Type index too large or bond-length is negative."); // possible user error
  if( bond_class > 2)
    throw RUMD_Error("MoleculeData", __func__, "Something's wrong; invalid bond_class received"); // shouldn't be possible for a user to trigger this

  h_bplist[bond_type].x = length_param;
  h_bplist[bond_type].y = stiffness_param;

  h_btlist_int[bond_type].x = bond_class ;
  
  size_t nbytes = sizeof(float2)*num_btypes;
  cudaMemcpy(d_bplist, h_bplist, nbytes, cudaMemcpyHostToDevice);
  
  nbytes = sizeof(int1)*num_btypes;
  cudaMemcpy(d_btlist_int, h_btlist_int, nbytes, cudaMemcpyHostToDevice);
}


void MoleculeData::SetAngleParams( unsigned angle_type, float mtheta0, float mktheta ){

  h_aplist[angle_type].x = mtheta0;
  h_aplist[angle_type].y = mktheta;

  size_t nbytes = sizeof(float2)*num_atypes;
  cudaMemcpy(d_aplist, h_aplist, nbytes, cudaMemcpyHostToDevice);
}

void MoleculeData::SetDihedralParams(unsigned dihedral_type, float prm0, float prm1, float prm2, float prm3, float prm4, float prm5) {
 if ( num_dihedrals == 0 )
    throw RUMD_Error("MoleculeData",__func__, "No dihedrals defined in top-file.");

  unsigned offset = dihedral_type*6;

  h_dplist[offset]   = prm0;
  h_dplist[offset+1] = prm1;
  h_dplist[offset+2] = prm2;
  h_dplist[offset+3] = prm3;
  h_dplist[offset+4] = prm4;
  h_dplist[offset+5] = prm5;

  size_t nbytes = sizeof(float)*6*num_dtypes;
  cudaMemcpy(d_dplist, h_dplist, nbytes, cudaMemcpyHostToDevice);
}


void MoleculeData::SetExclusionBond(PairPotential* potential, unsigned etype){
  potential->SetExclusionBond(&(h_btlist[0]), &(h_blist[0]), num_bonds, etype);
}

void MoleculeData::SetExclusionMolecule(PairPotential *potential, unsigned molindex){
  potential->SetExclusionMolecule(h_mlist, molindex, max_num_uau, num_mol);
}

void MoleculeData::SetExclusionMolecule(PairPotential *potential){
  for ( unsigned n=0; n<num_mol; n++ ) potential->SetExclusionMolecule(h_mlist, n, max_num_uau, num_mol);
}

void MoleculeData::SetExclusionAngle(PairPotential* potential){
  potential->SetExclusionAngle(&(h_alist[0]), num_angles);
}

void MoleculeData::SetExclusionDihedral(PairPotential* potential){
  potential->SetExclusionDihedral(&(h_dlist[0]), num_dihedrals);
}


void MoleculeData::SetAngleConstant(unsigned type, float ktheta){
  h_aplist[type].y = ktheta;
  size_t nbytes = sizeof(float2)*num_atypes;
  cudaMemcpy(d_aplist, h_aplist, nbytes, cudaMemcpyHostToDevice);
}

void MoleculeData::SetDihedralRyckaertConstants(unsigned type, float c0, float c1, float c2, float c3, float c4, float c5){
  unsigned offset = type*6;

  h_dplist[offset]   = c0;
  h_dplist[offset+1] = c1;
  h_dplist[offset+2] = c2;
  h_dplist[offset+3] = c3;
  h_dplist[offset+4] = c4;
  h_dplist[offset+5] = c5;
    
  size_t nbytes = sizeof(float)*6*num_dtypes;
  cudaMemcpy(d_dplist, h_dplist, nbytes, cudaMemcpyHostToDevice);
}

void MoleculeData::SetPeriodicDihedralConstants(unsigned type, float phiEq, float v1, float v2, float v3){
  unsigned offset = type*6;

  h_dplist[offset]   = phiEq;
  h_dplist[offset+1] = v1;
  h_dplist[offset+2] = v2;
  h_dplist[offset+3] = v3;
  h_dplist[offset+4] = 0.f;
  h_dplist[offset+5] = 0.f;

  size_t nbytes = sizeof(float)*6*num_dtypes;
  cudaMemcpy(d_dplist, h_dplist, nbytes, cudaMemcpyHostToDevice);
}


void MoleculeData::UpdateAfterSorting( unsigned int* __attribute__((unused))old_index, unsigned int* new_index ){
  
  int1 *mlist_offset = &d_mlist[num_mol];
  unsigned length_mlist_offset = lmlist - num_mol;
  unsigned num_threads = 128;
  unsigned num_mlist_blocks = length_mlist_offset/num_threads + 1;
  
  UpdateMlistAfterSorting<<<num_mlist_blocks,num_threads>>>(  new_index, S->GetNumberOfParticles(),	mlist_offset, length_mlist_offset);

  if ( num_bonds > 0 ) {
    unsigned int num_bond_blocks = num_bonds/num_threads + 1;
    updateBondsAfterSorting<<<num_bond_blocks,num_threads>>>(  new_index, S->GetNumberOfParticles(), d_blist, num_bonds);
  }
  if( num_angles > 0 ) {
    unsigned int num_angle_blocks = num_angles/num_threads + 1;
    updateAnglesAfterSorting<<<num_angle_blocks,num_threads>>>(  new_index, S->GetNumberOfParticles(), d_alist, num_angles);
  }
  if( num_dihedrals > 0 ) {
    num_threads = 16;
    unsigned int num_dihedral_blocks = num_dihedrals/num_threads + 1;
    updateDihedralsAfterSorting<<<num_dihedral_blocks,num_threads>>>(  new_index, S->GetNumberOfParticles(), d_dlist, num_dihedrals);
  }

}

///////////////////////////////////////////////////////
// Methods for changing the volume
// Currently called by box changing functions of Sample
///////////////////////////////////////////////////////

void MoleculeData::IsotropicScaleCM( double factor ){
  // Scaling the box with the centers of mass of the molecules
  // while keeping the intra atomic distances fixed.

  unsigned num_part = S->GetNumberOfParticles();
  const ParticleData* P = S->GetParticleData();
  unsigned int num_threads_block = 32;
  dim3 numBlocks( (num_mol+num_threads_block-1)/num_threads_block );

  RectangularSimulationBox* test_RSB = dynamic_cast<RectangularSimulationBox*>(S->GetSimulationBox());
  if(!test_RSB) throw RUMD_Error("MoleculeData","IsotropicScaleCM","Unsupported SimulationBox");

  float4* d_r_intra; // Contains atomic positions with respect to its molecule's center of mass
  if( cudaMalloc( (void**) &d_r_intra, num_part*sizeof(float4) ) == cudaErrorMemoryAllocation ) 
    throw( RUMD_Error("MoleculeData","ChangeBoxKeepingCMConstant","Malloc failed on r_intra") );
  
  // Get atom positions relative to centers of mass, and scale the box
  EvalCM(true);

  // Get atom positions relative to centers of mass
  get_positions_relative_to_CM_kernel<<<numBlocks, num_threads_block>>>(num_mol, max_num_uau,
									d_r_intra, d_mlist, d_cm, d_cm_im,
									P->d_r, P->d_im,
									test_RSB, test_RSB->GetDevicePointer());  

  // Scale box
  test_RSB->ScaleBox(factor);

  // Scale molecules
  scale_CM_kernel<<<numBlocks, num_threads_block>>>(num_mol, max_num_uau, factor,
						    d_r_intra, d_mlist, d_cm, d_cm_im,
						    P->d_r, P->d_im,
						    test_RSB, test_RSB->GetDevicePointer());
  
  cudaFree(d_r_intra);
}


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
						    float  *simBoxPointer){ 
  
  unsigned int mol_idx = blockDim.x*blockIdx.x + threadIdx.x;
  
  if(mol_idx < num_mol) {
    unsigned offset = max_num_uau*mol_idx + num_mol;
    unsigned n_ua_units = mlist[mol_idx].x;

    for ( unsigned idx=0; idx<n_ua_units; idx++ ){
      unsigned at_idx = mlist[offset+idx].x;
      r_intra[at_idx] = simbox->calculateDistanceWithImages(atom_positions[at_idx], rcm[mol_idx],
							    atom_images[at_idx], imcm[mol_idx],
							    simBoxPointer);
    }
  }
}


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
				float  *simBoxPointer){ 
  
  unsigned int mol_idx = blockDim.x*blockIdx.x + threadIdx.x;
  
  if(mol_idx < num_mol) {
    unsigned offset = max_num_uau*mol_idx + num_mol;
    unsigned n_ua_units = mlist[mol_idx].x;

    // Scaling of the center of mass
    rcm[mol_idx].x *= factor;
    rcm[mol_idx].y *= factor;
    rcm[mol_idx].z *= factor;

    // Loop over atoms in this molecule
    for ( unsigned idx=0; idx<n_ua_units; idx++ ){
      unsigned at_idx = mlist[offset+idx].x;
      atom_images[at_idx] = imcm[mol_idx];

      // Get new position
      float4 r = rcm[mol_idx];
      r.x += r_intra[at_idx].x;
      r.y += r_intra[at_idx].y;
      r.z += r_intra[at_idx].z;
      r.w = atom_positions[at_idx].w;
      
      // Take care of boundaries, images
      float4 image = simbox->applyBoundaryCondition(r, simBoxPointer);
      image.x += imcm[mol_idx].x;
      image.y += imcm[mol_idx].y;
      image.z += imcm[mol_idx].z;

      // Update the device data
      atom_positions[at_idx] = r;
      atom_images[at_idx] = image;
    }
  }
}


///////////////////////////////////////////////////////
// Debugging methods
///////////////////////////////////////////////////////

// Print mlist and blist (for debugging mostly)
void MoleculeData::PrintLists(void){

  std::cout << "--- blist: ---" << std::endl;
  for ( unsigned i=0; i<num_bonds; i++ )
    std::cout << h_blist[i].x << " " << h_blist[i].y << std::endl;
 	
  std::cout << "--- mlist: ---" << std::endl;
  for ( unsigned i=0; i<lmlist; i++)
    std::cout << h_mlist[i].x << std::endl;
}


// Print mlist and blist for i (for debugging mostly)
void MoleculeData::PrintLists(unsigned i){
  
  if(i>S->GetNumberOfParticles())
    std::cout << "index " << i << " too large (number of particles is " << S->GetNumberOfParticles() << ")" << std::endl; 
    else {
  std::cout << "Atom index " << i << " has " <<  h_blist[i].x 
	    << " bonds." << " Binding indices "; 
  for ( unsigned n=0; n<h_blist[i].x; n++ )
    std::cout << h_blist[S->GetNumberOfVirtualParticles() 
			 + i*max_num_bonds + n].x << " ";
  std::cout << std::endl;
  }
}

// Print mlist to file - convenient for post run data analysis
void MoleculeData::PrintMlist(){
  FILE *fout = fopen("mlist.top", "w");

  if ( fout == NULL )
    throw RUMD_Error("MoleculeData","PrintMlist",
		     "Couldn't open file for writing - something seriously wrong!");

  std::cout << "Writing mlist to mlist.top" << std::endl;

  for ( unsigned n=0; n<lmlist; n++ )
    fprintf(fout, "%d\n", h_mlist[n].x);

  fclose(fout);
}

/////////////////////////////////
// Eval methods
////////////////////////////////////

void MoleculeData::EvalMolMasses(){
  if(!S)
    throw RUMD_Error("MoleculeData","EvalMolMasses","No Sample has been set");

  const ParticleData* particleData = S->GetParticleData();

  particleData->CopyVelFromDevice();
    
  for ( unsigned n=0; n<num_mol; n++ )  h_cm[n].w = 0.0f;
  
  for ( unsigned n=0; n<num_mol; n++ ){
    unsigned offset = max_num_uau*n + num_mol;
    
    for ( int m=0; m<h_mlist[n].x; m++ ){
      unsigned a = h_mlist[offset+m].x;
      h_cm[n].w += 1.0/particleData->h_v[a].w;
    }
  }
}


void MoleculeData::EvalVelCM(bool copy_to_host){
  const ParticleData* P = S->GetParticleData();
  unsigned int num_threads_block = 32;
  dim3 numBlocks( (num_mol+num_threads_block-1)/num_threads_block );
  
  LeesEdwardsSimulationBox* test_LESB = dynamic_cast<LeesEdwardsSimulationBox*>(S->GetSimulationBox());
  RectangularSimulationBox* test_RSB = dynamic_cast<RectangularSimulationBox*>(S->GetSimulationBox());

  if(test_LESB)
    eval_vel_CM_kernel<<<numBlocks, num_threads_block>>>(num_mol, max_num_uau, d_mlist, P->d_v, d_vcm, test_LESB, test_LESB->GetDevicePointer());
  else if(test_RSB)
    eval_vel_CM_kernel<<<numBlocks, num_threads_block>>>(num_mol, max_num_uau, d_mlist, P->d_v, d_vcm, test_RSB, test_RSB->GetDevicePointer());
  else
    throw RUMD_Error("MoleculeData","EvalVelCM","Unknown SimulationBox");

  size_t nbytes = sizeof(float3)*num_mol;
  if(copy_to_host)
    cudaMemcpy(h_vcm, d_vcm, nbytes, cudaMemcpyDeviceToHost);
}


template <class Simbox>
__global__ void eval_vel_CM_kernel(unsigned int num_mol, 
				   unsigned int max_num_uau,
				   int1* mlist,
				   float4 *atom_velocities,
				   float3 *vcm, 
				   Simbox *simbox, 
				   float *simBoxPointer){
  unsigned int mol_idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(mol_idx < num_mol) {

    unsigned offset = max_num_uau*mol_idx + num_mol;
    unsigned n_ua_units = mlist[mol_idx].x;
    float3 this_vcm = {0.f, 0.f, 0.f};
    float total_mass = 0.f;

    for ( unsigned idx=0; idx<n_ua_units; idx++ ){
      unsigned at_idx = mlist[offset+idx].x;
      float4 v = atom_velocities[at_idx];
      float mass = 1.f/v.w;

      this_vcm.x += v.x*mass;
      this_vcm.y += v.y*mass;
      this_vcm.z += v.z*mass;
      total_mass += mass;
    } // end loop over particles this molecule

    this_vcm.x /= total_mass;
    this_vcm.y /= total_mass;
    this_vcm.z /= total_mass;
    
    vcm[mol_idx] = this_vcm;
  } // if(mol_idx .... )
}



void MoleculeData::EvalCM(bool copy_to_host){
  const ParticleData* P = S->GetParticleData();
  unsigned int num_threads_block = 32;
  dim3 numBlocks( (num_mol+num_threads_block-1)/num_threads_block );
  
  LeesEdwardsSimulationBox* test_LESB = dynamic_cast<LeesEdwardsSimulationBox*>(S->GetSimulationBox());
  RectangularSimulationBox* test_RSB = dynamic_cast<RectangularSimulationBox*>(S->GetSimulationBox());

  if(test_LESB)
    eval_CM_kernel<<<numBlocks, num_threads_block>>>(num_mol, max_num_uau, d_mlist,
						     P->d_r, P->d_v, P->d_im,
						     d_cm, d_cm_im,
						     test_LESB, test_LESB->GetDevicePointer());
  else if(test_RSB)
    eval_CM_kernel<<<numBlocks, num_threads_block>>>(num_mol, max_num_uau, d_mlist,
						     P->d_r, P->d_v, P->d_im,
						     d_cm, d_cm_im,
						     test_RSB, test_RSB->GetDevicePointer());
  else
    throw RUMD_Error("MoleculeData","EvalCM","Unknown SimulationBox");
  
  size_t nbytes = sizeof(float4)*num_mol;
  if(copy_to_host) {
    cudaMemcpy(h_cm,    d_cm,    nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cm_im, d_cm_im, nbytes, cudaMemcpyDeviceToHost);
  }
}

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
			       float *simBoxPointer){ 
  unsigned int mol_idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(mol_idx < num_mol) {
    unsigned offset = max_num_uau*mol_idx + num_mol;
    unsigned n_ua_units = mlist[mol_idx].x;
    unsigned at_idx = mlist[offset].x;
    float4 atom_pos = atom_positions[at_idx];
    float4 atom_vel = atom_velocities[at_idx];
    float4 atom_im  = atom_images[at_idx];
    
    float x = atom_pos.x;
    float y = atom_pos.y;
    float z = atom_pos.z;

    float atom_mass = 1.0/atom_vel.w;
    float4 this_rcm = {atom_mass*x, atom_mass*y, atom_mass*z, atom_mass};
    float4 this_imcm = atom_im;

    for ( unsigned idx=1; idx<n_ua_units; idx++ ){
      at_idx = mlist[offset+idx].x;
      float4 next_atom_pos = atom_positions[at_idx];
      float4 next_atom_im  = atom_images[at_idx];
      atom_mass = 1./atom_velocities[at_idx].w;

      float4 displacement = simbox->calculateDistanceWithImages(next_atom_pos,
								atom_pos,
								next_atom_im,
								atom_im,
								simBoxPointer);

      x += displacement.x; y += displacement.y; z += displacement.z;
      
      this_rcm.x += atom_mass*x;
      this_rcm.y += atom_mass*y;
      this_rcm.z += atom_mass*z;
      this_rcm.w += atom_mass;
      
      atom_pos.x = next_atom_pos.x;
      atom_pos.y = next_atom_pos.y;
      atom_pos.z = next_atom_pos.z;

      atom_im.x = next_atom_im.x;
      atom_im.y = next_atom_im.y;
      atom_im.z = next_atom_im.z;
    } // for ( unsigned idx=1 ... )

    this_rcm.x /= this_rcm.w;
    this_rcm.y /= this_rcm.w;
    this_rcm.z /= this_rcm.w;
    float4 image = simbox->applyBoundaryCondition(this_rcm, simBoxPointer);
    this_imcm.x += image.x;
    this_imcm.y += image.y;
    this_imcm.z += image.z;
    
    rcm[mol_idx]  = this_rcm;
    imcm[mol_idx] = this_imcm;
  } //  if(mol_idx < num_mol)

}


// Could-Should be implemented as device function
void MoleculeData::EvalSpin(){
  const ParticleData* particleData = S->GetParticleData();
  particleData->CopyVelFromDevice();
  EvalCM();

  for ( unsigned n=0; n<num_mol; n++ ){
    
    unsigned nuau   = h_mlist[n].x;
    unsigned offset = max_num_uau*n + num_mol;
    unsigned i_offset = n*9;
    
    h_s[n].x = 0.0; h_s[n].y = 0.0; h_s[n].z = 0.0;
    for ( int k=0; k<9; k++ ) h_inertia[i_offset + k] = 0.0;
    
    for ( unsigned i=0; i<nuau; i++ ){

      unsigned in = h_mlist[offset+i].x;
      float mass = 1.0/particleData->h_v[in].w;
   
      LeesEdwardsSimulationBox* test_LESB = dynamic_cast<LeesEdwardsSimulationBox*>(S->GetSimulationBox());
      RectangularSimulationBox* test_RSB = dynamic_cast<RectangularSimulationBox*>(S->GetSimulationBox());
      float4 dcm;
      if(test_LESB)
	dcm = test_LESB->calculateDistance(particleData->h_r[in], h_cm[n], test_LESB->GetHostPointer());	
      else if(test_RSB)
	dcm = test_RSB->calculateDistance(particleData->h_r[in], h_cm[n], test_RSB->GetHostPointer());
      else
	throw RUMD_Error("MoleculeData","EvalSpin","Unknown SimulationBox");
      
      float4 vel = particleData->h_v[in];
      
      // Angular momentum
      h_s[n].x += dcm.y*mass*vel.z - dcm.z*mass*vel.y;
      h_s[n].y += dcm.z*mass*vel.x - dcm.x*mass*vel.z;
      h_s[n].z += dcm.x*mass*vel.y - dcm.y*mass*vel.x;

      // Inertia tensor
      float dcmx2 = mass*dcm.x*dcm.x;  
      float dcmy2 = mass*dcm.y*dcm.y; 
      float dcmz2 = mass*dcm.z*dcm.z;

      h_inertia[i_offset]   += dcmy2 + dcmz2;
      h_inertia[i_offset+4] += dcmx2 + dcmz2;
      h_inertia[i_offset+8] += dcmy2 + dcmx2;

      h_inertia[i_offset+1] -= mass*dcm.x*dcm.y;
      h_inertia[i_offset+2] -= mass*dcm.x*dcm.z;
      h_inertia[i_offset+5] -= mass*dcm.y*dcm.z;
    }

    // Symmetri 
    h_inertia[i_offset+3] = h_inertia[i_offset+1];
    h_inertia[i_offset+6] = h_inertia[i_offset+2];
    h_inertia[i_offset+7] = h_inertia[i_offset+5];

    // Matrix rank should be checked: det neq 0 -
    if ( fabs(Determinant3(&h_inertia[i_offset])) < FLT_EPSILON ){
      float inert =  h_inertia[i_offset] + h_inertia[i_offset+4] +  h_inertia[i_offset+8];
      inert /= 3.0;

      h_omega[n].x = h_s[n].x/inert;
      h_omega[n].y = h_s[n].y/inert;
      h_omega[n].z = h_s[n].z/inert;
    }
    else {
      float s[3], omega[3];  // Solve for ang. velocity

      s[0] = h_s[n].x; s[1] = h_s[n].y; s[2] = h_s[n].z;

      GaussPivotBackCPU(omega, s, &h_inertia[i_offset], 3);
      
      h_omega[n].x = omega[0]; h_omega[n].y = omega[1]; h_omega[n].z = omega[2];
    }
  }
}


void MoleculeData::SetRigidBodyVelocities() {
  if(!S)
    throw RUMD_Error("MoleculeData", __func__, "No Sample has been set");


  EvalVelCM(false); // don't copy to host

  // for now use the existing host-based EvalSpin and copy to device.
  // This also calcualtes the center of mass positions
  EvalSpin();
  size_t nbytes = sizeof(float3)*num_mol;
  cudaMemcpy(d_omega,    h_omega,    nbytes, cudaMemcpyHostToDevice);
  
  // now have CM positions and velocities
  const ParticleData* P = S->GetParticleData();
  unsigned int num_threads_block = 32;
  dim3 numBlocks( (num_mol+num_threads_block-1)/num_threads_block );


  LeesEdwardsSimulationBox* test_LESB = dynamic_cast<LeesEdwardsSimulationBox*>(S->GetSimulationBox());
  RectangularSimulationBox* test_RSB = dynamic_cast<RectangularSimulationBox*>(S->GetSimulationBox());

  if(test_LESB)
    setrigidbodyvelocities_kernel<<<numBlocks, num_threads_block>>>(num_mol, max_num_uau, d_mlist, P->d_r, P->d_v, d_cm, d_vcm, d_omega, test_LESB, test_LESB->GetDevicePointer());
  else if(test_RSB)
    setrigidbodyvelocities_kernel<<<numBlocks, num_threads_block>>>(num_mol, max_num_uau, d_mlist, P->d_r, P->d_v, d_cm, d_vcm, d_omega, test_RSB, test_RSB->GetDevicePointer());
  else
    throw RUMD_Error("MoleculeData","EvalCM","Unknown SimulationBox");
  
	  
}


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
					      float *simBoxPointer){
  unsigned int mol_idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(mol_idx < num_mol) {
    unsigned offset = max_num_uau*mol_idx + num_mol;
    unsigned n_ua_units = mlist[mol_idx].x;
    
    float4 pos_cm = rcm[mol_idx];
    float3 vel_cm = vcm[mol_idx];
    float3 ang_vel = omega[mol_idx];
    ang_vel.x = 0.f;
    ang_vel.y = 0.f;
    
    // Load the simulation box in local memory to avoid bank conflicts.
    float simBoxPtr_local[simulationBoxSize];
    simbox->loadMemory(simBoxPtr_local, simBoxPointer);
    
    for ( unsigned idx=0; idx<n_ua_units; idx++ ){
      unsigned at_idx = mlist[offset+idx].x;
      float4 atom_pos = atom_positions[at_idx];
      float4 dcm = simbox->calculateDistance(atom_pos, pos_cm, simBoxPtr_local);
      float3 atom_vel = vel_cm;
      atom_vel.x += ang_vel.y * dcm.z - ang_vel.z * dcm.y;
      atom_vel.y +=  ang_vel.z * dcm.x - ang_vel.x * dcm.z;
      atom_vel.z += ang_vel.x * dcm.y - ang_vel.y * dcm.x;
      
      atom_velocities[at_idx].x = atom_vel.x;
      atom_velocities[at_idx].y = atom_vel.y;
      atom_velocities[at_idx].z = atom_vel.z;
    }
  } // if mol_idx ...
}

///////////////////////////////////
// Write and Get methods
///////////////////////////////

void MoleculeData::WriteMolConf(const std::string&fstr){

  EvalVelCM();
  EvalCM();

  float4 box = S->GetSimulationBox()->GetSimulationBox();

  FILE *fptr = fopen(fstr.c_str(), "w");
  if ( fptr == NULL )
    throw RUMD_Error("MoleculeData", __func__, "Couldn't open file.");

  fprintf(fptr, "%d\n", num_mol);
  fprintf(fptr, "ioformat=1 boxLengths=%f,%f,%f columns=type,x,y,z,vx,vy,vz\n", 
	  box.x, box.y, box.z);

  // Note: Molecule types not implemented yet.
  for ( unsigned n=0; n<num_mol; n++ ){
    fprintf(fptr, "0 %f %f %f %f %f %f\n", h_cm[n].x, h_cm[n].y, h_cm[n].z, 
	    h_vcm[n].x, h_vcm[n].y, h_vcm[n].z);
  } 
 
  fclose(fptr);
}


void MoleculeData::GetBonds(){
  size_t nbytes = num_bonds*sizeof(float);
  cudaMemcpy(h_bonds, d_bonds, nbytes, cudaMemcpyDeviceToHost);
}


void MoleculeData::GetAngles(){
  size_t nbytes = num_angles*sizeof(float);
  cudaMemcpy(h_angles, d_angles, nbytes, cudaMemcpyDeviceToHost);
}


void MoleculeData::GetDihedrals(){
  size_t nbytes = num_dihedrals*sizeof(float);
  cudaMemcpy(h_dihedrals, d_dihedrals, nbytes, cudaMemcpyDeviceToHost);
}


void MoleculeData::WriteMolxyz(const std::string& fstr, unsigned mol_id, 
			       const std::string& mode){
 
  FILE *fout = fopen(fstr.c_str(), mode.c_str());
  if ( fout == NULL )
    throw RUMD_Error("MoleculeData", __func__, "Couldn't open file.");

  const ParticleData* particleData = S->GetParticleData();
  particleData->CopyPosFromDevice();
  
  unsigned nuau = h_mlist[mol_id].x;
  
  fprintf(fout, "%u\n", nuau);
  fprintf(fout, "\n");

  size_t offset = max_num_uau*mol_id + num_mol;
  for ( unsigned i=offset; i<offset+nuau; i++ ){
    unsigned j = h_mlist[i].x;
    fprintf(fout, "%d %f %f %f\n", particleData->h_Type[j], 
	    particleData->h_r[j].x, particleData->h_r[j].y, 
	    particleData->h_r[j].z);
  }

  fclose(fout);

}

void MoleculeData::WriteMolxyz_filtered(const std::string& fstr, unsigned filter_uau,
                               const std::string& mode){
// Print out configuration excluding molecules with 'filter_uau' particles 
   
  unsigned nuau = 0;
  for ( unsigned i=0; i< num_mol; i++ ){ 
    unsigned nuau_i = h_mlist[i].x;
    if (nuau_i!=filter_uau)
      nuau += nuau_i;
  }     

  FILE *fout = fopen(fstr.c_str(), mode.c_str());
  if ( fout == NULL )
    throw RUMD_Error("MoleculeData", __func__, "Couldn't open file.");

  const ParticleData* particleData = S->GetParticleData();
  particleData->CopyPosFromDevice();
  
  fprintf(fout, "%u\n", nuau);
  fprintf(fout, "\n");

  for ( unsigned mol=0; mol< num_mol; mol++ ){ 
   nuau =  h_mlist[mol].x;
   if (nuau!=filter_uau) {
     size_t offset = max_num_uau*mol + num_mol;
     for ( unsigned i=offset; i<offset+nuau; i++ ){
       unsigned j = h_mlist[i].x;
       fprintf(fout, "%d %f %f %f\n", nuau,//S->particleData.h_Type[j], 
       	    particleData->h_r[j].x, particleData->h_r[j].y, 
  	    particleData->h_r[j].z);
       }
    }
  }
  fclose(fout);

}

//////////////////////////////////////////////////////
/// Methods for updating datastructures after sorting
//////////////////////////////////////////////////////


__global__ void updateBondsAfterSorting(  unsigned int *new_index,
					     unsigned int nParticles,
					     uint2 *blist,
					     unsigned num_bonds ){

  unsigned bond_index = blockIdx.x*blockDim.x + threadIdx.x;
  
  if ( bond_index < num_bonds ){
    uint2 old_bond = blist[bond_index];
    uint2 new_bond = {new_index[old_bond.x], new_index[old_bond.y]};
    blist[bond_index] = new_bond;
  }
}


__global__ void updateAnglesAfterSorting( unsigned int *new_index,
					  unsigned int nParticles,
					  uint4 *alist,
					  unsigned num_angles ){

  unsigned angle_index = blockIdx.x*blockDim.x + threadIdx.x;
  
  if ( angle_index < num_angles ){

    uint4 old_angle = alist[angle_index];
    uint4 new_angle = { new_index[old_angle.x],
			new_index[old_angle.y],
			new_index[old_angle.z],
			old_angle.w};
    alist[angle_index] = new_angle;
  }
}


__global__ void updateDihedralsAfterSorting(  unsigned int *new_index, 
						 unsigned int nParticles,
						 uint4 *dlist,
						 unsigned num_dihedrals ){

  unsigned dihedral_index = blockIdx.x*blockDim.x + threadIdx.x;
  
  if ( dihedral_index < num_dihedrals ){
    uint4 old_dihedral = dlist[dihedral_index];
    uint4 new_dihedral = {new_index[old_dihedral.x],
			  new_index[old_dihedral.y],
			  new_index[old_dihedral.z],
			  new_index[old_dihedral.w]};
    dlist[dihedral_index] = new_dihedral;
  }
}


// Kernel for updating d_mlist after sorting
__global__ void UpdateMlistAfterSorting(unsigned int *new_index, unsigned nParticles,
                                        int1 *mlist_offset, unsigned length_mlist_offset ){
  
  unsigned total_threadIdx = blockIdx.x*blockDim.x + threadIdx.x;

  if ( total_threadIdx < length_mlist_offset && mlist_offset[total_threadIdx].x != -1 ) {
    unsigned old_idx = mlist_offset[total_threadIdx].x;
    unsigned new_idx = new_index[old_idx];
    mlist_offset[total_threadIdx].x = new_idx;
  }

}
