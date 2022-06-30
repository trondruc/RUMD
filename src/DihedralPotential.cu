
/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/DihedralPotential.h"
#include "rumd/MoleculeData.h"
#include "rumd/Sample.h"
#include "rumd/rumd_algorithms.h"


#define DIHEDRAL_EPS 1.0e-3

__global__ void DihedralResetForces( float4 *f, float4 *w ){

  f[blockIdx.x].x = f[blockIdx.x].y = f[blockIdx.x].z = f[blockIdx.x].w = 0.0;
  w[blockIdx.x].w = 0.0;

}

///////////////////////////////////////////////////////
// Dihedral common functions
///////////////////////////////////////////////////////


void DihedralPotential::SetParams(unsigned dihedral_type, std::vector<float> coeffs) {
  if(coeffs.size() != num_dihedral_params)
    throw RUMD_Error("DihedralPotential", __func__, "Wrong number of parameters in SetParams");
  dihedral_params[dihedral_type] = coeffs;

  if(sample && sample->GetMoleculeData())
    CopyParamsToGPU();
}


void DihedralPotential::Initialize() {
  MoleculeData* moleculeData = sample->GetMoleculeData();

  if(!moleculeData)
    throw RUMD_Error("DihedralPotential", __func__, "No molecular data available, call ReadMoleculeData");

  CopyParamsToGPU();
}


void DihedralPotential::CopyParamsToGPU() {
  MoleculeData* moleculeData = sample->GetMoleculeData();
  float params_array[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  
  std::map<unsigned, std::vector<float> >::iterator params_it;
  for(params_it = dihedral_params.begin(); params_it != dihedral_params.end(); params_it++) {
    unsigned a_type = params_it->first;
    std::vector<float> a_params = params_it->second;
    for (unsigned idx=0; idx<a_params.size(); idx++)
      params_array[idx] = a_params[idx];
    moleculeData->SetDihedralParams(a_type, params_array[0], params_array[1],
				    params_array[2], params_array[3],
				    params_array[4], params_array[5]);
  }
}


void DihedralPotential::SetExclusions(PairPotential* non_bond_pot) {
  if(exclude_dihedral)
    sample->GetMoleculeData()->SetExclusionDihedral(non_bond_pot);
}


///////////////////////////////////////////////////////
// Ryckaert potential
///////////////////////////////////////////////////////

DihedralRyckaert::DihedralRyckaert() : DihedralPotential() {
  ID_String = "DihedralRyckaert";
  num_dihedral_params = 6;
}

void DihedralRyckaert::CalcF(bool initialize, bool __attribute__((unused))calc_stresses){
  
  if ( initialize )
    DihedralResetForces<<<particleData->GetNumberOfParticles(),1>>>(particleData->d_f, particleData->d_w);

  RectangularSimulationBox* RSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  LeesEdwardsSimulationBox* LESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);

  MoleculeData* M = sample->GetMoleculeData();
  unsigned num_dihedrals = M->GetNumberOfDihedrals();
  unsigned num_blocks = num_dihedrals/num_threads + 1;
  
  if ( LESB )
    Ryckaert<1><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_dlist, M->d_dtype, M->d_dplist, 
	M->d_epot_dihedral, M->d_dihedrals, 
	num_dihedrals, LESB, LESB->GetDevicePointer() );
  else if ( RSB )
    Ryckaert<1><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_dlist, M->d_dtype, M->d_dplist, 
	M->d_epot_dihedral, M->d_dihedrals, 
	num_dihedrals, RSB, RSB->GetDevicePointer() );
  else
    throw RUMD_Error("DihedralRyckaert","CalcF","unknown simulation box");

}

double DihedralRyckaert::GetPotentialEnergy(){

  RectangularSimulationBox* RSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  LeesEdwardsSimulationBox* LESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);
  
  MoleculeData* M = sample->GetMoleculeData();
  unsigned num_dihedrals = M->GetNumberOfDihedrals();
  unsigned num_blocks = num_dihedrals/num_threads + 1;

  if ( LESB )
    Ryckaert<2><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_dlist, M->d_dtype, M->d_dplist, 
	M->d_epot_dihedral, M->d_dihedrals, 
	num_dihedrals, LESB, LESB->GetDevicePointer() );
  else if ( RSB )
    Ryckaert<2><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_dlist, M->d_dtype, M->d_dplist, 
	M->d_epot_dihedral, M->d_dihedrals, 
	num_dihedrals, RSB, RSB->GetDevicePointer() );
  else
    throw RUMD_Error("DihedralRyckaert","GetPotentialEnergy","unknown simulation box");
  
  // copy to host
  size_t nbytes = num_dihedrals*sizeof(float);
  cudaMemcpy(M->h_epot_dihedral, M->d_epot_dihedral, nbytes, cudaMemcpyDeviceToHost);
  
  // sum over bonds
  double epot = 0.0;
  for ( unsigned n=0; n<num_dihedrals; n++ )  
    epot += (double)M->h_epot_dihedral[n];

  return epot;
}


template <int energy, class Simbox> 
__global__ void Ryckaert( float4 *r, float4 *f, 
			  uint4 *dlist, uint1 *dtype, float *plist, 
			  float *d_epot_dihedral, float *d_dihedrals,
			  unsigned num_dihedrals, Simbox *simbox, float *simboxpointer ){

  // First approach is to let one thread work on a dihedral
  float4 dr1, dr2, dr3;    
  float f1, f2, p[6];

  
  unsigned dihedral_index = blockIdx.x*blockDim.x + threadIdx.x;

  if ( dihedral_index < num_dihedrals ){
    unsigned dihedral_type = dtype[dihedral_index].x;

    unsigned a = dlist[dihedral_index].x;
    unsigned b = dlist[dihedral_index].y;
    unsigned c = dlist[dihedral_index].z;
    unsigned d = dlist[dihedral_index].w;

    unsigned offset = 6*dihedral_type;
    for ( int k=0; k<6; k++ ) p[k] = plist[offset+k];
  
    // Load the simulation box in local memory to avoid bank conflicts.
    float array[simulationBoxSize];
    simbox->loadMemory(array, simboxpointer);
  
    dr1 = simbox->calculateDistance(r[b], r[a], array);
    dr2 = simbox->calculateDistance(r[c], r[b], array);
    dr3 = simbox->calculateDistance(r[d], r[c], array);

    float c11 = dr1.w;
    float c12 = dr1.x*dr2.x + dr1.y*dr2.y + dr1.z*dr2.z;
    float c13 = dr1.x*dr3.x + dr1.y*dr3.y + dr1.z*dr3.z;
    float c22 = dr2.w;
    float c23 = dr2.x*dr3.x + dr2.y*dr3.y + dr2.z*dr3.z;
    float c33 = dr3.w;

    float cA = c13*c22 - c12*c23;
    float cB1 = c11*c22 - c12*c12;
    float cB2 = c22*c33 - c23*c23;

    if ( fabs(c23*c23/(c22*c33) - 1) < DIHEDRAL_EPS ||
	 fabs(c12*c12/(c11*c22) - 1) < DIHEDRAL_EPS ){

      d_dihedrals[dihedral_index] = 0.0;

    }
    else {

      float cD = sqrt(cB1*cB2);
      float cc = cA/cD;
      
      d_dihedrals[dihedral_index] = acos(cc);
      
      float epot = p[0] + (p[1]+(p[2]+(p[3]+(p[4]+p[5]*cc)*cc)*cc)*cc)*cc;
      
      if ( energy == 2 ){
	d_epot_dihedral[dihedral_index] = epot; 
      }
      else if ( energy == 1) {
	
	float t1 = cA;
	float t2 = c11*c23 - c12*c13;
	float t3 = -cB1;
	float t4 = cB2;
	float t5 = c13*c23 - c12*c33;
	float t6 = -cA;
	float cR1 = c12/c22; 
	float cR2 = c23/c22;
	
	float ff = -(p[1]+(2.f*p[2]+(3.f*p[3]+(4.f*p[4] + 5.f*p[5]*cc)*cc)*cc)*cc);
	
	ff *= c22;
	
	f1 = ff*(t1*dr1.x + t2*dr2.x + t3*dr3.x)/(cD*cB1);
	f2 = ff*(t4*dr1.x + t5*dr2.x + t6*dr3.x)/(cD*cB2);

	
	atomicFloatAdd(&(f[a].x), f1);
	atomicFloatAdd(&(f[b].x), -(1.f+cR1)*f1 + cR2*f2);
	atomicFloatAdd(&(f[c].x), cR1*f1 - (1.f + cR2)*f2);
	atomicFloatAdd(&(f[d].x), f2);
	
	f1 = ff*(t1*dr1.y + t2*dr2.y + t3*dr3.y)/(cD*cB1);
	f2 = ff*(t4*dr1.y + t5*dr2.y + t6*dr3.y)/(cD*cB2);

	atomicFloatAdd(&(f[a].y), f1);
	atomicFloatAdd(&(f[b].y), -(1.f+cR1)*f1 + cR2*f2);
	atomicFloatAdd(&(f[c].y), cR1*f1 - (1.f + cR2)*f2);
	atomicFloatAdd(&(f[d].y), f2);
	
	f1 = ff*(t1*dr1.z + t2*dr2.z + t3*dr3.z)/(cD*cB1);
	f2 = ff*(t4*dr1.z + t5*dr2.z + t6*dr3.z)/(cD*cB2);

	atomicFloatAdd(&(f[a].z), f1);
	atomicFloatAdd(&(f[b].z), -(1.f+cR1)*f1 + cR2*f2);
	atomicFloatAdd(&(f[c].z), cR1*f1 - (1.f + cR2)*f2);
	atomicFloatAdd(&(f[d].z), f2);
	
	// Potential energy
	atomicFloatAdd(&(f[a].w), epot);
      }
    }

  }

}



///////////////////////////////////////////////////////
// Periodic dihedral potential
///////////////////////////////////////////////////////

PeriodicDihedral::PeriodicDihedral() :  DihedralPotential() {
  ID_String = "PeriodicDihedral";
  num_dihedral_params = 4;
}

void PeriodicDihedral::CalcF(bool initialize, bool __attribute__((unused))calc_stresses){

  if ( initialize )
    DihedralResetForces<<<particleData->GetNumberOfParticles(),1>>>(particleData->d_f, particleData->d_w);

  RectangularSimulationBox* RSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  LeesEdwardsSimulationBox* LESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);

  MoleculeData* M = sample->GetMoleculeData();
  unsigned num_dihedrals = M->GetNumberOfDihedrals();
  unsigned num_blocks = num_dihedrals/num_threads + 1;

  if ( LESB )
    PeriodicDih<1><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_dlist, M->d_dtype, M->d_dplist, 
	M->d_epot_dihedral, M->d_dihedrals, 
	num_dihedrals, LESB, LESB->GetDevicePointer() );
  else if ( RSB )
    PeriodicDih<1><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_dlist, M->d_dtype, M->d_dplist, 
	M->d_epot_dihedral, M->d_dihedrals, 
	num_dihedrals, RSB, RSB->GetDevicePointer() );
  else
    throw RUMD_Error("PeriodicDihedral","CalcF","unknown simulation box");

}

double PeriodicDihedral::GetPotentialEnergy(){

  RectangularSimulationBox* RSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  LeesEdwardsSimulationBox* LESB = dynamic_cast<LeesEdwardsSimulationBox*>(simBox);

  MoleculeData* M = sample->GetMoleculeData();
  unsigned num_dihedrals = M->GetNumberOfDihedrals();
  unsigned num_blocks = num_dihedrals/num_threads + 1;

  if ( LESB )
    PeriodicDih<2><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_dlist, M->d_dtype, M->d_dplist, 
	M->d_epot_dihedral, M->d_dihedrals, 
	num_dihedrals, LESB, LESB->GetDevicePointer() );
  else if ( RSB )
    PeriodicDih<2><<<num_blocks, num_threads>>>
      ( particleData->d_r, particleData->d_f,
	M->d_dlist, M->d_dtype, M->d_dplist, 
	M->d_epot_dihedral, M->d_dihedrals, 
	num_dihedrals, RSB, RSB->GetDevicePointer() );
  else
    throw RUMD_Error("PeriodicDihedral","GetPotentialEnergy","unknown simulation box");
  
  // copy to host
  size_t nbytes = num_dihedrals*sizeof(float);
  cudaMemcpy(M->h_epot_dihedral, M->d_epot_dihedral, nbytes, cudaMemcpyDeviceToHost);
  
  // sum over bonds
  double epot = 0.0;
  for ( unsigned n=0; n<num_dihedrals; n++ )  
    epot += (double)M->h_epot_dihedral[n];

  return epot;
}


template <int energy, class Simbox> 
__global__ void PeriodicDih( float4 *r, float4 *f, 
			  uint4 *dlist, uint1 *dtype, float *plist, 
			  float *d_epot_dihedral, float *d_dihedrals,
			  unsigned num_dihedrals, Simbox *simbox, float *simboxpointer ){

  // First approach is to let one thread work on a dihedral
  float4 dr1, dr2, dr3;    
  float f1, f2, p[6];

  
  unsigned dihedral_index = blockIdx.x*blockDim.x + threadIdx.x;

  if ( dihedral_index < num_dihedrals ){
    unsigned dihedral_type = dtype[dihedral_index].x;

    unsigned a = dlist[dihedral_index].x;
    unsigned b = dlist[dihedral_index].y;
    unsigned c = dlist[dihedral_index].z;
    unsigned d = dlist[dihedral_index].w;

    unsigned offset = 6*dihedral_type;
    for ( int k=0; k<6; k++ ) p[k] = plist[offset+k];
  
    // Load the simulation box in local memory to avoid bank conflicts.
    float array[simulationBoxSize];
    simbox->loadMemory(array, simboxpointer);
  
    dr1 = simbox->calculateDistance(r[b], r[a], array);
    dr2 = simbox->calculateDistance(r[c], r[b], array);
    dr3 = simbox->calculateDistance(r[d], r[c], array);

    float c11 = dr1.w;
    float c12 = dr1.x*dr2.x + dr1.y*dr2.y + dr1.z*dr2.z;
    float c13 = dr1.x*dr3.x + dr1.y*dr3.y + dr1.z*dr3.z;
    float c22 = dr2.w;
    float c23 = dr2.x*dr3.x + dr2.y*dr3.y + dr2.z*dr3.z;
    float c33 = dr3.w;

    float cA = c13*c22 - c12*c23;
    float cB1 = c11*c22 - c12*c12;
    float cB2 = c22*c33 - c23*c23;

    if ( fabs(c23*c23/(c22*c33) - 1) < DIHEDRAL_EPS ||
         fabs(c12*c12/(c11*c22) - 1) < DIHEDRAL_EPS ){

      d_dihedrals[dihedral_index] = 0.0;
    //  return;
    } else {

      float cD = sqrt(cB1*cB2);

      float cc = cA/cD; //cos(phi)
      float ceq = cos(p[0]); //cos(phiEq)
      float seq = sin(p[0]); //in(phi)
      float ss = sqrt(fabs(1.f-cc*cc)); //sin(phiEq)
      float cceq = cc*ceq+ss*seq; //cos(phi-phiEq)
      float sseq = ss*ceq-cc*seq; //sin(phi-phiEq)

      //if (cc - 1.f > 0.0) cc = 1.f;
      //if (cc + 1.f < 0.0) cc = -1.f;
      //float phi = acos(cc);
  
      d_dihedrals[dihedral_index] = acos(cc);

      //float epot = p[0] + (p[1]+(p[2]+(p[3]+(p[4]+p[5]*cc)*cc)*cc)*cc)*cc;
      //float epot = p[1]/2.f*(1.f-cos(phi-p[0])) + p[2]/2.f*(1.f-cos(2.f*(phi-p[0]))) + p[3]/2.f*(1.f-cos(3.f*(phi-p[0])));
      float epot = 0.5*(p[1]*(1.f-cceq) + p[2]*(1.f-cceq*cceq+sseq*sseq) + p[3]*(1.f+3.f*cceq-4.f*cceq*cceq*cceq));

      if ( energy == 2 ){
        d_epot_dihedral[dihedral_index] = epot; 
      }
      else if ( energy == 1) {

        float t1 = cA;
        float t2 = c11*c23 - c12*c13;
        float t3 = -cB1;
        float t4 = cB2;
        float t5 = c13*c23 - c12*c33;
        float t6 = -cA;
        float cR1 = c12/c22; 
        float cR2 = c23/c22;


        //float ff = -(p[1]+(2.f*p[2]+(3.f*p[3]+(4.f*p[4] + 5.f*p[5]*cc)*cc)*cc)*cc);
        //float ff = p[1]/2.f*sin(phi-p[0]) + p[2]*sin(2.f*(phi-p[0])) + p[3]*1.5f*sin(3.f*(phi-p[0]));
        float ff = -0.5*p[1]*sseq - 2.f*p[2]*sseq*cceq - 1.5f*p[3]*(3.f*sseq-4.f*sseq*sseq*sseq);
        
        
        ff *= c22;

        f1 = ff*(t1*dr1.x + t2*dr2.x + t3*dr3.x)/(cD*cB1);
        f2 = ff*(t4*dr1.x + t5*dr2.x + t6*dr3.x)/(cD*cB2);

        atomicFloatAdd(&(f[a].x), f1);
        atomicFloatAdd(&(f[b].x), -(1.f+cR1)*f1 + cR2*f2);
        atomicFloatAdd(&(f[c].x), cR1*f1 - (1.f + cR2)*f2);
        atomicFloatAdd(&(f[d].x), f2);
      
        f1 = ff*(t1*dr1.y + t2*dr2.y + t3*dr3.y)/(cD*cB1);
        f2 = ff*(t4*dr1.y + t5*dr2.y + t6*dr3.y)/(cD*cB2);

        atomicFloatAdd(&(f[a].y), f1);
        atomicFloatAdd(&(f[b].y), -(1.f+cR1)*f1 + cR2*f2);
        atomicFloatAdd(&(f[c].y), cR1*f1 - (1.f + cR2)*f2);
        atomicFloatAdd(&(f[d].y), f2);
      
        f1 = ff*(t1*dr1.z + t2*dr2.z + t3*dr3.z)/(cD*cB1);
        f2 = ff*(t4*dr1.z + t5*dr2.z + t6*dr3.z)/(cD*cB2);

        atomicFloatAdd(&(f[a].z), f1);
        atomicFloatAdd(&(f[b].z), -(1.f+cR1)*f1 + cR2*f2);
        atomicFloatAdd(&(f[c].z), cR1*f1 - (1.f + cR2)*f2);
        atomicFloatAdd(&(f[d].z), f2);
	
        // Potential energy
	atomicFloatAdd(&(f[a].w), epot);
        }
    }
  }
}


