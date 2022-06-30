
/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/WallPotential.h"
#include "rumd/Sample.h"

///////////////////////////////////////////////////////
// Wall potential
///////////////////////////////////////////////////////

Wall_LJ_9_3::Wall_LJ_9_3(){
  SetID_String("FWall_LJ_9_3");
  wallOne = 1.f;
  wallTwo = 1.f;
  sigma1 = 1.f;
  epsilon1 = 1.f;
  sigma2 = 1.f;
  epsilon2 = 1.f;
  rhoWall = 0.6f;
}

void Wall_LJ_9_3::CalcF(bool initialize, bool __attribute__((unused))calc_stresses){
  RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(simBox);
  
  unsigned number = 16;
  unsigned num_blocks = ((particleData->GetNumberOfParticles())+number-1)/number;

  if ( initialize ){
    kernelWallLennardJones<1, 1, true><<< num_blocks, number >>>( particleData->d_r, particleData->d_f, particleData->d_w, particleData->d_sts, particleData->GetNumberOfParticles(), wallOne, wallTwo, 
								  sigma1, epsilon1, sigma2, epsilon2, rhoWall, scale, testRSB, testRSB->GetDevicePointer() );
  }
  else{
    kernelWallLennardJones<1, 1, false><<< num_blocks, number >>>( particleData->d_r, particleData->d_f, particleData->d_w, particleData->d_sts, particleData->GetNumberOfParticles(), wallOne, wallTwo, 
								   sigma1, epsilon1, sigma2, epsilon2, rhoWall, scale, testRSB, testRSB->GetDevicePointer() );
  }
}

void Wall_LJ_9_3::SetParams( float set_wallOne, float set_wallTwo, float set_sigma1, float set_epsilon1, 
			     float set_sigma2, float set_epsilon2, float set_rhoWall, float set_scale ){ 
  wallOne = set_wallOne; 
  wallTwo = set_wallTwo; 

  sigma1 = set_sigma1;
  epsilon1 = set_epsilon1;

  sigma2 = set_sigma2;
  epsilon2 = set_epsilon2;

  rhoWall = set_rhoWall;
  scale = set_scale;
}

void Wall_LJ_9_3::WritePotential(){}

void Wall_LJ_9_3::ScaleWalls( float factor ){ wallOne *= factor; wallTwo *= factor; }

double Wall_LJ_9_3::GetPotentialEnergy(){ return 0; }

///////////////////////////////////////////////////////
// Kernel implementations of wall potentials
///////////////////////////////////////////////////////

template <int stress, int energy, bool initialize, class Simbox> 
__global__ void kernelWallLennardJones( float4* position, float4* force, float4* virial, float4* my_stress, 
					unsigned numParticles, float wallOne, float wallTwo, float sigma1,
					float epsilon1, float sigma2, float epsilon2, float rho_wall, float scale, 
					Simbox* simBox, float* simBoxPointer ){
  
  if( MyGP < numParticles ){
    // Local variables
    float4 my_r = position[MyGP];
    float4 my_f = { 0, 0, 0, 0 };
    float4 my_w = { 0, 0, 0, 0 };
    float4 my_sts = { 0, 0, 0, 0 };
    unsigned type = __float_as_int(my_r.w);

    float sigma = 0; float epsilon = 0;

    if(type == 0){
      sigma = sigma1;
      epsilon = epsilon1;
    }
    
    if(type == 1){
      sigma = sigma2;
      epsilon = epsilon2;
    }

    // Load the simulation box in local memory to avoid bank conflicts.
    float array[simulationBoxSize];
    simBox->loadMemory(array, simBoxPointer);
    
    float pre_factor = scale * (4.f * (float) M_PI * epsilon * rho_wall * sigma * sigma * sigma) / 3.f; 
    
    float lengthOne = sqrtf( ( wallOne - my_r.z ) * ( wallOne - my_r.z ) );    
    float lengthTwo = sqrtf( ( my_r.z - wallTwo ) * ( my_r.z - wallTwo ) );
    
    // Wall one.
    float force1 = - (pre_factor/sigma) * ( (9.f/15.f) * powf( sigma / lengthOne, 10.f ) - (3.f/2.f) * powf( sigma / lengthOne, 4.f ) );
    my_f.z += force1;
    my_f.w += pre_factor * ( (1.f/15.f) * powf( sigma / lengthOne, 9.f ) - 0.5f * powf( sigma / lengthOne, 3.f ) );
    
    // Wall two.
    float force2 = (pre_factor/sigma) * ( (9.f/15.f) * powf( sigma / lengthTwo, 10.f ) - (3.f/2.f) * powf( sigma / lengthTwo, 4.f ) );
    my_f.z += force2;
    my_f.w += pre_factor * ( (1.f/15.f) * powf( sigma / lengthTwo, 9.f ) - 0.5f * powf( sigma / lengthTwo, 3.f ) );
    
    // Virial. Factor of 2 because summation in Sample corrects for double counting of pair virial.
    my_w.w += 2 * (force1 * (my_r.z - wallOne) + force2 * (my_r.z - wallTwo));
    
    // zz-component of the stress tensor.
    my_sts.z -= 2 * (force1 * (my_r.z - wallOne) + force2 * (my_r.z - wallTwo));
    
    if( initialize ){
      force[MyGP] = my_f;
      virial[MyGP] = my_w;
      my_stress[MyGP] = my_sts;
    }
    else{
      force[MyGP].z += my_f.z;
      force[MyGP].w += my_f.w;
      virial[MyGP].w += my_w.w;
      my_stress[MyGP].z = my_sts.z;
    }
  }
} 
