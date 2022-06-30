#include "rumd/SquaredDisplacementCalculator.h"
#include "rumd/Sample.h"
#include "rumd/Potential.h"
#include "rumd/PairPotential.h"
#include "rumd/rumd_algorithms.h"


SquaredDisplacementCalculator::SquaredDisplacementCalculator(Sample* S, Sample* S_ref) : mainSample(S), num_allocated(0), d_r_unsrt(0), d_r_ref_unsrt(0), d_im_unsrt(0), d_im_ref_unsrt(0)
{
  
  Allocate(S->GetNumberOfParticles());

  // We copy the reference positions at the beginning and don't update them. But one could imagine working with two samples which are independently being evolved (ie undergoing dynamics) and comparing their separation e.g. to calculate Lyaponov exponents
  
  if(S_ref)
    S_ref->GetParticleData()->CopyPosImagesDevice(d_r_ref_unsrt, d_im_ref_unsrt);
  else
    S->GetParticleData()->CopyPosImagesDevice(d_r_ref_unsrt, d_im_ref_unsrt);
}


void SquaredDisplacementCalculator::Allocate(unsigned nParticles) {

  if(num_allocated && nParticles != num_allocated)
    Free();

  if( cudaMalloc( (void**) &d_sum_sq_images, nParticles * sizeof(int3) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("SquaredDisplacementCalculator", __func__, "Malloc failed on d_sum_sq_images") );


   if( cudaMalloc( (void**) &d_sum_dbl_comp, nParticles * sizeof(double4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("SquaredDisplacementCalculator", __func__, "Malloc failed on d_sum_dbl_comp") );


    if( cudaMalloc( (void**) &d_r_unsrt, nParticles * sizeof(float4) ) == cudaErrorMemoryAllocation )
    throw( RUMD_Error("SquaredDisplacementCalculator", __func__, "Malloc failed on d_r_unsrt") );

    if( cudaMalloc( (void**) &d_r_ref_unsrt, nParticles * sizeof(float4) ) == cudaErrorMemoryAllocation )
      throw( RUMD_Error("SquaredDisplacementCalculator", __func__, "Malloc failed on d_r_ref_unsrt") );
    
    
    if( cudaMalloc( (void**) &d_im_unsrt, nParticles * sizeof(float4) ) == cudaErrorMemoryAllocation )
      throw( RUMD_Error("SquaredDisplacementCalculator", __func__, "Malloc failed on d_im_unsrt") );
    
    
    if( cudaMalloc( (void**) &d_im_ref_unsrt, nParticles * sizeof(float4) ) == cudaErrorMemoryAllocation )
      throw( RUMD_Error("SquaredDisplacementCalculator", __func__, "Malloc failed on d_im_ref_unsrt") );

}


void SquaredDisplacementCalculator::Free() {
  cudaFree(d_sum_sq_images);
  cudaFree(d_sum_dbl_comp);

  cudaFree(d_r_unsrt);
  cudaFree(d_r_ref_unsrt);
  cudaFree(d_im_unsrt);
  cudaFree(d_im_ref_unsrt);
}

SquaredDisplacementCalculator::~SquaredDisplacementCalculator() {
  Free();
}

void SquaredDisplacementCalculator::GetDataInfo(std::map<std::string, bool> &active, std::map<std::string, std::string> &columnIDs) {
  active["sq_disp"] = true;
  columnIDs["sq_disp"] = "sq_disp";
}

void SquaredDisplacementCalculator::RemoveDataInfo(std::map<std::string, bool> &active, std::map<std::string, std::string> &columnIDs) {
  active.erase("sq_disp");
  columnIDs.erase("sq_disp");
}

void SquaredDisplacementCalculator::GetDataValues(std::map<std::string, float> &dataValues, std::map<std::string, bool> &active) {

  if(active["sq_disp"]) {
  
    // call kernel to compute differences in double precision. Keep track separately of integer (image) and float parts---more precisley image-image, imega-float, float-float
    unsigned nParticles = mainSample->GetNumberOfParticles();
    unsigned num_threads = 128;
    unsigned num_blocks = (nParticles + num_threads - 1)/num_threads;
    mainSample->GetParticleData()->CopyPosImagesDevice(d_r_unsrt, d_im_unsrt);
    
    
    squared_displacement_kernel<<<num_blocks, num_threads >>>(nParticles, d_im_ref_unsrt, d_r_ref_unsrt, d_im_unsrt, d_r_unsrt, d_sum_sq_images, d_sum_dbl_comp);
    
    // then sum up
    sumIdenticalArrays( d_sum_sq_images, nParticles, 1, 32 );
    sumIdenticalArrays( d_sum_dbl_comp, nParticles, 1, 32 );
    // Copy sums to host
    
    int3 sum_sq_images;
    double4 sum_dbl_comp;
    
    cudaMemcpy(&sum_sq_images, d_sum_sq_images, sizeof(int3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_dbl_comp, d_sum_dbl_comp, sizeof(double4), cudaMemcpyDeviceToHost);
    
    // form complete sum of squared displacements
    SimulationBox* simbox = mainSample->GetSimulationBox();
    RectangularSimulationBox* testRSB = dynamic_cast<RectangularSimulationBox*>(simbox);
    if(!testRSB)
      throw RUMD_Error("SquaredDisplacementCalculator", __func__, "Only implemented for RectangularSimulationBox");
    
    double Lx = simbox->GetLength(0), Ly = simbox->GetLength(1), Lz = simbox->GetLength(2);
    double Lx2 = Lx*Lx, Ly2 = Ly*Ly, Lz2 = Lz*Lz;

    dataValues["sq_disp"] = (sum_sq_images.x*Lx2 + sum_sq_images.y*Ly2 + sum_sq_images.z*Lz2) + (sum_dbl_comp.x*Lx + sum_dbl_comp.y*Ly + sum_dbl_comp.z*Lz) + sum_dbl_comp.w;
    }
} 

__global__ void squared_displacement_kernel(unsigned nParticles, float4* im1, float4* pos1, float4* im2, float4* pos2, int3* sum_sq_images, double4* sum_dbl_comp) {
  if(MyGP < nParticles) {
    float4 image1 = im1[MyGP];
    float4 image2 = im2[MyGP];
    float4 p1 = pos1[MyGP];
    float4 p2 = pos2[MyGP];

    int3 delta_im = {(int) (image1.x - image2.x),
		     (int) (image1.y - image2.y),
		     (int) (image1.z - image2.z)};
    int3 my_sum_sq_im = {delta_im.x*delta_im.x,
			 delta_im.y*delta_im.y,
			 delta_im.z*delta_im.z};
    double4 diff_pos = {p1.x - p2.x, p1.y - p2.y, p1.z - p2.z, 0.};
    double4 my_dbl_comp;
    my_dbl_comp.x = 2 * delta_im.x * diff_pos.x;
    my_dbl_comp.y = 2 * delta_im.y * diff_pos.y;
    my_dbl_comp.z = 2 * delta_im.z * diff_pos.z;
    my_dbl_comp.w = diff_pos.x*diff_pos.x + diff_pos.y*diff_pos.y + diff_pos.z*diff_pos.z;

    sum_sq_images[MyGP] = my_sum_sq_im;
    sum_dbl_comp[MyGP] = my_dbl_comp;
  }
}
