#ifndef SIMULATIONBOX_H
#define SIMULATIONBOX_H

/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/RUMD_Error.h"
#include "rumd/rumd_base.h"
#include <cmath>
#include <vector>

// The abstract Simulation Box class. Inherit this class to build your own SimulationBox.
class SimulationBox{
 
 private:
  SimulationBox(const SimulationBox&); 
  SimulationBox& operator=(const SimulationBox&); 
  
 protected:
  float* h_boxDetails;
  float* d_boxDetails;
  
 public:
  SimulationBox() : h_boxDetails(0), d_boxDetails(0){};
  virtual ~SimulationBox(){};
  
  __host__ __device__ float4 calculateDistance( float4 position1, float4 position2, float* simBoxPtr );
  __host__ __device__ float4 calculateDistanceWithImages( float4 position1, float4 position2, float4 image1, float4 image2, float* simBoxPtr );
  __host__ __device__ float4 applyBoundaryCondition( float4& position, float* simBoxPtr );
  __host__ __device__ bool detectBoundingBoxOverlap( float4 Rc1, float4 dR1, float4 Rc2, float4 dR2, float* simBoxPtr);
  __host__ __device__ void loadMemory( float* toMem, float* fromMem );

  virtual void ScaleBox( float factor ) = 0;
  virtual void ScaleBoxFraction( double epsilon ) = 0;
  virtual void CopyAnotherBox(const SimulationBox* box_to_copy) = 0;
  virtual SimulationBox* MakeCopy() = 0;
  virtual void ScaleBoxDirection( float factor, int dir ) = 0;

  // Set methods.
  virtual void SetBoxParameters( std::vector<float>& parameterList ) = 0;

  // Get methods.
  virtual float GetVolume() const { return -1.f; };
  virtual float GetLength(int dir) const { return -1.f*dir; };
  virtual float4 GetSimulationBox() const { float4 value = {0,0,0,0}; return value; };

  virtual float* GetHostPointer(){
    if(!h_boxDetails)
      throw( RUMD_Error("SimulationBox","SimulationBox","Host pointer is invalid") );
    else
      return h_boxDetails;
  }

  virtual float* GetDevicePointer(){
    if(!d_boxDetails)
      throw( RUMD_Error("SimulationBox","SimulationBox","Device pointer is invalid") );
    else
      return d_boxDetails;
  }

  virtual std::string GetInfoString(unsigned int precision) const = 0;
};

///////////////////////////////////////////////////
// Avaliable simulation boxes
////////////////////////////////////////////////////

class BaseRectangularSimulationBox : public SimulationBox {
private:
  void AllocateBoxDetails();
protected:
  void CopyBoxDetailsToGPU();

public:
  BaseRectangularSimulationBox();
  BaseRectangularSimulationBox( float Lx, float Ly, float Lz );
  
  ~BaseRectangularSimulationBox();

  void CopyAnotherBox(const SimulationBox* box_to_copy);  
  void SetBoxParameters( std::vector<float>& parameterList );
  void ScaleBox( float factor );
  void ScaleBoxFraction( double epsilon );
  void ScaleBoxDirection( float factor, int dir );
  float GetVolume() const { return h_boxDetails[1] * h_boxDetails[2] * h_boxDetails[3]; }
  float GetLength(int dir) const { return h_boxDetails[dir+1]; }
  float4 GetSimulationBox() const { float4 value = { h_boxDetails[1], h_boxDetails[2], h_boxDetails[3], GetVolume() };  return value; }
};


// A rectangular simulation box that assumes periodic boundaries.
class RectangularSimulationBox : public BaseRectangularSimulationBox{

private:
  RectangularSimulationBox(const RectangularSimulationBox&);
  RectangularSimulationBox& operator=(const RectangularSimulationBox&);

public:
  RectangularSimulationBox();
  RectangularSimulationBox( float Lx, float Ly, float Lz );
  ~RectangularSimulationBox() {}

  SimulationBox* MakeCopy();
  
  __host__ __device__ float4 calculateDistance( float4 position1, float4 position2, float* simBoxPtr ){
    float dr_x = position1.x - position2.x;
    float dr_y = position1.y - position2.y;
    float dr_z = position1.z - position2.z;

    if(periodicInX)
      dr_x -= simBoxPtr[1] * rintf( dr_x * simBoxPtr[4] );
    if(periodicInY)
      dr_y -= simBoxPtr[2] * rintf( dr_y * simBoxPtr[5] );
    if(periodicInZ)
      dr_z -= simBoxPtr[3] * rintf( dr_z * simBoxPtr[6] );
    
    float4 value = { dr_x, dr_y, dr_z, dr_x * dr_x + dr_y * dr_y + dr_z * dr_z }; 
    return value;
  }

  __host__ __device__ float4 calculateDistanceMoved( float4 position, float4 last_position, float* simBoxPtr, float2*  ){
    return calculateDistance(position, last_position, simBoxPtr);
  }

  __host__ __device__ float4 calculateDistanceWithImages( float4 position1, float4 position2, float4 image1, float4 image2, float* simBoxPtr ){
    float dr_x = (position1.x - position2.x) + simBoxPtr[1] * ( image1.x - image2.x );
    float dr_y = (position1.y - position2.y) + simBoxPtr[2] * ( image1.y - image2.y );
    float dr_z = (position1.z - position2.z) + simBoxPtr[3] * ( image1.z - image2.z );
    
    float4 value = { dr_x, dr_y, dr_z, dr_x * dr_x + dr_y * dr_y + dr_z * dr_z }; 
    return value;
  }
  
  __host__ __device__ float4 applyBoundaryCondition( float4& position, float* simBoxPtr ){
    float4 image = { rintf( position.x * simBoxPtr[4] ), 
		     rintf( position.y * simBoxPtr[5] ), 
		     rintf( position.z * simBoxPtr[6] ), 
		     0.f };
    
    // Periodic boundary condition.
    if(periodicInX)
      position.x -= simBoxPtr[1] * image.x;
    if(periodicInY)
      position.y -= simBoxPtr[2] * image.y;
    if(periodicInZ)
      position.z -= simBoxPtr[3] * image.z;
    
    return image; 
  }


  __host__ __device__ bool detectBoundingBoxOverlap( float4 Rc1, float4 dR1, float4 Rc2, float4 dR2, float* simBoxPtr) {

    float4 Dist = calculateDistance(Rc1, Rc2, simBoxPtr);
    return ( fabs(Dist.x)<(dR1.x + dR2.x) && fabs(Dist.y)<(dR1.y + dR2.y)  && fabs(Dist.z)<(dR1.z + dR2.z) );
      
  }
  
  __host__ __device__ void loadMemory( float* toMem, float* fromMem ){
    if((unsigned) fromMem[0]){
      toMem[1] = fromMem[1]; // boxLength
      toMem[2] = toMem[1]; 
      toMem[3] = toMem[1];
      toMem[4] = 1.f / toMem[1]; // 1 / boxLength
      toMem[5] = toMem[4];
      toMem[6] = toMem[4];
    }
    else{
      for(unsigned int i=0; i < simulationBoxSize; i++)
	toMem[i] = fromMem[i];
    }
  }
  
  // Get methods

  std::string GetInfoString(unsigned int precision) const;
};


////////////////////////////////////////////////////////////////
// A rectangular simulation box that assumes periodic boundaries but with 
// Lees-Edwards boundary conditions to allow relative shear of the box and
// its images.
////////////////////////////////////////////////////////////////

class LeesEdwardsSimulationBox : public BaseRectangularSimulationBox{

 private:
  LeesEdwardsSimulationBox(const LeesEdwardsSimulationBox&);
  LeesEdwardsSimulationBox& operator=(const LeesEdwardsSimulationBox&);
  
  double box_shift_dbl;
  
 public:
  LeesEdwardsSimulationBox();
  LeesEdwardsSimulationBox( float Lx, float Ly, float Lz, float boxShift );
  LeesEdwardsSimulationBox(const SimulationBox* box_to_copy);
  ~LeesEdwardsSimulationBox() {}

  void SetBoxParameters( std::vector<float>& parameterList );

  SimulationBox* MakeCopy();
  void CopyAnotherBox(const SimulationBox* box_to_copy);

  __host__ __device__ float4 calculateDistance( float4 position1, float4 position2, float* simBoxPtr ){
    float dr_x = position1.x - position2.x;
    float dr_y = position1.y - position2.y;
    float dr_z = position1.z - position2.z;

    float y_wrap = rintf( dr_y * simBoxPtr[5] );

    dr_y -= simBoxPtr[2] * y_wrap;
    dr_x -= y_wrap * simBoxPtr[7];

    dr_x -= simBoxPtr[1] * rintf( dr_x * simBoxPtr[4] );
    dr_z -= simBoxPtr[3] * rintf( dr_z * simBoxPtr[6] );
    
    float4 value = { dr_x, dr_y, dr_z, dr_x * dr_x + dr_y * dr_y + dr_z * dr_z }; 
    return value;
  }

  __host__ __device__ float4 calculateDistanceMoved( float4 position, float4 last_position, float* simBoxPtr, float2* dStrain ){
    // this function calculates the (square of the) non-affine displacment
    float deltaStrain = dStrain[0].x;
    float dr_x = position.x - last_position.x;
    float dr_y = position.y - last_position.y;
    float dr_z = position.z - last_position.z;

    // some of these will be reused
    float Lx = simBoxPtr[1];
    float Ly = simBoxPtr[2];
    float boxShift = simBoxPtr[7];
    float Lx_inv = simBoxPtr[4];
    float Ly_inv = simBoxPtr[5];

    float y_wrap = rintf( dr_y * Ly_inv );


    dr_y -= Ly * y_wrap;
    dr_z -= simBoxPtr[3] * rintf( dr_z * simBoxPtr[6] );
    // x-component : subtract off the affine part
    dr_x -= ( y_wrap * boxShift + (y_wrap*Ly+position.y) * deltaStrain );
    dr_x -= Lx * rintf( dr_x * Lx_inv);

    
    float4 value = { dr_x, dr_y, dr_z, dr_x * dr_x + dr_y * dr_y + dr_z * dr_z }; 
    return value;
  }
  
  __host__ __device__ float4 calculateDistanceWithImages( float4 position1, float4 position2, float4 image1, float4 image2, float* simBoxPtr ){
    
    float dr_x = (position1.x - position2.x) + simBoxPtr[1] * ( image1.x - image2.x ) + simBoxPtr[7] * ( image1.y - image2.y ) ;
    
    float dr_y = (position1.y - position2.y) + simBoxPtr[2] * ( image1.y - image2.y );
    float dr_z = (position1.z - position2.z) + simBoxPtr[3] * ( image1.z - image2.z );
    
    float4 value = { dr_x, dr_y, dr_z, dr_x * dr_x + dr_y * dr_y + dr_z * dr_z }; 
    return value;
  }
  
  __host__ __device__ float4 applyBoundaryCondition( float4& position, float* simBoxPtr ){
    float y_wrap = rintf( position.y * simBoxPtr[5] );
    position.x -= simBoxPtr[7] * y_wrap;

    float4 image = { rintf( position.x * simBoxPtr[4] ), 
		     y_wrap,
		     rintf( position.z * simBoxPtr[6] ), 
		     0.f };
    
    // Periodic boundary condition.
    position.x -= simBoxPtr[1] * image.x;
    position.y -= simBoxPtr[2] * image.y;
    position.z -= simBoxPtr[3] * image.z;
    
    return image; 
  }


 __host__ __device__ bool detectBoundingBoxOverlap( float4 Rc1, float4 dR1, float4 Rc2, float4 dR2, float* simBoxPtr ) {
   // check z overlap first, if there's none then we're done
   // [for X and XY sorting there will almost always be a z-overlap]
   float dr_z = Rc1.z - Rc2.z;
   dr_z -= simBoxPtr[3] * rintf( dr_z * simBoxPtr[6] );
   if( fabs(dr_z) > (dR1.z + dR2.z) )
     return false;

   // if have z-overlap, then we check y and x in that order, but
   // for y need to keep track of the different wrap-possibilities 
   // which determine which shifts in the x-direction are relevant

   // have two main cases to deal with, depending on whether summed box height
   // is less than length in the y-direction [note that dR is half of the height
   // of the box]  or not
   float sum_dR_x =  dR1.x + dR2.x;
   float sum_dR_y =  dR1.y + dR2.y;
   float dRc_y = Rc1.y - Rc2.y;
   float dRc_x_noshift = Rc1.x - Rc2.x;

   if(sum_dR_y < simBoxPtr[2]/2.) {
     // (strictly) less than : only overlap in one direction possible
     // The easy case:  minimum image convention applies
     float y_wrap = rintf( dRc_y * simBoxPtr[5] );
     dRc_y -= simBoxPtr[2] * y_wrap;
     
     if( fabs(dRc_y) > sum_dR_y ) // no y-overlap
       return false;
     else {
       // Check x overlap with this single y-wrap
       float dRc_x_shift_wrap = dRc_x_noshift - y_wrap * simBoxPtr[7];
       dRc_x_shift_wrap -= simBoxPtr[1] * rintf( dRc_x_shift_wrap * simBoxPtr[4] );
       return fabs(dRc_x_shift_wrap) < sum_dR_x;
     }
       
   }
   else {
     // cannot assume minimum image convention. Try this image, and
     // images above and below

     // start with y_wrap = 0 (this image)
     
     if(  (fabs(dRc_y) < sum_dR_y) &&
	  ( fabs(dRc_x_noshift - simBoxPtr[1] * rintf( dRc_x_noshift * simBoxPtr[4] )) < sum_dR_x) )
       return true;
     
     if( fabs(dRc_y - simBoxPtr[2]) < sum_dR_y ) {
       // corresponds to y_wrap = 1.
       float dRc_x_shift = dRc_x_noshift - simBoxPtr[7];
       dRc_x_shift -= simBoxPtr[1] * rintf( dRc_x_shift * simBoxPtr[4] );   
       if(fabs(dRc_x_shift) < sum_dR_x)
	 return true;
     }
     
     if( fabs(dRc_y + simBoxPtr[2]) < sum_dR_y ) {
       // y_wrap = -1.
       float dRc_x_neg_shift = dRc_x_noshift + simBoxPtr[7];
       dRc_x_neg_shift -= simBoxPtr[1] * rintf( dRc_x_neg_shift * simBoxPtr[4] );
       if( fabs(dRc_x_neg_shift) < sum_dR_x )
	 return true;
     }
     // after checking all possibilities, arriving here means no overlaps
     return false;
   } // else [sum_dR_y]
 }
  


  
  __host__ __device__ void loadMemory( float* toMem, float* fromMem ){
    if((unsigned) fromMem[0]){
      toMem[1] = fromMem[1]; // boxLength
      toMem[2] = toMem[1]; 
      toMem[3] = toMem[1];
      toMem[4] = 1.f / toMem[1]; // 1 / boxLength
      toMem[5] = toMem[4];
      toMem[6] = toMem[4];
      toMem[7] = fromMem[7];
    }
    else{
      for(unsigned int i=0; i < simulationBoxSize; i++)
	toMem[i] = fromMem[i];
    }
  }

  // Set methods
  void ScaleBox( float factor );
  float IncrementBoxStrain(double inc_boxStrain);
  void SetBoxShift(double set_boxShift);
  void ScaleBoxDirection( float factor, int dir );


  // Get methods
  double GetBoxShift() const { return box_shift_dbl; }
  //float GetLength(int dir) const { return h_boxDetails[dir+1]; }

  std::string GetInfoString(unsigned int precision) const;
};

class SimulationBoxFactory{
 public:
  SimulationBoxFactory(){}
  virtual ~SimulationBoxFactory() {};
  SimulationBox* CreateSimulationBox(const std::string& infoStr);
 private:
};

#endif // SIMULATIONBOX_H
