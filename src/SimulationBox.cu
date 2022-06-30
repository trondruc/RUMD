/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/RUMD_Error.h"
#include "rumd/SimulationBox.h"
#include "rumd/ParseInfoString.h"

#include <iostream>
#include <sstream>
#include <iomanip>

const std::string LE_Error_Code1("simulationBoxSize is too small; it must be at least 8. Change in rumd_base.h and recompile");

const unsigned int min_simulationBoxSize_LE = 8;

const float cubic_tol = 1.e-8;


//////////////////////////////////////////////////////////
// Base-Rectangular Simulation Box implementation
//////////////////////////////////////////////////////////



void BaseRectangularSimulationBox::CopyBoxDetailsToGPU(){

  // Check whether cubic according to tolerance cubic_tol
  if( fabs(h_boxDetails[1]/h_boxDetails[2] - 1.) < cubic_tol && fabs(h_boxDetails[2]/h_boxDetails[3] - 1.) < cubic_tol )
    h_boxDetails[0] = 1;
  else
    h_boxDetails[0] = 0;
  
  cudaMemcpy( d_boxDetails, h_boxDetails, simulationBoxSize * sizeof(float), cudaMemcpyHostToDevice );
  
  if( cudaDeviceSynchronize() != cudaSuccess ) 
    throw( RUMD_Error("RectangularSimulationBox", __func__,"CudaMemcpyHostToDevice failed on h_boxDetails => d_boxDetails") );
}

BaseRectangularSimulationBox::BaseRectangularSimulationBox() : SimulationBox() {
  AllocateBoxDetails();
  float L = 20.f;

  h_boxDetails[1] = h_boxDetails[2] = h_boxDetails[3] = L; 
  h_boxDetails[4] = h_boxDetails[5] = h_boxDetails[6] = 1.f / L;
  
  CopyBoxDetailsToGPU();
}

BaseRectangularSimulationBox::BaseRectangularSimulationBox( float Lx, float Ly, float Lz ) : SimulationBox() {
  AllocateBoxDetails();

  h_boxDetails[1] = Lx; 
  h_boxDetails[2] = Ly; 
  h_boxDetails[3] = Lz;
  h_boxDetails[4] = 1.f / Lx; 
  h_boxDetails[5] = 1.f / Ly; 
  h_boxDetails[6] = 1.f / Lz;
  
  CopyBoxDetailsToGPU();
}

BaseRectangularSimulationBox::~BaseRectangularSimulationBox(){
  cudaFreeHost(h_boxDetails);
  cudaFree(d_boxDetails);
}


void BaseRectangularSimulationBox::AllocateBoxDetails(){
  // CPU.
  if( cudaMallocHost( (void**) &h_boxDetails, simulationBoxSize * sizeof(float) ) != cudaSuccess )
    throw( RUMD_Error("RectangularSimulationBox", __func__, "Malloc failed on h_boxDetails" ) );
  
  // GPU.
  if( cudaMalloc( (void**) &d_boxDetails, simulationBoxSize * sizeof(float) ) != cudaSuccess )
    throw( RUMD_Error("RectangularSimulationBox", __func__,"Malloc failed on d_BoxDetails") );
}


void BaseRectangularSimulationBox::ScaleBox( float factor ){
// new_L += epsilon*L, and all double
  h_boxDetails[1] *= factor;
  h_boxDetails[2] *= factor;
  h_boxDetails[3] *= factor;
  h_boxDetails[4] = 1.f / h_boxDetails[1];
  h_boxDetails[5] = 1.f / h_boxDetails[2];
  h_boxDetails[6] = 1.f / h_boxDetails[3];
  CopyBoxDetailsToGPU();
}

void BaseRectangularSimulationBox::ScaleBoxFraction( double epsilon ){
  // L += epsilon*L, and all double
  h_boxDetails[1] += epsilon*h_boxDetails[1];
  h_boxDetails[2] += epsilon*h_boxDetails[2];
  h_boxDetails[3] += epsilon*h_boxDetails[3];
  h_boxDetails[4] = 1.f / h_boxDetails[1];
  h_boxDetails[5] = 1.f / h_boxDetails[2];
  h_boxDetails[6] = 1.f / h_boxDetails[3];
  CopyBoxDetailsToGPU();
}

void BaseRectangularSimulationBox::ScaleBoxDirection( float factor, int dir ){
  h_boxDetails[dir+1] *= factor;
  h_boxDetails[dir+4] = 1.f / h_boxDetails[dir+1];

  CopyBoxDetailsToGPU();  
}

void BaseRectangularSimulationBox::SetBoxParameters(std::vector<float>& parameterList) {
  if (parameterList.size() <  3)
    throw RUMD_Error("BaseRectangularSimulationBox", __func__,"Not enough parameters passed");
  for(unsigned i=0; i < 3; i++) {
    if(parameterList[i] < 0.f)
      throw RUMD_Error("BaseRectangularSimulationBox", __func__,"Negative length passed");
      h_boxDetails[i+1] = parameterList[i];
  }
  h_boxDetails[4] = 1.f / h_boxDetails[1]; 
  h_boxDetails[5] = 1.f / h_boxDetails[2]; 
  h_boxDetails[6] = 1.f / h_boxDetails[3];

  CopyBoxDetailsToGPU();
}


void BaseRectangularSimulationBox::CopyAnotherBox(const SimulationBox* box_to_copy){
  const BaseRectangularSimulationBox* rect_box_to_copy = dynamic_cast<const BaseRectangularSimulationBox*> (box_to_copy);
  if(rect_box_to_copy){
    // works also if copying a LeesEdwards box, though boxShift is discarded
    float4 box_data = rect_box_to_copy->GetSimulationBox();
    float Lx = box_data.x;
    float Ly = box_data.y;
    float Lz = box_data.z;
    h_boxDetails[1] = Lx; 
    h_boxDetails[2] = Ly; 
    h_boxDetails[3] = Lz;
    h_boxDetails[4] = 1.f / Lx; 
    h_boxDetails[5] = 1.f / Ly; 
    h_boxDetails[6] = 1.f / Lz;
    
    CopyBoxDetailsToGPU();
  }
  else 
    throw RUMD_Error("RectangularSimulationBox", __func__, "Tried to copy box of a different type");
}


//////////////////////////////////////////////////////////
// Rectangular Simulation Box implementation
//////////////////////////////////////////////////////////

RectangularSimulationBox::RectangularSimulationBox() : BaseRectangularSimulationBox() {

}

RectangularSimulationBox::RectangularSimulationBox( float Lx, float Ly, float Lz ) : BaseRectangularSimulationBox( Lx, Ly, Lz ) {

}

SimulationBox* RectangularSimulationBox::MakeCopy(){
  SimulationBox* newBox = new RectangularSimulationBox( h_boxDetails[1], h_boxDetails[2], h_boxDetails[3] );
  return newBox;
}


std::string RectangularSimulationBox::GetInfoString(unsigned int precision) const {
  std::ostringstream infoStream;
  infoStream << "RectangularSimulationBox";
  infoStream << ","  << std::setprecision(precision) << h_boxDetails[1];
  infoStream << ","  << std::setprecision(precision) << h_boxDetails[2];
  infoStream << ","  << std::setprecision(precision) << h_boxDetails[3];
  return infoStream.str();
}

//////////////////////////////////////////////////////////
// Lees Edwards Simulation Box for shear.
//////////////////////////////////////////////////////////

LeesEdwardsSimulationBox::LeesEdwardsSimulationBox() : BaseRectangularSimulationBox() {
  if(simulationBoxSize < min_simulationBoxSize_LE)
    throw RUMD_Error("LeesEdwardsSimulationBox", __func__,
		     LE_Error_Code1);
  
  // may need to make h_boxDetails double for very slow strain rates
  h_boxDetails[7] = 0.f;
  box_shift_dbl = 0.; 
  CopyBoxDetailsToGPU();
}

LeesEdwardsSimulationBox::LeesEdwardsSimulationBox( float Lx, float Ly, float Lz, float boxShift ) : BaseRectangularSimulationBox(Lx, Ly, Lz){

  if(simulationBoxSize < min_simulationBoxSize_LE)
    throw RUMD_Error("LeesEdwardsSimulationBox", __func__, LE_Error_Code1);

  box_shift_dbl = boxShift;  
  h_boxDetails[7] = box_shift_dbl;

  CopyBoxDetailsToGPU();
}

LeesEdwardsSimulationBox::LeesEdwardsSimulationBox(const SimulationBox* box_to_copy) : BaseRectangularSimulationBox() {
  CopyAnotherBox(box_to_copy);
}

SimulationBox* LeesEdwardsSimulationBox::MakeCopy(){
  SimulationBox* newBox = new LeesEdwardsSimulationBox( h_boxDetails[1], h_boxDetails[2], h_boxDetails[3], h_boxDetails[7] );
  return newBox;
}

void LeesEdwardsSimulationBox::CopyAnotherBox(const SimulationBox* box_to_copy){
  BaseRectangularSimulationBox::CopyAnotherBox(box_to_copy);
  const LeesEdwardsSimulationBox* LE_box_to_copy = dynamic_cast<const LeesEdwardsSimulationBox*> (box_to_copy);
  if(LE_box_to_copy)
    box_shift_dbl = LE_box_to_copy->GetBoxShift();
  else
    box_shift_dbl = 0.;

  h_boxDetails[7] = box_shift_dbl;
  CopyBoxDetailsToGPU();

}

void LeesEdwardsSimulationBox::ScaleBox( float factor )  {
  BaseRectangularSimulationBox::ScaleBox(factor);
  // do the calculation in double precision
  box_shift_dbl *= factor;
  h_boxDetails[7] = box_shift_dbl; 
  CopyBoxDetailsToGPU();  
}

void LeesEdwardsSimulationBox::ScaleBoxDirection( float factor, int dir ){
  BaseRectangularSimulationBox::ScaleBoxDirection(factor, dir);
  if(dir == 1) {
    box_shift_dbl *= factor;
    h_boxDetails[7] = box_shift_dbl; 
  }

  CopyBoxDetailsToGPU();  
}

void LeesEdwardsSimulationBox::SetBoxShift( double set_boxShift ) {
  box_shift_dbl = set_boxShift;
  h_boxDetails[7] = box_shift_dbl;

  CopyBoxDetailsToGPU();
}

float LeesEdwardsSimulationBox::IncrementBoxStrain( double inc_boxStrain ) {
  
  box_shift_dbl += inc_boxStrain * h_boxDetails[1];
  double wrap =  rint(box_shift_dbl/h_boxDetails[1]);
  box_shift_dbl -= h_boxDetails[1] * wrap;

  h_boxDetails[7] = box_shift_dbl;
  CopyBoxDetailsToGPU();
  return float(wrap);
}

void LeesEdwardsSimulationBox::SetBoxParameters(std::vector<float>& parameterList) {
  if (parameterList.size() != 4)
    throw RUMD_Error("LeesEdwardsSimulationBox", __func__,"Wrong number of parameters passed");

  BaseRectangularSimulationBox::SetBoxParameters(parameterList);

  h_boxDetails[7] = parameterList[3];
  box_shift_dbl = h_boxDetails[7];

  CopyBoxDetailsToGPU();

}

std::string LeesEdwardsSimulationBox::GetInfoString(unsigned int precision) const {
  std::ostringstream infoStream;
  infoStream << "LeesEdwardsSimulationBox";
  infoStream << ","  << std::setprecision(precision) << h_boxDetails[1];
  infoStream << ","  << std::setprecision(precision) << h_boxDetails[2];
  infoStream << ","  << std::setprecision(precision) << h_boxDetails[3];
  infoStream << ","  << std::setprecision(precision) << h_boxDetails[7];
  return infoStream.str();
}

SimulationBox* SimulationBoxFactory::CreateSimulationBox(const std::string& infoStr) {
  std::vector<float> parameterList;
  std::string simBoxType = ParseInfoString(infoStr, parameterList);

  SimulationBox* newSimBox = 0;
  if (simBoxType == "RectangularSimulationBox") 
    newSimBox = new RectangularSimulationBox();
  else if (simBoxType == "LeesEdwardsSimulationBox") 
    newSimBox = new LeesEdwardsSimulationBox();
  else
    throw RUMD_Error("SimulationBoxFactory", __func__, std::string("Unrecognized simulation box class ")+simBoxType);

  newSimBox->SetBoxParameters(parameterList);
  return newSimBox;
}
