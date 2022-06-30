/*
    Copyright (C) 2010  Thomas Schrøder

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/
#include "rumd/Device.h"

#include <iostream>
#include <cstdlib>
#include <vector>
#include <sys/utsname.h>
#include <sstream>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include "rumd/RUMD_Error.h"



Device::Device() {
  cudaEventCreate( &event1 );
  cudaEventCreate( &event2 );
  cudaEventRecord( event1, 0);
  cudaEventSynchronize( event1 );
}

Device::~Device() {
// We used to do a cudaDeviceReset() at this point. Destruction of static objects
// happens too late, so the CUDA device may already be released

  cudaEventDestroy(event1);
  cudaEventDestroy(event2);


}

std::string Device::Report()
{
  int dev;
  if (cudaSuccess != cudaGetDevice(&dev))
    throw RUMD_Error("Device", __func__, "Cannot get cuda device");
  cudaDeviceProp deviceProp;
  cudaError err = cudaGetDeviceProperties(&deviceProp, dev);
  if (err != cudaSuccess)
    throw RUMD_Error("Device","Report",std::string("[cudaGetDeviceProperties] ")+cudaGetErrorString(err));

  struct utsname buf;
  uname(&buf); 
  std::ostringstream device_report;

  device_report
    << " node=" << buf.nodename
    << " device=" << dev
    << " name=\"" << deviceProp.name << "\"";

  return device_report.str();
}

const std::string Device::GetDeviceName() {
  int dev;
  if (cudaSuccess != cudaGetDevice(&dev))
    throw RUMD_Error("Device", __func__, "Cannot get cuda device");
  cudaDeviceProp deviceProp;
  cudaError err = cudaGetDeviceProperties(&deviceProp, dev);
  if (err != cudaSuccess)
    throw RUMD_Error("Device", __func__, std::string("[cudaGetDeviceProperties] ")+cudaGetErrorString(err));
  std::ostringstream outputStr;
  outputStr << deviceProp.name;
  return outputStr.str();
}


const std::string Device::GetDeviceReport() {  
  return GetDevice().Report();
}

Device& Device::GetDevice() {
  static Device instance;
  return instance;
}

unsigned Device::GetMaxThreadsPerBlock() {
  int dev;
  if (cudaSuccess != cudaGetDevice(&dev))
    throw RUMD_Error("Device", __func__, "Cannot get cuda device");

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  return deviceProp.maxThreadsPerBlock;
}

unsigned Device::GetMaximumGridDimensionX() {
  int dev;
  if (cudaSuccess != cudaGetDevice(&dev))
    throw RUMD_Error("Device", __func__, "Cannot get cuda device");
  int result;
  cudaDeviceGetAttribute(&result , cudaDevAttrMaxGridDimX, dev);
  return (unsigned) result;
}

unsigned Device::GetComputeCapability() {
  int dev;
  if (cudaSuccess != cudaGetDevice(&dev))
    throw RUMD_Error("Device", __func__, "Cannot get cuda device");
  cudaDeviceProp deviceProp;
  cudaError err = cudaGetDeviceProperties(&deviceProp, dev);
  if (err != cudaSuccess)
    throw RUMD_Error("Device", __func__, std::string("[cudaGetDeviceProperties] ")+cudaGetErrorString(err));
  
  unsigned cc = 100 * deviceProp.major + 10 * deviceProp.minor;
  return cc;
}

size_t Device::GetSharedMemPerBlock() {
  int dev;
  if (cudaSuccess != cudaGetDevice(&dev))
    throw RUMD_Error("Device", __func__, "Cannot get cuda device");
  cudaDeviceProp deviceProp;
  cudaError err = cudaGetDeviceProperties(&deviceProp, dev);
  if (err != cudaSuccess)
    throw RUMD_Error("Device", __func__, std::string("[cudaGetDeviceProperties] ")+cudaGetErrorString(err));
  return deviceProp.sharedMemPerBlock;
}


size_t Device::GetDeviceMemory() {
  int dev;
  if (cudaSuccess != cudaGetDevice(&dev))
    throw RUMD_Error("Device", __func__, "Cannot get cuda device");

  cudaDeviceProp deviceProp;
  cudaError err = cudaGetDeviceProperties(&deviceProp, dev);
  if (err != cudaSuccess)
    throw RUMD_Error("Device", __func__, std::string("[cudaGetDeviceProperties] ")+cudaGetErrorString(err));
  return deviceProp.totalGlobalMem;
}



void Device::Init()
{
  int deviceCount;
  cudaError err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess)
    throw RUMD_Error("Device","Init",std::string("[cudaGetDeviceCount]") + cudaGetErrorString(err));

  if (deviceCount == 0)
    throw RUMD_Error("Device","Init","error: no devices supporting CUDA.");

  if (deviceCount == 1) {
    // just one device or an emulated device present, no choice
    int dev=0;
    cudaDeviceProp deviceProp;
    cudaError err = cudaGetDeviceProperties(&deviceProp, dev);
    if (err != cudaSuccess)
      throw RUMD_Error("Device","Init",std::string("[cudaGetDeviceProperties]") + cudaGetErrorString(err));
    if (deviceProp.major < 1) 
      throw RUMD_Error("Device","Init","error: device does not support CUDA.");
    if ((deviceProp.major == 1) )// && (deviceProp.minor < 3))
      throw RUMD_Error("Device","Init","error: device compute capability >= 2.0 needed");
    cudaSetDevice(dev);
  }
  else {
    // several devices present, so make list of usable devices
    // and have one choosen among the currently available ones
    std::vector<int> usable_devices;
    for (int dev=0; dev<deviceCount; dev++) { 
      cudaDeviceProp deviceProp;
      cudaError err = cudaGetDeviceProperties(&deviceProp, dev);
      if (err != cudaSuccess)
	throw RUMD_Error("Device","Init",std::string("[cudaGetDeviceProperties] ") + cudaGetErrorString(err));
      if (((deviceProp.major > 1) 
	   //|| ((deviceProp.major == 1) && (deviceProp.minor >= 3))
	   ) && 
	(deviceProp.multiProcessorCount >= 2) &&
	 (deviceProp.computeMode != cudaComputeModeProhibited)) {
        usable_devices.push_back(dev); 
      }
    }
    if (usable_devices.size() == 0)
      throw RUMD_Error("Device","Init","error: no usable devices supporting CUDA.");
    cudaError err = cudaSetValidDevices(&usable_devices[0], usable_devices.size());
    if (err != cudaSuccess )
      throw RUMD_Error("Device","Init",std::string("[cudaSetValidDevices] ") + cudaGetErrorString(err));
    // trigger device initialization by a non-device management function call
    cudaError err2 = cudaDeviceSynchronize();
    if (err2 != cudaSuccess )
      throw RUMD_Error("Device","Init",std::string("[cudaDeviceSynchronize] ") + cudaGetErrorString(err));

  }

  // test for incorrect compute capability (thrust is useful for this
  // since it immediately raises an exception; an ordinary kernel will simply
  // not run but give no exception)
  thrust::device_vector<float> thrust_temp(1000);
  thrust::sequence(thrust_temp.begin(), thrust_temp.end());
  
  std::cout << "CUDA:" << Report() << std::endl;
}

void Device::Synchronize() {
  cudaDeviceSynchronize();
}

void Device::CheckErrors() {
  cudaDeviceSynchronize();
  cudaError err = cudaGetLastError();
    if(err != cudaSuccess)
  	throw( RUMD_Error("Device","CheckErrors", cudaGetErrorString(err) ) );
}

float Device::Time() {

  float elapsed_time = 0.f;
  cudaEventRecord( event2, 0 );
  cudaEventSynchronize( event2 );
  cudaEventElapsedTime( &elapsed_time, event1, event2);
  return elapsed_time/1000.; // return time in seconds
}
