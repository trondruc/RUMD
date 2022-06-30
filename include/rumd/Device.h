#ifndef DEVICE_H
#define DEVICE_H
/*
    Copyright (C) 2010  Thomas Schrøder

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/

#include <iostream>
// following is for access to cudaEvent_t when compiling without nvcc
#include <cuda_runtime_api.h>
class Device
{
public:

  virtual ~Device();
  static Device& GetDevice();
  void Init();
  void Synchronize();
  const std::string GetDeviceName();
  const std::string GetDeviceReport();
  unsigned GetMaxThreadsPerBlock();
  unsigned GetMaximumGridDimensionX();
  unsigned GetComputeCapability();
  size_t GetSharedMemPerBlock();
  size_t GetDeviceMemory();
  float Time();
  void CheckErrors();
protected:
  Device();
private:
  Device(const Device& d);
  Device& operator=(const Device& d);
  std::string Report();
  cudaEvent_t event1;
  cudaEvent_t event2;
};

#endif
