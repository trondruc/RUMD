#ifndef RUMD_TECHNICAL_H
#define RUMD_TECHNICAL_H

#include <vector_types.h>

////////////////////////////////////////////////////////////
// This header defines types/names used within the RUMD code.
////////////////////////////////////////////////////////////


#define PPerBlock  blockDim.x                  // Number of particles in each block
#define TPerPart   blockDim.y                  // Number of threads used for each particle
#define NumBlocks  gridDim.x                   // Number of blocks

#define MyB        blockIdx.x                  // Block number for 'this' thread
#define MyP        threadIdx.x                 // (local) particle number of 'this' thread
#define MyT        threadIdx.y                 // (local) thread number of 'this' thread
#define MyGP       (MyP + MyB * PPerBlock)     // Global particle number of 'this' thread
#if __CUDA_ARCH__ >= 350
#define LOAD(x) __ldg(&(x))
#else
#define LOAD(x) (x)
#endif


#endif // RUMD_TECHNICAL_H
