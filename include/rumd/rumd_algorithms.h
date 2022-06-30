#ifndef RUMD_ALGORITHMS_H
#define RUMD_ALGORITHMS_H

/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include <string>
#include <cfloat>

////////////////////////////////////////////////////////////
// This header defines a collection of useful MD algorithms.
// Topics: Operator Overloads, PRNG, Gaussian and Reduction.
//
// DISCLAIMER: Due to device function support in CUDA <= 3.1,
// this library is only avaliable from the host side
////////////////////////////////////////////////////////////

// Common operator overloads.
__host__ __device__ float4 operator+ (const float4 &a, const float4 &b);
__host__ __device__ void operator+= (float4 &a, const float4 &b);
__host__ __device__ void operator+= (volatile float4 &a, volatile const float4 &b);
__host__ __device__ void operator+= (float2 &a, const float2 &b);
__host__ __device__ void operator+= (volatile float2 &a, volatile const float2 &b);


__host__ __device__ double4 operator+ (const double4 &a, const double4 &b);
__host__ __device__ void operator+= (double4 &a, const double4 &b);
__host__ __device__ void operator+= (volatile double4 &a, volatile const double4 &b);
__host__ __device__ void operator+= (double2 &a, const double2 &b);
__host__ __device__ void operator+= (volatile double2 &a, volatile const double2 &b);
  
// The HybridTaus generator state is comprised of 4 unsigned integers. 
// When initializing the state you MUST set (x,y,z) > 128. 
// (w) can be any random number, for instance between [0, RAND_MAX].
// Input:  A state of 4 unsigned integers.
// Output: A float between [0, 1].
__host__ __device__ float HybridTausFloat( uint4* state );
// double precision version
__host__ __device__ double HybridTausDouble( uint4* state );

// Box Muller Gaussian random number generation.
// Input:  Two random uniform numbers in (0, 1].
// Output: Two random Gaussian distributed numbers.
__host__ __device__ float2 BoxMullerFloat( float u0, float u1 );
// double precision version
__host__ __device__ double2 BoxMullerDouble( double u0, double u1 );

// Performs a summation of numArrays array's of size numParticles.
// Note: Any unsupported types must be added in rumd_templates.cu.
// Output: Elements [0], [numParticles], [2*numParticles], etc.
template <class T> void sumIdenticalArrays( T* array, unsigned int numParticles, unsigned int numArrays, unsigned int maxThreadsPerBlock );

// Generic to string. Format: std::hex, std::dec or std::oct.
template <class T> std::string toString( T t, std::ios_base& (*f)(std::ios_base&) );

// Solving systems of linear equations Ax = b with possible different dimensionality. 
// The systems are laid out in sequential memory row-wise ie. 
// A11...A1N, A21...A2N, ..., B11...B1N, ..., C11...C1N etc.
template <class T> void solveLinearSystems( T* x, T* b, T* A, unsigned* dimLinearSystems, unsigned* dimConsecLinearSystems, unsigned* dimConsecSqLinearSystems, unsigned nLinearSystems, unsigned maxConstraintsPerMolecule);

template <class T> void solveTridiagonalLinearSystems( T* x, T* b, T* A, unsigned* dimLinearSystems, unsigned* dimConsecLinearSystems, unsigned* dimConsecSqLinearSystems, unsigned nLinearSystems, unsigned maxConstraintsPerMolecule);

// Atomic add is only available for single precision; for double precision
// we implement using atomicCAS

// Putting them here with almost similar names makes it easy to do the
// float->double conversion
__device__ inline void atomicFloatAdd(float *address, float val){
  atomicAdd(address, val);
}

__device__ inline void atomicDoubleAdd(double *address, double val){
  unsigned long long int i_val = __double_as_longlong(val);
  unsigned long long int tmp0 = 0;
  unsigned long long int tmp1;
  while( (tmp1 = atomicCAS((unsigned long long int *)address, tmp0, i_val)) != tmp0){
    tmp0 = tmp1;
    i_val = __double_as_longlong(val + __longlong_as_double(tmp1));
  }
}

#endif // RUMD_ALGORITHMS_H
