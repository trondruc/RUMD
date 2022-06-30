
/*
  Copyright (C) 2010  Thomas Schrøder
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  LICENSE.txt file for license details.
*/

#include "rumd/rumd_algorithms.h"
#include "rumd/rumd_technical.h"
#include <sstream>
#include <iostream>

__host__ __device__ float4 operator+ (const float4 &a, const float4 &b){
  float4 r;
  r.x = a.x + b.x;
  r.y = a.y + b.y;
  r.z = a.z + b.z;
  r.w = a.w + b.w;
  return r;
}

__host__ __device__ void operator+= (float4 &a, const float4 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

__host__ __device__ void operator+= (volatile float4 &a, volatile const float4 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

__host__ __device__ double4 operator+ (const double4 &a, const double4 &b){
  double4 r;
  r.x = a.x + b.x;
  r.y = a.y + b.y;
  r.z = a.z + b.z;
  r.w = a.w + b.w;
  return r;
}

__host__ __device__ void operator+= (double4 &a, const double4 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

__host__ __device__ void operator+= (volatile double4 &a, volatile const double4 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}


__host__ __device__ void operator+= (float2 &a, const float2 &b){
  a.x += b.x;
  a.y += b.y;
}

__host__ __device__ void operator+= (volatile float2 &a, volatile const float2 &b){
  a.x += b.x;
  a.y += b.y;
}

__host__ __device__ void operator+= (double2 &a, const double2 &b){
  a.x += b.x;
  a.y += b.y;
}

__host__ __device__ void operator+= (volatile double2 &a, volatile const double2 &b){
  a.x += b.x;
  a.y += b.y;
}

__host__ __device__ void operator+= (int3 &a, const int3 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

__host__ __device__ void operator+= (volatile int3 &a, volatile const int3 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

__device__ inline float fast_rsqrtf(float a) {
  return rsqrtf(a);
}

__device__ inline double fast_rsqrt(double a) {
  return rsqrt(a);
}

template <class T> std::string toString( T t, std::ios_base& (*f)(std::ios_base&) ){
  std::ostringstream oss;
  oss << f << t;
  return oss.str();
}

//////////////////////////////////////////////////////////
// Reduction algorithm: [Mark Harris, NVIDIA, SDK]
///////////////////////////////////////////////////////////

// Forward declaration.
template <class T, unsigned int blockSize> __global__ void reduce( T* g_idata, T* g_odata, unsigned int numParticles, unsigned stride );

// Dynamically allocated shared memory is declared extern. Thus we need to define
// different names corresponding to different variable types.
template <typename T> struct SharedMemory{
  __device__ T* getPointer(){
    extern __device__ void error(void); // Undefined. Ensures that we won't compile any un-specialized types.
    error();
    return NULL;
  }
  __device__ T getZero(){ return NULL; }
};

template <> struct SharedMemory <float>{
  __device__ float* getPointer(){ extern __shared__ float s_float[]; return s_float; }    
  __device__ float getZero(){ return 0.f; }
};

template <> struct SharedMemory <float2>{
  __device__ float2* getPointer(){ extern __shared__ float2 s_float2[]; return s_float2; }    
  __device__ float2 getZero(){ float2 zero = { 0.f, 0.f }; return zero; }
};

template <> struct SharedMemory <float4>{
  __device__ float4* getPointer(){ extern __shared__ float4 s_float4[]; return s_float4; }    
  __device__ float4 getZero(){ float4 zero = { 0.f, 0.f, 0.f, 0.f }; return zero; }
};

template <> struct SharedMemory <double>{
  __device__ double* getPointer(){ extern __shared__ double s_double[]; return s_double; }    
  __device__ double getZero(){ return 0.; }
};

template <> struct SharedMemory <double2>{
  __device__ double2* getPointer(){ extern __shared__ double2 s_double2[]; return s_double2; }
  __device__ double2 getZero(){ double2 zero = { 0., 0. }; return zero; }
};

template <> struct SharedMemory <double4>{
  __device__ double4* getPointer(){ extern __shared__ double4 s_double4[]; return s_double4; }    
  __device__ double4 getZero(){ double4 zero = { 0., 0., 0., 0. }; return zero; }
};

template <> struct SharedMemory <int3>{
  __device__ int3* getPointer(){ extern __shared__ int3 s_int3[]; return s_int3; }    
  __device__ int3 getZero(){ int3 zero = { 0, 0, 0}; return zero; }
};

inline void getThreadsAndBlocks( unsigned int size, unsigned int& numThreads, unsigned int& numBlocks, unsigned int maxThreadsPerBlock ){
  if(size == 1) 
    numThreads = 1;
  else
    numThreads = (size < 2 * maxThreadsPerBlock) ? size / 2 : maxThreadsPerBlock;
  
  numBlocks = size / (numThreads * 2);
}

template <class T> void sumIdenticalArrays( T* array, unsigned int numParticles, unsigned int numArrays, unsigned int maxThreadsPerBlock ){
  unsigned int numThreads = 0; unsigned int numBlocks = 0;
  unsigned int s = (unsigned) pow( 2, ceil( log( numParticles ) / log( 2 ) ) );

  // Largest nearest power of 2.
  unsigned stride = 1;
  while(s > 1){
    getThreadsAndBlocks( s , numThreads, numBlocks, maxThreadsPerBlock );  
    dim3 dimGrid(numBlocks, numArrays);
    dim3 dimBlock(numThreads, 1);  
    
    // We must allocate 2 * numThreads * sizeof(T) due to warp unrolled in the reduction algorithm.
    unsigned int size = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T); 
    //std::cout << "s = " << s << "; numThreads= " << numThreads  << "; numBlocks = " << numBlocks << "; stride = " << stride << std::endl;
    switch(numThreads){
    case 512:
      reduce<T,512><<< dimGrid, dimBlock, size >>>(array, array, numParticles, stride); break;
    case 256:
      reduce<T,256><<< dimGrid, dimBlock, size >>>(array, array, numParticles, stride); break;
    case 128:
      reduce<T,128><<< dimGrid, dimBlock, size >>>(array, array, numParticles, stride); break;
    case 64:
      reduce<T,64><<< dimGrid, dimBlock, size >>>(array, array, numParticles, stride); break;
    case 32:
      reduce<T,32><<< dimGrid, dimBlock, size >>>(array, array, numParticles, stride); break;
    case 16:
      reduce<T,16><<< dimGrid, dimBlock, size >>>(array, array, numParticles, stride); break;
    case 8:
      reduce<T,8><<< dimGrid, dimBlock, size >>>(array, array, numParticles, stride); break;
    case 4:
      reduce<T,4><<< dimGrid, dimBlock, size >>>(array, array, numParticles, stride); break;
    case 2:
      reduce<T,2><<< dimGrid, dimBlock, size >>>(array, array, numParticles, stride); break;
    case 1:
      reduce<T,1><<< dimGrid, dimBlock, size >>>(array, array, numParticles, stride); break;
    }
    stride *= numThreads*2;
    s = s / (numThreads*2);
  }
}

// Performs a reduction.
template <class T, unsigned int blockSize> __global__ void reduce( T* g_idata, T* g_odata, unsigned int numParticles, unsigned stride ){
  
  // Shared mem size is determined by the host app at run time.
  SharedMemory<T> smem;
  T* sdata = smem.getPointer();
  
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
  unsigned int offset = numParticles * blockIdx.y;
  
  T mySum = (i*stride < numParticles) ? g_idata[offset+i*stride] : smem.getZero();
  if( (i + blockSize)*stride < numParticles )
    mySum += g_idata[offset+(i+blockSize)*stride];
  //if( i + blockSize < numParticles ) 
  // mySum += g_idata[offset+i+blockSize];  

  sdata[tid] = mySum;
  
  __syncthreads();
  
  // Unroll the initial iterations. We have to synchronize, since only 32 threads are atomic.
  if(blockSize >= 512) { if(tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if(blockSize >= 256) { if(tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if(blockSize >= 128) { if(tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
  
  if(tid < 32){
    volatile T* localSdata = sdata; // We must declare volatile amongst the 32 warp threads.
    
    // Unroll the last warp.
    if(blockSize >= 64) localSdata[tid] += localSdata[tid + 32]; 
    if(blockSize >= 32) localSdata[tid] += localSdata[tid + 16]; 
    if(blockSize >= 16) localSdata[tid] += localSdata[tid +  8]; 
    if(blockSize >= 8)  localSdata[tid] += localSdata[tid +  4]; 
    if(blockSize >= 4)  localSdata[tid] += localSdata[tid +  2]; 
    if(blockSize >= 2)  localSdata[tid] += localSdata[tid +  1]; 
  }
    
  // Write result for this block to global mem 
  if(tid == 0 && blockIdx.x*(blockDim.x*2)*stride < numParticles)
    g_odata[offset+blockIdx.x*(blockDim.x*2)*stride] = sdata[0];

}

/////////////////////////////////////////////////////////////////////
// PRNG algorithm: [L. Howes and D. Thomas, GPU Gems 3, Chapter 37]
/////////////////////////////////////////////////////////////////////

__host__ __device__ inline unsigned TausStep( unsigned &z, int S1, int S2, int S3, unsigned M ){
  unsigned b = ( ( ( z << S1 ) ^ z ) >> S2 );
  return z = ( ( ( z & M ) << S3 ) ^ b );
}

__host__ __device__ inline unsigned LCGStep( unsigned &z, unsigned A, unsigned C ){
  return z = ( A * z + C );
}

// Different prefactor will produce (0, 1].
__host__ __device__ inline float HybridTausFloat( uint4* state ){
  return 2.3283064365387e-10f* (
				TausStep(state->x, 13, 19, 12, 4294967294UL) ^
				TausStep(state->y,  2, 25,  4, 4294967288UL) ^
				TausStep(state->z,  3, 11, 17, 4294967280UL) ^
				LCGStep( state->w,  1664525,   1013904223UL)
				);
}



// Have changed return type to float and removed the f from the constant.
// This may need to be looked at more carefully; for example should state have
// type unsigned long long int?
__host__ __device__ inline double HybridTausDouble( uint4* state ){
  return 2.3283064365387e-10* (
				TausStep(state->x, 13, 19, 12, 4294967294UL) ^
				TausStep(state->y,  2, 25,  4, 4294967288UL) ^
				TausStep(state->z,  3, 11, 17, 4294967280UL) ^
				LCGStep( state->w,  1664525,   1013904223UL)
				);
}


/////////////////////////////////////////////////////////////////////
// Box Muller Transformation.
/////////////////////////////////////////////////////////////////////

__host__ __device__ inline float2 BoxMuller( float u0, float u1 ){
  float r = sqrtf( -2.0f * logf(u0) );
  float theta = 2.f * float(M_PI) * u1;
  float2 value = { r*sinf(theta), r*cosf(theta) };
  return value;
}

//////////////////////////////////////////////////////////////////////
// Gaussian Elimination with rowise (partial) pivoting and 
// backsubstitution
//////////////////////////////////////////////////////////////////////

template <class T> __global__ void GaussPivotBack( T* x, T* b, T* A, unsigned* dimlist, unsigned* dimConsecList, unsigned* dimConsecSqList){
  unsigned i=0, j=0, k, kk, ir;
  unsigned lineIdx = (threadIdx.y*blockDim.x + threadIdx.x);
  unsigned lineDim = (blockDim.x*blockDim.y);
  unsigned nrc = dimlist[blockIdx.x]; // Number of constraints
  unsigned nra = nrc+1;
  unsigned vec_offset = dimConsecList[blockIdx.x];
  unsigned mtx_offset = dimConsecSqList[blockIdx.x];
  T tmp;

  
  extern __shared__ T s_A[];
  T* s_c = &s_A[nra*nrc];

  // Load system from global to shared
  for ( k=lineIdx; k < nrc*nrc; k+=lineDim )
    s_A[ (k/nrc)*nra + k%nrc ] = A[mtx_offset+k];
    
  for ( k=lineIdx; k < nrc; k+=lineDim )
  s_A[ k*nra + nrc ] = b[vec_offset+k];

  while ( i<nrc && j<nrc ){

    __syncthreads();

    // Find pivot
    ir = i;
    for ( k=i+1; k<nrc; k++ ){
      if ( fabs(s_A[k*nra +j]) > fabs(s_A[ir*nra+j]) )
	ir = k;
    }

    // Check that pivot is non-zero
    //if ( fabs(s_A[ir*nra+j]) > FLT_EPSILON ){
    if ( fabs(s_A[ir*nra+j]) > 0. ){    
      // Swap rows of s_A
      for ( k=lineIdx; k < nra; k+=lineDim ){
	tmp = s_A[i*nra+k];
	s_A[i*nra+k] = s_A[ir*nra+k];
	s_A[ir*nra+k] = tmp;
      }

      __syncthreads();

      // Temporary copy of pivot column
      for ( k=lineIdx; k < nrc; k+=lineDim )
	s_c[k] = s_A[k*nra+j];

      __syncthreads();

      // Divide row i with s_A(i,j)
      for ( k=lineIdx; k < nra; k+=lineDim )
	s_A[i*nra+k] /= s_c[i];

      __syncthreads();
     
      // Multiply and subtract - Gaussian part
      for ( k=threadIdx.x; k < nra; k+=blockDim.x )
	for ( kk=i+1+threadIdx.y; kk<nrc; kk+=blockDim.y )
	  s_A[kk*nra+k] -= s_c[kk]*s_A[i*nra+k];
   
      i++;
    }
    j++;
  }

  __syncthreads();

  // Backsubstitution
  for ( int i = nrc-1; i>=0; i-- )
    for ( k=lineIdx; k < i; k+=lineDim )
      s_A[k*nra+nrc] -= s_A[k*nra+i]*s_A[i*nra+nrc];

  __syncthreads();

  // Copy solution back to global
  for ( k=lineIdx; k < nrc; k+=lineDim ) {
      x[vec_offset+k] = s_A[k*nra+nrc];
  }
}

template <class T> void solveLinearSystems( T* x, T* b, T* A, unsigned* dimLinearSystems, unsigned* dimConsecLinearSystems, unsigned* dimConsecSqLinearSystems, unsigned nLinearSystems, unsigned maxc){
  dim3 threadsPerBlock(maxc+1, 2);
  unsigned shared_size = maxc * (maxc + 2) * sizeof(T);

  GaussPivotBack<<<nLinearSystems, threadsPerBlock, shared_size>>>( x, b, A,
								    dimLinearSystems,
								    dimConsecLinearSystems,
								    dimConsecSqLinearSystems);
}


//////////////////////////////////////////////////////////////////////
// Fast solver for tridiagonal matrices
// See Press et al: Numerical recipes
// Only the data transfer of this algorithm is parallel,
// So for large systems it can be optimized significantly
//////////////////////////////////////////////////////////////////////

template <class T> __global__ void solve_tridiagonal_systems_kernel( T* x, T* b, T* A, unsigned* dimlist, unsigned* dimConsecList, unsigned* dimConsecSqList){
  unsigned k;
  unsigned nrc = dimlist[blockIdx.x]; // Number of constraints
  unsigned lineIdx = (threadIdx.y*blockDim.x + threadIdx.x);
  unsigned lineDim = (blockDim.x*blockDim.y);
  unsigned vec_offset = dimConsecList[blockIdx.x];
  unsigned mtx_offset = dimConsecSqList[blockIdx.x];
  
  // Three diagonals in shared memory
  extern __shared__ T s_da[];
  T* s_db = &s_da[nrc];
  T* s_dc = &s_db[nrc];
  T* s_b =  &s_dc[nrc];
  T* s_g =  &s_b[nrc];
  T* s_x =  &s_g[nrc];

  // Load system from global to shared
  for ( k=lineIdx; k < nrc; k+=lineDim ){
    s_b[k] = b[vec_offset+k];
    s_db[k]   = A[ mtx_offset +  k*nrc    + k ];
  }
  for ( k=lineIdx; k < nrc-1; k+=lineDim ){
    s_da[k+1] = A[ mtx_offset + (k+1)*nrc + k ];
    s_dc[k]   = A[ mtx_offset +  k*nrc +1 + k ];
  }

  __syncthreads();

  if(lineIdx == 0){
    T bet = s_db[0];
    s_x[0] = s_b[0]/bet;
    
    // Decomposition and forward substitution
    for ( unsigned j=1; j<nrc; j++ ){
      s_g[j] = s_dc[j-1]/bet;
      bet    = s_db[j] - s_da[j]*s_g[j];
      s_x[j] = ( s_b[j] - s_da[j]*s_x[j-1] )/bet;
    }
    // Backsubstitution
    for ( int j=(nrc-2); j>=0; j-- )
      s_x[j] -= s_g[j+1]*s_x[j+1];
  }

  __syncthreads();

  // Copy solution back to global
  for ( k=lineIdx; k < nrc; k+=lineDim )
      x[vec_offset+k] = s_x[k];
}

template <class T> void solveTridiagonalLinearSystems( T* x, T* b, T* A, unsigned* dimLinearSystems, unsigned* dimConsecLinearSystems, unsigned* dimConsecSqLinearSystems, unsigned nLinearSystems, unsigned maxConstraintsPerMolecule){
  dim3 threadsPerBlock(maxConstraintsPerMolecule);
  unsigned shared_size = 6 * maxConstraintsPerMolecule * sizeof(T);

  solve_tridiagonal_systems_kernel<<<nLinearSystems, threadsPerBlock, shared_size>>>( x, b, A, dimLinearSystems, dimConsecLinearSystems, dimConsecSqLinearSystems);
}


template void sumIdenticalArrays<float>( float*, unsigned, unsigned, unsigned ); 
template void sumIdenticalArrays<float2>( float2*, unsigned, unsigned, unsigned );
template void sumIdenticalArrays<float4>( float4*, unsigned, unsigned, unsigned );

template void sumIdenticalArrays<double>( double*, unsigned, unsigned, unsigned ); 
template void sumIdenticalArrays<double2>( double2*, unsigned, unsigned, unsigned );
template void sumIdenticalArrays<double4>( double4*, unsigned, unsigned, unsigned );

template void sumIdenticalArrays<int3>( int3*, unsigned, unsigned, unsigned );

template std::string toString<unsigned long int>( unsigned long int, std::ios_base& (*f)(std::ios_base&) );

template void solveLinearSystems<float>( float*, float*, float*, unsigned*, unsigned*, unsigned*, unsigned, unsigned);
template void solveTridiagonalLinearSystems<float>( float*, float*, float*, unsigned*,unsigned*,unsigned*, unsigned, unsigned);

// and the double precision versions. There are problems having both these
// and the single precions versions available at the same time, will be fixed
// later. For now one set must be commented out.

//template void solveLinearSystems<double>( double*, double*, double*, unsigned*, unsigned*, unsigned*, unsigned, unsigned);
//template void solveTridiagonalLinearSystems<double>( double*, double*, double*, unsigned*,unsigned*,unsigned*, unsigned, unsigned);

