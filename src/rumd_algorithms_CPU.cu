#include "rumd/rumd_algorithms_CPU.h"

#include <cfloat>


void GaussPivotBackCPU(float *x, float *b, float *A, unsigned nrc){
  unsigned i=0, j=0, k, kk; 
  float tmp, c;

  for ( unsigned int n=0; n<nrc; n++ ) x[n] = b[n];

  while ( i<nrc && j<nrc ){
    
    // Find pivot
    unsigned ir = i;
    for ( k=i+1; k<nrc; k++ ){
      if ( fabs(A[k*nrc+j]) > fabs(A[ir*nrc+j]) )
	ir = k;
    }
    
    if ( fabs(A[ir*nrc+j])>FLT_EPSILON ){

      // Swap rows
      for ( k=0; k<nrc; k++ ){
	tmp = A[i*nrc+k];
	A[i*nrc+k] = A[ir*nrc+k];
	A[ir*nrc+k] = tmp;
      }
      
      tmp = b[i];
      b[i] = b[ir];
      b[ir] = tmp;

      // Divide row i (Eq. (i)) with A(i,j)
      c = A[i*nrc+j];
      b[i] /= c;
      for ( k=0; k<nrc; k++ ) A[i*nrc+k] /= c;
            
      // Multiply and substract A(i,j) - Gaussian part
      for ( k=i+1; k<nrc; k++ ){
	c = A[k*nrc + j];
	for ( kk=0; kk<3; kk++ )
	  A[k*nrc+kk] -= c*A[i*nrc+kk];
	b[k] -= c*b[i];
      }

      i++;
    }

    j++;
  }

  // Backsubstitution
  for ( int i=nrc-1; i>=0; i-- ){
    
    float s = 0.0;
    for ( int j=i+1; j<(signed)nrc; j++ )
      s += A[i*nrc+j]*b[j];

    b[i] = (b[i] - s)/A[i*nrc+i];
  }

  
  for ( unsigned int n=0; n<nrc; n++ ){
    float tmp = x[n];
    x[n] = b[n];
    b[n] = tmp;
  }

}


double Determinant3(float *A){
  
  double a = A[0]*A[4]*A[8];
  double b = A[0]*A[5]*A[7];
  double c = A[1]*A[3]*A[8];
  double d = A[1]*A[4]*A[6];
  double e = A[2]*A[3]*A[7];
  double f = A[2]*A[4]*A[6];

  return a - b - c + d + e - f;

}
