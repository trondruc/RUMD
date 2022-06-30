#ifndef RUMD_UTILS_H
#define RUMD_UTILS_H
 
/*
    Copyright (C) 2010  Thomas Schrøder

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/

// modernised utility macro from CUDA SDK to hide checking of CUDA API return values

#include <iostream>

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        std::cerr << "CUDA error in file: " << __FILE__  << " in line: " << __LINE__ << " : " << cudaGetErrorString(err) << std::endl;  \
        exit(EXIT_FAILURE);                                                  \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);

#endif
