////////////////////////////////////////////////////////////////
//Name: 5stereo/jcudautil.h
//Created date: 01-02-2012
//Modified date: 01-02-2012
//Author: Jie Shen
//Discription: cuda utilities
////////////////////////////////////////////////////////////////

#ifndef _JUTIL_CUDA_H_
#define _JUTIL_CUDA_H_

#include <stdio.h>
#include <stdlib.h>

void __cudaSafeCall(cudaError_t err){
	if(err != cudaSuccess){
		printf("Error: CUDA runtime API error %d: %s\n", (int)err, cudaGetErrorString(err));
		exit(-1);
	}
}


void __cudaKernelCheck(void){
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("Error: CUDA kernel launch error %d: %s\n", (int)err, cudaGetErrorString(err));
	exit(-1);
	} 
}

#endif
