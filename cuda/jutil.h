////////////////////////////////////////////////////////////////
//Name: 4stereo/jutil.h
//Created date: 06-12-2011
//Modified date: 20-12-2011
//Author: Jie Shen
//Discription: utilities
//Comment: getTimeInMSec() precision is not determined
////////////////////////////////////////////////////////////////

#ifndef _JUTIL_H_
#define _JUTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define MAX_RAND_FP 100
#define GET_RAND_FP ( (float)MAX_RAND_FP * (float)rand() / ((float)(RAND_MAX)+(float)(1)) )
#define GET_RAND(x) (rand()%x)
#define MAX_RELATIVE_ERR 1e-5



double getTimeInSec(){
	struct timeval t;
	gettimeofday(&t, NULL);
	return (double)(t.tv_sec + t.tv_usec * 1e-6); 
}

double getTimeInMSec(){
	struct timeval t;
	gettimeofday(&t, NULL);
	return (double)(t.tv_sec * 1e+3 + t.tv_usec * 1e-3);
}

double getTimeInUSec(){
	struct timeval t;
	gettimeofday(&t, NULL);
	return (double)(t.tv_sec * 1e+6 + t.tv_usec);
}

void fmatrixInit(float* matrix, const int rows, const int cols){
	int i, j;
	srand(time(NULL));
	for(i = 0; i < rows; i++)
		for(j = 0; j < cols; j++)
			matrix[i * cols + j] = GET_RAND_FP;
}

void cmatrixInit(unsigned char* matrix, const int rows, const int cols){
	int i, j;
	srand(time(NULL));
        for(i = 0; i < rows; i++)
                for(j = 0; j < cols; j++)
                        matrix[i * cols + j] = GET_RAND(256);
}

void cmatrixInit_3(unsigned char* matrix, const int rows, const int cols){
	int i, j, k;
	srand(time(NULL));
        for(i = 0; i < rows; i++)
                for(j = 0; j < cols; j++)
			for(k = 0; k < 3; k++)
			matrix[(i * cols + j) * 3 + k] = GET_RAND(256);
}


void matrixPrint(const float* matrix, const int rows, const int cols){
	int i, j;
	for(i = 0; i < rows; i++){
		for(j = 0; j < cols; j++)
			printf("%f ", matrix[i * cols + j]);
		printf("\n");
	}
}

void matrixVerify(const float* mTar, const float* mRef, const int rows, const int cols){
	int i, j, status=1, count=0;
	for(i = 0; i < rows; i++)
		for(j = 0; j < cols; j++)
			if(fabs(mTar[i * cols + j] - mRef[i * cols + j]) / mRef[i * cols + j] > MAX_RELATIVE_ERR){
				status = 0;
				count++;
			}
	if(status)
		printf("Success: Programm passed\n");
	else	printf("Error: Program failed\n");	
}


void ptrErrorCheck(void* ptr){
	if(ptr == NULL){
		printf("Error: Pointer is NULL. Memory allocation error\n");
		exit(-1);
	}
}

#endif
