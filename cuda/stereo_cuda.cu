///////////////////////////////////////////////////////////////////////////////
//Path: 5stereo_rearrange/main.cu
//Created date: 01-02-2012
//Modified date: 01-02-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: stereo cuda host code
//comment: The problem of first 2-3 methods
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <sys/time.h>
#include "pixelBasedCost_kernel.cu"
#include "boxAggregate_kernel.cu"
#include "WTA_kernel.cu"
#include "rankFilter_kernel.cu"
#include "censusTransform_kernel.cu"
#include "convolveImage_kernel.cu"	
#include "nccDisparity_kernel.cu"
#include "computeWeight_kernel.cu"
#include "crossStereo_kernel.cu"


#define DIMBLOCK 16
#define DIMGRID(x) x / DIMBLOCK + ((x % DIMBLOCK == 0)?0:1);
#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5)) 

//#define VERIFY
#define TIME

void __cudaSafeCall(cudaError_t err);
void __cudaKernelCheck(void);
double getTimeInMSec();
void ptrErrorCheck(void* ptr);
void calcProxWeight(float *proxWeight, float gamma_proximity, int maskRad);

unsigned char* dispRef, * dispTar, * imCrossCheck, * imRankFilter;
bool* imCensusTransRef, * imCensusTransTar;
float* imCost, * imLOGFilterRef, * imLOGFilterTar, * proxWeight;

unsigned char* d_imRef, * d_imTar, * d_dispRef, * d_dispTar, * d_imCrossCheck, * d_imRankFilterL, * d_imRankFilterR, * d_imLABRef, * d_imLABTar;
unsigned char *d_imRGBRef,*d_imRGBTar;
float* d_imCost, * d_imLOGFilterRef, * d_imLOGFilterTar, * d_F_LOG, * d_proxWeight, * d_weightRef, * d_weightTar, *d_cost_new_image;
bool*  d_imCensusTransRef, * d_imCensusTransTar;

void allocate(int rows, int cols, int dispRange, int maskCensusSize, int maskAdapSize)
{
	//output
        //host memory allocation
    dispRef = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));
	dispTar = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));
	imCrossCheck = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));
	imCost = (float*)malloc(rows * cols * dispRange * sizeof(float));
	imRankFilter = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));
	imCensusTransRef = (bool*)malloc(rows * cols * maskCensusSize * sizeof(bool));
	imCensusTransTar = (bool*)malloc(rows * cols * maskCensusSize * sizeof(bool));
	imLOGFilterRef = (float*)malloc(rows * cols * sizeof(float));
	imLOGFilterTar = (float*)malloc(rows * cols * sizeof(float));
	proxWeight = (float*)malloc(maskAdapSize * sizeof(float));

	ptrErrorCheck(dispRef);
	ptrErrorCheck(dispTar);
	ptrErrorCheck(imCrossCheck);
	ptrErrorCheck(imCost);
	ptrErrorCheck(imRankFilter);
	ptrErrorCheck(imCensusTransRef);
	ptrErrorCheck(imCensusTransTar);
	ptrErrorCheck(imLOGFilterRef);
	ptrErrorCheck(imLOGFilterTar);
	ptrErrorCheck(proxWeight);

	//device memory allocation
    __cudaSafeCall(cudaMalloc((void**)&d_imRef, rows * cols * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_imTar, rows * cols * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_dispRef, rows * cols * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_dispTar, rows * cols * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_imCrossCheck, rows * cols * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_imCost, rows * cols * dispRange * sizeof(float)));
	__cudaSafeCall(cudaMalloc((void**)&d_cost_new_image, rows * cols * dispRange * sizeof(float)));
	__cudaSafeCall(cudaMalloc((void**)&d_imRankFilterL, rows * cols * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_imRankFilterR, rows * cols * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_imCensusTransRef, rows * cols * maskCensusSize * sizeof(bool)));
	__cudaSafeCall(cudaMalloc((void**)&d_imCensusTransTar, rows * cols * maskCensusSize * sizeof(bool)));
	__cudaSafeCall(cudaMalloc((void**)&d_imLOGFilterRef, rows * cols * sizeof(float)));
	__cudaSafeCall(cudaMalloc((void**)&d_imLOGFilterTar, rows * cols * sizeof(float)));
	__cudaSafeCall(cudaMalloc((void**)&d_F_LOG, 25 * sizeof(float)));
	__cudaSafeCall(cudaMalloc((void**)&d_proxWeight, maskAdapSize * sizeof(float)));
	__cudaSafeCall(cudaMalloc((void**)&d_imLABRef, rows * cols * 3 * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_imLABTar, rows * cols * 3 * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_imRGBRef, rows * cols * 3 * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_imRGBTar, rows * cols * 3 * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_weightRef, rows * cols * maskAdapSize * sizeof(float)));
	__cudaSafeCall(cudaMalloc((void**)&d_weightTar, rows * cols * maskAdapSize * sizeof(float)));

}


void deallocate()
{
	//free device memory
	if(d_imRef){
		printf("free d_imRef\n");
		__cudaSafeCall(cudaFree(d_imRef));
	}
        if(d_imTar){
                printf("free d_imTar\n");
                __cudaSafeCall(cudaFree(d_imTar));
        }
        if(d_dispRef){
                printf("free d_dispRef\n");
                __cudaSafeCall(cudaFree(d_dispRef));
        }
        if(d_dispTar){
                printf("free d_dispTar\n");
                __cudaSafeCall(cudaFree(d_dispTar));
        }
        if(d_imCrossCheck){
                printf("free d_imCrossCheck\n");
                __cudaSafeCall(cudaFree(d_imCrossCheck));
        }
        if(d_imCost){
                printf("free d_imCost\n");
                __cudaSafeCall(cudaFree(d_imCost));
        }
	if(d_imRankFilterL){
		printf("free d_imRankFilterL\n");
		__cudaSafeCall(cudaFree(d_imRankFilterL));
	}
	if(d_imRankFilterR){
		printf("free d_imRankFilterR\n");
		__cudaSafeCall(cudaFree(d_imRankFilterR));
	}
	if(d_imCensusTransRef){
		printf("free d_imCensusTransRef\n");
		__cudaSafeCall(cudaFree(d_imCensusTransRef));
	}
	if(d_imCensusTransTar){
		printf("free d_imCensusTransTar\n");
		__cudaSafeCall(cudaFree(d_imCensusTransTar));
	}
	if(d_imLOGFilterRef){
		printf("free d_imLOGFilterRef\n");
		__cudaSafeCall(cudaFree(d_imLOGFilterRef));
	}
        if(d_imLOGFilterTar){
                printf("free d_imLOGFilterTar\n");
                __cudaSafeCall(cudaFree(d_imLOGFilterTar));
        }
	if(d_F_LOG){
		printf("free d_F_LOG\n");
		__cudaSafeCall(cudaFree(d_F_LOG));	
	}
	if(d_proxWeight){
		printf("free d_proxWeight\n");
		__cudaSafeCall(cudaFree(d_proxWeight));
	}
	if(d_imLABRef){
		 printf("free d_imLABRef\n");  
                __cudaSafeCall(cudaFree(d_imLABRef));
	}
	if(d_imLABTar){ 
                 printf("free d_imLABTar\n");  
                __cudaSafeCall(cudaFree(d_imLABTar));
        }
	if(d_weightRef){
		printf("free d_weightRef\n");
		__cudaSafeCall(cudaFree(d_weightRef));
	}
	if(d_weightTar){
                printf("free d_weightTar\n");
                __cudaSafeCall(cudaFree(d_weightTar));
        }
		

	//free host memory
	if(dispRef){
		printf("free dispRef\n");
		free(dispRef);
	}
	if(dispTar){
		printf("free dispTar\n");
		free(dispTar);
	} 
	if(imCrossCheck){
		printf("free imCrossCheck\n");
		free(imCrossCheck);
	}
	if(imCost){
		printf("free imCost\n");
		free(imCost);
	}
	if(imRankFilter){
		printf("free imRankFilter\n");
		free(imRankFilter);
	}
	if(imCensusTransRef){
		printf("free imCensusTransRef\n");
		free(imCensusTransRef);
	}
	if(imCensusTransTar){
		printf("free imCensusTransTar\n");
		free(imCensusTransTar);
	}
	if(imLOGFilterRef){
		printf("free imLOGFilterRef\n");
		free(imLOGFilterRef);
	}
	if(imLOGFilterTar){
                printf("free imLOGFilterTar\n");
                free(imLOGFilterTar);
        }
	if(proxWeight){
		printf("free proxWeight\n");
		free(proxWeight);
	}
}


void H2D(unsigned char* imRef, unsigned char* imTar,  
	unsigned char *imRGBRef, unsigned char *imRGBTar,
	unsigned char* imLABRef, unsigned char* imLABTar, 
	int rows, int cols, float* F_LOG)
{
#ifdef TIME
	double H2D_timeStart = getTimeInMSec(); 
#endif
	float F_LOG_[25] = { 0.0239, 0.046, 0.0499, 0.046, 0.0239,
			    0.046, 0.0061, -0.0923, 0.0061, 0.046,
			    0.0499, -0.0923, -0.3182, -0.0923, 0.0499,
			    0.046, 0.0061, -0.0923, 0.0061, 0.046,
			    0.0239, 0.046, 0.0499, 0.046, 0.0239 };
	//host to device data transfer
	__cudaSafeCall(cudaMemcpy(d_imRef, imRef, rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice));
	__cudaSafeCall(cudaMemcpy(d_imTar, imTar, rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice));
	__cudaSafeCall(cudaMemcpy(d_F_LOG, F_LOG_, 25 * sizeof(float), cudaMemcpyHostToDevice));
	__cudaSafeCall(cudaMemcpy(d_imLABRef, imLABRef, rows * cols * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
	__cudaSafeCall(cudaMemcpy(d_imLABTar, imLABTar, rows * cols * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
	__cudaSafeCall(cudaMemcpy(d_imRGBRef, imRGBRef, rows * cols * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
	__cudaSafeCall(cudaMemcpy(d_imRGBTar, imRGBTar, rows * cols * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

#ifdef TIME
	double H2D_timeStop = getTimeInMSec();
	printf("Time H2D = %.3fms\n", (H2D_timeStop - H2D_timeStart));
#endif	
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void* pixelBasedCostL2R_Cuda(int rdim, int cdim, int dispRange, int maskSize)
{
      int dimGridCols = DIMGRID(cdim);
      int dimGridRows = DIMGRID(rdim);
      dim3 dimGrid(dimGridCols, dimGridRows);
      dim3 dimBlock(DIMBLOCK, DIMBLOCK);

#ifdef TIME
	double KADBM_timeStart = getTimeInMSec();
#endif
	k_pixelBasedCostL2R<<<dimGrid, dimBlock>>>(d_imRef, d_imTar, d_imCost, dispRange, rdim, cdim);
	__cudaKernelCheck();
	
#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
#else
        __cudaSafeCall(cudaThreadSynchronize());
#endif
	
#ifdef TIME
	double KADBM_timeStop = getTimeInMSec();
#endif


#ifdef TIME
	double KADBM_D2H_timeStop = getTimeInMSec();
	printf("Time KADBM = %.3fms\n", (KADBM_timeStop - KADBM_timeStart));
	printf("Time KADBM_D2H = %.3fms\n", (KADBM_D2H_timeStop - KADBM_timeStop));
#endif
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
void* pixelBasedCostR2L_Cuda(int rdim, int cdim, int dispRange, int maskSize)
{
      int dimGridCols = DIMGRID(cdim);
      int dimGridRows = DIMGRID(rdim);
      dim3 dimGrid(dimGridCols, dimGridRows);
      dim3 dimBlock(DIMBLOCK, DIMBLOCK);

#ifdef TIME
	double KADBM_timeStart = getTimeInMSec();
#endif
	k_pixelBasedCostR2L<<<dimGrid, dimBlock>>>(d_imRef, d_imTar, d_imCost, dispRange, rdim, cdim);
	__cudaKernelCheck();
	
#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
#else
        __cudaSafeCall(cudaThreadSynchronize());
#endif
	
#ifdef TIME
	double KADBM_timeStop = getTimeInMSec();
#endif


#ifdef TIME
	double KADBM_D2H_timeStop = getTimeInMSec();
	printf("Time KADBM = %.3fms\n", (KADBM_timeStop - KADBM_timeStart));
	printf("Time KADBM_D2H = %.3fms\n", (KADBM_D2H_timeStop - KADBM_timeStop));
#endif
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

void* pixelBasedCostL2R_Float_Cuda(int rdim, int cdim, int dispRange, int maskSize)
{
      int dimGridCols = DIMGRID(cdim);
      int dimGridRows = DIMGRID(rdim);
      dim3 dimGrid(dimGridCols, dimGridRows);
      dim3 dimBlock(DIMBLOCK, DIMBLOCK);

#ifdef TIME
	double KADBM_timeStart = getTimeInMSec();
#endif
	k_pixelBasedCostL2R_Float<<<dimGrid, dimBlock>>>(d_imLOGFilterRef, d_imLOGFilterTar, d_imCost, dispRange, rdim, cdim);
	__cudaKernelCheck();
	
#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
#else
        __cudaSafeCall(cudaThreadSynchronize());
#endif
	
#ifdef TIME
	double KADBM_timeStop = getTimeInMSec();
#endif


#ifdef TIME
	double KADBM_D2H_timeStop = getTimeInMSec();
	printf("Time KADBM = %.3fms\n", (KADBM_timeStop - KADBM_timeStart));
	printf("Time KADBM_D2H = %.3fms\n", (KADBM_D2H_timeStop - KADBM_timeStop));
#endif
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

void* pixelBasedCostR2L_Float_Cuda(int rdim, int cdim, int dispRange, int maskSize)
{
      int dimGridCols = DIMGRID(cdim);
      int dimGridRows = DIMGRID(rdim);
      dim3 dimGrid(dimGridCols, dimGridRows);
      dim3 dimBlock(DIMBLOCK, DIMBLOCK);

#ifdef TIME
	double KADBM_timeStart = getTimeInMSec();
#endif
	k_pixelBasedCostR2L_Float<<<dimGrid, dimBlock>>>(d_imLOGFilterRef, d_imLOGFilterTar, d_imCost, dispRange, rdim, cdim);
	__cudaKernelCheck();
	
#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
#else
        __cudaSafeCall(cudaThreadSynchronize());
#endif
	
#ifdef TIME
	double KADBM_timeStop = getTimeInMSec();
#endif


#ifdef TIME
	double KADBM_D2H_timeStop = getTimeInMSec();
	printf("Time KADBM = %.3fms\n", (KADBM_timeStop - KADBM_timeStart));
	printf("Time KADBM_D2H = %.3fms\n", (KADBM_D2H_timeStop - KADBM_timeStop));
#endif
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

void* pixelBasedCostL2R_Color_Cuda(int rdim, int cdim, int dispRange, int maskSize,int normOrNot)
{
      int dimGridCols = DIMGRID(cdim);
      int dimGridRows = DIMGRID(rdim);
      dim3 dimGrid(dimGridCols, dimGridRows);
      dim3 dimBlock(DIMBLOCK, DIMBLOCK);

#ifdef TIME
	double KADBM_timeStart = getTimeInMSec();
#endif
	k_pixelBasedCostL2R_Color<<<dimGrid, dimBlock>>>(d_imRGBRef, d_imRGBTar, d_imCost, dispRange, rdim, cdim,normOrNot);
	__cudaKernelCheck();
	
#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
#else
        __cudaSafeCall(cudaThreadSynchronize());
#endif
	
#ifdef TIME
	double KADBM_timeStop = getTimeInMSec();
#endif


#ifdef TIME
	double KADBM_D2H_timeStop = getTimeInMSec();
	printf("Time KADBM = %.3fms\n", (KADBM_timeStop - KADBM_timeStart));
	printf("Time KADBM_D2H = %.3fms\n", (KADBM_D2H_timeStop - KADBM_timeStop));
#endif
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

void* pixelBasedCostR2L_Color_Cuda(int rdim, int cdim, int dispRange, int maskSize,int normOrNot)
{
      int dimGridCols = DIMGRID(cdim);
      int dimGridRows = DIMGRID(rdim);
      dim3 dimGrid(dimGridCols, dimGridRows);
      dim3 dimBlock(DIMBLOCK, DIMBLOCK);

#ifdef TIME
	double KADBM_timeStart = getTimeInMSec();
#endif
	k_pixelBasedCostR2L_Color<<<dimGrid, dimBlock>>>(d_imRGBRef, d_imRGBTar, d_imCost, dispRange, rdim, cdim,normOrNot);
	__cudaKernelCheck();
	
#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
#else
        __cudaSafeCall(cudaThreadSynchronize());
#endif
	
#ifdef TIME
	double KADBM_timeStop = getTimeInMSec();
#endif


#ifdef TIME
	double KADBM_D2H_timeStop = getTimeInMSec();
	printf("Time KADBM = %.3fms\n", (KADBM_timeStop - KADBM_timeStart));
	printf("Time KADBM_D2H = %.3fms\n", (KADBM_D2H_timeStop - KADBM_timeStop));
#endif
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

void* pixelBasedCostL2R_Color_Trunc_Cuda(int rdim, int cdim, int dispRange, int maskSize, 
	float trunc, int normOrNot)
{
      int dimGridCols = DIMGRID(cdim);
      int dimGridRows = DIMGRID(rdim);
      dim3 dimGrid(dimGridCols, dimGridRows);
      dim3 dimBlock(DIMBLOCK, DIMBLOCK);

#ifdef TIME
	double KADBM_timeStart = getTimeInMSec();
#endif
	k_pixelBasedCostL2R_Color_Trunc<<<dimGrid, dimBlock>>>(d_imRGBRef, d_imRGBTar, d_imCost, dispRange, 
		rdim, cdim, trunc, normOrNot);
	__cudaKernelCheck();
	
#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
#else
        __cudaSafeCall(cudaThreadSynchronize());
#endif
	
#ifdef TIME
	double KADBM_timeStop = getTimeInMSec();
#endif


#ifdef TIME
	double KADBM_D2H_timeStop = getTimeInMSec();
	printf("Time KADBM = %.3fms\n", (KADBM_timeStop - KADBM_timeStart));
	printf("Time KADBM_D2H = %.3fms\n", (KADBM_D2H_timeStop - KADBM_timeStop));
#endif
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

void* pixelBasedCostR2L_Color_Trunc_Cuda(int rdim, int cdim, int dispRange, int maskSize, 
	float trunc, int normOrNot)
{
      int dimGridCols = DIMGRID(cdim);
      int dimGridRows = DIMGRID(rdim);
      dim3 dimGrid(dimGridCols, dimGridRows);
      dim3 dimBlock(DIMBLOCK, DIMBLOCK);

#ifdef TIME
	double KADBM_timeStart = getTimeInMSec();
#endif
	k_pixelBasedCostR2L_Color_Trunc<<<dimGrid, dimBlock>>>(d_imRGBRef, d_imRGBTar, d_imCost, dispRange, 
		rdim, cdim, trunc, normOrNot);
	__cudaKernelCheck();
	
#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
#else
        __cudaSafeCall(cudaThreadSynchronize());
#endif
	
#ifdef TIME
	double KADBM_timeStop = getTimeInMSec();
#endif


#ifdef TIME
	double KADBM_D2H_timeStop = getTimeInMSec();
	printf("Time KADBM = %.3fms\n", (KADBM_timeStop - KADBM_timeStart));
	printf("Time KADBM_D2H = %.3fms\n", (KADBM_D2H_timeStop - KADBM_timeStop));
#endif
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

void *


unsigned char* stereo_cuda_2(unsigned char* imRef, unsigned char* imTar, unsigned char* imLABRef, unsigned char* imLABTar, int rows, int cols, int dispRange, int maskSize, int maskRankSize, int maskHrad, int maskWrad, int maskCensusSize, float* F_LOG, int maskNCCHrad, int maskNCCWrad, int maskNCCSize, int maskAdapRad, float gamma_similarity, float gamma_proximity, int maskAdapSize)
{
	int dimGridCols = DIMGRID(cols);
    int dimGridRows = DIMGRID(rows);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);

#ifdef TIME
	double KRankBM_timeStart = getTimeInMSec();
#endif
        ///////////////////////////////////////////////////////////////////////////////////////////////
	//kernel - Rank Box Matching
        ///////////////////////////////////////////////////////////////////////////////////////////////
	k_rankFilter<<<dimGrid, dimBlock>>>(d_imRef, d_imRankFilterL, maskRankSize, rows, cols);
	__cudaKernelCheck();
	k_rankFilter<<<dimGrid, dimBlock>>>(d_imTar, d_imRankFilterR, maskRankSize, rows, cols);
	__cudaKernelCheck();
	k_pixelBasedCostL2R<<<dimGrid, dimBlock>>>(d_imRankFilterL, d_imRankFilterR, d_imCost, dispRange, rows, cols);
	__cudaKernelCheck();
	k_boxAggregate<<<dimGrid, dimBlock>>>(d_imCost, dispRange, maskSize, rows, cols);
	__cudaKernelCheck();
	k_WTA<<<dimGrid, dimBlock>>>(d_imCost, d_dispRef, dispRange, rows, cols);
	__cudaKernelCheck();

#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
#else
        __cudaSafeCall(cudaThreadSynchronize());
#endif

#ifdef TIME
        double KRankBM_timeStop = getTimeInMSec();
#endif

        //device to host data transfer
        __cudaSafeCall(cudaMemcpy(dispRef, d_dispRef, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToHost));

#ifdef TIME
        double KRankBM_D2H_timeStop = getTimeInMSec();
	printf("Time KRankBM = %.3fms\n", (KRankBM_timeStop - KRankBM_timeStart));
	printf("Time KRank_D2H = %.3fms\n", (KRankBM_D2H_timeStop - KRankBM_timeStop));
#endif
	printf("--2--\n");	
	return dispRef;
}

unsigned char* stereo_cuda_3(unsigned char* imRef, unsigned char* imTar, unsigned char* imLABRef, unsigned char* imLABTar, int rows, int cols, int dispRange, int maskSize, int maskRankSize, int maskHrad, int maskWrad, int maskCensusSize, float* F_LOG, int maskNCCHrad, int maskNCCWrad, int maskNCCSize, int maskAdapRad, float gamma_similarity, float gamma_proximity, int maskAdapSize)
{
	int dimGridCols = DIMGRID(cols);
      	int dimGridRows = DIMGRID(rows);
      	dim3 dimGrid(dimGridCols, dimGridRows);
      	dim3 dimBlock(DIMBLOCK, DIMBLOCK);

#ifdef TIME
        double KCensusBM_timeStart = getTimeInMSec();
#endif
        ///////////////////////////////////////////////////////////////////////////////////////////////
	//kernel - Census Box Matching
        ///////////////////////////////////////////////////////////////////////////////////////////////
	k_censusTransform<<<dimGrid, dimBlock>>>(d_imRef, d_imCensusTransRef, maskHrad, maskWrad, rows, cols);
	__cudaKernelCheck();
	k_censusTransform<<<dimGrid, dimBlock>>>(d_imTar, d_imCensusTransTar, maskHrad, maskWrad, rows, cols);
	__cudaKernelCheck();
	k_hammingDist<<<dimGrid, dimBlock>>>(d_imCensusTransRef, d_imCensusTransTar, d_imCost, maskHrad, maskWrad, rows, cols,dispRange);
	__cudaKernelCheck();
	k_boxAggregate<<<dimGrid, dimBlock>>>(d_imCost, dispRange, maskSize, rows, cols);
	__cudaKernelCheck();
	k_WTA<<<dimGrid, dimBlock>>>(d_imCost, d_dispRef, dispRange, rows, cols);
	__cudaKernelCheck();

#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
#else
        __cudaSafeCall(cudaThreadSynchronize());
#endif

#ifdef TIME
        double KCensusBM_timeStop = getTimeInMSec();
#endif

        //device to host data transfer
        __cudaSafeCall(cudaMemcpy(dispRef, d_dispRef, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToHost));

#ifdef TIME
        double KCensusBM_D2H_timeStop = getTimeInMSec();
	printf("Time KCensusBM = %.3fms\n", (KCensusBM_timeStop - KCensusBM_timeStart));
	printf("Time KCensusBM_D2H = %.3fms\n", (KCensusBM_D2H_timeStop - KCensusBM_timeStop));
#endif
	printf("--3--\n");	
	return dispRef;
}

unsigned char* stereo_cuda_4(unsigned char* imRef, unsigned char* imTar, unsigned char* imLABRef, unsigned char* imLABTar, int rows, int cols, int dispRange, int maskSize, int maskRankSize, int maskHrad, int maskWrad, int maskCensusSize, float* F_LOG, int maskNCCHrad, int maskNCCWrad, int maskNCCSize, int maskAdapRad, float gamma_similarity, float gamma_proximity, int maskAdapSize)
{
	int dimGridCols = DIMGRID(cols);
      	int dimGridRows = DIMGRID(rows);
      	dim3 dimGrid(dimGridCols, dimGridRows);
      	dim3 dimBlock(DIMBLOCK, DIMBLOCK);
#ifdef TIME
        double KLOGBM_timeStart = getTimeInMSec();
#endif	
        ///////////////////////////////////////////////////////////////////////////////////////////////
	//kernel - LOG Box Matching
        ///////////////////////////////////////////////////////////////////////////////////////////////
	k_convolveImage<<<dimGrid, dimBlock>>>(d_imRef, d_imLOGFilterRef, d_F_LOG, 2, 2, rows, cols);
	__cudaKernelCheck();
	k_convolveImage<<<dimGrid, dimBlock>>>(d_imTar, d_imLOGFilterTar, d_F_LOG, 2, 2, rows, cols);
	__cudaKernelCheck();
	k_pixelBasedCostL2R_Float<<<dimGrid, dimBlock>>>(d_imLOGFilterRef, d_imLOGFilterTar, d_imCost, dispRange, rows, cols);
	__cudaKernelCheck();
	k_boxAggregate<<<dimGrid, dimBlock>>>(d_imCost, dispRange, maskSize, rows, cols);
	__cudaKernelCheck();
	k_WTA<<<dimGrid, dimBlock>>>(d_imCost, d_dispRef, dispRange, rows, cols);
	__cudaKernelCheck();

#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
#else
        __cudaSafeCall(cudaThreadSynchronize());
#endif

#ifdef TIME
        double KLOGBM_timeStop = getTimeInMSec();
#endif

        //device to host data transfer
        __cudaSafeCall(cudaMemcpy(dispRef, d_dispRef, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToHost));

#ifdef TIME
        double KLOGBM_D2H_timeStop = getTimeInMSec();
	printf("Time KLOGBM = %.3fms\n", (KLOGBM_timeStop - KLOGBM_timeStart));
	printf("Time KLOGBM_D2H = %.3fms\n", (KLOGBM_D2H_timeStop - KLOGBM_timeStop));
#endif
	printf("--4--\n");	
	return dispRef;
}

unsigned char* stereo_cuda_5(unsigned char* imRef, unsigned char* imTar, unsigned char* imLABRef, unsigned char* imLABTar, int rows, int cols, int dispRange, int maskSize, int maskRankSize, int maskHrad, int maskWrad, int maskCensusSize, float* F_LOG, int maskNCCHrad, int maskNCCWrad, int maskNCCSize, int maskAdapRad, float gamma_similarity, float gamma_proximity, int maskAdapSize)
{
	int dimGridCols = DIMGRID(cols);
      	int dimGridRows = DIMGRID(rows);
      	dim3 dimGrid(dimGridCols, dimGridRows);
      	dim3 dimBlock(DIMBLOCK, DIMBLOCK);

#ifdef TIME
        double KNCCBM_timeStart = getTimeInMSec();
#endif
        ///////////////////////////////////////////////////////////////////////////////////////////////
	//kernel - NCC Box Matching
        ///////////////////////////////////////////////////////////////////////////////////////////////
	k_nccDisparityL2R<<<dimGrid, dimBlock>>>(d_imRef, d_imTar, d_imCost, dispRange, maskNCCHrad, maskNCCWrad,maskNCCSize, rows, cols);
	__cudaKernelCheck();
	k_boxAggregate<<<dimGrid, dimBlock>>>(d_imCost, dispRange, maskSize, rows, cols);
	__cudaKernelCheck();
	k_WTA_NCC<<<dimGrid, dimBlock>>>(d_imCost, d_dispRef, dispRange, rows, cols);
	__cudaKernelCheck();

#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
#else
        __cudaSafeCall(cudaThreadSynchronize());
#endif

#ifdef TIME
        double KNCCBM_timeStop = getTimeInMSec();
#endif

        //device to host data transfer
        __cudaSafeCall(cudaMemcpy(dispRef, d_dispRef, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToHost));

#ifdef TIME
        double KNCCBM_D2H_timeStop = getTimeInMSec();
	printf("Time KNCCBM = %.3fms\n", (KNCCBM_timeStop - KNCCBM_timeStart));
	printf("Time KNCCBM_D2H = %.3fms\n", (KNCCBM_D2H_timeStop - KNCCBM_timeStop)); 
#endif
	printf("--5--\n");	
	return dispRef;
}

unsigned char* stereo_cuda_6(unsigned char* imRef, unsigned char* imTar, unsigned char* imLABRef, unsigned char* imLABTar, int rows, int cols, int dispRange, int maskSize, int maskRankSize, int maskHrad, int maskWrad, int maskCensusSize, float* F_LOG, int maskNCCHrad, int maskNCCWrad, int maskNCCSize, int maskAdapRad, float gamma_similarity, float gamma_proximity, int maskAdapSize)
{
	int dimGridCols = DIMGRID(cols);
      	int dimGridRows = DIMGRID(rows);
      	dim3 dimGrid(dimGridCols, dimGridRows);
      	dim3 dimBlock(DIMBLOCK, DIMBLOCK);

#ifdef TIME
        double KAdapW_timeStart = getTimeInMSec();
#endif
        ///////////////////////////////////////////////////////////////////////////////////////////////
	//Adaptive Weighting Aggregation
        ///////////////////////////////////////////////////////////////////////////////////////////////
	k_pixelBasedCostL2R<<<dimGrid, dimBlock>>>(d_imRef, d_imTar, d_imCost, dispRange, rows, cols);
	__cudaKernelCheck();
	calcProxWeight(proxWeight, gamma_proximity, maskAdapRad);
        __cudaSafeCall(cudaMemcpy(d_proxWeight, proxWeight, maskAdapSize * sizeof(float), cudaMemcpyHostToDevice));
	k_computeWeight<<<dimGrid, dimBlock>>>(d_weightRef, d_proxWeight, d_imLABRef, gamma_similarity, rows, cols, maskAdapRad);
	__cudaKernelCheck();
	k_computeWeight<<<dimGrid, dimBlock>>>(d_weightTar, d_proxWeight, d_imLABTar, gamma_similarity, rows, cols, maskAdapRad);
	__cudaKernelCheck();
	k_calcAWCostL2R<<<dimGrid, dimBlock>>>(d_weightRef, d_weightTar, d_imCost, d_cost_new_image, rows, cols, dispRange, maskAdapRad);
	__cudaKernelCheck();
	k_WTA<<<dimGrid, dimBlock>>>(d_cost_new_image, d_dispRef, dispRange, rows, cols);
	__cudaKernelCheck();
			
#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
#else
        __cudaSafeCall(cudaThreadSynchronize());
#endif

#ifdef TIME
        double KAdapW_timeStop = getTimeInMSec();
#endif

        //device to host data transfer
        __cudaSafeCall(cudaMemcpy(dispRef, d_dispRef, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToHost));

#ifdef TIME
        double KAdapW_D2H_timeStop = getTimeInMSec();
	printf("Time KAdapW = %.3fms\n", (KAdapW_timeStop - KAdapW_timeStart));
	printf("Time KAdapW_D2H = %.3fms\n", (KAdapW_D2H_timeStop - KAdapW_timeStop));
#endif
	printf("--6--\n");	
	return dispRef;
}

unsigned char* stereo_cuda_7(unsigned char* imRef, unsigned char* imTar, unsigned char* imLABRef, unsigned char* imLABTar, int rows, int cols, int dispRange, int maskSize, int maskRankSize, int maskHrad, int maskWrad, int maskCensusSize, float* F_LOG, int maskNCCHrad, int maskNCCWrad, int maskNCCSize, int maskAdapRad, float gamma_similarity, float gamma_proximity, int maskAdapSize)
{
	int dimGridCols = DIMGRID(cols);
      	int dimGridRows = DIMGRID(rows);
      	dim3 dimGrid(dimGridCols, dimGridRows);
      	dim3 dimBlock(DIMBLOCK, DIMBLOCK);
		//1 dimension
		//aggregate_cost_horizontal(float *cost_image, int rdim, int cdim, int dispRange)
		//2 dimension
		//findCross(const unsigned char *image, int *HorzOffset, int *VertOffset, int rdim, int cdim)
		//k_crossStereo<<<dimGrid, dimBlock>>>(d_imRef, d_imTar, d_dispRef, d_dispTar, d_imCost, dispRange, rows, cols)
#ifdef TIME
        double KCrossBased_timeStart = getTimeInMSec();
#endif
	
        ///////////////////////////////////////////////////////////////////////////////////////////////	
	//kernel - Cross Based Aggregation
        ///////////////////////////////////////////////////////////////////////////////////////////////
	k_crossStereo<<<dimGrid, dimBlock>>>(d_imRef, d_imTar, d_dispRef, d_dispTar, d_imCost, dispRange, rows, cols);
	__cudaKernelCheck();

#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
#else
        __cudaSafeCall(cudaThreadSynchronize());
#endif

#ifdef TIME
        double KCrossBased_timeStop = getTimeInMSec();
#endif

        //device to host data transfer
        __cudaSafeCall(cudaMemcpy(dispRef, d_dispRef, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToHost));

#ifdef TIME
        double KCrossBased_D2H_timeStop = getTimeInMSec();
	printf("Time KCrossBased = %.3fms\n", (KCrossBased_timeStop - KCrossBased_timeStart));
	printf("Time KCrossBased = %.3fms\n", (KCrossBased_D2H_timeStop - KCrossBased_timeStop));
#endif
	printf("--7--\n");	
	return dispRef;
}


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

double getTimeInMSec(){
	//struct timeval t;
	//gettimeofday(&t, NULL);
	return (double) 1;//(t.tv_sec * 1e+3 + t.tv_usec * 1e-3);
}

void ptrErrorCheck(void* ptr){
	if(ptr == NULL){
		printf("Error: Pointer is NULL. Memory allocation error\n");
		exit(-1);
	}
}

void calcProxWeight(float* proxWeight, float gamma_proximity, int maskRad)
{
	int k = 0, x, y;
	for(y = -maskRad; y <= maskRad; y++){
		for(x = -maskRad; x <= maskRad; x++){
			proxWeight[k] = exp(-sqrt((float)(y * y + x * x) / gamma_proximity));
			k++;
		}
	}
}

