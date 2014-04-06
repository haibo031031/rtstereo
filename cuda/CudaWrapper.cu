//////////////////////////////////////////////////////////////////////
//Name: AdaptiveWeightAggregation.cu 
//Created date: 4-2-2012
//Modified date: 4-2-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: Cuda Wrapper for cuda functions
//////////////////////////////////////////////////////////////////////
#include "AdaptiveWeightAggregation.cu"
#include "CensusTransform.cu"
#include "ConstBlockAggregation.cu"
#include "ConvolveImage.cu"
#include "CrossBasedAggregation.cu"
#include "CrossCheck.cu"
#include "NCCDisparity.cu"
#include "PixelBasedCost.cu"
#include "RankFilter.cu"
#include "WTA.cu"
#include "CombineCost.cu"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "jutil_cuda.h"

#define DIMBLOCK 16
#define DIMGRID(x) x / DIMBLOCK + ((x % DIMBLOCK == 0)?0:1);
#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5))
#define TIME

void __cudaSafeCall(cudaError_t err);
void __cudaKernelCheck(void);
//double getTimeInMSec();
//void ptrErrorCheck(void* ptr);
void calcProxWeight(float gamma_proximity, int maskRad);
//#define VERIFY



unsigned char *d_ref_image, *d_tar_image, *d_ref_disparity,*dispReturn;
unsigned char *d_ref_rgb_image, *d_tar_rgb_image;
float *d_cost_image1, *d_cost_image2, *d_new_cost_image;
unsigned char *d_rankFiltImL, *d_rankFiltImR;
bool *d_censusTrnsR, *d_censusTrnsT;
float *d_LoG_ref_image, *d_LoG_tar_image, *d_F_LOG;
unsigned char  *d_ref_lab_image, *d_tar_lab_image;
float *d_proxWeight, *d_weightRef, *d_weightTar,*proxWeight;
int *d_horz_offset_ref, *d_horz_offset_tar, *d_vert_offset_ref, *d_vert_offset_tar;



void allocateDeviceMem(int rdim, int cdim, int dispRange, int maskCensusArea, 
	int maskAdapArea, int costChoice1, int costChoice2, int aggrChoice){
	
	proxWeight = (float *)malloc(maskAdapArea*sizeof(float));
	dispReturn = (unsigned char*)malloc(rdim*cdim*sizeof(unsigned char));
	__cudaSafeCall(cudaMalloc((void**)&d_ref_image, rdim * cdim * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_tar_image, rdim * cdim * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_ref_disparity, rdim * cdim * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_ref_rgb_image, rdim * cdim * 3 * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_tar_rgb_image, rdim * cdim * 3 * sizeof(unsigned char)));
	__cudaSafeCall(cudaMalloc((void**)&d_cost_image1, rdim * cdim * dispRange * sizeof(float)));
	__cudaSafeCall(cudaMalloc((void**)&d_F_LOG, 25 * sizeof(float)));
	if(costChoice2 != 0)
		__cudaSafeCall(cudaMalloc((void**)&d_cost_image2, rdim * cdim * dispRange * sizeof(float)));
	if(aggrChoice == 3 || aggrChoice == 2)
		__cudaSafeCall(cudaMalloc((void**)&d_new_cost_image, rdim * cdim * dispRange * sizeof(float)));
	if(costChoice1 == 2 || costChoice2 == 2){
		__cudaSafeCall(cudaMalloc((void**)&d_rankFiltImL, rdim * cdim * sizeof(unsigned char)));
		__cudaSafeCall(cudaMalloc((void**)&d_rankFiltImR, rdim * cdim * sizeof(unsigned char)));
	}
	if(costChoice1 == 3 || costChoice2 == 3){
		__cudaSafeCall(cudaMalloc((void**)&d_censusTrnsR, rdim * cdim * maskCensusArea * sizeof(bool)));
		__cudaSafeCall(cudaMalloc((void**)&d_censusTrnsT, rdim * cdim * maskCensusArea * sizeof(bool)));
	}
	if(costChoice1 == 4 || costChoice2 == 4){
		__cudaSafeCall(cudaMalloc((void**)&d_LoG_ref_image, rdim * cdim * sizeof(float)));
		__cudaSafeCall(cudaMalloc((void**)&d_LoG_tar_image, rdim * cdim * sizeof(float)));
	}
	if(aggrChoice == 2){
		__cudaSafeCall(cudaMalloc((void**)&d_proxWeight, maskAdapArea * sizeof(float)));
		__cudaSafeCall(cudaMalloc((void**)&d_weightRef, rdim * cdim * maskAdapArea * sizeof(float)));
		__cudaSafeCall(cudaMalloc((void**)&d_weightTar, rdim * cdim * maskAdapArea * sizeof(float)));
		__cudaSafeCall(cudaMalloc((void**)&d_ref_lab_image, rdim * cdim * 3 * sizeof(unsigned char)));
		__cudaSafeCall(cudaMalloc((void**)&d_tar_lab_image, rdim * cdim * 3 * sizeof(unsigned char)));
	}
	if(aggrChoice == 3){
		__cudaSafeCall(cudaMalloc((void**)&d_horz_offset_ref, rdim * cdim * 2 * sizeof(int)));
		__cudaSafeCall(cudaMalloc((void**)&d_horz_offset_tar, rdim * cdim * 2 * sizeof(int)));
		__cudaSafeCall(cudaMalloc((void**)&d_vert_offset_ref, rdim * cdim * 2 * sizeof(int)));
		__cudaSafeCall(cudaMalloc((void**)&d_vert_offset_tar, rdim * cdim * 2 * sizeof(int)));
	}
}
void copyHost2Device(unsigned char *ref_image, unsigned char *tar_image, unsigned char *RGB_left_pnt,
	unsigned char *RGB_right_pnt, unsigned char *LAB_left_pnt, unsigned char *LAB_right_pnt,
	int rdim, int cdim, int aggrChoice){

	float F_LOG[25] = { 0.0239, 0.046, 0.0499, 0.046, 0.0239,
						0.046, 0.0061, -0.0923, 0.0061, 0.046,
						0.0499, -0.0923, -0.3182, -0.0923, 0.0499,
						0.046, 0.0061, -0.0923, 0.0061, 0.046,
						0.0239, 0.046, 0.0499, 0.046, 0.0239 };
	__cudaSafeCall(cudaMemcpy(d_F_LOG, F_LOG, 25 * sizeof(float), cudaMemcpyHostToDevice));
	__cudaSafeCall(cudaMemcpy(d_ref_image, ref_image, rdim * cdim * sizeof(unsigned char), cudaMemcpyHostToDevice));
	__cudaSafeCall(cudaMemcpy(d_tar_image, tar_image, rdim * cdim * sizeof(unsigned char), cudaMemcpyHostToDevice));
	__cudaSafeCall(cudaMemcpy(d_ref_rgb_image, RGB_left_pnt, rdim * cdim * 3* sizeof(unsigned char), cudaMemcpyHostToDevice));
	__cudaSafeCall(cudaMemcpy(d_tar_rgb_image, RGB_right_pnt, rdim * cdim * 3* sizeof(unsigned char), cudaMemcpyHostToDevice));
	if(aggrChoice == 2){
		__cudaSafeCall(cudaMemcpy(d_ref_lab_image, LAB_left_pnt, rdim * cdim * 3* sizeof(unsigned char), cudaMemcpyHostToDevice));
		__cudaSafeCall(cudaMemcpy(d_tar_lab_image, LAB_right_pnt, rdim * cdim * 3* sizeof(unsigned char), cudaMemcpyHostToDevice));
	}
}
void deallocateDeviceMem(int costChoice1, int costChoice2,int aggrChoice){
	if(d_ref_image)
		__cudaSafeCall(cudaFree(d_ref_image));
	if(d_tar_image)
		__cudaSafeCall(cudaFree(d_tar_image));
	if(d_ref_disparity)
		__cudaSafeCall(cudaFree(d_ref_disparity));
	if(d_ref_rgb_image)
		__cudaSafeCall(cudaFree(d_ref_rgb_image));
	if(d_tar_rgb_image)
		__cudaSafeCall(cudaFree(d_tar_rgb_image));
	if(d_cost_image1)
		__cudaSafeCall(cudaFree(d_cost_image1));
	if(costChoice2 != 0)
		if(d_cost_image2)
			__cudaSafeCall(cudaFree(d_cost_image2));
	if(aggrChoice == 3 || aggrChoice == 2)
		if(d_new_cost_image)
			__cudaSafeCall(cudaFree(d_new_cost_image));
	if(costChoice1 == 2 || costChoice2 == 2){
		if(d_rankFiltImL)
			__cudaSafeCall(cudaFree(d_rankFiltImL));
		if(d_rankFiltImR)
			__cudaSafeCall(cudaFree(d_rankFiltImR));
	}
	if(costChoice1 == 3 || costChoice2 == 3){
		if(d_censusTrnsR)
			__cudaSafeCall(cudaFree(d_censusTrnsR));
		if(d_censusTrnsT)
			__cudaSafeCall(cudaFree(d_censusTrnsT));
	}
	if(costChoice1 == 4 || costChoice2 == 4){
		if(d_LoG_ref_image)
			__cudaSafeCall(cudaFree(d_LoG_ref_image));
		if(d_LoG_tar_image)
			__cudaSafeCall(cudaFree(d_LoG_tar_image));
		if(d_F_LOG)
			__cudaSafeCall(cudaFree(d_F_LOG));
	}
	if(aggrChoice == 2){
		if(d_proxWeight)
			__cudaSafeCall(cudaFree(d_proxWeight));
		if(d_weightRef)
			__cudaSafeCall(cudaFree(d_weightRef));
		if(d_weightTar)
			__cudaSafeCall(cudaFree(d_weightTar));
		if(d_ref_lab_image)
			__cudaSafeCall(cudaFree(d_ref_lab_image));
		if(d_tar_lab_image)
			__cudaSafeCall(cudaFree(d_tar_lab_image));
	}
	if(aggrChoice == 3){
		if(d_horz_offset_ref)
			__cudaSafeCall(cudaFree(d_horz_offset_ref));
		if(d_horz_offset_tar)
			__cudaSafeCall(cudaFree(d_horz_offset_tar));
		if(d_vert_offset_ref)
			__cudaSafeCall(cudaFree(d_vert_offset_ref));
		if(d_vert_offset_tar)
			__cudaSafeCall(cudaFree(d_vert_offset_tar));
	}

}
void calcProxWeight(float gamma_proximity, int maskRad)
{
	int k = 0, x, y;
	for(y = -maskRad; y <= maskRad; y++){
		for(x = -maskRad; x <= maskRad; x++){
			proxWeight[k] = exp(-sqrt((float)(y * y + x * x) / gamma_proximity));
			k++;
		}
	}
}
void pixelBasedCostL2R_cuda(int rdim, int cdim, int dispRange, int costChoice){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	if(costChoice == 1){
		pixelBasedCostL2R<<<dimGrid, dimBlock>>>(d_ref_image, d_tar_image, d_cost_image1, dispRange, rdim, cdim);
		__cudaKernelCheck();
	}
	else{
		pixelBasedCostL2R<<<dimGrid, dimBlock>>>(d_ref_image, d_tar_image, d_cost_image2, dispRange, rdim, cdim);
		__cudaKernelCheck();
	}
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}
void pixelBasedCostR2L_cuda(int rdim, int cdim, int dispRange, int costChoice){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	if(costChoice == 1){
		pixelBasedCostR2L<<<dimGrid, dimBlock>>>(d_ref_image, d_tar_image, d_cost_image1, dispRange, rdim, cdim);
		__cudaKernelCheck();
	}
	else{
		pixelBasedCostR2L<<<dimGrid, dimBlock>>>(d_ref_image, d_tar_image, d_cost_image2, dispRange, rdim, cdim);
		__cudaKernelCheck();
	}
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}
void pixelBasedCostL2R_Float_cuda(int rdim, int cdim, int dispRange,int costChoice){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	if(costChoice == 1){
		pixelBasedCostL2R_Float<<<dimGrid, dimBlock>>>(d_LoG_ref_image, d_LoG_tar_image, d_cost_image1, dispRange, rdim, cdim);
		__cudaKernelCheck();
	}
	else{
		pixelBasedCostL2R_Float<<<dimGrid, dimBlock>>>(d_LoG_ref_image, d_LoG_tar_image, d_cost_image2, dispRange, rdim, cdim);
		__cudaKernelCheck();
	}
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}
void pixelBasedCostR2L_Float_cuda(int rdim, int cdim, int dispRange, int costChoice){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	if(costChoice == 1){
		pixelBasedCostR2L_Float<<<dimGrid, dimBlock>>>(d_LoG_ref_image, d_LoG_tar_image, d_cost_image1, dispRange, rdim, cdim);
		__cudaKernelCheck();
	}
	else{
		pixelBasedCostR2L_Float<<<dimGrid, dimBlock>>>(d_LoG_ref_image, d_LoG_tar_image, d_cost_image2, dispRange, rdim, cdim);
		__cudaKernelCheck();
	}
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}
void pixelBasedCostL2R_Color_cuda(int rdim, int cdim, int dispRange, int normOrNot, int costChoice){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	if(costChoice == 1){
		pixelBasedCostL2R_Color<<<dimGrid, dimBlock>>>(d_ref_rgb_image, d_tar_rgb_image, d_cost_image1, dispRange, rdim, cdim, normOrNot);
		__cudaKernelCheck();
	}
	else{
		pixelBasedCostL2R_Color<<<dimGrid, dimBlock>>>(d_ref_rgb_image, d_tar_rgb_image, d_cost_image2, dispRange, rdim, cdim, normOrNot);
		__cudaKernelCheck();
	}
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}
void pixelBasedCostR2L_Color_cuda(int rdim, int cdim, int dispRange, int normOrNot, int costChoice){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	if(costChoice == 1){
		pixelBasedCostR2L_Color<<<dimGrid, dimBlock>>>(d_ref_rgb_image, d_tar_rgb_image, d_cost_image1, dispRange, rdim, cdim, normOrNot);
		__cudaKernelCheck();
	}
	else{
		pixelBasedCostR2L_Color<<<dimGrid, dimBlock>>>(d_ref_rgb_image, d_tar_rgb_image, d_cost_image2, dispRange, rdim, cdim, normOrNot);
		__cudaKernelCheck();
	}
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}
void pixelBasedCostL2R_Color_Trunc_cuda(int rdim, int cdim, int dispRange, float trunc, int normOrNot, int costChoice){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	if(costChoice == 1){
		pixelBasedCostL2R_Color_Trunc<<<dimGrid, dimBlock>>>(d_ref_rgb_image, d_tar_rgb_image, d_cost_image1, dispRange, rdim, cdim, trunc, normOrNot);
		__cudaKernelCheck();
	}
	else{
		pixelBasedCostL2R_Color_Trunc<<<dimGrid, dimBlock>>>(d_ref_rgb_image, d_tar_rgb_image, d_cost_image2, dispRange, rdim, cdim, trunc, normOrNot);
		__cudaKernelCheck();
	}
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}
void pixelBasedCostR2L_Color_Trunc_cuda(int rdim, int cdim, int dispRange, float trunc, int normOrNot, int costChoice){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	if(costChoice == 1){
		pixelBasedCostR2L_Color_Trunc<<<dimGrid, dimBlock>>>(d_ref_rgb_image, d_tar_rgb_image, d_cost_image1, dispRange, rdim, cdim, trunc, normOrNot);
		__cudaKernelCheck();
	}
	else{
		pixelBasedCostR2L_Color_Trunc<<<dimGrid, dimBlock>>>(d_ref_rgb_image, d_tar_rgb_image, d_cost_image2, dispRange, rdim, cdim, trunc, normOrNot);
		__cudaKernelCheck();
	}
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}
void calcRankCost_cuda(int rdim, int cdim, int dispRange, int maskSize, int costChoice){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	calculateRankFilt<<<dimGrid, dimBlock>>>(d_ref_image,d_rankFiltImL,maskSize,rdim,cdim);
	__cudaKernelCheck();
	calculateRankFilt<<<dimGrid, dimBlock>>>(d_tar_image,d_rankFiltImR,maskSize,rdim,cdim);
	__cudaKernelCheck();
	if(costChoice == 1){
		pixelBasedCostL2R<<<dimGrid, dimBlock>>>(d_rankFiltImL,d_rankFiltImR,d_cost_image1,dispRange,rdim,cdim);
		__cudaKernelCheck();
	}
	else{
		pixelBasedCostL2R<<<dimGrid, dimBlock>>>(d_rankFiltImL,d_rankFiltImR,d_cost_image2,dispRange,rdim,cdim);
		__cudaKernelCheck();
	}
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}
void calcCensusCost_cuda(int rdim, int cdim, int dispRange, int maskHrad, int maskWrad, int costChoice){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	calcCensusTrans<<<dimGrid, dimBlock>>>(d_ref_image,d_censusTrnsR,maskHrad,maskWrad,rdim,cdim);
	__cudaKernelCheck();
	calcCensusTrans<<<dimGrid, dimBlock>>>(d_tar_image,d_censusTrnsT,maskHrad,maskWrad,rdim,cdim);
	__cudaKernelCheck();
	if(costChoice == 1){
		calcHammingDist<<<dimGrid, dimBlock>>>(d_censusTrnsR,d_censusTrnsT,d_cost_image1,maskHrad,maskWrad,rdim,cdim,dispRange);
		__cudaKernelCheck();
	}
	else{
		calcHammingDist<<<dimGrid, dimBlock>>>(d_censusTrnsR,d_censusTrnsT,d_cost_image2,maskHrad,maskWrad,rdim,cdim,dispRange);
		__cudaKernelCheck();
	}
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}
void calcLoGCost_cuda(int rdim, int cdim, int dispRange, int costChoice){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	convolveImage<<<dimGrid, dimBlock>>>(d_ref_image,d_LoG_ref_image,d_F_LOG,2,2,rdim,cdim);
	__cudaKernelCheck();
	convolveImage<<<dimGrid, dimBlock>>>(d_tar_image,d_LoG_tar_image,d_F_LOG,2,2,rdim,cdim);
	__cudaKernelCheck();
	if(costChoice == 1){
		pixelBasedCostL2R_Float<<<dimGrid, dimBlock>>>(d_LoG_ref_image, d_LoG_tar_image, d_cost_image1, dispRange, rdim, cdim);
		__cudaKernelCheck();
	}
	else{
		pixelBasedCostL2R_Float<<<dimGrid, dimBlock>>>(d_LoG_ref_image, d_LoG_tar_image, d_cost_image2, dispRange, rdim, cdim);
		__cudaKernelCheck();
	}
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}
void calcBoxAggregationCost_cuda(int rdim, int cdim, int dispRange, int maskRad){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	boxAggregate<<<dimGrid, dimBlock>>>(d_cost_image1,dispRange,maskRad,rdim,cdim);
	__cudaKernelCheck();
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}
void calcAdaptiveAggregationCost_cuda(float gamma_proximity, float gamma_similarity, 
	int rdim, int cdim, int dispRange, int maskAdapRad, int maskAdapArea){
	
	calcProxWeight(gamma_proximity,maskAdapRad);
	__cudaSafeCall(cudaMemcpy(d_proxWeight, proxWeight, maskAdapArea * sizeof(float), cudaMemcpyHostToDevice));
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	computeWeights<<<dimGrid, dimBlock>>>(d_weightRef,d_proxWeight,d_ref_lab_image,gamma_similarity,rdim,cdim,maskAdapRad);
	__cudaKernelCheck();
	computeWeights<<<dimGrid, dimBlock>>>(d_weightTar,d_proxWeight,d_tar_lab_image,gamma_similarity,rdim,cdim,maskAdapRad);
	__cudaKernelCheck();
	calcAWCostL2R<<<dimGrid, dimBlock>>>(d_weightRef,d_weightTar,d_cost_image1,d_new_cost_image,rdim,cdim,dispRange,maskAdapRad);
	__cudaKernelCheck();
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}
void calcCrossAggregation_cuda(int rdim, int cdim, int dispRange){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
	
	//set 1 dimension
	dim3 dimGrid1(1,dimGridRows);   
    dim3 dimBlock1(1,DIMBLOCK*DIMBLOCK);

	//set 2 dimension
    dim3 dimGrid2(dimGridCols, dimGridRows);
    dim3 dimBlock2(DIMBLOCK, DIMBLOCK);	

	findCross<<<dimGrid2, dimBlock2>>>(d_ref_rgb_image, d_horz_offset_ref, d_vert_offset_ref, rdim, cdim);
	__cudaKernelCheck();
	findCross<<<dimGrid2, dimBlock2>>>(d_tar_rgb_image, d_horz_offset_tar, d_vert_offset_tar, rdim, cdim);
	__cudaKernelCheck();
	aggregate_cost_horizontal<<<dimGrid1,dimBlock1>>>(d_cost_image1, rdim, cdim, dispRange);
	__cudaKernelCheck();
	cross_stereo_aggregation<<<dimGrid2, dimBlock2>>>(d_cost_image1, d_new_cost_image, d_horz_offset_ref, d_horz_offset_tar,
		d_vert_offset_ref, d_vert_offset_tar, rdim, cdim, dispRange);
	__cudaKernelCheck();
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}
unsigned char *WTA_cuda(int aggrChoice, int rdim, int cdim, int dispRange){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	if(aggrChoice == 1){
		WTA<<<dimGrid, dimBlock>>>(d_cost_image1,d_ref_disparity,dispRange,rdim,cdim);
		__cudaKernelCheck();
	}
	else{
		WTA<<<dimGrid, dimBlock>>>(d_new_cost_image,d_ref_disparity,dispRange,rdim,cdim);
		__cudaKernelCheck();
	}
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
	__cudaSafeCall(cudaMemcpy(dispReturn, d_ref_disparity, rdim * cdim * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	return dispReturn;
}
void calcCombinedCost_cuda(float lambda1, float lambda2, int rdim, int cdim, int dispRange){
	int dimGridCols = DIMGRID(cdim);
    int dimGridRows = DIMGRID(rdim);
    dim3 dimGrid(dimGridCols, dimGridRows);
    dim3 dimBlock(DIMBLOCK, DIMBLOCK);
	combineCost<<<dimGrid, dimBlock>>>(d_cost_image1,d_cost_image2,lambda1,lambda2,rdim,cdim,dispRange);
	__cudaKernelCheck();
	#if CUDART_VERSION >= 4000
        __cudaSafeCall(cudaDeviceSynchronize());
	#else
        __cudaSafeCall(cudaThreadSynchronize());
	#endif
}