//////////////////////////////////////////////////////////////////////
//Name: PixelBasedCost.cu
//Created date: 3-2-2012
//Modified date: 3-2-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: calculates pixel based cost of stereo image - cuda kernel
/////////////////////////////////////////////////////////////////////

#ifndef _PIXELBASEDCOST_KERNEL_
#define _PIXELBASEDCOST_KERNEL_


__global__ void pixelBasedCostL2R(const unsigned char* ref_image, const unsigned char* tar_image,
	 float* cost, const int dispRange, const int rdim, const int cdim){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * cdim + x;
	if(x<cdim && y<rdim){
		for(int d=0; d<dispRange; d++) {
			int tarc = x-d;
			if(tarc<0) tarc += cdim;
			cost[d+dispRange*idx] = abs(ref_image[idx]-tar_image[y * cdim + tarc]);
		}
	}
}


__global__ void pixelBasedCostR2L(const unsigned char *ref_image, 
	const unsigned char *tar_image, float *cost, 
	const int dispRange, const int rdim, const int cdim){
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x<cdim && y<rdim){
		int idx = y*cdim+x, xtar = 0;
		for(int d=0; d<dispRange; d++){
			int xtar = x+d;
			if(xtar>=cdim) xtar-=cdim;
			cost[d+dispRange*idx] = abs(ref_image[idx]-tar_image[y*cdim + xtar]);
		}
	}
}

__global__ void pixelBasedCostL2R_Float(const float *ref_image, 
	const float *tar_image, float *cost, 
	const int dispRange, const int rdim, const int cdim){
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y*cdim+x, xtar = 0;
	float res = 0;
	for(int d=0; d<dispRange; d++){
		xtar = x-d;
		if(xtar<0) xtar+=cdim;
		res = ref_image[idx]-tar_image[y * cdim + xtar];
		if(res<0) res=-res;
			cost[d+dispRange*idx] = res;
	}
}


__global__ void pixelBasedCostR2L_Float(const float *ref_image, 
	const float *tar_image, float *cost, 
	const int dispRange, const int rdim, const int cdim){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y*cdim+x,xtar = 0;
	float res = 0;
	for(int d=0; d<dispRange; d++){
		xtar = x+d;
		if(xtar>=cdim) xtar-=cdim;
		res = ref_image[idx]-tar_image[y * cdim + xtar];
		if(res<0) res=-res;
			cost[d+dispRange*idx] = res;
	}
}


__global__ void pixelBasedCostL2R_Color(const unsigned char *ref_image, 
	const unsigned char *tar_image, float *cost, 
	const int dispRange, const int rdim, const int cdim,  const int normOrNot){
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y*cdim+x, idxtar = 0, xtar = 0;
	for(int d=0; d<dispRange; d++){
		xtar = x-d;
		if(xtar<0) xtar+=cdim;
		idxtar = y*cdim + xtar;
		float coeff = (float)1/(float)3;
		if(normOrNot == 0)
			cost[d+dispRange*idx] = abs(ref_image[idx*3]-tar_image[idxtar*3]) +
									abs(ref_image[idx*3+1]-tar_image[idxtar*3+1])+
									abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]);
		else
			cost[d+dispRange*idx] = coeff*(abs(ref_image[idx*3]-tar_image[idxtar*3]) +
									abs(ref_image[idx*3+1]-tar_image[idxtar*3+1])+
									abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]));
	}
}


__global__ void pixelBasedCostL2R_Color_Trunc(const unsigned char *ref_image, 
	const unsigned char *tar_image, float *cost, 
	const int dispRange, const int rdim, const int cdim, const float trunc, const int normOrNot){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y*cdim+x, idxtar = 0, xtar = 0;
	float tempCost = 0;
	for(int d=0; d<dispRange; d++){
		xtar = x-d;
		if(xtar<0) xtar+=cdim;
		idxtar = y*cdim + xtar;
		float coeff = (float)1/(float)3;
		if(normOrNot == 0)
			tempCost = abs(ref_image[idx*3]-tar_image[idxtar*3]) +
					   abs(ref_image[idx*3+1]-tar_image[idxtar*3+1]) +
					   abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]);
		else
			tempCost = coeff*(abs(ref_image[idx*3]-tar_image[idxtar*3]) +
					          abs(ref_image[idx*3+1]-tar_image[idxtar*3+1])+
							  abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]));
		if(trunc<tempCost)
			cost[d+dispRange*idx] = trunc;
		else
			cost[d+dispRange*idx] = tempCost;
	}
}

__global__ void pixelBasedCostR2L_Color(const unsigned char *ref_image, 
	const unsigned char *tar_image, float *cost, 
	const int dispRange, const int rdim, const int cdim, const int normOrNot){
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y*cdim+x, idxtar = 0, xtar = 0;
	for(int d=0; d<dispRange; d++){
		xtar = x+d;
		if(xtar>=cdim) xtar-=cdim;
		idxtar = y*cdim + xtar;
		float coeff = (float)1/(float)3;
		if(normOrNot==0)
			cost[d+dispRange*idx] = (abs(ref_image[idx*3]-tar_image[idxtar*3]) +
									abs(ref_image[idx*3+1]-tar_image[idxtar*3+1]) + 
									abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]));
		else
			cost[d+dispRange*idx] = coeff*(abs(ref_image[idx*3]-tar_image[idxtar*3]) +
									abs(ref_image[idx*3+1]-tar_image[idxtar*3+1]) + 
									abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]));
	}
}

__global__ void pixelBasedCostR2L_Color_Trunc(const unsigned char *ref_image, 
	const unsigned char *tar_image, float *cost, 
	const int dispRange, const int rdim, const int cdim, const float trunc, const int normOrNot){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y*cdim+x, idxtar = 0, xtar = 0;
	float tempCost = 0;
	idx = y*cdim+x;
	for(int d=0; d<dispRange; d++){
		xtar = x+d;
		if(xtar>=cdim) xtar-=cdim;
		idxtar = y*cdim + xtar;
		float coeff = (float)1/(float)3;
		if(normOrNot == 0)
			tempCost = abs(ref_image[idx*3]-tar_image[idxtar*3]) +
					   abs(ref_image[idx*3+1]-tar_image[idxtar*3+1]) +
					   abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]);
		else
			tempCost = coeff*(abs(ref_image[idx*3]-tar_image[idxtar*3]) +
					          abs(ref_image[idx*3+1]-tar_image[idxtar*3+1])+
							  abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]));
		if(trunc<tempCost)
			cost[d+dispRange*idx] = trunc;
		else
			cost[d+dispRange*idx] = tempCost;
	}
}

#endif

