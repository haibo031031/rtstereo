//////////////////////////////////////////////////////////////////////
//Name: PixelBasedCost.cpp
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: Calculates pixel based cost
//////////////////////////////////////////////////////////////////////

#include "PixelBasedCost.h"
#define maxCostColor 765
#define maxCostInten 255

void pixelBasedL2R(unsigned char *ref_image, unsigned char *tar_image, float *cost, const int dispRange, const int rdim, const int cdim){
	int idx = 0,xtar = 0;
	for (int y=0; y<rdim; y++)
		for(int x=0; x<cdim;x++){
			idx = y*cdim+x;
			for(int d=0; d<dispRange; d++){
				xtar = x-d;
				if(xtar<0) xtar+=cdim;
				cost[d+dispRange*idx] = abs(ref_image[idx]-tar_image[y*cdim + xtar]);
			}
		}
}

void pixelBasedR2L(unsigned char *ref_image, unsigned char *tar_image, float *cost, const int dispRange, const int rdim, const int cdim){
	int idx = 0,xtar = 0;
	for (int y=0; y<rdim; y++)
		for(int x=0; x<cdim;x++){
			idx = y*cdim+x;
			for(int d=0; d<dispRange; d++){
				xtar = x+d;
				if(xtar>=cdim) xtar-=cdim;
				cost[d+dispRange*idx] = abs(ref_image[idx]-tar_image[y*cdim + xtar]);
			}
		}
}

void pixelBasedCostL2R_Float(float *ref_image, float *tar_image, float *cost, const int dispRange, const int rdim, const int cdim){
	int idx = 0,xtar = 0;
	float res = 0;
	for (int y=0; y<rdim; y++)
		for(int x=0; x<cdim;x++){
			idx = y*cdim+x;
			for(int d=0; d<dispRange; d++){
				xtar = x-d;
				if(xtar<0) xtar+=cdim;
				res = ref_image[idx]-tar_image[y * cdim + xtar];
				if(res<0) res=-res;
				cost[d+dispRange*idx] = res;
			}
		}
}

void pixelBasedCostR2L_Float(float *ref_image, float *tar_image, float *cost, const int dispRange, const int rdim, const int cdim){
	int idx = 0,xtar = 0;
	float res = 0;
	for (int y=0; y<rdim; y++)
		for(int x=0; x<cdim;x++){
			idx = y*cdim+x;
			for(int d=0; d<dispRange; d++){
				xtar = x+d;
				if(xtar>=cdim) xtar-=cdim;
				res = ref_image[idx]-tar_image[y * cdim + xtar];
				if(res<0) res=-res;
				cost[d+dispRange*idx] = res;
			}
		}
}


void pixelBasedCostL2R_Color(unsigned char *ref_image, unsigned char *tar_image, float *cost, const int dispRange, const int rdim, const int cdim, int normOrNot){
	int idx = 0,idxtar = 0,xtar = 0;
	for (int y=0; y<rdim; y++)
		for(int x=0; x<cdim;x++){
			idx = y*cdim+x;
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
}

void pixelBasedCostL2R_Color(unsigned char *ref_image, unsigned char *tar_image, float *cost, const int dispRange, const int rdim, const int cdim,float trunc, int normOrNot){
	int idx = 0,idxtar = 0,xtar = 0;
	for (int y=0; y<rdim; y++)
		for(int x=0; x<cdim;x++){
			idx = y*cdim+x;
			for(int d=0; d<dispRange; d++){
				xtar = x-d;
				if(xtar<0) xtar+=cdim;
				idxtar = y*cdim + xtar;
				float coeff = (float)1/(float)3;
				if(normOrNot == 0)
					cost[d+dispRange*idx] = min(abs(ref_image[idx*3]-tar_image[idxtar*3]) +
											abs(ref_image[idx*3+1]-tar_image[idxtar*3+1])+
											abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]),trunc);
				else
					cost[d+dispRange*idx] = coeff*(min(abs(ref_image[idx*3]-tar_image[idxtar*3]) +
											abs(ref_image[idx*3+1]-tar_image[idxtar*3+1])+
											abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]),trunc));
			}
		}
}

void pixelBasedR2L_Color(unsigned char *ref_image, unsigned char *tar_image, float *cost, const int dispRange, const int rdim, const int cdim, int normOrNot){
	int idx = 0,idxtar = 0,xtar = 0;
	for (int y=0; y<rdim; y++)
		for(int x=0; x<cdim;x++){
			idx = y*cdim+x;
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
}

void pixelBasedR2L_Color(unsigned char *ref_image, unsigned char *tar_image, float *cost, const int dispRange, const int rdim, const int cdim,float trunc, int normOrNot){
	int idx = 0,idxtar = 0,xtar = 0;
	for (int y=0; y<rdim; y++)
		for(int x=0; x<cdim;x++){
			idx = y*cdim+x;
			for(int d=0; d<dispRange; d++){
				xtar = x+d;
				if(xtar>=cdim) xtar-=cdim;
				idxtar = y*cdim + xtar;
				float coeff = (float)1/(float)3;
				if(normOrNot == 0)
					cost[d+dispRange*idx] = min(abs(ref_image[idx*3]-tar_image[idxtar*3]) +
											abs(ref_image[idx*3+1]-tar_image[idxtar*3+1]) + 
											abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]),trunc);
				else
					cost[d+dispRange*idx] = coeff*(min(abs(ref_image[idx*3]-tar_image[idxtar*3]) +
											abs(ref_image[idx*3+1]-tar_image[idxtar*3+1]) + 
											abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]),trunc));
			}
		}
}
