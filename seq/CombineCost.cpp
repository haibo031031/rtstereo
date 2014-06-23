//////////////////////////////////////////////////////////////////////
//Name: CombineCost.cpp
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: Combines different cost matrices
//////////////////////////////////////////////////////////////////////

#include "CombineCost.h"

void combineCost(float *cost_image1,float *cost_image2, float lambda1, float lambda2, int rdim, int cdim, int dispRange){
	float cost1=0,cost2=0;
	int idx=0,a=0;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++)
			for(int d=0; d<dispRange; d++){
				idx = d+dispRange*(x+cdim*y);
				cost1 = cost_image1[idx];
				cost2 = cost_image2[idx];
				if(cost1 != cost2)
					a = 1;
				cost1 = 1-exp(-cost1/lambda1);
				cost2 = 1-exp(-cost2/lambda2);
				cost_image1[idx] = cost1+cost2;
			}
}