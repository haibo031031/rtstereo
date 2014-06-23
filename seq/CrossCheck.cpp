//////////////////////////////////////////////////////////////////////
//Name: CrossBasedAggregation.cpp
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: Cross-Check final disparity results
//////////////////////////////////////////////////////////////////////

#include "CrossCheck.h"
void cross_check(unsigned char *ref_disparity, unsigned char *tar_disparity, unsigned char *cross_check_image, const int rdim, const int cdim){
	int idx = 0,refVal=0,xtar=0,tarVal=0;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			idx = y*cdim+x;
			refVal = ref_disparity[idx];
			xtar = x - refVal;
			if(xtar<0) xtar += cdim;
			tarVal = tar_disparity[y*cdim+xtar];
			if(refVal == tarVal)
				cross_check_image[idx] = 1;
			else
				cross_check_image[idx] = 0;
		}
}