//////////////////////////////////////////////////////////////////////
//Name: CensusTransform.cpp
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: Census transform of image and haar distance 
//////////////////////////////////////////////////////////////////////

#include "CensusTransform.h"

void calcCensusTrans(unsigned char *img, bool *censusTrans, const int maskHrad, const int maskWrad, const int rdim, const int cdim){
	int boolSize = (2*maskHrad+1)*(2*maskWrad+1);
	int boolcdim = 2*maskWrad+1; 
	int boolInd = 0;
	int ytar = 0,xtar = 0,idx = 0;
	unsigned char centPixVal = 0,tarPixVal = 0;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			idx = y*cdim+x;
			centPixVal = img[idx];
			for(int y_=-maskHrad; y_<=maskHrad; y_++){
				ytar = y+y_;
				if(ytar < 0) ytar += rdim;
				else if(ytar >= rdim) ytar -= rdim;
				for(int x_=-maskWrad; x_<=maskWrad; x_++){
					xtar = x+x_;
					if(xtar < 0) xtar += cdim;
					else if(xtar >= cdim) xtar -= cdim;
					boolInd = (y_+maskHrad)*boolcdim+(x_+maskWrad);
					tarPixVal = img[ytar*cdim+xtar];
					if(centPixVal>tarPixVal)
						censusTrans[boolInd+boolSize*idx] = 1;
					else censusTrans[boolInd+boolSize*idx] = 0;
				}
			}
		}
}

void calcHammingDist(bool *censusTrnsRef, bool *censusTrnsTar, float *cost, const int maskHrad, const int maskWrad, int rdim, int cdim, int dispRange){
	int boolSize = (2*maskHrad+1)*(2*maskWrad+1);
	int boolInd = 0,cnt = 0,xtar=0,idx=0;
	bool refVal=1,tarVal=1;
	int boolcdim = 2*maskWrad+1;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			idx = y*cdim+x;
			for(int d=0; d<dispRange; d++){
				cnt = 0;
				xtar = x-d;
				if(xtar<0) xtar += cdim;
				for(int y_=-maskHrad; y_<= maskHrad; y_++)
					for(int x_=-maskWrad; x_<=maskWrad; x_++){
						boolInd = (y_+maskHrad)*boolcdim+(x_+maskWrad);
						refVal = censusTrnsRef[boolInd+boolSize*idx];
						tarVal = censusTrnsTar[boolInd+boolSize*(xtar+cdim*y)];
						if(refVal!=tarVal)
							cnt++;
					}
				cost[d+dispRange*idx] = (float)cnt;
			}
		}
}