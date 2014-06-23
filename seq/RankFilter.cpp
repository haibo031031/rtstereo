//////////////////////////////////////////////////////////////////////
//Name: RankFilter.cpp
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: Rank filtering of an image
//////////////////////////////////////////////////////////////////////

#include "RankFilter.h"

void calculateRankFilt(unsigned char *img, unsigned char *outImg, const int maskSize, const int rdim, const int cdim){
	int idx=0,tarIdx=0,cnt=0,xInd=0,yInd=0;
	unsigned char centPixVal,tarPixVal;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			cnt=1;
			idx = y*cdim+x;
			centPixVal = img[idx];
			for(int y_=-maskSize; y_<=maskSize; y_++){
				yInd = y+y_;
				if(yInd>=rdim) yInd -= rdim;
				else if(yInd<0) yInd += rdim;
				for(int x_=-maskSize; x_<=maskSize; x_++){
					xInd = x+x_;
					if(xInd>=cdim) xInd -= cdim;
					else if(xInd<0) xInd += cdim;
					tarIdx = yInd*cdim+xInd;
					tarPixVal = img[tarIdx];
					if(tarPixVal<centPixVal)
						cnt++;	
				}
			}
			outImg[idx] = (unsigned char)cnt;
		}
}