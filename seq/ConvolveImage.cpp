//////////////////////////////////////////////////////////////////////
//Name: ConvolveImage.cpp
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: Convolves image with different masks
//////////////////////////////////////////////////////////////////////

#include "ConvolveImage.h"

void convolveImage(unsigned char *img, float *outIm, float *mask, const int rdim, const int cdim, const int maskHrad, const int maskWrad){
	int maskH = 2*maskHrad+1;
	int maskW = 2*maskWrad+1;
	float convRes=0;
	int ytar,xtar=0;
	int idx=0;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			idx = y*cdim+x;
			convRes = 0;
			for(int y_=-maskHrad; y_<=maskHrad; y_++){
				ytar = y+y_;
				if(ytar < 0) ytar+=rdim;
				else if(ytar >= rdim) ytar -= rdim;
				for(int x_=-maskWrad; x_<=maskWrad; x_++){
					xtar = x+x_;
					if(xtar<0) xtar += cdim;
					else if(xtar>=cdim) xtar -= cdim;
					convRes += (float)((float)(img[ytar*cdim+xtar])*(float)(mask[((y_+maskHrad)*maskW)+(x_+maskWrad)]));
				}
			}
			convRes = (convRes>0)? convRes:-convRes;
			outIm[idx] = (convRes>255)? 255:convRes;
		}
}