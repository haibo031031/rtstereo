//////////////////////////////////////////////////////////////////////
//Name: ConvolveImage.cu 
//Created date: 3-2-2012
//Modified date: 3-2-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: convolve image with any mask - cuda kernel
/////////////////////////////////////////////////////////////////////

#ifndef _CONVOLVEIMAGE_KERNEL_
#define _CONVOLVEIMAGE_KERNEL_

__global__ void convolveImage(const unsigned char *img, 
	float *outIm, const float *mask, const int maskHrad, const int maskWrad,
	const int rdim, const int cdim){
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int maskH = 2*maskHrad+1;
	int maskW = 2*maskWrad+1;
	float convRes = 0;
	int ytar, xtar = 0;
	int idx = y*cdim+x;
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
#endif
