//////////////////////////////////////////////////////////////////////
//Name: RankFilter.cu
//Created date: 3-2-2012
//Modified date: 3-2-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: rank filter of an image - cuda kernel
/////////////////////////////////////////////////////////////////////

#ifndef _RANKFILTER_KERNEL_
#define _RANKFILTER_KERNEL_

__global__ void calculateRankFilt(const unsigned char *img, 
	unsigned char *outImg, const int maskSize,
	const int rdim, const int cdim){
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y*cdim + x;
	int tarIdx=0, cnt = 1, xInd=0, yInd=0;
	unsigned char centPixVal,tarPixVal;
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

#endif
