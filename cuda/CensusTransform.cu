//////////////////////////////////////////////////////////////////////
//Name: CensusTransform.cu 
//Created date: 3-2-2012
//Modified date: 3-2-2012
//Author: Gorkem Saygili and Jie Shen
//Discription: census transform of an image and calculating hamming distance - cuda kernel
/////////////////////////////////////////////////////////////////////


#ifndef _CENSUSTRANSFORM_KERNEL_
#define _CENSUSTRANSFORM_KERNEL_

//Calculate census transform
__global__ void calcCensusTrans(const unsigned char *img, 
	bool *censusTrans, const int maskHrad,
	const int maskWrad, const int rdim, 
	const int cdim){
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int boolSize = (2*maskHrad+1)*(2*maskWrad+1);
	int boolcdim = 2*maskWrad+1; 
	int boolInd = 0;
	int ytar = 0,xtar = 0,idx = y*cdim+x;
	unsigned char centPixVal = img[idx], tarPixVal = 0;
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


//calculate hamming distance 
__global__ void calcHammingDist(const bool *censusTrnsRef, 
	const bool *censusTrnsTar, 
	float *cost, const int maskHrad, 
	const int maskWrad, const int rdim, const int cdim, 
	const int dispRange){
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int boolSize = (2*maskHrad+1)*(2*maskWrad+1);
	int boolInd = 0,cnt = 0,xtar=0,idx=y*cdim+x;
	bool refVal=1,tarVal=1;
	int boolcdim = 2*maskWrad+1;
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

#endif
