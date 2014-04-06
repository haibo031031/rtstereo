//////////////////////////////////////////////////////////////////////
//Name: CensusTransform.cl 
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: census transform of an image and calculating hamming distance - opencl kernel
/////////////////////////////////////////////////////////////////////


//Calculate census transform
__kernel void calcCensusTrans(const __global unsigned char *img, 
	__global bool *censusTrans, 
	const int maskHrad, const int maskWrad, const int rdim, 
	const int cdim){
	
	int x = get_global_id(0);
	int y = get_global_id(1);
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
__kernel void calcHammingDist(const __global bool *censusTrnsRef, 
	const __global bool *censusTrnsTar, 
	__global float *cost, const int maskHrad, 
	const int maskWrad, const int rdim, const int cdim, 
	const int dispRange){
	
	int x = get_global_id(0);
	int y = get_global_id(1);
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
