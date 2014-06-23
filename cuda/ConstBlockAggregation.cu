//////////////////////////////////////////////////////////////////
//Name: ConstantBlockAggregation.cu
//Created date: 3-2-2012
//Modified date: 3-2-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: aggregating cost on a constant block - cuda kernel
//////////////////////////////////////////////////////////////////

#ifndef _BOXAGGREGATE_KERNEL_
#define _BOXAGGREGATE_KERNEL_

__global__ void boxAggregate(float *cost,
		const int dispRange, const int wndwRad,
		const int rdim, const int cdim){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int wndwSize = (2*wndwRad+1)*(2*wndwRad+1);
	int ytar=0, xtar=0;
	float costAggr=0;
	for(int d=0; d<dispRange; d++){
		costAggr=0;
		for(int y_=-wndwRad; y_ <= wndwRad; y_++){
			ytar = y+y_;
			if(ytar>=rdim) ytar -= rdim;
			else if(ytar<0) ytar += rdim;
			for(int x_=-wndwRad; x_ <= wndwRad; x_++){
				xtar = x+x_;
				if(xtar>=cdim) xtar -= cdim;
				else if(xtar<0) xtar += cdim;
				costAggr += cost[d+dispRange*(xtar+cdim*ytar)]; 
			}
		}
		costAggr = (float)(costAggr)/(float)(wndwSize);
		cost[d+dispRange*(x+cdim*y)] = costAggr;
	}
}

#endif
