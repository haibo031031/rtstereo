//////////////////////////////////////////////////////////////////////
//Name: WTA.cu
//Created date: 3-2-2012
//Modified date: 3-2-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: winner-takes-all(WTA) - cuda kernel
/////////////////////////////////////////////////////////////////////

#ifndef _WTA_KERNEL_
#define _WTA_KERNEL_

__global__ void WTA(float* d_imCost , unsigned char* winners, const int dispRange, const int rows, const int cols)
{
        //determine matrix element that each thread works on        
        //using blockIdx and threadIdx
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        int r = blockIdx.y * blockDim.y + threadIdx.y;
        if(c < cols && r < rows){
		int idx = r * cols + c;
		float min_aggr_cost = d_imCost[0 + dispRange * idx];
		unsigned char min_disp = 0;
		int d;
		for(d = 1; d < dispRange; d++){
			float aggr_cost = d_imCost[d + dispRange * idx];
			if(aggr_cost < min_aggr_cost){
				min_aggr_cost = aggr_cost;
				min_disp = d;
			}		
		}
		winners[idx] = min_disp;
	}
}

__global__ void WTA_NCC(float* d_imCost, unsigned char* winners, const int dispRange, const int rows, const int cols)
{
        //determine matrix element that each thread works on        
        //using blockIdx and threadIdx
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        int r = blockIdx.y * blockDim.y + threadIdx.y;
        if(c < cols && r < rows){
		int idx = r * cols + c;
		float max_aggr_cost = d_imCost[0 + dispRange * idx];
		unsigned char max_disp = 0;
		int d;
		for(d = 1; d < dispRange; d++){
			float aggr_cost = d_imCost[d + dispRange * idx];
			if(aggr_cost > max_aggr_cost){
				max_aggr_cost = aggr_cost;
				max_disp = d;
			}
		}
		winners[idx] = max_disp;
	}
}



#endif
