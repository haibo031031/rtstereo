//////////////////////////////////////////////////////////////////
//Path: 4stereo/nccDisparity_kernel.cu
//Created date: 20-12-2011
//Modified date: 20-12-2011
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: stereo cuda device code - nccDisparity_kernel
//////////////////////////////////////////////////////////////////

#ifndef _NCCDISPARITY_KERNEL_
#define _NCCDISPARITY_KERNEL_

__global__ void k_nccDisparityL2R(const unsigned char* d_imRef, const unsigned char* d_imTar, float* d_imCost, const int dispRange, const int maskHrad, const int maskWrad, const int maskNCCSize, const int rows, const int cols)
{
        //determine matrix element that each thread works on        
        //using blockIdx and threadIdx
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        int r = blockIdx.y * blockDim.y + threadIdx.y;
        if(c < cols && r < rows){
		int idx = r * cols + c;
		int xref = 0, yref = 0, xtar = 0, ytar = 0;
		float nom = 0.0, denom = 0.0;
		float refVal = 0.0, tarVal = 0.0;
		float meanRef = 0.0, meanTar = 0.0;
        	int x, y ,d;
		for(y = - maskHrad; y <= maskHrad; y++){
			yref = r + y;
			if(yref >= rows) yref -= rows;
			else if(yref < 0) yref += rows;
			for(x = - maskWrad; x <= maskWrad; x++){
				xref = c + x;
				if(xref >= cols) xref -= cols;
				else if(xref < 0) xref += cols;
				meanRef += (float)d_imRef[yref * cols + xref];
			}
		}
		meanRef /= (float)maskNCCSize;
		for(d = 0; d < dispRange; d++){
			xref = 0, yref = 0;
			meanTar = 0.0;
			nom = 0.0, denom = 0.0;
			d_imCost[d + dispRange * idx] = 0.0;
			for(y = - maskHrad; y <= maskHrad; y++){
				ytar = r + y;
				if(ytar >= rows) ytar -= rows;
				else if(ytar < 0) ytar += rows;
				for(x = -maskWrad; x <= maskWrad; x++){
					xtar = c - d + x;
					if(xtar >= cols) xtar -= cols;
					else if(xtar < 0) xtar += cols;
					meanTar += (float)d_imTar[ytar * cols + xtar];
				}	
			}
			meanTar /= (float)maskNCCSize;
			for(y = -maskHrad; y <= maskHrad; y++){
				yref = r + y;
				if(yref >= rows) yref -= rows;
				else if(yref < 0) yref += rows;
				ytar = yref;
				for(x = -maskWrad; x <= maskWrad; x++){
					xref = c + x;
					xtar = c - d + x;
					if(xref >= cols) xref -= cols;
					else if(xref < 0) xref += cols;
					if(xtar >= cols) xtar -= cols;
					else if(xtar < 0) xtar += cols;
					refVal = (float)d_imRef[yref * cols + xref] - meanRef;
					tarVal = (float)d_imTar[ytar * cols + xtar] - meanTar;
					nom += refVal * tarVal;
					denom += (refVal * refVal) * (tarVal * tarVal);
				}	
			}
			denom = (float)sqrt(denom);
			if(denom != 0) d_imCost[d + dispRange * idx] = (float)(nom)/(float)denom;
		}
	}
}


#endif

