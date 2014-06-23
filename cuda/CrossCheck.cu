//////////////////////////////////////////////////////////////////////
//Name: ConvolveImage.cu
//Created date: 3-2-2012
//Modified date: 3-2-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: cross check - cuda kernel
/////////////////////////////////////////////////////////////////////

#ifndef _CROSSCHECK_KERNEL_
#define _CROSSCHECK_KERNEL_


__global__ void cross_check(unsigned char* d_dispRef, unsigned char* d_dispTar, unsigned char* d_imCrossCheck, const int rows, const int cols)
{
	unsigned char val = 0;
        //determine matrix element that each thread works on        
        //using blockIdx and threadIdx
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        int r = blockIdx.y * blockDim.y + threadIdx.y;
        if(c < cols && r < rows){
		int idx_ref = r * cols + c;
		int val_disp_ref = d_dispRef[idx_ref];
		
		int x_tar = c - val_disp_ref;   // ?
		if(x_tar < 0) x_tar += cols;			
		int idx_tar = r * cols + x_tar;
		int val_disp_tar = d_dispTar[idx_tar];
			
		val=(val_disp_ref == val_disp_tar)?1:0;
		d_imCrossCheck[idx_ref] = val;
	}
}

#endif


