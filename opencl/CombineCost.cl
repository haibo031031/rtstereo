//////////////////////////////////////////////////////////////////////
//Name: CombineCost.cl 
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: combine initial cost with state-of-the-art opencl kernel
///////////////////////////////////////////////////////////////////////



__kernel void combineCost(__global float *cost_image1,
	__global float *cost_image2, const float lambda1, 
	const float lambda2, const int rdim, const int cdim, 
	int dispRange){

	int x = get_global_id(0);
	int y = get_global_id(1);
	float cost1=0,cost2=0;
	int idx=0,a=0;
	for(int d=0; d<dispRange; d++){
		idx = d+dispRange*(x+cdim*y);
		cost1 = cost_image1[idx];
		cost2 = cost_image2[idx];
		cost1 = 1-exp(-cost1/lambda1);
		cost2 = 1-exp(-cost2/lambda2);
		cost_image1[idx] = cost1+cost2;
	}
}