//////////////////////////////////////////////////////////////////////
//Name: AdaptiveWeightAggregation.cu 
//Created date: 3-2-2012
//Modified date: 3-2-2012
//Author: Gorkem Saygili and Jie Shen
//Discription: aggregating cost with adaptive aggragation - cuda kernel
//////////////////////////////////////////////////////////////////////

#ifndef _COMPUTEWEIGHT_KERNEL_
#define _COMPUTEWEIGHT_KERNEL_
#define square(x) x*x

__global__ void computeWeights(float *weights, 
	float *proxWeight, const unsigned char *LABImage, 
	const float gamma_similarity, const int rdim, 
	const int cdim, const int maskRad){
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y*cdim+x;
	int lref=0, ltar=0, aref=0, atar=0, bref=0, btar=0, ytar=0, xtar=0;
	float diff0=0,diff1=0,diff2=0;
	int maskEdge = 2*maskRad+1;
	int maskArea = square(maskEdge);
	int taridx=0;
	float colorDiff=0;
	lref = (int)LABImage[idx*3];
	aref = (int)LABImage[idx*3+1];
	bref = (int)LABImage[idx*3+2];
	for(int k=0,y_=-maskRad; y_<=maskRad; y_++){
		ytar = y+y_;
		if(ytar>=rdim || ytar<0)
			for(int x_=-maskRad; x_<=maskRad; x_++,k++)
				weights[k+maskArea*idx] = 0;
		else{
			for(int x_=-maskRad; x_<=maskRad; x_++,k++){
				//if(weights[k+maskArea*idx] == 0){
				xtar = x+x_;
				if(xtar<0 || xtar>=cdim)
					weights[k+maskArea*idx] = 0;
				else{
					taridx = ytar*cdim+xtar;
					ltar = (int)LABImage[taridx*3];
					atar = (int)LABImage[taridx*3+1];
					btar = (int)LABImage[taridx*3+2];
					diff0 = (float)(lref - ltar);
					diff1 = (float)(aref - atar);
					diff2 = (float)(bref - btar);
					colorDiff = (float)sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2);
					weights[k+maskArea*idx] = (float)(proxWeight[k]*
						exp(-colorDiff/gamma_similarity));
					//weights[(maskArea-1-k)+maskArea*taridx]=weights[k+maskArea*idx];
				}
			}
		}
	}
}


__global__ void calcAWCostL2R(float *weightRef, 
	float *weightTar, float *cost, float *cost_new_image,
	 const int rdim, const int cdim,
	const int dispRange, const int maskRad){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int maskEdge = 2*maskRad+1;
	int maskArea = maskEdge*maskEdge;
	int idx=y*cdim+x, refidx=0, taridx=0, yref=0, xref=0, ytar=0, \
		xtar=0, maskInd=0, refInd=0, tarInd=0;
	int idx_t = 0;
	float weight,sum_weight,score;
	idx = y*cdim+x;
	for(int d=0; d<dispRange; d++){
		sum_weight = 0;
		maskInd = 0;
		score = 0;
		xtar = x-d;
		if(xtar<0) xtar += cdim;
		idx_t = y*cdim+xtar;
		for(int y_=-maskRad; y_<=maskRad; y_++){
			yref = y+y_;
			if(yref<0) yref+=rdim;
			else if(yref>=rdim) yref-=rdim;
			ytar = yref;
			for(int x_=-maskRad; x_<=maskRad; x_++){
				xref = x+x_;
				if(xref<0) xref += cdim;
				else if(xref>=cdim) xref -= cdim;
				refInd = yref*cdim+xref;
				refidx = maskInd+maskArea*idx;
				taridx = maskInd+maskArea*idx_t;
				weight = weightRef[refidx]*weightTar[taridx];
				sum_weight += weight;
				score += cost[d+dispRange*refInd]*weight;
				//score += cost[d+dispRange*idx];
				maskInd++;
			}
		}
			cost_new_image[d+dispRange*idx] = (float)(score/sum_weight);
	}
}


__global__ void calcAWCostR2L(float *weightRef, 
	float *weightTar, float *cost, 
	const int rdim, const int cdim,\
	const int dispRange, const int maskRad){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int maskEdge = 2*maskRad+1;
	int maskArea = maskEdge*maskEdge;
	int idx=y*cdim+x, refidx=0, taridx=0, yref=0, xref=0, ytar=0,
		xtar=0,maskInd=0,refInd=0,tarInd=0;
	float weight,sum_weight,score;
	for(int d=0; d<dispRange; d++){
		sum_weight = 0;
		maskInd = 0;
		score = 0;
		for(int y_=-maskRad; y_<=maskRad; y_++){
			yref = y+y_;
			if(yref<0) yref+=rdim;
			else if(yref>=rdim) yref-=rdim;
			ytar = yref;
			for(int x_=-maskRad; x_<=maskRad; x_++){
				xref = x+x_;
				xtar = xref+d;
				if(xref<0) xref += cdim;
				else if(xref>=cdim) xref -= cdim;
				if(xtar<0) xtar += cdim;
				else if(xtar>=cdim) xtar-=cdim;
				refInd = yref*cdim+xref;
				tarInd = ytar*cdim+xtar;
				refidx = maskInd+maskArea*refInd;
				taridx = maskInd+maskArea*tarInd;
				weight = weightRef[refidx]*weightTar[taridx];
				//weight = 1;
				sum_weight += weight;
				//sum_weight = 1;
				score += cost[d+dispRange*refInd]*weight;
				//score += 1;
				maskInd++;
			}
		}
		if(sum_weight != 0)
			cost[d+dispRange*idx] = (float)(score/sum_weight);
		else
			cost[d+dispRange*idx] = (float)score;
	}
}


#endif

