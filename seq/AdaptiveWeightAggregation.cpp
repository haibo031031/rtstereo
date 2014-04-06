//////////////////////////////////////////////////////////////////////
//Name: AdaptiveWeightAggregation.cpp
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: aggregating cost with adaptive aggragation 
//////////////////////////////////////////////////////////////////////

#include "AdaptiveWeightAggreagtion.h"

void calcProxWeight(float *proxWeight,float gamma_proximity,int maskRad){
	int k=0;
	for(int y=-maskRad; y<=maskRad; y++)
		for(int x=-maskRad; x<=maskRad; x++){
			proxWeight[k] = exp(-sqrt((float)(square(y)+square(x)))/gamma_proximity);
			k++;
		}
}
float calcEuclideanDist3P(int l0,int l1, int a0,int a1,int b0,int b1){
	float diff0 = (float)l0-(float)l1;
	float diff1 = (float)a0-(float)a1;
	float diff2 = (float)b0-(float)b1;
	return sqrt(square(diff0) + square(diff1) + square(diff2));	
}

float calcEuclideanDist3P_float(float l0,float l1, float a0,float a1,float b0,float b1){
	float diff0 = l0-l1;
	float diff1 = a0-a1;
	float diff2 = b0-b1;
	return sqrt(square(diff0) + square(diff1) + square(diff2));
}

void computeWeights(float *weights, float *proxWeight, unsigned char *LABImage, const float gamma_similarity, const int rdim, const int cdim,const int maskRad){
	int idx=0;
	int lref=0,ltar=0,aref=0,atar=0,bref=0,btar=0,ytar=0,xtar=0;
	int maskEdge = 2*maskRad+1;
	int maskArea = square(maskEdge);
	int taridx=0;
	float colorDiff=0;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			idx = y*cdim+x;
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
							colorDiff = calcEuclideanDist3P(lref,ltar,aref,atar,bref,btar);
							weights[k+maskArea*idx] = (float)(proxWeight[k]*exp(-colorDiff/gamma_similarity));
								//weights[(maskArea-1-k)+maskArea*taridx]=weights[k+maskArea*idx];
						}
					}
				}
			}

		}
}

void calcAWCostL2R(float *weightRef, float *weightTar, float *cost, float * newCost, const int rdim, const int cdim,\
	const int dispRange, const int maskRad){
	int maskEdge = 2*maskRad+1;
	int maskArea = maskEdge*maskEdge;
	int idx=0,refidx=0,taridx=0,yref=0,xref=0,ytar=0,xtar=0,maskInd=0,refInd=0,tarInd=0;
	int idx_t=0;
	float weight,sum_weight,score;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
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
						maskInd++;
					}
				}
				newCost[d+dispRange*idx] = (float)(score/sum_weight);
			}
		}
}

void calcAWCostR2L(float *weightRef, float *weightTar, float *cost, const int rdim, const int cdim,\
	const int dispRange, const int maskRad){
	int maskEdge = 2*maskRad+1;
	int maskArea = maskEdge*maskEdge;
	int idx=0,refidx=0,taridx=0,yref=0,xref=0,ytar=0,xtar=0,maskInd=0,refInd=0,tarInd=0;
	float weight,sum_weight,score;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			idx = y*cdim+x;
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
						sum_weight += weight;
						score += cost[d+dispRange*refInd]*weight;
						maskInd++;
					}
				}
				cost[d+dispRange*idx] = (float)(score/sum_weight);
			}
		}
}