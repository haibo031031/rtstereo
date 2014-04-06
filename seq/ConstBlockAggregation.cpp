//////////////////////////////////////////////////////////////////////
//Name: ConstBlockAggregation.cpp
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: Aggregation of cost matrix over constant block
//////////////////////////////////////////////////////////////////////

#include "ConstBlockAggregation.h"

void boxAggregate(float *cost,const int rdim, const int cdim, const int dispRange, const int wndwRad){
	int wndwSize = (2*wndwRad+1)*(2*wndwRad+1);
	int ytar=0, xtar=0;
	float costAggr=0, refCost=0,tarCost=0;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			for(int d=0; d<dispRange; d++){
				costAggr=0;
				for(int y_=-wndwRad; y_<=wndwRad; y_++){
					ytar = y+y_;
					if(ytar>=rdim) ytar -= rdim;
					else if(ytar<0) ytar += rdim;
					for(int x_=-wndwRad; x_<=wndwRad; x_++){
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

}