//////////////////////////////////////////////////////////////////////
//Name: NCCDisparity.cpp
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: Normalized Cross Correlation cost calculation
//////////////////////////////////////////////////////////////////////

#include "NCCDisparity.h"

void calcNCCDispL2R(unsigned char *ref_image, unsigned char *tar_image, float *cost, const int dispRange, const int rdim, const int cdim, const int maskHrad, const int maskWrad){
	int maskSize = (2*maskHrad+1)*(2*maskWrad+1);
	float nom=0,denom=0;
	int yref,ytar,xref,xtar;
	float refVal = 0;
	float tarVal = 0;
	float meanRef=0, meanTar=0;
	int idx = 0;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			meanRef=0;
			idx = y*cdim+x;
			for(int y_=-maskHrad; y_<=maskHrad; y_++){
				yref = y+y_;
				if(yref<0) yref += rdim;
				else if(yref>=rdim) yref -= rdim;
				for(int x_=-maskWrad; x_<=maskWrad; x_++){
					xref = x+x_;
					if(xref<0) xref+=cdim;
					else if(xref>=cdim) xref-=cdim;
					meanRef += (float) ref_image[yref*cdim+xref];
				}
			}
			meanRef = meanRef/(float)maskSize;
			for(int d=0; d<dispRange; d++){
				nom = 0;
				denom = 0;
				meanTar = 0;
				for(int y_=-maskHrad; y_<=maskHrad; y_++){
					ytar = y+y_;
					if(ytar<0) ytar += rdim;
					else if(ytar>=rdim) ytar -= rdim;
					for(int x_=-maskWrad; x_<=maskWrad; x_++){
						xtar = x+x_-d;
						if(xtar<0) xtar += cdim;
						else if(xtar>=cdim) xtar -= cdim;
						meanTar += (float)tar_image[ytar*cdim+xtar];
					}
				}
				meanTar = meanTar/(float)maskSize;
				for(int y_=-maskHrad; y_<=maskHrad; y_++){
					yref = y+y_;
					if(yref<0) yref += rdim;
					else if(yref>=rdim) yref -= rdim;
					ytar = yref;
					for(int x_=-maskWrad; x_<=maskWrad; x_++){
						xref = x+x_;
						xtar = x+x_-d;
						if(xref<0) xref += cdim;
						else if(xref>=cdim) xref -= cdim;
						if(xtar<0) xtar += cdim;
						else if(xtar>=cdim) xtar -= cdim;
						refVal = (float)(ref_image[yref*cdim+xref])-meanRef;
						tarVal = (float)(tar_image[ytar*cdim+xtar])-meanTar;
						nom += refVal*tarVal;
						denom += refVal*refVal*tarVal*tarVal;
					}
				}
				denom = (float)sqrt(denom);
				if(denom != 0)
					cost[d+dispRange*idx] = (float)(nom)/(float)(denom);
			}
		}
}