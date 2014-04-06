__kernel void calcNCCDispL2R(const __global unsigned char * ref_image, const __global unsigned char * tar_image, __global float * cost, \
	const int dispRange, const int rdim, const int cdim, const int maskHrad, const int maskWrad){
	int c = get_global_id(0);//Thread Index: x dimension
	int r = get_global_id(1);//Thread Index: y dimension
	int maskSize = (2*maskHrad+1)*(2*maskWrad+1);
	int idx = r * cdim + c;
	int yref,ytar,xref,xtar;
	float nom=0,denom=0;
	float refVal = 0;
	float tarVal = 0;
	float meanRef=0, meanTar=0; 
	for(int y=-maskHrad; y <= maskHrad; y++){
		yref = r+y;
		if(yref<0) yref+=rdim;
		else if(yref>=rdim) yref-=rdim;
		for(int x = -maskWrad; x <= maskWrad; x++){
			xref = c+x;
			if(xref<0) xref+=cdim;
			else if(xref>=cdim) xref-=cdim;
			meanRef += (float)ref_image[yref*cdim+xref];
		}
	}
	meanRef = meanRef/(float)maskSize;
	for(int d=0; d<dispRange; d++){
		//find the mean of matching box
		nom = 0;
		denom = 0;
		cost[d+dispRange*(c+cdim*r)] = 0;
		meanTar = 0;
		yref = 0;
		xref = 0;
		for(int y = -maskHrad; y <= maskHrad; y++){
			ytar = r+y;
			if(ytar<0) ytar += rdim;
			else if(ytar>=rdim) ytar-=rdim;
			for(int x = -maskWrad; x <= maskWrad; x++){
				xtar = c-d+x;
				if(xtar<0) xtar+=cdim;
				else if(xtar>=cdim) xtar-=cdim;
				meanTar += (float)tar_image[ytar*cdim+xtar];
			}
		}
		meanTar = meanTar/(float)maskSize;
		for(int y = -maskHrad; y <= maskHrad; y++){
			yref = r+y;
			if(yref	>= rdim) yref-=rdim;
			else if(yref < 0) yref+=rdim;
			ytar = yref;
			for(int x = -maskWrad; x <= maskWrad; x++){
				xref = c+x;
				xtar = c+x-d;
				if(xref >= cdim)xref-=cdim;
				else if(xref < 0)xref+=cdim;
				if(xtar >= cdim)xtar-=cdim;
				else if(xtar < 0)xtar+=cdim;
				refVal = (float)ref_image[yref*cdim+xref]-meanRef;
				tarVal = (float)tar_image[ytar*cdim+xtar]-meanTar;
				nom += refVal*tarVal;
				denom += (refVal*refVal)*(tarVal*tarVal);
			}
		}
		denom = (float)sqrt(denom);
		if(denom!=0)
			cost[d+dispRange*(c+cdim*r)] = (float)nom/(float)denom;
	}

}