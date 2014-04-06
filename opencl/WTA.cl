__kernel void WTA(__global float *totCost,__global unsigned char *winners, const int rdim, const int cdim, const int dispRange){
	int c = get_global_id(0);//Thread Index: x dimension
	int r = get_global_id(1);//Thread Index: y dimension
	float minCost = totCost[0+dispRange*(c+cdim*r)];
	unsigned char disp=0;
	for(int d=1; d<dispRange; d++){
		if(totCost[d+dispRange*(c+cdim*r)]<minCost){
					minCost=totCost[d+dispRange*(c+cdim*r)];
					disp = d;
		}
	}
	winners[r*cdim+c]=disp;
}

__kernel void WTA_NCC(__global float *totCost,__global unsigned char *winners, const int rdim, const int cdim, const int dispRange){
	int c = get_global_id(0);//Thread Index: x dimension
	int r = get_global_id(1);//Thread Index: y dimension
	float maxCost = totCost[0+dispRange*(c+cdim*r)];
	unsigned char disp=0;
	for(int d=1; d<dispRange; d++){
		if(totCost[d+dispRange*(c+cdim*r)]>maxCost){
					maxCost=totCost[d+dispRange*(c+cdim*r)];
					disp = d;
		}
	}
	winners[r*cdim+c]=disp;
}

