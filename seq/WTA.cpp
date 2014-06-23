//////////////////////////////////////////////////////////////////////
//Name: WTA.cpp
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: winner-takes-all
//////////////////////////////////////////////////////////////////////


void WTA(float *cost, unsigned char *winners, const int rdim, const int cdim, const int dispRange){
	float minCost=0;
	unsigned char disp=0;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			minCost = cost[0+dispRange*(x+cdim*y)];
			for(int d=1; d<dispRange; d++){
				if(cost[d+dispRange*(x+cdim*y)]<minCost){
					minCost = cost[d+dispRange*(x+cdim*y)];
					disp = d;
				}
			}
			winners[y*cdim+x]=disp;
		}
}

void WTA_NCC(float *cost, unsigned char *winners, const int rdim, const int cdim, const int dispRange){
	float maxCost=0;
	unsigned char disp=0;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			maxCost = cost[0+dispRange*(x+cdim*y)];
			for(int d=1; d<dispRange; d++){
				if(cost[d+dispRange*(x+cdim*y)]>maxCost){
					maxCost = cost[d+dispRange*(x+cdim*y)];
					disp = d;
				}
			}
			winners[y*cdim+x]=disp;
		}
}