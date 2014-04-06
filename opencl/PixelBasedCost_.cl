#define min(a,b) (a>b) ? b:a
#define max(a,b) (b>a) ? b:a
#define max3(a,b,c) (a>max(b,c)) ? a:max(b,c)
__kernel void pixelBasedCostL2R(const __global unsigned char * ref_image, const __global unsigned char * tar_image, __global float * cost, 
	const int dispRange, const int rdim, const int cdim){
	int c = get_global_id(0);//Thread Index: x dimension
	int r = get_global_id(1);//Thread Index: y dimension
if(c<cdim && r<rdim){
	int idx = r * cdim + c;
	for(int d=0; d<dispRange; d++) {
		int tarc = c-d;
		if(tarc<0) tarc += cdim;
		cost[d+dispRange*idx] = abs(ref_image[idx]-tar_image[r * cdim + tarc]);
	}
}
}

__kernel void pixelBasedCostR2L(const __global unsigned char * ref_image, const __global unsigned char * tar_image, __global float * cost, 
	const int dispRange, const int rdim, const int cdim){
	int c = get_global_id(0);//Thread Index: x dimension
	int r = get_global_id(1);//Thread Index: y dimension
if(c<cdim && r<rdim){
	int idx = r * cdim + c;
	for(int d=0; d<dispRange; d++) {
		int tarc = c+d;
		if(tarc>=cdim) tarc -= cdim;
		cost[d+dispRange*idx] = abs(ref_image[idx]-tar_image[r * cdim + tarc]);
	}
}
}

__kernel void pixelBasedCostL2R_Float(const __global float * ref_image, const __global float * tar_image, __global float * cost, 
	const int dispRange, const int rdim, const int cdim){
	int c = get_global_id(0);//Thread Index: x dimension
	int r = get_global_id(1);//Thread Index: y dimension
if(c<cdim && r<rdim){
	int idx = r * cdim + c;
	float res=0;
	for(int d=0; d<dispRange; d++) {
		int tarc = c-d;
		if(tarc<0) tarc += cdim;
		res = ref_image[idx]-tar_image[r * cdim + tarc];
		if(res<0) res=-res;
		cost[d+dispRange*idx] = res;
	}
}
}

__kernel void pixelBasedCostR2L_Float(const __global float *ref_image, const __global float *tar_image, __global float *cost, 
	const int dispRange, const int rdim, const int cdim){
	int c = get_global_id(0);//Thread Index: x dimension
	int r = get_global_id(1);//Thread Index: y dimension
if(c<cdim && r<rdim){
	float res=0;
	int idx = r * cdim + c;
	for(int d=0; d<dispRange; d++) {
		int tarc = c+d;
		if(tarc>=cdim) tarc -= cdim;
		res = ref_image[idx]-tar_image[r * cdim + tarc];
		if(res<0) res=-res;
		cost[d+dispRange*idx] = res;
	}
}
}
__kernel void pixelBasedCostL2R_Color(const __global unsigned char *ref_image, const __global unsigned char *tar_image, __global float *cost,\
	 const int dispRange, const int rdim, const int cdim){
	int idx = 0,idxtar = 0,xtar = 0;
	int x = get_global_id(0);//Thread Index: x dimension
	int y = get_global_id(1);//Thread Index: y dimension
	if(x<cdim && y<rdim){
		idx = y*cdim+x;
		for(int d=0; d<dispRange; d++){
			xtar = x-d;
			if(xtar<0) xtar+=cdim;
			idxtar = y*cdim + xtar;
			float coeff = (float)1/3;
			cost[d+dispRange*idx] = coeff * (abs(ref_image[idx*3]-tar_image[idxtar*3]) +
								abs(ref_image[idx*3+1]-tar_image[idxtar*3+1])+
								abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]));
		}
	}	
}

__kernel void pixelBasedCostL2R_Color_tran(const __global unsigned char *ref_image, const __global unsigned char *tar_image, __global float *cost,\
	 const int dispRange, const int rdim, const int cdim,float trunc){
	int idx = 0,idxtar = 0,xtar = 0;
	int x = get_global_id(0);//Thread Index: x dimension
	int y = get_global_id(1);//Thread Index: y dimension
	if(x<cdim && y<rdim){
		idx = y*cdim+x;
		for(int d=0; d<dispRange; d++){
			xtar = x-d;
			if(xtar<0) xtar+=cdim;
			idxtar = y*cdim + xtar;
			//float coeff = (float)1/3;
			cost[d+dispRange*idx] = min(abs(ref_image[idx*3]-tar_image[idxtar*3]) +
									abs(ref_image[idx*3+1]-tar_image[idxtar*3+1])+
									abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]),trunc);
		}
	}
}

__kernel void pixelBasedR2L_Color(const __global unsigned char *ref_image, const __global unsigned char *tar_image, __global float *cost,\
	 const int dispRange, const int rdim, const int cdim){
	int idx = 0,idxtar = 0,xtar = 0;
	int x = get_global_id(0);//Thread Index: x dimension
	int y = get_global_id(1);//Thread Index: y dimension
	if(x<cdim && y<rdim){
		idx = y*cdim+x;
		for(int d=0; d<dispRange; d++){
			xtar = x+d;
			if(xtar>=cdim) xtar-=cdim;
			idxtar = y*cdim + xtar;
			float coeff = (float)1/3;
			cost[d+dispRange*idx] = coeff * (abs(ref_image[idx*3]-tar_image[idxtar*3]) +
									abs(ref_image[idx*3+1]-tar_image[idxtar*3+1]) + 
									abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]));
		}
	}
}

__kernel void pixelBasedR2L_Color_tran(const __global unsigned char *ref_image, const __global unsigned char *tar_image, __global float *cost,\
	 const int dispRange, const int rdim, const int cdim,float trunc){
	int idx = 0,idxtar = 0,xtar = 0;
	int x = get_global_id(0);//Thread Index: x dimension
	int y = get_global_id(1);//Thread Index: y dimension
	if(x<cdim && y<rdim){
		idx = y*cdim+x;
		for(int d=0; d<dispRange; d++){
			xtar = x+d;
			if(xtar>=cdim) xtar-=cdim;
			idxtar = y*cdim + xtar;
			float coeff = (float)1/3;
			cost[d+dispRange*idx] = min(abs(ref_image[idx*3]-tar_image[idxtar*3]) +
									abs(ref_image[idx*3+1]-tar_image[idxtar*3+1]) + 
									abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]),trunc);
		}
	}
}
