//////////////////////////////////////////////////////////////////////
//Name: CrossBasedAggregation.cl 
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: cross based aggregation- opencl kernel
/////////////////////////////////////////////////////////////////////

#define min(a,b) (a>b) ? b:a
#define max(a,b) (b>a) ? b:a 
#define L 17
#define TAO 20

void calc_x_range(const __global unsigned char * image, int y, int x, int rdim, int cdim, int * leftOffset, int * rightOffset){
	int idx_ref = y*cdim + x;
	int x_ = 0;
	unsigned char val_ref_g = image[idx_ref*3];
	unsigned char val_ref_b = image[idx_ref*3+1];
	unsigned char val_ref_r = image[idx_ref*3+2];
	unsigned char val_tar_g,val_tar_b,val_tar_r;
	float delta = 0, delta_;
	int idx_tar,tarx=0;
	while((tarx>=0) && (delta<TAO) && (x_<L)){
		x_++;
		tarx = x-x_;
		if(tarx>=0){
			idx_tar = y*cdim + tarx;
			val_tar_g = image[idx_tar*3];
			val_tar_b = image[idx_tar*3+1];
			val_tar_r = image[idx_tar*3+2];
			delta_ = (float)max(abs(val_ref_b - val_tar_b),abs(val_ref_r - val_tar_r));
			delta = (float) max(abs(val_ref_g - val_tar_g),delta_);
		}
		else x_--;
	}
	if(x_==0 && x != 0)
		x_++;
	*leftOffset = x_;
	x_ = 0;
	delta = 0;
	while((tarx<cdim) && (delta<TAO) && (x_<L)){
		x_++;
		tarx = x+x_;
		if(tarx<cdim){
			idx_tar = y*cdim + tarx;
			val_tar_g = image[idx_tar*3];
			val_tar_b = image[idx_tar*3+1];
			val_tar_r = image[idx_tar*3+2];
			delta_ = (float)max(abs(val_ref_b - val_tar_b),abs(val_ref_r - val_tar_r));
			delta = (float) max(abs(val_ref_g - val_tar_g),delta_);
		}
		else x_--;
	}
	if(x_ == 0 && x < cdim-1)
		x_++;
	*rightOffset = x_;
}

void calc_y_range(const __global unsigned char * image, int y, int x, int rdim, int cdim, int * topOffset, int * bottomOffset){
	int idx_ref = y*cdim + x;
	int y_ = 0;
	unsigned char val_ref_g = image[idx_ref*3];
	unsigned char val_ref_b = image[idx_ref*3+1];
	unsigned char val_ref_r = image[idx_ref*3+2];
	unsigned char val_tar_g,val_tar_b,val_tar_r;
	float delta = 0,delta_;
	int idx_tar,tary=0;
	while((tary>=0) && (delta<TAO) && (y_<L)){
		y_++;
		tary = y-y_;
		if(tary>=0){
			idx_tar = tary*cdim + x;
			val_tar_g = image[idx_tar*3];
			val_tar_b = image[idx_tar*3+1];
			val_tar_r = image[idx_tar*3+2];
			delta_ = (float)max(abs(val_ref_b - val_tar_b),abs(val_ref_r - val_tar_r));
			delta = (float) max(abs(val_ref_g - val_tar_g),delta_);
		}
		else y_--;
	}
	if(y_ == 0 && y != 0)
		y_++;
	*topOffset = y_;
	y_=0;
	delta = 0;
	while((tary<rdim) && (delta<TAO) && (y_<L)){
		y_++;
		tary = y+y_;
		if(tary<rdim){
			idx_tar = tary*cdim+x;
			val_tar_g = image[idx_tar*3];
			val_tar_b = image[idx_tar*3+1];
			val_tar_r = image[idx_tar*3+2];
			delta_ = (float)max(abs(val_ref_b - val_tar_b),abs(val_ref_r - val_tar_r));
			delta = (float) max(abs(val_ref_g - val_tar_g),delta_);
		}
		else y_--;
	}
	if(y_ == 0 && y < rdim-1)
		y_++;
	*bottomOffset = y_;
}

__kernel void findCross(const __global unsigned char *image, __global int *HorzOffset, __global int *VertOffset, int rdim, int cdim){
	int topOffset,bottomOffset,leftOffset,rightOffset,idx;
	float costVal = 0,min_cost;
	int x = get_global_id(0);//Thread Index: x dimension
	int y = get_global_id(1);//Thread Index: y dimension
	if(x<cdim && y<rdim){
		idx = y*cdim+x;
		calc_y_range(image, y, x, rdim, cdim, &topOffset, &bottomOffset);
		VertOffset[2*idx] = topOffset;
		VertOffset[1+2*idx] = bottomOffset;
		calc_x_range(image, y, x, rdim, cdim, &leftOffset, &rightOffset);
		HorzOffset[2*idx] = leftOffset;
		HorzOffset[1+2*idx] = rightOffset;
	}
}

__kernel void aggregate_cost_horizontal(__global float *cost_image, int rdim, int cdim, int dispRange){
	int idx,idx_prev;
	int y = get_global_id(1);
	if(y<rdim){
		for(int x=1; x<cdim; x++)
			for(int d=0; d<dispRange; d++){
				//idx = y*cdim+x;
				cost_image[d+dispRange*(y*cdim+x)] += cost_image[d+dispRange*(y*cdim+x-1)]; 
			}	
	}
}


__kernel void cross_stereo_aggregation(const __global float * cost_image, __global float *cost_new_image,\
	__global int *HorzOffset_ref, __global int *HorzOffset_tar, __global int *VertOffset_ref, __global int *VertOffset_tar,  \
	int rdim, int cdim, int dispRange){
	int idx_ref,idx_tar,tarx,tOffset,bOffset,lOffset,rOffset,idx,idx_t,idx_left,idx_right;
	int cury, curx,curx_l,curx_r,cnt;
	float costSum;
	int x = get_global_id(0);//Thread Index: x dimension
	int y = get_global_id(1);//Thread Index: y dimension
	if(x<cdim && y<rdim){
		idx = x + y*cdim;
		for(int d=0; d<dispRange; d++){
			tarx = x-d;
			if(tarx < 0) tarx += cdim;
			idx_t = tarx + y*cdim;
			tOffset = min(VertOffset_ref[2*idx],VertOffset_tar[2*idx_t]);
			bOffset = min(VertOffset_ref[1+2*idx],VertOffset_tar[1+2*idx_t]);
			costSum = 0;
			cnt = 0;
			for(int y_=-tOffset; y_<=bOffset; y_++){
				cury = y+y_;
				if(cury >= 0 && cury < rdim){
					idx_ref = x + cury*cdim;
					idx_tar = tarx + cury*cdim;
					lOffset = min(HorzOffset_ref[2*idx_ref],HorzOffset_tar[2*idx_tar]);
					rOffset = min(HorzOffset_ref[1+2*idx_ref],HorzOffset_tar[1+2*idx_tar]);
					cnt += rOffset+lOffset+1;
					curx_l = x-lOffset-1;
					curx_r = x+rOffset;
					if(curx_l >= 0 && curx_r < cdim){
						idx_left = cury*cdim+curx_l;
						idx_right = cury*cdim+curx_r;
						costSum += (cost_image[d+dispRange*idx_right]-cost_image[d+dispRange*idx_left]);
					}
				}
			}
			costSum /= cnt;
			cost_new_image[d+dispRange*idx] = costSum;
		}
	}
}
