//////////////////////////////////////////////////////////////////////
//Name: CrossBasedAggregation.cpp
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: State-of-the-art Cross Based Aggregation
//////////////////////////////////////////////////////////////////////

#include "CrossBasedAggregation.h"


void calc_x_range(unsigned char * image, int y, int x, int rdim, int cdim, int * leftOffset, int * rightOffset){
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

void calc_y_range(unsigned char * image, int y, int x, int rdim, int cdim, int * topOffset, int * bottomOffset){
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

void findCross(unsigned char *image, int *HorzOffset, int *VertOffset, int rdim, int cdim){
	int topOffset,bottomOffset,leftOffset,rightOffset,idx;
	float costVal = 0,min_cost;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			idx = y*cdim+x;
			calc_y_range(image, y, x, rdim, cdim, &topOffset, &bottomOffset);
			VertOffset[2*idx] = topOffset;
			VertOffset[1+2*idx] = bottomOffset;
			calc_x_range(image, y, x, rdim, cdim, &leftOffset, &rightOffset);
			HorzOffset[2*idx] = leftOffset;
			HorzOffset[1+2*idx] = rightOffset;
		}
}

void aggregate_cost_horizontal(float *cost_image, int rdim, int cdim, int dispRange){
	int idx,idx_prev;
	for(int y=0; y<rdim; y++)
		for(int x=1; x<cdim; x++)
			for(int d=0; d<dispRange; d++){
				idx = y*cdim+x;
				cost_image[d+dispRange*idx] += cost_image[d+dispRange*(idx-1)]; 
			}	
}

void cross_stereo_aggregation(float * cost_image, float *cost_new_image,\
	int *HorzOffset_ref, int *HorzOffset_tar, int *VertOffset_ref, int *VertOffset_tar,  \
	int rdim, int cdim, int dispRange){
	int idx_ref,idx_tar,tarx,tOffset,bOffset,lOffset,rOffset,idx,idx_t,idx_left,idx_right;
	int cury, curx,curx_l,curx_r,cnt;
	float costSum;
	for(int y=0; y<rdim; y++)	
		for(int x=0; x<cdim; x++){
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
						curx_l = x-lOffset-1;
						curx_r = x+rOffset;
						if(curx_l < 0)
							curx_l = 0;
						else if(curx_r >= cdim)
							curx_r = cdim-1;
						if(curx_l >= 0 && curx_r < cdim){
							idx_left = cury*cdim+curx_l;
							idx_right = cury*cdim+curx_r;
							costSum += (cost_image[d+dispRange*idx_right]-cost_image[d+dispRange*idx_left]);
							cnt += rOffset+lOffset+1;
						}
					}
				}
				costSum /= cnt;
				cost_new_image[d+dispRange*idx] = costSum;
			}
		}
}

void disp_refinement(unsigned char *dispIm,int *HorzOffset, int *VertOffset, int rdim, int cdim, int dispRange){
	int idx,idx_left,idx_right,idx_top,idx_bottom,idx_ref;
	int tOffset,bOffset,lOffset,rOffset;
	int cury,curx_l,curx_r,dispVal;
	int *dispHist = (int *) malloc(dispRange*sizeof(int));
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			idx =y*cdim+x;
			tOffset = VertOffset[2*idx];
			bOffset = VertOffset[2*idx+1];
			memset(dispHist,0,dispRange*sizeof(int));
			for(int y_=-tOffset; y_<bOffset; y_++){
				cury = y+y_;
				if(cury >= 0 && cury < rdim){
					idx_ref = x + cury*cdim;
					lOffset = HorzOffset[2*idx_ref];
					rOffset = HorzOffset[2*idx_ref+1];
					curx_l = x-lOffset;
					curx_r = x+rOffset;
					if(curx_l < 0)
						curx_l = 0;
					else if(curx_r >= cdim)
						curx_r = cdim-1;
					for(int x_=curx_l; x_<=curx_r; x_++){
						idx_ref = cury*cdim+x_;
						dispVal = dispIm[idx_ref];
						dispHist[dispVal]++;
					}
				}
			}
			int maxVal = dispHist[0];
			int maxDisp = 0;
			for(int i=1; i<dispRange; i++){
				if(dispHist[i]>maxVal){
					maxVal = dispHist[i];
					maxDisp = i;
				}
			}
			dispIm[idx] = (unsigned char)maxDisp;

		}
	
}