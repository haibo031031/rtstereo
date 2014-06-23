#include "math.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define min(a,b) (a>b) ? b:a
#define max(a,b) (b>a) ? b:a 
#define L 17
#define TAO 20
//#define TRUNC 24

void medianFilt(unsigned char *image, int maskSize);
void calc_x_range(unsigned char * image, int y, int x, int rdim, int cdim, int * leftOffset, int * rightOffset);
void  calc_y_range(unsigned char * image, int y, int x, int rdim, int cdim, int * topOffset, int * bottomOffset);
void findCross(unsigned char *image, int *HorzOffset, int *VertOffset, int rdim, int cdim);
void aggregate_cost_horizontal(float *cost_image, int rdim, int cdim, int dispRange);
void cross_stereo_aggregation(float * cost_image, float *cost_new_image,\
	int *HorzOffset_ref, int *HorzOffset_tar, int *VertOffset_ref, int *VertOffset_tar, \
	int rdim, int cdim, int dispRange);
void disp_refinement(unsigned char *dispIm,int *HorzOffset, int *VertOffset, int rdim, int cdim, int dispRange);