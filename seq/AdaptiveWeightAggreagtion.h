#include <math.h>
#define square(x) x*x

void calcProxWeight(float *proxWeight,float gamma_proximity,int maskRad);
float calcEuclideanDist3P(int l0,int l1, int a0,int a1,int b0,int b1);
float calcEuclideanDist3P_float(float l0,float l1, float a0,float a1,float b0,float b1);
void computeWeights(float *weights, float *proxWeight,unsigned char *LABImage,const float gamma_similarity,const int rdim, const int cdim,const int maskRad);
void calcAWCostL2R(float *weightRef, float *weightTar, float *cost, float *newcost, const int rdim, const int cdim,\
	const int dispRange, const int maskRad);
void calcAWCostR2L(float *weightRef, float *weightTar, float *cost, const int rdim, const int cdim,\
	const int dispRange, const int maskRad);