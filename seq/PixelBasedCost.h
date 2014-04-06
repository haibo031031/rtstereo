#include "math.h"
#define min(a,b) (a>b) ? b:a
#define max(a,b) (b>a) ? b:a
#define max3(a,b,c) (a>max(b,c)) ? a:max(b,c) 

void pixelBasedL2R(unsigned char *ref_image, unsigned char *tar_image, float *cost, const int dispRange, const int rdim, const int cdim);
void pixelBasedR2L(unsigned char *ref_image, unsigned char *tar_image, float *cost, const int dispRange, const int rdim, const int cdim);
void pixelBasedCostL2R_Float(float *ref_image, float *tar_image, float *cost, const int dispRange, const int rdim, const int cdim);
void pixelBasedCostR2L_Float(float *ref_image, float *tar_image, float *cost, const int dispRange, const int rdim, const int cdim);
void pixelBasedCostL2R_Color(unsigned char *ref_image, unsigned char *tar_image, float *cost, const int dispRange, const int rdim, const int cdim, int normOrNot);
void pixelBasedCostR2L_Color(unsigned char *ref_image, unsigned char *tar_image, float *cost, const int dispRange, const int rdim, const int cdim, int normOrNot);
void pixelBasedCostL2R_Color(unsigned char *ref_image, unsigned char *tar_image, float *cost, const int dispRange, const int rdim, const int cdim, float trunc, int normOrNot);
void pixelBasedCostR2L_Color(unsigned char *ref_image, unsigned char *tar_image, float *cost, const int dispRange, const int rdim, const int cdim, float trunc, int normOrNot);