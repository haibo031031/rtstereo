//////////////////////////////////////////////////////////////////////
//Name: Main.cpp 
//Created date: 4-2-2012
//Modified date: 4-2-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen 
//Discription: Main Function for CUDA
//////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5)) 
#define square(x) x*x
#define TRUNC 24
using namespace cv;

//void* pixelBasedCostL2R(unsigned char* ref_image, unsigned char* tar_image,
	 //float* cost, int dispRange, int rdim, int cdim);
void allocateDeviceMem(int rdim, int cdim, int dispRange, int maskCensusArea, 
	int maskAdapArea, int costChoice1, int costChoice2, int aggrChoice);
void copyHost2Device(unsigned char *ref_image, unsigned char *tar_image, unsigned char *RGB_left_pnt,
	unsigned char *RGB_right_pnt, unsigned char *LAB_left_pnt, unsigned char *LAB_right_pnt,
	int rdim, int cdim, int aggrChoice);
void deallocateDeviceMem(int costChoice1, int costChoice2,int aggrChoice);
void calcProxWeight(float gamma_proximity, int maskRad);
void pixelBasedCostL2R_cuda(int rdim, int cdim, int dispRange, int costChoice);
void pixelBasedCostR2L_cuda(int rdim, int cdim, int dispRange, int costChoice);
void pixelBasedCostL2R_Float_cuda(int rdim, int cdim, int dispRange, int costChoice);
void pixelBasedCostR2L_Float_cuda(int rdim, int cdim, int dispRange, int costChoice);
void pixelBasedCostL2R_Color_cuda(int rdim, int cdim, int dispRange, int normOrNot, int costChoice);
void pixelBasedCostR2L_Color_cuda(int rdim, int cdim, int dispRange, int normOrNot, int costChoice);
void pixelBasedCostL2R_Color_Trunc_cuda(int rdim, int cdim, int dispRange, float trunc, int normOrNot, int costChoice);
void pixelBasedCostR2L_Color_Trunc_cuda(int rdim, int cdim, int dispRange, float trunc, int normOrNot, int costChoice);
void calcRankCost_cuda(int rdim, int cdim, int dispRange, int maskSize, int costChoice);
void calcCensusCost_cuda(int rdim, int cdim, int dispRange, int maskHrad, int maskWrad, int costChoice);
void calcLoGCost_cuda(int rdim, int cdim, int dispRange, int costChoice);
void calcBoxAggregationCost_cuda(int rdim, int cdim, int dispRange, int maskRad);
void calcAdaptiveAggregationCost_cuda(float gamma_proximity, float gamma_similarity, 
	int rdim, int cdim, int dispRange, int maskAdapRad, int maskAdapArea);
void calcCrossAggregation_cuda(int rdim, int cdim, int dispRange);
unsigned char *WTA_cuda(int aggrChoice, int rdim, int cdim, int dispRange);
void calcCombinedCost_cuda(float lambda1, float lambda2, int rdim, int cdim, int dispRange);

void array2cvMat(unsigned char *src,unsigned char *dst,int rdim, int cdim, float coeff);
void array2cvMat(unsigned char *src,unsigned char *dst,int rdim, int cdim);
void array2cvMatFloat(float *src,unsigned char *dst,int rdim, int cdim);

int main(int argc, char ** argv)
{
try{
	if( argc != 11 ){
		printf( "Need two image data and disparity range\n" );
		printf( "Argument 1 : Reference Image Name\n");
		printf( "Argument 2 : Target Image Name\n");
		printf( "Argument 3 : Disparity Range\n");
		printf( "Argument 4 : First Cost Evaluation Type: (1) AD, (2) Rank, (3) Census, (4) LoG \n");
		printf( "Argument 5 : Second Cost Evaluation Type: (0) Single Cost (1) AD, (2) Rank, (3) Census, (4) LoG \n");
		printf( "Argument 6 : Cost Aggregation Type: (1) Block, (2) Adaptive Weight, (3) Cross-Based \n");
		printf( "Argument 7 : Auto Scale or Not: (0)Scale with Pre-determined Coefficient, (1) Auto Scale \n");
		printf( "Argument 8 : Coefficient of First Cost (First Lambda) \n");
		printf( "Argument 9 : Coefficient of Second Cost (Second Lambda) \n");
		printf( "Argument 10: Name of Dataset(Image) \n");
		return -1;
	}
	/////////////////////////////////////////////////////////////////////////
	///////////////Initialization////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////
	Mat RGB_left_image,RGB_right_image;
	Mat LAB_left_image,LAB_right_image;
	unsigned char *RGB_left_pnt, *RGB_right_pnt;
	unsigned char *LAB_left_pnt, *LAB_right_pnt;
	Mat left_image = imread(argv[1],0);//Read left image as gray-scale
	Mat right_image = imread(argv[2],0);//Read right image as gray-scale
	if( !left_image.data || !right_image.data){
		printf( "incorrect image data \n" );
		return -1;
	}
	unsigned char *ref_image = left_image.data;
	unsigned char *tar_image = right_image.data;
	int dispRange = atoi(argv[3]);//disparity range
	int costChoice1 = atoi(argv[4]);
	int costChoice2 = atoi(argv[5]);
	int aggrChoice = atoi(argv[6]);
	int autoScaleOrNot = atoi(argv[7]);
	float lambdaFirst = atoi(argv[8]);
	float lambdaSecond = atoi(argv[9]);
	RGB_left_image = imread(argv[1],1);
	RGB_right_image = imread(argv[2],1);
	RGB_left_pnt = RGB_left_image.data;
	RGB_right_pnt = RGB_right_image.data;
	
	//Parameter Initialization
	int rdim = left_image.rows;
	int cdim = left_image.cols;
	cvtColor(RGB_left_image, LAB_left_image,CV_RGB2Lab); //convert to Lab
	cvtColor(RGB_right_image, LAB_right_image,CV_RGB2Lab);
	LAB_left_pnt = LAB_left_image.data;
	LAB_right_pnt = LAB_right_image.data;
	//Aggregation Window Radius
	int mask_Rad = 1;
	//Sobel Mask Initialization
	float F_SX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float F_SY[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	//Census Mask Initialization
	int maskHrad = 4;
	int maskWrad = 3;
	int maskT = (2*maskHrad+1)*(2*maskWrad+1);
	//Rank Mask Initialization
	int maskRankSize = 7;
	//LoG Mask Initialization
	float F_LOG[25] = { 0.0239, 0.046, 0.0499, 0.046, 0.0239,
						0.046, 0.0061, -0.0923, 0.0061, 0.046,
						0.0499, -0.0923, -0.3182, -0.0923, 0.0499,
						0.046, 0.0061, -0.0923, 0.0061, 0.046,
						0.0239, 0.046, 0.0499, 0.046, 0.0239 };
	//NCC window size
	int NCC_hRadSize = 1;
	int NCC_wRadSize = 1;\
	//Adaptive-Weight Parameters
	int adapWinRad = 5;
	float gamma_similarity = 7.0f;
	float gamma_proximity = 36.0f;
	int adapWinEdge = 2*adapWinRad+1;
	int adapWinArea = adapWinEdge*adapWinEdge;
	printf("win area : %d\n",adapWinArea);
	//Cross-Based Aggregation Parameters
	float trunc = 30;
	//output
	string outputName;

	/////////////////////////////////////////////////////////////////////////
	///////////////Host & Device Memory Allocation///////////////////////////
	/////////////////////////////////////////////////////////////////////////
	//Host Allocation
	Mat show_image(rdim, cdim, CV_8UC1, Scalar(0));
	unsigned char *shwImPnt = show_image.data;

	unsigned char * ref_disparity = (unsigned char *)malloc( rdim * cdim * sizeof(unsigned char));
	unsigned char * rankFiltImage =  (unsigned char *)malloc(rdim * cdim * sizeof(unsigned char));
	float *proxWeight = (float *)malloc(adapWinArea*sizeof(float));

	//Device Allocation	
	allocateDeviceMem(rdim,cdim,dispRange,maskT,adapWinArea,costChoice1,costChoice2,aggrChoice);
	copyHost2Device(ref_image,tar_image,RGB_left_pnt,RGB_right_pnt,LAB_left_pnt,LAB_right_pnt,rdim,cdim,aggrChoice);
	switch(costChoice1){
	case 1: //AD
		if(aggrChoice!=3)
			pixelBasedCostL2R_Color_cuda(rdim,cdim,dispRange,1,1);
		else
			pixelBasedCostL2R_Color_Trunc_cuda(rdim,cdim,dispRange,TRUNC,0,1);
		break;
	case 2:
		calcRankCost_cuda(rdim,cdim,dispRange,maskRankSize,1);
		break;
	case 3:
		calcCensusCost_cuda(rdim,cdim,dispRange,maskHrad,maskWrad,1);
		break;
	case 4:
		calcLoGCost_cuda(rdim,cdim,dispRange,1);
		break;
	}
	if(costChoice2 != 0){
		switch(costChoice2){
		case 1:
			if(aggrChoice != 3)
				pixelBasedCostL2R_Color_cuda(rdim,cdim,dispRange,1,2);
			else
				pixelBasedCostL2R_Color_Trunc_cuda(rdim,cdim,dispRange,TRUNC,0,2);
			break;
		case 2:
			calcRankCost_cuda(rdim,cdim,dispRange,maskRankSize,2);
			break;
		case 3:
			calcCensusCost_cuda(rdim,cdim,dispRange,maskHrad,maskWrad,2);
			break;
		case 4:
			calcLoGCost_cuda(rdim,cdim,dispRange,2);
			break;
		}
		calcCombinedCost_cuda(lambdaFirst,lambdaSecond,rdim,cdim,dispRange);
	}
	switch(aggrChoice){
	case 1:
		calcBoxAggregationCost_cuda(rdim,cdim,dispRange,mask_Rad);
		break;
	case 2:
		calcAdaptiveAggregationCost_cuda(gamma_proximity,gamma_similarity,rdim,cdim,dispRange,adapWinRad,adapWinArea);
		break;
	case 3:
		calcCrossAggregation_cuda(rdim,cdim,dispRange);
		break;
	}
	ref_disparity = WTA_cuda(aggrChoice,rdim,cdim,dispRange);
	if(autoScaleOrNot == 1)
		array2cvMat(ref_disparity,shwImPnt,rdim,cdim);
	else{
		if(dispRange==16){
			array2cvMat(ref_disparity,shwImPnt,rdim,cdim,16);
		}
		else if(dispRange == 20){
			array2cvMat(ref_disparity,shwImPnt,rdim,cdim,8);
		}
		else if(dispRange == 60){
			array2cvMat(ref_disparity,shwImPnt,rdim,cdim,4);
		}
		else {
			printf("Invalid disparity range! \n");
			return 0;
		}
	}
	imshow("Combined Image",show_image);
	waitKey(0);
	deallocateDeviceMem(costChoice1,costChoice2,aggrChoice);

}
catch(string msg){
	printf("ERR:%s\n", msg.c_str());
	printf("Error catched\n");
}
return 1;
}

void array2cvMat(unsigned char *src,unsigned char *dst,int rdim, int cdim, float coeff){
	float res = 0;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			res = round((float)(src[y*cdim+x])*coeff);
			res = (res>255)?255:res; 
			dst[y*cdim+x]=(unsigned char)res;
		}
}

void array2cvMat(unsigned char *src,unsigned char *dst,int rdim, int cdim){
	unsigned char upprBound = 0;
	float coeff = 0,res = 0;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			if(src[y*cdim+x]>upprBound)
				upprBound = src[y*cdim+x];
		}
	coeff = (float)255/(float)upprBound;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			res = round((float)(src[y*cdim+x])*coeff);
			res = (res>255)?255:res; 
			dst[y*cdim+x]=(unsigned char)res;
		}
}

void array2cvMatFloat(float *src,unsigned char *dst,int rdim, int cdim){
	float upprBound = 0;
	float coeff = 0, res = 0;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			if(src[y*cdim+x]>upprBound)
				upprBound = src[y*cdim+x];
		}
	coeff = (float)255/(float)upprBound;
	for(int y=0; y<rdim; y++)
		for(int x=0; x<cdim; x++){
			res = round((float)(src[y*cdim+x])*coeff);
			res = (res>255)?255:res; 
			dst[y*cdim+x]=(unsigned char)res;
		}
}