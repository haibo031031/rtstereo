//////////////////////////////////////////////////////////////////////
//Name: Main.cpp (Using OpenCV)
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: SOFTWARE FOR COMPARING STEREO MATCHING ALGORITHMS
/////////////////////////////////////////////////////////////////////


#include <stdlib.h>
#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "CLHelper.h"
#include "Math.h"

#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5)) 
#define square(x) x*x
#define TRUNC 24
using namespace cv;

////////////////////////////////////////////////////////////////
////////////////HOST FUNCTION DECLERATIONS//////////////////////
////////////////////////////////////////////////////////////////
void array2cvMat(unsigned char *src,unsigned char *dst,int rdim, int cdim, float coeff);
void array2cvMat(unsigned char *src,unsigned char *dst,int rdim, int cdim);
void array2cvMatFloat(float *src,unsigned char *dst,int rdim, int cdim);
void calcProxWeight(float *proxWeight, float gamma_proximity, int maskRad);
////////////////////////////////////////////////////////////////
//////////////DEVICE FUNCTION DECLERATIONS//////////////////////
////////////////////////////////////////////////////////////////
// Kernel Id 0 //
void pixelBasedCostL2R(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost, 
	int dispRange, int rdim, int cdim);
// Kernel Id 1 //
void pixelBasedCostR2L(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost, 
	int dispRange, int rdim, int cdim);
// Kernel Id 2 //
void pixelBasedCostL2R_Float(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost, 
	int dispRange, int rdim, int cdim);
// Kernel Id 3 //
void pixelBasedCostR2L_Float(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost, 
	int dispRange, int rdim, int cdim);
// Kernel Id 4 //
void pixelBasedCostL2R_Color(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost, 
	int dispRange, int rdim, int cdim, int normOrNot);
// Kernel Id 5 //
void pixelBasedCostL2R_Color_Trunc(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost,
	int dispRange, int rdim, int cdim, float trunc, int normOrNot);
// Kernel Id 6 //
void pixelBasedCostR2L_Color(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost, 
	int dispRange, int rdim, int cdim, int normOrNot);
// Kernel Id 7 //
void pixelBasedCostR2L_Color_Trunc(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost,
	int dispRange, int rdim, int cdim, float trunc, int normOrNot);
// Kernel Id 8 //
void calculateRankFilt(cl_mem d_ref_image, cl_mem d_rankFiltIm, int maskSize, int rdim, 
	int cdim);
// Kernel Id 9 //
void calcCensusTrans(cl_mem image, cl_mem censusTrns, int maskHrad, int maskWrad, 
	int rdim, int cdim);
// Kernel Id 10 //
void calcHammingDist(cl_mem d_censusTrnsRef, cl_mem d_censusTrnsTar, cl_mem d_cost, 
	int maskHrad, int maskWrad, int rdim, int cdim, int dispRange);
// Kernel Id 11 //
void calcNCCDispL2R(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost, 
	int dispRange, int rdim, int cdim, int maskHrad, int maskWrad);
// Kernel Id 12 //
void combineCost(cl_mem d_cost_image1, cl_mem d_cost_image2, float lambda1, 
	float lambda2, int rdim, int cdim, int dispRange);
// Kernel Id 13 //
void convolveImage(cl_mem d_img,cl_mem d_outIm, cl_mem d_mask, int rdim, int cdim, 
	int maskHrad, int maskWrad);
// Kernel Id 14 //
void boxAggregate(cl_mem d_cost, int rdim, int cdim, int dispRange, int wndwRad);
// Kernel Id 15 //
void computeWeights(cl_mem d_weights, cl_mem d_proxWeight, cl_mem d_LABImage,
	float gamma_similarity, int rdim, int cdim, int maskRad);
// Kernel Id 16 //
void calcAWCostL2R(cl_mem d_weightRef, cl_mem d_weightTar, cl_mem d_cost,cl_mem d_cost_new_image,
	 int rdim, int cdim, int dispRange, int maskRad);
// Kernel Id 17 //
void calcAWCostR2L(cl_mem d_weightRef, cl_mem d_weightTar, cl_mem d_cost, int rdim,
	 int cdim, int dispRange, int maskRad);
// Kernel Id 18 //
void findCross(cl_mem d_image, cl_mem d_HorzOffset, cl_mem d_VertOffset, 
	int rdim, int cdim);
// Kernel Id 19 //
void aggregate_cost_horizontal(cl_mem d_cost_image, int rdim, int cdim, 
	int dispRange);
// Kernel Id 20 //
void cross_stereo_aggregation(cl_mem d_cost_image, cl_mem d_cost_new_image, 
	cl_mem d_HorzOffset_ref, cl_mem d_HorzOffset_tar, cl_mem d_VertOffset_ref,   
	cl_mem d_VertOffset_tar, int rdim, int cdim, int dispRange);
// Kernel Id 21 //
void WTA(cl_mem d_totCost, cl_mem d_winners, int rdim, int cdim, int dispRange);
// Kernel Id 22 //
void WTA_NCC(cl_mem d_totCost, cl_mem d_winners, int rdim, int cdim, int dispRange);
// Kernel Id 23 //
void cross_check(cl_mem ref_disparity, cl_mem tar_disparity, cl_mem cross_check_image, 
	int rows, int cols);

/////////////////////////////////////////////////////////////////////////
////////////////MAIN FUNCTION////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
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
	_clInit(0, "gpu", 0);
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
	if(aggrChoice == 2){
		cvtColor(RGB_left_image, LAB_left_image,CV_RGB2Lab); //convert to Lab
		cvtColor(RGB_right_image, LAB_right_image,CV_RGB2Lab);
		LAB_left_pnt = LAB_left_image.data;
		LAB_right_pnt = LAB_right_image.data;
	}
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

	/////////////////////////////////
	//Device Allocation
	/////////////////////////////////
	cl_mem d_ref_image = _clMalloc(rdim * cdim * sizeof(unsigned char));
	cl_mem d_tar_image = _clMalloc(rdim * cdim * sizeof(unsigned char));
	cl_mem d_ref_disparity = _clMalloc(rdim * cdim * sizeof(unsigned char));
	/////////////////////////////////////////////////////////////////////////
	cl_mem d_ref_rgb_image = _clMalloc(rdim * cdim *3* sizeof(unsigned char));
	cl_mem d_tar_rgb_image = _clMalloc(rdim * cdim *3* sizeof(unsigned char));
	/////////////////////////////////////////////////////////////////////////
	cl_mem d_cost_image1, d_cost_image2, d_new_cost_image;
	cl_mem d_rankFiltImL, d_rankFiltImR;
	cl_mem d_censusTrnsR, d_censusTrnsT;
	cl_mem d_LoG_ref_image, d_LoG_tar_image,d_F_LOG;
	cl_mem d_proxWeight, d_weightRef, d_weightTar,d_ref_lab_image,d_tar_lab_image;
	cl_mem d_horz_offset_ref, d_horz_offset_tar, d_vert_offset_ref, d_vert_offset_tar;
	/////////////////////////////////////////////////////////////////////////
	d_cost_image1 = _clMalloc(rdim * cdim * dispRange * sizeof(float));
	if(costChoice2 != 0)
		d_cost_image2 = _clMalloc(rdim * cdim * dispRange * sizeof(float));
	if(aggrChoice == 3 || aggrChoice == 2){
		printf("1 \n");
		d_new_cost_image = _clMalloc(rdim * cdim * dispRange * sizeof(float));
	}

	if(costChoice1 == 2 || costChoice2 == 2){
		d_rankFiltImL = _clMalloc(rdim * cdim * sizeof(unsigned char));
		d_rankFiltImR = _clMalloc(rdim * cdim * sizeof(unsigned char));
	}

	if(costChoice1 == 3 || costChoice2 == 3){
		d_censusTrnsR = _clMalloc(rdim*cdim*maskT*sizeof(bool));
		d_censusTrnsT = _clMalloc(rdim*cdim*maskT*sizeof(bool));
	}

	if(costChoice1 == 4 || costChoice2 == 4){
		d_LoG_ref_image = _clMalloc(rdim * cdim * sizeof(float));
		d_LoG_tar_image = _clMalloc(rdim * cdim * sizeof(float));
		d_F_LOG = _clMalloc(25*sizeof(float));
		_clMemcpyH2D(d_F_LOG, F_LOG, 25 * sizeof(float));
	}
	
	if(aggrChoice == 2){//adaptive weighting
		d_proxWeight = _clMalloc(adapWinArea*sizeof(float));
		d_weightRef = _clMalloc(rdim*cdim*adapWinArea*sizeof(float));
		d_weightTar = _clMalloc(rdim*cdim*adapWinArea*sizeof(float));
		//_clMemset(d_weightRef, 0, rdim*cdim*adapWinArea*sizeof(float));
		//_clMemset(d_weightTar, 0, rdim*cdim*adapWinArea*sizeof(float));
		d_ref_lab_image = _clMalloc(rdim * cdim *3* sizeof(unsigned char));
		d_tar_lab_image = _clMalloc(rdim * cdim *3* sizeof(unsigned char));
		_clMemcpyH2D(d_ref_lab_image,LAB_left_pnt,rdim*cdim*3*sizeof(unsigned char));
		_clMemcpyH2D(d_tar_lab_image,LAB_right_pnt,rdim*cdim*3*sizeof(unsigned char));
	}

	if(aggrChoice == 3){
		d_horz_offset_ref = _clMalloc(rdim*cdim*2*sizeof(int));
		d_horz_offset_tar = _clMalloc(rdim*cdim*2*sizeof(int));
		d_vert_offset_ref = _clMalloc(rdim*cdim*2*sizeof(int));
		d_vert_offset_tar = _clMalloc(rdim*cdim*2*sizeof(int));
	}
	
	
	
	/* upload data (i.e. transfer data from the host to the device) */
	_clMemcpyH2D(d_ref_image, ref_image, rdim * cdim * sizeof(unsigned char));
	_clMemcpyH2D(d_tar_image, tar_image, rdim * cdim * sizeof(unsigned char));
	_clMemcpyH2D(d_ref_rgb_image, RGB_left_pnt, rdim * cdim * 3 * sizeof(unsigned char));
	_clMemcpyH2D(d_tar_rgb_image, RGB_right_pnt, rdim * cdim * 3 * sizeof(unsigned char));
	
	switch(costChoice1){
	case 1: //AD
		if(aggrChoice!=3){
			pixelBasedCostL2R_Color(d_ref_rgb_image,d_tar_rgb_image,d_cost_image1,dispRange,rdim,cdim,0);
		}
		else
			pixelBasedCostL2R_Color_Trunc(d_ref_rgb_image,d_tar_rgb_image,d_cost_image1,dispRange,rdim,cdim,TRUNC,0);
		break;
	case 2:
		calculateRankFilt(d_ref_image,d_rankFiltImL,maskRankSize,rdim,cdim);
		calculateRankFilt(d_tar_image,d_rankFiltImR,maskRankSize,rdim,cdim);
		pixelBasedCostL2R(d_rankFiltImL,d_rankFiltImR,d_cost_image1,dispRange,rdim,cdim);
		break;
	case 3:
		calcCensusTrans(d_ref_image,d_censusTrnsR,maskHrad,maskWrad,rdim,cdim);
		calcCensusTrans(d_tar_image,d_censusTrnsT,maskHrad,maskWrad,rdim,cdim);
		calcHammingDist(d_censusTrnsR,d_censusTrnsT,d_cost_image1,maskHrad,maskWrad, \
		rdim,cdim,dispRange);
		break;
	case 4:
		convolveImage(d_ref_image,d_LoG_ref_image,d_F_LOG,rdim,cdim,2,2);
		convolveImage(d_tar_image,d_LoG_tar_image,d_F_LOG,rdim,cdim,2,2);
		pixelBasedCostL2R_Float(d_LoG_ref_image,d_LoG_tar_image,d_cost_image1,dispRange,rdim,cdim);
		break;
	}
	if(costChoice2 != 0){
		switch(costChoice2){
		case 1:
			if(aggrChoice != 3)
				pixelBasedCostL2R_Color(d_ref_rgb_image,d_tar_rgb_image,d_cost_image2,dispRange,rdim,cdim,1);//1 for normalizing
			else
				pixelBasedCostL2R_Color_Trunc(d_ref_rgb_image,d_tar_rgb_image,d_cost_image2,dispRange,rdim,cdim,TRUNC,0);//1 for normalizing
			break;
		case 2:
			calculateRankFilt(d_ref_image,d_rankFiltImL,maskRankSize,rdim,cdim);
			calculateRankFilt(d_tar_image,d_rankFiltImR,maskRankSize,rdim,cdim);
			pixelBasedCostL2R(d_rankFiltImL,d_rankFiltImR,d_cost_image2,dispRange,rdim,cdim);
			break;
		case 3:
			calcCensusTrans(d_ref_image,d_censusTrnsR,maskHrad,maskWrad,rdim,cdim);
			calcCensusTrans(d_tar_image,d_censusTrnsT,maskHrad,maskWrad,rdim,cdim);
			calcHammingDist(d_censusTrnsR,d_censusTrnsT,d_cost_image2,maskHrad,maskWrad, \
			rdim,cdim,dispRange);
			break;
		case 4:
			convolveImage(d_ref_image,d_LoG_ref_image,d_F_LOG,rdim,cdim,2,2);
			convolveImage(d_tar_image,d_LoG_tar_image,d_F_LOG,rdim,cdim,2,2);
			pixelBasedCostL2R_Float(d_LoG_ref_image,d_LoG_tar_image,d_cost_image1,dispRange,rdim,cdim);
			break;
		}
		combineCost(d_cost_image1,d_cost_image2, lambdaFirst,lambdaSecond,rdim,cdim,dispRange);
	}
	switch(aggrChoice){
	case 1:
		boxAggregate(d_cost_image1,rdim,cdim,dispRange,mask_Rad);
		break;
	case 2:
		calcProxWeight(proxWeight,gamma_proximity,adapWinRad);
		printf("Here 1\n");
		_clMemcpyH2D(d_proxWeight,proxWeight,adapWinArea*sizeof(float));
		printf("Here 2\n");
		computeWeights(d_weightRef,d_proxWeight,d_ref_lab_image,gamma_similarity,rdim,cdim,adapWinRad);
		printf("Here 3\n");
		computeWeights(d_weightTar,d_proxWeight,d_tar_lab_image,gamma_similarity,rdim,cdim,adapWinRad);
		printf("Here 4\n");
		calcAWCostL2R(d_weightRef,d_weightTar,d_cost_image1,d_new_cost_image,rdim,cdim,dispRange,adapWinRad);
		printf("Here 5\n");
		break;
	case 3:
		medianBlur(RGB_left_image,RGB_left_image,3);
		medianBlur(RGB_right_image,RGB_right_image,3);
		_clMemcpyH2D(d_ref_rgb_image,RGB_left_pnt,rdim*cdim*3*sizeof(unsigned char));
		_clMemcpyH2D(d_tar_rgb_image,RGB_right_pnt,rdim*cdim*3*sizeof(unsigned char));
		findCross(d_ref_rgb_image, d_horz_offset_ref, d_vert_offset_ref,rdim,cdim);
		findCross(d_tar_rgb_image, d_horz_offset_tar, d_vert_offset_tar,rdim,cdim);
		aggregate_cost_horizontal(d_cost_image1, rdim, cdim, dispRange);
		cross_stereo_aggregation(d_cost_image1,d_new_cost_image,d_horz_offset_ref,d_horz_offset_tar,
			d_vert_offset_ref,d_vert_offset_tar,rdim,cdim,dispRange);
		break;
	}
	if(aggrChoice == 1)
		WTA(d_cost_image1,d_ref_disparity,rdim,cdim,dispRange);
	else
		WTA(d_new_cost_image,d_ref_disparity,rdim,cdim,dispRange);
	_clMemcpyD2H(ref_disparity,d_ref_disparity,rdim*cdim*sizeof(unsigned char));
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
	array2cvMat(ref_disparity,shwImPnt,rdim,cdim);
	string dash = "_";
	string ext  = ".png";
	outputName = argv[10]+dash+argv[4]+dash+argv[5]+dash+argv[6]+dash+argv[7]+dash+argv[8]+dash+argv[9]+ext;
	std::vector<int> compression_params;
	imwrite(outputName,show_image);
	imshow("Combined Image",show_image);
	waitKey(0);
	_clFree(d_ref_image);
	_clFree(d_tar_image);
	_clFree(d_ref_disparity);
	_clFree(d_cost_image1);
	/*printf("Here!\n");
	if(d_ref_rgb_image != NULL)
		_clFree(d_ref_rgb_image);
	printf("Here 1!\n");
	if(d_tar_rgb_image != NULL)
		_clFree(d_tar_rgb_image);
	printf("Here 2!\n");
	if(d_cost_image2 != NULL)
		_clFree(d_cost_image2);
	printf("Here 3!\n");
	if(d_new_cost_image != NULL)
		_clFree(d_new_cost_image);
	printf("Here 4!\n");
	if(d_rankFiltImL != NULL)
		_clFree(d_rankFiltImL);
	if(d_rankFiltImR != NULL)
		_clFree(d_rankFiltImR);
	if(d_censusTrnsR != NULL)
		_clFree(d_censusTrnsR);
	if(d_censusTrnsT != NULL)
		_clFree(d_censusTrnsT);
	if(d_LoG_ref_image != NULL)
		_clFree(d_LoG_ref_image);
	if(d_LoG_tar_image != NULL)
		_clFree(d_LoG_tar_image);
	if(d_ref_lab_image != NULL)
		_clFree(d_ref_lab_image);
	printf("Here 3!\n");
	if(d_tar_lab_image != NULL)
		_clFree(d_tar_lab_image);
	if(d_weightRef != NULL)
		_clFree(d_weightRef);
	if(d_weightTar != NULL)
		_clFree(d_weightTar);
	if(d_horz_offset_ref != NULL)
		_clFree(d_horz_offset_ref);
	if(d_horz_offset_tar != NULL)
		_clFree(d_horz_offset_tar);
	if(d_vert_offset_ref != NULL)
		_clFree(d_vert_offset_ref);
	if(d_vert_offset_tar != NULL)
		_clFree(d_vert_offset_tar);*/
	//_clRelease();	

}
catch(string msg){
	printf("ERR:%s\n", msg.c_str());
	printf("Error catched\n");
}
return 1;
}

/////////////////////////////////////////////////////HOST FUNCTIONS//////////////////////////////////////////////
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

void calcProxWeight(float *proxWeight, float gamma_proximity, int maskRad){
	int k=0;
	for(int y=-maskRad; y<=maskRad; y++)
		for(int x=-maskRad; x<=maskRad; x++){
			proxWeight[k] = exp(-sqrt((float)(square(y)+square(x)))/gamma_proximity);
			k++;
		}
}

//////////////////////////////////////////////////////GPU FUNCTIONS//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////PIXEL-BASED COST////////////////////////////////
/////////////////////////////////////////////////////////////////////////
void pixelBasedCostL2R(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost, 
	int dispRange, int rdim, int cdim){
	
	int kernel_id = 0;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_ref_image);
	_clSetArgs(kernel_id, arg_idx++, d_tar_image);
	_clSetArgs(kernel_id, arg_idx++, d_cost);
	_clSetArgs(kernel_id, arg_idx++, &dispRange, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

void pixelBasedCostR2L(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost, int dispRange, int rdim, int cdim){
	
	int kernel_id = 1;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_ref_image);
	_clSetArgs(kernel_id, arg_idx++, d_tar_image);
	_clSetArgs(kernel_id, arg_idx++, d_cost);
	_clSetArgs(kernel_id, arg_idx++, &dispRange, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

void pixelBasedCostL2R_Float(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost, int dispRange, int rdim, int cdim){
	
	int kernel_id = 2;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_ref_image);
	_clSetArgs(kernel_id, arg_idx++, d_tar_image);
	_clSetArgs(kernel_id, arg_idx++, d_cost);
	_clSetArgs(kernel_id, arg_idx++, &dispRange, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

void pixelBasedCostR2L_Float(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost, int dispRange, int rdim, int cdim){
	
	int kernel_id = 3;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_ref_image);
	_clSetArgs(kernel_id, arg_idx++, d_tar_image);
	_clSetArgs(kernel_id, arg_idx++, d_cost);
	_clSetArgs(kernel_id, arg_idx++, &dispRange, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

void pixelBasedCostL2R_Color(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost, 
	int dispRange, int rdim, int cdim, int normOrNot){
	
	int kernel_id = 4;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_ref_image);
	_clSetArgs(kernel_id, arg_idx++, d_tar_image);
	_clSetArgs(kernel_id, arg_idx++, d_cost);
	_clSetArgs(kernel_id, arg_idx++, &dispRange, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &normOrNot, sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

void pixelBasedCostL2R_Color_Trunc(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost,
	int dispRange, int rdim, int cdim, float trunc, int normOrNot){

	int kernel_id = 5;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_ref_image);
	_clSetArgs(kernel_id, arg_idx++, d_tar_image);
	_clSetArgs(kernel_id, arg_idx++, d_cost);
	_clSetArgs(kernel_id, arg_idx++, &dispRange, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &trunc, sizeof(float));
	_clSetArgs(kernel_id, arg_idx++, &normOrNot, sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

void pixelBasedCostR2L_Color(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost, 
	int dispRange, int rdim, int cdim, int normOrNot){

	int kernel_id = 6;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_ref_image);
	_clSetArgs(kernel_id, arg_idx++, d_tar_image);
	_clSetArgs(kernel_id, arg_idx++, d_cost);
	_clSetArgs(kernel_id, arg_idx++, &dispRange, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &normOrNot, sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

void pixelBasedCostR2L_Color_Trunc(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost,
	int dispRange, int rdim, int cdim, float trunc, int normOrNot){

	int kernel_id = 7;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_ref_image);
	_clSetArgs(kernel_id, arg_idx++, d_tar_image);
	_clSetArgs(kernel_id, arg_idx++, d_cost);
	_clSetArgs(kernel_id, arg_idx++, &dispRange, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &trunc, sizeof(float));
	_clSetArgs(kernel_id, arg_idx++, &normOrNot, sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////RANK FILTER/////////////////////////////////
/////////////////////////////////////////////////////////////////////////
void calculateRankFilt(cl_mem d_ref_image, cl_mem d_rankFiltIm,int maskSize,int rows,int cols){

	int kernel_id = 8;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_ref_image);
	_clSetArgs(kernel_id, arg_idx++, d_rankFiltIm);
	_clSetArgs(kernel_id, arg_idx++, &maskSize, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &rows, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cols, sizeof(int));
	int range_x = cols;
	int range_y = rows;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	return;
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////CENSUS TRANSFORM////////////////////////////////
/////////////////////////////////////////////////////////////////////////
void calcCensusTrans(cl_mem d_image,cl_mem d_censusTrns,int maskHrad,int maskWrad,int rdim,int cdim){

	int kernel_id = 9;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_image);
	_clSetArgs(kernel_id, arg_idx++, d_censusTrns);
	_clSetArgs(kernel_id, arg_idx++, &maskHrad,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &maskWrad,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &rdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim,sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	return;
}
/////////////////////////////////////////////////////////////////////////
/////////////////////////HAMMING DISTANCE////////////////////////////////
/////////////////////////////////////////////////////////////////////////
void  calcHammingDist(cl_mem d_censusTrnsRef, cl_mem d_censusTrnsTar,cl_mem d_cost, int maskHrad, int maskWrad,int rdim, \
	int cdim,int dispRange){

	int kernel_id = 10;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_censusTrnsRef);
	_clSetArgs(kernel_id, arg_idx++, d_censusTrnsTar);
	_clSetArgs(kernel_id, arg_idx++, d_cost);
	_clSetArgs(kernel_id, arg_idx++, &maskHrad,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &maskWrad,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &rdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &dispRange,sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////NCC DISPARITY///////////////////////////////////
/////////////////////////////////////////////////////////////////////////
void calcNCCDispL2R(cl_mem d_ref_image, cl_mem d_tar_image, cl_mem d_cost,
	int dispRange, int rdim, int cdim, int maskHrad, int maskWrad){

	int kernel_id = 11;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_ref_image);
	_clSetArgs(kernel_id, arg_idx++, d_tar_image);
	_clSetArgs(kernel_id, arg_idx++, d_cost);
	_clSetArgs(kernel_id, arg_idx++, &dispRange, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &maskHrad, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &maskWrad, sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

/////////////////////////////////////////////////////////////////////////
///////////////////////////COMBINE COST//////////////////////////////////
/////////////////////////////////////////////////////////////////////////
void combineCost(cl_mem d_cost_image1, cl_mem d_cost_image2, float lambda1, 
	float lambda2, int rdim, int cdim, int dispRange){

	int kernel_id = 12;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_cost_image1);
	_clSetArgs(kernel_id, arg_idx++, d_cost_image2);
	_clSetArgs(kernel_id, arg_idx++, &lambda1,sizeof(float));
	_clSetArgs(kernel_id, arg_idx++, &lambda2,sizeof(float));
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &dispRange, sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

/////////////////////////////////////////////////////////////////////////
///////////////////////////CONVOLVE IMAGE////////////////////////////////
/////////////////////////////////////////////////////////////////////////
void convolveImage(cl_mem d_img,cl_mem d_outIm, cl_mem d_mask, int rdim, int cdim, int maskHrad, int maskWrad){

	int kernel_id = 13;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_img);
	_clSetArgs(kernel_id, arg_idx++, d_outIm);
	_clSetArgs(kernel_id, arg_idx++, d_mask);
	_clSetArgs(kernel_id, arg_idx++, &rdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &maskHrad,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &maskWrad,sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////BOX COST AGGREGATION////////////////////////////
/////////////////////////////////////////////////////////////////////////
void boxAggregate(cl_mem d_cost, int rdim, int cdim, int dispRange, int wndwRad){

	int kernel_id = 14;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_cost);
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &dispRange, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &wndwRad, sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////ADAPTIVE WEIGHTING AGGREGATION//////////////////
/////////////////////////////////////////////////////////////////////////
void computeWeights(cl_mem d_weights, cl_mem d_proxWeight, cl_mem d_LABImage,
	float gamma_similarity, int rdim, int cdim, int maskRad){

	int kernel_id = 15;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_weights);
	_clSetArgs(kernel_id, arg_idx++, d_proxWeight);
	_clSetArgs(kernel_id, arg_idx++, d_LABImage);
	_clSetArgs(kernel_id, arg_idx++, &gamma_similarity,sizeof(float));
	_clSetArgs(kernel_id, arg_idx++, &rdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &maskRad,sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

void calcAWCostL2R(cl_mem d_weightRef, cl_mem d_weightTar, cl_mem d_cost, cl_mem d_cost_new_image,
	int rdim, int cdim, int dispRange, int maskRad){

	int kernel_id = 16;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_weightRef);
	_clSetArgs(kernel_id, arg_idx++, d_weightTar);
	_clSetArgs(kernel_id, arg_idx++, d_cost);
	_clSetArgs(kernel_id, arg_idx++, d_cost_new_image);
	_clSetArgs(kernel_id, arg_idx++, &rdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &dispRange,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &maskRad,sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

void calcAWCostR2L(cl_mem d_weightRef, cl_mem d_weightTar, cl_mem d_cost, int rdim, int cdim,\
	int dispRange, int maskRad){

	int kernel_id = 17;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_weightRef);
	_clSetArgs(kernel_id, arg_idx++, d_weightTar);
	_clSetArgs(kernel_id, arg_idx++, d_cost);
	_clSetArgs(kernel_id, arg_idx++, &rdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &dispRange,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &maskRad,sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////CROSS-BASED AGGREGATION/////////////////////
/////////////////////////////////////////////////////////////////////////
void findCross(cl_mem d_image, cl_mem d_HorzOffset, cl_mem d_VertOffset, int rdim, int cdim){

	int kernel_id = 18;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_image);
	_clSetArgs(kernel_id, arg_idx++, d_HorzOffset);
	_clSetArgs(kernel_id, arg_idx++, d_VertOffset);
	_clSetArgs(kernel_id, arg_idx++, &rdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim,sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

void aggregate_cost_horizontal(cl_mem d_cost_image, int rdim, int cdim, int dispRange){

	int kernel_id = 19;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_cost_image);
	_clSetArgs(kernel_id, arg_idx++, &rdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &dispRange,sizeof(int));
	int range_x = 1;
	int range_y = rdim;
	int group_x = 1;
	int group_y = 64;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);	
}

void cross_stereo_aggregation(cl_mem d_cost_image, cl_mem d_cost_new_image, cl_mem d_HorzOffset_ref, cl_mem d_HorzOffset_tar, 
	cl_mem d_VertOffset_ref, cl_mem d_VertOffset_tar, int rdim, int cdim, int dispRange){

	int kernel_id = 20;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_cost_image);
	_clSetArgs(kernel_id, arg_idx++, d_cost_new_image);
	_clSetArgs(kernel_id, arg_idx++, d_HorzOffset_ref);
	_clSetArgs(kernel_id, arg_idx++, d_HorzOffset_tar);
	_clSetArgs(kernel_id, arg_idx++, d_VertOffset_ref);
	_clSetArgs(kernel_id, arg_idx++, d_VertOffset_tar);
	_clSetArgs(kernel_id, arg_idx++, &rdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &dispRange,sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////WINNER-TAKES-ALL////////////////////////////////
/////////////////////////////////////////////////////////////////////////
void WTA(cl_mem d_totCost, cl_mem d_winners,int rdim,int cdim,int dispRange){

	int kernel_id = 21;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_totCost);
	_clSetArgs(kernel_id, arg_idx++, d_winners);
	_clSetArgs(kernel_id, arg_idx++, &rdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &dispRange,sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////WINNER-TAKES-ALL FOR NCC////////////////////////
/////////////////////////////////////////////////////////////////////////
void WTA_NCC(cl_mem d_totCost, cl_mem d_winners,int rdim,int cdim,int dispRange){

	int kernel_id = 22;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_totCost);
	_clSetArgs(kernel_id, arg_idx++, d_winners);
	_clSetArgs(kernel_id, arg_idx++, &rdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim,sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &dispRange,sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////CROSS-CHECK/////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
void cross_check(cl_mem d_ref_disparity, cl_mem d_tar_disparity, cl_mem d_cross_check_image, int rows, int cols)
{

	int kernel_id = 23;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_ref_disparity);
	_clSetArgs(kernel_id, arg_idx++, d_tar_disparity);
	_clSetArgs(kernel_id, arg_idx++, d_cross_check_image);
	_clSetArgs(kernel_id, arg_idx++, &rows, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cols, sizeof(int));
	int range_x = cols;
	int range_y = rows;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}
