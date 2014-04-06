__kernel void cross_check(__global unsigned char * ref_disparity, __global unsigned char * tar_disparity, \
						__global unsigned char * cross_check_image, const int rows, const int cols)
{
	int c = get_global_id(0);
	int r = get_global_id(1);
	unsigned char val = 0;
	int idx = r * cols + c;
	int ref_val = ref_disparity[idx];
	int tar_x = c - ref_val;
	if(tar_x<0) tar_x += cols;
	int tar_val = tar_disparity[r * cols + tar_x];	//x_tar ??
	if(ref_val == tar_val)
		cross_check_image[idx] = 1;
	else
		cross_check_image[idx] = 0;
	
}