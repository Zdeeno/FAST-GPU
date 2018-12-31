
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>

#define PADDING 3
#define BLOCK_SIZE 32  // max 32
#define CIRCLE_SIZE 16
#define PI 12		   // contiguous pixels
#define THRESHOLD 75
#define MASK_SIZE 7	   // Non-maximal suppression (must be odd nummber here), when MASK_SIZE > 7 it access memory outside the image
//#define USE_SHARED

// host
unsigned char *h_img;
unsigned short *h_candidates;
int *h_circle;
int *h_mask;
// time
clock_t start, end;
double time_measured;

// device
unsigned char *d_img;
unsigned short *d_candidates;
__constant__ int d_circle[CIRCLE_SIZE];
__constant__ int d_mask[MASK_SIZE*MASK_SIZE];


static void HandleError(cudaError_t error, const char *file, int line) {
	if (error != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
		scanf(" ");
		exit(EXIT_FAILURE);
	}
}

#define CHECK_ERROR( error ) ( HandleError( error, __FILE__, __LINE__ ) )

__host__ void create_circle(int *circle, int w) {
	// create surrounding circle using given width
	circle[0] = -3*w;
	circle[1] = -3*w + 1;
	circle[2] = -2*w + 2;
	circle[3] = -w + 3;
	
	circle[4] = 3;
	circle[5] = w + 3;
	circle[6] = 2*w + 2;
	circle[7] = 3*w + 1;

	circle[8] = 3*w;
	circle[9] = 3*w - 1;
	circle[10] = 2*w - 2;
	circle[11] = w - 3;

	circle[12] = -3;
	circle[13] = -w - 3;
	circle[14] = -2*w - 2;
	circle[15] = -3*w - 1;
}

__host__ void create_mask(int *mask, int w) {
	// create mask with given defined mask size and width
	int start = (int)-MASK_SIZE / 2;
	int end = (int)MASK_SIZE / 2;
	int index = 0;
	for (int i = start; i <= end; i++)
	{
		for (int j = start; j <= end; j++)
		{
			mask[index] = i * w + j;
			index++;
		}
	}
}

__device__ char comparator(unsigned char pixel_val, unsigned char circle_val) {
	// very similar to get_score, only returns normalised values
	if (circle_val > (pixel_val + THRESHOLD)) {
		return 1;
	}
	else {
		if (circle_val < (pixel_val - THRESHOLD)) {
			return -1;
		}
		else {
			return 0;
		}
	}
}

__device__ char get_score(unsigned char pixel_val, unsigned char circle_val) {
	// returns circle element score, positive when higher, negative when lower intensity
	char val = pixel_val + THRESHOLD;
	if (circle_val > val) {
		return circle_val - val;
	}
	else {
		val = pixel_val - THRESHOLD;
		if (circle_val < val) {
			return -(val - circle_val);
		}
		else {
			return 0;
		}
	}
}

__device__ int coords_2to1(int x, int y, int width, int height, bool eliminate_padding) {
	// recalculate 2d indexes into 1d array
	if (eliminate_padding && ((x - PADDING) < 0 || (x + PADDING) >= width || (y - PADDING) < 0 || (y + PADDING) >= height)) {
		// cutout the borders of image
		return -1;
	}
	else {
		return x + y * width;
	}
}

__global__ void FAST_shared(unsigned char *input, unsigned short *output, int width, int height)
{
	extern __shared__ unsigned char sData[];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	// get 1d coordinates and cutout borders
	int id1d = coords_2to1(idx, idy, width, height, true);
	if (id1d == -1) {
		return;
	}
	// fill in shared memory
	int shared_width = BLOCK_SIZE + (2*PADDING);
	int s_mem_half_size = ((shared_width)*(shared_width))/2;
	int index1 = coords_2to1(threadIdx.x, threadIdx.y, shared_width, shared_width, false);
	if (index1 < s_mem_half_size) {
		int index2 = index1 + s_mem_half_size;
		int global_x1 = -PADDING + (index1 % shared_width) + blockIdx.x * blockDim.x;
		int global_y1 = -PADDING + (index1 / shared_width) + blockIdx.y * blockDim.y;
		int global_x2 = -PADDING + (index2 % shared_width) + blockIdx.x * blockDim.x;
		int global_y2 = -PADDING + (index2 / shared_width) + blockIdx.y * blockDim.y;
		sData[index1] = input[coords_2to1(global_x1, global_y1, width, height, false)];
		sData[index2] = input[coords_2to1(global_x2, global_y2, width, height, false)];
	}
	// fast test
	int s_id1d = coords_2to1(threadIdx.x + PADDING, threadIdx.y + PADDING, shared_width, shared_width, false);
	unsigned char pixel = sData[s_id1d];
	char top = comparator(pixel, sData[s_id1d + d_circle[0]]);
	char down = comparator(pixel, sData[s_id1d + d_circle[8]]);
	char right = comparator(pixel, sData[s_id1d + d_circle[4]]);
	char left = comparator(pixel, sData[s_id1d + d_circle[12]]);
	if (abs(top + down + right + left) < 2 || (abs(top + down) < 2 && abs(left + right) < 2)) {
		return;
	}
	// make complex test and calculate score
	char score;
	int score_sum = 0;
	int max_score = 0;
	char val;
	char last_val = -2;
	unsigned char consecutive = 0;
	bool corner = false;
	for (size_t i = 0; i < (CIRCLE_SIZE + PI); i++) // iterate over whole circle
	{
		if (consecutive >= 12) {
			corner = true;
		}
		score = get_score(pixel, sData[s_id1d + d_circle[i % CIRCLE_SIZE]]);
		val = (score < 0) ? -1 : (score > 0);  // signum
		if (val == last_val) {
			consecutive++;
			score_sum += abs(score);
		}
		else {
			if (score_sum > max_score) {
				max_score = score_sum;
			}
			consecutive = 1;
			score_sum = abs(score);
		}
		last_val = val;
	}
	if (score_sum > max_score) {
		max_score = score_sum;
	}
	if (corner) {
		output[id1d] = (unsigned short) max_score;
	}
	else {
		return;
	}
	__syncthreads();
	// non-maximal suppresion (very time consuming)
	for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)
	{
		if (output[id1d + d_mask[i]] > max_score) {
			return;
		}
	}
	for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)	// if this thread has max value on id1d delete everything around in filter
	{
		if (d_mask[i]) {
			output[id1d + d_mask[i]] = 0;
		}
	}
	return;
}

__global__ void FAST_global(unsigned char *input, unsigned short *output, int width, int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx == 0 && idy == 0) {
		for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)
		{
			printf("mask: %d\n", d_mask[i]);
		}
	}
	// get 1d coordinates and cutout borders
	int id1d = coords_2to1(idx, idy, width, height, true);
	if (id1d == -1) {
		return;
	}
	// fast test
	unsigned char pixel = input[id1d];
	char top = comparator(pixel, input[id1d + d_circle[0]]);
	char down = comparator(pixel, input[id1d + d_circle[8]]);
	char right = comparator(pixel, input[id1d + d_circle[4]]);
	char left = comparator(pixel, input[id1d + d_circle[12]]);
	if (abs(top + down + right + left) < 2 || (abs(top + down) < 2 && abs(left + right) < 2)) {
		return;
	}
	// make complex test and calculate score
	char score;
	int score_sum = 0;
	int max_score = 0;
	char val;
	char last_val = -2;
	unsigned char consecutive = 0;
	bool corner = false;
	for (size_t i = 0; i < (CIRCLE_SIZE+PI); i++) // iterate over whole circle
	{
		if (consecutive >= 12) {
			corner = true;
		}
		score = get_score(pixel, input[id1d + d_circle[i % CIRCLE_SIZE]]);
		val = (score < 0) ? -1 : (score > 0);  // signum
		if (val == last_val) {
			consecutive++;
			score_sum += abs(score);
		}
		else {
			if (score_sum > max_score) {
				max_score = score_sum;
			}
			consecutive = 1;
			score_sum = abs(score);
		}
		last_val = val;
	}
	if (score_sum > max_score) {
		max_score = score_sum;
	}
	if (corner) {
		output[id1d] = (unsigned short) max_score;
	}
	else {
		return;
	}
	__syncthreads();
	printf("MADE IT HERE!\n");
	// non-maximal suppresion (very time consuming)
	for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)
	{
		if (output[id1d + d_mask[i]] > max_score) {
			return;
		}
	}
	for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)	// if this thread has max value on id1d delete everything around in filter
	{
		if (d_mask[i]) {
			output[id1d + d_mask[i]] = 0;
		}
	}
	return;
}

void show_image(cv::Mat img) {
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	//cv::Size size(140, 100);
	//cv::resize(img, img, size);
	cv::imshow("Display window", img);
	cv::waitKey(0);
}

int main(int argc, char **argv)
{
	// load image
	cv::Mat image;
	image = cv::imread("..\\..\\cvut.png", 0);

	// resize image for testing small image
	cv::Size size(768, 1024);
	resize(image, image, size);

	// get dimension of image
	int width = image.cols;
	int height = image.rows;
	int length = width * height;
	int shared_width = BLOCK_SIZE + (2 * PADDING);
	size_t char_size = length * sizeof(unsigned char);
	size_t short_size = length * sizeof(unsigned short);
	printf("\n --- Image loaded --- \n");

	// allocate memory
	h_img = (unsigned char*)malloc(char_size);
	h_candidates = (unsigned short*)malloc(short_size);
	h_circle = (int*)malloc(CIRCLE_SIZE*sizeof(int));
	h_mask = (int*)malloc(MASK_SIZE*MASK_SIZE*sizeof(int));
	CHECK_ERROR(cudaMalloc((void**)&d_img, char_size));
	CHECK_ERROR(cudaMalloc((void**)&d_candidates, short_size));
	CHECK_ERROR(cudaMemset(d_candidates, 0, short_size));

	// create array from image
	for (int i = 0; i < length; i++)
	{
		h_img[i] = image.at<unsigned char>((int)i / image.cols, i % image.cols);
	}

	// create circle and copy to device
	#ifdef USE_SHARED
	create_circle(h_circle, shared_width);
	create_mask(h_mask, width);
	#else
	create_circle(h_circle, width);
	create_mask(h_mask, width);
	#endif
	CHECK_ERROR(cudaMemcpyToSymbol(d_circle, h_circle, CIRCLE_SIZE * sizeof(int)));
	CHECK_ERROR(cudaMemcpyToSymbol(d_mask, h_mask, MASK_SIZE * MASK_SIZE * sizeof(int)));

	// copy image to device
	CHECK_ERROR(cudaMemcpy(d_img, h_img, char_size, cudaMemcpyHostToDevice));

	// define grid and block sizes
	dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(((int) (width-1) / BLOCK_SIZE) + 1, ((int) (height-1) / BLOCK_SIZE) + 1);

	// run kernel and measure the time
	printf(" --- Memory allocated, running kernel --- \n");
	start = clock();
	#ifdef USE_SHARED
	FAST_shared <<< grid, blocks, shared_width*shared_width*sizeof(unsigned char) >>> (d_img, d_candidates, image.cols, image.rows);
	#else
	FAST_global <<< grid, blocks >>> (d_img, d_candidates, image.cols, image.rows);
	#endif
	CHECK_ERROR(cudaDeviceSynchronize());
	end = clock();
	time_measured = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf(" --- Image with size (%d, %d) was processed in %f sec --- \n", width, height, time_measured);

	// copy result to host
	CHECK_ERROR(cudaMemcpy(h_candidates, d_candidates, short_size, cudaMemcpyDeviceToHost));

	printf(" --- Result copied from device to host --- \n");
	// draw corners 
	cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			if (h_candidates[i + j * width]) {
				cv::circle(image, cv::Point(i, j), 3, cv::Scalar(0, 255, 0));
			}
		}
	}

	// show image
	show_image(image);

	// free all memory
	CHECK_ERROR(cudaFree(d_img));
	CHECK_ERROR(cudaFree(d_candidates));
	free(h_img);
	free(h_candidates);
	free(h_mask);
	free(h_circle);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	CHECK_ERROR(cudaDeviceReset());

    return 0;
}
