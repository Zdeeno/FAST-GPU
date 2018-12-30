
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>

unsigned char *h_img;
unsigned char *d_img;
unsigned char *d_candidates;


static void HandleError(cudaError_t error, const char *file, int line) {
	if (error != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
		scanf(" ");
		exit(EXIT_FAILURE);
	}
}

#define CHECK_ERROR( error ) ( HandleError( error, __FILE__, __LINE__ ) )

__global__ void rgb2gray_kernel(unsigned char *input, unsigned char *output)
{
	
}

using namespace cv;

void show_image(Mat img) {
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	Size size(768, 1024);
	resize(img, img, size);
	imshow("Display window", img);
	waitKey(0);
}

int main(int argc, char **argv)
{

	Mat image;
	image = imread("..\\..\\simple.png", 0);

	int length = image.rows * image.cols;
	size_t my_size = length * sizeof(unsigned char);
	h_img = (unsigned char*) malloc(length * sizeof(unsigned char));

	printf("cols: %d\n", image.cols);

	// create array from image
	for (int i = 0; i < length; i++)
	{
		h_img[i] = image.at<unsigned char>((int)i / image.cols, i % image.cols);
	}

	CHECK_ERROR(cudaMalloc((void**)&d_img, my_size));
	CHECK_ERROR(cudaMalloc((void**)&d_candidates, my_size));
	CHECK_ERROR(cudaMemset(d_candidates, 0, my_size));
	CHECK_ERROR(cudaMemcpy(d_img, h_img, my_size, cudaMemcpyHostToDevice));



	show_image(image);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	CHECK_ERROR(cudaDeviceReset());

    return 0;
}
