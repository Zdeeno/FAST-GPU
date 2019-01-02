#ifndef CUDA_H
#define CUDA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>

#define PADDING 3
#define BLOCK_SIZE 32	/// max 32
#define CIRCLE_SIZE 16
#define MASK_SIZE 5
#define CHECK_ERROR( error ) ( HandleError( error, __FILE__, __LINE__ ) )

static void HandleError(cudaError_t error, const char *file, int line) {
	if (error != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
		scanf(" ");
		exit(EXIT_FAILURE);
	}
}

/// device variables
static unsigned char *d_img;
static unsigned int *d_candidates;
__constant__ int d_circle[CIRCLE_SIZE];
__constant__ int d_mask[MASK_SIZE*MASK_SIZE];

/// kernel.cu methods
__global__ void FAST_global(unsigned char *input, unsigned int *output, int width, int height, int threshold, int pi);
__global__ void FAST_shared(unsigned char *input, unsigned int *output, int width, int height, int threshold, int pi);
__host__ void fill_const_mem(int *h_circle, int *h_mask);
#endif