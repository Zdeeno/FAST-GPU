#ifndef CUDA_H
#define CUDA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#define PADDING 3
#define BLOCK_SIZE 32	/// max 32
#define CIRCLE_SIZE 16
#define MASK_SIZE 3
#define CHECK_ERROR( error ) ( HandleError( error, __FILE__, __LINE__ ) )

static void HandleError(cudaError_t error, const char *file, int line) {
	if (error != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
		scanf(" ");
		exit(EXIT_FAILURE);
	}
}

typedef struct corner {
	unsigned score;
	unsigned x;
	unsigned y;

	__host__ __device__ bool operator()(const corner &c1, const corner &c2) {
		return c1.score > c2.score;
	}

} corner;

/// device variables
static unsigned char *d_img;
static unsigned *d_corner_bools;
static unsigned *d_scores;
static corner *d_corners;
__constant__ int d_circle[CIRCLE_SIZE];
__constant__ int d_mask[MASK_SIZE*MASK_SIZE];
__constant__ int d_mask_shared[MASK_SIZE*MASK_SIZE];

/// kernel.cu methods
__global__ void FAST_global(unsigned char *input, unsigned *scores, unsigned *corner_bools, int width, int height, int threshold, int pi);
__global__ void FAST_shared(unsigned char *input, unsigned *scores, unsigned *corner_bools, int width, int height, int threshold, int pi);
__host__ void fill_const_mem(int *h_circle, int *h_mask, int *h_mask_shared);
__global__ void find_corners(unsigned *scanned_array, corner *result, unsigned *scores, int length, int width);
__device__ __host__ char fast_test(unsigned char *input, int *circle, int threshold, int id);
__device__ __host__ int complex_test(unsigned char *input, unsigned *scores, unsigned *corner_bools, int *circle, int threshold, int pi, int s_id, int g_id);
#endif