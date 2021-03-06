#include "cuda.cuh"


/**
 * @brief Comparator using threshold
 * 
 * @param pixel_val value of center pixel
 * @param circle_val value of pixel in circle
 * @param threshold
 * @param sign modifies function of comparator
 * @return char boolean
 */
__device__ __host__ char comparator(unsigned char pixel_val, unsigned char circle_val, int threshold, char sign) {
	/// return boolean if true ... sign parameter gives us criterion
	if (sign == 1) {
		return circle_val > (pixel_val + threshold);
	}
	else {
		return circle_val < (pixel_val - threshold);
	}
}

/**
 * @brief Calculate element of score of given pixel
 * 
 * @param pixel_val value of center pixel
 * @param circle_val value of pixel in circle
 * @param threshold
 * @return int element of score
 */
__device__ __host__ int get_score(int pixel_val, int circle_val, int threshold) {
	/// returns score of circle element, positive when higher, negative when lower intensity
	int val = pixel_val + threshold;
	if (circle_val > val) {
		return circle_val - val;
	}
	else {
		val = pixel_val - threshold;
		if (circle_val < val) {
			return circle_val - val;
		}
		else {
			return 0;
		}
	}
}

/**
 * @brief Recalculate 2D indexing into 1D
 * 
 * @param x
 * @param y
 * @param width width of image
 * @param height height of image
 * @param eliminate_padding boolean telling whether to eliminate borders of image
 * @return int element of score
 */
__device__ int coords_2to1(int x, int y, int width, int height, bool eliminate_padding) {
	if (eliminate_padding && ((x - PADDING) < 0 || (x + PADDING) >= width || (y - PADDING) < 0 || (y + PADDING) >= height)) {
		/// cutout the borders of image, only active when eliminate_padding == true
		return -1;
	}
	else {
		return x + y * width;
	}
}

/**
 * @brief Loads circle and mask from host to device constant memory
 * 
 * @param h_circle circle array
 * @param h_mask mask array
 * @param h_mask_shared mask array for shared memory
 */
__host__ void fill_const_mem(int *h_circle, int *h_mask, int *h_mask_shared) {
	CHECK_ERROR(cudaMemcpyToSymbol(d_circle, h_circle, CIRCLE_SIZE * sizeof(int)));
	CHECK_ERROR(cudaMemcpyToSymbol(d_mask, h_mask, MASK_SIZE * MASK_SIZE * sizeof(int)));
	CHECK_ERROR(cudaMemcpyToSymbol(d_mask_shared, h_mask_shared, MASK_SIZE * MASK_SIZE * sizeof(int)));
	CHECK_ERROR(cudaDeviceSynchronize());
	return;
}

/**
 * @brief Perform fast test on pixel with given id
 * 
 * @param input image array
 * @param circle 
 * @param threshold
 * @param id pixel 1D index
 * @return boolean telling whether it is corner candidate
 */
__device__ __host__ char fast_test(unsigned char *input, int *circle, int threshold, int id) {
	unsigned char pixel = input[id];
	unsigned char top = input[id + d_circle[0]];
	unsigned char right = input[id + d_circle[4]];
	unsigned char down = input[id + d_circle[8]];
	unsigned char left = input[id + d_circle[12]];

	unsigned char sum = comparator(pixel, top, threshold, 1) + comparator(pixel, right, threshold, 1) +
						comparator(pixel, down, threshold, 1) + comparator(pixel, left, threshold, 1);
	if (sum < 3) {
		sum = comparator(pixel, top, threshold, -1) + comparator(pixel, right, threshold, -1) +
			  comparator(pixel, down, threshold, -1) + comparator(pixel, left, threshold, -1);
		if (sum < 3) {
			return 1;
		}
	}
	return 0;
}

/**
 * @brief Run complex test on pixel with given id
 * 
 * @param input image array
 * @param scores array to output score
 * @param corner_bools array to output whether pixel is corner or not
 * @param circle 
 * @param threshold
 * @param pi
 * @param s_id 1D index in shared memory (same as g_id when using only global memory)
 * @param g_id 1D index in global memory
 * @return int score of pixel with given id
 */
__device__ __host__ int complex_test(unsigned char *input, unsigned *scores, unsigned *corner_bools, int *circle, int threshold, int pi, int s_id, int g_id) {	
	/// make complex test and calculate score
	unsigned char pixel = input[s_id];
	int score;
	int score_sum = 0;
	int max_score = 0;
	char val;
	char last_val = -2;
	unsigned char consecutive = 1;
	bool corner = false;
	/// iterate over whole circle
	for (size_t i = 0; i < (CIRCLE_SIZE + pi); i++) 
	{
		if (consecutive >= pi) {
			corner = true;
			if (score_sum > max_score) {
				max_score = score_sum;
			}
		}
		score = get_score(pixel, input[s_id + circle[i % CIRCLE_SIZE]], threshold);
		/// signum
		val = (score < 0) ? -1 : (score > 0); 
		if (val != 0 && val == last_val) {
			consecutive++;
			score_sum += abs(score);
		}
		else {
			consecutive = 1;
			score_sum = abs(score);
		}
		last_val = val;
	}
	if (corner) {
		if (score_sum > max_score) {
			max_score = score_sum;
		}
		corner_bools[g_id] = 1;
		scores[g_id] = max_score;
		return max_score;
	}
	else {
		return 0;
	}
}

/**
 * @brief Kernel computing FAST algorithm using global memory
 * 
 * @param input image array
 * @param scores array to output score
 * @param corner_bools array to output whether pixel is corner or not
 * @param width width of image
 * @param height height of image
 * @param threshold
 * @param pi
 */
__global__ void FAST_global(unsigned char *input, unsigned *scores, unsigned *corner_bools, int width, int height, int threshold, int pi)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	/// get 1d coordinates and cutout borders
	int id1d = coords_2to1(idx, idy, width, height, true);
	if (id1d == -1) {
		return;
	}
	/// fast test, it turns out that it slows the code a little bit
	/*
	if (fast_test(input, d_circle, threshold, id1d)) {
		return;
	}
	*/
	/// complex test
	int max_score = complex_test(input, scores, corner_bools, d_circle, threshold, pi, id1d, id1d);
	/// non-maximal suppresion
	__syncthreads();

	bool erase = false;
	for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)
	{
		if (scores[id1d + d_mask[i]] > max_score) {
			erase = true;
			break;
		}
	}
	__syncthreads();
	if (erase) {
		scores[id1d] = 0;
		corner_bools[id1d] = 0;
	}
	return;
}

/**
 * @brief Kernel computing FAST algorithm using shared memory
 * 
 * @param input image array
 * @param scores array to output score
 * @param corner_bools array to output whether pixel is corner or not
 * @param width width of image
 * @param height height of image
 * @param threshold
 * @param pi
 */
__global__ void FAST_shared(unsigned char *input, unsigned *scores, unsigned *corner_bools, int width, int height, int threshold, int pi)
{
	extern __shared__ unsigned char sData[];
	int max_score = 0;	/// final score of corner in particular thread
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	/// get 1d coordinates and cutout borders
	int id1d = coords_2to1(idx, idy, width, height, true);
	int length = height * width;
	/// fill in shared memory
	int shared_width = BLOCK_SIZE + (2 * PADDING);
	int s_mem_half_size = ((shared_width)*(shared_width)) / 2;
	int index1 = coords_2to1(threadIdx.x, threadIdx.y, BLOCK_SIZE, BLOCK_SIZE, false);
	int index2 = index1 + s_mem_half_size;
	int global_x1 = -PADDING + (index1 % shared_width) + blockIdx.x * blockDim.x;
	int global_y1 = -PADDING + (index1 / shared_width) + blockIdx.y * blockDim.y;
	int global_x2 = -PADDING + (index2 % shared_width) + blockIdx.x * blockDim.x;
	int global_y2 = -PADDING + (index2 / shared_width) + blockIdx.y * blockDim.y;
	int g1 = coords_2to1(global_x1, global_y1, width, height, false);
	int g2 = coords_2to1(global_x2, global_y2, width, height, false);
	if (index1 < s_mem_half_size && g1 > 0 && g2 < length) {
		sData[index1] = input[g1];
		sData[index2] = input[g2];
	}
	__syncthreads();

	int s_id1d = coords_2to1(threadIdx.x + PADDING, threadIdx.y + PADDING, shared_width, shared_width, false);
	

	if (id1d != -1) {
		/// fast test
		if ( // fast_test(sData, d_circle, threshold, s_id1d) &&
			true) {
			/// make complex test and calculate score
			max_score = complex_test(sData, scores, corner_bools, d_circle, threshold, pi, s_id1d, id1d);
		}
	}
	__syncthreads();

	/// refill shared memory
	unsigned *s_data = (unsigned*)sData;
	if (index1 < s_mem_half_size && g1 > 0 && g2 < length) {
		s_data[index1] = scores[g1];
		s_data[index2] = scores[g2];
	}
	__syncthreads();
	
	/// non-max suppresion
	bool erase = false;
	if (id1d != -1) {
		for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)
		{
			if (s_data[s_id1d + d_mask_shared[i]] > max_score) {
				erase = true;
				break;
			}
		}
	}
	__syncthreads();
	if (erase) {
		scores[id1d] = 0;
		corner_bools[id1d] = 0;
	}
	return;
}

/**
 * @brief Kernel to obtain array of corners from scanned array
 * 
 * @param scanned_array array which is output of parallel scan over array of booleans
 * @param result output corners
 * @param scores array of scores of all pixels
 * @param length number of pixels in image
 * @param width width of image
 */
__global__ void find_corners(unsigned *scanned_array, corner *result, unsigned *scores, int length, int width) {
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < length && idx > 0) {
		int prev = idx - 1;
		int val = scanned_array[idx];
		if (scanned_array[prev] < val) {
			result[val - 1].x = idx % width;
			result[val - 1].y = idx / width;
			result[val - 1].score = scores[idx];
		}
	}
}
