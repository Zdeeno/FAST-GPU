#include "cuda.cuh"


__device__ char comparator(unsigned char pixel_val, unsigned char circle_val, int threshold) {
	/// very similar to get_score, only returns -1,0,1
	if (circle_val > (pixel_val + threshold)) {
		return 1;
	}
	else {
		if (circle_val < (pixel_val - threshold)) {
			return -1;
		}
		else {
			return 0;
		}
	}
}

__device__ char get_score(unsigned char pixel_val, unsigned char circle_val, int threshold) {
	/// returns score of circle element, positive when higher, negative when lower intensity
	char val = pixel_val + threshold;
	if (circle_val > val) {
		return circle_val - val;
	}
	else {
		val = pixel_val - threshold;
		if (circle_val < val) {
			return -(val - circle_val);
		}
		else {
			return 0;
		}
	}
}

__device__ int coords_2to1(int x, int y, int width, int height, bool eliminate_padding) {
	/// recalculate 2d indexes into 1d array
	if (eliminate_padding && ((x - PADDING) < 0 || (x + PADDING) >= width || (y - PADDING) < 0 || (y + PADDING) >= height)) {
		/// cutout the borders of image, only active when eliminate_padding == true
		return -1;
	}
	else {
		return x + y * width;
	}
}

__host__ void fill_const_mem(int *h_circle, int *h_mask) {
	CHECK_ERROR(cudaMemcpyToSymbol(d_circle, h_circle, CIRCLE_SIZE * sizeof(int)));
	CHECK_ERROR(cudaMemcpyToSymbol(d_mask, h_mask, MASK_SIZE * MASK_SIZE * sizeof(int)));
	return;
}

__global__ void FAST_global(unsigned char *input, unsigned *scores, unsigned char *corner_bools, int width, int height, int threshold, int pi)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	/// get 1d coordinates and cutout borders
	int id1d = coords_2to1(idx, idy, width, height, true);
	if (id1d == -1) {
		return;
	}
	/// fast test
	unsigned char pixel = input[id1d];
	char top = comparator(pixel, input[id1d + d_circle[0]], threshold);
	char down = comparator(pixel, input[id1d + d_circle[8]], threshold);
	char right = comparator(pixel, input[id1d + d_circle[4]], threshold);
	char left = comparator(pixel, input[id1d + d_circle[12]], threshold);
	if (abs(top + down + right + left) < 2 || (abs(top + down) < 2 && abs(left + right) < 2)) {
		return;
	}
	/// make complex test and calculate score
	char score;
	int score_sum = 0;
	int max_score = 0;
	char val;
	char last_val = -2;
	unsigned char consecutive = 0;
	bool corner = false;
	for (size_t i = 0; i < (CIRCLE_SIZE + pi); i++) /// iterate over whole circle
	{
		if (consecutive >= 12) {
			corner = true;
		}
		score = get_score(pixel, input[id1d + d_circle[i % CIRCLE_SIZE]], threshold);
		val = (score < 0) ? -1 : (score > 0);  /// signum
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
		scores[id1d] = (unsigned int)max_score;
		corner_bools[id1d] = 1;
	}
	else {
		return;
	}
	__syncthreads();
	/// non-maximal suppresion
	for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)
	{
		if (scores[id1d + d_mask[i]] > max_score) {
			return;
		}
	}
	for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)	/// if this thread has max value on id1d delete everything around in filter
	{
		if (d_mask[i]) {
			scores[id1d + d_mask[i]] = 0;
			corner_bools[id1d + d_mask[i]] = 0;
		}
	}
	return;
}


__global__ void FAST_shared(unsigned char *input, unsigned *scores, unsigned char *corner_bools, int width, int height, int threshold, int pi)
{
	extern __shared__ unsigned char sData[];
	int max_score = 0;	/// final score of corner in particular thread
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	/// get 1d coordinates and cutout borders
	int id1d = coords_2to1(idx, idy, width, height, true);
	/// fill in shared memory
	int shared_width = BLOCK_SIZE + (2 * PADDING);
	int s_mem_half_size = ((shared_width)*(shared_width)) / 2;
	int index1 = coords_2to1(threadIdx.x, threadIdx.y, BLOCK_SIZE, BLOCK_SIZE, false);
	if (index1 < s_mem_half_size) {
		int index2 = index1 + s_mem_half_size;
		int global_x1 = -PADDING + (index1 % shared_width) + blockIdx.x * blockDim.x;
		int global_y1 = -PADDING + (index1 / shared_width) + blockIdx.y * blockDim.y;
		int global_x2 = -PADDING + (index2 % shared_width) + blockIdx.x * blockDim.x;
		int global_y2 = -PADDING + (index2 / shared_width) + blockIdx.y * blockDim.y;
		sData[index1] = input[coords_2to1(global_x1, global_y1, width, height, false)];
		sData[index2] = input[coords_2to1(global_x2, global_y2, width, height, false)];
	}
	if (id1d != -1) {

		__syncthreads();
		/// fast test
		int s_id1d = coords_2to1(threadIdx.x + PADDING, threadIdx.y + PADDING, shared_width, shared_width, false);
		unsigned char pixel = sData[s_id1d];
		char top = comparator(pixel, sData[s_id1d + d_circle[0]], threshold);
		char down = comparator(pixel, sData[s_id1d + d_circle[8]], threshold);
		char right = comparator(pixel, sData[s_id1d + d_circle[4]], threshold);
		char left = comparator(pixel, sData[s_id1d + d_circle[12]], threshold);
		if (!(abs(top + down + right + left) < 2 || (abs(top + down) < 2 && abs(left + right) < 2))) { /// exclude a lot of pixels

			/// make complex test and calculate score
			char score;
			int score_sum = 0;
			char val;
			char last_val = -2;
			unsigned char consecutive = 0;
			bool corner = false;
			for (size_t i = 0; i < (CIRCLE_SIZE + pi); i++) // iterate over whole circle
			{
				if (consecutive >= 12) {
					corner = true;
				}
				score = get_score(pixel, sData[s_id1d + d_circle[i % CIRCLE_SIZE]], threshold);
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
				scores[id1d] = (unsigned int)max_score;
			}
			else {
				return;
			}
		}
	}
	__syncthreads();

	if (max_score > 0) {
		/// non-maximal suppresion (very time consuming)
		for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)
		{
			if (scores[id1d + d_mask[i]] > max_score) {
				return;
			}
		}
		for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)	/// if this thread has max value on id1d delete everything around in filter
		{
			if (d_mask[i]) {
				scores[id1d + d_mask[i]] = 0;
			}
		}
	}
	return;
}

__global__ void scan(unsigned* out, const unsigned* in, unsigned* sums, const unsigned n) {

	unsigned int id = threadIdx.x;
	unsigned int id_offset = n * blockIdx.x;
	unsigned int offset = 1;

	// shared memory:
	extern __shared__ unsigned int shared[];
	shared[2 * id] = in[id_offset + 2 * id];
	shared[2 * id + 1] = in[id_offset + 2 * id + 1];

	// upsweep
	for (int i = n >> 1; i > 0; i = i >> 1)
	{
		__syncthreads();
		if (id < i)
		{
			int a = offset * (2 * id + 1) - 1;
			int b = offset * (2 * id + 2) - 1;
			shared[b] += shared[a];
		}
		offset *= 2;
	}
	if (id == 0) {
		if (sums) sums[blockIdx.x] = shared[n - 1];
		shared[n - 1] = 0;
	}

	// downsweep
	for (int i = 1; i < n; i *= 2)
	{
		offset = offset >> 1;
		__syncthreads();
		if (id < i)
		{
			int a = offset * (2 * id + 1) - 1;
			int b = offset * (2 * id + 2) - 1;
			float tmp = shared[a];
			shared[a] = shared[b];
			shared[b] += tmp;
		}
	}
	__syncthreads();
	out[id_offset + 2 * id] = shared[2 * id];
	out[id_offset + 2 * id + 1] = shared[2 * id + 1];
}