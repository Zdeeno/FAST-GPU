#include "FAST.hpp"

void show_image(cv::Mat img) {
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Display window", img);
	cv::waitKey(0);
}

__host__ void create_circle(int *circle, int w) {
	// create surrounding circle using given width
	circle[0] = -3 * w;
	circle[1] = -3 * w + 1;
	circle[2] = -2 * w + 2;
	circle[3] = -w + 3;

	circle[4] = 3;
	circle[5] = w + 3;
	circle[6] = 2 * w + 2;
	circle[7] = 3 * w + 1;

	circle[8] = 3 * w;
	circle[9] = 3 * w - 1;
	circle[10] = 2 * w - 2;
	circle[11] = w - 3;

	circle[12] = -3;
	circle[13] = -w - 3;
	circle[14] = -2 * w - 2;
	circle[15] = -3 * w - 1;
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

int main(int argc, char **argv)
{
	if (argc < 2) {
		printf("\n --- Path to image must be specified in arguments ... quiting ---");
		return 1;
	}

	/*
	/// parse arguments
	int opt;
	while ((opt = getopt(argc, argv, "s")) != -1)
	{
		switch (opt)
		{
		case 's':
		case ‘l’:
		case ‘:’:
			printf(“option needs a value\n”);
			break;
			case ‘ ? ’ :
			printf(“unknown option : %c\n”, optopt);
			break;
		}
	}*/

	const int threshold = 75;
	const int device = 2;
	const int pi = 12;

	/// load image
	cv::Mat image;
	printf("\n --- Runing with argument: %s --- \n", argv[1]);
	image = cv::imread(argv[1], 0);
	cv::Size size(768, 1024);
	resize(image, image, size);

	/// get dimension of image
	int width = image.cols;
	int height = image.rows;
	int length = width * height;
	int shared_width = BLOCK_SIZE + (2 * PADDING);
	size_t char_size = length * sizeof(unsigned char);
	size_t int_size = length * sizeof(unsigned int);
	printf(" --- Image loaded --- \n");

	/// allocate memory
	h_img = (unsigned char*)malloc(char_size);
	h_candidates = (unsigned int*)malloc(int_size);
	h_circle = (int*)malloc(CIRCLE_SIZE*sizeof(int));
	h_mask = (int*)malloc(MASK_SIZE*MASK_SIZE*sizeof(int));
	CHECK_ERROR(cudaMalloc((void**)&d_img, char_size));
	CHECK_ERROR(cudaMalloc((void**)&d_candidates, int_size));
	CHECK_ERROR(cudaMemset(d_candidates, 0, int_size));

	/// create array from image
	for (int i = 0; i < length; i++)
	{
		h_img[i] = image.at<unsigned char>((int)i / image.cols, i % image.cols);
	}

	/// create circle and copy to device
	if (device == 2) {
		printf(" --- Using shared memory --- \n");
		create_circle(h_circle, shared_width);
		create_mask(h_mask, width);
	}
	else {
		printf(" --- Using global memory --- \n");
		create_circle(h_circle, width);
		create_mask(h_mask, width);
	}

	fill_const_mem(h_circle, h_mask);

	/// copy image to device
	CHECK_ERROR(cudaMemcpy(d_img, h_img, char_size, cudaMemcpyHostToDevice));

	/// define grid and block sizes
	dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(((int) (width-1) / BLOCK_SIZE) + 1, ((int) (height-1) / BLOCK_SIZE) + 1);

	/// run kernel and measure the time
	printf(" --- Memory allocated, running kernel --- \n");
	start = clock();
	if (device == 2) {
		int sh_mem = shared_width * shared_width * sizeof(unsigned char);
		FAST_shared <<<grid, blocks, sh_mem>>> (d_img, d_candidates, width, height, threshold, pi);
	}
	else {
		FAST_global <<<grid, blocks>>> (d_img, d_candidates, width, height, threshold, pi);
	}
	
	
	CHECK_ERROR(cudaDeviceSynchronize());
	end = clock();
	time_measured = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf(" --- Image with size (%d, %d) was processed in %f sec --- \n", width, height, time_measured);

	/// copy result to host
	CHECK_ERROR(cudaMemcpy(h_candidates, d_candidates, int_size, cudaMemcpyDeviceToHost));

	/// draw corners 
	printf(" --- Result displayed by host --- \n");
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
