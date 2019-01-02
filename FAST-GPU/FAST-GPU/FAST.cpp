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

void parse_args(int argc, char **argv){
	for (size_t i = 1; i < argc; i++)
	{
		std::string arg = std::string(argv[i]);
		if (arg == "-f") filename = argv[i + 1];
		if (arg == "-m") mode = atoi(argv[i + 1]);
		if (arg == "-p") pi = atoi(argv[i + 1]);
		if (arg == "-i") foto = true;
		if (arg == "-v") video = true;
		if (arg == "-t") threshold = atoi(argv[i + 1]);
	}
	if (filename == NULL) {
		printf("\n --- Path to image must be specified in arguments ... quiting ---");
		exit(1);
	}
	if (mode < 0 || mode > 20) {
		printf("\n --- Mode must be in range 0 - 2 ... quiting ---");
		exit(1);
	}
	if (pi < 9 || pi > 12) {
		printf("\n --- Pi must be in range 9 - 12 ... quiting ---");
		exit(1);
	}
	if (threshold < 0 || threshold > 255) {
		printf("\n --- Threshold must be in range 0 - 255 ... quiting ---");
		exit(1);
	}
	printf("\n --- Runing with following setup: --- \n");
	printf("     Threshold: %d\n", threshold);
	printf("     Pi: %d\n", pi);
	printf("     Mode: %d\n", mode);
	printf("     File name: %s\n", filename);
	return;
}

int main(int argc, char **argv)
{

	parse_args(argc, argv);

	/// load image
	cv::Mat image;
	image = cv::imread(filename, 0);
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

	/// define grid and block sizes
	dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(((int)(width - 1) / BLOCK_SIZE) + 1, ((int)(height - 1) / BLOCK_SIZE) + 1);

	/// allocate memory
	h_img = (unsigned char*)malloc(char_size);
	h_corner_bools = (unsigned char*)malloc(int_size);
	h_circle = (int*)malloc(CIRCLE_SIZE*sizeof(int));
	h_mask = (int*)malloc(MASK_SIZE*MASK_SIZE*sizeof(int));
	CHECK_ERROR(cudaMalloc((void**)&d_img, char_size));
	CHECK_ERROR(cudaMalloc((void**)&d_corner_bools, char_size));
	CHECK_ERROR(cudaMalloc((void**)&d_scores, int_size));
	CHECK_ERROR(cudaMalloc((void**)&d_scores, int_size));
	CHECK_ERROR(cudaMemset(d_corner_bools, 0, char_size));
	CHECK_ERROR(cudaMemset(d_scores, 0, int_size));

	/// create array from image and copy image to device
	h_img = image.data;
	CHECK_ERROR(cudaMemcpy(d_img, h_img, char_size, cudaMemcpyHostToDevice));

	/// create circle and copy to device
	if (mode == 2) {
		printf(" --- Using shared memory --- \n");
		create_circle(h_circle, shared_width);
		create_mask(h_mask, width);
		fill_const_mem(h_circle, h_mask);
		printf(" --- Memory allocated, running kernel --- \n");
		/// run kernel and measure the time
		start = clock();
		int sh_mem = shared_width * shared_width * sizeof(unsigned char);
		FAST_shared << <grid, blocks, sh_mem >> > (d_img, d_scores, d_corner_bools, width, height, threshold, pi);
	}
	else {
		printf(" --- Using global memory --- \n");
		create_circle(h_circle, width);
		create_mask(h_mask, width);
		fill_const_mem(h_circle, h_mask);
		printf(" --- Memory allocated, running kernel --- \n");
		/// run kernel and measure the time
		start = clock();
		FAST_global << <grid, blocks >> > (d_img, d_scores, d_corner_bools, width, height, threshold, pi);
	}

	/// create new CUDA array of corners with appropriate length
	thrust::device_ptr<unsigned char> dev_corners(d_corner_bools);						/// 
	thrust::inclusive_scan(dev_corners, dev_corners + length, dev_corners);		/// scanned values
	unsigned number_of_corners;
	d_corner_bools = thrust::raw_pointer_cast(dev_corners);
	CHECK_ERROR(cudaMemcpy(&number_of_corners, &d_corner_bools[length - 1], sizeof(unsigned), cudaMemcpyDeviceToHost));
	printf("number of corners: %d \n", number_of_corners);


	CHECK_ERROR(cudaDeviceSynchronize());
	end = clock();
	time_measured = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf(" --- Image with size (%d, %d) was processed in %f sec --- \n", width, height, time_measured);

	/// copy result to host
	CHECK_ERROR(cudaMemcpy(h_corner_bools, d_corner_bools, char_size, cudaMemcpyDeviceToHost));

	/// draw corners 
	printf(" --- Result displayed by host --- \n");
	cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			if (h_corner_bools[i + j * width]) {
				cv::circle(image, cv::Point(i, j), 3, cv::Scalar(0, 255, 0));
			}
		}
	}

	/// show image
	show_image(image);

	/// free all memory
	CHECK_ERROR(cudaFree(d_img));
	CHECK_ERROR(cudaFree(d_corner_bools));
	free(h_corner_bools);
	free(h_mask);
	free(h_circle);

    /// cudaDeviceReset must be called before exiting in order for profiling and
    /// tracing tools such as Nsight and Visual Profiler to show complete traces.
	CHECK_ERROR(cudaDeviceReset());

    return 0;
}
