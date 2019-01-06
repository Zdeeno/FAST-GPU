#include "FAST.hpp"

void show_image(cv::Mat img) {
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Display window", img);
	cv::waitKey(0);
}

void print_device_array(unsigned int *device_arr, int length) {
	int *print = (int*)malloc(length * sizeof(int));
	CHECK_ERROR(cudaMemcpy(print, device_arr, sizeof(int)*length, cudaMemcpyDeviceToHost));

	for (size_t i = 0; i < length; i++)
	{
		printf("%d, ", print[i]);
	}
	free(print);
}

void create_circle(int *circle, int w) {
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

void create_mask(int *mask, int w) {
	// create mask with given defined mask size and width
	int start = -(int)MASK_SIZE / 2;
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

std::vector<corner> cpu_FAST(unsigned char *input, unsigned *scores, int *mask, int *circle, int width, int height) {
	/// fast test
	std::vector<corner> ret;
	int id1d;
	for (size_t y = PADDING; y < height - PADDING; y++)
	{
		for (size_t x = PADDING; x < width - PADDING; x++)
		{
			id1d = (width * y) + x;
			scores[id1d] = fast_test(input, circle, threshold, id1d);
		}
	}
	/// complex test
	for (size_t y = PADDING; y < height - PADDING; y++)
	{
		for (size_t x = PADDING; x < width - PADDING; x++)
		{
			id1d = (width * y) + x;
			if (scores[id1d] > 0) {
				scores[id1d] = complex_test(input, scores, scores, circle, threshold, pi, id1d, id1d);
			}
		}
	}
	/// non-max suppression
	bool is_max;
	int val;
	for (size_t y = PADDING; y < height - PADDING; y++)
	{
		for (size_t x = PADDING; x < width - PADDING; x++)
		{
			id1d = (width * y) + x;
			val = scores[id1d];
			if (val > 0) {
				is_max = true;
				for (size_t i = 0; i < MASK_SIZE*MASK_SIZE; i++)
				{
					if (val < scores[id1d + mask[i]]) {
						is_max = false;
						break;
					}
				}
				if (is_max) {
					corner c;
					c.score = (unsigned)val;
					c.x = (unsigned)x;
					c.y = (unsigned)y;
					ret.push_back(c);
				}
			}
		}
	}
	return ret;
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

void fill_const_mem(int width, int shared_width) {
	/// create circle and mask and copy to device
	if (mode == 3) {
		printf(" --- Using shared memory --- \n");
		create_circle(h_circle, shared_width);
		create_mask(h_mask_shared, shared_width);
		create_mask(h_mask, width);
		fill_const_mem(h_circle, h_mask, h_mask_shared);
	}
	else {
		printf(" --- Using global memory --- \n");
		create_circle(h_circle, width);
		create_mask(h_mask_shared, shared_width);
		create_mask(h_mask, width);
		fill_const_mem(h_circle, h_mask, h_mask_shared);
	}
}

void preallocate_mem(cv::Mat image, int length) {
	size_t char_size = length * sizeof(unsigned char);
	size_t int_size = length * sizeof(unsigned int);
	printf(" --- Image loaded --- \n");

	/// allocate memory
	h_img = (unsigned char*)malloc(char_size);
	h_corner_bools = (unsigned*)malloc(int_size);
	h_circle = (int*)malloc(CIRCLE_SIZE * sizeof(int));
	h_mask = (int*)malloc(MASK_SIZE*MASK_SIZE * sizeof(int));
	h_mask_shared = (int*)malloc(MASK_SIZE*MASK_SIZE * sizeof(int));
	CHECK_ERROR(cudaMalloc((void**)&d_img, char_size));
	CHECK_ERROR(cudaMalloc((void**)&d_corner_bools, int_size));
	CHECK_ERROR(cudaMalloc((void**)&d_scores, int_size));
	CHECK_ERROR(cudaMemset(d_corner_bools, 0, int_size));
	CHECK_ERROR(cudaMemset(d_scores, 0, int_size));

	/// create array from image and copy image to device
	h_img = image.data;
	CHECK_ERROR(cudaMemcpy(d_img, h_img, char_size, cudaMemcpyHostToDevice));
}

corner* get_corners_gpu(cv::Mat image, int shared_width, int length, int* corners_num) {
	/// define grid and block sizes
	dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(((int)(image.cols - 1) / BLOCK_SIZE) + 1, ((int)(image.rows - 1) / BLOCK_SIZE) + 1);

	if (mode == 3) {
		printf(" --- Memory allocated, running kernel --- \n");
		/// run kernel and measure the time
		start = clock();
		int sh_mem = shared_width * shared_width * sizeof(int);
		FAST_shared << <grid, blocks, sh_mem >> > (d_img, d_scores, d_corner_bools, image.cols, image.rows, threshold, pi);
	}
	else {

		printf(" --- Memory allocated, running kernel --- \n");
		/// run kernel and measure the time
		start = clock();
		FAST_global << <grid, blocks >> > (d_img, d_scores, d_corner_bools, image.cols, image.rows, threshold, pi);
	}
	CHECK_ERROR(cudaDeviceSynchronize());

	/// create new CUDA array of corners with appropriate length
	thrust::device_ptr<unsigned> dev_bools(d_corner_bools);
	thrust::inclusive_scan(dev_bools, dev_bools + length, dev_bools);		/// scanned values
	unsigned number_of_corners;
	d_corner_bools = thrust::raw_pointer_cast(&dev_bools[0]);						/// cast pointer
	CHECK_ERROR(cudaMemcpy(&number_of_corners, &d_corner_bools[length - 1], sizeof(unsigned), cudaMemcpyDeviceToHost));		/// get number of corners from device

	//print!!!
	//print_device_array(d_corner_bools, length);

	printf(" --- Corners found: %d --- \n", number_of_corners);
	*corners_num = number_of_corners;
	if (number_of_corners == 0) {
		return NULL;
	}

	/// alocate array for results
	corner *h_corners;
	h_corners = (corner*)malloc(number_of_corners * sizeof(corner));
	CHECK_ERROR(cudaMalloc((void**)&d_corners, number_of_corners * sizeof(corner)));

	/// find results, sort and transfer to host
	find_corners << < length / (BLOCK_SIZE*BLOCK_SIZE), BLOCK_SIZE*BLOCK_SIZE >> > (d_corner_bools, d_corners, d_scores, length, image.cols);
	CHECK_ERROR(cudaDeviceSynchronize());
	end = clock();
	time_measured = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf(" --- Image with size (%d, %d) was processed in %f sec --- \n", image.cols, image.rows, time_measured);

	thrust::device_ptr<corner> dev_corners(d_corners);

	thrust::sort(dev_corners, dev_corners + number_of_corners, corner());
	d_corners = thrust::raw_pointer_cast(&dev_corners[0]);						/// cast pointer

	CHECK_ERROR(cudaMemcpy(h_corners, d_corners, sizeof(corner)*number_of_corners, cudaMemcpyDeviceToHost));
	return h_corners;
}

void run_on_gpu(cv::Mat image) {
	
	/// get dimension of image
	int length = image.cols * image.rows;
	preallocate_mem(image, length);

	int shared_width = BLOCK_SIZE + (2 * PADDING);
	fill_const_mem(image.cols, shared_width);

	int number_of_corners;
	corner* h_corners = get_corners_gpu(image, shared_width, length, &number_of_corners);

	/// draw corners 
	printf(" --- Result displayed by host --- \n");
	cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
	float start = (float)h_corners[number_of_corners - 1].score;
	float end = h_corners[0].score;
	float rgb_k = 255 / (end - start);
	for (int i = 0; i < number_of_corners; i++)
	{
		// printf("score: %d, ", h_corners[i].score);
		unsigned inc = (h_corners[i].score - start)*rgb_k;
		cv::Scalar color = cv::Scalar(0, inc, 255 - inc);
		cv::circle(image, cv::Point(h_corners[i].x, h_corners[i].y), 3, color);
	}

	/// show image
	show_image(image);

	/// free all memory
	CHECK_ERROR(cudaFree(d_img));
	CHECK_ERROR(cudaFree(d_corner_bools));
	CHECK_ERROR(cudaFree(d_scores));
	free(h_corner_bools);
	free(h_mask);
	free(h_circle);

	/// cudaDeviceReset must be called before exiting in order for profiling and
	/// tracing tools such as Nsight and Visual Profiler to show complete traces.
	CHECK_ERROR(cudaDeviceReset());
}


void run_on_cpu(cv::Mat image) {
	if (mode == 1) {
		std::vector<cv::KeyPoint> keypointsD;

		start = clock();
		cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold, true);
		detector->detect(image, keypointsD, cv::Mat());
		end = clock();

		time_measured = ((double)(end - start)) / CLOCKS_PER_SEC;
		printf(" --- Image with size (%d, %d) was processed in %f sec --- \n", image.cols, image.rows, time_measured);

		printf(" --- Corners found: %d --- \n", keypointsD.size());

		// cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
		for (int i = 0; i < keypointsD.size(); i++) {
			cv::circle(image, keypointsD[i].pt, 3, cv::Scalar(0, 255, 0));
		}
	}
	else {
		cv::Mat gray_img = image.clone();	// create gray copy
		cv::cvtColor(gray_img, gray_img, cv::COLOR_BGR2GRAY);
		h_circle = (int*)malloc(CIRCLE_SIZE * sizeof(int));
		h_mask = (int*)malloc(MASK_SIZE*MASK_SIZE * sizeof(int));
		unsigned *h_scores = (unsigned*)malloc(image.cols*image.rows * sizeof(int));
		create_circle(h_circle, image.cols);
		create_mask(h_mask, image.cols);
		start = clock();
		std::vector<corner> points = cpu_FAST(gray_img.data, h_scores, h_mask, h_circle, image.cols, image.rows);
		end = clock();
		time_measured = ((double)(end - start)) / CLOCKS_PER_SEC;
		printf(" --- Image with size (%d, %d) was processed in %f sec --- \n", image.cols, image.rows, time_measured);

		printf(" --- Corners found: %d --- \n", points.size());

		for (int i = 0; i < points.size(); i++) {
			cv::circle(image, cv::Point(points[i].x, points[i].y), 5, cv::Scalar(0, 255, 0));
		}
	}

	cv::Size size(1280, 720);	// resize for testing
	//resize(image, image, size);
	//show_image(image);
}

int main(int argc, char **argv)
{

	parse_args(argc, argv);

	/// load image
	if (foto) {
		cv::Mat image;
		image = cv::imread(filename, 0);
		cv::Size size(600, 800);	// resize for testing
		resize(image, image, size);

		if (mode > 1) {
			run_on_gpu(image);
		}
		else {
			run_on_cpu(image);
		}
	}
	if (video){
		cv::VideoCapture cap = cv::VideoCapture(filename);
		cv::Mat frame;
		cap >> frame;
		// Capture frame-by-frame
		cv::VideoWriter video = cv::VideoWriter("outcpp.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 24, frame.size(), true);
		while (1) {
			run_on_cpu(frame);
			video.write(frame);
			cap >> frame;
			// If the frame is empty, break immediately
			if (frame.empty())
				break;
		}
		cap.release();
		video.release();
	}
	
    return 0;
}
