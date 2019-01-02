//----------------------------------------------------------------------------------------
/**
 * \file       FAST.cu
 * \author     Zdenek Rozsypalek
 * \date       2018
 * \brief      CUDA FAST-n algorithm implementation
 *
 *	
 *
*/
//----------------------------------------------------------------------------------------

#ifndef FAST_H
#define FAST_H

#include "cuda.cuh"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

/// host variables
unsigned char *h_img;
unsigned int *h_candidates;
int *h_circle;
int *h_mask;

/// time measurement
clock_t start, end;
double time_measured;

#endif