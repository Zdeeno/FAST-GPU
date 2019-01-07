//----------------------------------------------------------------------------------------
/**
 * \file       FAST.cpp
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
#include <string>
#include <iostream>
#include <vector>

/// argument parsing
int threshold = 75;
int mode = 1;
int pi = 12;
char *filename = NULL;
bool video = false;
bool foto = false;
int circle_size = 5;

/// host variables
unsigned char *h_img;
unsigned *h_corner_bools;
int *h_circle;
int *h_mask;
int *h_mask_shared;

/// streams for gpu video
cudaStream_t memory_s, work_s;

/// time measurement
clock_t start, end;
double time_measured;

#endif