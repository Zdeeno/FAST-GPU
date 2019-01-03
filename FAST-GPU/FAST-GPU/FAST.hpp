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

/// argument parsing
int threshold = 75;
int mode = 0;
int pi = 12;
char *filename = NULL;
bool video = false;
bool foto = false;

/// host variables
unsigned char *h_img;
unsigned char *h_corner_bools;
int *h_circle;
int *h_mask;
corner *h_corners;

/// time measurement
clock_t start, end;
double time_measured;

#endif