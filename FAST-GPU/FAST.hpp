/**
 * @file FAST.hpp
 * @author Zdenek Rozsypalek (rozsyzde@fel.cvut.cz)
 * @brief Main header for this program. It has all important includes and global variables. 
 * @version 1.0
 * @date 2019-01-07
 * 
 * @copyright Copyright (c) 2019
 * 
 */

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