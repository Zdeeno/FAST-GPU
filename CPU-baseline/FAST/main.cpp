#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs/imgcodecs.hpp"

using namespace std;
using namespace cv;


const int TRESHOLD = 75;


vector<KeyPoint> detect_opencv(Mat image){
    vector<KeyPoint> keypointsD;
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(TRESHOLD);
    detector->detect(image, keypointsD, Mat());
    return keypointsD;
}


void draw_keypoints(Mat &image, vector<KeyPoint> keypoints){
    for(int i = 0; i < keypoints.size(); i++) {
        circle(image, keypoints[i].pt, 14, Scalar(0, 255, 0), 2);
    }
}


int main(int argc, char** argv) {
    string imageName;
    if (argc > 1){
        imageName = argv[1];
    } else {
        imageName = "../cvut.png";
    }

    // load and invert to grayscale
    Mat img;
    Mat gray;
    img = imread(imageName, IMREAD_COLOR);
    cvtColor(img, gray, COLOR_RGB2GRAY);

    // draw keypoints
    draw_keypoints(img, detect_opencv(gray));

    // resize and show
    Size size(768, 1024);
    resize(img, img, size);
    imshow("image", img);
    waitKey(0);

    return 0;
}