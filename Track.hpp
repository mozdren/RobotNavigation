#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.hpp>

#include "Track.hpp"

class Track{
public:
    cv::Mat track_image;
    cv::Mat track_image_score;
    cv::Point2f start;
    cv::Point2f finish;
    cv::Mat n;
    float finish_radius;
    unsigned int width, height;
    
    Track();
    Track(const char *filename);
    Track(const Track &t);
    
    void drawTrack(cv::Mat &dest);
    void drawTrackScore(cv::Mat &dest);
};
