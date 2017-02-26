#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.hpp>

#include "Car.hpp"
#include "Track.hpp"

class Simulation{
public:
    Car car;
    Track track;
    cv::Mat playground;
    
    Simulation(Track &t, Car &c);
    
    float simulate(void *inteligence = NULL, bool show = true);
};

