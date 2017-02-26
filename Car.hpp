#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.hpp>

#include "Track.hpp"
#include "NeuralNetwork.hpp"

class Car{
public:
    cv::Point2f position;
    cv::Mat direction;
    std::vector<cv::Point2f> path;
    float speed;
    float width;
    float score;
    bool colided;
    
    Car();
    Car(float x, float y, float nx, float ny, float speed = 3.0f, float width = 20.0f);
    Car(Track &t, float speed = 3.0f, float width = 20.0f);
    Car(const Car &c);
    
    float move();
    void decide(const Track &t, void *inteligence = NULL);
    void drawCar(cv::Mat &dest);
    void drawPath(cv::Mat &dest);
    void drawSensors(cv::Mat &dest, const Track &t);
    bool isColision(const Track &track);
    bool finished(const Track &track);
    
};

