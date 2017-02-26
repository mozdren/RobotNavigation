#pragma once

#include <vector>

class Perceptron{
public:
    std::vector<float> inputs;
    std::vector<float> weights;
    float y,t,z,lambda;
    int inputSize;
    
    Perceptron(int newInputSize, float *parameters);
    Perceptron(int newInputSize, float newLambda, float newT);
    void computeOutput();
    void mutate(float probability, float magnitude);
};

