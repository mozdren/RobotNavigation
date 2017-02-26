#pragma once

#include <vector>

#include "Perceptron.hpp"

class Layer{
public:
    std::vector<Perceptron> perceptrons;
    std::vector<float> output;
    int layerSize;
    int inputSize;
    
    Layer(int newLayerSize, int newInputSize);
    Layer(int newLayerSize, int newInputSize, float *params);
    std::vector<float> computeOutput(std::vector<float> input);
    void mutate(float probability, float magnitude);
};
