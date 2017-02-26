#pragma once

#include "Layer.hpp"

class NeuralNetwork{
public:
    std::vector<Layer> layers;
    std::vector<float> output;
    int layersCount;
    int inputSize;
    float score;
    
    NeuralNetwork();
    NeuralNetwork(float *data);
    NeuralNetwork(int newInputSize, int newLayersCount, int *layer_sizes);
    NeuralNetwork(const char *filename);

    void initFromData(float *data);
    void save(const char *filename);
    void load(const char *filename);
    std::vector<float> computeOutput(std::vector<float> input);
    void mutate(float probability, float magnitude);
    NeuralNetwork crossover(NeuralNetwork nn, float probability);    
};

