#include "Layer.hpp"

#include <cstdlib>

Layer::Layer(int newLayerSize, int newInputSize){
    int i;
    layerSize = newLayerSize;
    inputSize = newInputSize;
    for (i=0;i<newLayerSize;i++){
        perceptrons.push_back(
            Perceptron(
                inputSize,
                (float)(rand()%10000)/10000.0f,
                (float)(rand()%10000)/10000.0f
            )
        );
    }
}

Layer::Layer(int newLayerSize, int newInputSize, float *params){
    int i;
    layerSize = newLayerSize;
    inputSize = newInputSize;
    for (i=0;i<newLayerSize;i++){
        perceptrons.push_back(Perceptron(inputSize, params+i*(newInputSize+2)));
    }
}

std::vector<float> Layer::computeOutput(std::vector<float> input){
    int i;
    output.clear();
    for (i=0;i<layerSize;i++){
        perceptrons[i].inputs = input;
        perceptrons[i].computeOutput();
        output.push_back(perceptrons[i].y);
    }
    return output;
}
    
void Layer::mutate(float probability, float magnitude){
    int i;
    for (i=0;i<perceptrons.size();i++){
        perceptrons[i].mutate(probability, magnitude);
    }
}

