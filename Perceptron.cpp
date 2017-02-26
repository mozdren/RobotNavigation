#include "Perceptron.hpp"

#include <cstdlib>
#include <cmath>

Perceptron::Perceptron(int newInputSize, float *parameters){
    int i;
    for (i=0;i<newInputSize;i++){
        inputs.push_back(0.0f);
        weights.push_back(parameters[i]);
    }
    t = parameters[newInputSize];
    lambda = parameters[newInputSize+1];
    inputSize = newInputSize;
}
    
Perceptron::Perceptron(int newInputSize, float newLambda, float newT){
    int i;
    for (i=0;i<newInputSize;i++){
        inputs.push_back(0.0f);
        weights.push_back((float)(rand()%10000)/10000.0f);
    }
    lambda = newLambda;
    t = newT;
    inputSize = newInputSize;
}
    
void Perceptron::computeOutput(){
    int i;
    float sum = 0.0f;
    for (i=0;i<inputSize;i++){
        sum += inputs[i]*weights[i];
    }
    z = sum;
    y = 1.0f/(1.0f+std::exp(-lambda*(z-t)));
}
    
void Perceptron::mutate(float probability, float magnitude){
    int i;
    float x,d;
    for (i=0;i<weights.size();i++){
        x = (float)(rand()%10000)/10000.0f;
        d = (float)(rand()%10000)/10000.0f - (float)(rand()%10000)/10000.0f;
        if (x < probability) weights[i] += d*magnitude;
    }
    x = (float)(rand()%10000)/10000.0f;
    d = (float)(rand()%10000)/10000.0f - (float)(rand()%10000)/10000.0f;
    if (x < probability) t += d*magnitude;
    x = (float)(rand()%10000)/10000.0f;
    d = (float)(rand()%10000)/10000.0f - (float)(rand()%10000)/10000.0f;
    if (x < probability) lambda += d*magnitude;
}

