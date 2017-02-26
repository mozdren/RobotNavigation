#include "NeuralNetwork.hpp"

#include <cstdio>
#include <cstdlib>

NeuralNetwork::NeuralNetwork(){
}
    
NeuralNetwork::NeuralNetwork(float *data){
    initFromData(data);
    score = 0.0f;
}
    
NeuralNetwork::NeuralNetwork(
    int newInputSize,
    int newLayersCount,
    int *layer_sizes)
{
    inputSize = newInputSize;
    layersCount = newLayersCount;
    int i;
    int layerInputSize;
    for (i=0;i<layersCount;i++){
       if (i==0) layerInputSize = inputSize;
       else layerInputSize = layer_sizes[i-1];
       layers.push_back(Layer(layer_sizes[i], layerInputSize));
    }
    score = 0.0f;
}
    
NeuralNetwork::NeuralNetwork(const char *filename){
    load(filename);
    score = 0.0f;
}
    
void NeuralNetwork::initFromData(float *data){
   layersCount = static_cast<int>(*data); // 0
   printf("Neural Network:\n\tLayers: %d\n", layersCount);
   int i,j;
   int layerSize, layerInputSize;
   inputSize = static_cast<int>(*(data+1)); // 1
   printf("\tInputs dimension: %d\n", inputSize);
   float *pointer = data+1+layersCount+1; // zaciname po hlavicce
   for (i=0;i<layersCount;i++){ // a ted vrstva po vrstve
       printf("\tLayer %d\n", i);
       layerSize = static_cast<int>(*(data+2+i));
       printf("\t\tNeurons in layer: %d\n", layerSize);
       if (i==0) layerInputSize = inputSize;
       else layerInputSize = static_cast<int>(*(data+2+i-1));
       printf("\t\tLayer input dimension: %d\n", layerInputSize);
       layers.push_back(Layer(layerSize, layerInputSize, pointer));
       pointer = pointer + layerSize*(layerInputSize+2);
    }
    score = 0.0f;
}
    
void NeuralNetwork::save(const char *filename){
    FILE *file;
    file = fopen(filename, "w");
    int i,j,k;
    printf("Saving neural network: Layers count:%f Input dimension: %f\n", 
           (float)layers.size(),
           (float)inputSize
    );
    fprintf(file, "%f %f\n", (float)layers.size(), (float)inputSize);
    for (i=0;i<layers.size();i++){
       fprintf(file, "%f ", (float)layers[i].perceptrons.size());
    }
    fprintf(file, "\n");
    for (i=0;i<layers.size();i++){
        for (j=0;j<layers[i].perceptrons.size();j++){
            for (k=0;k<layers[i].perceptrons[j].weights.size();k++){
               fprintf(file, "%f ", layers[i].perceptrons[j].weights[k]);
            }
            fprintf(file, "%f %f\n", 
                    layers[i].perceptrons[j].t, 
                    layers[i].perceptrons[j].lambda);
        }
    }
    fclose(file);
}
    
void NeuralNetwork::load(const char *filename){
    float lc, is;
    FILE *file;
    file = fopen(filename, "r");
    fscanf(file, "%f %f", &lc, &is);
    layersCount = static_cast<int>(lc); // 0
    printf("Neural network from file:\n\tLayer count: %d\n", layersCount);
    int i,j;
    int layerSize, layerInputSize;
    inputSize = static_cast<int>(is); // 1
    printf("\tInput dimension: %d\n", inputSize);
    float *layer_sizes = (float*)malloc(layersCount*sizeof(float));
    for (i=0;i<layersCount;i++){
        fscanf(file, "%f ", layer_sizes+i);
    }
    for (i=0;i<layersCount;i++){ // a ted vrstva po vrstve
        printf("\tLayer %d\n", i);
        layerSize = layer_sizes[i];
        printf("\t\tNeurons in layer: %d\n", layerSize);
        if (i==0) layerInputSize = inputSize;
        else layerInputSize = layer_sizes[i-1];
        printf("\t\tLayer input dimesion: %d\n", layerInputSize);
        float *layer_data = 
           (float*)malloc(layerSize*(layerInputSize+2)*sizeof(float));
        for (j=0;j<layerSize*(layerInputSize+2);j++){
            fscanf(file, "%f ", layer_data+j);
        }
        layers.push_back(Layer(layerSize, layerInputSize, layer_data));
        free(layer_data);
    }
    free(layer_sizes);
    score = 0.0f;
}
    
std::vector<float> NeuralNetwork::computeOutput(std::vector<float> input){
    int i;
    std::vector<float> ret, temp;
    ret = layers[0].computeOutput(input);
    for (i=1;i<layersCount;i++){
       temp = layers[i].computeOutput(ret);
       ret = temp;
    }
    return ret;
}
    
void NeuralNetwork::mutate(float probability, float magnitude){
    int i;
    for (i=0;i<layers.size();i++){
        layers[i].mutate(probability, magnitude);
    }
}
   
NeuralNetwork NeuralNetwork::crossover(NeuralNetwork nn, float probability){
    int i,j,k;
    float x;
    int *shape = (int*)malloc(layersCount*sizeof(int));
    for (i=0;i<layersCount;i++) shape[i] = layers[i].perceptrons.size();
    NeuralNetwork ret = NeuralNetwork(inputSize, layersCount, shape);
    free(shape);
    for (i=0;i<layersCount;i++){
        for (j=0;j<layers[i].perceptrons.size();j++){
            for (k=0;k<layers[i].perceptrons[j].weights.size();k++){
                x = (float)(rand()%10000)/10000.0f;
                if (x < probability)
                {
                    ret.layers[i].perceptrons[j].weights[k] = 
                        nn.layers[i].perceptrons[j].weights[k];
                }
                else
                {
                    ret.layers[i].perceptrons[j].weights[k] = 
                        layers[i].perceptrons[j].weights[k];
                }
            }
            x = (float)(rand()%10000)/10000.0f;
            if (x < probability)
            {
                ret.layers[i].perceptrons[j].t =
                    nn.layers[i].perceptrons[j].t;
            }
            else
            {
                ret.layers[i].perceptrons[j].t = layers[i].perceptrons[j].t;
            }
            x = (float)(rand()%10000)/10000.0f;
            if (x < probability)
            {
                ret.layers[i].perceptrons[j].lambda = 
                    nn.layers[i].perceptrons[j].lambda;
            }
            else
            {
                ret.layers[i].perceptrons[j].lambda =
                    layers[i].perceptrons[j].lambda;
            }
        }
    }
    return ret;
}

