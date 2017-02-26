#pragma once

#include <vector>

#include "NeuralNetwork.hpp"
#include "Simulation.hpp"
#include "Car.hpp"

class GeneticTraining{
public:
    std::vector<NeuralNetwork> chromosomes;
    NeuralNetwork best;
    int iterations;
    
    GeneticTraining(int population);
    GeneticTraining(int population, const char* filename);
    
    void train(int maxiter, float mutprob, float mutmag, float crossprob, Track t);    
    void train(int maxiter, float mutprob, float mutmag, float crossprob, std::vector<Track> tracks);
};
