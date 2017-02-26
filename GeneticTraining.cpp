#include "GeneticTraining.hpp"

#include <sys/stat.h>

GeneticTraining::GeneticTraining(int population){
    int shape[3] = {4, 3, 1};
    int i;
    for (i=0;i<population;i++){
        chromosomes.push_back(NeuralNetwork(21, 3, shape));
    }
    best = NeuralNetwork(21, 3, shape);
}
    
GeneticTraining::GeneticTraining(int population, const char* filename){
    struct stat buffer;   
    int shape[3] = {4, 3, 1};
    int i;
    for (i=0;i<population;i++){
        chromosomes.push_back(NeuralNetwork(21, 3, shape));
    }
    if (stat (filename, &buffer) == 0){
        best = NeuralNetwork(filename);
    }else{
        best = NeuralNetwork(21, 3, shape);
    }
}
    
void GeneticTraining::train(int maxiter, float mutprob, float mutmag, float crossprob, Track t){
    int iter = 0;
    float score;
    Car fc(t);
    Simulation fs(t,fc);
    score = fs.simulate(&best,false);
    best.score = score;
    while (iter < maxiter){
        int i;
        for (i=0;i<chromosomes.size();i++){
            chromosomes[i] = best.crossover(chromosomes[i], crossprob);
            chromosomes[i].mutate(mutprob, mutmag);
        }
        for (i=0;i<chromosomes.size();i++){
            Car c(t);
            Simulation s(t,c);
            score = s.simulate(&chromosomes[i],false);
            chromosomes[i].score = score;
            if (score > best.score){
                printf("New best: %f\n", score);
                best = chromosomes[i];
                best.save("best.txt");
            }
        }
        printf("ITERATION %d, best score: %f\n", iter, best.score);
        Car c(t);
        Simulation s(t,c);
        s.simulate(&best,true);
        iter++;
    }
}
    
void GeneticTraining::train(int maxiter, float mutprob, float mutmag, float crossprob, std::vector<Track> tracks){
    int iter = 0;
    int i,j;
    float score;
    score = 0.0f;
    for (j=0;j<tracks.size();j++){
        Car fc(tracks[j]);
        Simulation fs(tracks[j],fc);
        score += fs.simulate(&best,false);
    }
    best.score = score;
    while (iter < maxiter){
        for (i=0;i<chromosomes.size();i++){
            chromosomes[i] = best.crossover(chromosomes[i], crossprob);
            chromosomes[i].mutate(mutprob, mutmag);
        }
        for (i=0;i<chromosomes.size();i++){
            score = 0.0f;
            for (j=0;j<tracks.size();j++){
                Car c(tracks[j]);
                Simulation s(tracks[j],c);
                score += s.simulate(&chromosomes[i],false);
            }
            chromosomes[i].score = score;
            printf("iteration %d, chromosome %d, score: %f, best score: %f\n", iter, i, score, best.score);
            if (score > best.score){
                printf("New best: %f\n", score);
                best = chromosomes[i];
                best.save("best.txt");
            }
        }
        printf("ITERATION %d, best score: %f\n", iter, best.score);
        for (j=0;j<tracks.size();j++){
            Car c(tracks[j]);
            Simulation s(tracks[j],c);
            //s.simulate(&best,true);
        }
        iter++;
    }
}

