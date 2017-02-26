#include <vector>

#include "Track.hpp"
#include "GeneticTraining.hpp"

int main()
{    
    std::vector<Track> tracks;
    tracks.push_back(Track("draha1.txt"));
    tracks.push_back(Track("U.txt"));
    tracks.push_back(Track("S.txt"));
    tracks.push_back(Track("S2.txt"));
    tracks.push_back(Track("zigzag.txt"));
    tracks.push_back(Track("T.txt"));
    GeneticTraining gt(100, "best.txt");
    gt.train(300, 0.1f, 100.0f, 0.1f, tracks);
}

