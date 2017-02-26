#include "Simulation.hpp"

Simulation::Simulation(Track &t, Car &c){
    this->track = Track(t);
    this->car = Car(c);
    this->playground = cv::Mat(track.height, track.width, cv::DataType<cv::Vec3b>::type);
}
    
float Simulation::simulate(void *inteligence, bool show){
    if (show) cv::namedWindow("Simulation", 0);
    int i=0;
    char filename[100];
    float score = 0.0f;
    while(!car.finished(this->track) && i < 1000){
        car.decide(this->track, inteligence);
        car.move();
        if (show){
            track.drawTrack(this->playground);
            car.drawSensors(this->playground, this->track);
            car.drawPath(this->playground);
            car.drawCar(this->playground);
            cv::imshow("Simulation", this->playground);
            cv::waitKey(10);
        }
        i++;
        if (track.track_image_score.at<unsigned char>(car.position.y, car.position.x) > score){
            score = track.track_image_score.at<unsigned char>(car.position.y, car.position.x);
        }
    }
    if (show) cv::destroyWindow("Simulation");
    return score;
}

