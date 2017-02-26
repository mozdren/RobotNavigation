#include "Track.hpp"

Track::Track(){
}
    
Track::Track(const char *filename){
    FILE *file;
    char track_img_fn[100];
    char track_score_img_fn[100];
    float x,y,r;
    file = fopen(filename, "r");
    fscanf(file, "%s", track_img_fn);
    fscanf(file, "%s", track_score_img_fn);
    this->track_image = cv::imread(track_img_fn, 0);
    this->width = this->track_image.cols;
    this->height = this->track_image.rows;
    this->track_image_score = cv::imread(track_score_img_fn, 0);
    fscanf(file, "%f,%f", &x, &y);
    this->start = cv::Point2f(x,y);
    fscanf(file, "%f,%f,%f", &x, &y, &r);
    this->finish = cv::Point2f(x,y);
    this->finish_radius = r;
    fscanf(file, "%f,%f", &x, &y);
    this->n = cv::Mat(2,1,cv::DataType<float>::type);
    n.at<float>(0,0) = x;
    n.at<float>(1,0) = y;
}
    
Track::Track(const Track &t){
    this->track_image = t.track_image;
    this->width = t.width;
    this->height = t.height;
    this->track_image_score = t.track_image_score;
    this->start = t.start;
    this->finish = t.finish;
    this->finish_radius = t.finish_radius;
    this->n = t.n;
}
    
void Track::drawTrack(cv::Mat &dest){
    int y,x,val;
    for (y = 0; y < this->height; y++){
        for (x = 0; x < this->width; x++){
            val = this->track_image.at<unsigned char>(y,x);
            if (val == 0){
                dest.at<cv::Vec3b>(y,x) = cv::Vec3b(0,127,0);
            }else{
                dest.at<cv::Vec3b>(y,x) = cv::Vec3b(63,63,63);
            }
        }
    }
    cv::circle(dest, this->finish, (int)this->finish_radius, cv::Scalar(255,255,255), -1);
}
    
void Track::drawTrackScore(cv::Mat &dest){
    int y,x,val;
    for (y = 0; y < this->height; y++){
        for (x = 0; x < this->width; x++){
            val = this->track_image_score.at<unsigned char>(y,x);
            dest.at<cv::Vec3b>(y,x) = cv::Vec3b(val,val,val);
        }
    }
}

