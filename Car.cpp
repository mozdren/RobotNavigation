#include "Car.hpp"

float getProbeDistance(
    const Track &track,
    const cv::Point2f &origin,
    const cv::Point2f &direction
){
    float dist = 0.0f;
    float mag = std::sqrt(direction.x*direction.x + direction.y*direction.y);
    cv::Point2f position = origin;
    while (position.x >= 0 &&
           position.y >= 0 &&
           track.track_image.at<unsigned char>(position.y, position.x) > 0){
        position.x += direction.x;
        position.y += direction.y;
        dist += mag;
    }
    return dist;
}
    
Car::Car(){
}
  
Car::Car(
    float x, 
    float y, 
    float nx, 
    float ny, 
    float speed, 
    float width
){
    this->position = cv::Point2f(x,y);
    this->direction = cv::Mat(2,1,cv::DataType<float>::type);
    this->direction.at<float>(0,0) = nx;
    this->direction.at<float>(1,0) = ny;
    this->speed = speed;
    this->width = width;
    this->colided = false;
    this->score = 0.0;
}
    
Car::Car(Track &t, float speed, float width){
    this->position = t.start;
    this->direction = t.n.clone();
    this->speed = speed;
    this->width = width;
    this->colided = false;
    this->score = 0.0;
}
    
Car::Car(const Car &c){
    this->position = c.position;
    this->direction = c.direction;
    this->speed = c.speed;
    this->width = c.width;
    this->colided = c.colided;
    this->score = c.score;
}
    
float Car::move(){
    this->path.push_back(this->position);
    this->position.x += this->speed * this->direction.at<float>(0,0);
    this->position.y += this->speed * this->direction.at<float>(1,0);
}
    
void Car::decide(const Track &t, void *inteligence){
    cv::Point2f v,d,r;
    int i;
    float pi = 3.1415926f;
    float step = (2.0f*pi)/(float)21;
    float start = -(21-1)/2*step;
    float w, mag;
    r.x = 0;
    r.y = 0;
    d.x = this->direction.at<float>(0,0);
    d.y = this->direction.at<float>(1,0);
    NeuralNetwork *nn = (NeuralNetwork*)inteligence;
    std::vector<float> input;
    for (i=0;i<21;i++){
        v.x = d.x*cos(start+step*i) - d.y*sin(start+step*i);
        v.y = d.x*sin(start+step*i) + d.y*cos(start+step*i);
        w = getProbeDistance(t, this->position, v);
        input.push_back(w);
    }
    float out = nn->computeOutput(input)[0];
    out = out*2.0f*pi - pi;
    r.x = d.x*cos(out) - d.y*sin(out);
    r.y = d.x*sin(out) + d.y*cos(out);
    mag = std::sqrt(r.x*r.x + r.y*r.y);
    r.x /= mag;
    r.y /= mag;
    this->direction.at<float>(0,0) = r.x;
    this->direction.at<float>(1,0) = r.y;
}
    
void Car::drawCar(cv::Mat &dest){
    cv::circle(
        dest,
        this->position,
        (int)(this->width/2.0f),
        cv::Scalar(196,0,0), 
        -1
    );
    cv::circle(
        dest,
        this->position,
        (int)(this->width/2.0f),
        cv::Scalar(127,0,0),
        3
    );
    cv::line(dest,
        this->position,
        cv::Point(
            this->position.x + this->direction.at<float>(0,0)*this->width,
            this->position.y + this->direction.at<float>(1,0)*this->width
        ),
        cv::Scalar(0,0,255),
        3
    );
}

void Car::drawPath(cv::Mat &dest){
    cv::Point p,old;
    std::vector<cv::Point2f>::iterator it;
    it = this->path.begin();
    p = *it;
    old = p;
    for (it = this->path.begin(); it!=this->path.end(); ++it){
        old = p;
        p = *it;
        cv::line(dest, old, p, cv::Scalar(0,255,0), 2);
    }
}
    
void Car::drawSensors(cv::Mat &dest, const Track &t){
    cv::Point2f v,d,r,o;
    int i;
    float pi = 3.1415926f;
    float step = (2.0*pi)/(float)21;
    float start = -(21-1)/2*step;
    float w, mag;
    r.x = 0;
    r.y = 0;
    d.x = this->direction.at<float>(0,0);
    d.y = this->direction.at<float>(1,0);
    for (i=0;i<21;i++){
        v.x = d.x*cos(start+step*i) - d.y*sin(start+step*i);
        v.y = d.x*sin(start+step*i) + d.y*cos(start+step*i);
        w = getProbeDistance(t, this->position, v);
        o.x = this->position.x + w*v.x;
        o.y = this->position.y + w*v.y;
        cv::line(dest, this->position, o, cv::Scalar(0,255,255), 2);
    }
}
    
bool Car::isColision(const Track &track){
    int y,x;
    for (y=this->position.y-this->width/2.0f;y<this->position.y+this->width/2.0f;y++){
        for (x=this->position.x-this->width/2.0f;x<this->position.x+this->width/2.0f;x++){
           if (x>0 && y>0 && x<track.track_image.cols && y<track.track_image.rows){
               if (std::sqrt((this->position.x-x)*(this->position.x-x) + (this->position.y-y)*(this->position.y-y)) <= this->width/2.0f){
                    if (track.track_image.at<unsigned char>(y,x) != 255){
                        this->colided = true;
                        return true;
                    }
                }
            }
        }
    }
    this->colided = false;
    return false;
}
    
bool Car::finished(const Track &track){
    if (this->isColision(track)) return true;
    if (
        std::sqrt(
            (this->position.x-track.finish.x)*(this->position.x-track.finish.x) +
            (this->position.y-track.finish.y)*(this->position.y-track.finish.y)
        ) <= track.finish_radius)
    {
        return true;
    }
    return false;
}

