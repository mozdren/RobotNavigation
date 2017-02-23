#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <sys/stat.h>
#include <cstdio>
#include <cmath>

class Perceptron{
public:
    std::vector<float> inputs;
    std::vector<float> weights;
    float y,t,z,lambda;
    int inputSize;
    
    Perceptron(int newInputSize, float *parameters){
        int i;
        for (i=0;i<newInputSize;i++){
            inputs.push_back(0.0f);
            weights.push_back(parameters[i]);
        }
        t = parameters[newInputSize];
        lambda = parameters[newInputSize+1];
        inputSize = newInputSize;
    }
    
    Perceptron(int newInputSize, float newLambda, float newT){
        int i;
        for (i=0;i<newInputSize;i++){
            inputs.push_back(0.0f);
            weights.push_back((float)(rand()%10000)/10000.0f);
        }
        lambda = newLambda;
        t = newT;
        inputSize = newInputSize;
    }
    
    void computeOutput(){
        int i;
        float sum = 0.0f;
        for (i=0;i<inputSize;i++){
            sum += inputs[i]*weights[i];
        }
        z = sum;
        y = 1.0f/(1.0f+std::exp(-lambda*(z-t)));
    }
    
    void mutate(float probability, float magnitude){
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
};

class Layer{
public:
    std::vector<Perceptron> perceptrons;
    std::vector<float> output;
    int layerSize;
    int inputSize;
    
    Layer(int newLayerSize, int newInputSize){
        int i;
        layerSize = newLayerSize;
        inputSize = newInputSize;
        for (i=0;i<newLayerSize;i++){
            perceptrons.push_back(Perceptron(inputSize, (float)(rand()%10000)/10000.0f, (float)(rand()%10000)/10000.0f));
        }
    }
    
    Layer(int newLayerSize, int newInputSize, float *params){
        int i;
        layerSize = newLayerSize;
        inputSize = newInputSize;
        for (i=0;i<newLayerSize;i++){
            perceptrons.push_back(Perceptron(inputSize, params+i*(newInputSize+2))); // vsechny vahy a 2 dalsi parametry
        }
    }
    
    std::vector<float> computeOutput(std::vector<float> input){
        int i;
        output.clear();
        for (i=0;i<layerSize;i++){
            perceptrons[i].inputs = input;
            perceptrons[i].computeOutput();
            output.push_back(perceptrons[i].y);
        }
        return output;
    }
    
    void mutate(float probability, float magnitude){
        int i;
        for (i=0;i<perceptrons.size();i++){
            perceptrons[i].mutate(probability, magnitude);
        }
    }
};

class NeuralNetwork{
public:
    std::vector<Layer> layers;
    std::vector<float> output;
    int layersCount;
    int inputSize;
    float score;
    
    NeuralNetwork(){
    }
    
    NeuralNetwork(float *data){
        initFromData(data);
        score = 0.0f;
    }
    
    NeuralNetwork(int newInputSize, int newLayersCount, int *layer_sizes){
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
    
    NeuralNetwork(const char *filename){
        load(filename);
        score = 0.0f;
    }
    
    void initFromData(float *data){
        layersCount = static_cast<int>(*data); // 0
        printf("Neuronova Sit:\n\tVrstev: %d\n", layersCount);
        int i,j;
        int layerSize, layerInputSize;
        inputSize = static_cast<int>(*(data+1)); // 1
        printf("\tdimenze vstupu: %d\n", inputSize);
        float *pointer = data+1+layersCount+1; // zaciname po hlavicce
        for (i=0;i<layersCount;i++){ // a ted vrstva po vrstve
            printf("\tVrstva %d\n", i);
            layerSize = static_cast<int>(*(data+2+i));
            printf("\t\tNeuronu ve vrstve: %d\n", layerSize);
            if (i==0) layerInputSize = inputSize;
            else layerInputSize = static_cast<int>(*(data+2+i-1));
            printf("\t\tVelikost vstupu pro vrstvu: %d\n", layerInputSize);
            layers.push_back(Layer(layerSize, layerInputSize, pointer));
            pointer = pointer + layerSize*(layerInputSize+2);
        }
        score = 0.0f;
    }
    
    void save(const char *filename){
        FILE *file;
        file = fopen(filename, "w");
        int i,j,k;
        printf("Ukladam neuronku: pocet vrstev:%f velikost vstupu: %f\n", (float)layers.size(), (float)inputSize);
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
                fprintf(file, "%f %f\n", layers[i].perceptrons[j].t, layers[i].perceptrons[j].lambda);
            }
        }
        fclose(file);
    }
    
    void load(const char *filename){
        float lc, is;
        FILE *file;
        file = fopen(filename, "r");
        fscanf(file, "%f %f", &lc, &is);
        layersCount = static_cast<int>(lc); // 0
        printf("Neuronova Sit ze souboru:\n\tVrstev: %d\n", layersCount);
        int i,j;
        int layerSize, layerInputSize;
        inputSize = static_cast<int>(is); // 1
        printf("\tdimenze vstupu: %d\n", inputSize);
        float *layer_sizes = (float*)malloc(layersCount*sizeof(float));
        for (i=0;i<layersCount;i++){
            fscanf(file, "%f ", layer_sizes+i);
        }
        for (i=0;i<layersCount;i++){ // a ted vrstva po vrstve
            printf("\tVrstva %d\n", i);
            layerSize = layer_sizes[i];
            printf("\t\tNeuronu ve vrstve: %d\n", layerSize);
            if (i==0) layerInputSize = inputSize;
            else layerInputSize = layer_sizes[i-1];
            printf("\t\tVelikost vstupu pro vrstvu: %d\n", layerInputSize);
            float *layer_data = (float*)malloc(layerSize*(layerInputSize+2)*sizeof(float));
            for (j=0;j<layerSize*(layerInputSize+2);j++){
                fscanf(file, "%f ", layer_data+j);
            }
            layers.push_back(Layer(layerSize, layerInputSize, layer_data));
            free(layer_data);
        }
        free(layer_sizes);
        score = 0.0f;
    }
    
    std::vector<float> computeOutput(std::vector<float> input){
        int i;
        std::vector<float> ret, temp;
        ret = layers[0].computeOutput(input);
        for (i=1;i<layersCount;i++){
            temp = layers[i].computeOutput(ret);
            ret = temp;
        }
        return ret;
    }
    
    void mutate(float probability, float magnitude){
        int i;
        for (i=0;i<layers.size();i++){
            layers[i].mutate(probability, magnitude);
        }
    }
    
    NeuralNetwork crossover(NeuralNetwork nn, float probability){
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
                        ret.layers[i].perceptrons[j].weights[k] = nn.layers[i].perceptrons[j].weights[k];
                    else
                        ret.layers[i].perceptrons[j].weights[k] = layers[i].perceptrons[j].weights[k];
                }
                x = (float)(rand()%10000)/10000.0f;
                if (x < probability)
                    ret.layers[i].perceptrons[j].t = nn.layers[i].perceptrons[j].t;
                else
                    ret.layers[i].perceptrons[j].t = layers[i].perceptrons[j].t;
                x = (float)(rand()%10000)/10000.0f;
                if (x < probability)
                    ret.layers[i].perceptrons[j].lambda = nn.layers[i].perceptrons[j].lambda;
                else
                    ret.layers[i].perceptrons[j].lambda = layers[i].perceptrons[j].lambda;
            }
        }
        return ret;
    }
    
};

class Track{
public:
    cv::Mat track_image;
    cv::Mat track_image_score;
    cv::Point2f start;
    cv::Point2f finish;
    cv::Mat n;
    float finish_radius;
    unsigned int width, height;
    
    Track(){
    }
    
    Track(const char *filename){
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
    
    Track(const Track &t){
        this->track_image = t.track_image;
        this->width = t.width;
        this->height = t.height;
        this->track_image_score = t.track_image_score;
        this->start = t.start;
        this->finish = t.finish;
        this->finish_radius = t.finish_radius;
        this->n = t.n;
    }
    
    void drawTrack(cv::Mat &dest){
        int y,x,val;
        for (y = 0; y < this->height; y++){
            for (x = 0; x < this->width; x++){
                val = this->track_image.at<unsigned char>(y,x);
                if (val == 0){
                    dest.at<cv::Vec3b>(y,x) = cv::Vec3b(0,127,0);
                }else{
                    dest.at<cv::Vec3b>(y,x) = cv::Vec3b(63,63,63);
                }
                //dest.at<unsigned char>(y,x) = this->track_image.at<unsigned char>(y,x);
            }
        }
        cv::circle(dest, this->finish, (int)this->finish_radius, cv::Scalar(255,255,255), -1);
    }
    
    void drawTrackScore(cv::Mat &dest){
        int y,x,val;
        for (y = 0; y < this->height; y++){
            for (x = 0; x < this->width; x++){
                val = this->track_image_score.at<unsigned char>(y,x);
                dest.at<cv::Vec3b>(y,x) = cv::Vec3b(val,val,val);
            }
        }
    }
};

float getProbeDistance(const Track &track, const cv::Point2f &origin, const cv::Point2f &direction){
    float dist = 0.0f;
    float mag = std::sqrt(direction.x*direction.x + direction.y*direction.y);
    cv::Point2f position = origin;
    while (track.track_image.at<unsigned char>(position.y, position.x) > 0){
        position.x += direction.x;
        position.y += direction.y;
        //std::cout << position << std::endl;
        dist += mag;
        //cv::waitKey(1000);
    }
    return dist;
}

class Car{
public:
    cv::Point2f position;
    cv::Mat direction;
    std::vector<cv::Point2f> path;
    float speed;
    float width;
    float score;
    bool colided;
    
    Car(){
    }
    
    Car(float x, float y, float nx, float ny, float speed = 3.0f, float width = 20.0f){
        this->position = cv::Point2f(x,y);
        this->direction = cv::Mat(2,1,cv::DataType<float>::type);
        this->direction.at<float>(0,0) = nx;
        this->direction.at<float>(1,0) = ny;
        this->speed = speed;
        this->width = width;
        this->colided = false;
        this->score = 0.0;
    }
    
    Car(Track &t, float speed = 3.0f, float width = 20.0f){
        this->position = t.start;
        this->direction = t.n.clone();
        this->speed = speed;
        this->width = width;
        this->colided = false;
        this->score = 0.0;
    }
    
    Car(const Car &c){
        this->position = c.position;
        this->direction = c.direction;
        this->speed = c.speed;
        this->width = c.width;
        this->colided = c.colided;
        this->score = c.score;
    }
    
    float move(){
        this->path.push_back(this->position);
        this->position.x += this->speed * this->direction.at<float>(0,0);
        this->position.y += this->speed * this->direction.at<float>(1,0);
    }
    
    void decide(const Track &t, void *inteligence = NULL){
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
            //if (w <= this->width+5) w = 0.0f;
            //if (w > 100) w = 100;
            input.push_back(w);
        }
        float out = nn->computeOutput(input)[0];
        out = out*2.0f*pi - pi;
        //printf("%f\r", out);
        //fflush(stdout);
        r.x = d.x*cos(out) - d.y*sin(out);
        r.y = d.x*sin(out) + d.y*cos(out);
        mag = std::sqrt(r.x*r.x + r.y*r.y);
        r.x /= mag;
        r.y /= mag;
        this->direction.at<float>(0,0) = r.x;
        this->direction.at<float>(1,0) = r.y;
    }
    
    void drawCar(cv::Mat &dest){
        cv::circle(dest, this->position, (int)(this->width/2.0f), cv::Scalar(196,0,0), -1);
        cv::circle(dest, this->position, (int)(this->width/2.0f), cv::Scalar(127,0,0), 3);
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
    
    void drawPath(cv::Mat &dest){
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
    
    void drawSensors(cv::Mat &dest, const Track &t){
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
            //if (w <= this->width+10) w = 0.0f;
            //if (w > 400) w = 400;
            //r.x += w*v.x;
            //r.y += w*v.y;
            o.x = this->position.x + w*v.x;
            o.y = this->position.y + w*v.y;
            cv::line(dest, this->position, o, cv::Scalar(0,255,255), 2);
        }
        //mag = std::sqrt(r.x*r.x + r.y*r.y);
        //r.x /= mag;
        //r.y /= mag;
        //this->direction.at<float>(0,0) = r.x;
        //this->direction.at<float>(1,0) = r.y;
    }
    
    bool isColision(const Track &track){
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
    
    bool finished(const Track &track){
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
    
};

class Simulation{
public:
    Car car;
    Track track;
    cv::Mat playground;
    
    Simulation(Track &t, Car &c){
        this->track = Track(t);
        this->car = Car(c);
        this->playground = cv::Mat(track.height, track.width, cv::DataType<cv::Vec3b>::type);
    }
    
    float simulate(void *inteligence = NULL, bool show = true){
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
};

class GeneticTraining{
public:
    std::vector<NeuralNetwork> chromosomes;
    NeuralNetwork best;
    int iterations;
    
    GeneticTraining(int population){
        int shape[3] = {4, 3, 1};
        int i;
        for (i=0;i<population;i++){
            chromosomes.push_back(NeuralNetwork(21, 3, shape));
        }
        best = NeuralNetwork(21, 3, shape);
    }
    
    GeneticTraining(int population, const char* filename){
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
    
    void train(int maxiter, float mutprob, float mutmag, float crossprob, Track t){
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
                //printf("iteration %d, chromosome %d, score: %f\n", iter, i, score);
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
    
    void train(int maxiter, float mutprob, float mutmag, float crossprob, std::vector<Track> tracks){
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
                s.simulate(&best,true);
            }
            iter++;
        }
    }
};

int main()
{    
    std::vector<Track> tracks;
    tracks.push_back(Track("draha1.txt"));
    tracks.push_back(Track("U.txt"));
    tracks.push_back(Track("S.txt"));
    tracks.push_back(Track("S2.txt"));
    tracks.push_back(Track("zigzag.txt"));
    tracks.push_back(Track("T.txt"));
    GeneticTraining gt(1000, "best.txt");
    gt.train(300, 0.1f, 100.0f, 0.1f, tracks);
}

