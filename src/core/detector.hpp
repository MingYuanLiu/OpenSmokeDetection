#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include "model.hpp"
#include "features.hpp"
#include "statistical_features.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>
#include <mutex>

using std::string;
using std::vector;
namespace smoke_adaboost
{
    
class Detector
{

public:
    enum {VIDEO=0, IMAGE=1};
    struct detectorParams
    {
        string modelPath;
        string videoPath;
        string imagePath;
        int detectorModal;
        uint8_t threadNums;
        vector<uint16_t> detWindowSize;
    };
    struct predictRes
    {
        predictRes() {label=0; x=0; y=0; w=0; h=0;}
        int label;
        uint16_t x;
        uint16_t y;
        uint16_t w;
        uint16_t h;
    };
    
    Detector(const detectorParams& _param) { param = _param; }
    ~Detector();
    void run();
    void detectVideo(const string& _videoPath, const string& _modelPath);
    void detectVideoRaw(const string& _videoPath, const string& _modelPath);
    void detectImage(const string& _imagePath, const string& _modelPath);
    void detectImageRaw(const string &_imagePath, const string &_modelPath);
    void drawResult(Mat &frame);

private:
    detectorParams param;
    vector<predictRes> result;
};


} // namespace smoke_adaboost

#endif