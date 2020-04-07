#include <iostream>
#include <opencv2/core.hpp>
#include "main.hpp"
#include <vector>
#include <stdlib.h>

const std::string WORK_HOME = "/Users/mac/Desktop/摄像头烟雾检测/代码/cpp_adaboost/";
void testResult(smoke_adaboost::smokeCascadeDetector& cls, const cv::Ptr<cv::ml::TrainData>& dataset);

int main()
{
    // 流程:
    // 读取图片数据并计算特征
    // 设置参数
    // 开始训练
    // 验证

    /*
    std::string datasetAnnotationFile = "data.txt";
    std::string saveFeaturesPath = "feature_big_new2.yaml";
    smoke_adaboost::smokeFeatureGenerator dataset;
    smoke_adaboost::smokeFeatureGenerator::FeatureParam fp;
    fp.stride = 2;
    fp.windowSize = 8;
    // dataset.loadAnnotationFromFile(datasetAnnotationFile);
    dataset.readFeaturesFromFile(saveFeaturesPath);
    dataset.setFeatureParam(fp);
    // dataset.runGenerator();
    cv::Ptr<cv::ml::TrainData> Data =  dataset.getTrainData();
    // dataset.writeFeaturesToFile(saveFeaturesPath);

    Data->setTrainTestSplitRatio(0.8, true);
    
    cv::Mat trainData = Data->getTrainSamples();
    cv::Mat trainSIdx = Data->getTrainSampleIdx();
    cv::Mat trainResponse = Data->getTrainResponses();
    cv::Mat VarIdx = Data->getVarIdx();
    cv::Ptr<cv::ml::TrainData> trainDataset = cv::ml::TrainData::create(trainData, 
                                                        cv::ml::ROW_SAMPLE, 
                                                        trainResponse, VarIdx, trainSIdx);
    cv::Mat testData = Data->getTestSamples();
    cv::Mat testResponse = Data->getTestResponses();
    cv::Mat testSIdx = Data->getTestSampleIdx();
    cv::Ptr<cv::ml::TrainData> testDataset = cv::ml::TrainData::create(testData,
                                                                        cv::ml::ROW_SAMPLE,
                                                                        testResponse,VarIdx, testSIdx);
    smoke_adaboost::smokeCascadeDetector::DetectorParam dp;
    dp.singleLayerMaxFPR = 0.2;
    dp.singleLayerMinTPR = 0.95;
    dp.stageNums = 20;
    dp.targetFPR = 0.001;
    dp.boostParam.boostType = cv::ml::Boost::DISCRETE;
    dp.boostParam.weakCount = 1500;
    dp.boostParam.weightTrimRate = 0.99;
    dp.DTreeparam.setMaxDepth(2);
    dp.featureParam.windowSize = fp.windowSize;
    dp.featureParam.stride = fp.stride;
    dp.featureParam.tiledCutNums = fp.tiledCutNums;
    dp.featureParam.ringedCutNums = fp.ringedCutNums;
    smoke_adaboost::smokeCascadeDetector classifier(dp);

    classifier.train(trainDataset, testDataset);

    // cv::Mat testImage = cv::imread("/Users/mac/Downloads/烟雾数据集/dataset/smoke/Yuan_3425.jpg");

    // std::vector<float> res;
    // classifier.predict(, res);
    
    // std::cout << "predict:" << std::endl;
    // std::cout << "/Users/mac/Downloads/烟雾数据集/dataset/smoke/Yuan_3425.jpg" << std::endl;
    // std::cout << "label:" << res[0];
    


    // std::vector<float> res;
    
   //  classifier.predict(mmmtest, res);
    // for (int i=0; i<res.size(); i++)
    //      std::cout << "predict:" << res[i] << std::endl;
    
    classifier.evaluate(testDataset);
    double fpr = classifier.readFPR();
    char str[10];
    gcvt(fpr, 4, str);
    std::string saveModelPath = "model-" + std::string(str) + ".yaml";
    std::cout << saveModelPath << std::endl;
    classifier.saveModel(saveModelPath);
    testResult(classifier, Data);
    */

    
    std::string saveModelPath = "model-0.05373.yaml";
    std::string videoPath = "21.mpg";
    std::string imagePath = "51.jpg";
    smoke_adaboost::Detector::detectorParams param;
    param.videoPath = videoPath;
    param.imagePath = imagePath;
    param.modelPath = saveModelPath;
    vector<int> w = {50};
    param.detWindowSize = w;
    smoke_adaboost::Detector test(param);
    test.detectVideoRaw(videoPath, saveModelPath);
    // double t0 = cv::getTickCount();
    // test.detectImage(imagePath, saveModelPath);
    // double t2 = ((double) (cv::getTickCount() - t0) / cv::getTickFrequency()) * 1000000.0;
    // std::cout<<"test 's time"<<t2<<"us"<<std::endl;

        // Data.release();
    
}

void testResult(smoke_adaboost::smokeCascadeDetector& cls, const cv::Ptr<cv::ml::TrainData>& dataset)
{
    int nsamples = dataset->getNSamples();
    CV_Assert(nsamples > 0);
    cv::Mat samples = dataset->getSamples();
    std::vector<float> res;
    cls.predict(samples, res);
    cv::Mat response = dataset->getResponses();
    CV_Assert(res.size() == nsamples);

    for (int i=0; i<res.size(); i++)
    {
        if (res[i] != response.at<float>(i))
        {
            std::cout << "predict error! " << "sample number is :" << i << std::endl;
        }
    }

}