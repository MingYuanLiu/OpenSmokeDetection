#include "main.hpp"

using smoke_adaboost::Detector;
using smoke_adaboost::smokeCascadeDetector;
using std::string;

// work path
// const std::string WORK_HOME = "/Users/mac/Desktop/摄像头烟雾检测/代码/cpp_adaboost/";

// function declaration
void trainModel(const string &datasetAnnotationFile, const string &saveFeaturesPath,
                uint8_t stride, uint8_t windowSize, uint16_t tiledCutNums, uint16_t ringedCutNums,
                float dataSetSplitRatio, const smokeCascadeDetector::DetectorParam &dp);
void testResult(smoke_adaboost::smokeCascadeDetector &cls, const cv::Ptr<cv::ml::TrainData> &dataset);
void detetction(const Detector::detectorParams &_param);  

int main(int argc, char *argv[])
{
    if (argc <= 1)
    {
        std::cout << "Input valid parametor; \n";
        std::cout << "1. train: train model using dataset. \n";
        std::cout << "2. detetcion: detetction image or video using trained model. \n";
        return 0;
    }
    if (string(argv[1]) == "train")
    {
        // 训练参数
        string annotationFiles = "data.txt";  // 标注文件
        string saveFeaturesPath = "feature_big_new2.yaml"; // 特征保存的文件
        uint8_t stride = 2; // 特征计算时的步长
        uint8_t windowSize = 8; // 特征计算时的窗口大小
        float ratio = 0.8;  // 训练集与验证集的分割比列
        uint16_t tiledCutNums = 30; // 平铺分割次数
        uint16_t ringedCutNums = 5; // 环绕分割次数
        smoke_adaboost::smokeCascadeDetector::DetectorParam dp; // 分类器参数
        dp.singleLayerMaxFPR = 0.2;  // cascade中每一层分类器的最大误判率
        dp.singleLayerMinTPR = 0.95; // cascade中每一层分类器的最小识别率
        dp.stageNums = 20; // cascade的最大层数
        dp.targetFPR = 0.001; // 整体分类器的总期望误判率
        dp.boostParam.boostType = cv::ml::Boost::DISCRETE; // boost类型：可为 DISCRETE、REAL、LOGITIC等
        dp.boostParam.weakCount = 1500; // 弱分类器的数量
        dp.boostParam.weightTrimRate = 0.99; // 随机裁减掉一部分无用样本
        dp.DTreeparam.setMaxDepth(2); // 决策树的最大深度
        dp.featureParam.windowSize = windowSize;
        dp.featureParam.stride = stride;
        dp.featureParam.tiledCutNums = tiledCutNums;
        dp.featureParam.ringedCutNums = ringedCutNums;

        // 开始训练
        trainModel(annotationFiles, saveFeaturesPath, stride, windowSize,
                   tiledCutNums, ringedCutNums, ratio, dp);
    }
    else if (string(argv[1]) == "detection")
    {

        std::string saveModelPath = "model-0.05373.yaml"; // 模型保存路径
        std::string videoPath = "21.mpg"; // 视频文件
        std::string imagePath = "51.jpg"; // 图像文件
        smoke_adaboost::Detector::detectorParams param; // 检测参数
        param.videoPath = videoPath;
        param.imagePath = imagePath;
        param.modelPath = saveModelPath; 
        vector<uint16_t> w = {50}; // 检测窗口大小，默认为50pixel
        param.detWindowSize = w; 
        param.threadNums = 4; // 线程数，默认为4；可调，但建议为2的倍数
        param.detectorModal = smoke_adaboost::Detector::VIDEO; // 选择检测模式，图片(IMAGE) or 视频(VIDEO)
        detetction(param);
    }
    else
    {
        std::cout << "Invalid parametors. " << std::endl;
        return 0;
    }

    return 0;
}

// 训练模型
// Args:
//      datasetAnnotationFile: 数据集的标注信息，包括图片的路径和图片的标签
//      saveFeaturesPath: 如果不存在该文件，那么计算特征并保存到该文件中； 如果已经存在，则直接从该文件中读取特征
//      stride: 计算特征的窗口步长
//      windowSize: 计算特征的窗口大小
//      dataSetSplitRatio: 训练验证样本集的分割比例
//      dp: reference of DetectorParam 分类器参数
//
void trainModel(const string &datasetAnnotationFile,
                const string &saveFeaturesPath,
                uint8_t stride, uint8_t windowSize,
                uint16_t tiledCutNums, uint16_t ringedCutNums,
                float dataSetSplitRatio,
                const smokeCascadeDetector::DetectorParam &dp)
{
    // 流程:
    // 读取图片数据并计算特征
    // 设置参数
    // 开始训练
    // 验证

    smoke_adaboost::smokeFeatureGenerator dataset;
    smoke_adaboost::smokeFeatureGenerator::FeatureParam fp;
    fp.stride = stride;
    fp.windowSize = windowSize;
    fp.tiledCutNums = tiledCutNums;
    fp.ringedCutNums = ringedCutNums;

    dataset.setFeatureParam(fp);
    struct stat buff; // check feature file is or is not existing. 
    if (stat(saveFeaturesPath.c_str(), &buff) != 0)
    {
        std::cout << "Features file does not exist. createing... \n";
        dataset.loadAnnotationFromFile(datasetAnnotationFile);
        dataset.runGenerator();
        dataset.writeFeaturesToFile(saveFeaturesPath);
    }
    else
    {
        dataset.readFeaturesFromFile(saveFeaturesPath);
    }
    cv::Ptr<cv::ml::TrainData> Data = dataset.getTrainData();

    Data->setTrainTestSplitRatio(dataSetSplitRatio, true);

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
                                                                       testResponse, VarIdx, testSIdx);

    smoke_adaboost::smokeCascadeDetector classifier(dp);

    classifier.train(trainDataset, testDataset);

    classifier.evaluate(testDataset);
    double fpr = classifier.readFPR();
    char str[10];
    gcvt(fpr, 4, str);
    std::string saveModelPath = "model-" + std::string(str) + ".yaml";
    std::cout << "--- \n Model save to " << saveModelPath << std::endl;
    classifier.saveModel(saveModelPath);
    testResult(classifier, Data);
}

// 使用训练好的模型进行检测
//
void detetction(const Detector::detectorParams &_param)
{
    Detector detector(_param);
    detector.run();
}

// 测试训练模型， 输出预测错误的样本信息
void testResult(smoke_adaboost::smokeCascadeDetector &cls, const cv::Ptr<cv::ml::TrainData> &dataset)
{
    int nsamples = dataset->getNSamples();
    CV_Assert(nsamples > 0);
    cv::Mat samples = dataset->getSamples();
    std::vector<float> res;
    cls.predict(samples, res);
    cv::Mat response = dataset->getResponses();
    CV_Assert(res.size() == nsamples);

    for (int i = 0; i < res.size(); i++)
    {
        if (res[i] != response.at<float>(i))
        {
            std::cout << "predict error! "
                      << "sample number is :" << i << std::endl;
        }
    }
}