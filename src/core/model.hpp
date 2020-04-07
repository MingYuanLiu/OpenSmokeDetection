// project: adaboost cpp code
// data: 2020.03
// author: MingYuan Liu
#ifndef MODEL_HPP
#define MODEL_HPP

#include "cv_dtree_header.hpp"
#include "features.hpp"
#include "statistical_features.hpp"


#include <opencv2/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <fstream>
#include <iostream>
#include <time.h>
#include <iomanip>


namespace smoke_adaboost
{

// 打印出程序运行进度
inline static void printProcess(float value, const std::string& infor)
{
    if (!(value >= 0 && value <= 1))
        CV_Error(cv::Error::Code::StsOutOfRange, "Input Value must be range from 0 to 1. ");

    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2);
    std::string tag = "[ " + infor + ":]" + std::string((int)value * 10, '*') + "[";
    std::cout << std::flush << '\r' << tag << value * 100 << "%]";
}

// 对cv::TrainData的拓展，实现原类中没有实现的虚方法
class smokeCascadeDetectorTrainData : public TrainData
{
    public:
        smokeCascadeDetectorTrainData() {}
        virtual int getLayout() const {return 0;}
        virtual int getNTrainSamples() const {return 0;}
        virtual int getNTestSamples() const {return 0;} 
        virtual int getCatCount(int varIdx) const {return 0;} // 在离散型分类决策问题中某个特征的不同取值的个数
        virtual Mat getVarType() const {return Mat();} // 获取特征中不同的取值类型
        virtual int getResponseType() const {return 0;} // 根据responses的数值类型判断是分类问题还是回归问题
};

// 数据生成器，根据标注文件读取图片数据，计算特征
// 有两种模式可供选择： 单线程模式和多线程模式
class smokeFeatureGenerator
{
    public:
        smokeFeatureGenerator(bool _multiThread=false, int _threadNums=1);
        ~smokeFeatureGenerator(){}
        void runGenerator(); 
        void runGeneratorMultiThread(){}
        bool isFinished() {return finished;}
        Ptr<TrainData> getTrainData();
        void loadAnnotationFromFile(const std::string& _annotationFilePath);
        void writeFeaturesToFile(const std::string& _featrueFilePath);
        void readFeaturesFromFile(const std::string& _filePath);

        struct FeatureParam
        {
            FeatureParam() 
            {cols=0; rows=0; ddepth=0; windowSize=8; stride=2; padding=false; 
            tiledCutNums=30; ringedCutNums=5;}
            // feature map 相关
            uint16_t cols;
            uint16_t rows;
            uint8_t ddepth;
            uint8_t windowSize;
            uint8_t stride;
            bool padding;
            // 统计特征相关
            uint16_t tiledCutNums;
            uint16_t ringedCutNums;
        };
        void setFeatureParam(FeatureParam& _fmParam) {fmParam = _fmParam;};

    private:        
        FeatureParam fmParam;
        // 样本及特征相关
        struct sample
        {
            std::string filePath;
            uint8_t label;
        };
        vector<sample> samples;  // 从文件中读取到的样本
        vector<vector<float> > rawFeatures; // 特征向量
        vector<float> labels;
        vector<int> sidx, vidx;
        std::string annotationFilePath;  // 标记文件所在路径

        // 多线程相关
        bool finished;       
        // std::atomic<size_t> finishedCount;
        bool multithread;
        int threadNums;

};

// 带有cascade特性的boost决策树
//
class smokeCascadeBoostDescisionTree : public DTreesImpl
{
    public:
        smokeCascadeBoostDescisionTree() 
        { 
            boostParam.boostType = Boost::DISCRETE;
            boostParam.weakCount = 1500;
            boostParam.weightTrimRate = 0.95;
            miniHitRate = 0.995;
            maxFalseAlarmRate = 0.5;
            _isClassifier = true;
        }
        smokeCascadeBoostDescisionTree(int boostType, int weakcnts, 
                                        float weightTrimRate, float _miniHitRate, 
                                        float _maxFalseAlarmRate, int _stage, 
                                        const TreeParams &_tp) 
        { 

            boostParam.boostType = boostType;
            boostParam.weakCount = weakcnts;
            boostParam.weightTrimRate = weightTrimRate;
            miniHitRate = _miniHitRate;
            maxFalseAlarmRate = _maxFalseAlarmRate;
            if (boostType == Boost::DISCRETE)
                _isClassifier = true;
            stage = _stage;
            DTreesImpl::setDParams(_tp);
        }        
        virtual ~smokeCascadeBoostDescisionTree() { }
        virtual void beforeTraining(const Ptr<TrainData>& trainData, int flags);
        virtual bool trainBoost(const Ptr<TrainData>& trainData, const Ptr<TrainData>& evalData, int flags);
        virtual float predictTree(const Range& range, const Mat& sample, int flags0);
        inline void setClassifier(bool set) {_isClassifier = set;}
        virtual void calcValue(int nidx, const std::vector<int>& _sidx) override;
        inline void setminiHitRate(float _miniHitRate) {miniHitRate = _miniHitRate;}
        inline void setmaxFalseAlarmRate(float _maxFalseAlarmRate) {maxFalseAlarmRate = _maxFalseAlarmRate;}
        float predictSamples(Mat& samples, vector<float>& result);
        virtual void writeTrainingParams( FileStorage& fs ) const override;
        virtual void write(FileStorage& fs) const override;

        virtual void readParams(const FileNode& fn) override;
        virtual void read(const FileNode& fn) override; 

    private:
        static inline double log_ratio( double val )
        {
            const double eps = 1e-5;
            val = std::max( val, eps );
            val = std::min( val, 1. - eps );
            return log( val/(1. - val) );
        }
        void updateWeightsAndTrim(int treeidx, vector<int>& sidx);
        void normalizeWeights();
        void scaleTree(int root, float scale);
        bool reachDesireRate(int weak_count, const Ptr<TrainData>& _evalData);
        bool reachDesireRate2(int weak_count); // 以训练集作为测试数据
        
        BoostTreeParams boostParam;
        float miniHitRate, maxFalseAlarmRate;
        int stage; // 当前级数
        // 验证数据集
        Ptr<TrainData> evalDataset;
};
 
// 级联分类器
class smokeCascadeDetector
{
    public:
        void setCrossValidationFold(int _cvFold);
        // void setTrainData(Ptr<TrainData> _cascadeTrainData);
        void train(Ptr<TrainData> _cascadeTrainData, Ptr<TrainData> cascadeEvalData);
        void evaluate(Ptr<TrainData> _evalData);
        void saveModel(const std::string& modelFileName);
        void loadModel(const std::string& modelFileName);
        void printResult();
        // 根据前一级强分类器的结果更新训练数据集
        Ptr<TrainData> updateTrainData(Ptr<TrainData>  _stageTrainData,
                                         Ptr<smokeCascadeBoostDescisionTree> _stageClassfier); 
        float predict(const Mat& _samples, vector<float>& _res);
        inline bool empty() {return stageStrongClassifiers.size() == 0;}

        struct DetectorParam
        {
            DetectorParam()
            {
                singleLayerMaxFPR = 0.5;
                singleLayerMinTPR = 0.99;
                targetFPR = pow(singleLayerMaxFPR, stageNums);
                stageNums = 10;
                useCrossValidate = false;
                cvFold = 0;
            }
            float singleLayerMaxFPR; // 每一层adaboost的最大误识率
            float singleLayerMinTPR; // 每一层adaboost的检测率
            float targetFPR; // 识别器总的误识率
            int stageNums;
            bool useCrossValidate;
            int cvFold;

            BoostTreeParams boostParam;
            TreeParams DTreeparam;
            smokeFeatureGenerator::FeatureParam featureParam;
            bool isClassfier;
        };
        smokeCascadeDetector();
        smokeCascadeDetector(const DetectorParam& _detParams);
        ~smokeCascadeDetector() { }
        void setDetParams(const DetectorParam& _detParams) { detParams = _detParams; }
        void readDetParams(DetectorParam& _detParams) { _detParams = detParams; }
        double readFPR() { return tmpFalsePositiveRate; }

    private:
        void writeParams(FileStorage& fs) const;
        void readParams(const FileStorage& fs);
        inline void clear()
        {
            newSampleIdx.clear();
            predictPositiveLabels.clear();
        }
        enum {CAT_NEG=0, CAT_POS=1, ORD_NEG=-1, ORD_POS=1};
        DetectorParam detParams;
        double tmpFalsePositiveRate; // 训练时，用于记录每次训练后的误识率
        vector<Ptr<smokeCascadeBoostDescisionTree> > stageStrongClassifiers;  // 强分类器

        // 更新数据相关
        vector<int> newSampleIdx; // 新数据集的样本索引
        vector<int> predictPositiveLabels;        
};
} // namespace smoke_adaboost


#endif