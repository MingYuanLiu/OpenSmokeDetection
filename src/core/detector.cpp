#include "detector.hpp"

namespace smoke_adaboost
{

// 读取视频并进行烟雾检测
// Args:
// _videoPath: 视频文件路径
// _modelPath: 模型文件路径
// 
// Notes:
//  
void Detector::detectVideo(const string &_videoPath, const string &_modelPath)
{
    if (_videoPath.empty())
    {
        std::cout << "Video Path is empty, give detector a video path. " << std::endl;
        return;
    }
    VideoCapture cap;
    cap.open(_videoPath);
    if (!cap.isOpened())
        CV_Error(Error::StsObjectNotFound, "Can not open the video file. ");

    Mat frame;
    // 读取模型
    smokeCascadeDetector cls;
    cls.loadModel(_modelPath);
    if (cls.empty())
        CV_Error(Error::StsNullPtr, "Model is empty, please check.");

    smokeCascadeDetector::DetectorParam classifierParam;
    cls.readDetParams(classifierParam);
    for (;;)
    {
        cap >> frame;
        if (frame.empty())
            break;
        Mat grayFrame;
        if (frame.channels() != 1)
            cvtColor(frame, grayFrame, CV_BGR2GRAY);
        else
            grayFrame = frame.clone();

        uint16_t originDetWindowSize = param.detWindowSize[0]; // 原图上的检测窗口大小，这里只采用了一种窗口大小
        uint16_t originDetStride = originDetWindowSize / 2;    // 检测窗口在特征图像上的步长，默认为检测窗口的一半

        uint8_t strideOnFeatureMap = classifierParam.featureParam.stride; // 计算特征图像时的滑动窗口步长
        uint8_t windowSizeOnFeatureMap = classifierParam.featureParam.windowSize; // 计算特征图像时的滑动窗口大小
        uint8_t tiledCutNums = classifierParam.featureParam.tiledCutNums; // 计算统计特征时的平铺分割次数
        uint8_t ringedCutNums = classifierParam.featureParam.ringedCutNums; // 环绕切割次数

        uint16_t detWindowSizeOnFeatureMap = (originDetWindowSize - windowSizeOnFeatureMap) / strideOnFeatureMap; // 将原图上的检测窗口映射到特征图像上

        uint16_t featureMapCols = 0, featureMapRows = 0;
        uint8_t featureMapDepth = 0;

        vector<float> featureMap;

        int threadNums = param.threadNums;
        assert(threadNums > 0);
        
            // debug
            // double t0 = cv::getTickCount();

        // multi thread generate map
        generatFeatureMapMultiThread(grayFrame, featureMap, featureMapCols, featureMapRows, featureMapDepth, strideOnFeatureMap, windowSizeOnFeatureMap, 8);
        assert(!featureMap.empty());
        uint16_t detSrtideOnFeatureMap = originDetStride / 2;
        uint8_t rowsNum = (grayFrame.rows - originDetWindowSize) / originDetStride;
        uint8_t colsNum = (grayFrame.cols - originDetWindowSize) / originDetStride;

        // multi thread predict
        int deltaRows = (rowsNum + 1) / threadNums;
        if (deltaRows <= 0)
        {
            std::cout << "---- \n Thread numbers must be less than rows of detection sliding window. " << std::endl;
            std::cout << "Force threadNums to be rows of detection window. " << std::endl;
            deltaRows = 1;
        }
        std::thread predictThreads[threadNums];
        vector<predictRes> tmpRes; // 线程计算后保存的临时结果
        std::mutex result_lock;  // 对全局变量tmpRes读写时的互斥量

        for (int t = 0; t < threadNums; t++)
        {
            int rowStart = t * deltaRows;
            int rowEnd = (t + 1) * deltaRows;
            // 使用匿名函数创建线程，方便利用上下文环境
            predictThreads[t] = std::thread([=, &featureMap, &cls, &tmpRes, &result_lock] {
                size_t total = detWindowSizeOnFeatureMap * detWindowSizeOnFeatureMap * featureMapDepth;
                vector<float> tmpStatisticalFeature;
                tmpStatisticalFeature.reserve((tiledCutNums + ringedCutNums) * featureMapDepth * 4);
                // 根据每个线程所分配的行数来调整每行之间的跨度
                int passedRows = 0;
                if (deltaRows <= 1)
                    passedRows = (t % 2 == 0) ? t * 12.5 : t * 12;
                else
                    passedRows = t * 25;
                
                for (uint8_t i = rowStart; i < rowEnd; i++)
                {
                    float tmpRow[featureMapCols * detWindowSizeOnFeatureMap * featureMapDepth] = {0};
                    size_t start = featureMapCols * passedRows * featureMapDepth;
                    passedRows += ((i % 2) != 0 ? detSrtideOnFeatureMap : detSrtideOnFeatureMap + 1);
                    size_t end = start + featureMapCols * detWindowSizeOnFeatureMap * featureMapDepth;
                    if (end >= featureMap.size())
                        break;

                    // 读取一整行窗口存入到tmp变量中，提高缓存利用率
                    std::copy(featureMap.begin() + start, featureMap.begin() + end, tmpRow);
                    uint16_t colsStep = 0;
                    for (uint8_t j = 0; j < colsNum + 1; j++)
                    {
                        // 读取一个窗口
                        float tmpWindow[detWindowSizeOnFeatureMap * detWindowSizeOnFeatureMap * featureMapDepth] = {0};
                        for (int m = 0; m < detWindowSizeOnFeatureMap; m++)
                        {
                            for (int n = 0; n < detWindowSizeOnFeatureMap; n++)
                            {
                                for (int k = 0; k < featureMapDepth; k++)
                                {
                                    float val = tmpRow[k + featureMapDepth * (n + colsStep + m * featureMapCols)];
                                    tmpWindow[k + featureMapDepth * (n + m * detWindowSizeOnFeatureMap)] = val;
                                }
                            }
                        }
                        colsStep += ((j % 2) == 0 ? detSrtideOnFeatureMap : detSrtideOnFeatureMap + 1);
                        std::vector<float> featureWindow(tmpWindow, tmpWindow + total);
                        BlocksAndStatisticalFeatures sf(featureWindow,
                                                        detWindowSizeOnFeatureMap,
                                                        detWindowSizeOnFeatureMap, featureMapDepth,
                                                        tiledCutNums, ringedCutNums);
                        sf.getStatisticalFeatures(tmpStatisticalFeature);
                        vector<float> resv;
                        cls.predict(Mat(1, tmpStatisticalFeature.size(), CV_32F, &tmpStatisticalFeature[0]), resv);
                        if (resv[0] == 1)
                        {
                            predictRes res;
                            res.label = 1;
                            res.x = j * 25;
                            res.y = i * 25;
                            res.w = res.h = 50;
                            result_lock.lock();
                            tmpRes.push_back(res);
                            result_lock.unlock();
                        }
                        tmpStatisticalFeature.clear();
                    }
                }
            });
        }
                //debug
                // double t2 = ((double)(cv::getTickCount() - t0) / cv::getTickFrequency()) * 1000000.0;
                // if (t2 > 50000.0)
                //     std::cout << "origin 's time" << t2 << "us" << std::endl;
        for (auto &t : predictThreads)
        {
            t.join();
        }
        result = tmpRes;
        drawResult(frame);
    }
}

// 视频检测的朴素版设计
// Args:
//  _videoPath: 视频文件路径
//  _modelPath: 模型文件路径
//
void Detector::detectVideoRaw(const string &_videoPath, const string &_modelPath)
{
    if (_videoPath.empty())
    {
        std::cout << "Video Path is empty, give detector a video path. " << std::endl;
        return;
    }
    VideoCapture cap;
    cap.open(_videoPath);
    if (!cap.isOpened())
    {
        CV_Error(Error::StsObjectNotFound, "Can not open the video file. ");
    }
    // 读取模型
    smokeCascadeDetector cls;
    cls.loadModel(_modelPath);
    if (cls.empty())
        CV_Error(Error::StsNullPtr, "Model is empty, please check.");

    smokeCascadeDetector::DetectorParam classifierParam;
    cls.readDetParams(classifierParam);
    for (;;)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        Mat grayFrame;
        if (frame.channels() != 1)
            cvtColor(frame, grayFrame, CV_BGR2GRAY);
        else
            grayFrame = frame.clone();
        resize(grayFrame, grayFrame, Size(330, 250));
        /*
        * use original image to predict
        */
        vector<float> tmpFeatureData;
        vector<float> tmpStatisticalFeature;
        for (int i = 0; i < param.detWindowSize.size(); i++)
        {
            int windowSize = param.detWindowSize[i];
            for (int x = 0; x < frame.cols - windowSize; x += windowSize / 2)
            {
                for (int y = 0; y < frame.rows - windowSize; y += windowSize / 2)
                {
                    uint16_t out_cols = 0, out_rows = 0;
                    uint8_t out_ddepth = 0;
                    generateFeatureMap(grayFrame(Range(y, y + windowSize), Range(x, x + windowSize)),
                                       tmpFeatureData, out_cols, out_rows, out_ddepth,
                                       classifierParam.featureParam.stride, classifierParam.featureParam.windowSize);
                    BlocksAndStatisticalFeatures sf(tmpFeatureData,
                                                    out_cols, out_rows, out_ddepth,
                                                    classifierParam.featureParam.tiledCutNums,
                                                    classifierParam.featureParam.ringedCutNums);
                    sf.getStatisticalFeatures(tmpStatisticalFeature);
                    vector<float> resv;
                    cls.predict(Mat(1, tmpStatisticalFeature.size(), CV_32F, &tmpStatisticalFeature[0]), resv);
                    if (resv[0] == 1.0)
                    {
                        predictRes res;
                        res.label = 1;
                        res.x = x;
                        res.y = y;
                        res.w = res.h = windowSize;
                        result.push_back(res);
                    }
                    tmpStatisticalFeature.clear();
                    tmpFeatureData.clear();
                    resv.clear();
                }
            }
        }
        drawResult(frame);
    }
}

// 朴素版检测图片中的烟雾
// Args:
//  _imagePath: 图片路径
//  _modelPath: 模型文件路径
//
void Detector::detectImageRaw(const string &_imagePath, const string &_modelPath)
{
    CV_Assert(!_imagePath.empty());
    Mat image = imread(_imagePath);
    if (image.empty())
        CV_Error(Error::Code::StsNullPtr, "image is empty");

    Mat grayImage;
    if (image.channels() != 1)
        cvtColor(image, grayImage, CV_BGR2GRAY);

    // resize(grayImage, grayImage, Size(330,250));

    if (param.detWindowSize.empty())
        param.detWindowSize.push_back(100);

    // load model
    smokeCascadeDetector cls;
    cls.loadModel(_modelPath);
    smokeCascadeDetector::DetectorParam classifierParam;
    cls.readDetParams(classifierParam);
    /*
    * use original image to predict
    */
    vector<float> tmpFeatureData;
    vector<float> tmpStatisticalFeature;

    for (int i = 0; i < param.detWindowSize.size(); i++)
    {
        int windowSize = param.detWindowSize[i];
        for (int x = 0; x < image.cols - windowSize; x += windowSize / 2)
        {
            for (int y = 0; y < image.rows - windowSize; y += windowSize / 2)
            {
                uint16_t out_cols = 0, out_rows = 0;
                uint8_t out_ddepth = 0;
                generateFeatureMap(grayImage(Range(y, y + windowSize), Range(x, x + windowSize)),
                                   tmpFeatureData, out_cols, out_rows, out_ddepth,
                                   classifierParam.featureParam.stride, classifierParam.featureParam.windowSize);
                BlocksAndStatisticalFeatures sf(tmpFeatureData,
                                                out_cols, out_rows, out_ddepth,
                                                classifierParam.featureParam.tiledCutNums,
                                                classifierParam.featureParam.ringedCutNums);
                sf.getStatisticalFeatures(tmpStatisticalFeature);
                vector<float> resv;
                cls.predict(Mat(1, tmpStatisticalFeature.size(), CV_32F, &tmpStatisticalFeature[0]), resv);
                if (resv[0] == 1)
                {
                    predictRes res;
                    res.label = 1;
                    res.x = x;
                    res.y = y;
                    res.w = res.h = windowSize;
                    result.push_back(res);
                }
                tmpStatisticalFeature.clear();
                tmpFeatureData.clear();
                resv.clear();
            }
        }
    }

    drawResult(image);
}

// 检测烟雾图片
// Args:
//  _imagePath: 图片文件路径
//  _modelPath: 模型文件路径
//
void Detector::detectImage(const string &_imagePath, const string &_modelPath)
{
    CV_Assert(!_imagePath.empty());
    Mat image = imread(_imagePath);
    if (image.empty())
        CV_Error(Error::Code::StsNullPtr, "image is empty");

    Mat grayImage;
    if (image.channels() != 1)
        cvtColor(image, grayImage, CV_BGR2GRAY);

    if (param.detWindowSize.empty())
        param.detWindowSize.push_back(50);

    // load model
    smokeCascadeDetector cls;
    cls.loadModel(_modelPath);
    smokeCascadeDetector::DetectorParam classifierParam;
    cls.readDetParams(classifierParam);

    uint16_t originDetWindowSize = param.detWindowSize[0]; // 原图上的窗口大小
    uint16_t originDetStride = originDetWindowSize / 2;    // 检测窗口在特征图像上的步长，默认为检测窗口的一半

    uint8_t strideOnFeatureMap = classifierParam.featureParam.stride; // 计算特征时的相关参数
    uint8_t windowSizeOnFeatureMap = classifierParam.featureParam.windowSize;
    uint16_t tiledCutNums = classifierParam.featureParam.tiledCutNums;
    uint16_t ringedCutNums = classifierParam.featureParam.ringedCutNums;

    uint16_t detWindowSizeOnFeatureMap = (originDetWindowSize - windowSizeOnFeatureMap) / strideOnFeatureMap; // 将原图上的检测窗口映射到特征图像上

    uint16_t featureMapCols = 0, featureMapRows = 0;
    uint8_t featureMapDepth = 0;

    vector<float> featureMap;

    int threadNums = param.threadNums;
    assert(threadNums > 0);
    // debug
    double t0 = cv::getTickCount();

    generatFeatureMapMultiThread(grayImage, featureMap, featureMapCols, featureMapRows, featureMapDepth,
                                 strideOnFeatureMap, windowSizeOnFeatureMap, threadNums);

    assert(!featureMap.empty());
    uint16_t detSrtideOnFeatureMap = originDetStride / 2;
    int rowsNum = (image.rows - originDetWindowSize) / originDetStride;
    int colsNum = (image.cols - originDetWindowSize) / originDetStride;

    // multi thread predict
    int deltaRows = (rowsNum + 1) / threadNums;
    std::thread predictThreads[threadNums];
    vector<predictRes> tmpRes;
    std::mutex result_lock;

    for (int t = 0; t < threadNums; t++)
    {
        int rowStart = t * deltaRows;
        int rowEnd = (t + 1) * deltaRows;
        predictThreads[t] = std::thread([=, &featureMap, &cls, &tmpRes, &result_lock] {
            size_t total = detWindowSizeOnFeatureMap * detWindowSizeOnFeatureMap * featureMapDepth;
            vector<float> tmpStatisticalFeature;
            tmpStatisticalFeature.reserve((tiledCutNums + ringedCutNums) * featureMapDepth * 4);
            // 根据每个线程所分配的行数来调整每行之间的跨度
            int passedRows = 0;
            if (deltaRows <= 1)
                passedRows = (t % 2 == 0) ? t * 12.5 : t * 12;
            else
                passedRows = t * 25;

            for (uint8_t i = rowStart; i < rowEnd; i++)
            {
                float tmpRow[featureMapCols * detWindowSizeOnFeatureMap * featureMapDepth] = {0};
                size_t start = featureMapCols * passedRows * featureMapDepth;
                passedRows += ((i % 2) != 0 ? detSrtideOnFeatureMap : detSrtideOnFeatureMap + 1);
                size_t end = start + featureMapCols * detWindowSizeOnFeatureMap * featureMapDepth;
                if (end >= featureMap.size())
                    break;

                // 读取一整行窗口存入到tmp变量中，提高缓存利用率
                std::copy(featureMap.begin() + start, featureMap.begin() + end, tmpRow);
                uint16_t colsStep = 0;
                for (uint8_t j = 0; j < colsNum + 1; j++)
                {
                    // 读取一个窗口
                    float tmpWindow[detWindowSizeOnFeatureMap * detWindowSizeOnFeatureMap * featureMapDepth] = {0};
                    for (int m = 0; m < detWindowSizeOnFeatureMap; m++)
                    {
                        for (int n = 0; n < detWindowSizeOnFeatureMap; n++)
                        {
                            for (int k = 0; k < featureMapDepth; k++)
                            {
                                float val = tmpRow[k + featureMapDepth * (n + colsStep + m * featureMapCols)];
                                tmpWindow[k + featureMapDepth * (n + m * detWindowSizeOnFeatureMap)] = val;
                            }
                        }
                    }
                    colsStep += ((j % 2) == 0 ? detSrtideOnFeatureMap : detSrtideOnFeatureMap + 1);
                    std::vector<float> featureWindow(tmpWindow, tmpWindow + total);
                    BlocksAndStatisticalFeatures sf(featureWindow,
                                                    detWindowSizeOnFeatureMap,
                                                    detWindowSizeOnFeatureMap, featureMapDepth,
                                                    tiledCutNums, ringedCutNums);
                    sf.getStatisticalFeatures(tmpStatisticalFeature);
                    vector<float> resv;
                    cls.predict(Mat(1, tmpStatisticalFeature.size(), CV_32F, &tmpStatisticalFeature[0]), resv);
                    if (resv[0] == 1)
                    {
                        predictRes res;
                        res.label = 1;
                        res.x = j * 25;
                        res.y = i * 25;
                        res.w = res.h = 50;
                        result_lock.lock();
                        tmpRes.push_back(res);
                        result_lock.unlock();
                    }
                    tmpStatisticalFeature.clear();
                }
            }
        });
    }

    //debug
    double t2 = ((double)(cv::getTickCount() - t0) / cv::getTickFrequency()) * 1000000.0;
    std::cout << "origin 's time" << t2 << "us" << std::endl;

    for (auto &t : predictThreads)
    {
        t.join();
    }
    result = tmpRes;

    drawResult(image);
}

// 画检测框并显示图片
// Args:
//  frame: 图片帧
//
void Detector::drawResult(Mat &frame)
{
    const string windowName = "detection result";
    namedWindow(windowName, WINDOW_AUTOSIZE);
    if (result.empty())
    {
        // std::cout << "no predict. " << std::endl;
        imshow(windowName, frame);
        if (param.detectorModal == VIDEO)
            waitKey(2);
        else if (param.detectorModal == IMAGE)
            waitKey(0);
        
        return;
    }
    for (size_t i = 0; i < result.size(); i++)
    {
        Point p1, p2;
        p1.x = result[i].x;
        p1.y = result[i].y;
        p2.x = result[i].w + result[i].x;
        p2.y = result[i].h + result[i].y;
        rectangle(frame, Rect(p1, p2), Scalar(0, 255, 0), 1);
        // std::cout << "point1:" << p1 << "point2:" << p2 << std::endl;
    }
    result.clear();
    imshow(windowName, frame);
    if (param.detectorModal == VIDEO)
        waitKey(2);
    else if (param.detectorModal == IMAGE)
        waitKey(0);
}

// 根据参数选择检测视频模式和检测图像模式
// 
void Detector::run()
{
    assert(!param.modelPath.empty());
    if (param.detectorModal == VIDEO)
    {
        assert(!param.videoPath.empty());
        detectVideo(param.videoPath, param.modelPath);
    }
    else if (param.detectorModal == IMAGE)
    {
        assert(!param.imagePath.empty());
        detectImage(param.imagePath, param.modelPath);
    }
}

Detector::~Detector()
{
}

} // namespace smoke_adaboost