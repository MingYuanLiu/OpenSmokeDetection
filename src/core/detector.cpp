#include "detector.hpp"

namespace smoke_adaboost
{

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
    {
        CV_Error(Error::StsObjectNotFound, "Can not open the video file. ");
    }
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

        std::vector<float> featureImage;
        uint16_t out_cols = 0, out_rows = 0;
        uint8_t out_ddepth = 0;
        generateFeatureMap(grayFrame, featureImage, out_cols, out_rows, out_ddepth);
        // the feature image shape is out_cols x out_rows x out_ddepth
        CV_Assert(!featureImage.empty());
        CV_Assert(out_cols != 0 && out_rows != 0 && out_ddepth != 0);
        int i;
        if (param.detWindowSize.empty())
        {
            param.detWindowSize.push_back(100);
        }
        // produce the features of the frame
        for (i = 0; i < param.detWindowSize.size(); i++)
        {
            float *data = &featureImage[0];
            int windowSize = param.detWindowSize[i] -
                             classifierParam.featureParam.windowSize / classifierParam.featureParam.windowSize;
            // int numCols = out_cols / (windowSize / 2);
            // int numRows = out_rows / (windowSize / 2);
            vector<float> tmpWindowFeature; // 窗口的特征图像
            tmpWindowFeature.reserve(windowSize * windowSize * out_ddepth);
            for (int x = 0; x < out_cols - windowSize; x += (windowSize / 2))
            {
                for (int y = 0; y < out_rows - windowSize; y += (windowSize / 2))
                {
                    // 读取一个窗口的特征图像
                    for (int k = y; k < (y + windowSize); k++)
                    {
                        for (int m = x; m < (x + windowSize); m++)
                        {
                            for (int n = 0; n < out_ddepth; n++)
                                tmpWindowFeature.push_back(data[n + (k * out_cols + m) * out_ddepth]);
                        }
                    }

                    // 获取static features
                    Ptr<BlocksAndStatisticalFeatures> sf = makePtr<BlocksAndStatisticalFeatures>(tmpWindowFeature,
                                                                                                 windowSize, windowSize, out_ddepth,
                                                                                                 classifierParam.featureParam.tiledCutNums,
                                                                                                 classifierParam.featureParam.ringedCutNums);
                    vector<float> sfData;
                    sfData.reserve(out_ddepth *
                                   (classifierParam.featureParam.tiledCutNums +
                                    classifierParam.featureParam.ringedCutNums + 5) *
                                   4);
                    sf->getStatisticalFeatures(sfData);
                    int featureSize = sfData.size();
                    std::cout << "debug" << featureSize;
                    vector<float> res;
                    cls.predict(Mat(1, featureSize, CV_32F, &sfData[0]), res);
                    if (res[0] == 1)
                    {
                        predictRes res;
                        res.label = 1;
                        res.x = x * classifierParam.featureParam.stride + classifierParam.featureParam.windowSize;
                        res.y = y * classifierParam.featureParam.stride + classifierParam.featureParam.windowSize;
                        res.w = res.h = param.detWindowSize[i];
                        result.push_back(res);
                    }
                    tmpWindowFeature.clear();
                    sfData.clear();
                    res.clear();
                }
            }
        }

        drawResult(frame);
    }
}

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
                    // printf("predi:%f\n", resv[0]);
                    tmpStatisticalFeature.clear();
                    tmpFeatureData.clear();
                    resv.clear();
                }
            }
        }
        // imshow("test",frame);
        drawResult(frame);
    }
}

void Detector::detectImage(const string &_imagePath, const string &_modelPath)
{
    CV_Assert(!_imagePath.empty());
    Mat image = imread(_imagePath);
    if (image.empty())
        CV_Error(Error::Code::StsNullPtr, "image is empty");

    Mat grayImage;
    if (image.channels() != 1)
        cvtColor(image, grayImage, CV_BGR2GRAY);

    // resize(grayImage, grayImage, Size(320,240));

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

void Detector::drawResult(Mat &frame)
{
    const string windowName = "detection result";
    namedWindow(windowName, WINDOW_AUTOSIZE);
    if (result.empty())
    {
        std::cout << "no predict. " << std::endl;
        imshow(windowName, frame);
        waitKey(10);
        return;
    }
    for (int i = 0; i < result.size(); i++)
    {
        Point p1, p2;
        p1.x = result[i].x;
        p1.y = result[i].y;
        p2.x = result[i].w + result[i].x;
        p2.y = result[i].h + result[i].y;
        rectangle(frame, Rect(p1, p2), Scalar(0, 255, 0), 1);
        std::cout << "point1:" << p1 << "point2:" << p2 << std::endl;
    }
    result.clear();
    imshow(windowName, frame);
    waitKey(10);
}

Detector::~Detector()
{
}

} // namespace smoke_adaboost