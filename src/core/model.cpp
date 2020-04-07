#include "model.hpp"

namespace smoke_adaboost
{
/************************************* smokeCascadeDetectorTrainData ***************************************/

/**************************************  smokeFeatureEvaluator  ************************************************/

// 从标注文本文件中读取样本标签及路径
// Args:
//      annotationFilePath: 标注文件路径
// Note:
//      标注文件格式：
//      label filePath
//      如： 0 a/b/c/d/e
//      标签与路径间间隔一个空格
//      输入图片格式仅支持：.jpg .jpeg .png
void smokeFeatureGenerator::loadAnnotationFromFile(const std::string &_annotationFilePath)
{
    std::ifstream fin;
    fin.open(_annotationFilePath.c_str());
    if (!fin.is_open())
    {
        CV_Error(Error::Code::StsObjectNotFound, "File is not exsist!");
    }
    std::string s;
    while (getline(fin, s))
    {
        char c = s.c_str()[0];
        if (!(c >= 48 && c <= 57))
        {
            std::cout << "this item's format can not be recognized. skip and continue ->" << std::endl;
            continue;
        }
        uint8_t label = (uint8_t)c - 48;
        sample item;
        item.label = label;
        item.filePath = s.substr(2, (s.size() - 2));
        if (item.filePath.substr(item.filePath.size() - 4, item.filePath.size()) != ".jpg" &&
            item.filePath.substr(item.filePath.size() - 5, item.filePath.size()) != ".jpeg" &&
            item.filePath.substr(item.filePath.size() - 4, item.filePath.size()) != ".png") // 判断后缀是否合格
        {
            std::cout << "Do not support this format image file. skip and continue ->" << std::endl;
            continue;
        }
        samples.emplace_back(item);
        std::cout << "read form file-> "
                  << "label: " << (int)item.label
                  << " path: " << item.filePath << std::endl;
    }
    std::cout << "DONE. " << std::endl;
    fin.close();
}

// 构造函数完成从文件中读取样本标签和图片路径
smokeFeatureGenerator::smokeFeatureGenerator(bool _multiThread, int _threadNums)
{
    multithread = _multiThread;
    threadNums = _threadNums;
    finished = false;
}

// 读取图片数据并计算特征保存到内存中（非多线程模式）
void smokeFeatureGenerator::runGenerator()
{
    int nsamples = samples.size();
    CV_Assert(nsamples != 0);
    Mat sampleImage, sampleGrayImage;
    vector<float> tmpFeaturesMap, tmpRawFeatures; // 存储临时变量
    labels.reserve(nsamples);

    std::cout << std::endl
              << "--- start read image and calculate features ---" << std::endl;

    const std::string printInfro = "Read image and generate features";
    for (int i = 0; i < nsamples; ++i)
    {
        sampleImage = imread(samples[i].filePath);
        CV_Assert(!sampleImage.empty());
        cvtColor(sampleImage, sampleGrayImage, CV_BGR2GRAY);
        // debug
        // Mat resizedImage;
        // resize(sampleGrayImage, resizedImage, Size(101,99));  // resize image to fixed size
        CV_Assert(sampleGrayImage.channels() == 1);
        generateFeatureMap(sampleGrayImage, tmpFeaturesMap, fmParam.cols,
                           fmParam.rows, fmParam.ddepth, fmParam.stride,
                           fmParam.windowSize, fmParam.padding); // 获得特征映射图
        BlocksAndStatisticalFeatures sf(tmpFeaturesMap, fmParam.cols, fmParam.rows,
                                        fmParam.ddepth, fmParam.tiledCutNums, fmParam.ringedCutNums);
        // std::cout << "row: " << fmParam.rows << ", col: " << fmParam.cols << "i:" << i <<std::endl; 
        sf.getStatisticalFeatures(tmpRawFeatures);
        // std::cout << "size of feature: "<< tmpRawFeatures.size() <<std::endl;
        if (i == 0)
        {
            rawFeatures.reserve(nsamples * tmpRawFeatures.size());
        }
        rawFeatures.emplace_back(tmpRawFeatures);
        labels.push_back(samples[i].label);
        printProcess((float)i / (nsamples - 1), printInfro);
        tmpRawFeatures.clear();
    }
    std::cout << std::endl
              << "All samples have been finished. " << std::endl;
    finished = true;
}

// 根据计算出来的特征和标签值来生成供训练的TrainData
// Returns:
//      Ptr<TrainData>: 生成的数据集
Ptr<TrainData> smokeFeatureGenerator::getTrainData()
{
    if (multithread == false) // 非多线程模式
    {
        if (!rawFeatures.empty())
        {
            int featureSize = rawFeatures[0].size();
            int nsamples = labels.size();
            CV_Assert(nsamples == rawFeatures.size());
            // Ptr<float> totalRawFeatures = new float[nsamples * featureSize]();
            Mat X;
            for (int i = 0; i < nsamples; ++i)
            {
                float *data = &rawFeatures[i][0];
                Mat tmp(1, featureSize, CV_32F, data);
                X.push_back(tmp);
                tmp.release();
            }
            Mat y(labels, CV_32F);
            sidx.reserve(nsamples);
            vidx.reserve(featureSize);
            int i;
            for (i = 0; i < nsamples; i++)
            {
                sidx.push_back(i);
            }
            for (i = 0; i < featureSize; i++)
            {
                vidx.push_back(i);
            }
            std::cout << "Creat TrainData Object successfully!" << std::endl;
            std::cout << "debug type:"<< X.type() << std::endl;
            return TrainData::create(X, ROW_SAMPLE, y, vidx, sidx);
        }
    }
}

// 将计算出来的特征和标签写入到文件中
// Args:
//      _featureFilePath: 待写入文件路径
void smokeFeatureGenerator::writeFeaturesToFile(const std::string &_featureFilePath)
{
    FileStorage fs(_featureFilePath, FileStorage::WRITE);
    if (!fs.isOpened())
    {
        std::cout << "Can not open the feature file." << std::endl;
        return;
    }
    if (rawFeatures.empty())
    {
        CV_Error(Error::Code::StsObjectNotFound, "features is empty");
        return;
    }
    std::cout << "Start write features to file: " << _featureFilePath << std::endl;
    time_t rawTime;
    time(&rawTime);
    int nsamples = samples.size();
    CV_Assert(nsamples == labels.size());
    fs << "saveTime" << asctime(localtime(&rawTime)); // 写入时间戳
    fs << "nsamples" << nsamples;
    fs << "samples"
       << "[";
    const std::string printInfor = "Write features into file, Presistence";
    for (int i = 0; i < nsamples; i++)
    {
        fs << "{";
        fs << "label" << labels[i];
        fs << "feature" << rawFeatures[i];
        fs << "}";
        printProcess((float)i / nsamples, printInfor);
    }

    fs << "]";
    fs.release();
    std::cout << "Write features finished. " << std::endl;
}

// 从文件中读取特征
//
void smokeFeatureGenerator::readFeaturesFromFile(const std::string &_filePath)
{
    std::cout << "start read features from file: " << _filePath << std::endl;
    FileStorage fs(_filePath, FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "Can not open the feature file." << std::endl;
        return;
    }
    int nsamples = fs["nsamples"];
    FileNode fn = fs["samples"];
    FileNodeIterator it = fn.begin();
    CV_Assert(nsamples == fn.size());
    float tmp_label = 0;
    vector<float> tmpFeature;
    for (int i = 0; i < nsamples; i++, it++)
    {
        (*it)["label"] >> tmp_label;
        (*it)["feature"] >> tmpFeature;
        labels.emplace_back(tmp_label);
        rawFeatures.emplace_back(tmpFeature);
        // printProcess((float)i / (nsamples - 1));
    }
    std::cout << "Read from file finished. " << std::endl;
    fs.release();
}

/**************************************  smokeCascadeBoostDescisionTree  ****************************************/

// 在开始训练之前先对相应的变量以及内存空间作初始化
// 并根据boost的类型转化response的值
// Args:
//      trainData: 训练数据
//      flags: 预测时的类型： PREDICT_AUTO | PREDICT_MAX_VOTE | PREDICT_SUM | RAW_OUTPUT
//
void smokeCascadeBoostDescisionTree::beforeTraining(const Ptr<TrainData> &trainData, int flags)
{
    DTreesImpl::startTraining(trainData, flags);
    if (boostParam.boostType != Boost::DISCRETE)
    {
        _isClassifier = false;
        int i, n = (int)w->cat_responses.size();
        w->ord_responses.resize(n);

        double a = -1, b = 1;
        if (boostParam.boostType == Boost::LOGIT)
        {
            a = -2, b = 2;
        }
        for (i = 0; i < n; i++)
            w->ord_responses[i] = w->cat_responses[i] > 0 ? b : a;
    }
    normalizeWeights(); // 将样本权重初始为均等权重
}

// 归一化样本权重
//
void smokeCascadeBoostDescisionTree::normalizeWeights()
{
    int i, n = (int)w->sidx.size();
    double sumw = 0, a, b;

    for (i = 0; i < n; i++)
        sumw += w->sample_weights[w->sidx[i]];
    if (sumw > DBL_EPSILON)
    {
        a = 1. / sumw;
        b = 0;
    }
    else
    {
        a = 0;
        b = 1;
    }
    for (i = 0; i < n; i++)
    {
        double &wval = w->sample_weights[w->sidx[i]];
        wval = wval * a + b;
    }
}

// 深度优先遍历（前序遍历）整棵树并将节点的估计值乘上相应的比例
//
void smokeCascadeBoostDescisionTree::scaleTree(int root, float scale)
{
    int nidx = root, pidx = 0;
    Node *node = nullptr;

    for (;;)
    {
        for (;;) // 从左子树开始遍历
        {
            node = &nodes[nidx];
            node->value *= scale;
            if (node->left < 0) // 左节点遍历到底
                break;
            nidx = node->left;
        }
        for (pidx = node->parent; pidx >= 0 && nodes[pidx].right == nidx;
             nidx = pidx, pidx = nodes[nidx].parent)
            ; // 右节点向上回溯
        if (pidx < 0)
            break;
        nidx = nodes[pidx].right;
    }
}

// 根据boost算法的类型修改节点的值
void smokeCascadeBoostDescisionTree::calcValue(int nidx, const vector<int> &_sidx)
{
    DTreesImpl::calcValue(nidx, _sidx);
    WNode *wnode = &w->wnodes[nidx];
    if (boostParam.boostType == Boost::DISCRETE)
    {
        wnode->value = wnode->class_idx == 0 ? -1 : 1;
    }
    else
    {
        double p = (wnode->value + 1) * 0.5;
        wnode->value = 0.5 * log_ratio(p);
    }
}

// adaboost算法，根据训练误差在每次迭代后更新样本权重并对权重进行修剪
// Args:
//      treeidx: 树索引
//      sidx: 样本索引
void smokeCascadeBoostDescisionTree::updateWeightsAndTrim(int treeidx, vector<int> &sidx)
{
    int i, n = (int)w->sidx.size();
    int nvars = (int)varIdx.size();
    double sumw = 0., C = 1.;
    cv::AutoBuffer<double> buf(n + nvars);
    double *result = buf;
    float *sbuf = (float *)(result + n);
    Mat sample(1, nvars, CV_32F, sbuf);
    int predictFlags = boostParam.boostType == Boost::DISCRETE ? (PREDICT_MAX_VOTE | RAW_OUTPUT) : PREDICT_SUM;
    predictFlags |= COMPRESSED_INPUT;

    for (i = 0; i < n; i++)
    {
        w->data->getSample(varIdx, w->sidx[i], sbuf);
        result[i] = predictTrees(Range(treeidx, treeidx + 1), sample, predictFlags);
    }

    // now update weights and other parameters for each type of boosting
    if (boostParam.boostType == Boost::DISCRETE)
    {
        // Discrete AdaBoost:
        //   weak_eval[i] (=f(x_i)) is in {-1,1}
        //   err = sum(w_i*(f(x_i) != y_i))/sum(w_i)
        //   C = log((1-err)/err)
        //   w_i *= exp(C*(f(x_i) != y_i))
        double err = 0.;

        for (i = 0; i < n; i++)
        {
            int si = w->sidx[i];
            double wval = w->sample_weights[si];
            sumw += wval;
            err += wval * (result[i] != w->cat_responses[si]);
        }

        if (sumw != 0)
            err /= sumw;
        C = -log_ratio(err);
        double scale = std::exp(C);

        sumw = 0;
        for (i = 0; i < n; i++)
        {
            int si = w->sidx[i];
            double wval = w->sample_weights[si];
            if (result[i] != w->cat_responses[si])
                wval *= scale;
            sumw += wval;
            w->sample_weights[si] = wval; // w->sample_weights[si] = wval / sumw
        }

        scaleTree(roots[treeidx], C);
    }
    else if (boostParam.boostType == Boost::REAL || boostParam.boostType == Boost::GENTLE)
    {
        // Real AdaBoost:
        //   weak_eval[i] = f(x_i) = 0.5*log(p(x_i)/(1-p(x_i))), p(x_i)=P(y=1|x_i)
        //   w_i *= exp(-y_i*f(x_i))

        // Gentle AdaBoost:
        //   weak_eval[i] = f(x_i) in [-1,1]
        //   w_i *= exp(-y_i*f(x_i))
        for (i = 0; i < n; i++)
        {
            int si = w->sidx[i];
            CV_Assert(std::abs(w->ord_responses[si]) == 1); // -1 or 1
            double wval = w->sample_weights[si] * std::exp(-result[i] * w->ord_responses[si]);
            sumw += wval;
            w->sample_weights[si] = wval;
        }
    }
    else if (boostParam.boostType == Boost::LOGIT)
    {
        // LogitBoost:
        //   weak_eval[i] = f(x_i) in [-z_max,z_max]
        //   sum_response = F(x_i).
        //   F(x_i) += 0.5*f(x_i)
        //   p(x_i) = exp(F(x_i))/(exp(F(x_i)) + exp(-F(x_i))=1/(1+exp(-2*F(x_i)))
        //   reuse weak_eval: weak_eval[i] <- p(x_i)
        //   w_i = p(x_i)*1(1 - p(x_i))
        //   z_i = ((y_i+1)/2 - p(x_i))/(p(x_i)*(1 - p(x_i)))
        //   store z_i to the data->data_root as the new target responses
        const double lb_weight_thresh = FLT_EPSILON;
        const double lb_z_max = 10.;
        double sumResult = 0.0f;

        for (i = 0; i < n; i++)
        {
            int si = w->sidx[i];
            sumResult += 0.5 * result[i];
            double p = 1. / (1 + std::exp(-2 * sumResult));
            double wval = std::max(p * (1 - p), lb_weight_thresh), z;
            w->sample_weights[si] = wval;
            sumw += wval;
            if (w->ord_responses[si] > 0)
            {
                z = 1. / p;
                w->ord_responses[si] = std::min(z, lb_z_max);
            }
            else
            {
                z = 1. / (1 - p);
                w->ord_responses[si] = -std::min(z, lb_z_max);
            }
        }
    }
    else
        CV_Error(CV_StsNotImplemented, "Unknown boosting type");

    /*if( bparams.boostType != Boost::LOGIT )
        {
            double err = 0;
            for( i = 0; i < n; i++ )
            {
                sumResult[i] += result[i]*C;
                if( bparams.boostType != Boost::DISCRETE )
                    err += sumResult[i]*w->ord_responses[w->sidx[i]] < 0;
                else
                    err += sumResult[i]*w->cat_responses[w->sidx[i]] < 0;
            }
            printf("%d trees. C=%.2f, training error=%.1f%%, working set size=%d (out of %d)\n", (int)roots.size(), C, err*100./n, (int)sidx.size(), n);
        }*/

    // renormalize weights
    if (sumw > FLT_EPSILON)
        normalizeWeights();

    if (boostParam.weightTrimRate <= 0. || boostParam.weightTrimRate >= 1.)
        return;

    // trime sample weights
    for (i = 0; i < n; i++)
        result[i] = w->sample_weights[w->sidx[i]];
    std::sort(result, result + n);

    // as weight trimming occurs immediately after updating the weights,
    // where they are renormalized, we assume that the weight sum = 1.
    sumw = 1. - boostParam.weightTrimRate;

    for (i = 0; i < n; i++)
    {
        double wval = result[i];
        if (sumw <= 0)
            break;
        sumw -= wval;
    }

    double threshold = i < n ? result[i] : DBL_MAX;
    sidx.clear();

    for (i = 0; i < n; i++)
    {
        int si = w->sidx[i];
        if (w->sample_weights[si] >= threshold)
            sidx.push_back(si);
    }
}

// 训练主函数，根据boost算法添加简单的决策树弱分类器构成强分类器，然后根据预测结果更新样本的权重和分类器权重
// 同时根据cascade算法，当检测率和误识率达到设定水平时，停止训练
// Args:
//      trainData: 一级强分类器的训练数据集
//      flags: 预测标志， 初始化DTree时使用
// Return:
//      bool: 是否完成训练
bool smokeCascadeBoostDescisionTree::trainBoost(const Ptr<TrainData> &trainData,
                                                const Ptr<TrainData> &evalData,
                                                int flags)
{
    std::cout << "------------- stage: " << stage << " ----------------------" << std::endl;
    std::cout << "start training: " << std::endl;
    beforeTraining(trainData, flags);
    assert(!trainData.empty());
    int weak_count = 0; // 已经添加的弱分类器个数
    vector<int> sidx = w->sidx;

    do
    {
        int root = DTreesImpl::addTree(sidx);
        if (root < 0)
        {
            std::cout << "root is negtive, ealy terminate.";
            break;
        }
        updateWeightsAndTrim(weak_count, sidx);
        weak_count++;
    } while (weak_count < boostParam.weakCount && !reachDesireRate2(weak_count));

    bool isTrained = false;
    if (weak_count > 0)
    {
        isTrained = true;
    }

    return isTrained;
}

// 判断一级的adaboost是否达到了miniHitRate和maxFAR(maxFPR)
// Args:
//      weak_count: 弱分类器的个数
// Return:
//      bool: 是否满足条件
bool smokeCascadeBoostDescisionTree::reachDesireRate(int weak_count, const Ptr<TrainData> &_evalData)
{

    vector<float> responses;
    _evalData->getResponses().copyTo(responses); // 读取标签
    CV_Assert(!responses.empty());
    Mat evalData = _evalData->getSamples(); // 获取数据
    CV_Assert(!evalData.empty());
    vector<int> sidx;
    _evalData->getTrainSampleIdx().copyTo(sidx);
    CV_Assert(sidx.size() == evalData.rows);

    int numPos = 0, numNeg = 0, numPosTrue = 0;
    int flags = boostParam.boostType == Boost::DISCRETE ? PREDICT_AUTO : PREDICT_SUM;
    if (boostParam.boostType != Boost::DISCRETE)
    {
        for (int i = 0; i < responses.size(); i++)
            responses[i] = responses[i] == 0 ? -1 : 1;
    }

    std::cout << std::endl
              << "weak classfier:" << std::endl;
    int n = _evalData->getNSamples();
    for (int i = 0; i < n; ++i)
    {
        CV_Assert(!(w->sidx.empty()));
        float val = DTreesImpl::predictTrees(Range(0, roots.size()), evalData.row(i), flags);
        if (boostParam.boostType == Boost::DISCRETE)
        {
            if (val == 1)
            {
                numPos++;
                if (responses[i] == 1)
                    numPosTrue++;
                else
                    numNeg++;
            }
        }
        else
        {
            if (val > 0)
            {
                numPos++;
                if (responses[i] == 1)
                    numPosTrue++;
                else
                    numNeg++;
            }
        }
    }
    float hitRate = (float)numPosTrue / ((float)numPos + std::numeric_limits<float>::epsilon());
    float falseAlarmRate = (float)numNeg / ((float)numPos + +std::numeric_limits<float>::epsilon());
    std::cout << "weak count: " << weak_count << "|";
    std::cout << " hr: ";
    std::cout << hitRate << "|";
    std::cout << " fr: ";
    std::cout << falseAlarmRate;
    std::cout << std::endl;

    //
    return falseAlarmRate <= maxFalseAlarmRate && hitRate >= miniHitRate;
}

// 判断一级的adaboost是否达到了miniHitRate和maxFAR(maxFPR)
// Args:
//      weak_count: 弱分类器的个数
// Return:
//      bool: 是否满足条件
bool smokeCascadeBoostDescisionTree::reachDesireRate2(int weak_count)
{

    vector<float> responses;
    w->data->getResponses().copyTo(responses); // 读取标签
    CV_Assert(!responses.empty());
    Mat evalData = w->data->getSamples(); // 获取数据
    CV_Assert(!evalData.empty());
    vector<int> sidx;
    w->data->getTrainSampleIdx().copyTo(sidx);
    CV_Assert(sidx.size() == evalData.rows);

    int numPos = 0, numNeg = 0, numPosTrue = 0;
    int flags = boostParam.boostType == Boost::DISCRETE ? PREDICT_AUTO : PREDICT_SUM;
    if (boostParam.boostType != Boost::DISCRETE)
    {
        for (int i = 0; i < responses.size(); i++)
            responses[i] = responses[i] == 0 ? -1 : 1;
    }

    std::cout << std::endl
              << "weak classfier:" << std::endl;
    int n = w->data->getNSamples();
    for (int i = 0; i < n; ++i)
    {
        CV_Assert(!(w->sidx.empty()));
        float val = DTreesImpl::predictTrees(Range(0, roots.size()), evalData.row(i), flags);
        if (boostParam.boostType == Boost::DISCRETE)
        {
            if (val == 1)
            {
                numPos++;
                if (responses[i] == 1)
                    numPosTrue++;
                else
                    numNeg++;
            }
        }
        else
        {
            if (val > 0)
            {
                numPos++;
                if (responses[i] == 1)
                    numPosTrue++;
                else
                    numNeg++;
            }
        }
    }
    float hitRate = (float)numPosTrue / ((float)numPos + std::numeric_limits<float>::epsilon());
    float falseAlarmRate = (float)numNeg / ((float)numPos + +std::numeric_limits<float>::epsilon());
    std::cout << "weak cont: " << weak_count << "|";
    std::cout << " hr: ";
    std::cout << hitRate << "|";
    std::cout << " fr: ";
    std::cout << falseAlarmRate;
    std::cout << std::endl;

    //
    return falseAlarmRate <= maxFalseAlarmRate;
}

// 拓展DTreesImpl的predictTrees
// 根据预测类型修改成相应的预测值
// Args:
//      range: 弱分类器的范围
//      sample: 待预测样本
//      flags0: 预测标志
// Return:
//      float: 预测结果
//
float smokeCascadeBoostDescisionTree::predictTree(const Range &range, const Mat &sample, int flags0)
{
    int flags = (flags0 & ~PREDICT_MASK) | PREDICT_SUM;
    float val = DTreesImpl::predictTrees(range, sample, flags);
    if (flags != flags0)
    {
        int ival = (int)(val > 0);
        if (!(flags0 & RAW_OUTPUT))
            ival = classLabels[ival];
        val = (float)ival;
    }
    return val;
}

// boost分类器预测函数
// Args:
//      samples: 多个待预测样本
//      result: 结果
// Return:
//      第一个样本的预测结果
float smokeCascadeBoostDescisionTree::predictSamples(Mat &samples, std::vector<float> &result)
{
    int flags = PREDICT_AUTO;
    int nsamples = samples.rows;
    float retRes = 0;
    for (int i = 0; i < nsamples; i++)
    {
        float res = predictTree(Range(0, roots.size()), samples.row(i), flags);
        result.emplace_back(res);
        if (i == 0)
            retRes = res;
    }
    return retRes;
}

void smokeCascadeBoostDescisionTree::writeTrainingParams(FileStorage &fs) const
{
    fs << "boosting_type" << (boostParam.boostType == Boost::DISCRETE ? "DiscreteAdaboost" : boostParam.boostType == Boost::REAL ? "RealAdaboost" : boostParam.boostType == Boost::LOGIT ? "LogitBoost" : boostParam.boostType == Boost::GENTLE ? "GentleAdaboost" : "Unknown");

    fs << "miniHitRate" << miniHitRate;
    fs << "maxFalseAlarmRate" << maxFalseAlarmRate;
    fs << "stage" << stage;
    DTreesImpl::writeTrainingParams(fs);
    fs << "weight_trimming_rate" << boostParam.weightTrimRate;
}

void smokeCascadeBoostDescisionTree::write(FileStorage &fs) const
{
    if (roots.empty())
        CV_Error(cv::Error::Code::StsObjectNotFound, "DTree has not been trained.");
    writeFormat(fs);
    writeParams(fs);

    int n = roots.size();
    fs << "ntrees" << n << "trees"
       << "[";
    for (int k = 0; k < n; k++)
    {
        writeTree(fs, roots[k]);
    }
    fs << "]";
}

void smokeCascadeBoostDescisionTree::readParams(const FileNode &fn)
{
    DTreesImpl::readParams(fn);

    FileNode trainParams = fn["training_params"];
    std::string boostType = (std::string)(fn["boosting_type"].empty() ? trainParams["boosting_type"] : fn["boosting_type"]);
    boostParam.boostType = (boostType == "DiscreteAdaboost" ? Boost::DISCRETE : boostType == "RealAdaboost" ? Boost::REAL : boostType == "LogitBoost" ? Boost::LOGIT : boostType == "GentleAdaboost" ? Boost::GENTLE : 255);
    miniHitRate = (float)(fn["miniHitRate"].empty() ? trainParams["miniHitRate"] : fn["miniHitRate"]);
    maxFalseAlarmRate = (float)(fn["maxFalseAlarmRate"].empty() ? trainParams["maxFalseAlarmRate"] : fn["maxFalseAlarmRate"]);
    stage = (float)(fn["stage"].empty() ? trainParams["stage"] : fn["stage"]);
    fn["weight_trimming_rate"].empty() ? trainParams["weight_trimming_rate"] >> boostParam.weightTrimRate
                                       : fn["weight_trimming_rate"] >> boostParam.weightTrimRate;
}

void smokeCascadeBoostDescisionTree::read(const FileNode &fn)
{
    // clear();
    readParams(fn);
    int ntrees = fn["ntrees"];

    FileNode trees = fn["trees"];
    CV_Assert(ntrees == (int)trees.size());
    FileNodeIterator it = trees.begin();
    for (int treeIdx = 0; treeIdx < ntrees; treeIdx++, ++it)
    {
        FileNode treeFN = (*it)["nodes"];
        readTree(treeFN);
    }
}

/********************************** smokeCascadeDetector ******************************************************/

// 默认构造函数
smokeCascadeDetector::smokeCascadeDetector()
{
    detParams.singleLayerMaxFPR = 0.5;
    detParams.singleLayerMinTPR = 0.995;
    detParams.stageNums = 10;
    detParams.targetFPR = pow(detParams.singleLayerMaxFPR, detParams.stageNums);
    detParams.useCrossValidate = false;
    detParams.boostParam.boostType = Boost::DISCRETE;
    detParams.boostParam.weakCount = 1000;
    detParams.boostParam.weightTrimRate = 0.95;
    tmpFalsePositiveRate = 1.0;
}

// cascadeDetector有参构造函数
// Args:
//      _detParams: 检测器参数
//
smokeCascadeDetector::smokeCascadeDetector(const DetectorParam &_detParams)
{
    detParams = _detParams;
    tmpFalsePositiveRate = 1.0;
}

// 训练烟雾检测级联分类器
// 更新数据集
// 检测是否满足最大误检率
// 训练
// 保存模型

// 对交叉验证：
// 首先要将原始数据分为cvfold份，其中cvfold-1份作为训练集，1份作为验证集
// 对每一折进行同上述的操作，循环完成每折训练
// Args:
//      cascadeTrainData: 当前层所使用的数据集
//
void smokeCascadeDetector::train(Ptr<TrainData> cascadeTrainData, Ptr<TrainData> cascadeEvalData)
{

    std::cout << "---------smoke cascade detector now begin training----------" << std::endl;
    int nums = stageStrongClassifiers.size();
    Ptr<TrainData> stageTrainData = cascadeTrainData;
    double FPR = 1.0;
    for (int n = nums; n < detParams.stageNums; n++)
    {
        Ptr<smokeCascadeBoostDescisionTree> stageClassfier = makePtr<smokeCascadeBoostDescisionTree>(
            detParams.boostParam.boostType, detParams.boostParam.weakCount,
            detParams.boostParam.weightTrimRate,
            detParams.singleLayerMinTPR, detParams.singleLayerMaxFPR,
            n, detParams.DTreeparam);
        stageClassfier->trainBoost(stageTrainData, cascadeEvalData, smokeCascadeBoostDescisionTree::PREDICT_AUTO);
        stageStrongClassifiers.push_back(stageClassfier);
        FPR *= tmpFalsePositiveRate;
        stageTrainData = updateTrainData(stageTrainData, stageClassfier);
        std::cout << tmpFalsePositiveRate << std::endl;
        if (tmpFalsePositiveRate <= detParams.targetFPR)
        {
            std::cout << std::endl
                      << "Classfier reached the desired max False Positive Rate: " << tmpFalsePositiveRate << "%" << std::endl;
            break;
        }
        // std::cout << "fpr: " << FPR << std::endl; // debug

    }
    // saveModel();
}

// 根据当前层的adaboost分类器来更新数据集
// Args:
//      _stageTrainData: 当前层所使用的数据集
//      _stageClassfier: 当前层的分类器
// Return:
//      下一层所使用的数据集指针
//
Ptr<TrainData> smokeCascadeDetector::updateTrainData(Ptr<TrainData> _stageTrainData,
                                                     Ptr<smokeCascadeBoostDescisionTree> _stageClassfier)
{
    clear();
    Mat samples = _stageTrainData->getSamples();
    CV_Assert(!samples.empty());
    size_t nsamples = samples.rows;

    vector<int> labels, VarIdx, sampleIdx;
    _stageTrainData->getTrainSampleIdx().copyTo(sampleIdx);
    _stageTrainData->getResponses().copyTo(labels);

    CV_Assert(!labels.empty());
    _stageTrainData->getVarIdx().copyTo(VarIdx);
    CV_Assert(!VarIdx.empty());

    Mat newSamples;
    size_t negCount = 0, numPos = 0; // 用于计算虚警率
    vector<float> resAll;

    if (detParams.boostParam.boostType != Boost::DISCRETE)
    {
        for (int i = 0; i < labels.size(); i++)
            labels[i] = labels[i] == 0 ? -1 : 1;
    }

    for (size_t i = 0; i < nsamples; i++)
    {
        Mat rowSample = samples.row(i);
        CV_Assert(!rowSample.empty());
        float predictVal = _stageClassfier->predictSamples(rowSample, resAll);
        if (detParams.boostParam.boostType == Boost::DISCRETE)
        {

            if (predictVal == 1.0) // 预测为正的样本
            {
                newSamples.push_back(rowSample);
                predictPositiveLabels.emplace_back(labels[i]);

                if (labels[i] == 0)
                    negCount++;
                newSampleIdx.emplace_back(numPos);
                numPos++;
            }
            
           /*
           if (predictVal != labels[i])
           {
               newSamples.push_back(rowSample);
               predictPositiveLabels.emplace_back(labels[i]);
               if (labels[i] == 0)
                   negCount++;
               newSampleIdx.emplace_back(numPos);
               numPos++;
           }
           */
        }
        else 
        {
            if (predictVal > 0)
            {
                newSamples.push_back(rowSample);
                predictPositiveLabels.emplace_back(labels[i]);

                if (labels[i] == -1)
                    negCount++;
                newSampleIdx.emplace_back(numPos);
                numPos++;                
            }
        }
    }
    CV_Assert(!newSamples.empty());
    tmpFalsePositiveRate = (double)negCount / numPos;
    std::cout << std::endl
              << numPos << std::endl
              << negCount;

    Ptr<TrainData> newStageTrainData = TrainData::create(newSamples, ROW_SAMPLE,
                                                         predictPositiveLabels,
                                                         VarIdx, newSampleIdx);
    return newStageTrainData;
}

// cascade检测器的预测函数
// Args:
//      _samples: 预测样本，可以是多维
//      _res: 所有样本的预测结果
// Return:
//      val: 第一样本的预测值
//
float smokeCascadeDetector::predict(const Mat &_samples, vector<float> &_res)
{
    CV_Assert(!_samples.empty());
    int nsamples = _samples.rows;
    assert(!stageStrongClassifiers.empty());
    vector<float> tmpRes;
    for (int i = 0; i < nsamples; i++)
    {
        float predictRes = 0;
        for (std::vector<Ptr<smokeCascadeBoostDescisionTree>>::iterator it = stageStrongClassifiers.begin();
             it < stageStrongClassifiers.end(); it++)
        {
            Mat sample = _samples.row(i);
            if (detParams.boostParam.boostType == Boost::DISCRETE)
            {
                if ((*it)->predictSamples(sample, tmpRes) == 0)
                {
                    predictRes = 0;
                    break;
                }
                predictRes = 1;
            }
            else 
            {
                if ((*it)->predictSamples(sample, tmpRes) < 0)
                {
                    predictRes = -1;
                    break;
                }
                predictRes = 1;                
            }
        }
        _res.push_back(predictRes);
    }
    return _res[0];
}

// 对验证数据集进行测试
// Args:
//      _evalData: 验证数据集
//
void smokeCascadeDetector::evaluate(Ptr<TrainData> _evalData)
{
    Mat evalSamples = _evalData->getSamples();
    vector<float> labels;
    _evalData->getResponses().copyTo(labels);
    int nsamples = evalSamples.rows;
    CV_Assert(!labels.empty());
    size_t numNeg = 0, numPos = 0;

    for (std::vector<float>::iterator it = labels.begin(); it < labels.end(); it++)
    {

        if ((*it) == 0)
            numNeg++;
        else
            numPos++;
    }
    if (detParams.boostParam.boostType != Boost::DISCRETE)
    {
        for (int i=0; i<labels.size(); i++)
            labels[i] = labels[i] == 0 ? -1 : 1;
    }
    std::cout << "evaluation----" << std::endl;
    std::cout << "NumPos: " << numPos << ", NumNeg: " << numNeg << std::endl;

    vector<float> predictRes;
    predict(evalSamples, predictRes);
    CV_Assert(!predictRes.empty());
    CV_Assert(predictRes.size() == nsamples);

    int numPosTrue = 0, numNegTrue = 0, numPosFalse = 0, numNegFalse = 0;
    for (int i = 0; i < nsamples; i++)
    {
        if (detParams.boostParam.boostType == Boost::DISCRETE)
        {
            if (predictRes[i] == labels[i])
            {
                if (labels[i] == 0)
                    numNegTrue++;
                else
                    numPosTrue++;
            }
            else 
            {
                if (labels[i] == 0)
                    numNegFalse++;
                else
                    numPosFalse++;
            }
        }
        else 
        {
            if (predictRes[i] > 0)
            {
                if (labels[i] == 1)
                    numPosTrue++;
                else 
                    numPosFalse++;
            }
            else
            {
                if (labels[i] == 1)
                    numPosFalse++;
                else
                    numNegTrue++;
            }
        }
    }

    std::cout << "+----+-----+-----+" << std::endl;
    std::cout << "     |  T  |  F  |" << std::endl;
    std::cout << "------------------" << std::endl;
    std::cout << "| P  | " << numPosTrue << "   |  " << numPosFalse << std::endl;
    std::cout << "------------------" << std::endl;
    std::cout << "| N  | " << numNegTrue << "   |  " << numNegFalse << std::endl;
    std::cout << "------------------" << std::endl;

    float accuracy = (float)(numPosTrue + numNegTrue) / (float)nsamples;
    CV_Assert(accuracy >= 0 && accuracy <= 1);
    float precision = (float)numPosTrue / (float)(numPosTrue + numPosFalse);
    float recall = (float)numPosTrue / (float)(numPosTrue + numNegFalse);
    float FPR = (float)numNegFalse / (float)(numPosTrue + numNegFalse);
    std::cout << "Mterics: " << std::endl;
    std::cout << std::endl
              << "accuracy: " << accuracy * 100 << "%" << std::endl;
    std::cout << "precision: " << precision * 100 << "%" << std::endl;
    std::cout << "recall: " << recall * 100 << "%" << std::endl;
    std::cout << "FPR: " << FPR * 100 << "%" << std::endl;
    std::cout << "DONE";
}

void smokeCascadeDetector::writeParams(FileStorage &fs) const
{
    time_t rawTime;
    time(&rawTime);
    fs << "saveModelTime" << asctime(localtime(&rawTime));       // 写入时间戳
    fs << "maxFalsePositiveRate" << detParams.singleLayerMaxFPR; // 每一层adaboost分类器的最大误识率
    fs << "miniTruePositiveRate" << detParams.singleLayerMinTPR;
    fs << "targetFalsePositiveRate" << detParams.targetFPR;
    fs << "boostType" << detParams.boostParam.boostType;
    fs << "weakCount" << detParams.boostParam.weakCount;
    fs << "weightTrimRate" << detParams.boostParam.weightTrimRate;
    fs << "featureWindowSize" << detParams.featureParam.windowSize;
    fs << "featureStride" << detParams.featureParam.stride;
    fs << "featureTiledCutSize" << detParams.featureParam.tiledCutNums;
    fs << "featureRingedCutSize" << detParams.featureParam.ringedCutNums;
}

void smokeCascadeDetector::saveModel(const std::string &modelFileName)
{
    FileStorage fs(modelFileName, FileStorage::WRITE);
    std::cout << "*** Start saving model to :" << modelFileName << "***" << std::endl;
    if (stageStrongClassifiers.empty())
    {
        std::cout << "No strong classifier can be save, break out." << std::endl;
        return;
    }
    writeParams(fs);
    std::vector<Ptr<smokeCascadeBoostDescisionTree>>::const_iterator it = stageStrongClassifiers.begin();
    int k = 0;
    int stages = stageStrongClassifiers.size();
    fs << "stages" << stages;
    fs << "cascadeBoostClassifiers"
       << "[";
    for (; k < stageStrongClassifiers.size(); k++, it++)
    {

        fs << "{";
        (*it)->write(fs);
        fs << "}";
    }
    fs << "]";
    if (k != stageStrongClassifiers.size())
    {
        std::cout << " Some Classifiers have failed to save. " << std::endl;
        return;
    }
    std::cout << "Save Done. " << std::endl;
    fs.release();
}

void smokeCascadeDetector::readParams(const FileStorage &fs)
{
    CV_Assert(!fs["maxFalsePositiveRate"].empty());
    fs["maxFalsePositiveRate"] >> detParams.singleLayerMaxFPR;
    CV_Assert(!fs["miniTruePositiveRate"].empty());
    fs["miniTruePositiveRate"] >> detParams.singleLayerMinTPR;
    CV_Assert(!fs["targetFalsePositiveRate"].empty());
    fs["targetFalsePositiveRate"] >> detParams.targetFPR;
    CV_Assert(!fs["boostType"].empty());
    fs["boostType"] >> detParams.boostParam.boostType;
    CV_Assert(!fs["weakCount"].empty());
    fs["weakCount"] >> detParams.boostParam.weakCount;
    CV_Assert(!fs["weightTrimRate"].empty());
    fs["weightTrimRate"] >> detParams.boostParam.weightTrimRate;
    CV_Assert(!fs["featureWindowSize"].empty());
    fs["featureWindowSize"] >> detParams.featureParam.windowSize;
    CV_Assert(!fs["featureStride"].empty());
    fs["featureStride"] >> detParams.featureParam.stride;
    CV_Assert(!fs["featureTiledCutSize"].empty());
    fs["featureTiledCutSize"] >> detParams.featureParam.tiledCutNums;
    CV_Assert(!fs[ "featureRingedCutSize"].empty());
    fs[ "featureRingedCutSize"] >> detParams.featureParam.ringedCutNums; 
}

void smokeCascadeDetector::loadModel(const std::string &modelFileName)
{
    FileStorage fs;
    fs.open(modelFileName, FileStorage::READ);

    if (!fs.isOpened())
    {
        CV_Error(Error::Code::StsObjectNotFound, "File is not exsist!");
    }
    std::cout << "*** Start loading model from file: " << modelFileName << "***" << std::endl;

    readParams(fs);
    int stages = fs["stages"];
    FileNode cls = fs["cascadeBoostClassifiers"];
    CV_Assert(stages == (int)cls.size());
    FileNodeIterator cls_iter = cls.begin();

    for (int i = 0; i < stages; i++, cls_iter++)
    {
        Ptr<smokeCascadeBoostDescisionTree> boostClassifier = makePtr<smokeCascadeBoostDescisionTree>(
            detParams.boostParam.boostType, detParams.boostParam.weakCount,
            detParams.boostParam.weightTrimRate,
            detParams.singleLayerMinTPR, detParams.singleLayerMaxFPR, i, detParams.DTreeparam);
        boostClassifier->read((*cls_iter));
        stageStrongClassifiers.push_back(boostClassifier);
    }
    std::cout << "Load Model DONE. " << std::endl;

    fs.release();
}

} // namespace smoke_adaboost