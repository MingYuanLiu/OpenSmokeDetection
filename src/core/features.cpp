#include "features.hpp"

namespace smoke_adaboost
{
// 计算eoh和mag特征
void calculateEohAndMag(const cv::Mat &Mag, const cv::Mat &Theta,
                        uint16_t block_img_area,
                        std::vector<float_t> &eoh_histogram,
                        std::vector<float_t> &mag_histogram,
                        uint8_t eoh_bin,
                        uint8_t mag_bin)
{

    // cv::Mat block_img_magnitude, block_img_theta;
    // cv::cartToPolar(gradientX, gradientY, block_img_magnitude, block_img_theta, true); // 微分图像转化为幅相图

     double epsilon = 2e-21;
     //计算eoh
     // 根据eoh_bin来设置数组长度
     float eoh_[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
     float mag_[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
     float sum_eoh = 0.0;
     // 找到magnitude中的最大值
     double m_max = 0.0;
     
    cv::minMaxIdx(Mag, NULL, &m_max);
     if (!eoh_histogram.empty())
         eoh_histogram.clear();

     if (!mag_histogram.empty())
         mag_histogram.clear();

     m_max = m_max == 0 ? epsilon : m_max / 2; // 在求幅值直方图中使用

    int theta_img_cols = Theta.cols;
    int theta_img_rows = Theta.rows;
    // const float_t *theta_img_data_ptr;
    // const float_t *mag_img_data_ptr;

    // if (block_img_theta.isContinuous())
    // {
    //      theta_img_cols *= theta_img_rows;
    //     theta_img_rows = 1;
    // }
    // 角度值分量
    double bin_theta[9] = {0.0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180};

    for (int i = 0; i < theta_img_rows; i++)
    {
        // theta_img_data_ptr = block_img_theta.ptr<float_t>(i);
        // mag_img_data_ptr = block_img_magnitude.ptr<float_t>(i);
        for (int j = 0; j < theta_img_cols; j++)
        {
            // eoh计算部分
            // float_t theta = *theta_img_data_ptr;
            float theta = Theta.at<float>(i, j);
            if (theta > 180)
            {
                theta -= 180; // 将0～360度的相角转化为0～180度的相角
            }
            if (theta >= bin_theta[0] &&
                theta <= bin_theta[1])
            {
                // eoh_[0] += *mag_img_data_ptr;
                eoh_[0] += Mag.at<float>(i,j); // mag存储内存不连续
            }
            else if (theta > bin_theta[1] &&
                     theta <= bin_theta[2])
            {
                // eoh_[1] += *mag_img_data_ptr;
                eoh_[1] += Mag.at<float>(i,j);
            }
            else if (theta > bin_theta[2] &&
                     theta <= bin_theta[3])
            {
                // eoh_[2] += *mag_img_data_ptr;
                eoh_[2] += Mag.at<float>(i,j);
            }
            else if (theta > bin_theta[3] &&
                     theta <= bin_theta[4])
            {
                // eoh_[3] += *mag_img_data_ptr;
                eoh_[3] += Mag.at<float>(i,j);
            }
            else if (theta > bin_theta[4] &&
                     theta <= bin_theta[5])
            {
                // eoh_[4] += *mag_img_data_ptr;
                eoh_[4] += Mag.at<float>(i,j);
            }
            else if (theta > bin_theta[5] &&
                     theta <= bin_theta[6])
            {
                // eoh_[5] += *mag_img_data_ptr;
                eoh_[5] += Mag.at<float>(i,j);
            }

            else if (theta > bin_theta[6] &&
                     theta <= bin_theta[7])
            {
                // eoh_[6] += *mag_img_data_ptr;
                eoh_[6] += Mag.at<float>(i,j);
            }
            else if (theta > bin_theta[7] &&
                     theta <= bin_theta[8])
            {
                // eoh_[7] += *mag_img_data_ptr;
                eoh_[7] += Mag.at<float>(i,j); 
            }
            // mag计算部分
            uint16_t mq = uint16_t(Mag.at<float>(i,j) / (m_max / mag_bin)) > mag_bin - 1 ? mag_bin - 1 : uint16_t(Mag.at<float>(i,j) / (m_max / mag_bin));
            // note: 这里eoh_bin和mag_bin的数值相同，因此可以放在一个循环中
            switch (mq)
            {
            case 0:
                mag_[0] += 1;
                break;
            case 1:
                mag_[1] += 1;
                break;
            case 2:
                mag_[2] += 1;
                break;
            case 3:
                mag_[3] += 1;
                break;
            case 4:
                mag_[4] += 1;
                break;
            case 5:
                mag_[5] += 1;
                break;
            case 6:
                mag_[6] += 1;
                break;
            case 7:
                mag_[7] += 1;
                break;
            default:
                break;
            }

            // mag_img_data_ptr++;
            // theta_img_data_ptr++;
        }
    }

    for (auto i : eoh_)
    {
        sum_eoh += i;
    }
    for (auto &eoh : eoh_)
    {
        eoh /= (sum_eoh + epsilon); // 对eoh求归一化
        eoh_histogram.push_back(eoh);
    }
    CV_Assert(block_img_area != 0);

    float ed = sum_eoh / block_img_area; // eoh相对于图像面积的密度
    eoh_histogram.push_back(ed);
    for (auto &mag : mag_)
    {
        mag /= block_img_area; // 对mag作相对于图像大小归一化
        mag_histogram.push_back(mag);
    }
}


// 计算lbp(局部二进制模式)特征
void calculateModifiedLbp(const cv::Mat &block_img_gray_data,
                          uint16_t block_img_area,
                          std::vector<float_t> &lbp,
                          uint8_t radius,
                          uint8_t near_point_nums,
                          uint8_t lbp_bin)
{
   if (!lbp.empty())
        lbp.clear();

    std::array<float_t, 8> modified_lbp = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // lbp直方图
    // 将参数预先计算出来，进行查表
    // N=near_point_nums
    // x = sin(2*pi*(n/N))
    // y = cos(2*pi*(n/N))
    double x[near_point_nums] = {1, 0.707106781, 0, -0.707106781, -1, -0.707106781, 0, 0.707106781};
    double y[near_point_nums] = {0, -0.707106781, -1, -0.707106781, 0, 0.707106781, 1, 0.707106781};
    // fx, fy = floor(x), floor(y)
    int fx[near_point_nums] = {1, 0, 0, -1, -1, -1, 0, 0};
    int fy[near_point_nums] = {0, -1, -1, -1, 0, 0, 1, 0};
    // cx, cy = ceil(x), ceil(y)
    int cx[near_point_nums] = {1, 1, 0, 0, -1, 0, 0, 1};
    int cy[near_point_nums] = {0, 0, -1, 0, 0, 1, 1, 1};
    // tx = x - fx
    // ty = y - fy
    double tx[near_point_nums] = {0, 0.707106781, 0, 0.292893219, 0, 0.292893219, 0, 0.707106781};
    double ty[near_point_nums] = {0, 0.292893219, 0, 0.292893219, 0, 0.707106781, 0, 0.707106781};
    // w1 = (1-tx)*(1-ty)
    // w2 = tx*(1-ty)
    // w3 = (1-tx)*ty
    // w4 = tx*ty
    double w1[near_point_nums] = {1, 0.207106781, 1, 0.5, 1, 0.207106781, 1, 0.0857864377};
    double w2[near_point_nums] = {0, 0.5, 0, 0.207106781, 0, 0.0857864377, 0, 0.207106781};
    double w3[near_point_nums] = {0, 0.0857864377, 0, 0.207106781, 0, 0.5, 0, 0.207106781};
    double w4[near_point_nums] = {0, 0.207106781, 0, 0.0857864377, 0, 0.207106781, 0, 0.5};


    for (int i = radius; i < block_img_gray_data.rows - radius; i++)
    {
        for (int j = radius; j < block_img_gray_data.cols - radius; j++)
        {
            uint8_t tmp_lbp_value = 0;  // 临时储存求出来的lbp值
            uint8_t last_bit_value = 0; // 记录上一次的lbp位值
            uint8_t hop_times = 0;      // 记录01跳变次数
            uint8_t bit_value = 0;
            for (int n = 0; n < near_point_nums; n++)
            {

                // 插值点值
                float_t tmp = w1[n] * block_img_gray_data.at<uchar>(i + fx[n], j + fy[n]) +
                              w2[n] * block_img_gray_data.at<uchar>(i + cx[n], j + fy[n]) +
                              w3[n] * block_img_gray_data.at<uchar>(i + fx[n], j + cy[n]) +
                              w4[n] * block_img_gray_data.at<uchar>(i + cx[n], j + cy[n]);

                // 计算圆形lbp值
                if (n == 0)
                {
                    last_bit_value = (uint8_t)(tmp > block_img_gray_data.at<uchar>(i, j));
                }
                bit_value = (uint8_t)(tmp > block_img_gray_data.at<uchar>(i, j));
                if (bit_value^last_bit_value)
                    hop_times++;

                last_bit_value = bit_value;
                tmp_lbp_value += bit_value << n;

                if (hop_times > 2) // 剔除掉lbp多余部分，余下主模式
                {
                    tmp_lbp_value = 0;
                }
            }
            // lbp直方图
            modified_lbp[0] += static_cast<float_t>((tmp_lbp_value >> 0) & 0x01);
            modified_lbp[1] += static_cast<float_t>((tmp_lbp_value >> 1) & 0x01);
            modified_lbp[2] += static_cast<float_t>((tmp_lbp_value >> 2) & 0x01);
            modified_lbp[3] += static_cast<float_t>((tmp_lbp_value >> 3) & 0x01);
            modified_lbp[4] += static_cast<float_t>((tmp_lbp_value >> 4) & 0x01);
            modified_lbp[5] += static_cast<float_t>((tmp_lbp_value >> 5) & 0x01);
            modified_lbp[6] += static_cast<float_t>((tmp_lbp_value >> 6) & 0x01);
            modified_lbp[7] += static_cast<float_t>((tmp_lbp_value >> 7) & 0x01);
        }
    }
    float_t modified_lbp_sum = 0;
    // 求和
    for (int i = 0; i < near_point_nums; i++)
    {
        modified_lbp_sum += modified_lbp[i];
    }
    // 归一化
    for (int l = 0; l < near_point_nums; l++)
    {
        modified_lbp[l] /= modified_lbp_sum + std::numeric_limits<float_t>::epsilon();
        lbp.push_back(modified_lbp[l]);
    }
    // 计算lbp值相对于图像大小的密度
    float_t lbp_density = modified_lbp_sum / block_img_area;
    lbp.push_back(lbp_density);
}

//
// 计算产生一幅图像的特征图像
// notes: 
//      the size of feature_image is out_cols x out_rows x out_ddepth
void generateFeatureMap(const cv::Mat &input_image,
                        std::vector<float_t> &feature_image,
                        uint16_t &out_cols,
                        uint16_t &out_rows,
                        uint8_t &out_ddepth,
                        uint8_t stride,
                        uint8_t window_size,
                        bool padding)
{
    if (!feature_image.empty())
        feature_image.clear();

    CV_Assert(stride < window_size);
    if (padding) // 对图像进行填充，使输出图像保持原图大小
    {
        cv::Mat padding_image;
        uint8_t pad_rows = ((stride - 1) * input_image.rows + window_size - stride) / 2; // 填充的行数
        uint8_t pad_cols = ((stride - 1) * input_image.cols + window_size - stride) / 2; // 填充的列数
        cv::copyMakeBorder(input_image, padding_image, pad_rows, pad_rows, pad_cols, pad_cols,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
        out_rows = input_image.rows;
        out_cols = input_image.cols;
    }
    else
    {
        out_rows = (input_image.rows - window_size) / stride;
        out_cols = (input_image.cols - window_size) / stride;
    }
    if (out_ddepth != 0)
    {
        out_ddepth = 0;
    }
    std::vector<float_t> eoh_res, mag_res, lbp_res; // 用于存储临时值的变量
    eoh_res.reserve(9);
    mag_res.reserve(8);
    lbp_res.reserve(9);
    cv::Mat gradientX, gradientY, Mag, Theta;
    cv::Sobel(input_image, gradientX, CV_32F, 1, 0);
    cv::Sobel(input_image, gradientY, CV_32F, 0, 1);
    cv::cartToPolar(gradientX, gradientY, Mag, Theta, true);

    for (int i = 0; i < input_image.rows - window_size; i += stride)
    {
        for (int j = 0; j < input_image.cols - window_size; j += stride)
        {
            calculateEohAndMag(Mag(cv::Range(i, i + window_size), cv::Range(j, j + window_size)),
                                Theta(cv::Range(i, i + window_size), cv::Range(j, j + window_size)),
                               window_size * window_size, eoh_res, mag_res);
            calculateModifiedLbp(input_image(cv::Range(i, i + window_size), cv::Range(j, j + window_size)),
                                 window_size * window_size, lbp_res);

            if (i == 0 && j == 0) // cal only once
            {
                out_ddepth = eoh_res.size() + lbp_res.size() + mag_res.size();
            }
            for (std::vector<float_t>::const_iterator it = eoh_res.begin(); it < eoh_res.end(); ++it)
            {
                feature_image.push_back(*it);
            }
            for (std::vector<float_t>::const_iterator it = mag_res.begin(); it < mag_res.end(); ++it)
            {
                feature_image.push_back(*it);
            }
            for (std::vector<float_t>::const_iterator it = lbp_res.begin(); it < lbp_res.end(); ++it)
            {
                feature_image.push_back(*it);
            }
            eoh_res.clear();
            mag_res.clear();
            lbp_res.clear();
        }
    }
}



void featureMapThreadHandle(threadParams &param)
{
    std::vector<float_t> eoh_res, mag_res, lbp_res; // 用于存储临时值的变量
    eoh_res.reserve(9);
    mag_res.reserve(8);
    lbp_res.reserve(9);
   for (int i = 0; i < param.thread_image.rows - param.windowSize; i += param.stride)
    {
        for (int j = 0; j < param.thread_image.cols - param.windowSize; j += param.stride)
        {
            calculateEohAndMag(param.thread_Mag(cv::Range(i, i + param.windowSize), cv::Range(j, j + param.windowSize)),
                                param.thread_Theta(cv::Range(i, i + param.windowSize), cv::Range(j, j + param.windowSize)),
                               param.windowSize * param.windowSize, eoh_res, mag_res);
            calculateModifiedLbp(param.thread_image(cv::Range(i, i + param.windowSize), cv::Range(j, j + param.windowSize)),
                                 param.windowSize * param.windowSize, lbp_res);

            for (std::vector<float_t>::const_iterator it = eoh_res.begin(); it < eoh_res.end(); ++it)
            {
                param.res.push_back(*it);
            }
            for (std::vector<float_t>::const_iterator it = mag_res.begin(); it < mag_res.end(); ++it)
            {
                param.res.push_back(*it);
            }
            for (std::vector<float_t>::const_iterator it = lbp_res.begin(); it < lbp_res.end(); ++it)
            {
                param.res.push_back(*it);
            }
            eoh_res.clear();
            mag_res.clear();
            lbp_res.clear();
        }
    }
}

void generatFeatureMapMultiThread(const cv::Mat &input_image,
                                  std::vector<float_t> &feature_image,
                                  uint16_t &out_cols,
                                  uint16_t &out_rows,
                                  uint8_t &out_ddepth,
                                  uint8_t stride,
                                  uint8_t window_size,
                                  uint8_t thread_nums)
{
    int cols = input_image.cols;
    int rows = input_image.rows;
    out_cols = (cols - window_size) / stride;
    out_rows = (rows - window_size) / stride;
    out_ddepth = 26; // fixed value
    int threadDeltaRows = rows / thread_nums;
    assert(thread_nums <= 12);
    threadParams params[thread_nums];

    cv::Mat gradientX, gradientY, Mag, Theta;
    cv::Sobel(input_image, gradientX, CV_32F, 1, 0);
    cv::Sobel(input_image, gradientY, CV_32F, 0, 1);
    cv::cartToPolar(gradientX, gradientY, Mag, Theta, true);

    // 设置线程参数
    for (int i = 0; i < thread_nums; i++)
    {
        threadParams p;
        p.thread_id = i + 1;
        p.thread_nums = thread_nums;
        int startIndex = i * threadDeltaRows;
        int endIndex = ((i == thread_nums - 1) ? ((i+1) * threadDeltaRows) : ((i + 1) * threadDeltaRows + window_size));
        p.thread_image = input_image(cv::Range(startIndex, endIndex), cv::Range::all());
        p.thread_Mag = Mag(cv::Range(startIndex, endIndex), cv::Range::all());   // TODO:内存访问不连续
        p.thread_Theta = Theta(cv::Range(startIndex, endIndex), cv::Range::all());
        p.windowSize = window_size;
        p.stride = stride;
        params[i] = p;
    }

    // 创建线程
    std::thread threads[thread_nums];
    for (uint8_t i = 0; i < thread_nums; i++)
    {
        threads[i] = std::thread(featureMapThreadHandle, std::ref(params[i]));
    }
    // join thread
    for (auto& s:threads)
    {
        s.join();
    }
    size_t total = out_ddepth * out_cols * out_rows;
    feature_image.reserve(total);

    // move result
    for (auto &p:params)
    {
        assert(!p.res.empty());
        feature_image.insert(feature_image.end(), p.res.begin(), p.res.end());  // TODO:用右值完成数据的移动
    }
}

} // namespace smoke_adaboost