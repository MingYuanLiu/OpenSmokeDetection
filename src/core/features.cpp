#include "features.hpp"

namespace smoke_adaboost
{
// 计算eoh和mag特征
void calculateEohAndMag(const cv::Mat &block_img_gray_data,
                        uint16_t block_img_area,
                        std::vector<float_t> &eoh_histogram,
                        std::vector<float_t> &mag_histogram,
                        uint8_t eoh_bin,
                        uint8_t mag_bin)
{

    cv::Mat gradient_x, gradient_y;
    cv::Sobel(block_img_gray_data, gradient_x, CV_32F, 1, 0); // 计算微分图像
    cv::Sobel(block_img_gray_data, gradient_y, CV_32F, 0, 1);

    cv::Mat block_img_magnitude, block_img_theta;                                        // 微分图像对应的幅相图
    cv::cartToPolar(gradient_x, gradient_y, block_img_magnitude, block_img_theta, true); // 微分图像转化为幅相图

    float epsilon = 2e-21;
    //计算eoh
    // 根据eoh_bin来设置数组长度
    float eoh_[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float mag_[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float_t sum_eoh = 0.0;
    // 找到magnitude中的最大值
    double m_max = 0.0;
    double m_min = 0.0;
    cv::minMaxIdx(block_img_magnitude, &m_min, &m_max);

    if (!eoh_histogram.empty())
        eoh_histogram.clear();

    if (!mag_histogram.empty())
        mag_histogram.clear();

    if (m_max == 0)
    {
        m_max = epsilon;
    }
    else
    {
        m_max = m_max / 2; // 在求幅值直方图中使用
    }

    int theta_img_cols = block_img_magnitude.cols;
    int theta_img_rows = block_img_theta.rows;
    const float_t *theta_img_data_ptr;
    const float_t *mag_img_data_ptr;

    if (block_img_theta.isContinuous())
    {
        theta_img_cols *= theta_img_rows;
        theta_img_rows = 1;
    }

    for (int i = 0; i < theta_img_rows; i++)
    {
        theta_img_data_ptr = block_img_theta.ptr<float_t>(i);
        mag_img_data_ptr = block_img_magnitude.ptr<float_t>(i);
        for (int j = 0; j < theta_img_cols; j++)
        {
            // eoh计算部分
            float_t theta = *theta_img_data_ptr;
            if (theta > 180)
            {
                theta -= 180; // 将0～360度的相角转化为0～180度的相角
            }
            if ((theta >= (180.0f / eoh_bin) * 0) &&
                (theta <= (180.0f / eoh_bin) * (1)))
            {
                eoh_[0] += (float)(*mag_img_data_ptr);
            }
            else if ((theta > (180.0f / eoh_bin) * 1) &&
                     (theta <= (180.0f / eoh_bin) * (2)))
            {
                eoh_[1] += (float)(*mag_img_data_ptr);
            }
            else if ((theta > (180.0f / eoh_bin) * 2) &&
                     (theta <= (180.0f / eoh_bin) * (3)))
            {
                eoh_[2] += (float)*mag_img_data_ptr;
            }
            else if ((theta > (180.0f / eoh_bin) * 3) &&
                     (theta <= (180.0f / eoh_bin) * (4)))
            {
                eoh_[3] += (float)*mag_img_data_ptr;
            }
            else if ((theta > (180.0f / eoh_bin) * 4) &&
                     (theta <= (180.0f / eoh_bin) * (5)))
            {
                eoh_[4] += (float)*mag_img_data_ptr;
            }
            else if ((theta > (180.0f / eoh_bin) * 5) &&
                     (theta <= (180.0f / eoh_bin) * (6)))
            {
                eoh_[5] += (float)*mag_img_data_ptr;
            }

            else if ((theta > (180.0f / eoh_bin) * 6) &&
                     (theta <= (180.0f / eoh_bin) * (7)))
            {
                eoh_[6] += (float)*mag_img_data_ptr;
            }
            else if ((theta > (180.0f / eoh_bin) * 7) &&
                     (theta <= (180.0f / eoh_bin) * (8)))
            {
                eoh_[7] += (float)*mag_img_data_ptr;
            }
            // mag计算部分
            uint16_t mq = uint16_t(*mag_img_data_ptr / (m_max / mag_bin)) > mag_bin - 1 ? mag_bin - 1 : uint16_t(*mag_img_data_ptr / (m_max / mag_bin));
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

            mag_img_data_ptr++;
            theta_img_data_ptr++;
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
    /**
     * 程序逻辑:
     * 先对使用双线性插值对圆周上的像素点进行插值，然后计算出以图像中每个像素为中心的圆形lbp值，
     * 然后根据其二进制的01跳变次数来调整值，并统计直方图，最后归一化。
     * 对于直方图的密度特征，求出其和然后除以图像面积。
    */
    std::array<float_t, 8> modified_lbp = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // lbp直方图
    float x[near_point_nums] = {0};
    float y[near_point_nums] = {0};

    // 先计算出正余弦再查表
    for (int i = 0; i<near_point_nums; i++)
    {
        x[i] = static_cast<float_t>(radius) * cos(2 * CV_PI *
                                                  (i / static_cast<float_t>(near_point_nums)));
        y[i] = static_cast<float_t>(radius) * -sin(2 * CV_PI *
                                                   (i / static_cast<float_t>(near_point_nums)));

    }

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

                // 对x,y取整
                int fx = static_cast<int>(floor(x[n])); // 对x向下取整
                int fy = static_cast<int>(floor(y[n])); // 对y向下取整
                int cx = static_cast<int>(ceil(x[n]));  // 对x向上取整
                int cy = static_cast<int>(ceil(y[n]));  // 对y向上取整
                // printf("%d,%d,%d,%d\n", fx,fy,cx,cy);

                // 双线性插值参数
                float_t tx = x[n] - fx;
                float_t ty = y[n] - fy;
                float_t w1 = (1 - tx) * (1 - ty); // 双线性插值的周围四个点的权重
                float_t w2 = tx * (1 - ty);
                float_t w3 = (1 - tx) * ty;
                float_t w4 = tx * ty;

                // 插值点值
                float_t tmp = w1 * static_cast<float_t>(block_img_gray_data.at<uchar>(i + fx, j + fy)) +
                              w2 * static_cast<float_t>(block_img_gray_data.at<uchar>(i + cx, j + fy)) +
                              w3 * static_cast<float_t>(block_img_gray_data.at<uchar>(i + fx, j + cy)) +
                              w4 * static_cast<float_t>(block_img_gray_data.at<uchar>(i + cx, j + cy));

                // 计算圆形lbp值
                if (n == 0)
                {
                    last_bit_value = ((tmp > block_img_gray_data.at<uchar>(i, j)) &&
                                      (abs(tmp - block_img_gray_data.at<uchar>(i, j)) > std::numeric_limits<float_t>::epsilon()));
                }
                bit_value = ((tmp > block_img_gray_data.at<uchar>(i, j)) &&
                             (abs(tmp - block_img_gray_data.at<uchar>(i, j)) > std::numeric_limits<float_t>::epsilon()));
                if (bit_value != last_bit_value)
                {
                    hop_times++;
                }
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
    // 特征向量维数: eoh + ed + mag + lbp + lbp_bit_density + iDensity + sDensity = 8+1+8+8+1+1+1 = 28
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

    for (int i = 0; i < input_image.cols - window_size; i += stride)
    {
        for (int j = 0; j < input_image.rows - window_size; j += stride)
        {
            calculateEohAndMag(input_image(cv::Range(j, j + window_size), cv::Range(i, i + window_size)),
                               window_size * window_size, eoh_res, mag_res);
            calculateModifiedLbp(input_image(cv::Range(j, j + window_size), cv::Range(i, i + window_size)),
                                 window_size * window_size, lbp_res);

            if (i == 0 && j == 0) // run only once
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

} // namespace smoke_adaboost