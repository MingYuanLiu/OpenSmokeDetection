#include "statistical_features.hpp"

namespace smoke_adaboost
{

// 积分图像的构造函数：
IntegralImage::IntegralImage(const std::vector<float_t> &_feature_image,
                             uint16_t _cols,
                             uint16_t _rows,
                             uint8_t _ddepth)
{
  integral_feature_image = new float_t* [_ddepth]();
  for (int i = 0; i < _ddepth; ++i)
  {
    integral_feature_image[i] = new float_t[(_cols + 1) * (_rows + 1)]();
  }
  cols = _cols + 1;  // 此处为积分图像的cols
  rows = _rows + 1;
  ddepth = _ddepth;
  buildIntegralImage(_feature_image, _cols, _rows, _ddepth);
}

// 析构函数，释放开辟的动态内存空间
IntegralImage::~IntegralImage()
{
  for (int i = 0; i < ddepth; ++i)
  {
    delete[] integral_feature_image[i];
  }
  delete[] integral_feature_image;
}

// 计算积分图像
//
void IntegralImage::buildIntegralImage(const std::vector<float_t> &_feature_image,
                                       uint16_t _cols,
                                       uint16_t _rows,
                                       uint8_t _ddepth)
{
  int index = (_cols + 1) * _ddepth;                                  // 积分特征图像的坐标索引，跳过第一行留白；因为积分图像的第一行和第一列均为0
  float_t *sum = new float_t[_ddepth]();                             // 每一行的求和
  for (std::vector<float_t>::const_iterator it = _feature_image.begin(); it < _feature_image.end(); ++it)
  {
    if ((index / _ddepth) % (_cols + 1) == 0)
    {
      for (int i = 0; i < _ddepth; ++i)
      {
        sum[i] = 0.0; // 在新的一行开始时将每一行的和清零
      }
      index += _ddepth; // 跳过第一列，使积分图像的第一列为0
    }
    sum[index % _ddepth] += *it;
    integral_feature_image[index % _ddepth][index / _ddepth] = integral_feature_image[index % _ddepth][index / _ddepth - _cols - 1] +
                                                              sum[index % _ddepth];
    index++;
  }
}

// 利用积分图像计算区域的积分和
void IntegralImage::getIntegralData(uint16_t x0, uint16_t y0,
                                    uint16_t x1, uint16_t y1,
                                    float_t *integral_data)
{
  float_t tmp_integral_data;
  // handle range of input
  x0=CLIP_UP(x0, rows-1); x0=CLIP_DOWN(x0, 0);
  x1=CLIP_UP(x1, rows-1); x1=CLIP_DOWN(x1, 0);
  y0=CLIP_UP(y0, cols-1); y0=CLIP_DOWN(y0, 0);
  y1=CLIP_UP(y1, cols-1); y1=CLIP_DOWN(y1, 0);

  for (int i = 0; i < this->ddepth; ++i)
  {
    tmp_integral_data = integral_feature_image[i][x0 * cols + y0] + integral_feature_image[i][x1 * cols + y1] -
                        integral_feature_image[i][x0 * cols + y1] - integral_feature_image[i][x1 * cols + y0];
    integral_data[i] = tmp_integral_data;
  }
}

// 打印出积分图像，用于调试
void IntegralImage::debugPrintfIntegralData()
{
  for (int i = 0; i < cols; ++i)
  {
    for (int j = 0; j < rows; ++j)
    {
      std::cout << "(i" << i << ",j" << j << ")";
      for (int k = 0; k < ddepth; ++k)
      {
        std::cout << integral_feature_image[k][i * cols + j] << ",";
      }
      std::cout << std::endl;
    }
  }
}

// 类外函数，用于验证求和结果
void verifySumData(std::vector<float_t> &fm, int x0, int y0, int x1, int y1, float_t *sum)
{
  for (int i = x0; i < x1; ++i)
  {
    for (int j = y0; j < y1; ++j)
    {
      for (int k = 0; k < 26; ++k)
      {
        *(sum + k) += fm.at((i * 21 + j) * 26 + k);
      }
    }
  }
}

// 统计量类的构造函数
BlocksAndStatisticalFeatures::BlocksAndStatisticalFeatures(const std::vector<float_t> &_feature_image,
                                                           uint16_t _cols,
                                                           uint16_t _rows,
                                                           uint8_t _ddepth,
                                                           uint16_t _tiled_block_cut_times,
                                                           uint16_t _ringed_block_cut_times)
{
  this->integral_image = new IntegralImage(_feature_image, _cols, _rows, _ddepth);
  ringed_block_cut_times = _ringed_block_cut_times;
  tiled_block_cut_times = _tiled_block_cut_times;
  feature_image_cols = _cols;
  feature_image_rows = _rows;
  feature_image_ddepth = _ddepth;
  initRingCutInterval();
}

// 析构函数
BlocksAndStatisticalFeatures::~BlocksAndStatisticalFeatures()
{
  if (integral_image != nullptr)
    delete integral_image;
}

// 计算均值、方差、峭度、歪斜度统计量
void BlocksAndStatisticalFeatures::getStatisticalFeatures(std::vector<float_t>& features_vector)
{
  if (!features_vector.empty())
  {
    features_vector.clear();
  }

  cv::Ptr<float_t> tmp_integral_data = new float_t[feature_image_ddepth]();
  cv::Ptr<float_t> tmp_variance_data = new float_t[feature_image_ddepth]();
  cv::Ptr<float_t> tmp_skewness_data = new float_t[feature_image_ddepth]();
  cv::Ptr<float_t> tmp_kurtosis_data = new float_t[feature_image_ddepth]();
  cv::Ptr<float_t> tmp_mean_data = new float_t[feature_image_ddepth]();
  cv::Ptr<float_t> tmp_sum_data = new float_t[feature_image_ddepth]();

  // 获取整个特征图像的和
  integral_image->getIntegralData(0, 0, feature_image_cols, feature_image_rows, tmp_sum_data);

  // 平铺分割部分：
  // 以不同的长宽比将特征图像分成多个矩形块，求取统计量均值、方差、峭度、歪斜率
  uint16_t range_numbers = ceil(sqrt(tiled_block_cut_times)); // 在特征图像的一个维度上切块个数的变化范围
  int delta_cols = 0;
  int delta_rows = 0;
  for (int row_num = 2; row_num < range_numbers + 2; ++row_num) // 列切割个数
  {
    delta_rows = cvRound((float)feature_image_rows / row_num); // 每行的切割间隔
    for (int col_num = 2; col_num < range_numbers + 1; ++col_num) // 行切割个数
    {
      int block_nums = col_num * row_num;                    // 总切割个数
      delta_cols = cvRound((float)feature_image_cols / col_num); // 每列的切割间隔
      for (int i = 0; i < feature_image_ddepth; ++i)
      {
        tmp_mean_data[i] = tmp_sum_data[i] /  block_nums; // 获得均值
      }

      for (int i = 0; i < row_num; ++i)
      {
        for (int j = 0; j < col_num; ++j)
        {
          integral_image->getIntegralData(j * delta_cols, i * delta_rows, (j + 1) * delta_cols, (i+1) * delta_rows, tmp_integral_data);
          for (int i = 0; i < feature_image_ddepth; ++i) // 计算方差、斜度、歪斜率的分子部分求和
          {
            tmp_variance_data[i] += pow((tmp_integral_data[i] - tmp_mean_data[i]), 2);
            tmp_skewness_data[i] += pow((tmp_integral_data[i] - tmp_mean_data[i]), 3);
            tmp_kurtosis_data[i] += pow((tmp_integral_data[i] - tmp_mean_data[i]), 4);
          }
        }
      }
      for (int i = 0; i < this->feature_image_ddepth; ++i) // 计算方差、斜度、歪斜率
      {
        tmp_variance_data[i] /= block_nums;
        tmp_skewness_data[i] = (tmp_skewness_data[i] / block_nums) / (pow(tmp_variance_data[i], 1.5) + std::numeric_limits<float_t>::epsilon());
        tmp_kurtosis_data[i] = (tmp_kurtosis_data[i] / block_nums) / (pow(tmp_variance_data[i], 2) + std::numeric_limits<float_t>::epsilon());
        features_vector.push_back(tmp_mean_data[i]);
        features_vector.push_back(tmp_variance_data[i]);
        features_vector.push_back(tmp_skewness_data[i]);
        features_vector.push_back(tmp_kurtosis_data[i]);
      }
    }
  }

  // 环绕分割部分
  // 以不同的环绕宽度环绕切割特征图像，在分割块的基础上求取统计量均值、方差、歪斜率、峭度
  
  cv::Ptr<float_t> last_ring_integral_data = new float_t[feature_image_ddepth]();
  float block_nums = 0;

  for (std::vector<uint8_t>::const_iterator it = ring_cut_interval.begin(); it < ring_cut_interval.end(); ++it)
  {
    block_nums = floor(feature_image_cols / *it);
    for (int i = 0; i < feature_image_ddepth; ++i)
    {
      tmp_mean_data[i] = tmp_sum_data[i] / block_nums;
    }

    for (int i = 0; i <= block_nums; ++i)
    {
      if (i == block_nums)
      {
        integral_image->getIntegralData(i * (*it), i * (*it), feature_image_cols - i * (*it), feature_image_rows - i * (*it), tmp_integral_data);
      }
      else
      {
        integral_image->getIntegralData(i * (*it), i * (*it), feature_image_cols - i * (*it), feature_image_rows - i * (*it), last_ring_integral_data);
        integral_image->getIntegralData((i + 1) * (*it), (i + 1) * (*it), feature_image_cols - (i + 1) * (*it), feature_image_rows - (i + 1) * (*it), tmp_integral_data);
        for (uint8_t i = 0; i < feature_image_ddepth; i++)
        {
          tmp_integral_data[i] = last_ring_integral_data[i] - tmp_integral_data[i];
        }
      }
      for (uint8_t i = 0; i < feature_image_ddepth; ++i) // 对方差、斜度、歪斜率的分子部分求和
      {
        tmp_variance_data[i] += pow((tmp_integral_data[i] - tmp_mean_data[i]), 2);
        tmp_skewness_data[i] += pow((tmp_integral_data[i] - tmp_mean_data[i]), 3);
        tmp_kurtosis_data[i] += pow((tmp_integral_data[i] - tmp_mean_data[i]), 4);
      }
    }
    for (int i = 0; i < this->feature_image_ddepth; ++i) // 计算方差、斜度、歪斜率
    {
      tmp_variance_data[i] /= block_nums;
      tmp_skewness_data[i] = (tmp_skewness_data[i] / block_nums) / (pow(tmp_variance_data[i], 1.5) + std::numeric_limits<float_t>::epsilon());
      tmp_kurtosis_data[i] = (tmp_kurtosis_data[i] / block_nums) / (pow(tmp_variance_data[i], 2) + std::numeric_limits<float_t>::epsilon());
      features_vector.emplace_back(tmp_mean_data[i]);
      features_vector.emplace_back(tmp_variance_data[i]);
      features_vector.emplace_back(tmp_skewness_data[i]);
      features_vector.emplace_back(tmp_kurtosis_data[i]);
    }
  }

  tmp_integral_data.release();
  tmp_mean_data.release();
  tmp_variance_data.release();
  tmp_skewness_data.release();
  tmp_kurtosis_data.release();
  tmp_sum_data.release();
  last_ring_integral_data.release();
}

template <typename T>
void BlocksAndStatisticalFeatures::deepCopyPtr(T *input, T *output, int nums)
{
  for (int i = 0; i < nums; ++i)
  {
    output[i] = input[i];
  }
}

} // namespace adaboost