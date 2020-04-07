// project: adaboost cpp code
// data: 2020.02
// author: MingYuan Liu
// 在eoh, lbp等特征的基础上计算特征映射图的均值，方差，歪斜度和峭度等统计量:
#ifndef STATISTICAL_FEATURES_HPP
#define STATISTICAL_FEATURES_HPP

#include <opencv2/core.hpp>

#include <iostream>
#include <vector>
#include <math.h>

namespace smoke_adaboost
{

#define CLIP_UP(x, up) (x>up? up:x)
#define CLIP_DOWN(x, down) (x<down? down:x)

// 积分图像
class IntegralImage
{
public:
  // 积分图像的构造函数：
  // 1. 开辟积分图像内存；2. 设置积分图像的长宽以及深度；3.建立积分图
  //
  // Args:
  //      feature_image: 由features.hpp中的generateFeatureMap产生，数据顺序为 ddepth * cols * rows
  //
  // Usage:
  //      上一步调用generateFeatureMap，得到IntegralImage的参数；
  //      构造对象 adaboost::IntegralImage integral(fm, cols, rows, ddepth);
  IntegralImage(const std::vector<float_t> &_feature_image, uint16_t _cols, uint16_t _rows, uint8_t _ddepth);

  // 析构函数，释放开辟的动态内存空间
  ~IntegralImage();

  // 利用积分图像计算区域的积分和
  // Args:
  //    x0: 起始横坐标，范围为[0, cols]
  //    y0: 起始列坐标，范围为)[0, rows]
  //    x1: 终止横坐标，范围为[0, cols]
  //    x1: 终止列坐标，范围为[0, rows]
  //    integral_data: 返回的积分值
  // Notes:
  //      注意这里的cols和rows都表征的是原图的长宽
  //      特征图像上某一区域的和用积分图像计算公式为:
  //      sum(k) = I(k,x0,y0) + I(k,x1,y1) - I(k,x0,y1) - I(k,x1,y0)
  //      计算复杂度为O(4)
  void getIntegralData(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, float_t *integral_data);

  uint16_t getRows() const { return this->rows; }
  uint16_t getCols() const { return this->cols; }

  // 打印出积分图像，用于调试
  void debugPrintfIntegralData();

private:

  // 计算积分图像
  //
  // Args:
  //      feature_image:一维数组，保存特征数据；存储顺序为ddepth x cols x rows
  //      cols: 输入特征图像的列
  //      rows: 输入特征图像的行
  //
  // 实现算法：
  //      输入的特征图像维度为 ddepth * cols * rows，求积分图像就是将原图左上角的像素积分值作为积分图像的坐标值。
  //      为了降低计算复杂度，我们产用增量式的方法来计算积分图像，即在上一次计算的积分图像上加上当前坐标的像素值得当前的积分图像值。
  //      具体表达式为: I(x,y) = I(x, y-1) + I(x-1, y) - I(x-1,y-1) + ii(x,y) 其中I为积分图像，ii为原图像值
  //      我们可以进一步将其简化为：I(x,y) = I(x,y-1) + rowssum(x) 也就是上一行的积分图像加上当前行的积分值。
  //
  //      程序中对原特征图做一次循环，index表示积分图像上的坐标索引，由index % ddepth来控制当前求和的通道数
  //      index / ddepth 表示每个通道的二维坐标， index / ddepth -cols - 1表示上一行
  void buildIntegralImage(const std::vector<float_t> &_feature_image, uint16_t _cols, 
                          uint16_t _rows, uint8_t _ddepth);
  
  float_t** integral_feature_image; // 积分特征图，维度为 ddepth * cols * rows
  uint16_t cols;                    // 积分特征图像的二维列数
  uint16_t rows;                    // 积分特征图像的二维行数
  uint16_t ddepth;                  // 特征图像的通道数
};

// 验证积分和
void verifySumData(std::vector<float_t> &fm, int x0, int y0, int x1, int y1, float_t *sum);

// 划分块，并求统计量
class BlocksAndStatisticalFeatures
{
public:
  // 统计量类的构造函数
  // Args:
  //    feature_image: 输入是积分特征图像
  //    cols: 特征图像的列数
  //    rows: 特征图像的行数
  //    ddepth: 特征图像的深度
  //    tiled_block_cut_times: 平铺切割次数
  //    ringed_block_cut_times: 环绕切割次数
  //
  BlocksAndStatisticalFeatures(const std::vector<float_t>& _feature_image, uint16_t _cols, uint16_t _rows, 
                              uint8_t _ddepth, uint16_t tiled_block_cut_times, uint16_t ringed_block_cut_times);

  // 在析构函数中释放掉积分图像的内存空间
  ~BlocksAndStatisticalFeatures();

  //
  // 计算均值、方差、峭度、歪斜度统计量
  //    此部分函数分为两部分： 
  //      1） 平铺分割部分，将特征图像按不同的长宽比分割不同的矩形，长宽比由切割数确定
  //      2） 环绕分割部分，将特征图像以部分的间隔环绕切割特征图像
  //Args:
  //  features_vector: 待求特征向量的引用，用于获取特征值
  //
  void getStatisticalFeatures(std::vector<float_t> &features_vector);

  // 工具函数，拷贝指针所指向的内存空间
  template <typename T>
  void deepCopyPtr(T *input, T *output, int nums);

  
private:
  IntegralImage *integral_image;   // 特征图的积分图像
  uint16_t tiled_block_cut_times;  // 平铺切割的次数
  uint16_t ringed_block_cut_times; // 环绕切割的次数
  uint16_t feature_image_cols;
  uint16_t feature_image_rows;
  uint16_t feature_image_ddepth;
  std::vector<float_t> statistical_features_vector; // 统计量
  std::vector<uint8_t> ring_cut_interval;
  inline void initRingCutInterval() 
  {for (int i=3;i< 3+ringed_block_cut_times;i++) ring_cut_interval.emplace_back(i);}
};

} // namespace smoke_adaboost

#endif