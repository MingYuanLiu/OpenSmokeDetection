// project: adaboost cpp code
// data: 2020.02
// author: MingYuan Liu
// 计算特征:eoh, mag, lbp, bit, Id, Sd
#ifndef FEATURES_HPP
#define FEATURES_HPP

/***************************** eoh, mag *********************************/

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <math.h>
#include <vector>
#include <array>
#include <thread>
#include <sys/time.h>

namespace smoke_adaboost{

// 计算eoh和mag特征
//
// Args:
//      block_img_gray_data: Mat format，单通道灰度图像
//      block_img_area: 图像面积
//      eoh_histogram: eoh的计算结果
//      mag_histogram: mag的计算结果
//      eoh_bin: eoh直方图的位数
//      mag_bin: mag直方图的位数
// Usage:
//      cv::Mat img_gray = cv::imread("...");
//      std::vector<typename> eoh_res, mag_res;
//      calculateEohAndMag(img_gray, img_gray.cols*img_gray.rows, eoh_res,mag_res);
//
// 实现算法：
//      使用opencv api计算出传入图像块的sobel剃度, 然后将其转化为幅值和相位，保存在类变量中；
//      对于相位满足条件：
//      theta_block >= (180. / eoh_bin) * k) & (theta_block <= (180. / eoh_bin) * (k + 1)
//      的位置处保持幅值不变，否则至为0。其中k是从1到eoh_bin的取直。
//      对上述得到的图像求和，得到直方图的一个bin的值。
//      求出整个直方图，归一化后即为eoh histogram
//
//      对于mag直方图特征的计算，找到mag中最大值的一半m_max，然后将幅值除以m_max / mag_bin,并取整
//      最后统计每个bin值出现的次数。
//
void calculateEohAndMag(const cv::Mat &Mag, const cv::Mat &Theta, uint16_t block_img_area,
                        std::vector<float_t>& eoh_histogram,std::vector<float_t>& mag_histogram,uint8_t eoh_bin=8,
                        uint8_t mag_bin=8);

 // 计算lbp(局部二进制模式)特征
//
// Args:
//      block_img_gray_data: Mat format，单通道灰度图像
//      block_img_area: 图像面积
//      lbp: lbp直方图结果
//      radius: 圆形lbp计算半径
//      near_point_nums: 圆形lbp的相邻点数
//      lbp_bin: lbp直方图的位数
//
// Usage:
//      cv::Mat img_gray = cv::imread("...");
//      std::vector<typename> lbp_res;
//      calculateEohAndMag(img_gray, img_gray.cols*img_gray.rows, lbp_res;    
void calculateModifiedLbp(const cv::Mat &block_img_gray_data, uint16_t block_img_area, std::vector<float_t> &lbp, 
                          uint8_t radius=1, uint8_t near_point_nums=8, uint8_t lbp_bin=8);

// 
// 计算产生一幅图像的特征图像
// 
// Args:
//      input_image: 输入单通道灰度图像
//      fearture_image: 产生的特征图像
//      out_cols: 输出特征图像的宽度
//      out_rows: 输出特征图像的长度
//      out_ddepth: 输出 特征图像的深度
//      stride: 窗口移动的步长
//      window_size: 窗口大小
//      padding: 是否用0填充原图使特征图像与原图大小相等
//      
// Usage:
//      传入一张灰度图片，产生 26 * out_cols * out_rows 的一维特征向量
//      adaboost::generateFeatureMap(cv.mat, feature_map, )
//     
void generateFeatureMap(const cv::Mat &input_image, std::vector<float_t> &feature_image, uint16_t& out_cols,
                        uint16_t& out_rows, uint8_t& out_ddepth,  uint8_t stride=2, uint8_t window_size=8, bool padding=false);


// 多线程函数参数
struct threadParams
{
    uint8_t thread_id;
    uint8_t thread_nums;
    uint8_t windowSize;
    uint8_t stride;
    cv::Mat thread_image;
    cv::Mat thread_gradX;
    cv::Mat thread_gradY;
    cv::Mat thread_Mag;
    cv::Mat thread_Theta;
    std::vector<float> res;
};

// 计算特征图像的多线程版本
// 相对于单线程版本只多了一个线程数 `thread_nums`变量，用于设置运行线程数
void generatFeatureMapMultiThread(const cv::Mat &input_image, std::vector<float_t> &feature_image,
                                  uint16_t &out_cols, uint16_t &out_rows, uint8_t &out_ddepth,
                                  uint8_t stride, uint8_t window_size, uint8_t thread_nums);

} // namespace smoke_adaboost


#endif