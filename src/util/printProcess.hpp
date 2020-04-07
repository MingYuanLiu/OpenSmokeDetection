#ifndef PRINTPROCESS_HPP
#define PRINTPROCESS_HPP

#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>

// 打印出程序运行进度
void printProcess(float value, const std::string& infro)
{
    if (!(value >= 0 && value <= 1))
        CV_Error(cv::Error::Code::StsOutOfRange, "Input Value must be range from 0 to 1. ");

    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2);
    std::string tag = "[ " + infro + ":]" + std::string((int)value * 10, '*') + "[";
    std::cout << std::flush << '\r' << tag << value * 100 << "%]";
}

#endif