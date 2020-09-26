<<<<<<< HEAD
# OpenSmokeDetection:  A real-time smoke detector
# OpenSmokeDetetction
[toc]

![](https://github.com/MingYuanLiu/OpenSmokeDetection/blob/master/data/res2.png)

![](https://github.com/MingYuanLiu/OpenSmokeDetection/blob/master/data/result.png)

[English](README_en.md)

OpenSmokeDetction是一个实时检测烟雾的算法；算法核心思想是使用梯度直方图和局部二进制模式特征 + adaboost提升算法对烟雾图片进行识别分类，区分出有烟和无烟。

具体算法可参考：[A double mapping framework for extraction of shape-invariant features based on multi-scale partitions with AdaBoost for video smoke detection](https://www.sciencedirect.com/science/article/pii/S0031320312002786)

本项目对上述算法做了大量工程实现和优化工作：

1. 本项目实现了上述算法，并达到和作者提供的测试样例类似的识别效果；
2. 论文中仅讨论了对一个图像小块进行检测，但是对一幅图片或视频帧却没有一个完整的算法流程；我在本项目中对其进行了拓展，算法的应用价值得到提升；具体方法如下：a) 计算整幅图片或视频帧的特征图；b) 将原图的待检测框映射至特征图上；c) 使用检测框在特征图上滑动，计算对应的统计特征并送入adaboost分类中完成识别；如图1所示
3. 使用了积分图像技术加速特征图像的计算；
4. 使用特征图像加速图片的检测和特征的计算速度；

## 使用方法

**依赖环境：cmake、opencv、python(训练时用于产生样本标签文件)**

### 下载源码

```bash
git clone https://github.com/MingYuanLiu/OpenSmokeDetection
```
### 编译
```bash
mkdir build
cmake .. && make -j4 && make install
```

编译完成后会在`build`文件夹下生成可执行文件`smokeAboost`，头文件和链接库将保存到install文件夹。

如果需要画出预测结果，那么要添加预编译参数：

```bash
cmake -DDRAW_RESULTS=ON .. && make -j4 && make install
```

使用交叉编译工具链:

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=aarch64-himix100-linux.cmake .. && make -j4 && make insatll
```

这里以海思aarch64工具链为例，使用者可以自行替换成自己的工具链。

### 使用方法

#### 训练

1. 准备数据集（本人收集了3W张左右有烟和无烟的训练样本，如有需要请联系我：myliu327@zju.edu.cn）
2. 将数据集的文件以如下组织形式存放在文件系统中：

>dataset/
>
>​      \- non/
>
>​        \- *.jpg
>
>​        \- ...
>
>
>
>​      \- smoke/
>
>​        \- *.jpg
>
>​        \- ...

3. 生成训练集的标签文件:

```bash
cd src/util
python writeAnnotation.py -dir dataset-directory --annotation filename.txt
```

说明：命令中的`dataset-directory`就是第二步中的数据集文件夹`dataset`，名字可随意取。

4. 进入`main.cpp`根据注释修改训练参数（一般情况下多数参数可保持默认）
5. 重新编译`make smokeAdaboost`
6. 运行`./smokeAdaboost train`开始训练

**说明：我已经在model目录下提供了一个训练好的模型，可以拿来直接使用。**

#### 检测

1. 直接使用检测程序：

   `./smokeAdaboost detetcion <model_path> <video | image> <file_path>`

   `video | image`选择输入的文件类型是视频还是图片

2. 加入到程序中: 

```c++
#include "model.hpp"
#include "detector.hpp"

/* Detection */
{
	smoke_adaboost::Detector::detectorParams param; // 检测参数
	param.modelPath =  "your model path"; // 模型保存路径
	param.videoPath = "your video path";
	param.imagePath = "your image path"
	param.detectorModal = smoke_adaboost::Detector::VIDEO; // OR   smoke_adaboost::Detector::IMAGE
	vector<uint16_t> w = {50}; // 检测窗口大小，默认为50pixel
	param.detWindowSize = w; 
	param.threadNums = 4; // 线程数，默认为4；可调，但建议为2的倍数
	Detector detector(param);
	detector.run();
}

/*获取检测结果*/
{
    // 如果是视频，res 保存了最近5帧的检测结果；
    const std::vector<std::vector<smoke_adaboost::Detector::predictRes> >& res = detector.get_predictions();
}
```

