# OpenSmokeDetetction

![](https://github.com/MingYuanLiu/OpenSmokeDet/blob/master/data/res2.png)
![](https://github.com/MingYuanLiu/OpenSmokeDet/blob/master/data/result.png)

## Introduction 

A real time video smoke detection system. It firstly generates feature map from gray image using eoh and lbp characteristics, then applies statistical calculation on feature map , and finally feeds features  into adaboost classifier in order to produce detection result. 

## Compile Dependencies
- gcc >= 7.5.0
- opencv >= 3.4.1
- cmake
- python3

## Run
<u>Note: This repository is now only run on ubuntu. Later, I will try to transplant windows.</u>
Download the source code. 

```bash
git clone https://github.com/MingYuanLiu/OpenSmokeDet
```
### Training
  1. Generate annotation file from dataset using src/util/writeAnnotation.py. 
   ```bash
   cd src/util
   python writeAnnotation.py -dir your-dataset-directory -- annotation filename.txt
   ```   
  the dataset structure should be like this: 
   ```
   dataset/
      - non/
        - *.jpg
        - ...

      - smoke/
        - *.jpg
        - ...
   ```
2. Get into src/core/main.cpp, and modify the training parameters, including:
      1) annotationFiles -- the annotation file path last step generating; 
      2) saveFeaturesPath -- the save path of features.  if it exits, the system will directly read from this file,  but if not , it will calculate the feature from the images which are recorded  in the annotaion file. 
      3) other model parameters, details in code comments.

3. make new build directory, and run
	```bash
	mkdir build && cd build && cmake .. && make
	```
4.  run training mode, the suffix offeatures file and  model file is .yaml
	```bash
	./smokeAdaboost train
	```
### Detection
Use trained model to detect video or image. 
I provide a trained detection model which is in the model directory.
This model is trained by more than 30,000 pictures, and the positive sample ratio is 1/3.

1. Get into main.cpp, and modify the parameter ['saveModelPath']() into your model path last step training. 
2. Change the video file path or image file path. 
3. Change the ['param.detectorMode']() according to your detection mode(VIDEO or IMAGE)
4. compile and run: `cd build && make`; if you want to use the model provided, you should run:
```bash
mkdir build
cd model
cp model-0.05373.yaml ../build
cd build 
cmake .. && make
```
5. run the detection system `./smokeAdaboost detetcion`
> note: you can change the parameters according to the code comment in main.cpp
> if you want to get my dataset, please contact me by mail: myliu327@zju.edu.cn. 

## TODO
















