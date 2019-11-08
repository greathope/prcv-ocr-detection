
# prcv-ocr-detection
prcv [面向自动阅卷的OCR技术挑战赛](http://vipl.ict.ac.cn/homepage/prcv2019-OCR-challenge/index.htm)-检测任务一等奖代码

## 环境配置
* 参见[mmdetection](https://github.com/open-mmlab/mmdetection)的安装

## 数据下载
* [训练数据](https://pan.baidu.com/s/1LiM-MfBMdi7gLE6Ze4NcvA), 提取码g1gd
* [验证数据](https://pan.baidu.com/s/1PPXyOZdxEhDPxi3jB3o0rw ), 提取码nw5q
  * [验证数据标签](https://pan.baidu.com/s/1KywBdn0RkTJ6O9WuaSCNcA ), 提取码iw8i

## 训练
* python tools/ocr/make_dataset.py 
  * 会生成coco格式的训练集
* bash train.sh
  * 会按configs里的配置文件进行训练

## 预测
* python tools/ocr/make_dataset.py 
  * 会生成coco格式的预测集
* bash predict.sh
* python tools/ocr/make_submit.py 
    * 会生成提交格式

> 注意，以上的路径需要自行修改

## 性能

| 模型|阈值|box数目|分数 
|---|---|---|---|
| faster x101_32x8d | 0.9|6529|0.6754| 
| faster x101_32x8d| 0.95|6169|0.6526967198| 
| faster x101_32x8d| 0.85|6703|0.670| |
| cascade x101_32x8d| 0.85|6467|0.7069429582|
| cascade x101_32x8d| 0.8|6608|0.7121319398| 
| cascade dcn x101_32x8d| 0.8|6626|0.7250362477| 
