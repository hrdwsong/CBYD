## Introduction

This repository serves as the official source code for the paper titled "Explicit Location-Label-Guided Foreground Feature Optimization Learning for Few-Shot Classification"

Currently, the paper is under review. We will make the source code publicly available upon the publication of the paper.

<div align="center"><img src="assets/arch.bmp" width="800"></div>

## Quick Start

**1. Check Requirements**
* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 2.0 & [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch version.
* CUDA 12.1
* GCC >= 4.9

**2. Build CBYD**
* Clone Code
  ```angular2html
  git clone https://github.com/hrdwsong/CBYD
  cd CBYD
  ```
* Create a virtual environment (optional)
  ```angular2html
  conda create -n cbyd python=3.8
  conda activate cbyd
  ```
* Install PyTorch 2.3.0 with CUDA 12.1 
  ```shell
  pip install torch==2.3.0+cu121 torchvision==0.18.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
  ```
* Install Detectron2
  ```angular2html
  pip install detectron2==0.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
  ```
  - If you use other version of PyTorch/CUDA, check the latest version of Detectron2 in this page: [Detectron2](https://github.com/facebookresearch/detectron2/releases). 
* Install other requirements. 
  ```angular2html
  pip install -r requirements.txt
  ```

**3. Prepare Data and Weights**
* Data Preparation
  - We evaluate our models on five datasets:

    |     Dataset     |  Size   | 
    |:---------------:|:-------:|
    |  mini-ImageNet  | 2.88GB  |
    | tiered-ImageNet | 28.2GB  |
    |       CUB       | 1.16GB  |
    |  Stanford Dogs  | 0.753GB |
    |  Stanford Cars  | 1.85GB  |
  - Unzip the downloaded data-source to `datasets` and put it into your project directory:
    ```angular2html
      ...
      datasets
        | -- miniImageNet (miniImageNet/data/*.jpg, miniImageNet/miniImageNet-split, miniImageNet/*.json)
        | -- tieredImageNet
        | -- CUB
        | -- Dogs
        | -- Cars
      cbyd
      tools
      ...
    ```
* Weights Preparation
  - We use the imagenet pretrain weights to initialize our model. Download the same models from here: [GoogleDrive](https://drive.google.com/file/d/1rsE20_fSkYeIhFaNU04rBfEDkMENLibj/view?usp=sharing) [BaiduYun](https://pan.baidu.com/s/1IfxFq15LVUI3iIMGFT8slw)
  - The extract code for all BaiduYun link is **0000**
  - The pretrained weights are completely align with [DeFRCN](https://github.com/er-muyue/DeFRCN)

**4. Training and Evaluation**

For ease of training and evaluation over multiple runs, we integrate the whole pipeline of few-shot object detection into one script `*.bat`, including base pre-training and novel-finetuning.
* To reproduce the results on mini-ImageNet, `EXP_NAME` can be any string (e.g cbyd, or something):
  ```angular2html
  miniin_train.bat EXP_NAME
  ```
* Please read the details of pipeline in `*.bat`, you need change `IMAGENET_PRETRAIN*` to your path.

## Results on mini-ImageNet Benchmark

* Few-shot Object Detection

| Method            | Backbone | mini-ImageNet 5-way 1-shot | mini-ImageNet 5-way 5-shot | tiered-ImageNet 5-way 1-shot | tiered-ImageNet 5-way 5-shot | Avg. |
|:------------------| :--- | :---: | :---: | :---: | :---: | :---: |
| ProtoNet          | ResNet-12 | 60.37±0.83 | 78.02±0.57 | 65.65±0.92 | 83.40±0.65 | 71.86 |
| RFS               | ResNet-12 | 64.82±0.60 | 82.14±0.43 | 71.52±0.69 | 86.03±0.49 | 76.13 |
| S2M2              | WRN-28-10 | 64.93±0.18 | 83.18±0.11 | 73.71±0.22 | 88.59±0.14 | 77.60 |
| FRN               | ResNet-12 | 66.45±0.19 | 82.83±0.13 | 71.16±0.22 | 86.01±0.15 | 76.61 |
| DeepEMD           | ResNet-12 | 65.91±0.82 | 82.41±0.56 | 71.16±0.87 | 86.03±0.58 | 76.38 |
| FewTURE           | ViT-Small | 68.02±0.88 | <u>84.51±0.53</u> | 72.96±0.92 | 86.43±0.67 | 77.98 |
| UniSiam           | ResNet-18 | 64.10±0.36 | 82.26±0.25 | 67.01±0.39 | 84.47±0.28 | 74.46 |
| SPRM              | ResNet-12 | 66.35±0.34 | 82.24±0.27 | 70.70±0.33 | 85.40±0.25 | 76.17 |
| A-MET             | ResNet-12 | 68.47±0.51 | 80.89±0.33 | 74.08±0.54 | 84.91±0.35 | 77.09 |
| ESPT              | ResNet-12 | 68.36±0.19 | 84.11±0.12 | 72.68±0.22 | 87.49±0.14 | 78.16 |
| DFR               | ResNet-12 | 68.12±0.81 | 82.79±0.56 | 72.38±0.95 | 86.00±0.61 | 77.32 |
| CFS-space         | ResNet-12 | 66.42±0.84 | 82.69±0.59 | 72.07±0.99 | 86.36±0.67 | 76.89 |
| CFRN              | ResNet-12 | 67.49±0.46 | 84.30±0.31 | 73.37±0.50 | 87.65±0.36 | 78.20 |
| ALFA              | ResNet-12 | 66.61±0.28 | 81.43±0.25 | 70.29±0.40 | 86.17±0.35 | 76.13 |
| MCS               | ResNet-12 | 67.47±0.31 | 82.49±0.57 | 73.13±0.33 | 87.56±0.60 | 77.66 |
| LASTSHOT          | ResNet-12 | 67.35±0.20 | 82.58±0.14 | 72.43±0.23 | 85.82±0.16 | 77.05 |
| TRSN              | ResNet-12 | 67.14±0.20 | 84.14±0.15 | 72.16±0.37 | 86.90±0.24 | 77.59 |
| FA-adapter        | ResNet-12 | 67.79±0.42 | 83.24±0.28 | 72.86±0.50 | 86.60±0.32 | 77.62 |
| **CBYD (Ours)**   | F-RCNN | <u>68.85±1.15</u> | 84.45±0.84 | <u>74.53±1.02</u> | <u>89.50±0.91</u> | <u>79.33</u> |
| **CBYD++ (Ours)** | F-RCNN | **85.63±0.99** | **94.80±0.47** | **90.13±0.71** | **95.67±0.51** | **91.56** |
  


## Acknowledgement
This repo is developed based on [DeFRCN](https://github.com/ucbdrive/few-shot-object-detection) and [Detectron2](https://github.com/facebookresearch/detectron2). Please check them for more details and features.


