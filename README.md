# Optimization and quantization of a classification model for thoracic diseases using OpenVINO

## Introduction

Medical imaging is an indispensable technology for modern medicine, and the application of deep learning is also spreading to this field. 
A typical example is image reading using medical images such as X-rays, CT, MRI, etc.
By constructing a model that estimates the name of the disease and the location of the disease using convolutional networks (CNNs), etc., for medical images,
it is expected to reduce the burden on the image reading physician, equalize the diagnostic criteria,
and realize diagnosis that exceeds human capabilities, although diagnosis through reading is still the responsibility of the physician.

On the other hand, there are several challenges in deep learning for medical images. 
One is the collection and labeling of medical images, which requires collecting as many images as necessary for training, 
considering patient privacy, and attaching high-quality labels for training. 
In 2017, the National Institutes of Health (NIH) released a large dataset called ChestX-ray14, described below. 
Other medical institutions have also begun to release medical image datasets with case labels, and the environment for developing models for clinical use is now in place. 
*CheXNet*, which I used in this repository, is one of the models proposed in this study.

Another issue is that inference models with high accuracy require a lot of computational cost. 
Even if a model with high accuracy is obtained, if the computational cost of inference is too high, it will be difficult to introduce into the medical field.
For widespread adoption, it is important to realize inference for medical images at a practical computational cost.

One of the most effective ways to reduce the computational cost is to optimize and quantize the model. 
If computational costs can be reduced, processing can be done on existing medical devices (edge computing), lowering the barrier to adoption.

In this repository, I perform model optimization and quantization using OpenVINO on *CheXNet*, a trained model proposed for the ChestX-ray14 dataset, to verify the reduction of computational cost in inference.

## Dataset

I used ChestX-ray14 as a dataset. This dataset is a chest X-ray image dataset provided by NIH. 
112,120 chest X-ray images of 30,805 patients are associated with multiple labels corresponding to each image from 14 different diseases. 
This data set is divided into training set (70%), validation set (10%), and test set (20%).

- [ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) 

## Model

As a model, I use an improved version of *CheXNet*, a model proposed by Rajpurkar et al. that takes chest X-ray images as input and performs multi-label classification for each chest disease. 
The structure of the neural network is based on DenseNet121, and the output layer for classification into 14 diseases is added to the trained model by ImageNet, and fine-tuning is performed by ChestX-ray14.

- [CheXNet](https://arxiv.org/abs/1711.05225)

## Usage

First, run the script to download the dataset.

```bash
bash batch_download.sh
```

The next step is to set up a Python environment to optimize and quantize the model. This procedure is described in `run.sh`.

```bash
bash run.sh
```

The following scripts are used to perform inference on the PyTorch, FP32 optimized and INT8 quantized models, respectively.

```bash
python main.py --mode torch
python main.py --mode fp32
python main.py --mode int8
```

## Results

I used a trained *CheXNet* model to perform multi-label inference on a test split and compared the time taken to perform the inference. 
For the inference, I performed 10 crop runs for each sample and determined the predicted label from the average of the results.
As a result, I obtained 6.2 times performance improvement in the optimization of FP32 models by OpenVINO and 2.6 times performance improvement in the quantization to INT8 models. 
Although performance-first quantization was used for the quantization, the ROC-AUC evaluation showed that the average AUC for FP32 was 0.843, while the average AUC for INT8 was 0.842, indicating that even with quantization, there was only a slight decrease in accuracy and no practical problem.

## Contributions

The dataset `ChestX-ray14` was suggested by Noriaki Sato (Kyoto University).
The annotation converter for the dataset was developed by Hiroshi Ouchiyama (Intel). 
