# MobileNetV2 Experiments on CIFAR-10

This repository contains the implementation and experimentation of the MobileNetV2 architecture on the CIFAR-10 dataset. The project explores various configurations and modifications to improve the performance of the model.

## Table of Contents

- [Introduction](#introduction)
- [Baseline Model](#baseline-model)
- [Implementation](#implementation)
- [Experimental Setup](#experimental-setup)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Architectural Modifications](#architectural-modifications)
- [Results](#results)
- [Challenges and Learnings](#challenges-and-learnings)
- [Future Work](#future-work)
- [References](#references)

## Introduction

In this project, we aimed to explore various configurations of the MobileNetV2 architecture by implementing and evaluating its performance on the CIFAR-10 dataset for a classification task. The primary goals were to understand how the architecture responds to different settings and to identify potential areas for improvement.

## Baseline Model

Our baseline model uses the original MobileNetV2 architecture, trained on the CIFAR-10 dataset. The baseline model's performance served as a reference point for subsequent experiments and modifications.

## Implementation

The implementation of MobileNetV2 was done using PyTorch. The key components of the implementation include:

- **ConvBNActivation**: Defines a block with a convolutional layer, batch normalization, and ReLU6 activation.
- **InvertedResidualBlock**: Implements an inverted residual block, a key component of MobileNetV2.
- **CustomMobileNetV2**: Consists of the initial convolutional layers, inverted residual blocks, final convolutional layers, and the classifier.

### Code Files

- `Implementation.ipynb`: Contains the baseline implementation.
- `experiment1.ipynb`: Contains hyperparameter tuning experiments.
- `experiment2.ipynb`: Contains architectural modifications experiments.
- `experiment_test.ipynb`: Contains additional architectural modifications.
- `Modification_width.ipynb`: Contains experiments on width scaling.

## Experimental Setup

### Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images.

### Hyperparameters

For the experiments, various hyperparameters were tuned, including learning rates, number of epochs, optimizer settings, regularization techniques, and data augmentation strategies.

### Evaluation Metrics

The primary evaluation metric used was top-1 accuracy, which is standard for CIFAR-10 evaluations.

## Hyperparameter Tuning

We conducted hyperparameter tuning on the baseline model to optimize learning efficiency and generalization capability. Various configurations were tested, and the best-performing configurations were used for further experiments.

### Example Configuration

```python
config = {
    "learning_rate": 0.001,
    "epochs": 30,
    "optimizer": "Adam",
    "batch_size": 64,
    "scheduler_step_size": 15,
    "scheduler_gamma": 0.5
}
```

## Architectural Modifications

After establishing a baseline, we explored several architectural modifications, including changes to the expansion ratio, kernel size, activation functions, and the addition of Squeeze-and-Excitation (SE) blocks.

### Key Experiments

- **Modification of Expansion Ratio**
- **Changes in Kernel Size**
- **Experimenting with Different Activation Functions (LeakyReLU, SiLU, Hardswish, Hardsigmoid)**
- **Integration of SE Blocks**
- **Width Scaling**

## Results

The results of the experiments demonstrated that certain modifications, such as using the SiLU activation function and integrating SE blocks, improved the model's performance. However, issues like overfitting were observed, indicating the need for further tuning and regularization.

### Visualizations

**Architectural Modifications Performance - Part 1**

<img width="718" alt="image" src="https://github.com/mlk500/MobileNetsV2-DL/assets/57171298/e80d807b-995a-48d7-9cdd-8a637bdb1180">

[Interactive Visualization Link](https://wandb.ai/malak-y17/mobilenetv2_arch/reports/-Run-Summaries-Part-1--Vmlldzo3MzY4MDI4)

 

**Architectural Modifications Performance - Part 2**

<img width="716" alt="image" src="https://github.com/mlk500/MobileNetsV2-DL/assets/57171298/9d42cf73-7078-46c1-b6f6-5380bf035489">

[Interactive Visualization Link](https://wandb.ai/malak-y17/arch_3/reports/Run-Summaries-Part-2--Vmlldzo3MzczNzQ5)



**Width Scaling Experiment**

![image](https://github.com/mlk500/MobileNetsV2-DL/assets/57171298/60a5ed4f-1b62-40da-bf07-2bd1ed4df300)


## Challenges and Learnings

### Challenges

- **Long Training Times**: Extensive running times required for each model training session.
- **Initial Validation Mistake**: Models were initially validated on the training set, skewing performance perception.

### Learnings

- Importance of proper validation techniques.
- Valuable insights into model architecture and the impact of different modifications.

## Future Work

For future improvements, we propose:

- **Combining Width Scaling with SE Blocks**
- **Exploring Compound Scaling (EfficientNet's Technique)**
- **Extending Experiments to ImageNet**

## References

- Original MobileNetV2 Paper: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- CIFAR-10 Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

