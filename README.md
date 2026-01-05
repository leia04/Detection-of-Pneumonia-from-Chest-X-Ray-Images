# Detection of Pneumonia from Chest X-Ray Images

## Abstract
This project focuses on detecting pneumonia from pediatric chest X-ray images using a convolutional neural network (CNN) implemented from scratch in Python.  
Rather than relying on high-level deep learning frameworks, the model explicitly implements core CNN components such as convolution, pooling, and the training loop using NumPy, with the goal of understanding and demonstrating the internal mechanics of CNN-based image classification.

## Problem
Pneumonia is a leading cause of mortality among children worldwide, and chest X-ray imaging is commonly used for diagnosis.  
However, interpreting X-ray images requires clinical expertise and can be time-consuming. This project addresses the problem of automatically classifying chest X-ray images into pneumonia and normal cases using a CNN-based approach.


## Key Findings
- The CNN model successfully learned discriminative features from chest X-ray images.
- Training and validation accuracy improved consistently during training.
- Visualization of convolutional filters indicated meaningful low-level feature extraction from medical images.


## Approach
- Implemented a custom CNN architecture using NumPy, including convolutional and pooling layers.
- Constructed a data loading and preprocessing pipeline for grayscale chest X-ray images.
- Trained the model using mini-batch gradient descent and evaluated classification performance.
- Visualized learned convolutional filters to qualitatively assess feature extraction.


## Code
- `src/data_loader.py`: Loads and preprocesses chest X-ray images (grayscale conversion, resizing, label assignment).
- `src/layers/conv.py`: Implements the convolution forward pass.
- `src/layers/pooling.py`: Implements max-pooling forward pass.
- `src/layers/im2col.py`: Provides the im2col utility for efficient convolution and pooling operations.
- `src/model.py`: Defines the CNN architecture (SimpleConvNet).
- `src/trainer.py`: Implements the training loop, optimizer updates, and accuracy evaluation.
- `src/train.py`: Entry-point script that integrates data loading, model construction, and training.
- `notebooks/01_filter_visualization.ipynb`: Visualizes learned convolutional filters (optional).

## Tools and Libraries
- Python
- NumPy
- OpenCV
- Matplotlib
- tqdm

## Contribution
- Designed and implemented the entire project independently.
- Implemented convolution, pooling, and training components from scratch using NumPy.
- Developed the data loading and preprocessing pipeline for chest X-ray images.
- Conducted model training and result visualization.



