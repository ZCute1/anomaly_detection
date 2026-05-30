# Machine Vision Build Project Steel Defect Classifier
**By Zuriel Olu-Silas**

This project implements a custom Convolutional Neural Network (CNN) to detect and classify surface defects in steel images using the Severstal Steel Defect Detection Dataset.

Title: Steel Defect Classifier
DatasetSource: Severstal Steel Defect Detection 
DatasetClasses: no_defect, defect_1, defect_2, defect_3, defect_4
Total images: 1,000 (Balanced subset of 200 images per class)

## Dataset
| Class | Count |
| :--- | :--- |
| no_defect | 200 |
| defect_1 | 200 |
| defect_2 | 200 |
| defect_3 | 200 |
| defect_4 | 200 |

## Train / Test Split
The dataset was split using a stratified approach to maintain class balance across all sets.

| Split | Size |
| :--- | :--- |
| Train | 699 (70%) |
| Validation | 150 (15%) |
| Test | 151 (15%) |

## Model Architecture
The SteelCNN is a custom convolutional neural network designed for industrial surface inspection. It uses batch normalization for training stability and dropout in the fully connected layers to prevent overfitting.

```text
SteelCNN(
  (features): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2)
    (4): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(64)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=2, stride=2)
    (8): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): BatchNorm2d(128)
    (10): ReLU()
    (11): MaxPool2d(kernel_size=2, stride=2)
  )
  (pool): AdaptiveAvgPool2d(output_size=1)
  (classifier): Sequential(
    (0): Flatten()
    (1): Linear(in_features=128, out_features=64)
    (2): ReLU()
    (3): Dropout(p=0.3)
    (4): Linear(in_features=64, out_features=5)
  )
)
```

Trainable parameters: 102,277

## Training Results
The model was trained for 10 epochs using the Adam optimizer (with a learning rate of 0.001) and Cross-Entropy Loss.

- Best Validation Accuracy: 0.513 (Reached at Epoch 8)
- Final Training Loss: 1.2607
- Inference Latency: ~3ms per frame (on NVIDIA T4 GPU)

## Training Loss Curve
![Training Loss Curves](docs/images/loss_curve.png)

## Evaluation - Confusion Matrix
![Confusion Matrix](docs/images/confusion_matrix.png)
The model demonstrates an ability to distinguish between defect-free surfaces and specific defect types, though there is some overlap between similar defect classes.