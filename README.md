
# Video Capsule Abnormality Detection with ResNet + Transformer

This repository provides a hybrid computer vision model using **ResNet** as a feature extractor and a **Transformer encoder** for classifying abnormalities in medical capsule endoscopy or video frames, emphasizing class imbalance handling.

## Overview

- **Goal:** Detect and classify abnormalities in endoscopic images or video stills using deep learning with a focus on imbalanced class distribution.
- **Approach:** Combines a pretrained ResNet50 backbone with a custom Transformer encoder for robust spatial feature modeling, including targeted data augmentation.

## Pipeline

### 1. Imports and Dependencies

- Libraries: `torch`, `torchvision`, `numpy`, `sklearn`, `matplotlib`, `PIL`

### 2. Data Loading and Augmentation

- **Inputs:** Images organized by class in folders (`training`, `validating`)
- Augmentation is tailored:
  - Minority classes: Heavy augmentation (resize, random rotations, flips)
  - Majority classes: Minimal augmentation (resize)
- Custom `Dataset` classes for train and validation, with per-class augmentation strategy.

### 3. Sampler and Loader

- **WeightedRandomSampler**: Ensures training batches reflect target class balancing based on class frequency.
- `DataLoader`: Efficiently loads images with augmentation according to class.

### 4. Model Definition

- **ResNetTransformerClassifier**
  - ResNet50 pretrained on ImageNet, ending with a linear layer for feature size adaptation.
  - Transformer encoder (tunable `num_layers`, `num_heads`) for enhanced context modeling.
  - Final classifier head for multiclass output (default 4 classes).

### 5. Training Setup

- **Loss function:** Weighted CrossEntropyLoss to reflect class imbalance.
- **Optimizer:** AdamW with learning rate scheduling.
- Early stopping based on validation metrics.

### 6. Metrics

- Reports for both train and validation:
  - Balanced Accuracy
  - Macro F1-Score
  - Mean ROC-AUC Score

### 7. Training Loop

- Custom loop with:
  - On-epoch evaluation and best model checkpointing
  - Early stopping on plateau
  - Save best model as `best_model.pth`

***

## Requirements

- Python 3.7+
- torch
- torchvision
- numpy
- scikit-learn
- matplotlib
- PIL

To install main dependencies:
```bash
pip install torch torchvision numpy scikit-learn matplotlib pillow
```

## How to Run

1. Organize training and validation image folders as:
   ```
   ./trining-dataset/training/
   ./validation/validating/
   ```
   Each class should be a subfolder with images.
2. Launch Jupyter or Colab and open `resnet-trans 4.ipynb`.
3. Run all cells to preprocess, train, and evaluate.
4. Trained model checkpoint and logs will be generated.

***

## Notes

- Minor typo in directory (`trining-dataset` vs `training-dataset`): match with your folder structure.
- Hyperparameters (epochs, patience, batch size, transformer depth) are configurable at the notebook head.
- Class balancing is handled both with data augmentation and loss weighting.


_This README explains the overall workflow, technical requirements, and quick-start guide for the notebook._[1]

[1](https://github.com/hindu-muppala/VideoCapsuleAbnormalityDetection/blob/main/resnet-trans%204.ipynb)
