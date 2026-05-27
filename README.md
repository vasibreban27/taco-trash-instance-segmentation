# TACO Trash Instance Segmentation with Mask R-CNN

This project implements an instance segmentation pipeline for trash detection using the **TACO Trash Segmentation** dataset in COCO format.

The main model is based on **Mask R-CNN with a ResNet-50 FPN backbone**, fine-tuned for detecting, localizing, and segmenting waste objects in real-world images. The project also includes a multi-task extension with an auxiliary image-level classification head.

---

## Overview

The goal of this project is to train and evaluate a computer vision model capable of identifying trash objects in real-world images.

For each detected object, the model predicts:

- object class;
- bounding box;
- segmentation mask;
- confidence score.

The project compares three model variants:

1. **Baseline adapted pretrained Mask R-CNN**
2. **Fine-tuned Mask R-CNN**
3. **Mask R-CNN with an auxiliary multi-label classification head**

The final model demonstrates a complete deep learning workflow for instance segmentation, from dataset analysis and preprocessing to training, evaluation, and visualization.

---

## Dataset

The dataset used is **TACO Trash Segmentation**, available in COCO Segmentation format from Roboflow:

https://universe.roboflow.com/corey-hughes/taco-trash-segmentation

Raw dataset split:

| Split | Images | Annotations |
|---|---:|---:|
| Train | 1054 | 3382 |
| Validation | 300 | 895 |
| Test | 150 | 542 |

The dataset contains real-world images of trash objects in natural environments. It is challenging because of:

- cluttered backgrounds;
- small objects;
- overlapping objects;
- irregular object shapes;
- class imbalance;
- varied lighting and image resolutions.

---

## Selected Classes

The original dataset contains many categories, but several of them have very few examples. To make training more stable and feasible on Google Colab, the project keeps the 10 most frequent foreground classes.

Selected classes:

1. Cigarette
2. Unlabeled litter
3. Plastic film
4. Clear plastic bottle
5. Other plastic
6. Other plastic wrapper
7. Drink can
8. Plastic bottle cap
9. Plastic straw
10. Broken glass

After cleaning and filtering:

| Indicator | Value |
|---|---:|
| Raw images | 1504 |
| Clean images | 1123 |
| Raw annotations | 4819 |
| Clean annotations | 3185 |
| Selected foreground classes | 10 |
| Image size used by model | 512 |

---

## Main Features

The notebook includes:

- Google Colab setup with Drive persistence;
- reproducibility setup using a fixed random seed;
- COCO annotation loading;
- dataset statistics;
- class distribution analysis;
- image resolution analysis;
- annotation quality checks;
- class filtering;
- dataset cleaning;
- preprocessing and resizing;
- data augmentation;
- Mask R-CNN model definition;
- fine-tuning strategy;
- auxiliary multi-label classification head;
- training and validation loops;
- COCO-style evaluation;
- precision, recall, F1, and IoU metrics;
- visual comparison between ground truth and predictions;
- optional Gradio demo.

---

## Model Architecture

The base model is:

```text
Mask R-CNN + ResNet-50 + Feature Pyramid Network
```

### Mask R-CNN

Mask R-CNN is used because it can solve three tasks at the same time:

- classify each detected object;
- predict bounding boxes;
- generate segmentation masks.

This makes it suitable for instance segmentation tasks where each object instance must be separated individually.

### ResNet-50 Backbone

The ResNet-50 backbone is used as a feature extractor. It provides strong pretrained visual features and improves training stability through residual connections.

### Feature Pyramid Network

The FPN component helps the model detect objects at multiple scales. This is important for the TACO dataset because trash objects can be small, partially occluded, or placed in complex scenes.

### Auxiliary Head

In addition to the standard Mask R-CNN heads, the project adds an auxiliary image-level classification head.

This head predicts which trash classes are present in the image, independently of the number of object instances.

The auxiliary head is trained with multi-label binary classification and acts as an additional learning signal for the shared backbone.

---

## Training Setup

Main training configuration:

| Parameter | Value |
|---|---:|
| Image size | 512 |
| Batch size | 2 |
| Number of epochs | 5 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Seed | 42 |
| Device | Google Colab T4 |

The training uses transfer learning. The model starts from pretrained weights and is adapted to the selected TACO classes.

The fine-tuning strategy includes:

- replacing the original classification and mask heads;
- freezing and unfreezing parts of the model;
- using AdamW for stable optimization;
- using a cosine learning rate scheduler;
- applying gradient clipping;
- saving the best model checkpoints.

---

## Loss Functions

The model optimizes the standard Mask R-CNN losses:

- RPN objectness loss;
- RPN bounding box regression loss;
- detection classification loss;
- detection bounding box regression loss;
- mask segmentation loss.

For the multi-task model, an additional auxiliary loss is added:

```text
Total Loss = Mask R-CNN Loss + lambda_aux * Auxiliary BCE Loss
```

where:

```text
lambda_aux = 0.2
```

The auxiliary head uses `BCEWithLogitsLoss`, because the image-level classification task is multi-label.

---

## Evaluation Metrics

The project evaluates the models using both intuitive and standard instance segmentation metrics.

Main metrics:

- Precision at IoU 0.50;
- Recall at IoU 0.50;
- F1-score at IoU 0.50;
- mean matched mask IoU;
- COCO AP;
- COCO AP50;
- COCO AP75;
- COCO AR.

The auxiliary head is evaluated separately using:

- micro precision;
- micro recall;
- micro F1;
- macro precision;
- macro recall;
- macro F1;
- hamming accuracy;
- exact match accuracy;
- average precision.

---

## Results

### Main Test Metrics

| Model | Precision@0.50 | Recall@0.50 | F1@0.50 | Mean Mask IoU |
|---|---:|---:|---:|---:|
| Baseline adapted pretrained | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fine-tuned Mask R-CNN | 0.4245 | 0.1236 | 0.1915 | 0.8296 |
| Mask R-CNN + auxiliary head | 0.3526 | 0.1676 | 0.2272 | 0.8107 |

The fine-tuned model improves significantly over the adapted pretrained baseline. The multi-task model increases recall and F1-score, meaning it detects more objects, although with lower precision.

### COCO-Style Metrics

| Model | Type | AP | AP50 | AP75 | AR100 |
|---|---|---:|---:|---:|---:|
| Fine-tuned Mask R-CNN | bbox | 0.1234 | 0.1924 | 0.1357 | 0.2729 |
| Fine-tuned Mask R-CNN | segm | 0.1142 | 0.1848 | 0.1210 | 0.2606 |
| Mask R-CNN + auxiliary head | bbox | 0.1210 | 0.2163 | 0.1148 | 0.3016 |
| Mask R-CNN + auxiliary head | segm | 0.1157 | 0.1943 | 0.1275 | 0.2870 |

The auxiliary head improves AP50 and recall, which is useful in scenarios where missing objects is more problematic than producing additional false positives.

### Auxiliary Head Metrics

Best threshold used for the auxiliary head:

```text
threshold = 0.15
```

| Metric | Value |
|---|---:|
| Micro precision | 0.1972 |
| Micro recall | 0.8953 |
| Micro F1 | 0.3233 |
| Macro precision | 0.1705 |
| Macro recall | 0.7833 |
| Macro F1 | 0.2668 |
| Hamming accuracy | 0.3828 |
| Macro average precision | 0.1945 |

The auxiliary head is not used as a standalone detector. Its main role is to provide a global image-level learning signal that helps the shared backbone.

---

## Visual Results

The notebook generates visual comparisons between:

- ground truth bounding boxes and masks;
- predicted bounding boxes and masks;
- successful detections;
- failure cases.

The best examples show that the model can correctly detect and segment selected trash objects. The weakest examples usually involve crowded scenes, small objects, rare classes, or predictions below the confidence threshold.

---

## Requirements

Main libraries used:

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Albumentations
- Pillow
- pycocotools
- scikit-learn
- Roboflow
- Gradio

The full environment is stored in:

```text
requirements.txt
```

---

## Conclusions

This project shows that Mask R-CNN can be adapted for real-world trash instance segmentation using transfer learning.

The main findings are:

- the pretrained baseline is not enough without fine-tuning;
- fine-tuning improves detection and segmentation quality;
- the auxiliary multi-label head improves recall and F1-score;
- the multi-task model detects more objects but introduces more false positives;
- class imbalance remains a major limitation of the dataset.

Overall, the project demonstrates a complete computer vision workflow for detection, segmentation, and multi-head deep learning architectures.

---

## Future Improvements

Possible improvements:

- train for more epochs;
- use stronger augmentations;
- use class-aware sampling;
- add focal loss for class imbalance;
- experiment with ResNet-101 or transformer-based backbones;
- evaluate per-class AP more extensively;
- improve threshold tuning;
- deploy the model as a web demo;
- export the model for inference using ONNX.

---

## Academic Context

This project was developed for the **Intelligent Systems** course at the **Technical University of Cluj-Napoca**, Department of Computer Science.

The project covers concepts related to:

- deep learning;
- computer vision;
- convolutional neural networks;
- transfer learning;
- object detection;
- instance segmentation;
- multi-task learning;
- model evaluation.

---
