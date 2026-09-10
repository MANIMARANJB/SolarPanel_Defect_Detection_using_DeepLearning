# ☀️ SolarGuard: Intelligent Solar Panel Defect Detection Using Deep Learning

## 📌 Project Overview

SolarGuard is a deep learning-based computer vision application designed to automatically identify the visual condition of solar photovoltaic panels.

The system uses a fine-tuned **ResNet18 Convolutional Neural Network (CNN)** to classify solar panel images into six different conditions and provides maintenance recommendations through an interactive **Streamlit application**.

The project demonstrates the complete deep learning workflow from image preprocessing and transfer learning to model fine-tuning, evaluation, and application deployment.

---

## 🎯 Project Objective

The main objective of SolarGuard is to develop an AI-assisted solar panel inspection system that can:

- Automatically classify solar panel conditions from images
- Detect common surface and visible damage conditions
- Provide prediction confidence
- Identify maintenance priority
- Recommend suitable maintenance actions
- Support faster preliminary inspection of solar installations

---

## 🏷️ Classes

The model classifies solar panel images into six categories:

1. Bird-drop
2. Clean
3. Dusty
4. Electrical-damage
5. Physical-Damage
6. Snow-Covered

---

## 📊 Dataset

The dataset contains **869 labeled solar panel images**.

| Class | Images |
|---|---:|
| Bird-drop | 191 |
| Clean | 193 |
| Dusty | 190 |
| Electrical-damage | 103 |
| Physical-Damage | 69 |
| Snow-Covered | 123 |
| **Total** | **869** |

The dataset contains moderate class imbalance, particularly for the Physical-Damage and Electrical-damage categories.

A stratified **80:20 train-validation split** was used:

- Training images: **695**
- Validation images: **174**

A separate set of unlabeled images was kept outside the training dataset for inference testing.

---

## 🔄 Project Workflow

```text
Solar Panel Images
        ↓
Data Inspection & EDA
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
Train / Validation Split
        ↓
Class Weight Calculation
        ↓
Pretrained ResNet18
        ↓
Custom Classification Head
        ↓
Transfer Learning
        ↓
Model Evaluation
        ↓
Fine-Tuning
        ↓
Final Evaluation
        ↓
Model Saving
        ↓
Streamlit Application
