# Food Multi-Label Classification Pipeline

<p align="center">
  <img src="https://www.rededucom.org/img/proyectos/14/galeria/big/universidad-salesiana.jpg" width="200">
</p>

<p align="center">
  <strong>Multilabel Dataset Builder + MobileNetV2 Training + MLflow Tracking + Flask Web UI</strong>
</p>

<p align="center">
  <b>Authors:</b> Erika Contreras, Alexander Chuquipoma<br>
  <b>Institution:</b> Universidad Salesiana
</p>

---

## Overview

This repository implements a **complete end-to-end pipeline** for multilabel food image classification:

| Step | Description |
|------|-------------|
| 1 | Generate a **real multilabel dataset** from Food11 + FoodOrNot |
| 2 | Train a deep learning classifier using **MobileNetV2 (ImageNet)** |
| 3 | Track experiments and metrics with **MLflow** |
| 4 | Run predictions on images (single image inference) |
| 5 | Deploy a **Flask web interface** for visual analysis |

---

## Datasets Used

| Dataset | Source | Purpose |
|---------|--------|---------|
| **Food-11** | [Kaggle](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset) | Food categories (Dairy, Rice, Vegetables) |
| **FoodOrNot** | [Kaggle](https://www.kaggle.com/datasets/sciencelabwork/food-or-not-dataset) | Non-food negative samples |

---

## Target Classes (Multilabel)

The model predicts **independent probabilities** for each class:

| Class | Description |
|-------|-------------|
| `lacteos` | Dairy products |
| `arroz` | Rice |
| `frutas/verduras` | Fruits and vegetables |

**Example multilabel vectors:**

| Vector | Meaning |
|--------|---------|
| `[1, 0, 0]` | Dairy only |
| `[0, 1, 1]` | Rice + Fruits/Vegetables |
| `[1, 1, 1]` | All classes |
| `[0, 0, 0]` | None / Non-food |

---

## Repository Structure

```
Food-Multi-Label-Classification-Pipeline/
│
├── raw/                                    # Raw datasets (manual download)
│   ├── food11/
│   │   ├── training/
│   │   │   ├── Dairy product/
│   │   │   ├── Rice/
│   │   │   └── Vegetable-Fruit/
│   │   ├── validation/
│   │   └── evaluation/
│   └── foodornot/
│       ├── train/negative_non_food/
│       └── test/negative_non_food/
│
├── dataset/                                # Generated dataset (auto)
│   ├── images/
│   └── labels.csv
│
├── outputs/                                # Notebooks and trained model
│   ├── 1_prepare_dataset_multilabel_real.ipynb
│   ├── 2_train_multilabel.ipynb
│   ├── 3_predict_image_multilabel.ipynb
│   └── model.keras
│
├── mlruns/                                 # MLflow experiment logs (auto)
│
└── flask_app/                              # Web application
    ├── app.py
    ├── templates/index.html
    └── static/
        ├── style.css
        └── uploads/
```

---

## Requirements

```bash
pip install tensorflow pandas numpy pillow scikit-learn mlflow flask
```

| Package | Purpose |
|---------|---------|
| `tensorflow` | Deep learning framework (MobileNetV2, tf.data) |
| `pandas` | CSV handling and data manipulation |
| `numpy` | Numerical operations |
| `pillow` | Image loading and collage generation |
| `scikit-learn` | Train/validation split |
| `mlflow` | Experiment tracking |
| `flask` | Web interface for predictions |

---

## Pipeline Details

### Step 1: Dataset Preparation

**Notebook:** `outputs/1_prepare_dataset_multilabel_real.ipynb`

Generates a multilabel dataset by creating **collage images** that combine multiple food categories.

**Configuration:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_TRAIN` | 3,000 | Collages for training |
| `N_VAL` | 600 | Collages for validation |
| `N_TEST` | 600 | Collages for testing |
| `P_INCLUDE` | 0.60 | Probability each class appears |
| `MAX_NONFOOD_TRAIN` | 1,200 | Non-food samples for training |
| `IMG_SIZE` | 224×224 | Image dimensions |

**Output Statistics:**

| Metric | Value |
|--------|-------|
| Total images | 5,800 |
| `lacteos` positives | ~2,627 |
| `arroz` positives | ~2,565 |
| `frutas/verduras` positives | ~2,573 |
| Non-food (NONE) | 1,600 |

---

### Step 2: Model Training

**Notebook:** `outputs/2_train_multilabel.ipynb`

Uses **transfer learning** with MobileNetV2 pretrained on ImageNet.

**Architecture:**

```
Input (224×224×3)
       ↓
MobileNetV2 (ImageNet pretrained)
       ↓
GlobalAveragePooling2D
       ↓
Dropout (0.2)
       ↓
Dense (3 units, sigmoid)
       ↓
Output: [P(lacteos), P(arroz), P(frutas/verduras)]
```

**Two-Stage Training Strategy:**

| Stage | Description | Learning Rate | Trainable Params |
|-------|-------------|---------------|------------------|
| `head` | Train classification head only (backbone frozen) | 1e-3 | ~3,843 |
| `finetune` | Unfreeze last 30 layers of backbone | 1e-4 | ~1M+ |

**Training Results:**

| Epoch | Train Acc | Val Acc | Val AUC | Val Loss |
|-------|-----------|---------|---------|----------|
| 1 | 82.47% | 88.85% | 0.9606 | 0.2702 |
| 5 | 94.84% | 94.43% | 0.9877 | 0.1516 |
| 10 | 96.19% | 95.43% | 0.9915 | 0.1236 |

**MLflow Tracking:**

```bash
mlflow ui
# Open http://localhost:5000
```

---

### Step 3: Single Image Prediction

**Notebook:** `outputs/3_predict_image_multilabel.ipynb`

**Prediction Pipeline:**

```python
# Load and preprocess
img = tf.io.read_file(path)
img = tf.image.decode_image(img, channels=3)
img = tf.image.resize(img, (224, 224))
img = img / 255.0

# Predict
probs = model.predict(img)

# Apply thresholds
THRESH = 0.50
NONE_IF_MAX_BELOW = 0.45
```

**Decision Logic:**

| Condition | Result |
|-----------|--------|
| Any class ≥ 0.50 | Return all classes above threshold |
| Max probability < 0.45 | "Not found / None" |
| Otherwise | "Uncertain (most probable: X)" |

**Example Output:**

```
📌 Image: test_0000001.jpg

Probabilities:
 - lacteos: 92.94%
 - arroz: 2.33%
 - frutas/verduras: 99.95%

✅ Final Result: lacteos, frutas/verduras
```

---

### Step 4: Flask Web Application

**Location:** `flask_app/`

```bash
cd flask_app
python app.py
# Open http://localhost:5000
```

**Features:**
- Upload images via web form
- Display uploaded image preview
- Show per-class probabilities
- Display final multilabel prediction

---

## Quick Start Guide

```bash
# 1. Clone the repository
git clone <repo-url>
cd Food-Multi-Label-Classification-Pipeline

# 2. Install dependencies
pip install tensorflow pandas numpy pillow scikit-learn mlflow flask

# 3. Download datasets from Kaggle and place in raw/

# 4. Run notebooks in order:
#    - 1_prepare_dataset_multilabel_real.ipynb
#    - 2_train_multilabel.ipynb
#    - 3_predict_image_multilabel.ipynb

# 5. (Optional) Launch web interface
cd flask_app && python app.py
```

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **MobileNetV2** | Lightweight, efficient, suitable for mobile/edge deployment |
| **Sigmoid activation** | Independent probability per class (multilabel) |
| **Binary Cross-Entropy** | Standard loss for multilabel classification |
| **Collage generation** | Simulates real-world scenarios with multiple food items |
| **Non-food negatives** | Reduces false positives, improves rejection capability |
| **Sample weighting (NEG_WEIGHT=4.0)** | Penalizes false positives on NONE samples |
| **Two-stage training** | Head-only first, then fine-tune for better convergence |

---

## Conclusion: Why This Project Matters

### Real-World Applications

This pipeline can be adapted for various practical use cases:

| Application | Description |
|-------------|-------------|
| **Nutrition Apps** | Automatically identify food types in meal photos for dietary tracking |
| **Smart Refrigerators** | Detect and inventory food items inside refrigerators |
| **Restaurant Automation** | Verify dishes before serving to ensure correct components |
| **Healthcare** | Monitor patient meals for dietary compliance |
| **Food Delivery QA** | Validate order contents through image analysis |

### Educational Value

This project demonstrates key concepts in modern deep learning:

| Concept | Implementation |
|---------|----------------|
| **Transfer Learning** | Using pretrained MobileNetV2 as feature extractor |
| **Multilabel Classification** | Sigmoid outputs with binary cross-entropy loss |
| **Data Augmentation** | Collage generation to simulate complex scenarios |
| **Negative Sampling** | Including non-food images to reduce false positives |
| **Experiment Tracking** | MLflow for reproducible ML experiments |
| **Model Deployment** | Flask web application for inference |
| **Two-Stage Training** | Head training + fine-tuning strategy |

### Technical Achievements

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 95.43% |
| **Validation AUC** | 0.9915 |
| **Model Size** | ~8.6 MB (MobileNetV2) |
| **Inference Ready** | Yes (Keras .keras format) |
| **Web Deployment** | Flask application included |

### Why Multilabel Matters

Unlike traditional multiclass classification (one label per image), **multilabel classification** is essential for real-world scenarios where:

- A single image may contain **multiple food items**
- The model must recognize **combinations** (e.g., dairy + rice in the same meal)
- The model must correctly **reject** images that contain no relevant classes

This approach mirrors how humans perceive food - rarely do we see isolated ingredients; instead, we see complete dishes with multiple components.

---
