# Food Multi-Label Classification Pipeline (Food11 + FoodOrNot)
**Multilabel Dataset Builder + MobileNetV2 Training + MLflow Tracking + Flask Web UI**

This repository implements a complete end-to-end pipeline to:
1) Generate a **real multilabel dataset** from Food11 + FoodOrNot  
2) Train a deep learning multilabel classifier using **MobileNetV2 (ImageNet)**  
3) Track experiments and metrics with **MLflow**  
4) Run predictions on images (single image inference)  
5) Deploy a simple **Flask web interface** to upload and analyze images visually  

---a

## 1) Project Overview

### ✅ Goal
Train a model that can detect multiple food categories in the same image (multilabel classification), using:

- `lacteos` (dairy products)
- `arroz` (rice)
- `frutas/verduras` (fruits & vegetables)

Additionally, the dataset includes **non-food negative images** labeled as `(0,0,0)` to reduce false positives and improve rejection behavior.

---

## 2) Multilabel Output Format

The model produces probabilities per class:

| Class | Meaning |
|------|---------|
| `lacteos` | Dairy products |
| `arroz` | Rice |
| `frutas/verduras` | Fruits and vegetables |

Example multilabel vectors:
- `[1, 0, 0]` → dairy only  
- `[0, 1, 1]` → rice + fruits/vegetables  
- `[1, 1, 1]` → all classes  
- `[0, 0, 0]` → none / non-food  

---

## 3) Repository Structure

```text
Food-Multi-Label-Classification-Pipeline-with-TensorFlow-Dataset-Builder/
│
├─ raw/                              # Raw datasets (manual download)
│   ├─ food11/
│   │   ├─ training/
│   │   │   ├─ Dairy product/
│   │   │   ├─ Rice/
│   │   │   └─ Vegetable-Fruit/
│   │   ├─ validation/
│   │   └─ evaluation/
│   │
│   └─ foodornot/
│       ├─ train/negative_non_food/
│       └─ test/negative_non_food/
│
├─ dataset/
│   ├─ images/                       # Generated dataset images (auto)
│   └─ labels.csv                    # Generated multilabel annotations (auto)
│
├─ outputs/
│   └─ model.keras                   # Saved trained model (auto)
│
├─ mlruns/                           # MLflow experiment logs (auto)
│
├─ outputs/1_prepare_dataset_multilabel_real.ipynb
├─ outputs/2_train_multilabel.ipynb
├─ outputs/3_predict_image_multilabel.ipynb
│
└─ flask_app/
    ├─ app.py
    ├─ templates/
    │   └─ index.html
    └─ static/
        ├─ style.css
        └─ uploads/                  # Uploaded files (ignored from Git)
