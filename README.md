
# 🧠 Breast Cancer Treatment Prediction 

## 📌 Project Overview

This project aims to predict which type of treatment a breast cancer patient is likely to receive — specifically **chemotherapy**, **radiotherapy**, or **hormone therapy** — using clinical and molecular features. The core goal is to explore whether treatment decisions can be anticipated based on patient data and how well machine learning can capture such decisions. This can aid in **personalized medicine**, **treatment planning**, and potentially **identifying outliers** in current medical practices.

## 📊 Dataset

The data comes from the **METABRIC (Molecular Taxonomy of Breast Cancer International Consortium)** study, a joint Canada-UK project published in Nature Communications (Pereira et al., 2016). The dataset was obtained via [Kaggle](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric) and contains:

- Clinical information (age, tumor size, menopausal state, etc.)
- Molecular markers (ER/PR/HER2 status, mutation counts, etc.)
- Treatment flags (whether the patient received chemo-, radio-, or hormone therapy)

## 🧪 Methodology

We approached the prediction task in **two main modeling phases**:

### 1. **Binary Classification: Chemotherapy Prediction**

- **Objective**: Predict whether a patient receives chemotherapy (Yes/No).
- **Model**: Feedforward Neural Network (PyTorch)
- **Evaluation**: ROC AUC, Accuracy, Precision, Recall, F1-Score

### 2. **Multiclass Classification (Simplified)**

- **Objective**: Predict which *single* therapy the patient receives — **only** one of chemo, radio, or hormone (mutually exclusive).
- **Classes**:
  - `0`: Chemotherapy only
  - `1`: Radiotherapy only
  - `2`: Hormone therapy only
- **Model**: Multiclass Neural Network
- **Note**: Patients who received combinations of treatments were excluded in this simplified scenario.

## 📈 Feature Selection and Importance

Before training, we performed a **Random Forest-based feature importance analysis** to determine which features most influenced the decision for chemotherapy.

### 🏆 Selected Important Features:

- `age_at_diagnosis`
- `tumor_size`
- `tumor_stage`
- `lymph_nodes_examined_positive`
- `nottingham_prognostic_index`
- `cellularity`
- `neoplasm_histologic_grade`
- `inferred_menopausal_state`
- `er_status_measured_by_ihc`
- `pr_status`
- `her2_status`

These were used in all models to maintain consistency and reduce overfitting.

## 🛠 Model Architecture

Both models were trained using a simple fully connected feedforward neural network with:

- Two hidden layers
- ReLU activations
- Dropout (0.5) to prevent overfitting
- Early stopping based on validation loss

### 🚫 Handling Overfitting

We observed initial signs of overfitting (training loss ↓, validation loss ↑). To mitigate this, we applied:

- **Dropout regularization**
- **Early stopping** with patience = 10
- **Feature reduction** to the most important predictors only

## 📊 Results

### ✅ Binary Model – Chemotherapy Prediction

| Metric        | Score     |
|---------------|-----------|
| Accuracy      | 90.4 %    |
| Precision     | 80.0 %    |
| Recall        | 75.9 %    |
| F1 Score      | 77.9 %    |
| ROC AUC       | 95.5 %    |

➡️ **High-performing model** that reliably distinguishes patients who receive chemotherapy.

---

### 🔁 Multiclass Model – Therapy Type (Simplified)

| Class   | Therapy Type   | Precision | Recall | F1 Score |
|---------|----------------|-----------|--------|----------|
| 0       | Chemotherapy   | 1.00      | 0.80   | 0.89     |
| 1       | Radiotherapy   | 0.82      | 0.71   | 0.76     |
| 2       | Hormone        | 0.76      | 0.86   | 0.80     |

- **Overall Accuracy**: 79.1 %
- **Macro F1 Score**: 81.8 %
- **Weighted F1 Score**: 78.9 %

➡️ Excellent class separation and well-balanced performance across all therapy types.

## 🧠 Interpretation

- The binary model shows that **treatment decisions for chemotherapy are highly predictable**.
- The multiclass model proves that **treatment types can be predicted with nearly 80% accuracy**, assuming exclusive therapy application.
- Strong predictors include **tumor size**, **nodal involvement**, **hormonal receptor status**, and **Nottingham index** — all aligned with clinical reasoning.

## 📂 Project Structure

```
├── data/                          # Raw and processed data
├── models/                        # Saved model weights (optional)
├── outputs/
│   └── figures/
│       ├── nn_learning_curve_*.png
│       ├── confusion_matrix_*.png
│       └── correlation/*.png      # Feature-treatment plots
├── scripts/
│   └── train_model.py             # PyTorch training script
│   └── feature_importance.py      # Feature ranking
├── notebooks/
│   └── EDA and experiments.ipynb  # Jupyter notebooks
└── README.md
```

---

## 📌 Future Work

- Include patients with combined therapies and model them hierarchically
- Explore explainable AI (e.g., SHAP values) for clinical transparency
- Deploy model in a lightweight clinical dashboard for testing

---

## 📚 References

- Pereira et al., Nature Communications, 2016  
- METABRIC study: [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric)  
- Kaggle Dataset: [Breast Cancer - Gene Expression (METABRIC)](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric)
