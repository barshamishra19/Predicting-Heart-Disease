
# 🫀 Kaggle Playground Series 2026 - Heart Disease Prediction

[![Kaggle](https://img. ammunition.com/badge/Kaggle-Playground-blue.svg)](https://www.kaggle.com/competitions/playground-series-s6e2)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![ML](https://img.shields.io/badge/Machine%20Learning-Ensemble-orange.svg)](https://github.com/topics/machine-learning)

## 🎯 Competition Goal
The objective is to predict the probability of Heart Disease based on clinical features. Performance is evaluated using the **ROC-AUC** metric.

*   **Target Score**: 0.955+ ROC-AUC
*   **Leaderboard Goal**: Beat the 0.95393 benchmark.

---

## 📂 Repository Structure

| File | Version | Core Strategy | CV Strategy |
| :--- | :--- | :--- | :--- |
| `predicting-heart-disease.ipynb` | **Baseline** | LGBM, XGB, CatBoost | 5-Fold CV |
| `predicting-heart-disease-2.ipynb` | **Enhanced** | 5-Model Ensemble + Target Encoding | 10-Fold CV |
| `predicting-heart-disease-3.ipynb` | **Ultra** | 7-Model "Blend-of-Blends" + Seed Averaging | 10-Fold CV |

---

## 🔥 Evolution of the Approach

### 1. Baseline Version
*   **Models**: LightGBM, XGBoost, CatBoost.
*   **Engineering**: Basic health ratios (Cholesterol/Age).
*   **Ensemble**: Simple weighted average optimized via `scipy.minimize`.

### 2. Enhanced Version
*   **Models**: Added `HistGradientBoosting` and `ExtraTrees`.
*   **Engineering**: 30+ features including hypertension stages and composite risk scores.
*   **Preprocessing**: Cross-validated Target Encoding for categorical variables.
*   **Ensemble**: Introduced **Rank Averaging** and **Stacking** with a Logistic Regression meta-learner.

### 3. Ultra Optimized Version (The Benchmark)
This version focuses on variance reduction and diverse model signals:
*   **Seed Averaging**: Training across **5 different random seeds** to ensure predictions are robust to noise.
*   **Advanced Scaler**: Switched from `StandardScaler` to `QuantileTransformer` (Normal distribution) to better handle non-linear relationships.
*   **Expanded Diversity**: Added `RandomForest` and `GaussianNB` to the ensemble to capture different data perspectives.
*   **Blend-of-Blends**: A hierarchical ensemble method combining Rank Averaging, Geometric Means, and Stacked models.

---

## 🧠 Technical Methodology (Ultra Version)

### 🚀 7-Model Diversity
| Model | Role in Ensemble |
| :--- | :--- |
| **CatBoost** | Handles categorical relationships natively with high precision. |
| **XGBoost/LGBM** | Captures complex non-linear gradients. |
| **HistGradientBoosting** | Provides a fast, robust baseline for the boosting family. |
| **ExtraTrees/RF** | Adds bagging-based stability and reduces variance. |
| **GaussianNB** | Provides probabilistic diversity for the "Blend-of-Blends." |

### 🛠️ Feature Engineering Highlights
*   **Medical Interaction**: `BP_Product`, `Chol_Per_Age`, and `MAP` (Mean Arterial Pressure).
*   **Risk Categories**: Automated binning for `Hypertension Stage` and `Age Risk` (55+/65+).
*   **Statistical Rows**: Row-wise mean, std, and range across clinical indicators to capture "general health" variance.

### 🧪 Preprocessing Pipeline
1.  **Imputation**: Median strategy for numerical, Most Frequent for categorical.
2.  **Target Encoding**: 10-Fold out-of-fold encoding to prevent leakage.
3.  **Transformation**: `QuantileTransformer` mapping to a Normal distribution to stabilize neural-based and probabilistic models.

---

## 🚀 Quick Start

1.  **Clone the repo**:
    ```bash
    git clone https://github.com/your-username/heart-disease-prediction.git
    ```
2.  **Install dependencies**:
    ```bash
    pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna matplotlib seaborn scipy
    ```
3.  **Run the Ultra Notebook**: Open `predicting-heart-disease-3.ipynb` and execute. Expected runtime is approximately 25 minutes on a Kaggle P100 GPU or high-end CPU.

---

## 📊 Expected Results

*   **Validation (OOF) Score**: ~0.95547
*   **Consistency**: Seed averaging reduces the variance between local CV and Public Leaderboard scores, leading to more reliable submissions.

---

## 🛠️ Tools Used
*   **Optimization**: Optuna (Hyperparameter tuning)
*   **Analysis**: Seaborn/Matplotlib (Correlation heatmaps, distribution plots)
*   **Ensembling**: Scipy Optimize (Weighted blending)

---
*Developed by Barsha Mishra - Kaggle Playground Series S6E2*
