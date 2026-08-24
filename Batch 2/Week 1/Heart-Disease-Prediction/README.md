# Heart Disease Prediction Using Machine Learning
**Research Internship - Week 1 Submission**  
**Author:** Sardar Ahmed  
**Institution/Company:** Alphatron Technologies  
**Structure:** 11-Step Modular Data Science Pipeline with Master Orchestrator (`Main.py`)

---

## 📌 1. Project Overview
This project delivers a modular, end-to-end Machine Learning system for heart disease diagnosis and severity staging using the multicenter UCI Heart Disease cohort (Cleveland, Hungarian, Switzerland, and Long Beach V cohorts, total $N = 920$ patients).

The repository is structured into **11 sequential, decoupled Python modules (`Step_1_...py` to `Step_11_...py`)** coordinated by a master orchestrator (`Main.py`), featuring dedicated artifact directories, automated SMOTE class balancing, 5-fold cross-validation, and comprehensive model comparison.

---

## 📂 2. Repository & Output Folder Structure

```
Heart-Disease-Prediction/
├── Datapreparation/                       # Step 1 Outputs
│   └── cleaned_heart_disease.csv
├── EDA/                                   # Step 2 Outputs (Exploratory Figures & Stats)
│   ├── 01_missing_values_analysis.png
│   ├── 02_target_distribution.png
│   ├── 07_correlation_matrix.png
│   └── eda_summary_statistics.csv
├── TransformedData/                       # Steps 3, 4, 5, 8 Outputs
│   ├── transformed_data.csv
│   ├── selected_features_data.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── X_train_sampled.csv
│   └── y_train_sampled.csv
├── Models/                                # Steps 6, 10, 11 Outputs (Serialized Models)
│   ├── baseline_logistic_regression.pkl
│   ├── baseline_random_forest.pkl
│   ├── baseline_xgboost.pkl
│   ├── tuned_xgboost.pkl
│   └── best_model.pkl
├── ModelComparison/                       # Step 11 Outputs (Metrics & Plots)
│   ├── model_comparison_results.csv
│   └── model_accuracy_comparison.png
│
├── Step_1_DataPreparation.py              # Ingestion, zero-anomaly handling, string cleaning
├── Step_2_EDA.py                          # Statistical profiling & distribution plots
├── Step_3_Data_Transformation.py          # Imputation, One-Hot/Binary encoding, StandardScaler
├── Step_4_FeatureSelection.py             # Feature selection (dropping ca, thal, slope, id)
├── Step_5_Train_Test_DataSplit.py         # Stratified 80/20 train/test partitioning
├── Step_6_Model_Training.py               # Fitting baseline LR, RF, and XGBoost models
├── Step_7_ModelTesting_Performceevaluation.py # Test set evaluation & classification reports
├── Step_8_DataSampling.py                 # SMOTE class balancing on training data
├── Step_9_Cross_Validation.py             # 5-Fold Stratified Cross-Validation
├── Step_10_HyperparameterOptimization.py  # GridSearchCV tuning for XGBoost
├── Step_11_Model_Comparison.py            # Comparative evaluation & best model export
├── Main.py                                # Master orchestrator running Steps 1-11
│
├── HeartDiseasePrediction.ipynb           # Unified interactive master notebook
├── flowchart_Task1_Sardar_Ahmed_Heart_disease_prediction.pdf # 11-step pipeline flowchart PDF
├── Report.pdf                             # Formal research report PDF
├── requirements.txt                       # Curated environment dependencies
└── heart_disease_uci.csv                  # Raw dataset
```

---

## 🔄 3. 11-Step Machine Learning Pipeline

```
┌──────────────────────────────────────────────────────────┐
│ Step 1: Data Preparation                                 │
│  - Load UCI dataset, handle 0s in [chol, trestbps]       │
│  - Export: 'Datapreparation/cleaned_heart_disease.csv'   │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ Step 2: Exploratory Data Analysis (EDA)                  │
│  - Profiling, missingness, target balance, correlations  │
│  - Export: 'EDA/' charts & summary statistics            │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ Step 3: Data Transformation                              │
│  - Median/Mode Imputation, OHE, StandardScaler           │
│  - Export: 'TransformedData/transformed_data.csv'        │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ Step 4: Feature Selection                                │
│  - Drop non-predictive [id] & high-missing [ca,thal,slope]│
│  - Export: 'TransformedData/selected_features_data.csv'  │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ Step 5: Stratified Train / Test Split                    │
│  - 80% Train (736 samples) / 20% Test (184 samples)      │
│  - Export: X_train, X_test, y_train, y_test              │
└──────────────┬─────────────────────────────┬─────────────┘
               │                             │
               ▼ (Train Set)                 ▼ (Test Set - Unseen)
┌──────────────────────────────┐             │
│ Step 6: Baseline Training    │             │
│  - Fit LR, RF, XGBoost       │             │
│  - Export: 'Models/'         │             │
└──────────────┬───────────────┘             │
               │                             │
               ▼                             │
┌──────────────────────────────┐             │
│ Step 7: Baseline Testing     │             │
│  - Accuracy, F1, Report      │             │
└──────────────┬───────────────┘             │
               │                             │
               ▼                             │
┌──────────────────────────────┐             │
│ Step 8: Data Sampling (SMOTE)│             │
│  - Balance minority classes  │             │
│  - Export: X_train_sampled   │             │
└──────────────┬───────────────┘             │
               │                             │
               ▼                             │
┌──────────────────────────────┐             │
│ Step 9: 5-Fold Cross-Val     │             │
│  - StratifiedKFold checking  │             │
└──────────────┬───────────────┘             │
               │                             │
               ▼                             │
┌──────────────────────────────┐             │
│ Step 10: Grid Search Tuning  │             │
│  - Tune XGBoost depth & lr   │             │
│  - Export: 'tuned_xgboost'   │             │
└──────────────┬───────────────┘             │
               │                             │
               ▼                             ▼
┌────────────────────────────────────────────┴─────────────┐
│ Step 11: Model Comparison & Selection                    │
│  - Compare all models on unseen test data                │
│  - Export: 'model_comparison_results.csv',               │
│            'model_accuracy_comparison.png',              │
│            'Models/best_model.pkl'                       │
└──────────────────────────────────────────────────────────┘
```

---
## 📊 4. Experimental Results

| Model Architecture | Training Strategy | Test Accuracy | Macro F1 | Weighted F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | Original Split | **60.87%** | **0.3662** | **0.5801** |
| **Random Forest (Baseline)** | Original Split | 57.07% | 0.2752 | 0.5077 |
| **XGBoost (Baseline)** | Original Split | 60.33% | 0.3855 | 0.5859 |
| **XGBoost (Tuned + SMOTE)** | SMOTE Resampled | 56.52% | **0.4173** | 0.5787 |

---

## 🛠️ 5. Quick Start & Execution

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Entire 11-Step Pipeline in One Command
```bash
python Main.py
```

### 3. Run Individual Steps Separately
```bash
python Step_1_DataPreparation.py
python Step_2_EDA.py
python Step_3_Data_Transformation.py
python Step_4_FeatureSelection.py
python Step_5_Train_Test_DataSplit.py
python Step_6_Model_Training.py
python Step_7_ModelTesting_Performceevaluation.py
python Step_8_DataSampling.py
python Step_9_Cross_Validation.py
python Step_10_HyperparameterOptimization.py
python Step_11_Model_Comparison.py
```

### 4. Interactive Master Notebook
Launch Jupyter Notebook to run step-by-step EDA and benchmarks:
```bash
jupyter notebook HeartDiseasePrediction.ipynb
```
