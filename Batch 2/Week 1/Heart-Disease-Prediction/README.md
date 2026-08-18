# Heart Disease Prediction Using Machine Learning

## Overview
This project uses machine learning techniques to predict the severity of heart disease using the UCI Heart Disease dataset. The problem is treated as a multiclass classification task where the target variable (`num`) contains five classes (0–4).

## Dataset
The dataset contains clinical and demographic features such as:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol level
- Fasting blood sugar
- ECG results
- Maximum heart rate
- Exercise-induced angina
- ST depression

## Project Workflow

1. **Data Preprocessing**
   - Handled missing values using appropriate imputation techniques.
   - Removed columns with excessive missing values.
   - Cleaned invalid values.

2. **Feature Encoding**
   - Applied One-Hot Encoding to categorical variables.
   - Applied Label Encoding to binary categorical features.

3. **Model Training**
   - Split the dataset into training and testing sets using stratified sampling.
   - Used XGBoost Classifier for multiclass classification.

4. **Hyperparameter Optimization**
   - Applied GridSearchCV with 5-fold cross-validation.
   - Tuned parameters including:
     - Number of estimators
     - Maximum depth
     - Learning rate
     - Subsample ratio
     - Column sampling ratio

5. **Model Evaluation**
   - Evaluated the model using:
     - Accuracy Score
     - Classification Report
     - Confusion Matrix

## Results

Final XGBoost Model Performance:

- Cross-Validation Accuracy: ~60%
- Test Accuracy: ~60%

The model achieved reasonable performance while facing challenges due to class imbalance in the dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## Conclusion

This project demonstrates an end-to-end machine learning workflow, including preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation for a multiclass classification problem.
