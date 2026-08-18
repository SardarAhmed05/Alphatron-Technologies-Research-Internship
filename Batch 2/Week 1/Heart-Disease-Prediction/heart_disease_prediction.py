import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib

def main():
    # Converting the csv into a dataframe

    df = pd.read_csv("heart_disease_uci.csv")

    df['trestbps'] = df['trestbps'].replace(0, np.nan)
    imputer = SimpleImputer(strategy='median')
    df['trestbps'] = imputer.fit_transform(df[['trestbps']])

    df['chol'] = df['chol'].replace(0, np.nan)
    chol_imputer = IterativeImputer(random_state=42, max_iter=10)
    df['chol'] = chol_imputer.fit_transform(df[['chol']])

    fbs_imputer = SimpleImputer(strategy="median")
    df['fbs'] = fbs_imputer.fit_transform(df[['fbs']])

    restecg_imputer = SimpleImputer(strategy='most_frequent')
    df[['restecg']] = restecg_imputer.fit_transform(df[['restecg']])

    thalch_imputer = SimpleImputer(strategy='mean')
    df['thalch'] = thalch_imputer.fit_transform(df[['thalch']])

    exang_imputer = SimpleImputer(strategy='most_frequent')
    df[['exang']] = exang_imputer.fit_transform(df[['exang']])

    df.loc[df['oldpeak'] < 0, 'oldpeak'] = np.nan
    oldpeak_imputer = SimpleImputer(strategy='median')

    df['oldpeak'] = oldpeak_imputer.fit_transform(df[['oldpeak']])

    df.drop(columns=['slope', 'ca', 'thal'], inplace=True)

    # ONE HOT ENCODING / GET DUMMIES FOR 'dataset' and 'cp' column
    df = pd.get_dummies(df, columns=['dataset'], drop_first=True)
    df = pd.get_dummies(df, columns=['cp'], drop_first=True)
    df = pd.get_dummies(df, columns=['restecg'], drop_first=True)

    # LABEL ENCODING COLUMNS
    sex_encoder = LabelEncoder()
    df['sex'] = sex_encoder.fit_transform(df['sex'])

    fbs_encoder = LabelEncoder()
    df['fbs'] = fbs_encoder.fit_transform(df['fbs'])

    exang_encoder = LabelEncoder()
    df['exang'] = exang_encoder.fit_transform(df['exang'])

    df.drop('id', axis=1, inplace=True)

    X = df.drop('num', axis=1)
    y = df['num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

    # Define the model
    model = XGBClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best CV Accuracy:", grid.best_score_)

    best_xgb = grid.best_estimator_

    y_pred = best_xgb.predict(X_test)

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(best_xgb, "xgboost_model.pkl")

if __name__ == "__main__":
    main()