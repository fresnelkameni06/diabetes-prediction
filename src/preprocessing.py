import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Features du nouveau dataset
NUMERIC_FEATURES     = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
CATEGORICAL_FEATURES = ['gender', 'smoking_history']
BINARY_FEATURES      = ['hypertension', 'heart_disease']
ALL_FEATURES         = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET               = 'diabetes'

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"Diabétiques    : {df[TARGET].sum()} ({df[TARGET].mean()*100:.1f}%)")
    print(f"Non-diabét.    : {(df[TARGET]==0).sum()} ({(df[TARGET]==0).mean()*100:.1f}%)")
    return df

def get_preprocessor() -> ColumnTransformer:
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC_FEATURES + BINARY_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ])
    return preprocessor

def prepare_data(df: pd.DataFrame):
    X = df[ALL_FEATURES]
    y = df[TARGET]

    # Split 80/20 — test jamais touché
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Split 85/15 — train/val depuis train_val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
    )

    print(f"\nTrain      : {X_train.shape[0]} exemples | Diabétiques: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"Validation : {X_val.shape[0]} exemples   | Diabétiques: {y_val.sum()} ({y_val.mean()*100:.1f}%)")
    print(f"Test       : {X_test.shape[0]} exemples  | Diabétiques: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    print(f"Total      : {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")

    # Preprocessing — fit sur train uniquement
    preprocessor = get_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed   = preprocessor.transform(X_val)
    X_test_processed  = preprocessor.transform(X_test)

    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, preprocessor