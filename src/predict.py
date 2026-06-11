import pickle
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import (
    NUMERIC_FEATURES, BINARY_FEATURES,
    CATEGORICAL_FEATURES, ALL_FEATURES
)

MODEL_PATH       = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
EXPLAINER_PATH   = "models/shap_explainer.pkl"

def load_artifacts():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    with open(EXPLAINER_PATH, 'rb') as f:
        explainer = pickle.load(f)
    return model, preprocessor, explainer

def predict(patient_data: dict, threshold: float = 0.1) -> dict:
    """
    Predit si un patient est diabetique.

    Args:
        patient_data : dictionnaire avec les valeurs du patient
        threshold    : seuil de decision (0.1 retenu pour XGBoost)

    Returns:
        prediction   : 0 ou 1
        probability  : probabilite d etre diabetique
        label        : Diabetique ou Non-Diabetique
        confidence   : score de confiance
        shap_values  : contribution de chaque feature
        top_factors  : top 3 features qui ont influence la prediction
    """
    model, preprocessor, explainer = load_artifacts()

    # Convertir en DataFrame
    df_patient = pd.DataFrame([patient_data])[ALL_FEATURES]

    # Preprocessing
    X_processed = preprocessor.transform(df_patient)

    # Prediction
    proba      = model.predict_proba(X_processed)[0, 1]
    prediction = int(proba >= threshold)
    label      = 'Diabetique' if prediction == 1 else 'Non-Diabetique'
    confidence = proba if prediction == 1 else 1 - proba

    # SHAP
    shap_vals = explainer.shap_values(X_processed)[0]

    # Noms des features
    cat_encoder   = preprocessor.named_transformers_['cat']
    cat_names     = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = NUMERIC_FEATURES + BINARY_FEATURES + cat_names

    # Top 3 facteurs
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap':    shap_vals,
        'impact':  np.abs(shap_vals)
    }).sort_values('impact', ascending=False)

    top_factors = []
    for _, row in shap_df.head(3).iterrows():
        direction = 'augmente le risque' if row['shap'] > 0 else 'diminue le risque'
        top_factors.append({
            'feature':   row['feature'],
            'shap':      round(float(row['shap']), 4),
            'direction': direction
        })

    return {
        'prediction':    prediction,
        'probability':   round(float(proba), 4),
        'label':         label,
        'confidence':    round(float(confidence), 4),
        'threshold':     threshold,
        'shap_values':   shap_vals.tolist(),
        'feature_names': feature_names,
        'top_factors':   top_factors
    }


if __name__ == "__main__":
    # Test rapide
    test_patient = {
        'age':                 45.0,
        'bmi':                 28.5,
        'HbA1c_level':         6.5,
        'blood_glucose_level': 140,
        'hypertension':        1,
        'heart_disease':       0,
        'gender':              'Male',
        'smoking_history':     'never'
    }

    result = predict(test_patient)
    print(f"Prediction  : {result['label']}")
    print(f"Probabilite : {result['probability']:.4f}")
    print(f"Confiance   : {result['confidence']:.1%}")
    print(f"\nTop 3 facteurs :")
    for f in result['top_factors']:
        print(f"  {f['feature']:<35} SHAP={f['shap']:>8.4f}  -> {f['direction']}")