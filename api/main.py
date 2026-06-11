import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from src.predict import predict

# ─── Application FastAPI ───────────────────────────────────────
app = FastAPI(
    title       = "Diabetes Prediction API",
    description = "API de prédiction du diabète basée sur XGBoost + SHAP",
    version     = "1.0.0"
)

# ─── Schéma d'entrée — Pydantic valide automatiquement ────────
class PatientData(BaseModel):
    age:                 float = Field(..., ge=0,   le=120, description="Age du patient")
    bmi:                 float = Field(..., ge=10,  le=100, description="Indice de masse corporelle")
    HbA1c_level:         float = Field(..., ge=3.5, le=15,  description="Taux HbA1c")
    blood_glucose_level: int   = Field(..., ge=50,  le=400, description="Glycémie")
    hypertension:        int   = Field(..., ge=0,   le=1,   description="Hypertension (0=Non, 1=Oui)")
    heart_disease:       int   = Field(..., ge=0,   le=1,   description="Maladie cardiaque (0=Non, 1=Oui)")
    gender:              str   = Field(..., description="Genre (Male/Female/Other)")
    smoking_history:     str   = Field(..., description="Tabagisme (never/former/current/not current/ever/No Info)")

    class Config:
        json_schema_extra = {
            "example": {
                "age":                 45.0,
                "bmi":                 28.5,
                "HbA1c_level":         6.5,
                "blood_glucose_level": 140,
                "hypertension":        1,
                "heart_disease":       0,
                "gender":              "Male",
                "smoking_history":     "never"
            }
        }

# ─── Schéma de sortie ─────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction:    int
    probability:   float
    label:         str
    confidence:    float
    threshold:     float
    top_factors:   list

# ─── Endpoint GET /health ─────────────────────────────────────
@app.get("/health", tags=["Health"])
def health_check():
    """Vérifie que l'API est opérationnelle."""
    return {
        "status":  "ok",
        "message": "Diabetes Prediction API is running",
        "version": "1.0.0"
    }

# ─── Endpoint POST /predict ───────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_diabetes(patient: PatientData):
    """
    Prédit si un patient est diabétique.

    - **prediction** : 0 = Non-Diabétique, 1 = Diabétique
    - **probability** : probabilité d'être diabétique (0 à 1)
    - **label** : Diabetique ou Non-Diabetique
    - **confidence** : score de confiance
    - **threshold** : seuil de décision utilisé (0.1)
    - **top_factors** : top 3 features qui ont influencé la prédiction
    """
    try:
        result = predict(patient.model_dump())
        return PredictionResponse(
            prediction  = result['prediction'],
            probability = result['probability'],
            label       = result['label'],
            confidence  = result['confidence'],
            threshold   = result['threshold'],
            top_factors = result['top_factors']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Endpoint GET / ───────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "message": "Diabetes Prediction API",
        "docs":    "http://localhost:8000/docs",
        "health":  "http://localhost:8000/health",
        "predict": "http://localhost:8000/predict"
    }