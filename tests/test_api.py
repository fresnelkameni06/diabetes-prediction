import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# ─── Données de test ───────────────────────────────────────────
valid_patient = {
    "age":                 45.0,
    "bmi":                 28.5,
    "HbA1c_level":         6.5,
    "blood_glucose_level": 140,
    "hypertension":        1,
    "heart_disease":       0,
    "gender":              "Male",
    "smoking_history":     "never"
}

# ─── Tests /health ─────────────────────────────────────────────
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

# ─── Tests /predict ────────────────────────────────────────────
def test_predict_valid_patient():
    response = client.post("/predict", json=valid_patient)
    assert response.status_code == 200
    data = response.json()
    assert "prediction"   in data
    assert "probability"  in data
    assert "label"        in data
    assert "confidence"   in data
    assert "top_factors"  in data

def test_predict_returns_binary():
    response = client.post("/predict", json=valid_patient)
    data = response.json()
    assert data["prediction"] in [0, 1]

def test_predict_probability_range():
    response = client.post("/predict", json=valid_patient)
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0

def test_predict_label_valid():
    response = client.post("/predict", json=valid_patient)
    data = response.json()
    assert data["label"] in ["Diabetique", "Non-Diabetique"]

def test_predict_top_factors():
    response = client.post("/predict", json=valid_patient)
    data = response.json()
    assert len(data["top_factors"]) == 3
    for factor in data["top_factors"]:
        assert "feature"   in factor
        assert "shap"      in factor
        assert "direction" in factor

def test_predict_invalid_age():
    invalid = valid_patient.copy()
    invalid["age"] = -5
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422

def test_predict_invalid_bmi():
    invalid = valid_patient.copy()
    invalid["bmi"] = 200
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422

def test_predict_missing_field():
    incomplete = valid_patient.copy()
    del incomplete["age"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422

def test_predict_diabetic_patient():
    diabetic = {
        "age":                 65.0,
        "bmi":                 38.0,
        "HbA1c_level":         8.5,
        "blood_glucose_level": 250,
        "hypertension":        1,
        "heart_disease":       1,
        "gender":              "Female",
        "smoking_history":     "current"
    }
    response = client.post("/predict", json=diabetic)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1