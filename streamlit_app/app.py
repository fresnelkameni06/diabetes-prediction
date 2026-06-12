import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ─── Configuration ─────────────────────────────────────────────
import os
API_URL = os.getenv("API_URL", "https://diabetes-prediction-hcpe.onrender.com")

st.set_page_config(
    page_title = "Diabetes Prediction",
    page_icon  = "🩺",
    layout     = "wide"
)

# ─── CSS personnalisé ──────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .result-diabetic {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-healthy {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .shap-positive {
        color: #f44336;
        font-weight: bold;
    }
    .shap-negative {
        color: #4caf50;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─── Titre ─────────────────────────────────────────────────────
st.markdown('<div class="main-title">🩺 Diabetes Prediction Dashboard</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Prédiction du diabète basée sur XGBoost + Explainabilité SHAP</div>',
            unsafe_allow_html=True)

# ─── Vérification API ──────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=60)
        return r.status_code == 200
    except:
        return False

api_ok = check_api()
if api_ok:
    st.success("API connectée et opérationnelle")
else:
    st.warning("API en cours de démarrage... Patientez 30 secondes et rechargez la page.")
    st.stop()

st.markdown("---")

# ─── Layout : Formulaire | Résultats ──────────────────────────
col_form, col_result = st.columns([1, 1.5], gap="large")

# ══════════════════════════════════════════════════════════════
# COLONNE GAUCHE — Formulaire de saisie
# ══════════════════════════════════════════════════════════════
with col_form:
    st.subheader("📋 Données du Patient")

    with st.form("patient_form"):

        st.markdown("**Informations générales**")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Âge (années)",
                min_value=0.0, max_value=120.0,
                value=45.0, step=1.0
            )
            bmi = st.number_input(
                "BMI (kg/m²)",
                min_value=10.0, max_value=100.0,
                value=28.5, step=0.1
            )

        with col2:
            gender = st.selectbox(
                "Genre",
                options=["Male", "Female", "Other"]
            )
            smoking_history = st.selectbox(
                "Tabagisme",
                options=["never", "former", "current",
                         "not current", "ever", "No Info"]
            )

        st.markdown("**Données cliniques**")
        col3, col4 = st.columns(2)

        with col3:
            hba1c = st.number_input(
                "HbA1c (%)",
                min_value=3.5, max_value=15.0,
                value=5.5, step=0.1,
                help="Taux d'hémoglobine glyquée — normal < 5.7%"
            )
            blood_glucose = st.number_input(
                "Glycémie (mg/dL)",
                min_value=50, max_value=400,
                value=100, step=1,
                help="Glycémie à jeun — normal < 100 mg/dL"
            )

        with col4:
            hypertension = st.radio(
                "Hypertension",
                options=[0, 1],
                format_func=lambda x: "Non" if x == 0 else "Oui",
                horizontal=True
            )
            heart_disease = st.radio(
                "Maladie cardiaque",
                options=[0, 1],
                format_func=lambda x: "Non" if x == 0 else "Oui",
                horizontal=True
            )

        st.markdown("---")
        submitted = st.form_submit_button(
            "🔍 Prédire",
            use_container_width=True,
            type="primary"
        )

# ══════════════════════════════════════════════════════════════
# COLONNE DROITE — Résultats
# ══════════════════════════════════════════════════════════════
with col_result:
    st.subheader("📊 Résultats de la Prédiction")

    if not submitted:
        st.info("Remplissez le formulaire et cliquez sur **Prédire**")

    else:
        # Appel API
        patient_data = {
            "age":                 float(age),
            "bmi":                 float(bmi),
            "HbA1c_level":         float(hba1c),
            "blood_glucose_level": int(blood_glucose),
            "hypertension":        int(hypertension),
            "heart_disease":       int(heart_disease),
            "gender":              gender,
            "smoking_history":     smoking_history
        }

        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json=patient_data,
                    timeout=30
                )
                result = response.json()

            except Exception as e:
                st.error(f"Erreur API : {e}")
                st.stop()

        # ── Résultat principal ──
        prediction  = result['prediction']
        probability = result['probability']
        confidence  = result['confidence']
        label       = result['label']
        top_factors = result['top_factors']

        if prediction == 1:
            st.markdown(f"""
            <div class="result-diabetic">
                <h2>🔴 {label}</h2>
                <p>Ce patient présente un risque élevé de diabète.</p>
                <p>Une consultation médicale est fortement recommandée.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-healthy">
                <h2>🟢 {label}</h2>
                <p>Ce patient ne présente pas de risque détecté de diabète.</p>
                <p>Un suivi régulier reste conseillé.</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Métriques ──
        st.markdown("---")
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                label = "Probabilité",
                value = f"{probability:.1%}"
            )
        with m2:
            st.metric(
                label = "Confiance",
                value = f"{confidence:.1%}"
            )
        with m3:
            st.metric(
                label = "Seuil utilisé",
                value = f"{result['threshold']}"
            )

        # ── Jauge de probabilité ──
        st.markdown("**Jauge de risque**")
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = probability * 100,
            title = {"text": "Probabilité de Diabète (%)"},
            delta = {"reference": 10},
            gauge = {
                "axis": {"range": [0, 100]},
                "bar":  {"color": "darkred" if prediction == 1 else "darkgreen"},
                "steps": [
                    {"range": [0,  10],  "color": "#e8f5e9"},
                    {"range": [10, 30],  "color": "#fff9c4"},
                    {"range": [30, 100], "color": "#ffebee"}
                ],
                "threshold": {
                    "line":  {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 10
                }
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── SHAP — Top 3 facteurs ──
        st.markdown("---")
        st.markdown("**🔍 Explainabilité — Facteurs influents**")
        st.caption("Les facteurs qui ont le plus influencé cette prédiction :")

        features = [f['feature']   for f in top_factors]
        shap_vals = [f['shap']     for f in top_factors]
        directions = [f['direction'] for f in top_factors]

        colors = ['#f44336' if s > 0 else '#4caf50' for s in shap_vals]

        fig_shap = go.Figure(go.Bar(
            x           = shap_vals,
            y           = features,
            orientation = 'h',
            marker_color = colors,
            text        = [f"{s:+.4f}" for s in shap_vals],
            textposition = 'outside'
        ))
        fig_shap.update_layout(
            title    = "Valeurs SHAP — Top 3 features",
            xaxis_title = "Impact sur la prédiction",
            height   = 250,
            margin   = dict(t=40, b=20, l=150, r=80),
            showlegend = False
        )
        fig_shap.add_vline(x=0, line_color="black", line_width=1)
        st.plotly_chart(fig_shap, use_container_width=True)

        # ── Explication textuelle ──
        st.markdown("**📝 Interprétation :**")
        for f in top_factors:
            icon = "🔴" if f['shap'] > 0 else "🟢"
            st.markdown(
                f"{icon} **{f['feature']}** : {f['direction']} "
                f"(SHAP = {f['shap']:+.4f})"
            )

        # ── Avertissement médical ──
        st.markdown("---")
        st.warning(
            "⚠️ **Avertissement** : Cette prédiction est un outil d'aide à la décision. "
            "Elle ne remplace pas un diagnostic médical professionnel."
        )