# api/main.py

import os
import sys

# ── Path fix — Render pe zaroori hai ─────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)  # Working directory = project root

import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
from datetime import datetime


# ============================================
# LOGGING SETUP
# ============================================
os.makedirs('api', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# ============================================
# APP SETUP
# ============================================
app = FastAPI(
    title="Hospital Readmission Risk API",
    description="""
    AI-powered 30-day readmission risk prediction.

    ## Endpoints
    * **POST /predict** — Predict readmission risk
    * **GET /health**   — Health check
    * **GET /model-info** — Model details
    """,
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# MODEL MANAGER
# ============================================
class ModelManager:
    model        = None
    explainer    = None
    feature_cols = None
    scaler       = None
    loaded       = False

    @classmethod
    def load(cls):
        try:
            logger.info("Loading model artifacts...")

            # Model paths — BASE_DIR se relative
            model_path   = os.path.join(BASE_DIR, 'models', 'readmission_model.pkl')
            scaler_path  = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
            feature_path = os.path.join(BASE_DIR, 'models', 'feature_cols.pkl')

            logger.info(f"Looking for model at: {model_path}")
            logger.info(f"File exists: {os.path.exists(model_path)}")

            cls.model        = joblib.load(model_path)
            cls.feature_cols = joblib.load(feature_path)
            cls.scaler       = joblib.load(scaler_path)
            cls.explainer    = shap.TreeExplainer(cls.model)
            cls.loaded       = True

            logger.info("✅ All artifacts loaded successfully")
            logger.info(f"   Features: {len(cls.feature_cols)}")

        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            logger.error(f"   BASE_DIR: {BASE_DIR}")
            logger.error(f"   Files in models/: {os.listdir(os.path.join(BASE_DIR, 'models')) if os.path.exists(os.path.join(BASE_DIR, 'models')) else 'folder not found'}")
            cls.loaded = False


@app.on_event("startup")
async def startup_event():
    ModelManager.load()


# ============================================
# REQUEST / RESPONSE MODELS
# ============================================
class PatientFeatures(BaseModel):
    age:                  float = Field(..., ge=18,  le=95)
    gender_male:          int   = Field(..., ge=0,   le=1)
    insurance_medicare:   int   = Field(0,   ge=0,   le=1)
    insurance_medicaid:   int   = Field(0,   ge=0,   le=1)
    insurance_selfpay:    int   = Field(0,   ge=0,   le=1)
    length_of_stay:       float = Field(..., ge=0,   le=30)
    emergency_admission:  int   = Field(0,   ge=0,   le=1)
    urgent_admission:     int   = Field(0,   ge=0,   le=1)
    discharged_home:      int   = Field(0,   ge=0,   le=1)
    discharged_snf:       int   = Field(0,   ge=0,   le=1)
    n_diagnoses:          int   = Field(..., ge=0,   le=20)
    has_diabetes:         int   = Field(0,   ge=0,   le=1)
    has_heart_failure:    int   = Field(0,   ge=0,   le=1)
    has_copd:             int   = Field(0,   ge=0,   le=1)
    has_kidney_disease:   int   = Field(0,   ge=0,   le=1)
    has_sepsis:           int   = Field(0,   ge=0,   le=1)
    avg_glucose:          float = Field(100.0, ge=40, le=400)
    max_creatinine:       float = Field(1.0,  ge=0.3, le=15)
    min_hemoglobin:       float = Field(12.0, ge=3,   le=20)
    avg_sodium:           float = Field(140.0,ge=120, le=160)
    n_abnormal_labs:      int   = Field(0,   ge=0,   le=20)
    n_labs:               int   = Field(0,   ge=0,   le=30)
    n_medications:        int   = Field(0,   ge=0,   le=20)
    has_insulin:          int   = Field(0,   ge=0,   le=1)
    has_iv_drugs:         int   = Field(0,   ge=0,   le=1)
    n_prev_admissions:    int   = Field(0,   ge=0,   le=20)

    model_config = {'protected_namespaces': ()}

    class Config:
        json_schema_extra = {
            "example": {
                "age": 72,
                "gender_male": 1,
                "insurance_medicare": 1,
                "insurance_medicaid": 0,
                "insurance_selfpay": 0,
                "length_of_stay": 7,
                "emergency_admission": 1,
                "urgent_admission": 0,
                "discharged_home": 0,
                "discharged_snf": 1,
                "n_diagnoses": 4,
                "has_diabetes": 1,
                "has_heart_failure": 1,
                "has_copd": 0,
                "has_kidney_disease": 1,
                "has_sepsis": 0,
                "avg_glucose": 185.0,
                "max_creatinine": 2.1,
                "min_hemoglobin": 9.5,
                "avg_sodium": 138.0,
                "n_abnormal_labs": 3,
                "n_labs": 6,
                "n_medications": 5,
                "has_insulin": 1,
                "has_iv_drugs": 1,
                "n_prev_admissions": 2
            }
        }


class RiskFactor(BaseModel):
    feature:   str
    impact:    float
    direction: str


class PredictionResponse(BaseModel):
    model_config = {'protected_namespaces': ()}

    patient_id:       Optional[str]
    risk_score:       float
    risk_percentage:  str
    risk_level:       str
    risk_color:       str
    readmission_prob: str
    top_factors:      list[RiskFactor]
    recommendation:   str
    timestamp:        str
    model_version:    str


# ============================================
# HELPERS
# ============================================
def get_risk_level(prob: float):
    if prob >= 0.60:
        return "HIGH",   "#e74c3c", "🔴"
    elif prob >= 0.35:
        return "MEDIUM", "#f59e0b", "🟡"
    else:
        return "LOW",    "#2ecc71", "🟢"


def get_recommendation(risk_level: str):
    if risk_level == "HIGH":
        return (
            "⚠️ HIGH RISK: Schedule follow-up within 7 days. "
            "Consider care coordinator assignment and "
            "medication reconciliation before discharge."
        )
    elif risk_level == "MEDIUM":
        return (
            "⚡ MEDIUM RISK: Schedule follow-up within 14 days. "
            "Provide detailed discharge instructions and "
            "ensure patient has support at home."
        )
    else:
        return (
            "✅ LOW RISK: Standard discharge protocol. "
            "Schedule routine follow-up within 30 days."
        )


def prepare_features(patient: PatientFeatures) -> pd.DataFrame:
    feature_dict = patient.model_dump()
    df = pd.DataFrame([feature_dict])
    df = df[ModelManager.feature_cols]

    num_cols = [
        'age', 'length_of_stay', 'avg_glucose',
        'max_creatinine', 'min_hemoglobin', 'avg_sodium',
        'n_diagnoses', 'n_medications', 'n_labs',
        'n_abnormal_labs', 'n_prev_admissions'
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    df[num_cols] = ModelManager.scaler.transform(df[num_cols])

    return df


# ============================================
# ENDPOINTS
# ============================================
@app.get("/")
async def root():
    return {
        "message":      "Hospital Readmission Risk API 🏥",
        "version":      "1.0.0",
        "status":       "running",
        "model_loaded": ModelManager.loaded,
        "docs":         "/docs"
    }


@app.get("/health")
async def health_check():
    return {
        "status":       "healthy" if ModelManager.loaded else "degraded",
        "model_loaded": ModelManager.loaded,
        "timestamp":    datetime.now().isoformat(),
        "base_dir":     BASE_DIR,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_readmission(patient: PatientFeatures):
    # Model loaded nahi hai toh retry karo
    if not ModelManager.loaded:
        logger.warning("Model not loaded — retrying...")
        ModelManager.load()

    if not ModelManager.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please try again in a moment."
        )

    try:
        logger.info("Prediction request received")

        X = prepare_features(patient)

        risk_prob = float(
            ModelManager.model.predict_proba(X)[0][1]
        )

        shap_vals      = ModelManager.explainer.shap_values(X)[0]
        feature_impact = pd.Series(
            shap_vals,
            index=ModelManager.feature_cols
        ).sort_values(key=abs, ascending=False)

        top_factors = []
        for feat, impact in feature_impact.head(5).items():
            top_factors.append(RiskFactor(
                feature   = feat.replace('_', ' ').title(),
                impact    = round(float(impact), 4),
                direction = ("Increases Risk"
                             if impact > 0
                             else "Decreases Risk")
            ))

        risk_level, color, emoji = get_risk_level(risk_prob)
        recommendation = get_recommendation(risk_level)

        logger.info(f"Prediction done: {risk_level} | {risk_prob:.3f}")

        return PredictionResponse(
            patient_id       = None,
            risk_score       = round(risk_prob, 4),
            risk_percentage  = f"{risk_prob * 100:.1f}%",
            risk_level       = f"{emoji} {risk_level}",
            risk_color       = color,
            readmission_prob = (
                f"{risk_prob * 100:.1f}% chance of "
                f"readmission within 30 days"
            ),
            top_factors      = top_factors,
            recommendation   = recommendation,
            timestamp        = datetime.now().isoformat(),
            model_version    = "v1.0.0-xgboost"
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model-info")
async def model_info():
    if not ModelManager.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type":  "XGBoost Classifier",
        "features":    ModelManager.feature_cols,
        "n_features":  len(ModelManager.feature_cols),
        "version":     "1.0.0",
        "trained_on":  "Synthetic hospital data",
        "target":      "30-day readmission",
        "recall":      "73.2%",
        "roc_auc":     "62.8%",
    }