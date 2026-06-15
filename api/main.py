# api/main.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api/api.log')
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

    ## Features
    * Predict readmission risk for a patient
    * Get SHAP-based explanation of prediction
    * Health check endpoint
    """,
    version="1.0.0",
)

# CORS — Frontend se connect karne ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Production mein specific domain do
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# MODEL LOADING — Startup pe ek baar load hoga
# ============================================
class ModelManager:
    """
    Model ko memory mein rakhta hai.
    Har request pe dobara load nahi karna —
    warna slow ho jaata.
    """
    model        = None
    explainer    = None
    feature_cols = None
    scaler       = None

    @classmethod
    def load(cls):
        try:
            logger.info("Loading model artifacts...")
            cls.model        = joblib.load('models/readmission_model.pkl')
            cls.feature_cols = joblib.load('models/feature_cols.pkl')
            cls.scaler       = joblib.load('models/scaler.pkl')
            cls.explainer    = shap.TreeExplainer(cls.model)
            logger.info("✅ All artifacts loaded successfully")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise


@app.on_event("startup")
async def startup_event():
    """Server start hote hi model load karo."""
    ModelManager.load()


# ============================================
# REQUEST / RESPONSE MODELS
# ============================================
class PatientFeatures(BaseModel):
    """
    Patient ka data — doctor yeh fill karega UI mein.
    Field(...) = required field
    ge/le = greater/less than validation
    """
    # Patient info
    age:                  float = Field(..., ge=18,  le=95,
                                        description="Patient age")
    gender_male:          int   = Field(..., ge=0,   le=1,
                                        description="1=Male, 0=Female")

    # Insurance
    insurance_medicare:   int   = Field(0, ge=0, le=1)
    insurance_medicaid:   int   = Field(0, ge=0, le=1)
    insurance_selfpay:    int   = Field(0, ge=0, le=1)

    # Admission
    length_of_stay:       float = Field(..., ge=0,   le=30,
                                        description="Days in hospital")
    emergency_admission:  int   = Field(0, ge=0, le=1)
    urgent_admission:     int   = Field(0, ge=0, le=1)
    discharged_home:      int   = Field(0, ge=0, le=1)
    discharged_snf:       int   = Field(0, ge=0, le=1)

    # Diagnoses
    n_diagnoses:          int   = Field(..., ge=0, le=20)
    has_diabetes:         int   = Field(0, ge=0, le=1)
    has_heart_failure:    int   = Field(0, ge=0, le=1)
    has_copd:             int   = Field(0, ge=0, le=1)
    has_kidney_disease:   int   = Field(0, ge=0, le=1)
    has_sepsis:           int   = Field(0, ge=0, le=1)

    # Labs
    avg_glucose:          float = Field(100.0, ge=40,  le=400)
    max_creatinine:       float = Field(1.0,   ge=0.3, le=15)
    min_hemoglobin:       float = Field(12.0,  ge=3,   le=20)
    avg_sodium:           float = Field(140.0, ge=120, le=160)
    n_abnormal_labs:      int   = Field(0,     ge=0,   le=20)
    n_labs:               int   = Field(0,     ge=0,   le=30)

    # Medications
    n_medications:        int   = Field(0, ge=0, le=20)
    has_insulin:          int   = Field(0, ge=0, le=1)
    has_iv_drugs:         int   = Field(0, ge=0, le=1)

    # History
    n_prev_admissions:    int   = Field(0, ge=0, le=20)

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
    """API ka response — frontend ko yeh milega."""
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
# HELPER FUNCTIONS
# ============================================
def get_risk_level(prob: float):
    """Risk level aur color determine karta hai."""
    if prob >= 0.60:
        return "HIGH",   "#e74c3c", "🔴"
    elif prob >= 0.35:
        return "MEDIUM", "#f39c12", "🟡"
    else:
        return "LOW",    "#2ecc71", "🟢"


def get_recommendation(risk_level: str, top_factors: list):
    """Risk level se clinical recommendation."""
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
    """Patient data ko model-ready format mein convert karta hai."""
    feature_dict = patient.model_dump()
    df = pd.DataFrame([feature_dict])

    # Feature order same rakho jaise training mein tha
    df = df[ModelManager.feature_cols]

    # Numerical columns scale karo
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
    """API root — health check."""
    return {
        "message": "Hospital Readmission Risk API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs"
    }


@app.on_event("startup")
async def startup_event():
    try:
        ModelManager.load()
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Server start hoga — model load baad mein retry hoga

@app.get("/health")
async def health_check():
    """Model loaded hai ya nahi check karta hai."""
    model_loaded = ModelManager.model is not None
    return {
        "status":       "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp":    datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_readmission(patient: PatientFeatures):
    """
    Main prediction endpoint.

    Input  : Patient features
    Output : Risk score + explanation + recommendation
    """
    try:
        logger.info(f"Prediction request received")

        # Features prepare karo
        X = prepare_features(patient)

        # Risk score
        risk_prob = float(
            ModelManager.model.predict_proba(X)[0][1]
        )

        # SHAP explanation
        shap_vals    = ModelManager.explainer.shap_values(X)[0]
        feature_impact = pd.Series(
            shap_vals,
            index=ModelManager.feature_cols
        ).sort_values(key=abs, ascending=False)

        # Top 5 factors
        top_factors = []
        for feat, impact in feature_impact.head(5).items():
            top_factors.append(RiskFactor(
                feature   = feat.replace('_', ' ').title(),
                impact    = round(float(impact), 4),
                direction = ("Increases Risk"
                             if impact > 0
                             else "Decreases Risk")
            ))

        # Risk level
        risk_level, color, emoji = get_risk_level(risk_prob)

        # Recommendation
        recommendation = get_recommendation(
            risk_level, top_factors
        )

        response = PredictionResponse(
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

        logger.info(
            f"Prediction: {risk_level} | "
            f"Score: {risk_prob:.3f}"
        )
        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model-info")
async def model_info():
    """Model ki information return karta hai."""
    return {
        "model_type":    "XGBoost Classifier",
        "features":      ModelManager.feature_cols,
        "n_features":    len(ModelManager.feature_cols),
        "version":       "1.0.0",
        "trained_on":    "Synthetic hospital data",
        "target":        "30-day readmission",
    }