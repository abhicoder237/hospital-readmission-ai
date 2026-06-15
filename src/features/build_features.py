# src/features/build_features.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from sqlalchemy import text
from src.utils.db_connection import get_engine


def extract_features():
    """
    SQL se saare features ek saath nikalta hai.
    Yeh query 6 tables ko join karke ek flat feature table banati hai.
    """
    print("📊 Extracting features from database...")

    query = """
    SELECT
        -- ── IDENTIFIERS ──────────────────────────────
        a.admission_id,
        a.patient_id,

        -- ── PATIENT FEATURES ─────────────────────────
        EXTRACT(YEAR FROM AGE(a.admit_time, p.date_of_birth))
                                        AS age,
        CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END
                                        AS gender_male,
        CASE WHEN p.insurance_type = 'Medicare'  THEN 1 ELSE 0 END
                                        AS insurance_medicare,
        CASE WHEN p.insurance_type = 'Medicaid'  THEN 1 ELSE 0 END
                                        AS insurance_medicaid,
        CASE WHEN p.insurance_type = 'Self-Pay'  THEN 1 ELSE 0 END
                                        AS insurance_selfpay,

        -- ── ADMISSION FEATURES ───────────────────────
        EXTRACT(EPOCH FROM (a.discharge_time - a.admit_time))
            / 86400.0                   AS length_of_stay,
        CASE WHEN a.admission_type = 'EMERGENCY' THEN 1 ELSE 0 END
                                        AS emergency_admission,
        CASE WHEN a.admission_type = 'URGENT'    THEN 1 ELSE 0 END
                                        AS urgent_admission,
        CASE WHEN a.discharge_location = 'HOME'  THEN 1 ELSE 0 END
                                        AS discharged_home,
        CASE WHEN a.discharge_location = 'SNF'   THEN 1 ELSE 0 END
                                        AS discharged_snf,

        -- ── DIAGNOSIS FEATURES ───────────────────────
        COALESCE(diag.n_diagnoses, 0)   AS n_diagnoses,
        COALESCE(diag.has_diabetes, 0)  AS has_diabetes,
        COALESCE(diag.has_heart_failure,0) AS has_heart_failure,
        COALESCE(diag.has_copd, 0)      AS has_copd,
        COALESCE(diag.has_kidney_disease,0) AS has_kidney_disease,
        COALESCE(diag.has_sepsis, 0)    AS has_sepsis,

        -- ── LAB FEATURES ─────────────────────────────
        COALESCE(lab.avg_glucose, 100)  AS avg_glucose,
        COALESCE(lab.max_creatinine, 1) AS max_creatinine,
        COALESCE(lab.min_hemoglobin,12) AS min_hemoglobin,
        COALESCE(lab.avg_sodium, 140)   AS avg_sodium,
        COALESCE(lab.n_abnormal_labs, 0) AS n_abnormal_labs,
        COALESCE(lab.n_labs, 0)         AS n_labs,

        -- ── MEDICATION FEATURES ──────────────────────
        COALESCE(med.n_medications, 0)  AS n_medications,
        COALESCE(med.has_insulin, 0)    AS has_insulin,
        COALESCE(med.has_iv_drugs, 0)   AS has_iv_drugs,

        -- ── PREVIOUS ADMISSIONS ──────────────────────
        COALESCE(prev.n_prev_admissions, 0) AS n_prev_admissions,

        -- ── TARGET VARIABLE ──────────────────────────
        COALESCE(r.readmitted, 0)       AS readmitted

    FROM admissions a

    -- Patient info
    JOIN patients p ON a.patient_id = p.patient_id

    -- Diagnosis summary
    LEFT JOIN (
        SELECT
            admission_id,
            COUNT(*)                                AS n_diagnoses,
            MAX(CASE WHEN icd_code = 'E11.9' THEN 1 ELSE 0 END)
                                                    AS has_diabetes,
            MAX(CASE WHEN icd_code = 'I50.9' THEN 1 ELSE 0 END)
                                                    AS has_heart_failure,
            MAX(CASE WHEN icd_code = 'J44.1' THEN 1 ELSE 0 END)
                                                    AS has_copd,
            MAX(CASE WHEN icd_code = 'N18.3' THEN 1 ELSE 0 END)
                                                    AS has_kidney_disease,
            MAX(CASE WHEN icd_code = 'A41.9' THEN 1 ELSE 0 END)
                                                    AS has_sepsis
        FROM diagnoses
        GROUP BY admission_id
    ) diag ON a.admission_id = diag.admission_id

    -- Lab summary
    LEFT JOIN (
        SELECT
            admission_id,
            COUNT(*)                                AS n_labs,
            AVG(CASE WHEN lab_name = 'Glucose'
                THEN value END)                     AS avg_glucose,
            MAX(CASE WHEN lab_name = 'Creatinine'
                THEN value END)                     AS max_creatinine,
            MIN(CASE WHEN lab_name = 'Hemoglobin'
                THEN value END)                     AS min_hemoglobin,
            AVG(CASE WHEN lab_name = 'Sodium'
                THEN value END)                     AS avg_sodium,
            SUM(CASE
                WHEN value < normal_low
                  OR value > normal_high
                THEN 1 ELSE 0 END)                  AS n_abnormal_labs
        FROM lab_results
        GROUP BY admission_id
    ) lab ON a.admission_id = lab.admission_id

    -- Medication summary
    LEFT JOIN (
        SELECT
            admission_id,
            COUNT(*)                                AS n_medications,
            MAX(CASE WHEN drug_name = 'Insulin'
                THEN 1 ELSE 0 END)                  AS has_insulin,
            MAX(CASE WHEN route = 'IV'
                THEN 1 ELSE 0 END)                  AS has_iv_drugs
        FROM medications
        GROUP BY admission_id
    ) med ON a.admission_id = med.admission_id

    -- Previous admissions count
    LEFT JOIN (
        SELECT
            a2.admission_id,
            COUNT(a3.admission_id)                  AS n_prev_admissions
        FROM admissions a2
        LEFT JOIN admissions a3
            ON  a2.patient_id  = a3.patient_id
            AND a3.admit_time  < a2.admit_time
        GROUP BY a2.admission_id
    ) prev ON a.admission_id = prev.admission_id

    -- Target label
    LEFT JOIN readmissions r ON a.admission_id = r.admission_id

    -- Expired patients exclude karo
    WHERE a.hospital_expire_flag = FALSE;
    """

    engine = get_engine()
    df = pd.read_sql(query, engine)
    print(f"   ✅ {len(df)} rows, {len(df.columns)} features extracted")
    return df


def clean_features(df):
    """
    Data cleaning steps:
    1. Missing values handle karo
    2. Outliers fix karo
    3. Data types correct karo
    """
    print("🧹 Cleaning features...")
    original_shape = df.shape

    # ── STEP 1: Missing values check ──────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"   ⚠️  Missing values found:")
        print(missing)
    else:
        print("   ✅ No missing values!")

    # ── STEP 2: Outlier treatment ─────────────────────
    # Medical domain knowledge se limits set karo
    outlier_limits = {
        'age':             (18,  95),
        'length_of_stay':  (0,   30),
        'avg_glucose':     (40,  400),
        'max_creatinine':  (0.3, 15),
        'min_hemoglobin':  (3,   20),
        'avg_sodium':      (120, 160),
        'n_diagnoses':     (0,   20),
        'n_medications':   (0,   20),
        'n_labs':          (0,   30),
        'n_abnormal_labs': (0,   20),
    }

    for col, (low, high) in outlier_limits.items():
        if col in df.columns:
            before = df[col].between(low, high).sum()
            df[col] = df[col].clip(low, high)
            after  = df[col].between(low, high).sum()

    print(f"   ✅ Outliers clipped to medical valid ranges")

    # ── STEP 3: Data types ────────────────────────────
    int_cols = [
        'gender_male', 'insurance_medicare', 'insurance_medicaid',
        'insurance_selfpay', 'emergency_admission', 'urgent_admission',
        'discharged_home', 'discharged_snf', 'has_diabetes',
        'has_heart_failure', 'has_copd', 'has_kidney_disease',
        'has_sepsis', 'has_insulin', 'has_iv_drugs',
        'n_diagnoses', 'n_medications', 'n_labs',
        'n_abnormal_labs', 'n_prev_admissions', 'readmitted'
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    float_cols = [
        'age', 'length_of_stay', 'avg_glucose',
        'max_creatinine', 'min_hemoglobin', 'avg_sodium'
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    print(f"   ✅ Shape: {original_shape} → {df.shape}")
    return df


def split_and_scale(df):
    """
    Train-test split aur feature scaling.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib

    print("✂️  Splitting and scaling...")

    # ID columns aur target alag karo
    drop_cols = ['admission_id', 'patient_id', 'readmitted']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df['readmitted']

    print(f"   Features: {len(feature_cols)}")
    print(f"   Class distribution: {y.value_counts().to_dict()}")

    # ── Train-Test Split ──────────────────────────────
    # 80% train, 20% test
    # stratify=y → class ratio same rakho dono mein
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y        # IMPORTANT: imbalanced data mein zaroori
    )

    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Feature Scaling ───────────────────────────────
    # Kyun? Age=65, Glucose=180 → same scale pe laao
    # Sirf numerical columns scale karo
    num_cols = [
        'age', 'length_of_stay', 'avg_glucose',
        'max_creatinine', 'min_hemoglobin', 'avg_sodium',
        'n_diagnoses', 'n_medications', 'n_labs',
        'n_abnormal_labs', 'n_prev_admissions'
    ]
    num_cols = [c for c in num_cols if c in X.columns]

    scaler = StandardScaler()

    # IMPORTANT: fit sirf train data pe, transform dono pe
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    # Scaler save karo — prediction ke time bhi chahiye
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("   ✅ Scaler saved → models/scaler.pkl")

    return X_train, X_test, y_train, y_test, feature_cols


def run_preprocessing():
    """Poora preprocessing pipeline."""
    print("=" * 50)
    print("🚀 Preprocessing Pipeline")
    print("=" * 50)

    # Step 1: Extract
    df = extract_features()

    # Step 2: Clean
    df = clean_features(df)

    # Step 3: Save processed data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/features.csv', index=False)
    print(f"\n💾 Saved → data/processed/features.csv")

    # Step 4: Split & Scale
    X_train, X_test, y_train, y_test, feature_cols = split_and_scale(df)

    # Step 5: Save splits
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv',  index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv',  index=False)

    print("\n💾 Saved splits:")
    print("   data/processed/X_train.csv")
    print("   data/processed/X_test.csv")
    print("   data/processed/y_train.csv")
    print("   data/processed/y_test.csv")

    print("\n" + "=" * 50)
    print("✅ Preprocessing Complete!")
    print("=" * 50)

    return X_train, X_test, y_train, y_test, feature_cols


if __name__ == "__main__":
    run_preprocessing()