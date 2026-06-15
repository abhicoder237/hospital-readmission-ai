 # src/data/generate_data.py

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

# ============================================
# CONSTANTS
# ============================================
INSURANCE_TYPES    = ['Medicare', 'Medicaid', 'Private', 'Self-Pay']
ADMISSION_TYPES    = ['EMERGENCY', 'ELECTIVE', 'URGENT']
DISCHARGE_LOCATIONS = ['HOME', 'SNF', 'REHAB', 'LONG TERM CARE']

ICD_CODES = {
    'E11.9': 'Type 2 Diabetes',
    'I10':   'Hypertension',
    'I50.9': 'Heart Failure',
    'J18.9': 'Pneumonia',
    'N18.3': 'Chronic Kidney Disease',
    'J44.1': 'COPD',
    'I21.9': 'Heart Attack',
    'K92.1': 'GI Bleed',
    'A41.9': 'Sepsis',
    'F32.9': 'Depression',
    'E78.5': 'High Cholesterol',
    'M54.5': 'Back Pain',
}

# HIGH RISK ICD codes — yeh hote hain toh readmission zyada
HIGH_RISK_ICD = ['I50.9', 'N18.3', 'A41.9', 'I21.9', 'J44.1']

LAB_TESTS = {
    'Glucose':    (70,  100, 'mg/dL', 40,  400),
    'Creatinine': (0.6, 1.2, 'mg/dL', 0.3, 10),
    'Sodium':     (136, 145, 'mEq/L', 120, 160),
    'Potassium':  (3.5, 5.0, 'mEq/L', 2.5, 7.0),
    'Hemoglobin': (12,  17,  'g/dL',  5,   20),
    'WBC':        (4,   11,  'K/uL',  1,   30),
    'Platelets':  (150, 400, 'K/uL',  20,  800),
    'BUN':        (7,   20,  'mg/dL', 3,   100),
}

MEDICATIONS = [
    ('Metformin',    '500mg',   'ORAL'),
    ('Lisinopril',   '10mg',    'ORAL'),
    ('Furosemide',   '40mg',    'IV'),
    ('Aspirin',      '81mg',    'ORAL'),
    ('Insulin',      '10 units','IV'),
    ('Warfarin',     '5mg',     'ORAL'),
    ('Amoxicillin',  '500mg',   'ORAL'),
    ('Morphine',     '2mg',     'IV'),
    ('Heparin',      '5000u',   'IV'),
    ('Atorvastatin', '40mg',    'ORAL'),
]


# ============================================
# HELPER — Risk Score Calculator
# ============================================
def calculate_risk_score(age, n_prev_admissions,
                         has_high_risk_diagnosis,
                         n_diagnoses, insurance_type):
    """
    Patient ka readmission risk score calculate karta hai.
    Yeh score gap determine karega — high risk = chota gap.
    Score 0.0 to 1.0 ke beech.
    """
    score = 0.0

    # Age factor — elderly zyada risk
    if age >= 80:
        score += 0.30
    elif age >= 70:
        score += 0.20
    elif age >= 60:
        score += 0.10

    # Previous admissions — baar baar aane wale
    if n_prev_admissions >= 3:
        score += 0.25
    elif n_prev_admissions >= 1:
        score += 0.15

    # High risk diagnosis
    if has_high_risk_diagnosis:
        score += 0.25

    # Multiple diagnoses — sicker patient
    if n_diagnoses >= 4:
        score += 0.15
    elif n_diagnoses >= 2:
        score += 0.08

    # Insurance — self-pay kam follow-up
    if insurance_type == 'Self-Pay':
        score += 0.05
    elif insurance_type == 'Medicaid':
        score += 0.03

    return min(score, 1.0)  # Max 1.0


# ============================================
# GENERATORS
# ============================================
def generate_patients(n=1000):
    """1000 patients — age distribution realistic."""
    patients = []
    for _ in range(n):
        age = int(np.random.normal(65, 15))
        age = max(18, min(95, age))
        dob = datetime.now() - timedelta(days=age * 365)

        patients.append({
            'first_name':     fake.first_name(),
            'last_name':      fake.last_name(),
            'date_of_birth':  dob.strftime('%Y-%m-%d'),
            'gender':         random.choice(['M', 'F']),
            'insurance_type': random.choices(
                INSURANCE_TYPES,
                weights=[40, 25, 30, 5]
            )[0],
        })

    return pd.DataFrame(patients)


def generate_admissions(patient_ids, patients_df):
    """
    Risk-based admissions generate karta hai.
    High risk patients → chota gap → readmission zyada.
    """
    admissions = []
    start_date = datetime(2020, 1, 1)

    for idx, pid in enumerate(patient_ids):

        # Patient info lo
        patient    = patients_df.iloc[idx]
        age        = (datetime.now() -
                      pd.to_datetime(patient['date_of_birth'])
                     ).days // 365
        insurance  = patient['insurance_type']

        # Visits — sicker patients zyada baar aate hain
        if age >= 70:
            n_visits = random.choices(
                [1, 2, 3, 4, 5],
                weights=[15, 25, 30, 20, 10]
            )[0]
        else:
            n_visits = random.choices(
                [1, 2, 3, 4, 5],
                weights=[35, 30, 20, 10, 5]
            )[0]

        current_date      = (start_date +
                             timedelta(days=random.randint(0, 180)))
        n_prev_admissions = 0

        for visit_num in range(n_visits):

            # Random diagnoses for this visit
            n_diag = random.choices(
                [1, 2, 3, 4, 5],
                weights=[15, 25, 30, 20, 10]
            )[0]
            visit_icds = random.sample(
                list(ICD_CODES.keys()),
                min(n_diag, len(ICD_CODES))
            )
            has_high_risk = any(
                icd in HIGH_RISK_ICD for icd in visit_icds
            )

            # Risk score calculate karo
            risk = calculate_risk_score(
                age, n_prev_admissions,
                has_high_risk, n_diag, insurance
            )

            # Length of stay — sicker = longer stay
            if risk > 0.6:
                los = random.choices(
                    range(1, 15),
                    weights=[5,8,12,15,15,12,10,8,6,4,2,1,1,1]
                )[0]
            else:
                los = random.choices(
                    range(1, 15),
                    weights=[25,20,18,12,8,6,4,3,2,1,1,0,0,0]
                )[0]

            admit_time     = current_date
            discharge_time = admit_time + timedelta(days=los)
            expired        = (risk > 0.7 and
                              random.random() < 0.04)

            admissions.append({
                'patient_id':           pid,
                'admit_time':           admit_time.strftime(
                                            '%Y-%m-%d %H:%M:%S'),
                'discharge_time':       discharge_time.strftime(
                                            '%Y-%m-%d %H:%M:%S'),
                'admission_type':       random.choices(
                    ADMISSION_TYPES,
                    weights=[70, 15, 15]
                    if risk > 0.5
                    else [40, 40, 20]
                )[0],
                'discharge_location':   (
                    'EXPIRED' if expired
                    else random.choices(
                        DISCHARGE_LOCATIONS,
                        weights=[40, 25, 20, 15]
                        if risk > 0.5
                        else [70, 10, 10, 10]
                    )[0]
                ),
                'hospital_expire_flag': expired,
                '_risk':                risk,   # temp — load mein drop hoga
                '_icds':                visit_icds,  # temp
            })

            n_prev_admissions += 1

            # ── Gap — risk se decide hoga ──────────────
            if risk > 0.65:
                # High risk → 70% chance 30 din se pehle wapas
                gap = random.choices(
                    [7, 14, 21, 28, 45, 90, 180],
                    weights=[15, 20, 20, 15, 15, 10, 5]
                )[0]
            elif risk > 0.35:
                # Medium risk → 30% chance 30 din se pehle
                gap = random.choices(
                    [7, 14, 21, 28, 45, 90, 180],
                    weights=[5, 8, 10, 12, 25, 25, 15]
                )[0]
            else:
                # Low risk → mostly 30 din ke baad
                gap = random.choices(
                    [7, 14, 21, 28, 45, 90, 180],
                    weights=[2, 3, 4, 6, 20, 35, 30]
                )[0]

            current_date = discharge_time + timedelta(days=gap)

    return pd.DataFrame(admissions)


def generate_diagnoses(admissions_df):
    """
    Admissions ke stored _icds use karta hai —
    consistent diagnoses milenge.
    """
    diagnoses = []

    for _, row in admissions_df.iterrows():
        adm_id = row['admission_id']
        icds   = row.get('_icds', [])

        if not isinstance(icds, list) or len(icds) == 0:
            icds = random.sample(list(ICD_CODES.keys()), 2)

        for seq, icd in enumerate(icds, start=1):
            diagnoses.append({
                'admission_id': adm_id,
                'icd_code':     icd,
                'icd_version':  10,
                'seq_num':      seq,
            })

    return pd.DataFrame(diagnoses)


def generate_labs(admissions_df):
    """
    Risk-based labs — high risk patients ke labs
    zyada abnormal honge.
    """
    labs      = []
    lab_names = list(LAB_TESTS.keys())

    for _, row in admissions_df.iterrows():
        adm_id = row['admission_id']
        risk   = row.get('_risk', 0.3)

        n_labs = random.randint(4, 8)
        chosen = random.sample(
            lab_names, min(n_labs, len(lab_names))
        )

        for lab in chosen:
            low, high, unit, min_val, max_val = LAB_TESTS[lab]

            # High risk → zyada abnormal labs
            abnormal_prob = 0.20 + (risk * 0.50)

            if random.random() < abnormal_prob:
                # Abnormal value
                if random.random() < 0.5:
                    # Too low
                    value = round(
                        random.uniform(min_val, low * 0.9), 2
                    )
                else:
                    # Too high
                    value = round(
                        random.uniform(high * 1.1, max_val), 2
                    )
            else:
                value = round(random.uniform(low, high), 2)

            labs.append({
                'admission_id': adm_id,
                'lab_name':     lab,
                'value':        value,
                'unit':         unit,
                'normal_low':   low,
                'normal_high':  high,
                'chart_time':   fake.date_time_between(
                    start_date='-2y',
                    end_date='now'
                ).strftime('%Y-%m-%d %H:%M:%S'),
            })

    return pd.DataFrame(labs)


def generate_medications(admission_ids):
    """Medications generate karta hai."""
    meds = []

    for adm_id in admission_ids:
        n_meds = random.randint(1, 5)
        chosen = random.sample(
            MEDICATIONS, min(n_meds, len(MEDICATIONS))
        )

        for drug, dose, route in chosen:
            start = fake.date_time_between(
                start_date='-2y', end_date='-1y'
            )
            end = start + timedelta(days=random.randint(1, 7))

            meds.append({
                'admission_id': adm_id,
                'drug_name':    drug,
                'dose':         dose,
                'route':        route,
                'start_time':   start.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time':     end.strftime('%Y-%m-%d %H:%M:%S'),
            })

    return pd.DataFrame(meds)