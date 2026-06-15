# src/data/load_data.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from sqlalchemy import text
from src.utils.db_connection import get_engine
from src.data.generate_data import (
    generate_patients, generate_admissions,
    generate_diagnoses, generate_labs, generate_medications
)


def load_patients(engine):
    print("👤 Generating 1000 patients...")
    df = generate_patients(1000)
    df.to_sql('patients', engine, if_exists='append', index=False)
    print(f"   ✅ {len(df)} patients loaded")
    return df


def load_admissions(engine, patients_df):
    print("🏥 Generating admissions...")
    with engine.connect() as conn:
        result     = conn.execute(
            text("SELECT patient_id FROM patients ORDER BY patient_id")
        )
        patient_ids = [row[0] for row in result]

    df = generate_admissions(patient_ids, patients_df)

    # DB mein sirf valid columns daalo
    db_cols = [
        'patient_id', 'admit_time', 'discharge_time',
        'admission_type', 'discharge_location',
        'hospital_expire_flag'
    ]
    df[db_cols].to_sql(
        'admissions', engine,
        if_exists='append', index=False
    )
    print(f"   ✅ {len(df)} admissions loaded")
    return df  # _risk aur _icds ke saath return karo


def load_diagnoses(engine, admissions_df):
    print("🔬 Generating diagnoses...")

    # DB se admission_ids lo
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT admission_id FROM admissions ORDER BY admission_id")
        )
        adm_ids = [row[0] for row in result]

    # admissions_df mein admission_id add karo
    admissions_df = admissions_df.copy()
    admissions_df['admission_id'] = adm_ids

    df = generate_diagnoses(admissions_df)
    df.to_sql('diagnoses', engine, if_exists='append', index=False)
    print(f"   ✅ {len(df)} diagnoses loaded")


def load_labs(engine, admissions_df):
    print("🧪 Generating lab results...")

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT admission_id FROM admissions ORDER BY admission_id")
        )
        adm_ids = [row[0] for row in result]

    admissions_df = admissions_df.copy()
    admissions_df['admission_id'] = adm_ids

    df = generate_labs(admissions_df)
    df.to_sql('lab_results', engine, if_exists='append', index=False)
    print(f"   ✅ {len(df)} lab results loaded")


def load_medications(engine):
    print("💊 Generating medications...")
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT admission_id FROM admissions")
        )
        admission_ids = [row[0] for row in result]

    df = generate_medications(admission_ids)
    df.to_sql('medications', engine, if_exists='append', index=False)
    print(f"   ✅ {len(df)} medications loaded")


def calculate_readmissions(engine):
    print("🎯 Calculating readmission labels...")

    query = """
    INSERT INTO readmissions
        (admission_id, readmitted,
         days_to_readmission, next_admission_id)
    SELECT
        a1.admission_id,
        CASE
            WHEN a2.admission_id IS NOT NULL
             AND a2.admit_time <=
                 a1.discharge_time + INTERVAL '30 days'
            THEN 1 ELSE 0
        END AS readmitted,
        CASE
            WHEN a2.admission_id IS NOT NULL
            THEN EXTRACT(DAY FROM
                 a2.admit_time - a1.discharge_time)::INT
            ELSE NULL
        END AS days_to_readmission,
        a2.admission_id AS next_admission_id
    FROM admissions a1
    LEFT JOIN admissions a2
        ON  a1.patient_id  = a2.patient_id
        AND a2.admit_time  > a1.discharge_time
        AND a2.admit_time  = (
            SELECT MIN(a3.admit_time)
            FROM   admissions a3
            WHERE  a3.patient_id = a1.patient_id
            AND    a3.admit_time > a1.discharge_time
        )
    WHERE a1.hospital_expire_flag = FALSE;
    """

    with engine.connect() as conn:
        with conn.begin():
            conn.execute(text(query))

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT
                COUNT(*)                         AS total,
                SUM(readmitted)                  AS readmitted,
                ROUND(AVG(readmitted) * 100, 1)  AS rate
            FROM readmissions
        """))
        row = result.fetchone()
        print(f"   ✅ {row[0]} records | "
              f"{row[1]} readmitted | "
              f"{row[2]}% rate")


def run_pipeline():
    print("=" * 50)
    print("🚀 Hospital Data Ingestion Pipeline")
    print("=" * 50)

    engine = get_engine()

    patients_df    = load_patients(engine)
    admissions_df  = load_admissions(engine, patients_df)
    load_diagnoses(engine, admissions_df)
    load_labs(engine, admissions_df)
    load_medications(engine)
    calculate_readmissions(engine)

    print("=" * 50)
    print("✅ Pipeline complete!")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()