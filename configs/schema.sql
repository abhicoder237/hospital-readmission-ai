-- configs/schema.sql
-- Hospital Readmission Risk Engine — Database Schema

-- ============================================
-- TABLE 1: patients
-- Ek row = ek unique patient
-- ============================================
CREATE TABLE IF NOT EXISTS patients (
    patient_id      SERIAL PRIMARY KEY,
    first_name      VARCHAR(50) NOT NULL,
    last_name       VARCHAR(50) NOT NULL,
    date_of_birth   DATE NOT NULL,
    gender          VARCHAR(10) CHECK (gender IN ('M', 'F', 'Other')),
    insurance_type  VARCHAR(30),  -- Medicare, Medicaid, Private
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- TABLE 2: admissions
-- Ek row = ek hospital visit
-- ============================================
CREATE TABLE IF NOT EXISTS admissions (
    admission_id        SERIAL PRIMARY KEY,
    patient_id          INT NOT NULL REFERENCES patients(patient_id),
    admit_time          TIMESTAMP NOT NULL,
    discharge_time      TIMESTAMP,
    admission_type      VARCHAR(30),  -- EMERGENCY, ELECTIVE, URGENT
    discharge_location  VARCHAR(50),  -- HOME, SNF, REHAB
    hospital_expire_flag BOOLEAN DEFAULT FALSE,  -- patient died?
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index banao — queries fast hongi
CREATE INDEX IF NOT EXISTS idx_admissions_patient
    ON admissions(patient_id);
CREATE INDEX IF NOT EXISTS idx_admissions_admit_time
    ON admissions(admit_time);

-- ============================================
-- TABLE 3: diagnoses
-- Ek admission mein multiple diagnoses ho sakti hain
-- ============================================
CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id    SERIAL PRIMARY KEY,
    admission_id    INT NOT NULL REFERENCES admissions(admission_id),
    icd_code        VARCHAR(10) NOT NULL,  -- e.g. E11.9 = Type 2 Diabetes
    icd_version     INT CHECK (icd_version IN (9, 10)),
    seq_num         INT,  -- 1 = primary diagnosis
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_diagnoses_admission
    ON diagnoses(admission_id);

-- ============================================
-- TABLE 4: lab_results
-- Blood tests, glucose, creatinine etc.
-- ============================================
CREATE TABLE IF NOT EXISTS lab_results (
    lab_id          SERIAL PRIMARY KEY,
    admission_id    INT NOT NULL REFERENCES admissions(admission_id),
    lab_name        VARCHAR(100) NOT NULL,  -- 'Glucose', 'Creatinine'
    value           NUMERIC(10, 3),
    unit            VARCHAR(20),            -- 'mg/dL', 'mEq/L'
    normal_low      NUMERIC(10, 3),
    normal_high     NUMERIC(10, 3),
    chart_time      TIMESTAMP,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_labs_admission
    ON lab_results(admission_id);

-- ============================================
-- TABLE 5: medications
-- Har admission mein jo drugs diye gaye
-- ============================================
CREATE TABLE IF NOT EXISTS medications (
    medication_id   SERIAL PRIMARY KEY,
    admission_id    INT NOT NULL REFERENCES admissions(admission_id),
    drug_name       VARCHAR(100) NOT NULL,
    dose            VARCHAR(50),
    route           VARCHAR(30),   -- IV, ORAL, IM
    start_time      TIMESTAMP,
    end_time        TIMESTAMP,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_medications_admission
    ON medications(admission_id);

-- ============================================
-- TABLE 6: readmissions  ← YEH HAI HAMARA TARGET
-- Yeh hum khud calculate karenge admissions se
-- ============================================
CREATE TABLE IF NOT EXISTS readmissions (
    readmission_id      SERIAL PRIMARY KEY,
    admission_id        INT NOT NULL REFERENCES admissions(admission_id),
    readmitted          SMALLINT NOT NULL CHECK (readmitted IN (0, 1)),
    days_to_readmission INT,   -- NULL if readmitted=0
    next_admission_id   INT REFERENCES admissions(admission_id),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);