from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "credit_card_model.joblib"
METRICS_PATH = REPORT_DIR / "model_metrics.csv"
SUMMARY_PATH = REPORT_DIR / "training_summary.json"

TARGET_COLUMN = "APPROVED"

FEATURE_COLUMNS = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_FAM_MEMBERS",
    "OCCUPATION_TYPE",
    "MONTHS_WITH_OVERDUE",
    "CREDIT_STATUS_SCORE",
]

NUMERIC_COLUMNS = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_FAM_MEMBERS",
    "MONTHS_WITH_OVERDUE",
    "CREDIT_STATUS_SCORE",
]

CATEGORICAL_COLUMNS = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

FORM_OPTIONS = {
    "CODE_GENDER": ["F", "M"],
    "FLAG_OWN_CAR": ["N", "Y"],
    "FLAG_OWN_REALTY": ["Y", "N"],
    "NAME_INCOME_TYPE": [
        "Working",
        "Commercial associate",
        "Pensioner",
        "State servant",
        "Student",
    ],
    "NAME_EDUCATION_TYPE": [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ],
    "NAME_FAMILY_STATUS": [
        "Married",
        "Single / not married",
        "Civil marriage",
        "Separated",
        "Widow",
    ],
    "NAME_HOUSING_TYPE": [
        "House / apartment",
        "With parents",
        "Municipal apartment",
        "Rented apartment",
        "Office apartment",
        "Co-op apartment",
    ],
    "OCCUPATION_TYPE": [
        "Laborers",
        "Core staff",
        "Sales staff",
        "Managers",
        "Drivers",
        "High skill tech staff",
        "Accountants",
        "Medicine staff",
        "Security staff",
        "Unknown",
    ],
}

DEFAULT_FORM_VALUES = {
    "CODE_GENDER": "F",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "CNT_CHILDREN": 0,
    "AMT_INCOME_TOTAL": 180000,
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -2500,
    "CNT_FAM_MEMBERS": 2,
    "OCCUPATION_TYPE": "Core staff",
    "MONTHS_WITH_OVERDUE": 0,
    "CREDIT_STATUS_SCORE": 0,
}
