from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, DEFAULT_FORM_VALUES, FEATURE_COLUMNS, TARGET_COLUMN


def _status_to_risk(value: object) -> int:
    status = str(value).strip().upper()
    if status in {"C", "X", "0", ""}:
        return 0
    if status in {"1", "2", "3", "4", "5"}:
        return int(status)
    return 0


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column, default in DEFAULT_FORM_VALUES.items():
        if column not in prepared.columns:
            prepared[column] = default
    return prepared[FEATURE_COLUMNS]


def load_real_dataset(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, str]:
    app_path = data_dir / "application_record.csv"
    credit_path = data_dir / "credit_record.csv"

    if not app_path.exists() or not credit_path.exists():
        raise FileNotFoundError("application_record.csv and credit_record.csv were not found.")

    applications = pd.read_csv(app_path)
    credit = pd.read_csv(credit_path)

    if "ID" not in applications.columns or "ID" not in credit.columns or "STATUS" not in credit.columns:
        raise ValueError("Real dataset must contain ID in both files and STATUS in credit_record.csv.")

    applications = applications.drop_duplicates(subset=["ID"]).copy()
    credit = credit.copy()
    credit["RISK_VALUE"] = credit["STATUS"].map(_status_to_risk)

    credit_summary = (
        credit.groupby("ID")
        .agg(
            CREDIT_STATUS_SCORE=("RISK_VALUE", "max"),
            MONTHS_WITH_OVERDUE=("RISK_VALUE", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    credit_summary[TARGET_COLUMN] = np.where(credit_summary["CREDIT_STATUS_SCORE"] > 0, 0, 1)

    merged = applications.merge(credit_summary, on="ID", how="inner")
    features = _ensure_feature_columns(merged)
    dataset = features.copy()
    dataset[TARGET_COLUMN] = credit_summary.set_index("ID").loc[merged["ID"], TARGET_COLUMN].values
    dataset = dataset.dropna(subset=[TARGET_COLUMN]).copy()

    for column, default in DEFAULT_FORM_VALUES.items():
        dataset[column] = dataset[column].fillna(default)

    return dataset, "real Kaggle-style application_record.csv and credit_record.csv"


def generate_demo_dataset(rows: int = 1400, seed: int = 42) -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(seed)

    income_types = np.array(["Working", "Commercial associate", "Pensioner", "State servant", "Student"])
    education_types = np.array(
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ]
    )
    family_statuses = np.array(["Married", "Single / not married", "Civil marriage", "Separated", "Widow"])
    housing_types = np.array(
        [
            "House / apartment",
            "With parents",
            "Municipal apartment",
            "Rented apartment",
            "Office apartment",
            "Co-op apartment",
        ]
    )
    occupations = np.array(
        [
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
        ]
    )

    income_type = rng.choice(income_types, rows, p=[0.52, 0.22, 0.12, 0.11, 0.03])
    education = rng.choice(education_types, rows, p=[0.58, 0.26, 0.09, 0.05, 0.02])
    family = rng.choice(family_statuses, rows, p=[0.55, 0.22, 0.09, 0.08, 0.06])
    housing = rng.choice(housing_types, rows, p=[0.72, 0.1, 0.07, 0.05, 0.04, 0.02])
    occupation = rng.choice(occupations, rows)

    age_years = rng.integers(21, 67, rows)
    employed_years = np.clip(rng.normal(6, 5, rows), 0, 35)
    income = np.clip(rng.normal(185000, 85000, rows), 35000, 650000).round(-2)
    children = rng.choice([0, 1, 2, 3, 4], rows, p=[0.56, 0.22, 0.15, 0.05, 0.02])
    family_members = np.maximum(children + rng.choice([1, 2], rows, p=[0.28, 0.72]), 1)

    overdue_months = rng.choice([0, 1, 2, 3, 4, 5, 6], rows, p=[0.67, 0.12, 0.08, 0.05, 0.04, 0.025, 0.015])
    credit_status_score = np.where(
        overdue_months == 0,
        0,
        np.clip(rng.integers(1, 6, rows) + (overdue_months > 3), 1, 5),
    )

    risk_score = (
        0.000004 * (210000 - income)
        + 0.16 * overdue_months
        + 0.2 * credit_status_score
        + 0.04 * children
        - 0.055 * employed_years
        - 0.015 * (age_years - 35)
        + np.where(education == "Higher education", -0.25, 0)
        + np.where(income_type == "Pensioner", 0.22, 0)
        + np.where(housing == "Rented apartment", 0.2, 0)
        + rng.normal(0, 0.32, rows)
    )
    approved = (risk_score < 0.72).astype(int)

    dataset = pd.DataFrame(
        {
            "CODE_GENDER": rng.choice(["F", "M"], rows, p=[0.58, 0.42]),
            "FLAG_OWN_CAR": rng.choice(["N", "Y"], rows, p=[0.62, 0.38]),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], rows, p=[0.67, 0.33]),
            "CNT_CHILDREN": children,
            "AMT_INCOME_TOTAL": income,
            "NAME_INCOME_TYPE": income_type,
            "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS": family,
            "NAME_HOUSING_TYPE": housing,
            "DAYS_BIRTH": -age_years * 365,
            "DAYS_EMPLOYED": -(employed_years * 365).astype(int),
            "CNT_FAM_MEMBERS": family_members,
            "OCCUPATION_TYPE": occupation,
            "MONTHS_WITH_OVERDUE": overdue_months,
            "CREDIT_STATUS_SCORE": credit_status_score,
            TARGET_COLUMN: approved,
        }
    )

    return dataset, "generated demo dataset"


def load_training_dataset(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, str]:
    try:
        return load_real_dataset(data_dir)
    except (FileNotFoundError, ValueError):
        return generate_demo_dataset()


def coerce_form_payload(payload: dict[str, object]) -> pd.DataFrame:
    row = DEFAULT_FORM_VALUES.copy()
    row.update({key: value for key, value in payload.items() if key in row})

    numeric_columns = [
        "CNT_CHILDREN",
        "AMT_INCOME_TOTAL",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CNT_FAM_MEMBERS",
        "MONTHS_WITH_OVERDUE",
        "CREDIT_STATUS_SCORE",
    ]
    for column in numeric_columns:
        row[column] = float(row[column])

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
