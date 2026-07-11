from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_COLUMNS,
    REPORT_DIR,
    SUMMARY_PATH,
    TARGET_COLUMN,
)
from .data_utils import load_training_dataset


def _xgboost_model():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    return XGBClassifier(
        eval_metric="logloss",
        learning_rate=0.08,
        max_depth=4,
        n_estimators=140,
        random_state=42,
        subsample=0.9,
    )


def candidate_models() -> dict[str, object]:
    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=400, class_weight="balanced", solver="liblinear"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=70,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced",
        ),
    }
    xgb = _xgboost_model()
    if xgb is not None:
        models["XGBoost"] = xgb
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=70, random_state=42)
    return models


def build_pipeline(model: object) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def _evaluate(name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    predictions = pipeline.predict(x_test)
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    }


def train_and_save_model(model_path: Path = MODEL_PATH) -> dict[str, object]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataset, source = load_training_dataset()
    x = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.22,
        random_state=42,
        stratify=y,
    )

    rows = []
    trained_pipelines: dict[str, Pipeline] = {}
    for name, model in candidate_models().items():
        pipeline = build_pipeline(model)
        pipeline.fit(x_train, y_train)
        metrics = _evaluate(name, pipeline, x_test, y_test)
        rows.append(metrics)
        trained_pipelines[name] = pipeline

    metrics_df = pd.DataFrame(rows).sort_values(["f1_score", "accuracy"], ascending=False)
    best_name = str(metrics_df.iloc[0]["model"])
    best_pipeline = trained_pipelines[best_name]

    bundle = {
        "pipeline": best_pipeline,
        "best_model": best_name,
        "metrics": metrics_df.to_dict(orient="records"),
        "feature_columns": FEATURE_COLUMNS,
        "data_source": source,
        "target": TARGET_COLUMN,
    }

    joblib.dump(bundle, model_path)
    metrics_df.to_csv(METRICS_PATH, index=False)
    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "best_model": best_name,
                "data_source": source,
                "training_rows": int(len(dataset)),
                "approval_rate": round(float(y.mean()), 4),
                "metrics": bundle["metrics"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return bundle


def load_model(model_path: Path = MODEL_PATH) -> dict[str, object]:
    if not model_path.exists():
        return train_and_save_model(model_path)
    return joblib.load(model_path)


def predict_approval(bundle: dict[str, object], features: pd.DataFrame) -> dict[str, object]:
    pipeline: Pipeline = bundle["pipeline"]
    prediction = int(pipeline.predict(features)[0])

    approval_probability = None
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        approval_probability = float(pipeline.predict_proba(features)[0][1])

    if approval_probability is None:
        decision_score = float(prediction)
        approval_probability = decision_score

    return {
        "label": "Approved" if prediction == 1 else "Rejected",
        "prediction": prediction,
        "approval_probability": round(approval_probability * 100, 2),
        "model_name": str(bundle["best_model"]),
    }
