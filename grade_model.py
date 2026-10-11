"""Shared model code for the Student Grade Predictor (used by app.py and tests).

Mirrors the notebook: same feature engineering, same preprocessing pipeline,
same train/test split settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "student-mat.csv"
TARGET = "G3"
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV (handles the semicolon-separated original UCI file)."""
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    return df


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Same engineered features as the notebook."""
    d = d.copy()
    d["parent_edu"] = d[["Medu", "Fedu"]].mean(axis=1)
    d["alcohol"] = d["Dalc"] + d["Walc"]
    d["log_absences"] = np.log1p(d["absences"])
    d["has_failed"] = (d["failures"] > 0).astype(int)
    return d


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categoricals."""
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def train(use_previous_grades: bool = True) -> Dict:
    """Train a Random Forest and return the model plus info the UI needs.

    use_previous_grades=False drops G1/G2 (the harder 'early warning' setting).
    """
    df = load_data()
    y = df[TARGET]
    raw = df.drop(columns=[TARGET])
    if not use_previous_grades:
        raw = raw.drop(columns=["G1", "G2"])

    X = add_features(raw)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    model = Pipeline([
        ("pre", make_preprocessor(X_tr)),
        ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                    random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    # Defaults for every raw column the user does not fill in on the form
    defaults = {}
    for col in raw.columns:
        defaults[col] = raw[col].median() if pd.api.types.is_numeric_dtype(raw[col]) else raw[col].mode().iloc[0]

    names = [n.split("__", 1)[1] for n in model.named_steps["pre"].get_feature_names_out()]
    importances = pd.Series(model.named_steps["m"].feature_importances_, index=names)

    return {
        "model": model,
        "raw_columns": list(raw.columns),
        "defaults": defaults,
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "r2": float(r2_score(y_te, pred)),
        "n_rows": len(df),
        "importances": importances.groupby(importances.index).sum().sort_values(ascending=False),
    }


def predict_grade(bundle: Dict, inputs: Dict) -> float:
    """Predict G3 for one student. Unspecified columns use training defaults."""
    row = {**bundle["defaults"], **{k: v for k, v in inputs.items() if k in bundle["raw_columns"]}}
    frame = add_features(pd.DataFrame([row], columns=bundle["raw_columns"]))
    return float(np.clip(bundle["model"].predict(frame)[0], 0, 20))
