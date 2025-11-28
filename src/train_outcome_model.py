from pathlib import Path
from typing import List, Tuple

import json
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import xgboost as xgb
import joblib

import config
from build_dataset import OUTCOME_LABELS

def load_matchup_dataset() -> pd.DataFrame:
    path = config.MODELING_DIR / "matchups.parquet"
    if not path.exists():
        raise FileNotFoundError("matchups.parquet not found. Run build_dataset.py first.")
    df = pd.read_parquet(path)
    df = df.sort_values("date").reset_index(drop=True)
    return df

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    blacklist = {
        "date", "game_pk", "at_bat_number", "pitch_number",
        "batter", "pitcher",
        "is_hit", "outcome", "outcome_id",
    }
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns
    return [c for c in numeric_cols if c not in blacklist]

def split_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Ensure "date" is datetime64[ns]
    df["date"] = pd.to_datetime(df["date"])

    # Convert config dates (datetime.date) to pandas Timestamp
    train_end = pd.Timestamp(config.TRAIN_END)
    val_start = pd.Timestamp(config.VAL_START)
    val_end = pd.Timestamp(config.VAL_END)
    test_start = pd.Timestamp(config.TEST_START)

    train = df[df["date"] <= train_end]
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)]
    test = df[df["date"] >= test_start]

    print(f"Train: {len(train):,} rows")
    print(f"Val:   {len(val):,} rows")
    print(f"Test:  {len(test):,} rows")

    return train, val, test

def fill_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Logistic Regression requires no NaNs.
    We fill missing numeric values with the column median.
    """
    X_filled = X.copy()
    for col in X_filled.columns:
        if X_filled[col].isna().any():
            X_filled[col] = X_filled[col].fillna(X_filled[col].median())
    return X_filled

def make_xy_multiclass(df: pd.DataFrame, feature_cols: List[str]):
    X = df[feature_cols]
    y = df["outcome_id"].astype(int)
    return X, y

def train_xgb_multiclass(X_train, y_train, X_val, y_val):
    num_classes = len(OUTCOME_LABELS)

    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 500,
        "num_class": num_classes,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_probs = model.predict_proba(X_val)
    val_preds = val_probs.argmax(axis=1)

    acc = accuracy_score(y_val, val_preds)
    macro_f1 = f1_score(y_val, val_preds, average="macro")

    print("\nXGBoost Multiclass Model (val):")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    return model

def evaluate_multiclass_on_test(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds)

    print("\n=== Multiclass Test Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification report:")
    print(classification_report(
        y_test, preds,
        target_names=OUTCOME_LABELS,
        digits=4
    ))

def save_multiclass_model(model, feature_cols: List[str]):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = config.MODELS_DIR / "xgb_outcome_model.joblib"
    joblib.dump(model, model_path)
    print(f"Saved multiclass model to {model_path}")

    feats_path = config.MODELS_DIR / "outcome_feature_cols.json"
    with open(feats_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"Saved feature columns to {feats_path}")

    labels_path = config.MODELS_DIR / "outcome_labels.json"
    with open(labels_path, "w") as f:
        json.dump(OUTCOME_LABELS, f)
    print(f"Saved outcome labels to {labels_path}")

def main():
    print("Loading dataset...")
    df = load_matchup_dataset()

    df = df[df["outcome_id"].notna()].copy()

    print("Selecting features...")
    feature_cols = get_feature_columns(df)

    print("Splitting by date...")
    train, val, test = split_by_date(df)

    X_train, y_train = make_xy_multiclass(train, feature_cols)
    X_val, y_val = make_xy_multiclass(val, feature_cols)
    X_test, y_test = make_xy_multiclass(test, feature_cols)

    X_train = fill_missing_values(X_train)
    X_val = fill_missing_values(X_val)
    X_test = fill_missing_values(X_test)

    model = train_xgb_multiclass(X_train, y_train, X_val, y_val)

    evaluate_multiclass_on_test(model, X_test, y_test)

    save_multiclass_model(model, feature_cols)


if __name__ == "__main__":
    main()
