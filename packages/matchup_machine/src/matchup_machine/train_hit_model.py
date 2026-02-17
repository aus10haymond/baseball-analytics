from pathlib import Path
from typing import Tuple, List

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, precision_recall_curve
)

import xgboost as xgb
import joblib

import config

def load_matchup_dataset() -> pd.DataFrame:
    path = config.MODELING_DIR / "matchups.parquet"
    if not path.exists():
        raise FileNotFoundError("matchups.parquet not found. Run build_dataset.py first.")
    df = pd.read_parquet(path)
    df = df.sort_values("date").reset_index(drop=True)
    return df

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Select numeric feature columns only, excluding ID/date/target columns.
    """
    blacklist = {"date", "game_pk", "at_bat_number", "pitch_number", "batter", "pitcher", "is_hit"}

    # Keep only numeric columns for modeling
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

def make_xy(df: pd.DataFrame, feature_cols: List[str]):
    X = df[feature_cols]
    y = df["is_hit"].astype(int)
    return X, y

def train_logistic_baseline(X_train, y_train, X_val, y_val):
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, preds >= 0.5)

    print(f"\nBaseline Logistic Regression:")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")

    return model

def train_xgb_model(X_train, y_train, X_val, y_val):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 500,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, preds >= 0.5)

    print(f"\nXGBoost Model:")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")

    return model

def find_best_threshold(y_val, probs):
    """
    Use the validation set to choose a probability threshold.
    Here we maximize F1-score over possible thresholds.
    Returns (best_threshold, precision, recall, f1).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    # precision_recall_curve returns N+1 points, thresholds has length N.
    # Skip the first point (threshold undefined) when pairing with thresholds.
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)

    f1_for_thresholds = f1_scores[1:]          # align with thresholds
    idx = f1_for_thresholds.argmax()
    best_threshold = thresholds[idx]
    best_precision = precisions[idx + 1]
    best_recall = recalls[idx + 1]
    best_f1 = f1_scores[idx + 1]

    print("\n=== Threshold tuning on validation set ===")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Val Precision:  {best_precision:.4f}")
    print(f"Val Recall:     {best_recall:.4f}")
    print(f"Val F1:         {best_f1:.4f}")

    return best_threshold, best_precision, best_recall, best_f1

def evaluate_on_test(model, X_test, y_test, threshold: float = 0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"\n=== Test Set Evaluation (threshold = {threshold:.3f}) ===")
    print(f"AUC:        {auc:.4f}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

def save_feature_importance(model, feature_cols: List[str]):
    if hasattr(model, "feature_importances_"):
        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        out = config.RESULTS_DIR / "feature_importance.csv"
        importances.to_csv(out, index=False)
        print(f"Saved feature importance to {out}")

def save_model(model, name: str):
    path = config.MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Saved model to {path}")

def main():
    print("Loading dataset...")
    df = load_matchup_dataset()

    print("Selecting features...")
    feature_cols = get_feature_columns(df)

    print("Splitting by date...")
    train, val, test = split_by_date(df)

    X_train, y_train = make_xy(train, feature_cols)
    X_val, y_val = make_xy(val, feature_cols)
    X_test, y_test = make_xy(test, feature_cols)

    # Fill missing values for models that cannot handle NaNs
    X_train = fill_missing_values(X_train)
    X_val = fill_missing_values(X_val)
    X_test = fill_missing_values(X_test)

    # Baseline model
    baseline = train_logistic_baseline(X_train, y_train, X_val, y_val)
    save_model(baseline, "baseline_logistic")

    # XGBoost model
    xgb_model = train_xgb_model(X_train, y_train, X_val, y_val)
    save_model(xgb_model, "xgboost_hit_model")

    # Tune threshold on validation set
    val_probs = xgb_model.predict_proba(X_val)[:, 1]
    best_threshold, _, _, _ = find_best_threshold(y_val, val_probs)

    # Evaluate on test using tuned threshold
    evaluate_on_test(xgb_model, X_test, y_test, threshold=best_threshold)


    # Importance
    save_feature_importance(xgb_model, feature_cols)

    print("\nDone.")


if __name__ == "__main__":
    main()
