import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error
)

RANDOM_STATE = 42


# -----------------------------------------------------
#  API Key helper
# -----------------------------------------------------
def get_openai_key() -> str:
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY") or ""


# -----------------------------------------------------
#  Task type detection
# -----------------------------------------------------
def detect_task_type(y: pd.Series) -> str:
    """
    Heuristic to decide whether the problem is classification or regression.

    Rules:
    - If y is non-numeric → classification
    - If y is float numeric → regression (almost always continuous)
    - If y is integer numeric:
        * classification only when there are a small number of classes
          that repeat many times
        * otherwise → regression
    """
    # Non-numeric target → classification
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    nunq = y.nunique(dropna=True)
    total = len(y)

    # Float targets are almost always regression
    if pd.api.types.is_float_dtype(y):
        return "regression"

    # Integer targets:
    # treat as classification only if:
    #  - very few distinct values (≤ 10)
    #  - and they are not too sparse (≤ 10% of total samples)
    if nunq <= 10 and nunq <= 0.1 * total:
        return "classification"

    # Otherwise, treat as regression
    return "regression"

# -----------------------------------------------------
#  Safe train-test split with stratify fallback
# -----------------------------------------------------
def safe_train_test_split(X, y, test_size=0.2, stratify=True, random_state=RANDOM_STATE):
    """
    Tries stratified split for classification; falls back to random split if any class has < 2 samples.
    """
    if stratify and y is not None:
        try:
            vc = y.value_counts()
            if vc.min() < 2:
                raise ValueError("A class has fewer than 2 samples — disabling stratify.")
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        except Exception:
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


# -----------------------------------------------------
#  Imbalance handling 
# -----------------------------------------------------
def handle_imbalance(X_train, y_train, task_type: str):
    """
    Automatically balances training data for classification:
    - For small datasets (≤100k rows): apply SMOTE.
    - For larger ones: compute class weights instead.
    For regression, returns inputs unchanged.
    """
    if task_type != "classification":
        return X_train, y_train, None

    n_samples = len(y_train)
    if n_samples <= 100_000:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=RANDOM_STATE)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            return X_res, y_res, None
        except Exception:
            pass

    # fallback: class weights for larger data
    try:
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        return X_train, y_train, dict(zip(classes, weights))
    except Exception:
        return X_train, y_train, None


# -----------------------------------------------------
# Cross-validation metrics
# -----------------------------------------------------
def cross_val_metrics(model, X_train, y_train, task_type: str, folds: int = 5):
    """
    Performs k-fold CV (auto-reduces folds for large datasets).
    Returns average CV metric (f1 for classification, r2 for regression).
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

    n = len(X_train)
    if n > 100_000:
        folds = 3  # CV load

    try:
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
            return {"cv_score": float(np.mean(scores)), "cv_std": float(np.std(scores))}
        else:
            cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
            return {"cv_score": float(np.mean(scores)), "cv_std": float(np.std(scores))}
    except Exception:
        return {"cv_score": None, "cv_std": None}


# -----------------------------------------------------
# Metrics utilities
# -----------------------------------------------------
def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


# -----------------------------------------------------
# Best model selection (robust to missing metrics)
# -----------------------------------------------------
def best_model_by_task(task_type: str, results: pd.DataFrame) -> Tuple[str, dict]:
    """
    Choose the best model row from a leaderboard DataFrame without assuming
    that any specific metric column (like 'rmse') always exists.

    For classification:
        prefer f1 > accuracy > precision > recall (maximize).
    For regression:
        prefer r2 (max), then rmse (min), then mae (min), then mse (min).
    If none of these are present or all-NaN, fall back to the first row.
    """
    if results is None or results.empty:
        return "", {}

    # default index if everything else fails
    best_idx = 0

    if task_type == "classification":
        # larger is better
        for metric in ["f1", "accuracy", "precision", "recall"]:
            if metric in results.columns:
                s = results[metric]
                if s.notna().any():
                    best_idx = s.idxmax()
                    break
    else:
        # regression: r2 larger is better; rmse/mae/mse smaller is better
        metric_candidates = [
            ("r2", True),
            ("rmse", False),
            ("mae", False),
            ("mse", False),
        ]
        for metric, larger_is_better in metric_candidates:
            if metric in results.columns:
                s = results[metric]
                if s.notna().any():
                    best_idx = s.idxmax() if larger_is_better else s.idxmin()
                    break

    row = results.loc[best_idx].to_dict()
    name = str(row.get("model", best_idx))
    return name, row


# -----------------------------------------------------
# Label encoding helpers
# -----------------------------------------------------
def encode_labels_if_needed(y: pd.Series):
    from sklearn.preprocessing import LabelEncoder
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y_enc = pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=y.name)
        return y_enc, le
    return y, None


def inverse_transform_if_possible(arr, encoder):
    if encoder is None:
        return arr
    try:
        import numpy as np
        return encoder.inverse_transform(arr if isinstance(arr, (list, tuple)) else np.array(arr))
    except Exception:
        return arr


# =====================================================
# Metric helpers for defaulted tuning
# =====================================================

_CLASS_METRICS = {"f1", "accuracy", "precision", "recall"}
_REG_METRICS = {"r2", "mae", "rmse"}


def normalize_metric(metric: Optional[str]) -> Optional[str]:
    """
    Normalize user-entered metric text to our canonical set.
    Returns lower-case canonical name or None.
    """
    if metric is None:
        return None
    m = str(metric).strip().lower()
    aliases = {
        "f1_score": "f1",
        "f1weighted": "f1",
        "f1-weighted": "f1",
        "acc": "accuracy",
        "prec": "precision",
        "rec": "recall",
        "r^2": "r2",
        "r-2": "r2",
        "rmse": "rmse",
        "mae": "mae",
        "mse": "mse",  # not exposed to user, but we can map if needed
    }
    m = aliases.get(m, m)
    return m


def default_metric(task_type: str) -> str:
    """
    Our safe defaults when user skips the metric:
      - classification -> f1
      - regression     -> r2
    """
    return "f1" if task_type == "classification" else "r2"


def metric_direction(metric: str) -> str:
    """
    Direction used by Bayesian optimization:
      - maximize for: f1, accuracy, precision, recall, r2
      - minimize for: mae, rmse (and mse if ever used)
    """
    m = normalize_metric(metric) or ""
    if m in {"mae", "rmse", "mse"}:
        return "minimize"
    return "maximize"


def metric_to_sklearn_scorer(task_type: str, metric: Optional[str]) -> str:
    """
    Map our canonical metric names to sklearn scorer strings.
    This ensures RandomizedSearchCV (which always maximizes) gets the right sign.
      - classification: f1 -> 'f1_weighted', accuracy -> 'accuracy',
                        precision -> 'precision_weighted', recall -> 'recall_weighted'
      - regression:     r2 -> 'r2', mae -> 'neg_mean_absolute_error',
                        rmse -> 'neg_root_mean_squared_error' (fallback to 'neg_mean_squared_error')
    If metric is None/invalid, returns the default scorer for the task type.
    """
    m = normalize_metric(metric)

    if task_type == "classification":
        mapping = {
            "f1": "f1_weighted",
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
        }
        return mapping.get(m, "f1_weighted")

    # regression
    
    if m == "r2":
        return "r2"
    if m == "mae":
        return "neg_mean_absolute_error"
    if m == "rmse":
        try:
            # We just return the string; availability handled by sklearn at runtime.
            return "neg_root_mean_squared_error"
        except Exception:
            return "neg_mean_squared_error"
    # fallback default
    return "r2"


def validate_metric_for_task(task_type: str, metric: Optional[str]) -> str:
    """
    Ensure the chosen metric is valid for the task type.
    Falls back to default if not valid.
    """
    m = normalize_metric(metric)
    valid_set = _CLASS_METRICS if task_type == "classification" else _REG_METRICS
    if m in valid_set:
        return m
    return default_metric(task_type)
