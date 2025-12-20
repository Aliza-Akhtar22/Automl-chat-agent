#training.py
import warnings
import pandas as pd
from typing import Dict, Any

from .utils import (
    RANDOM_STATE,
    detect_task_type,
    safe_train_test_split,
    classification_metrics,
    regression_metrics,
    encode_labels_if_needed,
    inverse_transform_if_possible,
    handle_imbalance,
    cross_val_metrics,
)

warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------------------------------
# Build Preprocessor
# -----------------------------------------------------
def build_preprocessor(X: pd.DataFrame, model_kind: str):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    scale = model_kind in {"linear", "logistic", "knn"}

    transformers = []
    if cat_cols:
        # Auto-handle high-cardinality categorical columns
        high_card_cols = [c for c in cat_cols if X[c].nunique() > 1000]
        safe_cat_cols = [c for c in cat_cols if X[c].nunique() <= 1000]

        if high_card_cols:
            try:
                from sklearn.feature_extraction import FeatureHasher
                transformers.append(("hash", FeatureHasher(n_features=128, input_type="string"), high_card_cols))
            except Exception:
                pass

        if safe_cat_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), safe_cat_cols))

    if num_cols:
        transformers.append(("num", StandardScaler() if scale else "passthrough", num_cols))

    return ColumnTransformer(transformers, remainder="drop")


# -----------------------------------------------------
# Build Model Library
# -----------------------------------------------------
def build_models(task_type: str):
    models = []

    if task_type == "classification":
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier

        models.extend([
            ("LogisticRegression", LogisticRegression(max_iter=300, random_state=RANDOM_STATE), "logistic"),
            ("RandomForestClassifier", RandomForestClassifier(random_state=RANDOM_STATE), "tree"),
            ("KNNClassifier", KNeighborsClassifier(), "knn"),
        ])

       
        try:
            from xgboost import XGBClassifier
            models.append(("XGBoostClassifier", XGBClassifier(random_state=RANDOM_STATE, n_estimators=200, eval_metric="mlogloss"), "tree"))
        except Exception:
            pass
        try:
            from catboost import CatBoostClassifier
            models.append(("CatBoostClassifier", CatBoostClassifier(random_state=RANDOM_STATE, verbose=False), "tree"))
        except Exception:
            pass

    else:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor

        models.extend([
            ("LinearRegression", LinearRegression(), "linear"),
            ("RandomForestRegressor", RandomForestRegressor(random_state=RANDOM_STATE), "tree"),
            ("KNNRegressor", KNeighborsRegressor(), "knn"),
        ])

        try:
            from xgboost import XGBRegressor
            models.append(("XGBoostRegressor", XGBRegressor(random_state=RANDOM_STATE, n_estimators=300), "tree"))
        except Exception:
            pass
        try:
            from catboost import CatBoostRegressor
            models.append(("CatBoostRegressor", CatBoostRegressor(random_state=RANDOM_STATE, verbose=False), "tree"))
        except Exception:
            pass

    return models


# -----------------------------------------------------
# Train Baselines (Enhanced)
# -----------------------------------------------------
def train_baselines(X: pd.DataFrame, y: pd.Series, task_type: str, test_size: float = 0.2) -> Dict[str, Any]:
    """
    Enhanced baseline trainer with:
      - SMOTE/class-weight balancing for classification
      - Safe stratification
      - Cross-validation metrics
      - Graceful error handling
    """

    from sklearn.pipeline import Pipeline

    # Encode labels for classification
    y_encoded, label_encoder = encode_labels_if_needed(y) if task_type == "classification" else (y, None)

    # Safe train/test split
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y_encoded, test_size=test_size, stratify=(task_type == "classification"))

    # Handle imbalance (SMOTE or class weights)
    X_train_bal, y_train_bal, class_weights = handle_imbalance(X_train, y_train, task_type)

    rows = []
    predictions = {}
    fitted = {}

    for name, est, kind in build_models(task_type):
        try:
            pre = build_preprocessor(X, model_kind=kind)
            pipe = Pipeline([("pre", pre), ("model", est)])

            # Apply class weights if supported
            if class_weights and hasattr(pipe.named_steps["model"], "class_weight"):
                pipe.named_steps["model"].set_params(class_weight=class_weights)

            # Perform cross-validation
            cv_result = cross_val_metrics(pipe, X_train_bal, y_train_bal, task_type, folds=5)

            # Fit and test
            pipe.fit(X_train_bal, y_train_bal)
            y_pred = pipe.predict(X_test)

            metrics = (
                classification_metrics(y_test, y_pred)
                if task_type == "classification"
                else regression_metrics(y_test, y_pred)
            )

            metrics.update(cv_result)
            rows.append({"model": name, **metrics})

            # Store predictions (preview)
            y_true_preview = y_test[:20]
            y_pred_preview = y_pred[:20]

            if label_encoder is not None:
                y_true_preview = inverse_transform_if_possible(y_true_preview, label_encoder)
                y_pred_preview = inverse_transform_if_possible(y_pred_preview, label_encoder)

            predictions[name] = {
                "y_true": list(y_true_preview),
                "y_pred": list(y_pred_preview if hasattr(y_pred_preview, "tolist") else y_pred_preview),
            }

            fitted[name] = pipe

        except Exception as e:
            rows.append({"model": name, "error": str(e)})
            continue

    results_df = pd.DataFrame(rows)

    return {
        "results": results_df,
        "predictions": predictions,
        "fitted": fitted,
        "X_train": X_train_bal,
        "y_train": y_train_bal,
        "X_test": X_test,
        "y_test": y_test,
        "label_encoder": label_encoder,
    }
