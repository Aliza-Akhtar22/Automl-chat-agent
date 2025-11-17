# app/core/tuning.py
import warnings
from typing import Dict, Any, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from .utils import (
    RANDOM_STATE,
    classification_metrics,
    regression_metrics,
    encode_labels_if_needed,
    inverse_transform_if_possible,
    handle_imbalance,
    # NEW metric helpers
    default_metric,
    metric_direction,
    metric_to_sklearn_scorer,
    validate_metric_for_task,
)
from .training import build_preprocessor

warnings.filterwarnings("ignore", category=UserWarning)

# Optional Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False


# -----------------------------------------------------
# Model Builder for Tuning
# -----------------------------------------------------
def _estimator_from_name(model_name: str, task_type: str, params: Dict[str, Any] = None):
    params = params or {}
    if task_type == "classification":
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier

        if model_name == "LogisticRegression":
            return LogisticRegression(
                max_iter=300,
                random_state=RANDOM_STATE,
                **{k: v for k, v in params.items() if k in {"C", "penalty", "solver", "max_iter"}},
            )
        if model_name == "RandomForestClassifier":
            return RandomForestClassifier(random_state=RANDOM_STATE, **params)
        if model_name == "KNNClassifier":
            return KNeighborsClassifier(**params)
        if model_name == "XGBoostClassifier":
            from xgboost import XGBClassifier
            return XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss", **params)
        if model_name == "CatBoostClassifier":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(random_state=RANDOM_STATE, verbose=False, **params)
    else:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor

        if model_name == "LinearRegression":
            return LinearRegression(**params)
        if model_name == "RandomForestRegressor":
            return RandomForestRegressor(random_state=RANDOM_STATE, **params)
        if model_name == "KNNRegressor":
            return KNeighborsRegressor(**params)
        if model_name == "XGBoostRegressor":
            from xgboost import XGBRegressor
            return XGBRegressor(random_state=RANDOM_STATE, **params)
        if model_name == "CatBoostRegressor":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(random_state=RANDOM_STATE, verbose=False, **params)
    raise ValueError(f"Unknown model: {model_name}")


def _build_pipeline(X, model_name: str, task_type: str, params: Dict[str, Any] = None):
    if "Logistic" in model_name:
        kind = "logistic"
    elif "Linear" in model_name:
        kind = "linear"
    elif "KNN" in model_name:
        kind = "knn"
    else:
        kind = "tree"
    pre = build_preprocessor(X, model_kind=kind)
    est = _estimator_from_name(model_name, task_type, params or {})
    return Pipeline([("pre", pre), ("model", est)])


# -----------------------------------------------------
# Bayesian Tuning (Optuna)
# -----------------------------------------------------
def tune_with_optuna(
    X_train, y_train, X_test, y_test,
    task_type: str, model_name: str,
    n_trials: int = 40,                           # DEFAULT CHANGED: 40 trials
    timeout: Optional[int] = None,
    direction: Optional[str] = None,              # derive from metric if None
    seed: int = RANDOM_STATE,
    metric: Optional[str] = None,                 # ask user only this; defaults by task
):
    if not HAS_OPTUNA:
        raise RuntimeError("Optuna is not installed. Please install optuna to use Bayesian optimization.")

    # Normalize/validate metric & direction
    metric = validate_metric_for_task(task_type, metric or default_metric(task_type))
    direction = direction or metric_direction(metric)

    # Encode labels for classification
    y_enc, label_encoder = encode_labels_if_needed(y_train) if task_type == "classification" else (y_train, None)

    # Handle imbalance
    X_train_bal, y_train_bal, class_weights = handle_imbalance(X_train, y_enc, task_type)

    # Inner validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_bal, y_train_bal, test_size=0.2, random_state=seed,
        stratify=y_train_bal if task_type == "classification" else None
    )

    # Define hyperparameter search space
    def suggest_params(trial):
        params = {}
        if "RandomForest" in model_name:
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 600)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 30)
        elif "XGBoost" in model_name:
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
            params["n_estimators"] = trial.suggest_int("n_estimators", 200, 1000)
        elif "CatBoost" in model_name:
            params["depth"] = trial.suggest_int("depth", 4, 10)
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True)
            params["iterations"] = trial.suggest_int("iterations", 300, 1200)
        elif "LogisticRegression" in model_name:
            params["C"] = trial.suggest_float("C", 1e-2, 10.0, log=True)
            params["solver"] = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
        elif "KNN" in model_name:
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 41, step=2)
        return params

    # Objective function
    def objective(trial):
        params = suggest_params(trial)
        pipe = _build_pipeline(X_train_bal, model_name, task_type, params=params)
        if class_weights and hasattr(pipe.named_steps["model"], "class_weight"):
            pipe.named_steps["model"].set_params(class_weight=class_weights)

        pipe.fit(X_tr, y_tr)
        y_pred_val = pipe.predict(X_val)

        if task_type == "classification":
            m = classification_metrics(y_val, y_pred_val)
            return float(m.get(metric, 0.0))
        else:
            m = regression_metrics(y_val, y_pred_val)
            return float(m.get(metric, 0.0))

    # Run optimization
    study = optuna.create_study(direction=direction, sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_params
    best_score = float(study.best_value)

    # Refit best model
    best_pipe = _build_pipeline(X_train_bal, model_name, task_type, params=best_params)
    if class_weights and hasattr(best_pipe.named_steps["model"], "class_weight"):
        best_pipe.named_steps["model"].set_params(class_weight=class_weights)
    best_pipe.fit(X_train_bal, y_enc if label_encoder else y_train)

    # Evaluate
    y_pred_test = best_pipe.predict(X_test)
    test_metrics = (
        classification_metrics(y_test, y_pred_test)
        if task_type == "classification"
        else regression_metrics(y_test, y_pred_test)
    )

    y_true_prev = y_test[:20]
    y_pred_prev = y_pred_test[:20]
    if label_encoder is not None:
        y_true_prev = inverse_transform_if_possible(y_true_prev, label_encoder)
        y_pred_prev = inverse_transform_if_possible(y_pred_prev, label_encoder)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "test_metrics": test_metrics,
        "fitted": best_pipe,
        "preview_ap": {"y_true": list(y_true_prev), "y_pred": list(y_pred_prev)},
        "objective": {
            "metric": metric,
            "direction": direction,
            "n_trials": n_trials,
            "timeout": timeout
        },
    }


# -----------------------------------------------------
# Randomized Search
# -----------------------------------------------------
def tune_with_random_search(
    X_train, y_train, X_test, y_test,
    task_type: str, model_name: str,
    n_iter: int = 40,                         # DEFAULT CHANGED: 40 iterations
    cv: int = 3,                              # DEFAULT: 3-fold CV
    random_state: int = RANDOM_STATE,
    scoring: Optional[str] = None,            # if None, derived from metric
    metric: Optional[str] = None,             # user-chosen or defaulted by task
    max_depth_range: Optional[Tuple[int, int]] = None,
    n_estimators_range: Optional[Tuple[int, int]] = None,
):
    """
    RandomizedSearchCV with robust handling:
    - Smart defaults (n_iter=40, cv=3) and metric-driven scoring
    - Class weight / imbalance support
    - Fallback for models without hyperparams
    - Adaptive CV folds for large datasets (handled in utils.cross_val_metrics for baselines)
    """
    from scipy.stats import randint

    # Handle imbalance
    X_train_bal, y_train_bal, class_weights = handle_imbalance(X_train, y_train, task_type)

    # Determine scoring from metric if not provided
    chosen_metric = validate_metric_for_task(task_type, metric or default_metric(task_type))
    if scoring is None:
        scoring = metric_to_sklearn_scorer(task_type, chosen_metric)

    pipe = _build_pipeline(X_train, model_name, task_type, params={})

    # Search space (kept focused & safe)
    param_dist = {}
    if max_depth_range and ("RandomForest" in model_name or "XGBoost" in model_name):
        lo, hi = max_depth_range
        param_dist["model__max_depth"] = randint(int(lo), int(hi) + 1)
    if n_estimators_range and ("RandomForest" in model_name or "XGBoost" in model_name):
        lo, hi = n_estimators_range
        param_dist["model__n_estimators"] = randint(int(lo), int(hi) + 1)

    # If no tunables for this model, just fit and report
    if not param_dist:
        pipe.fit(X_train_bal, y_train_bal)
        y_pred_test = pipe.predict(X_test)
        test_metrics = (
            classification_metrics(y_test, y_pred_test)
            if task_type == "classification"
            else regression_metrics(y_test, y_pred_test)
        )
        return {
            "best_params": {},
            "best_score": float(test_metrics.get(chosen_metric, test_metrics.get("r2", 0.0))),
            "val_metrics": {"cv_score": None},
            "test_metrics": test_metrics,
            "fitted": pipe,
            "objective": {
                "metric": chosen_metric,
                "scoring": scoring,
                "n_iter": 0,
                "cv": None
            },
        }

    rs = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    if class_weights and hasattr(pipe.named_steps["model"], "class_weight"):
        pipe.named_steps["model"].set_params(class_weight=class_weights)

    rs.fit(X_train_bal, y_train_bal)

    best_pipe = rs.best_estimator_
    y_pred_test = best_pipe.predict(X_test)
    test_metrics = (
        classification_metrics(y_test, y_pred_test)
        if task_type == "classification"
        else regression_metrics(y_test, y_pred_test)
    )

    return {
        "best_params": {k.replace("model__", ""): v for k, v in rs.best_params_.items()},
        "best_score": float(rs.best_score_),
        "val_metrics": {"cv_score": float(rs.best_score_)},
        "test_metrics": test_metrics,
        "fitted": best_pipe,
        "objective": {
            "metric": chosen_metric,
            "scoring": scoring,
            "n_iter": n_iter,
            "cv": cv
        },
    }
