# app/agents/nodes.py

from typing import Dict, Any, List, Optional
import pandas as pd

# --- Import from core modules ---
from app.core.preprocessing import (
    process_dataframe,
    coerce_nulls,
    missing_report,
    drop_all_nan_cols,
)

from app.core.training import train_baselines
from app.core.utils import detect_task_type


# =========================================================
# 🔧 PREPROCESS TOOL
# =========================================================
def preprocess_tool(
    df: pd.DataFrame,
    drop_cols: List[str],
    duplicate_strategy: str = "drop",
    missing_strategy: Optional[Dict[str, str]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    type_overrides: Optional[Dict[str, str]] = None,
    preserve_column_names: bool = False,
) -> Dict[str, Any]:
    """
    Main preprocessing wrapper called from the Streamlit app.

    Steps:
        1. Coerce null-like strings ('?', 'None', 'NaN', '')
        2. Drop columns user selected as all-NaN
        3. Run full process_dataframe:
            - Clean / rename columns
            - Handle duplicates
            - Handle missing values
            - Infer and validate data types
    Returns:
        Dict with processed DataFrame, column types, stats, and preview.
    """
    # Step 1: normalize null-like tokens
    df1 = coerce_nulls(df.copy())

    # Step 2: drop user-selected all-NaN columns
    if drop_cols:
        df1 = drop_all_nan_cols(df1, drop_cols)

    # Step 3: full preprocessing
    df2, col_types, type_params, stats = process_dataframe(
        df1,
        column_mapping=column_mapping,
        type_overrides=type_overrides,
        duplicate_strategy=duplicate_strategy,
        missing_strategy=missing_strategy,
        preserve_column_names=preserve_column_names,
    )

    # Step 4: prepare return payload
    return {
        "df": df2,
        "preview": df2.head(15),
        "col_types": col_types,
        "type_params": type_params,
        "stats": stats,
    }


# =========================================================
# ⚙️ BASELINE TRAINING TOOL
# =========================================================
def baseline_training_tool(X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, Any]:
    """
    Wrapper for baseline model training (classification/regression).
    """
    return train_baselines(X, y, task_type)


# =========================================================
# 🧠 TASK TYPE DETECTION (fixed)
# =========================================================
def choose_task_type(y: Any) -> Optional[str]:
    """
    Detect whether the target is classification or regression based on target variable type.

    This version is safe even if:
      - y is accidentally a DataFrame (multiple columns)
      - y is empty or None
    """
    # ✅ Handle None or empty values
    if y is None:
        return None

    # ✅ If DataFrame is passed instead of Series, pick the first column
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 0:
            return None
        y = y.iloc[:, 0]

    # ✅ Handle empty Series
    if getattr(y, "empty", False):
        return None

    # ✅ Call the actual detector
    try:
        return detect_task_type(y)
    except Exception as e:
        # Log-safe return — do not raise to supervisor
        print(f"[choose_task_type] Warning: failed to detect task type → {e}")
        return None
