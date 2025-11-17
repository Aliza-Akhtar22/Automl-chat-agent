import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import re
from datetime import datetime
import logging

# --------------------------------
# Logging
# --------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------
# Null coercion helpers (your earlier behavior)
# --------------------------------
_CANON_NULLS = {"?", "none", "nan", ""}

def _coerce_single(x):
    if isinstance(x, str):
        sx = x.strip().lower()
        if sx in _CANON_NULLS:
            return np.nan
    return x

def coerce_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Convert '?', 'None', 'NaN', '' → np.nan across the whole frame."""
    return df.applymap(_coerce_single)

def missing_report(df: pd.DataFrame) -> Dict[str, Any]:
    miss_by_col = df.isna().sum()
    all_nan_cols = miss_by_col[miss_by_col == len(df)].index.tolist()
    return {"missing_by_column": miss_by_col.to_dict(), "all_nan_columns": all_nan_cols}

def drop_all_nan_cols(df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
    return df.drop(columns=cols_to_drop, errors="ignore")

def dtypes_dict(df: pd.DataFrame):
    return {c: str(t) for c, t in df.dtypes.to_dict().items()}


def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).to_pydatetime()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def clean_column_name(col_name: str) -> str:
    col_name = col_name.lower().replace(' ', '_')
    col_name = re.sub(r'[^a-z0-9_]', '', col_name)
    if len(col_name) == 0 or not col_name[0].isalpha():
        col_name = 'col_' + col_name
    return col_name

def handle_duplicates(df: pd.DataFrame, strategy: str = 'drop') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    duplicate_stats = {'total_duplicates': 0, 'duplicate_columns': [], 'duplicate_indices': []}
    if strategy == 'drop':
        original_len = len(df)
        df = df.drop_duplicates()
        duplicate_stats['total_duplicates'] = original_len - len(df)
        logger.info(f"Dropped {duplicate_stats['total_duplicates']} duplicate rows")
    elif strategy == 'keep_first':
        df = df.drop_duplicates(keep='first')
        duplicate_stats['total_duplicates'] = 0  # kept first; we report 0 removed for clarity
    elif strategy == 'keep_last':
        df = df.drop_duplicates(keep='last')
        duplicate_stats['total_duplicates'] = 0
    elif strategy == 'mark':
        df['is_duplicate'] = df.duplicated()
        duplicate_stats['total_duplicates'] = int(df['is_duplicate'].sum())
        logger.info(f"Marked {duplicate_stats['total_duplicates']} duplicate rows")
    return df, duplicate_stats

def handle_missing_values(df: pd.DataFrame, strategy: Dict[str, str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    strategy: dict per column → {'drop'|'mean'|'median'|'mode'|'fill'}
    If a column is not in 'strategy', we do a reasonable default:
      - numeric → fill 0
      - datetime → keep NaT
      - else → fill ''
    """
    missing_stats = {
        'total_missing': int(df.isna().sum().sum()),
        'missing_by_column': df.isna().sum().to_dict(),
        'missing_percentage': (df.isna().sum() / len(df) * 100).to_dict() if len(df) else {}
    }
    strategy = strategy or {}

    for col in df.columns:
        if col in strategy:
            s = (strategy[col] or "").lower()
            if s == 'drop':
                df = df.dropna(subset=[col])
            elif s == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif s == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif s == 'mode':
                if df[col].mode().shape[0] > 0:
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna('')
            elif s == 'fill':
                df[col] = df[col].fillna('')
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].fillna(pd.NaT)
            else:
                df[col] = df[col].fillna('')
    logger.info(f"Handled {missing_stats['total_missing']} missing values")
    return df, missing_stats

def validate_data_types(df: pd.DataFrame, expected_types: Dict[str, str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    validation_stats = {'type_errors': {}, 'converted_columns': []}
    expected_types = expected_types or {}
    for col in df.columns:
        if col in expected_types:
            try:
                exp = expected_types[col]
                if exp == 'int':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif exp == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif exp == 'boolean':
                    df[col] = df[col].astype(str).str.lower().map(
                        {'true': True, 'false': False, 'yes': True, 'no': False, '1': True, '0': False})
                elif exp == 'timestamp':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif exp == 'string':
                    df[col] = df[col].astype(str)
                validation_stats['converted_columns'].append(col)
            except Exception as e:
                validation_stats['type_errors'][col] = str(e)
                logger.warning(f"Error converting {col} to {expected_types[col]}: {str(e)}")
    return df, validation_stats

def infer_column_type(series: pd.Series) -> Tuple[str, Dict[str, Any]]:
    if pd.api.types.is_datetime64_any_dtype(series): return "timestamp", {}
    if pd.api.types.is_integer_dtype(series): return "int", {}
    if pd.api.types.is_float_dtype(series): return "float", {}
    if pd.api.types.is_bool_dtype(series): return "boolean", {}
    # categorical if <10% unique (safe heuristic)
    if len(series) > 0 and (series.nunique(dropna=True) / len(series) < 0.1):
        return "categorical", {"categories": pd.Series(series).dropna().unique().tolist()}
    return "string", {}

def clean_data_value(value: Any, col_type: str) -> Any:
    if pd.isna(value):
        return None
    try:
        if col_type == "int":
            return int(float(value))
        elif col_type == "float":
            return float(value)
        elif col_type == "boolean":
            if isinstance(value, str):
                return value.strip().lower() in ('true', 'yes', '1', 't', 'y')
            return bool(value)
        elif col_type == "timestamp":
            if isinstance(value, str):
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y%m%d']:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue
            return pd.to_datetime(value, errors='coerce')
        elif col_type == "string":
            return str(value).strip()
        return value
    except (ValueError, TypeError):
        return None

def process_dataframe(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None,
    type_overrides: Optional[Dict[str, str]] = None,
    duplicate_strategy: str = 'drop',
    missing_strategy: Optional[Dict[str, str]] = None,
    preserve_column_names: bool = False
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Any], Dict[str, Any]]:
    """
    Full preprocessing:
      1) Coerce '?','None','NaN','' → NaN
      2) (Optional) Clean column names unless preserve_column_names=True
      3) Rename by column_mapping (if provided)
      4) Handle duplicates (drop/keep_first/keep_last/mark)
      5) Handle missing values (per-column strategies allowed)
      6) Infer types + clean cell values
      7) (Optional) Enforce expected types via validate_data_types
    Returns: df, inferred_col_types, type_params, processing_stats
    """
    processing_stats = {
        'duplicates': {},
        'missing_values': {},
        'type_validation': {},
        'original_shape': df.shape,
        'final_shape': None
    }

    # Step 1: normalize known null tokens
    df = coerce_nulls(df.copy())

    # Step 2: clean columns
    if not preserve_column_names:
        df.columns = [clean_column_name(col) for col in df.columns]

    # Step 3: rename columns if requested
    if column_mapping:
        df = df.rename(columns=column_mapping)

    # Step 4: duplicates
    df, duplicate_stats = handle_duplicates(df, duplicate_strategy)
    processing_stats['duplicates'] = duplicate_stats

    # Step 5: missing values
    df, missing_stats = handle_missing_values(df, missing_strategy)
    processing_stats['missing_values'] = missing_stats

    # Step 6: infer types & clean values
    col_types: Dict[str, str] = {}
    type_params: Dict[str, Any] = {}

    for col in df.columns:
        if type_overrides and col in type_overrides:
            col_type = type_overrides[col]
            type_params[col] = {}
        else:
            col_type, params = infer_column_type(df[col])
            type_params[col] = params
        col_types[col] = col_type
        df[col] = df[col].apply(lambda x: clean_data_value(x, col_type))

    # Step 7: validate/enforce types
    df, validation_stats = validate_data_types(df, col_types)
    processing_stats['type_validation'] = validation_stats

    processing_stats['final_shape'] = df.shape
    processing_stats = convert_numpy_types(processing_stats)
    return df, col_types, type_params, processing_stats

def generate_column_mapping(df: pd.DataFrame, target_columns: Optional[List[str]] = None) -> Dict[str, str]:
    if not target_columns:
        return {}
    mapping: Dict[str, str] = {}
    current_cols = df.columns.tolist()
    for target in target_columns:
        best_match, best_score = None, 0.0
        for col in current_cols:
            clean_target = clean_column_name(target)
            clean_col = clean_column_name(col)
            if clean_target == clean_col:
                score = 1.0
            else:
                common = set(clean_target) & set(clean_col)
                score = len(common) / max(len(clean_target), len(clean_col))
            if score > best_score:
                best_score, best_match = score, col
        if best_score > 0.7 and best_match is not None:
            mapping[best_match] = target
    return mapping
