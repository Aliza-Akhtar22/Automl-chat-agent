# app/core/forecasting.py
from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd


def _infer_freq(ds: pd.Series) -> Optional[str]:
    """
    Best-effort frequency inference.
    Returns a pandas frequency string or None.
    """
    try:
        s = pd.to_datetime(ds, errors="coerce").dropna()
        if len(s) < 3:
            return None
        s = s.sort_values()
        return pd.infer_freq(s)
    except Exception:
        return None


def run_prophet_forecast(
    df: pd.DataFrame,
    ds_col: str,
    y_col: str,
    periods: int = 30,
    freq: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fit a Prophet model and return forecast.

    Requirements:
      - ds_col: datetime-like column
      - y_col: numeric column

    Returns:
      {
        "ds_col": str,
        "y_col": str,
        "periods": int,
        "freq": str,
        "model": fitted Prophet model (not JSON-serializable),
        "forecast_df": pd.DataFrame with columns:
            [ds, yhat, yhat_lower, yhat_upper],
      }
    """
    # -------------------- Validation --------------------
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    if ds_col not in df.columns:
        raise ValueError(f"ds_col not found in dataframe: {ds_col}")
    if y_col not in df.columns:
        raise ValueError(f"y_col not found in dataframe: {y_col}")

    # -------------------- Lazy import --------------------
    try:
        from prophet import Prophet  # type: ignore
    except Exception as e:
        raise ImportError(
            "Prophet is not installed. Install it with: pip install prophet"
        ) from e

    # -------------------- Prepare data --------------------
    work = df[[ds_col, y_col]].copy()

    # ds → datetime
    work[ds_col] = pd.to_datetime(work[ds_col], errors="coerce")
    work = work.dropna(subset=[ds_col])

    # y → numeric
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=[y_col])

    if work.empty:
        raise ValueError("After cleaning ds/y, no rows remain to fit Prophet.")

    # Prophet expects columns named ds, y
    prophet_df = (
        work.rename(columns={ds_col: "ds", y_col: "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )

    # -------------------- Frequency handling (CRITICAL FIX) --------------------
    if freq is None:
        freq = _infer_freq(prophet_df["ds"])

    # HARD FALLBACK: prevents Prophet crash
    if freq is None:
        freq = "MS"  # Month Start (safe default)

    # -------------------- Fit model --------------------
    m = Prophet()
    m.fit(prophet_df)

    # -------------------- Forecast --------------------
    future = m.make_future_dataframe(
        periods=int(periods),
        freq=freq,
    )
    forecast = m.predict(future)

    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    return {
        "ds_col": ds_col,
        "y_col": y_col,
        "periods": int(periods),
        "freq": freq,
        "model": m,
        "forecast_df": out,
    }
