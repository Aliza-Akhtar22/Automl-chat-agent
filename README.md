
# AutoML Agent (Streamlit + Tool Nodes)

This project implements a conversational AutoML workflow with Human-in-the-Loop checkpoints, aligned with your step-by-step spec. It uses Streamlit for UI and provides LangGraph/LangChain-friendly tool nodes.

## Features

- Upload CSV → Raw preview (head(5)), dtypes
- Minimal cleaning: `"?"`, `"None"`, `"NaN"`, `""` → `NaN`
- Missing summary & **all-NaN columns** list
- User can drop all-NaN columns (checkboxes) → preview again
- **Preprocess Tool (must call)**: apply missing value strategy (mean/median/mode/drop) → preview 15 rows
- Download preprocessed CSV
- Proceed to training? → Target selection → Auto detect **Classification/Regression**
- Baseline training across multiple models in parallel-like loop (pipelines w/ proper preprocessing)
- Results table + **best model recommendation**
- Download best model `.pkl`
- (Stub) Hyperparameter tuning entry (Optuna-ready)

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Optional: create `.env` with `OPENAI_API_KEY=...` (or `OPEN_API_KEY=...`) for future LLM explanations.

## Notes

- Encoding & scaling occur in sklearn Pipelines during training, so the preprocessed CSV remains human-readable (post-imputation, pre-encoding).
- K-Means is excluded when a target is selected (supervised task). A no-target unsupervised path can be added easily.
- CatBoost/XGBoost are optional; the app degrades gracefully if they aren't installed.
