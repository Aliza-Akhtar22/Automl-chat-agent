# main.py
from dotenv import load_dotenv
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, Response  # <-- NEW
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import json

from app.agents.chat_orchestrator import ChatOrchestrator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1) Create one orchestrator instance
orch = ChatOrchestrator()

# 2) Create one global state (same idea as Streamlit session_state)
STATE = {
    "stage": "await_upload",
    "raw_df": None,
    "clean_df": None,
    "pre_df": None,
    "pre_preview": None,
    "pre_stats": None,
    "pre_col_types": None,
    "pre_type_params": None,
    "show_only_preview": False,
    "messages": [
        {"role": "assistant", "content": "Hi 👋 Upload a CSV to get started."}
    ],
    # preprocessing scratch
    "pp_missing_strategy": {},
    "pp_duplicate_strategy": None,
    "pp_type_overrides": {},
    "pp_drop_all_nan_cols": [],
    "pp_column_mapping": {},
    "pp_preserve_column_names": False,
    "done_missing": False,
    "done_duplicates": False,
    "done_dtypes": False,
    "done_drop_all_nan": False,
    "done_rename": False,
    # training
    "target_col": None,
    "task_type": None,
    "train_result": None,
    "best_model_name": None,
    "best_model_row": None,
    "best_model_bytes": None,  # <-- NEW: for download
    # tuning
    "tuned_result": None,
    "chosen_tune_method": None,
    "tune_metric": None,
    # supervisor flags
    "want_preprocess": False,
    "want_train": False,
    "want_tune": False,
    "require_approval": False,
    "approved": False,
    "supervisor_reason": "",
    "history": [],
    "errors": [],
    "suppress_preview_once": False,
    "thread_id": "api-thread",
    "tuning_stage": None,
    "tuning_offered": False,
    "show_training_panel": False,
}


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


# ---------- small helper to build JSON-safe preview ----------
def build_preview(df: pd.DataFrame) -> dict:
    """
    Take a DataFrame and return a JSON-safe preview:
    - first 15 rows
    - NaN / NA / +/-inf converted to JSON null
    """
    # work on a copy of the first 15 rows so we don't mutate STATE
    head = df.head(15).copy()

    # replace infinities with NaN
    head = head.replace([np.inf, -np.inf], np.nan)

    # use pandas' to_json, which serializes NaN/NaT as null, then load back
    rows = json.loads(head.to_json(orient="records"))

    return {
        "columns": list(head.columns),
        "rows": rows,
    }


# ---------- 1) CSV upload endpoint ----------
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Receive a CSV, run ChatOrchestrator.start_after_upload, return first reply.
    """
    global STATE

    # Read file into pandas
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # Let your AutoML orchestrator initialize state with this dataset
    STATE = orch.start_after_upload(df, STATE)

    # Last assistant message (data summary + 'proceed with preprocessing?')
    last = STATE["messages"][-1]["content"]
    return {"reply": last}


# ---------- 2) Chat endpoint using AutoML orchestrator ----------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Take a user message, pass it into ChatOrchestrator.handle, return its reply.
    """
    global STATE

    # Let your "chat brain" decide what to do (preprocess, train, tune, QA...)
    STATE = orch.handle(req.message, STATE)

    # Grab the last assistant message as the reply to the frontend
    last = STATE["messages"][-1]["content"]
    return {"reply": last}


# ---------- 3) RAW preview endpoint (no extra preprocessing) ----------
@app.get("/raw_preview")
async def get_raw_preview():
    """
    Return a small preview of the *raw* data (as uploaded),
    without forcing the preprocessing graph to run.

    Used for the 'Raw data preview' table shown right after upload.
    """
    global STATE

    raw_df = STATE.get("raw_df")
    clean_df = STATE.get("clean_df")
    pre_df = STATE.get("pre_df")

    # Choose first non-None dataframe explicitly
    if raw_df is not None:
        df = raw_df
    elif clean_df is not None:
        df = clean_df
    elif pre_df is not None:
        df = pre_df
    else:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")

    return build_preview(df)


# ---------- 4) PREPROCESSED preview endpoint ----------
@app.get("/preview")
async def get_preview():
    """
    Run preprocessing (if needed) and return a small preview of the data.
    This is used for the 'Preprocessed data preview' table.
    """
    global STATE

    # If no data loaded yet
    if (
        STATE.get("clean_df") is None
        and STATE.get("pre_df") is None
        and STATE.get("raw_df") is None
    ):
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")

    # Ask the orchestrator to run preprocessing once
    STATE = orch.run_preprocess_now(STATE)

    pre_df = STATE.get("pre_df")
    clean_df = STATE.get("clean_df")

    if pre_df is not None:
        df = pre_df
    elif clean_df is not None:
        df = clean_df
    else:
        raise HTTPException(status_code=500, detail="Preview not available.")

    return build_preview(df)


# ---------- 5) TRAINING RESULTS (leaderboard) ----------
@app.get("/train_results")
async def get_train_results():
    """
    Return leaderboard rows (JSON-safe) plus an LLM explanation
    of why the best model is recommended.
    """
    global STATE

    tr = STATE.get("train_result")
    if not tr or tr.get("results") is None:
        raise HTTPException(status_code=400, detail="No training results yet.")

    df = tr["results"].copy()

    # Replace infinities -> NaN so JSON is happy
    df = df.replace([np.inf, -np.inf], np.nan)

    # Convert NaN to null via pandas' JSON serializer
    rows = json.loads(df.to_json(orient="records"))

    # Ask the orchestrator to build a short explanation
    explanation = orch.build_training_explanation(STATE)

    return {
        "leaderboard": rows,
        "explanation": explanation,
    }


# ---------- 6) DOWNLOAD preprocessed data as CSV ----------
@app.get("/download_preprocessed")
async def download_preprocessed():
    """
    Download the preprocessed dataset (or clean_df if pre_df is None) as CSV.
    """
    global STATE

    pre_df = STATE.get("pre_df")
    clean_df = STATE.get("clean_df")

    if isinstance(pre_df, pd.DataFrame):
        df = pre_df
    elif isinstance(clean_df, pd.DataFrame):
        df = clean_df
    else:
        raise HTTPException(status_code=400, detail="No preprocessed data available.")

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=preprocessed_data.csv"
        },
    )


# ---------- 7) DOWNLOAD best model as .pkl ----------
@app.get("/download_best_model")
async def download_best_model():
    """
    Download the best trained model as a pickle file.
    """
    global STATE

    best_bytes = STATE.get("best_model_bytes")
    best_name = STATE.get("best_model_name") or "best_model"

    if not best_bytes:
        raise HTTPException(status_code=400, detail="No trained model available yet.")

    return Response(
        content=best_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={best_name}.pkl"
        },
    )
