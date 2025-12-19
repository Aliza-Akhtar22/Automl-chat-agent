#main.py
from dotenv import load_dotenv
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, Response
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

# =========================================================
# Orchestrator
# =========================================================
orch = ChatOrchestrator()

# =========================================================
# GLOBAL STATE (single-user, like Streamlit session_state)
# =========================================================
STATE = {
    # lifecycle
    "stage": "await_upload",
    "thread_id": "api-thread",

    # data
    "raw_df": None,
    "clean_df": None,
    "pre_df": None,

    # previews / stats
    "pre_preview": None,
    "pre_stats": None,
    "pre_col_types": None,
    "pre_type_params": None,
    "show_only_preview": False,

    # chat
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

    # ======================
    # TRAINING
    # ======================
    "target_col": None,
    "task_type": None,
    "train_result": None,
    "best_model_name": None,
    "best_model_row": None,
    "best_model_bytes": None,

    # ======================
    # FORECASTING (Prophet)
    # ======================
    "ds_col": None,
    "y_col": None,
    "forecast_periods": 30,
    "forecast_freq": None,
    "forecast_result": None,
    "forecast_preview": None,

    # ======================
    # TUNING
    # ======================
    "tuned_result": None,
    "chosen_tune_method": None,
    "tune_metric": None,

    # ======================
    # SUPERVISOR FLAGS
    # ======================
    "want_preprocess": False,
    "want_train": False,
    "want_tune": False,
    "want_forecast": False,

    "require_approval": False,
    "approved": False,
    "supervisor_reason": "",
    "recommended_approach": None,

    # bookkeeping
    "history": [],
    "errors": [],
    "suppress_preview_once": False,
    "tuning_stage": None,
    "tuning_offered": False,
    "show_training_panel": False,
}


# =========================================================
# Request / Response models
# =========================================================
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


# =========================================================
# Helpers
# =========================================================
def build_preview(df: pd.DataFrame) -> dict:
    """
    JSON-safe preview:
    - first 15 rows
    - NaN / NaT / inf → null
    """
    head = df.head(10).copy()
    head = head.replace([np.inf, -np.inf], np.nan)
    rows = json.loads(head.to_json(orient="records"))

    return {
        "columns": list(head.columns),
        "rows": rows,
    }


# =========================================================
# 1) CSV UPLOAD
# =========================================================
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    global STATE

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    STATE = orch.start_after_upload(df, STATE)
    return {
        "messages": STATE["messages"]
    }


# =========================================================
# 2) CHAT
# =========================================================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global STATE

    STATE = orch.handle(req.message, STATE)
    last = STATE["messages"][-1]["content"]
    return {"reply": last}


# =========================================================
# 3) RAW PREVIEW
# =========================================================
@app.get("/raw_preview")
async def get_raw_preview():
    global STATE

    if STATE.get("raw_df") is not None:
        df = STATE["raw_df"]
    elif STATE.get("clean_df") is not None:
        df = STATE["clean_df"]
    elif STATE.get("pre_df") is not None:
        df = STATE["pre_df"]
    else:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")

    return build_preview(df)


# =========================================================
# 4) PREPROCESSED PREVIEW
# =========================================================
@app.get("/preview")
async def get_preview():
    global STATE

    if (
        STATE.get("raw_df") is None
        and STATE.get("clean_df") is None
        and STATE.get("pre_df") is None
    ):
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")

    STATE = orch.run_preprocess_now(STATE)

    if STATE.get("pre_df") is not None:
        df = STATE["pre_df"]
    elif STATE.get("clean_df") is not None:
        df = STATE["clean_df"]
    else:
        raise HTTPException(status_code=500, detail="Preview not available.")

    return build_preview(df)


# =========================================================
# 5) TRAINING RESULTS
# =========================================================
@app.get("/train_results")
async def get_train_results():
    global STATE

    tr = STATE.get("train_result")
    if not tr or tr.get("results") is None:
        raise HTTPException(status_code=400, detail="No training results yet.")

    df = tr["results"].replace([np.inf, -np.inf], np.nan)
    rows = json.loads(df.to_json(orient="records"))

    explanation = orch.build_training_explanation(STATE)

    return {
        "leaderboard": rows,
        "explanation": explanation,
    }


# =========================================================
# 6) FORECAST RESULTS (Prophet)
# =========================================================
@app.get("/forecast_results")
async def get_forecast_results():
    global STATE

    if not STATE.get("forecast_result"):
        raise HTTPException(status_code=400, detail="No forecast results yet.")

    return {
        "meta": {
            "ds_col": STATE["forecast_result"].get("ds_col"),
            "y_col": STATE["forecast_result"].get("y_col"),
            "periods": STATE["forecast_result"].get("periods"),
            "freq": STATE["forecast_result"].get("freq"),
        },
        "preview": (
            STATE["forecast_preview"].to_dict(orient="records")
            if hasattr(STATE.get("forecast_preview"), "to_dict")
            else STATE.get("forecast_preview")
        ),
    }

# =========================================================
# 7) DOWNLOAD PREPROCESSED DATA
# =========================================================
@app.get("/download_preprocessed")
async def download_preprocessed():
    global STATE

    if isinstance(STATE.get("pre_df"), pd.DataFrame):
        df = STATE["pre_df"]
    elif isinstance(STATE.get("clean_df"), pd.DataFrame):
        df = STATE["clean_df"]
    else:
        raise HTTPException(status_code=400, detail="No preprocessed data available.")

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=preprocessed_data.csv"},
    )


# =========================================================
# 8) DOWNLOAD BEST MODEL
# =========================================================
@app.get("/download_best_model")
async def download_best_model():
    global STATE

    best_bytes = STATE.get("best_model_bytes")
    best_name = STATE.get("best_model_name") or "best_model"

    if not best_bytes:
        raise HTTPException(status_code=400, detail="No trained model available yet.")

    return Response(
        content=best_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={best_name}.pkl"},
    )
