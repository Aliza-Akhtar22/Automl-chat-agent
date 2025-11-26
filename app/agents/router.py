# app/agents/router.py
from __future__ import annotations

from typing import Literal, TypedDict, Optional


# -----------------------------------------
# Types
# -----------------------------------------
Intent = Literal["simple", "pipeline", "unknown"]


class RouterResult(TypedDict, total=False):
    """
    Output structure returned by `classify_intent`.

    Fields:
        intent:        "simple" | "pipeline" | "unknown"
        action:        For simple intents only ("preview", "train", "tune")
        pipeline_goal: For pipeline intents only:
                           "preprocess_only",
                           "preprocess_train",
                           "preprocess_train_tune"
        raw_text:      Original user message
    """
    intent: Intent
    action: str
    pipeline_goal: str
    raw_text: str


# -----------------------------------------
# Helpers
# -----------------------------------------
def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _looks_like_pipeline(text: str) -> bool:
    """
    Detect multi-step / end-to-end AutoML requests.

    Covers patterns like:
        "do everything end to end"
        "full automl pipeline"
        "preprocess, train, tune"
        "clean data then train"
    """
    t = _normalize(text)

    strong_phrases = [
        "do everything end to end",
        "do everything end-to-end",
        "do everything for me",
        "run full automl pipeline",
        "run the full automl pipeline",
        "run full pipeline",
        "run the full pipeline",
        "full pipeline",
        "full automl pipeline",
        "end to end pipeline",
        "end-to-end pipeline",
        "just do everything",
        "just do it for me",
        "handle everything for me",
    ]

    if any(p in t for p in strong_phrases):
        return True

    # Stage-based detection
    has_preprocess = any(w in t for w in [
        "preprocess", "pre-processing", "clean data", "clean my data"
    ])
    has_train = "train" in t or "training" in t
    has_tune = any(w in t for w in [
        "tune", "tuning", "hyperparameter", "hyper parameter", "optimize"
    ])

    # Pipeline if multiple major stages appear together
    if has_preprocess and has_train:
        return True
    if has_train and has_tune:
        return True
    if has_preprocess and has_tune:
        return True

    return False


def _simple_action(text: str) -> Optional[str]:
    """
    Simple one-step commands.
    These bypass planner entirely.
    """
    t = _normalize(text)

    # Preview
    preview_phrases = [
        "preview", "show preview", "see preview",
        "show the data", "show data", "see the data",
        "show table", "see table", "data preview",
    ]
    if any(w in t for w in preview_phrases):
        return "preview"

    # Train
    train_phrases = [
        "go to training", "start training", "run training",
        "train baselines", "train the model", "proceed to training",
        "move to training", "move forward with training",
        "training part",
    ]
    if any(p in t for p in train_phrases):
        return "train"
    if t in {"train", "training"}:
        return "train"

    # Tune
    tune_words = [
        "tune", "tuning", "hyperparameter",
        "hyper parameter", "optimize", "improve model",
    ]
    if any(w in t for w in tune_words):
        return "tune"

    return None


# -----------------------------------------
# Public API
# -----------------------------------------
def classify_intent(user_text: str) -> RouterResult:
    """
    Lightweight intent recognizer.

    Identifies:
      - Pipeline requests: "do preprocess and train for me"
      - Simple commands: "preview", "train", "tune"
      - Unknown: fall back to orchestrator logic
    """
    raw = user_text or ""
    t = _normalize(raw)

    # 1) Detect multi-step / pipeline queries
    if _looks_like_pipeline(t):
        has_preprocess = any(w in t for w in [
            "preprocess", "pre-processing", "clean data"
        ])
        has_train = "train" in t or "training" in t
        has_tune = any(w in t for w in [
            "tune", "tuning", "hyperparameter", "optimize"
        ])

        # Determine the pipeline target
        if has_preprocess and has_train and has_tune:
            goal = "preprocess_train_tune"
        elif has_preprocess and has_train:
            goal = "preprocess_train"
        elif has_preprocess:
            goal = "preprocess_only"
        else:
            # Generic catch-all like "do everything"
            goal = "preprocess_train_tune"

        return RouterResult(
            intent="pipeline",
            pipeline_goal=goal,
            raw_text=raw,
        )

    # 2) Simple one-step actions
    action = _simple_action(t)
    if action:
        return RouterResult(
            intent="simple",
            action=action,
            raw_text=raw,
        )

    # 3) Default fallback
    return RouterResult(
        intent="unknown",
        raw_text=raw,
    )
