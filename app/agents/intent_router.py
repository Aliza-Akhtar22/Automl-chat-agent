# app/agents/intent_router.py
from __future__ import annotations
from typing import Dict, Any, List


class IntentRouter:
    """
    Detects RAW user intent from free-form text.

    Output contract:
    {
        "kind": "qa" | "simple_action" | "multi_step",
        "actions": ["preview" | "preprocess" | "train" | "tune" | "forecast"],
        "reason": str
    }
    """

    def classify(self, text: str, state: Dict[str, Any]) -> Dict[str, Any]:
        t = (text or "").strip().lower()

        if not t:
            return {"kind": "simple_action", "actions": [], "reason": "Empty input"}

        looks_like_question = self._looks_like_question(t)
        actions = self._extract_actions(t)

        if self._is_status_or_metric_question(t):
            return {"kind": "qa", "actions": [], "reason": "Status / metric question"}

        if looks_like_question and not actions:
            return {"kind": "qa", "actions": [], "reason": "Pure informational question"}

        if self._looks_like_multi_step(t, actions):
            return {"kind": "multi_step", "actions": actions, "reason": "Multiple workflow actions requested"}

        if actions:
            return {"kind": "simple_action", "actions": actions, "reason": "Single workflow action requested"}

        return {"kind": "qa", "actions": [], "reason": "Ambiguous input → respond safely"}

    def _looks_like_question(self, t: str) -> bool:
        if "?" in t:
            return True
        starters = (
            "what", "which", "how", "why", "when", "where",
            "did", "do ", "does", "have", "has",
            "is ", "are ", "was", "were",
            "can", "could", "should", "would",
        )
        return any(t.startswith(s) for s in starters)

    def _is_status_or_metric_question(self, t: str) -> bool:
        status_keywords = [
            "done", "completed", "finished",
            "preprocess", "preprocessed", "cleaned",
            "trained", "training",
            "tuned", "tuning",
            "forecast", "forecasted", "prophet",
            "result", "results",
        ]
        metric_keywords = [
            "accuracy", "f1", "precision", "recall",
            "r2", "rmse", "mae",
            "score", "metric", "leaderboard",
            "best model", "which model",
            "explain", "explanation",
        ]
        return any(k in t for k in status_keywords + metric_keywords) and self._looks_like_question(t)

    def _extract_actions(self, t: str) -> List[str]:
        actions: List[str] = []

        # preview
        if any(p in t for p in ["preview", "show data", "see data", "show table", "see table", "look at data"]):
            actions.append("preview")

        # preprocess
        if any(p in t for p in [
            "preprocess", "pre-processing",
            "clean data", "data cleaning",
            "handle missing", "fix missing",
            "drop duplicates", "clean the data",
        ]):
            actions.append("preprocess")

        # forecast horizon-only messages (CRITICAL FIX)
        if any(k in t for k in ["daily", "weekly", "monthly", "yearly"]) and any(
            x in t for x in ["day", "days", "week", "weeks", "month", "months", "year", "years"]
        ):
            actions.append("forecast")


        # forecast (Prophet / time series)
        if any(p in t for p in [
            "forecast", "forecasting",
            "time series", "timeseries",
            "predict future", "predict next",
            "prophet",
            "future sales", "next 30 days", "next 7 days",
        ]):
            actions.append("forecast")

        # train
        if any(p in t for p in [
            "train", "training",
            "train model", "build model",
            "run ml", "start training",
            "build churn model",
            "prediction model",
        ]):
            actions.append("train")

        # tune
        if any(p in t for p in [
            "tune", "tuning",
            "optimize", "optimization",
            "hyperparameter",
            "improve model",
            "random search",
            "bayesian",
        ]):
            actions.append("tune")

        # dedupe
        seen = set()
        uniq: List[str] = []
        for a in actions:
            if a not in seen:
                seen.add(a)
                uniq.append(a)
        return uniq

    def _looks_like_multi_step(self, t: str, actions: List[str]) -> bool:
        explicit_phrases = [
            "end to end", "end-to-end",
            "full pipeline", "full automl",
            "run everything", "do everything",
            "clean and train",
            "preprocess and train",
            "clean train tune",
            "complete workflow",
            "clean and forecast",
            "preprocess and forecast",
            "forecast and tune",
        ]
        if any(p in t for p in explicit_phrases):
            return True
        return len(actions) >= 2
