# app/agents/intent_router.py
from __future__ import annotations
from typing import Dict, Any, List


class IntentRouter:
    """
    Detects RAW user intent from free-form text.

    Output contract:
    {
        "kind": "qa" | "simple_action" | "multi_step",
        "actions": ["preview" | "preprocess" | "train" | "tune"],
        "reason": str
    }

    IMPORTANT:
    - This class does NOT enforce prerequisites
    - This class does NOT block execution
    - It ONLY detects intent signals
    - Safety + planning is handled by IntentNormalizer + Planner
    """

    # ------------------------ PUBLIC API ------------------------

    def classify(self, text: str, state: Dict[str, Any]) -> Dict[str, Any]:
        t = (text or "").strip().lower()

        if not t:
            return {
                "kind": "simple_action",
                "actions": [],
                "reason": "Empty input",
            }

        looks_like_question = self._looks_like_question(t)
        actions = self._extract_actions(t)

        # ------------------------------------------------------------
        # 1) STATUS / METRIC QUESTIONS → QA
        # ------------------------------------------------------------
        if self._is_status_or_metric_question(t):
            return {
                "kind": "qa",
                "actions": [],
                "reason": "Status / metric question",
            }

        # ------------------------------------------------------------
        # 2) PURE QUESTIONS → QA
        # ------------------------------------------------------------
        if looks_like_question and not actions:
            return {
                "kind": "qa",
                "actions": [],
                "reason": "Pure informational question",
            }

        # ------------------------------------------------------------
        # 3) MULTI-STEP INTENT
        # ------------------------------------------------------------
        if self._looks_like_multi_step(t, actions):
            return {
                "kind": "multi_step",
                "actions": actions,
                "reason": "Multiple workflow actions requested",
            }

        # ------------------------------------------------------------
        # 4) SINGLE ACTION COMMAND
        # ------------------------------------------------------------
        if actions:
            return {
                "kind": "simple_action",
                "actions": actions,
                "reason": "Single workflow action requested",
            }

        # ------------------------------------------------------------
        # 5) FALLBACK → QA-ISH GUIDANCE
        # ------------------------------------------------------------
        return {
            "kind": "qa",
            "actions": [],
            "reason": "Ambiguous input → respond safely",
        }

    # ------------------------ HELPERS ------------------------

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

        # ---------------- preview ----------------
        if any(p in t for p in [
            "preview", "show data", "see data",
            "show table", "see table",
            "look at data",
        ]):
            actions.append("preview")

        # ---------------- preprocess ----------------
        if any(p in t for p in [
            "preprocess", "pre-processing",
            "clean data", "data cleaning",
            "handle missing", "fix missing",
            "drop duplicates", "clean the data",
        ]):
            actions.append("preprocess")

        # ---------------- train ----------------
        if any(p in t for p in [
            "train", "training",
            "train model", "build model",
            "run ml", "start training",
            "build churn model",
            "prediction model",
        ]):
            actions.append("train")

        # ---------------- tune ----------------
        if any(p in t for p in [
            "tune", "tuning",
            "optimize", "optimization",
            "hyperparameter",
            "improve model",
            "random search",
            "bayesian",
        ]):
            actions.append("tune")

        # Deduplicate while preserving order
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
        ]

        if any(p in t for p in explicit_phrases):
            return True

        # More than one action detected
        return len(actions) >= 2
