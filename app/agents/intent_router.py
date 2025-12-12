# app/agents/intent_router.py
from __future__ import annotations
from typing import Dict, Any, List


class IntentRouter:
    """
    Classifies user messages into:
      - "qa"            (questions, metric queries, status checks)
      - "simple_action" (single-step workflow action)
      - "multi_step"    (multi-step workflows: preprocess→train, clean→train→tune, etc.)

    Option B (Safe Multi-Step):
      * Status questions remain QA.
      * Requests containing multiple actions become multi_step.
      * Analytical / metric questions → QA.
      * Single action commands → simple_action.
    """

    def __init__(self) -> None:
        pass

    # ------------------------ PUBLIC API ------------------------
    def classify(self, text: str, state: Dict[str, Any]) -> Dict[str, Any]:
        t = (text or "").strip().lower()

        if not t:
            return {
                "kind": "simple_action",
                "actions": [],
                "reason": "Blank message → simple_action",
            }

        looks_like_question = self._looks_like_question(t)
        actions = self._extract_actions(t)

        # ------------------------------------------------------------
        # 1) STATUS QUESTIONS (MUST ALWAYS BE QA)
        # ------------------------------------------------------------
        # Ex: "have we trained?", "is preprocessing done?", "did we tune?"
        if self._is_status_question(t):
            return {
                "kind": "qa",
                "actions": actions,
                "reason": "Status question (have we / did we / is X done?) → QA",
            }

        # ------------------------------------------------------------
        # 2) METRIC / ANALYSIS QUESTIONS → ALWAYS QA
        # ------------------------------------------------------------
        if looks_like_question and self._mentions_analysis_concepts(t):
            return {
                "kind": "qa",
                "actions": actions,
                "reason": "Analytical/metric question → QA",
            }

        # ------------------------------------------------------------
        # 3) MULTI-STEP COMMANDS (REQUEST TO DO MULTIPLE ACTIONS)
        # ------------------------------------------------------------
        # Ex: "preprocess and train", "clean data then train", "clean, train & tune"
        is_multi = self._looks_like_multi_step(t, actions)

        if is_multi and not looks_like_question:
            # Only classify as multi-step when it's a request, not a question.
            return {
                "kind": "multi_step",
                "actions": actions,
                "reason": "Multiple workflow actions detected → multi_step",
            }

        # ------------------------------------------------------------
        # 4) SIMPLE ACTION: SINGLE EXPLICIT ACTION
        # ------------------------------------------------------------
        if actions:
            return {
                "kind": "simple_action",
                "actions": actions,
                "reason": "Single workflow action found → simple_action",
            }

        # ------------------------------------------------------------
        # 5) NON-METRIC QUESTIONS → QA
        # ------------------------------------------------------------
        if looks_like_question:
            return {
                "kind": "qa",
                "actions": [],
                "reason": "Generic question → QA",
            }

        # ------------------------------------------------------------
        # 6) DEFAULT → SIMPLE ACTION
        # ------------------------------------------------------------
        return {
            "kind": "simple_action",
            "actions": [],
            "reason": "No strong signals → simple_action",
        }

    # ------------------------ HELPERS ------------------------

    def _looks_like_question(self, t: str) -> bool:
        if "?" in t:
            return True
        q_starts = (
            "what", "which", "how", "why", "when", "where",
            "did", "do ", "does", "have", "has",
            "is ", "are ", "was", "were",
            "can", "could", "should", "would",
        )
        return any(t.startswith(p) for p in q_starts)

    def _is_status_question(self, t: str) -> bool:
        """
        Detect questions like:
           - "have we done preprocessing?"
           - "is training finished?"
           - "did we tune the model?"
        These MUST be QA (Option B rule)
        """
        status_starts = (
            "have we", "have i", "did we", "did i",
            "has the", "is ", "are ", "was ", "were ",
        )

        status_keywords = [
            "done", "completed", "finished",
            "preprocess", "preprocessed", "preprocessing",
            "cleaned", "processed",
            "training", "trained",
            "model", "tuning", "tuned",
            "optimization", "optimized",
        ]

        return any(t.startswith(s) for s in status_starts) and any(
            k in t for k in status_keywords
        )

    def _mentions_analysis_concepts(self, t: str) -> bool:
        """
        Detect metric/explanation/status analytical queries → QA.
        """
        keywords = [
            "accuracy", "f1", "precision", "recall",
            "r2", "rmse", "mae", "mse",
            "score", "metric", "metrics",
            "leaderboard", "best model", "which model",
            "trained", "training done", "did training finish",
            "cleaning done", "preprocessed",
            "finished", "completed",
            "why is", "why my", "why the",
            "negative", "bad result",
            "result", "results", "explain", "explanation",
        ]
        return any(k in t for k in keywords)

    def _extract_actions(self, t: str) -> List[str]:
        """
        Detect high-level workflow intents.
        """
        actions: List[str] = []

        preview_words = [
            "preview", "show preview", "see preview",
            "show data", "see data", "data preview",
            "show table", "see table",
        ]
        if any(w in t for w in preview_words):
            actions.append("preview")

        preprocess_words = [
            "preprocess", "pre-processing", "clean data",
            "clean the data", "data cleaning",
            "handle missing", "fix missing",
            "drop duplicates", "clean up",
        ]
        if any(w in t for w in preprocess_words):
            actions.append("preprocess")

        train_words = [
            "train", "training", "train baselines",
            "start training", "run training",
            "train the model", "model training",
            "go to training", "move to training",
        ]
        if any(w in t for w in train_words):
            actions.append("train")

        tune_words = [
            "tune", "tuning", "hyperparameter",
            "optimize model", "improve model",
            "random search", "bayesian optimization",
        ]
        if any(w in t for w in tune_words):
            actions.append("tune")

        # Deduplicate while preserving order
        unique = []
        seen = set()
        for a in actions:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique

    def _looks_like_multi_step(self, t: str, actions: List[str]) -> bool:
        """
        Multi-step if:
           - The user *requests* multiple steps
           - OR uses explicit end-to-end wording
        """
        # Explicit multi-step phrases
        explicit_phrases = [
            "end to end", "end-to-end", "full pipeline",
            "full automl", "run everything",
            "do everything", "all steps",
            "from preprocessing to training",
            "from preprocess to train",
            "clean data and train",
            "clean data then train",
            "preprocess and then train",
            "preprocess then train",
            "clean, train", "clean, train, tune",
        ]
        if any(p in t for p in explicit_phrases):
            return True

        # Multiple workflow actions = multi-step
        return len(actions) >= 2
