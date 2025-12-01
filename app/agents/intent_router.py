# app/agents/intent_router.py
from __future__ import annotations
from typing import Dict, Any, List


class IntentRouter:
    """
    Lightweight intent router.

    It classifies a user message into:
      - kind: "qa" | "simple_action" | "multi_step"
      - actions: list of high-level actions mentioned: ["preprocess", "train", "tune", "preview"]
      - reason: short string explaining why
    """

    def __init__(self) -> None:
        # Could later accept config, thresholds, toggles, etc.
        pass

    # ------------------------ public API ------------------------
    def classify(self, text: str, state: Dict[str, Any]) -> Dict[str, Any]:
        t = (text or "").strip().lower()

        if not t:
            return {
                "kind": "simple_action",
                "actions": [],
                "reason": "Empty/blank message; defaulting to simple_action.",
            }

        looks_like_question = self._looks_like_question(t)
        actions = self._extract_actions(t)

        # --- 1) Explicit status / "have we done X" questions → QA ---
        # e.g. "have we done preprocessing and training?",
        #      "is tuning done?", "have we trained the model?"
        if self._is_status_question(t):
            return {
                "kind": "qa",
                "actions": actions,
                "reason": "Status-style question about whether steps (preprocess/train/tune) have been done.",
            }

        # --- 2) Multi-step / end-to-end detection ---
        if self._looks_like_multi_step(t, actions):
            return {
                "kind": "multi_step",
                "actions": actions,
                "reason": "Text mentions multiple workflow actions or end-to-end behavior.",
            }

        # --- 3) QA / analytical questions ---
        if looks_like_question and self._mentions_analysis_concepts(t):
            return {
                "kind": "qa",
                "actions": actions,
                "reason": "Looks like a question about metrics/status/results, not a navigation command.",
            }

        # --- 4) Simple action (single tool / navigation) ---
        if actions:
            return {
                "kind": "simple_action",
                "actions": actions,
                "reason": "Contains a single clear workflow action (preview/train/tune/preprocess).",
            }

        # --- 5) Fallbacks ---
        if looks_like_question:
            return {
                "kind": "qa",
                "actions": [],
                "reason": "Question-like text; treating as QA by default.",
            }

        return {
            "kind": "simple_action",
            "actions": [],
            "reason": "No clear QA or multi-step pattern; treat as simple navigation or small talk.",
        }

    # ------------------------ helpers ------------------------
    def _looks_like_question(self, t: str) -> bool:
        if "?" in t:
            return True
        q_starts = (
            "what", "which", "how", "why", "when", "where",
            "did", "do ", "does", "have", "has", "is ", "are ",
            "was", "were", "can", "could", "should", "would",
        )
        return any(t.startswith(p) for p in q_starts)

    def _is_status_question(self, t: str) -> bool:
        """
        Detect questions like:
          - "have we done preprocessing and training?"
          - "have we preprocessed the data?"
          - "is tuning done?"
        These should go straight to the QA helper instead of the multi-step planner.
        """
        status_starts = (
            "have we",
            "have i",
            "did we",
            "did i",
            "has the",
            "is ",
            "are ",
            "was ",
            "were ",
        )
        status_keywords = [
            "done",
            "completed",
            "finished",
            "preprocess",
            "preprocessed",
            "preprocessing",
            "cleaned",
            "processed",
            "training",
            "trained",
            "built",
            "model",
            "tuning",
            "tuned",
            "optimized",
            "optimization",
        ]
        return any(t.startswith(s) for s in status_starts) and any(
            k in t for k in status_keywords
        )

    def _mentions_analysis_concepts(self, t: str) -> bool:
        """
        Metric / status / explanation style queries = QA.
        """
        keywords = [
            "accuracy", "f1", "precision", "recall",
            "r2", "rmse", "mae", "mse",
            "score", "metric", "metrics",
            "leaderboard", "best model", "which model",
            "trained", "training done", "did training finish",
            "preprocess", "preprocessing", "cleaning done",
            "finished", "completed",
            "why is", "why my", "why the", "negative", "bad result",
            # NEW: explanation / results-oriented queries
            "result", "results", "explain", "explanation",
        ]
        return any(k in t for k in keywords)

    def _extract_actions(self, t: str) -> List[str]:
        """
        Map natural language phrases to high-level actions:
          - "preview"   → preview
          - "preprocess"→ preprocess
          - "train"     → train
          - "tune"      → tune
        """
        actions: List[str] = []

        # Preview
        preview_words = [
            "preview",
            "show preview",
            "see preview",
            "see the preview",
            "show the preview",
            "show data",
            "see data",
            "show the data",
            "see the data",
            "data preview",
            "show table",
            "see table",
        ]
        if any(w in t for w in preview_words):
            actions.append("preview")

        # Preprocess / clean
        preprocess_words = [
            "preprocess",
            "pre-processing",
            "clean data",
            "clean the data",
            "data cleaning",
            "handle missing",
            "handle duplicates",
            "fix missing",
            "fix nulls",
            "drop duplicates",
        ]
        if any(w in t for w in preprocess_words):
            actions.append("preprocess")

        # Train
        train_words = [
            "train",
            "training",
            "train baselines",
            "start training",
            "run training",
            "train the model",
            "model training",
            "go to training",
            "move to training",
        ]
        if any(w in t for w in train_words):
            actions.append("train")

        # Tune
        tune_words = [
            "tune",
            "tuning",
            "hyperparameter",
            "hyper parameter",
            "optimize model",
            "optimize the model",
            "improve model",
            "improve the model",
            "random search",
            "bayesian optimization",
        ]
        if any(w in t for w in tune_words):
            actions.append("tune")

        # De-duplicate while preserving order
        seen = set()
        unique_actions: List[str] = []
        for a in actions:
            if a not in seen:
                seen.add(a)
                unique_actions.append(a)
        return unique_actions

    def _looks_like_multi_step(self, t: str, actions: List[str]) -> bool:
        """
        Multi-step if:
          - multiple distinct actions in the same sentence, OR
          - explicit end-to-end phrases.
        """
        # Explicit end-to-end wording
        if any(
            phrase in t
            for phrase in [
                "end to end",
                "end-to-end",
                "full pipeline",
                "full automl",
                "run everything",
                "do everything",
                "do the whole thing",
                "all steps",
                "from preprocessing to training",
                "from preprocess to train",
                "clean data, train",
                "clean, train",
                "clean data and train",
                "clean data then train",
                "preprocess and then train",
                "preprocess then train",
            ]
        ):
            return True

        # Multiple workflow actions in same query
        if len(actions) >= 2:
            return True

        return False
