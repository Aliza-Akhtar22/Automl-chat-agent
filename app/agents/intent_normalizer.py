# app/agents/intent_normalizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class NormalizedIntent:
    kind: str
    actions: List[str]
    reason: str


class IntentNormalizer:

    AGREE_WITH_RECOMMENDATION = {
    "ok proceed",
    "okay proceed",
    "proceed",
    "go ahead",
    "do it",
    "use this",
    "use this approach",
    "use your suggestion",
    "use your recommendation",
    "sounds good",
    "let’s do it",
    "lets do it",
    "yes proceed",
    }
    YES_WORDS = {"yes", "y", "yeah", "yep", "ok", "okay", "sure", "please do", "do it", "go ahead", "proceed"}
    NO_WORDS = {"no", "n", "nope", "not now", "later", "stop", "cancel"}

    MUTATING_ACTIONS = {"preprocess", "train", "tune", "forecast"}

    def normalize(self, raw_intent: Dict[str, Any], state: Dict[str, Any], user_text: str) -> NormalizedIntent:
        kind = (raw_intent.get("kind") or "simple_action").strip()
        actions = list(raw_intent.get("actions") or [])
        reason = raw_intent.get("reason", "")
        requested_model = raw_intent.get("requested_model")
        t = (user_text or "").strip().lower()

        stage = state.get("stage")
        approval_pending = (state.get("require_approval") is True or stage == "plan_proposed")

        if approval_pending:
            if t in self.YES_WORDS or t in self.NO_WORDS:
                return NormalizedIntent(kind="confirm", actions=[], reason="User responded to plan approval")
            return NormalizedIntent(kind="qa", actions=[], reason="Plan pending; block new actions until confirmed")

        if kind == "qa":
            return NormalizedIntent(kind="qa", actions=[], reason=reason or "User asked a question")

        if kind == "multi_step":
            safe_actions = self._normalize_actions(actions)
            safe_actions = self._enforce_prerequisites(safe_actions, state)
            if requested_model:
                state["requested_model"] = requested_model
            return NormalizedIntent(kind="plan", actions=safe_actions, reason="Multi-step request → plan required")

        if (
            any(a in self.MUTATING_ACTIONS for a in actions)
            or "confirm_target" in actions
        ):
            safe_actions = self._normalize_actions(actions)
            safe_actions = self._enforce_prerequisites(safe_actions, state)
            if requested_model:
                state["requested_model"] = requested_model
            return NormalizedIntent(kind="plan", actions=safe_actions, reason="State-changing request → plan first")

        if "preview" in actions:
            return NormalizedIntent(kind="plan", actions=["preview"], reason="Preview may trigger preprocessing → plan first")

        return NormalizedIntent(kind=kind, actions=actions, reason=reason or "No deterministic action detected")

    def _normalize_actions(self, actions: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for a in actions:
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out

    def _enforce_prerequisites(self, actions: List[str], state: Dict[str, Any]) -> List[str]:
        out = list(actions)
        preprocessing_done = state.get("pre_df") is not None

        # Forecast prerequisites
        if "forecast" in out:
            if not preprocessing_done and "preprocess" not in out:
                out.insert(0, "preprocess")
            # ensure ds/y confirmation happens before forecast
            if "confirm_ds" not in out:
                idx = out.index("forecast")
                out.insert(idx, "confirm_ds")
            if "confirm_y" not in out:
                idx = out.index("forecast")
                out.insert(idx, "confirm_y")

        # Training prerequisites
        if "train" in out or "tune" in out:
            if not preprocessing_done and "preprocess" not in out:
                out.insert(0, "preprocess")
            if "confirm_target" not in out:
                idx = out.index("train") if "train" in out else len(out)
                out.insert(idx, "confirm_target")

        if "tune" in out and "train" not in out:
            idx = out.index("tune")
            out.insert(idx, "train")

        return out
