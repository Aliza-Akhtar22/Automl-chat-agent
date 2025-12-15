# app/agents/intent_normalizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class NormalizedIntent:
    """
    Production-safe intent after applying prerequisites + product rules.

    kind:
      - "qa"      → answer only (no planning, no execution)
      - "plan"    → propose a plan + request approval
      - "confirm" → user is responding yes/no to an existing plan
    """
    kind: str
    actions: List[str]
    reason: str


class IntentNormalizer:
    """
    Converts messy, natural user language into SAFE, SYSTEM-READY intent.

    Core production rules:
    - User intent is advisory
    - System prerequisites are mandatory
    - ANY state-changing action must go through:
        PLAN → USER APPROVAL → EXECUTION
    """

    YES_WORDS = {
        "yes", "y", "yeah", "yep", "ok", "okay", "sure",
        "please do", "do it", "go ahead", "proceed"
    }

    NO_WORDS = {
        "no", "n", "nope", "not now", "later", "stop", "cancel"
    }

    MUTATING_ACTIONS = {"preprocess", "train", "tune"}

    def normalize(
        self,
        raw_intent: Dict[str, Any],
        state: Dict[str, Any],
        user_text: str,
    ) -> NormalizedIntent:
        """
        raw_intent: output from IntentRouter.classify()
          example:
            {
              "kind": "simple_action",
              "actions": ["train"],
              "reason": "user wants training"
            }

        state: chat / AutoML state
        user_text: raw user message (needed for yes/no detection)
        """

        kind = (raw_intent.get("kind") or "simple_action").strip()
        actions = list(raw_intent.get("actions") or [])
        reason = raw_intent.get("reason", "")
        t = (user_text or "").strip().lower()

        stage = state.get("stage")
        approval_pending = (
            state.get("require_approval") is True
            or stage == "plan_proposed"
        )

        # ============================================================
        # 0) PLAN ALREADY PROPOSED → ONLY ACCEPT YES / NO
        # ============================================================
        if approval_pending:
            if t in self.YES_WORDS or t in self.NO_WORDS:
                return NormalizedIntent(
                    kind="confirm",
                    actions=[],
                    reason="User responded to plan approval",
                )

            return NormalizedIntent(
                kind="qa",
                actions=[],
                reason="Plan pending; block new actions until confirmed",
            )

        # ============================================================
        # 1) QA ALWAYS PASSES THROUGH
        # ============================================================
        if kind == "qa":
            return NormalizedIntent(
                kind="qa",
                actions=[],
                reason=reason or "User asked a question",
            )

        # ============================================================
        # 2) MULTI-STEP REQUEST → PLAN
        # ============================================================
        if kind == "multi_step":
            safe_actions = self._normalize_actions(actions)
            safe_actions = self._enforce_prerequisites(safe_actions, state)

            return NormalizedIntent(
                kind="plan",
                actions=safe_actions,
                reason="Multi-step request → plan required",
            )

        # ============================================================
        # 3) SINGLE MUTATING ACTION → STILL PLAN (PRODUCTION RULE)
        # ============================================================
        if any(a in self.MUTATING_ACTIONS for a in actions):
            safe_actions = self._normalize_actions(actions)
            safe_actions = self._enforce_prerequisites(safe_actions, state)

            return NormalizedIntent(
                kind="plan",
                actions=safe_actions,
                reason="State-changing request → plan first",
            )

        # ============================================================
        # 4) PREVIEW / AMBIGUOUS ACTIONS → PLAN (SAFE DEFAULT)
        # ============================================================
        if "preview" in actions:
            return NormalizedIntent(
                kind="plan",
                actions=["preview"],
                reason="Preview may trigger preprocessing → plan first",
            )

        # ============================================================
        # 5) DEFAULT → QA
        # ============================================================
        return NormalizedIntent(
            kind="qa",
            actions=[],
            reason="No deterministic action detected",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_actions(self, actions: List[str]) -> List[str]:
        """Deduplicate actions while preserving order."""
        seen = set()
        out: List[str] = []
        for a in actions:
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out

    def _enforce_prerequisites(
        self,
        actions: List[str],
        state: Dict[str, Any],
    ) -> List[str]:
        """
        Enforce mandatory prerequisites at the PLAN level.
        This NEVER executes anything — it only shapes the plan.
        """

        out = list(actions)

        preprocessing_done = state.get("pre_df") is not None

        # -------------------------
        # Training prerequisites
        # -------------------------
        if "train" in out or "tune" in out:
            if not preprocessing_done and "preprocess" not in out:
                out.insert(0, "preprocess")

            if "confirm_target" not in out:
                train_idx = out.index("train") if "train" in out else len(out)
                out.insert(train_idx, "confirm_target")

        # -------------------------
        # Tuning requires training
        # -------------------------
        if "tune" in out and "train" not in out:
            idx = out.index("tune")
            out.insert(idx, "train")

        return out
