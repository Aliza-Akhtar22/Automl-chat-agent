# app/agents/policies.py
from typing import Dict, Any

class SupervisorPolicy:
    """
    Encapsulates routing decisions & HITL checkpoints.
    Change thresholds here without touching the graph.
    """
    def route(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide next action and whether to require human approval.
        Returns dict with keys:
          - next: one of {"preprocess","choose_task","train","tune","end"}
          - require_approval: bool
          - reason: str (shown to user)
        """
        s = state
        # If we haven't preprocessed, do that first.
        if s.get("pre_df") is None and s.get("clean_df") is not None and s.get("want_preprocess", True):
            return {"next": "preprocess", "require_approval": False, "reason": "Preprocessing not done yet."}

        # If we have pre_df but no task_type/target, choose task
        if s.get("pre_df") is not None and s.get("target_col") and not s.get("task_type"):
            return {"next": "choose_task", "require_approval": False, "reason": "Task type not set yet."}

        # If we want to train and haven't, go train (HITL checkpoint)
        if s.get("want_train") and not s.get("train_result"):
            return {"next": "train", "require_approval": True, "reason": "Approve moving to training?"}

        # If we want to tune and we trained, go tune (HITL checkpoint)
        if s.get("want_tune") and s.get("train_result") and not s.get("tuned_result"):
            return {"next": "tune", "require_approval": True, "reason": "Approve hyperparameter tuning?"}

        # Otherwise, end
        return {"next": "end", "require_approval": False, "reason": "Workflow complete or idle."}
