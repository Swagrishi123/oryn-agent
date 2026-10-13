"""Centralized Policy & Human-in-the-Loop Enforcement Layer."""
import logging
from typing import Dict
from oryn.models.schemas import ActionProposal, ApprovalDecision, ApprovalStatus, PermissionCategory

logger = logging.getLogger("oryn.policy")


class PolicyEngine:
    def __init__(self, hitl_enabled: bool = True):
        self.hitl_enabled = hitl_enabled
        self._decisions: Dict[str, ApprovalDecision] = {}

    def evaluate(self, proposal: ActionProposal) -> ApprovalStatus:
        """Determines whether an action can execute immediately or requires human authorization."""
        if proposal.category == PermissionCategory.PROHIBITED:
            logger.warning(f"Action {proposal.action_id} [{proposal.tool_name}] rejected: Prohibited.")
            return ApprovalStatus.REJECTED

        if not self.hitl_enabled:
            # If HITL safety guardrails are off, read-only and reversible execute directly
            if proposal.category in (PermissionCategory.READ_ONLY, PermissionCategory.REVERSIBLE):
                return ApprovalStatus.APPROVED
            # Consequential actions still fail closed unless explicitly approved
            return ApprovalStatus.PENDING

        # Under HITL Safety Mode:
        if proposal.category == PermissionCategory.READ_ONLY:
            return ApprovalStatus.APPROVED

        # Check for pre-existing approved decision for this exact action ID
        if proposal.action_id in self._decisions:
            decision = self._decisions[proposal.action_id]
            if decision.status == ApprovalStatus.APPROVED:
                if proposal.timestamp > proposal.expires_at:
                    logger.warning(f"Approval for {proposal.action_id} expired.")
                    return ApprovalStatus.EXPIRED
                return ApprovalStatus.APPROVED
            return decision.status

        logger.info(f"Action {proposal.action_id} requires human approval: {proposal.description}")
        return ApprovalStatus.PENDING

    def record_decision(self, decision: ApprovalDecision) -> None:
        self._decisions[decision.action_id] = decision
        logger.info(f"Recorded approval decision: {decision.action_id} -> {decision.status}")
