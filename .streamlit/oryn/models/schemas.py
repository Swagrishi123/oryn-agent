"""Data schemas for ORYN.AI agent architecture."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time
import uuid


class PermissionCategory(str, Enum):
    READ_ONLY = "CATEGORY_1_READ_ONLY"
    REVERSIBLE = "CATEGORY_2_REVERSIBLE"
    CONSEQUENTIAL = "CATEGORY_3_CONSEQUENTIAL"
    PROHIBITED = "CATEGORY_4_PROHIBITED"


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class ActionProposal:
    tool_name: str
    params: Dict[str, Any]
    category: PermissionCategory
    description: str
    target: str
    action_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 300.0)


@dataclass
class ApprovalDecision:
    action_id: str
    status: ApprovalStatus
    decided_by: str = "user"
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolResult:
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    provenance: str = "direct"
    is_live: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMessage:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    content: str
    actions_executed: List[str] = field(default_factory=list)
    pending_approval: Optional[ActionProposal] = None
    plot_bytes: Optional[bytes] = None
    excel_bytes: Optional[bytes] = None
    execution_time: float = 0.0
