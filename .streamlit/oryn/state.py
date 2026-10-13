"""Bounded Conversation and Execution State Management."""
from typing import Dict, List
from oryn.models.schemas import AgentMessage


class SessionState:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.messages: List[AgentMessage] = []

    def add_message(self, role: str, content: str) -> None:
        if self.messages and self.messages[-1].role == role and self.messages[-1].content == content:
            return
        self.messages.append(AgentMessage(role=role, content=content))

    def get_context_window(self) -> List[Dict[str, str]]:
        """Returns bounded conversation history, preventing context overflow."""
        bounded = self.messages[-(self.max_turns * 2):]
        return [{"role": m.role, "content": m.content} for m in bounded]

    def clear(self) -> None:
        self.messages.clear()
