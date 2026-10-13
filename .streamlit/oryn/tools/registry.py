"""Type-safe Tool Registry and Base Contracts."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from oryn.models.schemas import PermissionCategory, ToolResult


class BaseTool(ABC):
    name: str
    description: str
    category: PermissionCategory
    timeout: float = 5.0

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, str]:
        return {name: tool.description for name, tool in self._tools.items()}
