"""Disabled legacy code runner.

Executing model-generated Python with exec() inside the Streamlit process is not a
security boundary. This module intentionally fails closed until a separately
isolated execution service with resource limits and filesystem/network controls
is implemented.
"""

from typing import Optional


def clean_code_snippet(code: str) -> str:
    """Retained for compatibility; does not execute or sanitize code for execution."""
    return (code or "").strip()


def tool_python_interpreter(
    code: str, hitl_enabled: bool = True
) -> tuple[str, Optional[bytes], Optional[bytes]]:
    """Fail closed. Approval toggles do not make in-process exec safe."""
    return (
        "Python execution is disabled for safety. A secure isolated execution "
        "environment has not been configured.",
        None,
        None,
    )
