"""
Cursor Agent CLI provider (`cursor_agent`).

This is the Agent CLI Connect `Run` route on `agentn`, not Cloud Agents
`/cursor` (`custom_llm_provider="cursor"`).
"""

from .chat.transformation import CursorAgentConfig

__all__ = ["CursorAgentConfig"]
