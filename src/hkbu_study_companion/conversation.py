from __future__ import annotations

from typing import Dict, List


class ConversationBuffer:
    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self.turns: List[Dict[str, str]] = []

    def add_user(self, text: str) -> None:
        self.turns.append({"role": "user", "text": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.turns.append({"role": "assistant", "text": text})
        self._trim()

    def _trim(self) -> None:
        if self.max_turns <= 0:
            self.turns = []
            return
        # Keep the latest max_turns user-assistant pairs (2 messages per pair).
        max_msgs = self.max_turns * 2
        if len(self.turns) > max_msgs:
            self.turns = self.turns[-max_msgs:]

    def format_for_prompt(self) -> str:
        if not self.turns:
            return ""
        lines: List[str] = []
        for t in self.turns:
            if t["role"] == "user":
                lines.append(f"User: {t['text']}")
            else:
                lines.append(f"Assistant: {t['text']}")
        return "\n".join(lines)
