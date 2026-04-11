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
        pairs: List[List[Dict[str, str]]] = []
        cur: List[Dict[str, str]] = []
        for t in self.turns:
            cur.append(t)
            if len(cur) == 2:
                pairs.append(cur)
                cur = []
        if len(pairs) > self.max_turns:
            pairs = pairs[-self.max_turns :]
        self.turns = [x for pair in pairs for x in pair]

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
