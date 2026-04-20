"""Conversation Management Module

This module is responsible for managing conversation history, including adding conversation turns, limiting conversation length, and formatting conversation history.
"""
from __future__ import annotations

from typing import Dict, List


class ConversationBuffer:
    """Conversation Buffer
    
    Used to store and manage conversation history, supporting the addition of user and assistant messages while limiting the length of the history.
    
    Attributes:
        max_turns: Maximum number of conversation turns (each turn includes a user and an assistant message).
        turns: List of conversation turns.
    """
    def __init__(self, max_turns: int):
        """Initializes the conversation buffer.
        
        Args:
            max_turns: Maximum number of conversation turns.
        """
        self.max_turns = max_turns
        self.turns: List[Dict[str, str]] = []

    def add_user(self, text: str) -> None:
        """Adds a user message.
        
        Args:
            text: User message text.
        """
        self.turns.append({"role": "user", "text": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        """Adds an assistant message.
        
        Args:
            text: Assistant message text.
        """
        self.turns.append({"role": "assistant", "text": text})
        self._trim()

    def _trim(self) -> None:
        """Trims the conversation history to stay within the maximum number of turns.
        
        Groups the conversation history by turns (user + assistant) and keeps only the latest `max_turns`.
        """
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
        """Formats the conversation history for use in prompt templates.
        
        Returns:
            A formatted conversation history string.
        """
        if not self.turns:
            return ""
        lines: List[str] = []
        for t in self.turns:
            if t["role"] == "user":
                lines.append(f"User: {t['text']}")
            else:
                lines.append(f"Assistant: {t['text']}")
        return "\n".join(lines)

