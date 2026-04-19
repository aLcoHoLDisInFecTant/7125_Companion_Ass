"""对话管理模块

该模块负责管理对话历史，包括添加对话轮次、限制对话长度和格式化对话历史。
"""
from __future__ import annotations

from typing import Dict, List


class ConversationBuffer:
    """对话缓冲区
    
    用于存储和管理对话历史，支持添加用户和助手的消息，并限制对话历史的长度。
    
    Attributes:
        max_turns: 最大对话轮次数（每轮包括用户和助手的消息）
        turns: 对话轮次列表
    """
    def __init__(self, max_turns: int):
        """初始化对话缓冲区
        
        Args:
            max_turns: 最大对话轮次数
        """
        self.max_turns = max_turns
        self.turns: List[Dict[str, str]] = []

    def add_user(self, text: str) -> None:
        """添加用户消息
        
        Args:
            text: 用户消息文本
        """
        self.turns.append({"role": "user", "text": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        """添加助手消息
        
        Args:
            text: 助手消息文本
        """
        self.turns.append({"role": "assistant", "text": text})
        self._trim()

    def _trim(self) -> None:
        """修剪对话历史，保持在最大轮次数以内
        
        将对话历史按轮次（用户+助手）分组，保留最新的max_turns轮。
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
        """格式化对话历史，用于提示模板
        
        Returns:
            格式化后的对话历史字符串
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
