"""Ollama客户端模块

该模块提供了与Ollama API交互的功能，用于生成文本响应。
主要功能包括：
1. 调用Ollama的generate API生成文本
2. 提取生成过程中的token使用统计信息
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import requests


def generate_raw(
    *,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    num_predict: int,
    stream: bool = False,
    base_url: str = "http://localhost:11434",
    timeout_s: int = 300,
) -> Dict[str, Any]:
    """调用Ollama API生成文本
    
    首先尝试使用ollama Python库，如果失败则使用HTTP请求直接调用API。
    
    Args:
        model: 模型名称
        prompt: 提示文本
        temperature: 温度参数，控制生成文本的随机性
        top_p: 顶部p参数，控制生成文本的多样性
        num_predict: 预测的token数量
        stream: 是否流式返回结果
        base_url: Ollama API的基础URL
        timeout_s: 请求超时时间（秒）
    
    Returns:
        API返回的响应字典
    """
    try:
        import ollama

        return ollama.generate(
            model=model,
            prompt=prompt,
            stream=stream,
            options={
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
            },
        )
    except Exception:
        # 如果ollama库不可用，使用HTTP请求
        url = base_url.rstrip("/") + "/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
            },
        }
        r = requests.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
        return r.json()


def token_stats(resp: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """提取token使用统计信息
    
    Args:
        resp: API返回的响应字典
    
    Returns:
        包含token统计信息的字典，包括prompt_eval_count和eval_count
    """
    return {
        "prompt_eval_count": resp.get("prompt_eval_count"),  # 提示文本的token数量
        "eval_count": resp.get("eval_count"),  # 生成文本的token数量
    }
