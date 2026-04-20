"""Ollama Client Module

This module provides functionality for interacting with the Ollama API to generate text responses.
Main features include:
1. Calling Ollama's `generate` API to generate text.
2. Extracting token usage statistics from the generation process.
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
    """Calls the Ollama API to generate text.
    
    First attempts to use the `ollama` Python library; if that fails, falls back to making direct HTTP requests to the API.
    
    Args:
        model: Model name.
        prompt: Prompt text.
        temperature: Temperature parameter, controlling the randomness of generated text.
        top_p: Top-p parameter, controlling the diversity of generated text.
        num_predict: Number of tokens to predict.
        stream: Whether to stream the results.
        base_url: Base URL of the Ollama API.
        timeout_s: Request timeout in seconds.
    
    Returns:
        The response dictionary returned by the API.
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
        # If the ollama library is unavailable, use HTTP requests
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
    """Extracts token usage statistics.
    
    Args:
        resp: The response dictionary returned by the API.
    
    Returns:
        A dictionary containing token statistics, including `prompt_eval_count` and `eval_count`.
    """
    return {
        "prompt_eval_count": resp.get("prompt_eval_count"),  # Number of tokens in the prompt
        "eval_count": resp.get("eval_count"),  # Number of tokens in the generated text
    }

