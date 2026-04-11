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
    return {
        "prompt_eval_count": resp.get("prompt_eval_count"),
        "eval_count": resp.get("eval_count"),
    }
