from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


def list_local_models(
    *,
    base_url: str = "http://localhost:11434",
    timeout_s: int = 5,
) -> List[str]:
    names: List[str] = []

    try:
        import ollama

        resp = ollama.list()
        models = []
        if isinstance(resp, dict):
            models = resp.get("models") or []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip()
            if name and name not in names:
                names.append(name)
        if names:
            return names
    except Exception:
        pass

    url = base_url.rstrip("/") + "/api/tags"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    models = data.get("models") if isinstance(data, dict) else []
    if not isinstance(models, list):
        return []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


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
