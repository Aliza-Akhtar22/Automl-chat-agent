# app/agents/llm_utils.py
import json
from typing import Any, Dict
from openai import OpenAI

# relies on OPENAI_API_KEY from environment (.env)
client = OpenAI()

def chat_once(
    system: str,
    user: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content

def chat_json(
    system: str,
    user: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """LLM should return JSON (we'll try to parse it)."""
    text = chat_once(system, user, model=model, temperature=temperature)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        
        return {"raw": text}
