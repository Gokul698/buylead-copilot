"""Thin LLM wrapper. Uses Groq (free tier) if GROQ_API_KEY is set, else returns None
so every caller can fall back to a deterministic path. The app always works offline."""
import json
import os
import re

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def llm_available() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def chat(system: str, user: str, json_mode: bool = False, temperature: float = 0.1):
    """Return model text (or parsed dict if json_mode). None if no key / on error."""
    if not llm_available():
        return None
    try:
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        kwargs = dict(
            model=MODEL,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        out = client.chat.completions.create(**kwargs).choices[0].message.content
        if json_mode:
            m = re.search(r"\{.*\}", out, re.S)
            return json.loads(m.group(0)) if m else None
        return out
    except Exception as e:  # network, rate limit, bad JSON -> graceful fallback
        print(f"[llm] falling back: {e}")
        return None
