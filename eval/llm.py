# ============================================================
# GROQ HELPER — shared client, .env loading, retry on rate limit
# ============================================================
# Only needed for answer generation, LLM-as-judge, and synthetic
# test-set generation. Retrieval metrics run without any API key.
# ============================================================

import json
import os
import re
import time
from pathlib import Path

from groq import Groq

JUDGE_MODEL = "llama-3.3-70b-versatile"   # bigger model judges the 8B generator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv():
    """Minimal .env loader so we don't add a dependency."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_groq_client() -> Groq:
    _load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY not set. Create a .env file in the project root "
            'containing:  GROQ_API_KEY="gsk_..."  (free key: console.groq.com)'
        )
    return Groq(api_key=api_key)


def chat(client: Groq, prompt: str, model: str, temperature: float = 0.0,
         max_retries: int = 5) -> str:
    """One-shot chat completion with exponential backoff on rate limits."""
    delay = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            transient = "rate" in str(e).lower() or "429" in str(e) or "503" in str(e)
            if not transient or attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def extract_json(text: str):
    """Parse the first JSON object found in an LLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in response: {text[:200]}")
    return json.loads(match.group(0))
