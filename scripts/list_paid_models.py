"""Zero-cost discovery of models visible to the *paid* API keys.

Lists every model id the paid OpenAI / Anthropic keys can see, so run configs can
reference real ids instead of guesses. Makes no billable completion calls — only
the providers' free model-list endpoints.

    uv run python scripts/list_paid_models.py

Reads OPENAI_API_KEY_PAID and ANTHROPIC_API_KEY_PAID from the environment / .env.
"""

from __future__ import annotations

import os
import re
import sys

import arabic_ocr_bench  # noqa: F401  -- side effect: loads .env

# ids the user specifically asked about; we want to know their real API strings
WANTED = ("fable", "gpt-5.6", "sol", "terra", "luna", "gpt-5", "opus-5", "sonnet-5")


def _hit(model_id: str) -> bool:
    m = model_id.lower()
    return any(w in m for w in WANTED)


def list_openai() -> list[str]:
    key = os.environ.get("OPENAI_API_KEY_PAID")
    if not key:
        print("OPENAI_API_KEY_PAID not set — skipping OpenAI\n")
        return []
    from openai import OpenAI

    client = OpenAI(api_key=key)
    ids = sorted(m.id for m in client.models.list().data)
    print(f"== OpenAI — {len(ids)} models visible to OPENAI_API_KEY_PAID ==")
    for i in ids:
        print(f"  {'*' if _hit(i) else ' '} {i}")
    print()
    return ids


def list_anthropic() -> list[str]:
    key = os.environ.get("ANTHROPIC_API_KEY_PAID")
    if not key:
        print("ANTHROPIC_API_KEY_PAID not set — skipping Anthropic\n")
        return []
    import httpx

    ids: list[str] = []
    url = "https://api.anthropic.com/v1/models?limit=1000"
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    r = httpx.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    for m in r.json().get("data", []):
        ids.append(m["id"])
    ids.sort()
    print(f"== Anthropic — {len(ids)} models visible to ANTHROPIC_API_KEY_PAID ==")
    for i in ids:
        print(f"  {'*' if _hit(i) else ' '} {i}")
    print()
    return ids


def main() -> int:
    oai = list_openai()
    ant = list_anthropic()

    print("== Resolving requested names ==")
    for want in ("gpt-5.6", "sol", "terra", "luna", "fable"):
        pool = ant if want == "fable" else oai
        hits = [i for i in pool if want in i.lower()]
        print(f"  {want:10s} -> {hits or '(not found on this key)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
