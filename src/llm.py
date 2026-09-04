"""Optional LLM access, over free-tier providers only.

Used for one narrow job: reading the subject's name out of messy search-result
titles when Google Lens has not already supplied it. That is a small text
extraction, so a free tier covers it with room to spare — one or two short
calls per run.

Two providers are supported, both with a genuinely free tier that needs no
payment card, and both speaking the OpenAI-compatible chat-completions shape,
so a single `requests` call serves either. No provider SDK is added as a
dependency.

The whole module is optional. With no key configured, `available()` is False,
the caller falls back to the deterministic path, and the pipeline behaves
identically — an LLM refines the identity, it is never required to find one.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from src.config import LLM_MAX_TOKENS, LLM_PROVIDERS, LLM_TIMEOUT_SECONDS, load_settings
from src.errors import PipelineError


class LLMError(PipelineError):
    code = "llm_failed"


@dataclass(frozen=True)
class LLMConfig:
    """A resolved provider: which one, which key, which model."""

    provider: str
    api_key: str
    base_url: str
    model: str


def resolve() -> LLMConfig | None:
    """Pick the first configured provider, or None if none is set.

    Providers are tried in the order declared in config, so a user holding keys
    for both gets a predictable choice rather than an arbitrary one.
    """
    settings = load_settings()
    for provider in LLM_PROVIDERS:
        key = settings.llm_keys.get(provider.env_var)
        if key:
            return LLMConfig(
                provider=provider.name,
                api_key=key,
                base_url=provider.base_url,
                model=settings.llm_model or provider.default_model,
            )
    return None


def available() -> bool:
    """Whether any LLM provider is configured."""
    return resolve() is not None


def complete(prompt: str, *, system: str | None = None) -> str:
    """Send one prompt and return the model's text reply.

    Raises:
        LLMError: no provider configured, the call failed, or the response had
            an unexpected shape.
    """
    config = resolve()
    if config is None:
        raise LLMError("No LLM provider configured; set GROQ_API_KEY or GEMINI_API_KEY")

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            f"{config.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.model,
                "messages": messages,
                "max_tokens": LLM_MAX_TOKENS,
                # Deterministic: the same titles should yield the same name, so
                # a re-run of the pipeline is reproducible.
                "temperature": 0,
            },
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise LLMError(f"Could not reach {config.provider}: {exc}") from exc

    if response.status_code != 200:
        raise LLMError(
            f"{config.provider} returned HTTP {response.status_code}",
            body=response.text[:200],
        )

    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"{config.provider} returned an unexpected response: {exc}") from exc

    if not isinstance(content, str):
        raise LLMError(f"{config.provider} returned no text content")
    return content.strip()
