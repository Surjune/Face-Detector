"""The optional LLM client.

Free-tier providers only, and never a hard dependency: with no key configured
the pipeline must behave exactly as it does without one.
"""

from __future__ import annotations

from typing import Any

import pytest

import src.llm as llm
from src.config import LLM_PROVIDERS
from src.llm import LLMError, complete, resolve


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


def settings_with(monkeypatch: pytest.MonkeyPatch, **keys: str) -> None:
    """Pretend a given set of provider keys is configured."""
    from src.config import Settings

    monkeypatch.setattr(
        llm,
        "load_settings",
        lambda: Settings(
            serpapi_key=None,
            sepolia_rpc_url="https://example.invalid",
            private_key=None,
            registry_address=None,
            github_raw_base=None,
            youtube_api_key=None,
            llm_keys=dict(keys),
            llm_model=None,
        ),
    )


class TestProviderResolution:
    def test_no_key_means_no_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings_with(monkeypatch)
        assert resolve() is None
        assert not llm.available()

    def test_picks_a_configured_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings_with(monkeypatch, GROQ_API_KEY="k")
        config = resolve()
        assert config is not None
        assert config.provider == "groq"

    def test_prefers_the_first_declared_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Holding both keys must give a predictable choice, not an arbitrary one."""
        settings_with(monkeypatch, GROQ_API_KEY="k", GEMINI_API_KEY="k")
        config = resolve()
        assert config is not None
        assert config.provider == LLM_PROVIDERS[0].name

    def test_every_default_model_is_declared(self) -> None:
        for provider in LLM_PROVIDERS:
            assert provider.default_model
            assert provider.base_url.startswith("https://")


class TestComplete:
    def test_errors_without_a_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings_with(monkeypatch)
        with pytest.raises(LLMError, match="No LLM provider"):
            complete("hello")

    def test_returns_the_reply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings_with(monkeypatch, GEMINI_API_KEY="k")
        monkeypatch.setattr(
            llm.requests,
            "post",
            lambda *a, **k: FakeResponse(
                {"choices": [{"message": {"content": " Ada Lovelace "}}]}
            ),
        )
        assert complete("who?") == "Ada Lovelace"

    def test_rejects_an_empty_reply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A reasoning model that runs out of budget answers 200 with no content.

        Groq's gpt-oss models do exactly this when max_tokens is too small: the
        reasoning consumes the budget and `content` comes back empty. Returning
        that as if it were an answer would put an empty name into a site: query.
        """
        settings_with(monkeypatch, GEMINI_API_KEY="k")
        monkeypatch.setattr(
            llm.requests,
            "post",
            lambda *a, **k: FakeResponse(
                {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
            ),
        )
        with pytest.raises(LLMError, match="no text content"):
            complete("who?")

    def test_surfaces_an_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings_with(monkeypatch, GEMINI_API_KEY="k")
        monkeypatch.setattr(
            llm.requests, "post", lambda *a, **k: FakeResponse({}, status_code=404)
        )
        with pytest.raises(LLMError, match="404"):
            complete("who?")

    def test_surfaces_an_unexpected_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings_with(monkeypatch, GEMINI_API_KEY="k")
        monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResponse({"oops": 1}))
        with pytest.raises(LLMError, match="unexpected response"):
            complete("who?")
