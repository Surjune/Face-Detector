"""Deriving the subject's name.

A name is what makes targeted platform search possible — you cannot run a
`site:` query against a face. It is a search term only: nothing is anchored on
it, and every lead it produces is still face-verified.
"""

from __future__ import annotations

from typing import Any

import pytest

import src.search.identity as identity_module
from src.search.identity import derive_identity
from src.search.provider import Candidate, Identity


def candidate(title: str, index: int = 0) -> Candidate:
    return Candidate(
        position=index + 1,
        title=title,
        page_url=f"https://example.com/{index}",
        image_urls=("https://cdn.example.com/x.jpg",),
        source="example.com",
    )


class TestLensIdentity:
    def test_prefers_what_lens_already_resolved(self) -> None:
        """Free, deterministic, and disambiguated by Google's own entity graph."""
        lens = Identity(name="Sundar Pichai", origin="lens_related_content")
        result = derive_identity(lens, [candidate("Someone Else - Wikipedia")])

        assert result is not None
        assert result.name == "Sundar Pichai"
        assert result.origin == "lens_related_content"

    def test_does_not_call_the_llm_when_lens_answered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*args: Any, **kwargs: Any) -> str:
            raise AssertionError("the LLM must not be called")

        monkeypatch.setattr(identity_module.llm, "complete", explode)
        lens = Identity(name="Sundar Pichai", origin="lens_related_content")
        assert derive_identity(lens, []) is not None


class TestTitleFrequency:
    def test_finds_the_name_repeated_across_titles(self) -> None:
        titles = [
            "Sundar Pichai - Wikipedia",
            "Sundar Pichai on the future of AI",
            "Google chief Sundar Pichai speaks",
        ]
        result = derive_identity(None, [candidate(t, i) for i, t in enumerate(titles)])

        assert result is not None
        assert result.name == "Sundar Pichai"
        assert result.origin == "title_frequency"

    def test_ignores_publication_names(self) -> None:
        """Mastheads recur more often than the subject in search results."""
        titles = [
            "Business Insider - Sundar Pichai on AI",
            "Business Insider - Sundar Pichai interview",
            "Business Insider - Sundar Pichai profile",
        ]
        result = derive_identity(None, [candidate(t, i) for i, t in enumerate(titles)])

        assert result is not None
        assert result.name == "Sundar Pichai"

    def test_ignores_a_name_seen_only_once(self) -> None:
        """One appearance is as likely to be a passing mention as the subject."""
        titles = ["Some Random Person visits", "An unrelated headline entirely"]
        assert derive_identity(
            None, [candidate(t, i) for i, t in enumerate(titles)], use_llm=False
        ) is None

    def test_returns_none_without_titles(self) -> None:
        assert derive_identity(None, [], use_llm=False) is None


class TestLLMFallback:
    def test_used_only_when_frequency_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(identity_module.llm, "available", lambda: True)
        monkeypatch.setattr(identity_module.llm, "complete", lambda *a, **k: "Ada Lovelace")

        result = derive_identity(None, [candidate("an ambiguous headline")])
        assert result is not None
        assert result.name == "Ada Lovelace"
        assert result.origin == "llm"

    def test_skipped_when_no_key_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pipeline must behave identically with no LLM available."""
        monkeypatch.setattr(identity_module.llm, "available", lambda: False)
        assert derive_identity(None, [candidate("an ambiguous headline")]) is None

    def test_honours_the_models_unknown_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(identity_module.llm, "available", lambda: True)
        monkeypatch.setattr(identity_module.llm, "complete", lambda *a, **k: "UNKNOWN")
        assert derive_identity(None, [candidate("an ambiguous headline")]) is None

    def test_rejects_a_rambling_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A sentence is not a name, and would poison every site: query."""
        monkeypatch.setattr(identity_module.llm, "available", lambda: True)
        monkeypatch.setattr(
            identity_module.llm,
            "complete",
            lambda *a, **k: "I believe this person is most likely Ada Lovelace, the mathematician",
        )
        assert derive_identity(None, [candidate("an ambiguous headline")]) is None

    def test_a_provider_failure_does_not_break_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.llm import LLMError

        monkeypatch.setattr(identity_module.llm, "available", lambda: True)

        def fail(*args: Any, **kwargs: Any) -> str:
            raise LLMError("provider is down")

        monkeypatch.setattr(identity_module.llm, "complete", fail)
        assert derive_identity(None, [candidate("an ambiguous headline")]) is None
