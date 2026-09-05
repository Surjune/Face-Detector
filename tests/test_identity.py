"""Deriving the subject's name.

A name is what makes targeted platform search possible — you cannot run a
`site:` query against a face. It is a search term only: nothing is anchored on
it, and every lead it produces is still face-verified.
"""

from __future__ import annotations

from typing import Any

import pytest

import src.search.identity as identity_module
from src.search.identity import confirm_identity, derive_identity
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


class TestConfirmIdentity:
    """What the run may claim about identity, judged only on verified results.

    This is the guard on the defect where a name read off visually similar pages
    was announced as the subject's identity, in a run whose every candidate was
    then rejected by the face check.
    """

    def test_nothing_verified_means_nothing_identified(self) -> None:
        guess = Identity(name="Rasindu Jayan", origin="title_frequency")
        verdict = confirm_identity(guess, [])

        assert verdict.is_confirmed is False
        assert verdict.name is None
        assert verdict.verified == 0
        # Kept, so the run can say what was searched for and why it means nothing.
        assert verdict.search_term == "Rasindu Jayan"

    def test_a_guess_named_beside_a_matched_face_is_confirmed(self) -> None:
        guess = Identity(name="Sundar Pichai", origin="lens_related_content")
        verdict = confirm_identity(guess, [candidate("Sundar Pichai on X")])

        assert verdict.is_confirmed is True
        assert verdict.name == "Sundar Pichai"
        assert verdict.supporting == 1
        assert verdict.origin == "lens_related_content"

    def test_matched_titles_outrank_a_guess_they_contradict(self) -> None:
        """The face check is the authority, so its results name the subject."""
        guess = Identity(name="Someone Else", origin="title_frequency")
        verified = [
            candidate("Ada Lovelace - profile", 0),
            candidate("Photos of Ada Lovelace", 1),
        ]
        verdict = confirm_identity(guess, verified)

        assert verdict.name == "Ada Lovelace"
        assert verdict.origin == "verified_titles"
        assert verdict.search_term == "Someone Else"

    def test_matches_without_names_identify_nobody(self) -> None:
        """A matched face on an untitled page proves a match, not a name."""
        verdict = confirm_identity(None, [candidate("", 0), candidate("", 1)])

        assert verdict.is_confirmed is False
        assert verdict.name is None
        assert verdict.verified == 2

    def test_one_verified_title_is_enough(self) -> None:
        """Unverified titles need two appearances; a verified one needs one."""
        verdict = confirm_identity(None, [candidate("Ada Lovelace at work")])

        assert verdict.name == "Ada Lovelace"
        assert verdict.supporting == 1

    def test_the_name_is_matched_case_insensitively(self) -> None:
        guess = Identity(name="Ada Lovelace", origin="llm")
        verdict = confirm_identity(guess, [candidate("ADA LOVELACE / gallery")])

        assert verdict.is_confirmed is True
        assert verdict.supporting == 1
