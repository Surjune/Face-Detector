"""Receipt canonicalisation.

These are the most important tests in the project. If the digest is not
byte-reproducible on a machine other than the one that produced it, the on-chain
record proves nothing, and every one of these cases is a way that could silently
go wrong.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.chain.canonical import (
    MatchRecord,
    Receipt,
    SearchRecord,
    canonical_bytes,
    receipt_digest,
    receipt_from_dict,
    receipt_to_dict,
    sha256_hex,
)
from src.config import SIMILARITY_DECIMALS
from src.errors import EvidenceError


def make_receipt(**overrides: object) -> Receipt:
    match = MatchRecord(
        post_url="https://example.com/p/abc123",
        image_url="https://cdn.example.com/abc123.jpg",
        image_sha256="a" * 64,
        page_title="A post by someone",
        similarity=0.7123,
    )
    search = SearchRecord(
        provider="serpapi_google_lens",
        query_image_sha256="b" * 64,
        candidate_count=18,
        retrieved_at="2026-09-03T15:40:00Z",
    )
    receipt = Receipt(
        input_image_sha256="c" * 64,
        embedding_sha256="d" * 64,
        match=match,
        search=search,
    )
    return replace(receipt, **overrides)  # type: ignore[arg-type]


def with_title(title: str) -> Receipt:
    receipt = make_receipt()
    return replace(receipt, match=replace(receipt.match, page_title=title))


class TestDigestStability:
    def test_is_deterministic(self) -> None:
        assert receipt_digest(make_receipt()) == receipt_digest(make_receipt())

    def test_has_the_shape_of_a_bytes32(self) -> None:
        digest = receipt_digest(make_receipt())
        assert digest.startswith("0x")
        assert len(digest) == 66
        bytes.fromhex(digest[2:])

    def test_keys_are_serialised_in_sorted_order(self) -> None:
        """Insertion order must not reach the hash."""
        payload = canonical_bytes(make_receipt()).decode("ascii")
        keys = list(json.loads(payload).keys())
        assert keys == sorted(keys)

    def test_output_carries_no_insignificant_whitespace(self) -> None:
        assert b", " not in canonical_bytes(make_receipt())
        assert b": " not in canonical_bytes(make_receipt())

    def test_output_is_pure_ascii(self) -> None:
        """Escaping non-ASCII removes any dependence on the platform encoding."""
        canonical_bytes(with_title("Café — Zürich 東京")).decode("ascii")


class TestDigestSensitivity:
    def test_a_single_character_edit_changes_the_digest(self) -> None:
        assert receipt_digest(with_title("A post by someonf")) != receipt_digest(make_receipt())

    def test_a_different_post_url_changes_the_digest(self) -> None:
        original = make_receipt()
        edited = replace(
            original, match=replace(original.match, post_url="https://example.com/p/abc124")
        )
        assert receipt_digest(edited) != receipt_digest(original)

    def test_a_swapped_image_changes_the_digest(self) -> None:
        original = make_receipt()
        edited = replace(original, match=replace(original.match, image_sha256="e" * 64))
        assert receipt_digest(edited) != receipt_digest(original)

    def test_a_meaningful_similarity_change_changes_the_digest(self) -> None:
        original = make_receipt()
        edited = replace(original, match=replace(original.match, similarity=0.8123))
        assert receipt_digest(edited) != receipt_digest(original)


class TestCrossPlatformEquivalence:
    def test_float_noise_below_the_rounding_floor_is_ignored(self) -> None:
        """Two machines can compute cosine similarity to different last digits."""
        original = make_receipt()
        nudged = replace(
            original,
            match=replace(original.match, similarity=0.7123 + 10 ** -(SIMILARITY_DECIMALS + 3)),
        )
        assert receipt_digest(nudged) == receipt_digest(original)

    def test_decomposed_and_precomposed_unicode_agree(self) -> None:
        """An accented letter as one code point, or as a letter plus a combining mark.

        Both render identically and mean the same thing, but are different byte
        sequences. Page titles scraped from the web arrive in both forms.
        """
        precomposed = "Café"
        decomposed = "Café"
        assert precomposed != decomposed

        assert receipt_digest(with_title(decomposed)) == receipt_digest(with_title(precomposed))

    def test_line_endings_agree(self) -> None:
        assert receipt_digest(with_title("line one\r\nline two")) == receipt_digest(
            with_title("line one\nline two")
        )

    def test_surrounding_whitespace_is_ignored(self) -> None:
        assert receipt_digest(with_title("  A post by someone  ")) == receipt_digest(
            make_receipt()
        )


class TestRoundTrip:
    def test_survives_a_json_round_trip(self) -> None:
        """This is exactly what `verify` does with a stored receipt.json."""
        original = make_receipt()
        stored = json.loads(json.dumps(receipt_to_dict(original)))
        assert receipt_digest(receipt_from_dict(stored)) == receipt_digest(original)

    def test_round_trip_preserves_a_non_ascii_title(self) -> None:
        original = with_title("Café 東京")
        stored = json.loads(json.dumps(receipt_to_dict(original)))
        assert receipt_digest(receipt_from_dict(stored)) == receipt_digest(original)

    def test_rejects_a_receipt_missing_fields(self) -> None:
        stored = receipt_to_dict(make_receipt())
        del stored["match"]["image_sha256"]
        with pytest.raises(EvidenceError):
            receipt_from_dict(stored)

    def test_rejects_a_receipt_with_the_wrong_types(self) -> None:
        stored = receipt_to_dict(make_receipt())
        stored["search"]["candidate_count"] = "not a number"
        with pytest.raises(EvidenceError):
            receipt_from_dict(stored)


class TestImageHashing:
    def test_matches_a_known_value(self) -> None:
        assert sha256_hex(b"") == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_differs_for_different_bytes(self) -> None:
        assert sha256_hex(b"one") != sha256_hex(b"two")
