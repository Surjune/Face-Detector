"""The run receipt and its canonical, hashable form.

Everything the pipeline discovered about a match is collected into a `Receipt`.
The receipt is serialised to one exact byte string and hashed; that hash is what
goes on chain.

The serialisation has to be reproducible on a machine that has never seen the
original run, or verification proves nothing. Three things would otherwise break
it, and each is handled explicitly below: key ordering, float formatting, and
Unicode representation.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Any

from src.config import PIPELINE_VERSION, RECEIPT_SCHEMA, SIMILARITY_DECIMALS
from src.errors import EvidenceError


@dataclass(frozen=True)
class MatchRecord:
    """The single web result the pipeline settled on."""

    post_url: str
    image_url: str
    image_sha256: str
    page_title: str
    similarity: float


@dataclass(frozen=True)
class SearchRecord:
    """How the match was found, so the search can be audited later."""

    provider: str
    query_image_sha256: str
    candidate_count: int
    retrieved_at: str


@dataclass(frozen=True)
class Receipt:
    """The complete, hashable record of one pipeline run."""

    input_image_sha256: str
    embedding_sha256: str
    match: MatchRecord
    search: SearchRecord
    schema: str = RECEIPT_SCHEMA
    pipeline_version: str = PIPELINE_VERSION


def sha256_hex(data: bytes) -> str:
    """Hex sha256 of raw bytes, used for image content hashes."""
    return hashlib.sha256(data).hexdigest()


def normalise_text(value: str) -> str:
    """Reduce a free-text field to one unambiguous representation.

    Page titles arrive from arbitrary web pages. The same visible title can be
    encoded as pre-composed or decomposed Unicode, and line endings differ by
    platform; either would produce a different digest for identical content.
    """
    text = unicodedata.normalize("NFC", value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def format_similarity(value: float) -> str:
    """Render a similarity score as a fixed-width decimal string.

    Emitted as a string, not a JSON number: float repr is not guaranteed
    identical across platforms and interpreters, and a single differing digit
    would change the digest and fail an otherwise valid verification.
    """
    return f"{round(float(value), SIMILARITY_DECIMALS):.{SIMILARITY_DECIMALS}f}"


def receipt_to_dict(receipt: Receipt) -> dict[str, Any]:
    """Convert a receipt to the plain structure that gets serialised."""
    return {
        "schema": receipt.schema,
        "pipeline_version": receipt.pipeline_version,
        "input_image_sha256": receipt.input_image_sha256,
        "embedding_sha256": receipt.embedding_sha256,
        "match": {
            "post_url": normalise_text(receipt.match.post_url),
            "image_url": normalise_text(receipt.match.image_url),
            "image_sha256": receipt.match.image_sha256,
            "page_title": normalise_text(receipt.match.page_title),
            "similarity": format_similarity(receipt.match.similarity),
        },
        "search": {
            "provider": receipt.search.provider,
            "query_image_sha256": receipt.search.query_image_sha256,
            "candidate_count": receipt.search.candidate_count,
            "retrieved_at": receipt.search.retrieved_at,
        },
    }


def receipt_from_dict(payload: dict[str, Any]) -> Receipt:
    """Rebuild a receipt from a stored `receipt.json`.

    Raises:
        EvidenceError: the file is missing fields or has the wrong shape.
    """
    try:
        match = payload["match"]
        search = payload["search"]
        return Receipt(
            schema=payload["schema"],
            pipeline_version=payload["pipeline_version"],
            input_image_sha256=payload["input_image_sha256"],
            embedding_sha256=payload["embedding_sha256"],
            match=MatchRecord(
                post_url=match["post_url"],
                image_url=match["image_url"],
                image_sha256=match["image_sha256"],
                page_title=match["page_title"],
                similarity=float(match["similarity"]),
            ),
            search=SearchRecord(
                provider=search["provider"],
                query_image_sha256=search["query_image_sha256"],
                candidate_count=int(search["candidate_count"]),
                retrieved_at=search["retrieved_at"],
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"Malformed receipt: {exc}") from exc


def canonical_bytes(receipt: Receipt) -> bytes:
    """Serialise a receipt to the exact bytes that get hashed.

    `sort_keys` removes any dependence on field insertion order, the separators
    strip insignificant whitespace, and `ensure_ascii` escapes every non-ASCII
    character so the output is pure ASCII and its encoding cannot vary.
    """
    return json.dumps(
        receipt_to_dict(receipt),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def receipt_digest(receipt: Receipt) -> str:
    """The 0x-prefixed sha256 of a receipt: the value anchored on chain."""
    return "0x" + hashlib.sha256(canonical_bytes(receipt)).hexdigest()
