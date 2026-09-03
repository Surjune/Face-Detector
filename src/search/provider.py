"""What a reverse-image-search backend has to provide.

Several backends sit behind one interface so the pipeline does not care where
candidates came from: a live provider, or a recorded response replayed from an
evidence folder. Whichever runs, the pipeline scores every candidate against the
probe face itself — the provider only supplies leads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable


def utc_now() -> str:
    """Timestamp in the one format used throughout the receipt."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Candidate:
    """One lead from a reverse image search, before any face check."""

    position: int
    title: str
    page_url: str
    image_url: str
    source: str

    def describe(self) -> str:
        """Short label for the terminal candidate table."""
        return self.source or self.page_url


@dataclass(frozen=True)
class SearchResponse:
    """Everything a provider returned for one query."""

    provider: str
    query_image_url: str
    retrieved_at: str
    candidates: list[Candidate]
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SearchProvider(Protocol):
    """A reverse image search backend."""

    name: str

    def search(self, image_url: str) -> SearchResponse:
        """Find pages showing this image. Raises SearchError on failure."""
        ...
