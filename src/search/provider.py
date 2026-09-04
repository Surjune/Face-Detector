"""What a search backend has to provide.

Several backends sit behind one interface so the pipeline does not care where
candidates came from: Google Lens, a site-scoped web search, the YouTube Data
API, or a recorded response replayed from an evidence folder. Whichever runs,
the pipeline scores every candidate against the probe face itself — a provider
only supplies leads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from src.search.platforms import SOCIAL_PLATFORMS, Platform, classify


def utc_now() -> str:
    """Timestamp in the one format used throughout the receipt."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Source(str, Enum):
    """Where a candidate was found.

    Recorded per candidate so the evidence shows which search produced each
    lead, rather than presenting one undifferentiated list.
    """

    LENS_VISUAL = "lens_visual_match"
    LENS_ORGANIC = "lens_organic_result"
    LENS_VIDEO = "lens_short_video"
    SITE_SEARCH = "site_search"
    YOUTUBE_API = "youtube_data_api"


@dataclass(frozen=True)
class Candidate:
    """One lead from a search, before any face check."""

    position: int
    title: str
    page_url: str
    image_urls: tuple[str, ...]
    source: str
    origin: Source = Source.LENS_VISUAL

    @property
    def image_url(self) -> str:
        """The preferred image URL.

        Candidates carry an ordered chain rather than a single URL: social
        platforms serve their canonical image through crawler endpoints that
        return HTML to anyone else, and the search engine's own thumbnail is
        the reliable fallback. See `fetch.download_first_available`.
        """
        return self.image_urls[0] if self.image_urls else ""

    @property
    def platform(self) -> Platform:
        return classify(self.page_url)

    @property
    def is_social(self) -> bool:
        return self.platform in SOCIAL_PLATFORMS

    def describe(self) -> str:
        """Short label for the terminal candidate table."""
        return self.source or self.platform.label


@dataclass(frozen=True)
class Identity:
    """The subject's likely name, used to drive targeted platform searches."""

    name: str
    origin: str
    confidence: str = "unconfirmed"


@dataclass(frozen=True)
class SearchResponse:
    """Everything a provider returned for one query."""

    provider: str
    query_image_url: str
    retrieved_at: str
    candidates: list[Candidate]
    raw: dict[str, Any] = field(default_factory=dict)
    identity: Identity | None = None


@runtime_checkable
class SearchProvider(Protocol):
    """A reverse image search backend."""

    name: str

    def search(self, image_url: str) -> SearchResponse:
        """Find pages showing this image. Raises SearchError on failure."""
        ...
