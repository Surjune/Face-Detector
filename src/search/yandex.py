"""Yandex reverse image search, through SerpApi.

Google Lens is a strong first pass, but it is weak at the case this project
cares about most: finding an ordinary person. Western search engines have
deliberately restricted public face matching for private individuals in
response to privacy regulation. Yandex never applied the same restriction, and
its image retrieval weighs facial geometry heavily, so it matches a face across
changes in pose, lighting and background that defeat Lens.

Published comparisons put Yandex at roughly 65-75% at finding other photographs
of the same person against roughly 30-40% for Google. That gap is the entire
difference between "works for celebrities" and "works for someone with a public
Instagram account", so both engines are queried and their results merged.

Same SerpApi key and the same free 250-searches-a-month budget as the Lens call.
"""

from __future__ import annotations

from typing import Any

import requests

from src.config import (
    DOWNLOAD_TIMEOUT_SECONDS,
    MAX_CANDIDATES,
    SERPAPI_ENDPOINT,
    SERPAPI_YANDEX_ENGINE,
)
from src.errors import SearchNotConfigured, SearchProviderError
from src.search.provider import Candidate, Identity, SearchResponse, Source, utc_now

# Only `image_results` is harvested, and the omission is deliberate.
#
# A Yandex response also carries `similar_images`, which sounds like exactly
# what this project wants. It is not usable here: those entries carry no source
# page. Their `link` points back into `yandex.com/images/search`, and the
# original page the image came from is never given. A candidate with no
# permalink cannot be evidence of a social media post no matter how well its
# face scores, so harvesting them would spend the candidate budget on leads
# that could never be anchored.
#
# `image_results` entries do carry a real `link` and `source`, which is what
# makes them citable.
RESULT_SETS: tuple[tuple[str, Source], ...] = (("image_results", Source.YANDEX_IMAGE),)


class YandexReverseImageProvider:
    """Reverse image search over Yandex's index."""

    name = "serpapi_yandex_images"

    def __init__(self, api_key: str, endpoint: str = SERPAPI_ENDPOINT) -> None:
        if not api_key:
            raise SearchNotConfigured("SERPAPI_KEY is not set; see .env.example")
        self._api_key = api_key
        self._endpoint = endpoint

    def search(self, image_url: str) -> SearchResponse:
        """Query Yandex and normalise its result sets into candidates."""
        params = {
            "engine": SERPAPI_YANDEX_ENGINE,
            "url": image_url,
            "api_key": self._api_key,
        }
        try:
            response = requests.get(
                self._endpoint, params=params, timeout=DOWNLOAD_TIMEOUT_SECONDS
            )
        except requests.RequestException as exc:
            raise SearchProviderError(f"Could not reach SerpApi: {exc}") from exc

        if response.status_code != 200:
            raise SearchProviderError(
                f"SerpApi returned HTTP {response.status_code}",
                body=response.text[:200],
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise SearchProviderError(f"SerpApi returned invalid JSON: {exc}") from exc

        if "error" in payload:
            # Yandex reports "nothing found" as an error rather than an empty
            # list. For an obscure subject that is an ordinary outcome, not a
            # failure, so it yields no candidates instead of aborting the run.
            return SearchResponse(
                provider=self.name,
                query_image_url=image_url,
                retrieved_at=utc_now(),
                candidates=[],
                raw=payload,
            )

        return SearchResponse(
            provider=self.name,
            query_image_url=image_url,
            retrieved_at=utc_now(),
            candidates=parse_yandex_response(payload),
            raw=payload,
            identity=extract_identity(payload),
        )


def parse_yandex_response(
    payload: dict[str, Any], start_position: int = 1
) -> list[Candidate]:
    """Pull candidates out of every result set in a Yandex response.

    Interleaved for the same reason as the Lens harvest: one array must not be
    able to consume the whole candidate budget and starve the others.
    """
    queues = [
        (origin, entries)
        for key, origin in RESULT_SETS
        if isinstance(entries := payload.get(key), list) and entries
    ]
    if not queues:
        return []

    candidates: list[Candidate] = []
    seen: set[str] = set()
    longest = max(len(entries) for _, entries in queues)

    for index in range(longest):
        for origin, entries in queues:
            if index >= len(entries):
                continue
            if len(candidates) >= MAX_CANDIDATES:
                return candidates

            candidate = _to_candidate(
                entries[index], origin, start_position + len(candidates)
            )
            if candidate is None or candidate.page_url in seen:
                continue
            seen.add(candidate.page_url)
            candidates.append(candidate)

    return candidates


def _to_candidate(entry: Any, origin: Source, position: int) -> Candidate | None:
    """Convert one Yandex result into a candidate, or None if unusable."""
    if not isinstance(entry, dict):
        return None

    page_url = entry.get("link")
    if not page_url:
        return None

    image_urls = _image_chain(entry)
    if not image_urls:
        return None

    return Candidate(
        position=position,
        title=str(entry.get("title") or entry.get("snippet") or ""),
        page_url=str(page_url),
        image_urls=image_urls,
        source=str(entry.get("source") or ""),
        origin=origin,
    )


def _image_chain(entry: dict[str, Any]) -> tuple[str, ...]:
    """Ordered image URLs to try, full resolution first then the thumbnail."""
    chain: list[str] = []
    for value in (_nested(entry, "original_image"), _nested(entry, "thumbnail")):
        if value and value not in chain:
            chain.append(value)
    return tuple(chain)


def _nested(entry: dict[str, Any], key: str) -> str | None:
    """Read `entry[key].link`, which is how Yandex nests its image URLs."""
    value = entry.get(key)
    if isinstance(value, dict):
        link = value.get("link")
        return link if isinstance(link, str) and link else None
    return value if isinstance(value, str) and value else None


def extract_identity(payload: dict[str, Any]) -> Identity | None:
    """Read a subject name from Yandex's knowledge graph, when it has one.

    Present only for recognisable entities, so it is a bonus rather than
    something to rely on. An ordinary person will have none, which is expected.
    """
    graph = payload.get("knowledge_graph")
    if not isinstance(graph, dict):
        return None

    title = graph.get("title")
    if isinstance(title, str) and title.strip():
        return Identity(
            name=title.strip(), origin="yandex_knowledge_graph", confidence="medium"
        )
    return None
