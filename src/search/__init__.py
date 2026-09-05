"""Search stage: find social media posts showing the same person, and prove which do."""

from __future__ import annotations

from src.search.expand import missing_platforms, search_platform
from src.search.fetch import download_first_available, download_image
from src.search.filter import (
    CANDIDATE_IMAGE_DIRNAME,
    ScoredCandidate,
    Status,
    score_candidates,
)
from src.search.identity import IdentityVerdict, confirm_identity, derive_identity
from src.search.imghost import public_url_for, upload_to_catbox
from src.search.platforms import (
    SOCIAL_PLATFORMS,
    TARGET_PLATFORMS,
    Platform,
    classify,
    is_social,
)
from src.search.provider import (
    Candidate,
    Identity,
    SearchProvider,
    SearchResponse,
    Source,
    utc_now,
)
from src.search.replay import RAW_RESPONSE_FILENAME, ReplayProvider, record_response
from src.search.serpapi_lens import (
    SerpApiLensProvider,
    extract_identity,
    parse_lens_response,
    parse_visual_matches,
)
from src.search.yandex import (
    YandexReverseImageProvider,
    parse_yandex_response,
)
from src.search.youtube import search_videos

__all__ = [
    "CANDIDATE_IMAGE_DIRNAME",
    "Candidate",
    "Identity",
    "IdentityVerdict",
    "Platform",
    "RAW_RESPONSE_FILENAME",
    "ReplayProvider",
    "SOCIAL_PLATFORMS",
    "ScoredCandidate",
    "SearchProvider",
    "SearchResponse",
    "SerpApiLensProvider",
    "YandexReverseImageProvider",
    "Source",
    "Status",
    "TARGET_PLATFORMS",
    "classify",
    "confirm_identity",
    "derive_identity",
    "download_first_available",
    "download_image",
    "extract_identity",
    "is_social",
    "missing_platforms",
    "parse_lens_response",
    "parse_yandex_response",
    "parse_visual_matches",
    "public_url_for",
    "record_response",
    "score_candidates",
    "search_platform",
    "search_videos",
    "upload_to_catbox",
    "utc_now",
]
