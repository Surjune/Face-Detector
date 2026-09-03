"""Search stage: find web pages showing the same person, and prove which ones do."""

from __future__ import annotations

from src.search.fetch import download_image
from src.search.filter import (
    CANDIDATE_IMAGE_DIRNAME,
    ScoredCandidate,
    Status,
    score_candidates,
)
from src.search.imghost import public_url_for, upload_to_catbox
from src.search.provider import Candidate, SearchProvider, SearchResponse, utc_now
from src.search.replay import RAW_RESPONSE_FILENAME, ReplayProvider, record_response
from src.search.serpapi_lens import SerpApiLensProvider, parse_visual_matches

__all__ = [
    "CANDIDATE_IMAGE_DIRNAME",
    "Candidate",
    "RAW_RESPONSE_FILENAME",
    "ReplayProvider",
    "ScoredCandidate",
    "SearchProvider",
    "SearchResponse",
    "SerpApiLensProvider",
    "Status",
    "download_image",
    "parse_visual_matches",
    "public_url_for",
    "record_response",
    "score_candidates",
    "upload_to_catbox",
    "utc_now",
]
