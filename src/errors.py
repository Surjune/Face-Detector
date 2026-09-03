"""Typed exceptions.

Every failure the pipeline can hit surfaces as one of these, carrying a stable
machine-readable ``code`` alongside a human-readable message. Nothing in the
pipeline falls back to placeholder data on failure: an invented match anchored on
a blockchain would be worse than no result at all.
"""

from __future__ import annotations

from typing import Any


class PipelineError(Exception):
    """Base class for every error raised by this project."""

    code = "pipeline_error"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if not self.context:
            return self.message
        detail = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))
        return f"{self.message} ({detail})"


# --- Face stage ----------------------------------------------------------


class FaceError(PipelineError):
    code = "face_error"


class ImageLoadError(FaceError):
    code = "image_load_failed"


class NoFaceFound(FaceError):
    code = "no_face_found"


# --- Search stage --------------------------------------------------------


class SearchError(PipelineError):
    code = "search_error"


class SearchNotConfigured(SearchError):
    """A provider was selected but its credentials or inputs are missing."""

    code = "search_not_configured"


class SearchProviderError(SearchError):
    """The provider was reachable but returned an error or unusable payload."""

    code = "search_provider_failed"


class ImageHostError(SearchError):
    code = "image_host_failed"


class DownloadError(SearchError):
    code = "download_failed"


class NoMatchFound(SearchError):
    """The search ran and returned candidates, but none cleared the threshold."""

    code = "no_match_found"


# --- Chain stage ---------------------------------------------------------


class ChainError(PipelineError):
    code = "chain_error"


class ChainNotConfigured(ChainError):
    code = "chain_not_configured"


class ContractCompileError(ChainError):
    code = "contract_compile_failed"


class AnchorFailed(ChainError):
    code = "anchor_failed"


class NotAnchored(ChainError):
    """The digest is absent from the registry — unknown or tampered data."""

    code = "not_anchored"


# --- Evidence ------------------------------------------------------------


class EvidenceError(PipelineError):
    code = "evidence_error"
