"""Re-running face recognition over the search results.

This is the step that makes the pipeline a face search rather than an
image-hash lookup. A reverse image search finds pages showing copies of the
*same file*; embedding every candidate and scoring it against the probe finds a
*different photograph of the same person*, and rejects the visually similar
pictures of other people that a search returns alongside.

Every candidate is kept with its score, including the rejected ones. That
record is the evidence the search actually ran.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

from src.chain.canonical import sha256_hex
from src.config import DOWNLOAD_WORKERS, FACE_MATCH_THRESHOLD
from src.errors import DownloadError, PipelineError
from src.face import Embedding, encode_best_match
from src.search.fetch import download_first_available
from src.search.platforms import Platform
from src.search.provider import Candidate

Status = Literal["match", "below_threshold", "no_face", "unreachable"]

# What one download attempt produced: the image bytes and the URL that served
# them, or the error explaining why every URL failed. The failure is carried
# rather than raised so one dead link cannot abort a batch.
Fetched = Union[tuple[bytes, str], DownloadError]

CANDIDATE_IMAGE_DIRNAME = "candidates"


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate after the face check."""

    candidate: Candidate
    status: Status
    similarity: float | None = None
    image_sha256: str | None = None
    image_path: Path | None = None
    image_url_used: str | None = None
    detail: str = ""

    @property
    def is_match(self) -> bool:
        return self.status == "match"

    @property
    def platform(self) -> Platform:
        return self.candidate.platform

    @property
    def is_social_match(self) -> bool:
        """A verified match that is also on a social platform."""
        return self.is_match and self.candidate.is_social


def score_candidates(
    reference: Embedding,
    candidates: list[Candidate],
    images_dir: Path,
    *,
    threshold: float = FACE_MATCH_THRESHOLD,
) -> list[ScoredCandidate]:
    """Download, embed and score every candidate.

    Downloading runs concurrently and scoring runs sequentially after it. About
    70% of the per-candidate cost is waiting on the network, so fetching one at
    a time left the run idle for most of its duration. Face embedding is CPU
    work on a shared model and stays serial.

    Ordered social matches first, then by score. The brief asks for a social
    media post specifically, and an exact-file copy on an encyclopaedia will
    otherwise outrank every genuine social post on raw similarity.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    fetched = _download_all(candidates)

    scored = [
        _score_one(reference, candidate, fetched[index], images_dir, threshold)
        for index, candidate in enumerate(candidates)
    ]

    return sorted(
        scored,
        key=lambda item: (
            item.is_social_match,
            item.is_match,
            item.similarity if item.similarity is not None else -2.0,
        ),
        reverse=True,
    )


def _download_all(candidates: list[Candidate]) -> list[Fetched]:
    """Fetch every candidate image concurrently, preserving input order.

    Results are placed back at their original index rather than collected as
    they complete, so the candidate list — and therefore the numbered image
    files and the receipt built from them — does not depend on which download
    happened to finish first.
    """
    results: list[Fetched] = [None] * len(candidates)  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(
                download_first_available,
                candidate.image_urls,
                referer=candidate.page_url,
            ): index
            for index, candidate in enumerate(candidates)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except DownloadError as exc:
                results[index] = exc

    return results


def _score_one(
    reference: Embedding,
    candidate: Candidate,
    fetched: Fetched,
    images_dir: Path,
    threshold: float,
) -> ScoredCandidate:
    """Score a single candidate, converting any failure into a status.

    One unreachable or faceless candidate must never abort the run — a search
    result set routinely contains dead links and pictures of scenery.
    """
    if isinstance(fetched, DownloadError):
        return ScoredCandidate(
            candidate=candidate, status="unreachable", detail=fetched.message
        )

    image_bytes, used_url = fetched
    digest = sha256_hex(image_bytes)
    path = images_dir / f"{candidate.position:02d}_{digest[:12]}{_suffix_for(used_url)}"
    path.write_bytes(image_bytes)

    try:
        best = encode_best_match(reference, image_bytes)
    except PipelineError as exc:
        return ScoredCandidate(
            candidate=candidate,
            status="no_face",
            image_sha256=digest,
            image_path=path,
            image_url_used=used_url,
            detail=exc.message,
        )

    if best is None:
        return ScoredCandidate(
            candidate=candidate,
            status="no_face",
            image_sha256=digest,
            image_path=path,
            image_url_used=used_url,
            detail="no face detected in the candidate image",
        )

    similarity, _ = best
    return ScoredCandidate(
        candidate=candidate,
        status="match" if similarity >= threshold else "below_threshold",
        similarity=similarity,
        image_sha256=digest,
        image_path=path,
        image_url_used=used_url,
    )


def _suffix_for(url: str) -> str:
    """Best-effort file extension, for images a human may want to open."""
    tail = url.split("?")[0].rsplit(".", 1)
    if len(tail) == 2 and 1 <= len(tail[1]) <= 4 and tail[1].isalnum():
        return "." + tail[1].lower()
    return ".img"
