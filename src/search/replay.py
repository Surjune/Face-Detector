"""Replay a recorded search response.

Every live run writes the provider's untouched payload into its evidence folder.
Replaying that file re-runs the pipeline over the exact leads the live search
returned, which is how someone without an API key reproduces a published run.

This is a recording of a real search, and is labelled as such wherever it is
reported. It is not a substitute for one: `--replay` never presents itself as a
live query.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.errors import SearchNotConfigured, SearchProviderError
from src.search.provider import SearchResponse
from src.search.serpapi_lens import extract_identity, parse_lens_response

RAW_RESPONSE_FILENAME = "search_response.json"


class ReplayProvider:
    """Serves a previously recorded response instead of querying the web."""

    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / RAW_RESPONSE_FILENAME
        if not self._path.exists():
            raise SearchNotConfigured(
                f"No recorded search in {run_dir}", expected=str(self._path)
            )
        self._record = _read_record(self._path)
        self.name = f"{self._record['provider']} (replayed)"

    def search(self, image_url: str) -> SearchResponse:
        """Return the recorded response.

        `image_url` is accepted for interface compatibility and ignored: the
        recording is tied to the URL that was queried at the time, which is
        preserved in the response rather than overwritten.
        """
        payload = self._record["raw"]
        return SearchResponse(
            provider=self.name,
            query_image_url=str(self._record["query_image_url"]),
            retrieved_at=str(self._record["retrieved_at"]),
            candidates=parse_lens_response(payload),
            raw=payload,
            identity=extract_identity(payload),
        )


def record_response(run_dir: Path, response: SearchResponse) -> Path:
    """Write a provider response so the run can be replayed later."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RAW_RESPONSE_FILENAME
    record = {
        "provider": response.provider,
        "query_image_url": response.query_image_url,
        "retrieved_at": response.retrieved_at,
        "raw": response.raw,
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _read_record(path: Path) -> dict[str, Any]:
    try:
        record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SearchProviderError(f"Could not read {path.name}: {exc}") from exc

    missing = {"provider", "query_image_url", "retrieved_at", "raw"} - set(record)
    if missing:
        raise SearchProviderError(
            f"{path.name} is missing fields", missing=sorted(missing)
        )
    return record
