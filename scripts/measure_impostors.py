"""Measure how high two *different* faces can score.

The match threshold is only as good as the impostor sample it was derived from.
An earlier calibration used sixteen impostor pairs drawn from three people and
concluded that different faces top out around 0.30. A real run then matched two
clearly different men at 0.65 and anchored the wrong person, because sixteen
pairs say nothing about the tail of a distribution.

This measures that tail properly: many distinct individuals, every cross-person
pair a known non-match, reported by percentile. The threshold belongs above the
impostor maximum, not above its average.

    python scripts/measure_impostors.py --people 40

Uses freely licensed Commons portraits and makes no face search of any kind.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DOWNLOAD_TIMEOUT_SECONDS, FACE_MATCH_THRESHOLD  # noqa: E402
from src.errors import PipelineError  # noqa: E402
from src.face import cosine_similarity, encode_faces  # noqa: E402

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "face-detector-calibration/1.0 (github.com/Surjune/Face-Detector)"}

# Searched to gather portraits of many different individuals. Breadth is the
# point: an impostor tail measured only on studio photographs of well-lit
# celebrities will be optimistic about the messy images a web search returns.
SEARCH_TERMS = (
    "portrait photograph man",
    "portrait photograph woman",
    "conference speaker portrait",
    "musician portrait",
    "athlete portrait",
    "scientist portrait",
    "actor headshot",
    "politician portrait",
)

ACCEPTABLE_LICENCES = ("cc0", "cc by", "cc-by", "public domain", "pd-", "attribution")

# A calibration image must show exactly one face, or the label is ambiguous.
REQUIRED_FACES = 1

THUMBNAIL_WIDTH = 600


def search_portraits(term: str, limit: int) -> list[dict[str, str]]:
    """Find freely licensed portrait files on Commons."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": term,
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime",
        "iiurlwidth": str(THUMBNAIL_WIDTH),
    }
    try:
        response = requests.get(
            COMMONS_API, params=params, headers=HEADERS, timeout=DOWNLOAD_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  search failed: {exc}")
        return []

    found: list[dict[str, str]] = []
    for page in response.json().get("query", {}).get("pages", {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        if not info.get("thumburl") or info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        licence = str(
            info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
        ).lower()
        if not any(token in licence for token in ACCEPTABLE_LICENCES):
            continue
        found.append({"title": str(page.get("title", "")), "url": info["thumburl"]})
    return found


def subject_key(title: str) -> str:
    """A crude identity key from the filename, used to avoid same-person pairs.

    Two files whose names share their leading words are very often the same
    person photographed twice. Treating them as different people would put a
    genuine pair into the impostor set and inflate the measured tail — the
    opposite of the error being corrected, but an error all the same.
    """
    cleaned = re.sub(r"^File:", "", title)
    cleaned = re.sub(r"\.(jpg|jpeg|png)$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^A-Za-z ]", " ", cleaned)
    words = [w for w in cleaned.split() if len(w) > 2]
    return " ".join(words[:2]).lower()


def collect(target: int) -> list[tuple[str, object]]:
    """Gather one embedded face per distinct individual."""
    people: dict[str, object] = {}

    for term in SEARCH_TERMS:
        if len(people) >= target:
            break
        print(f"searching: {term}")
        for entry in search_portraits(term, limit=target):
            if len(people) >= target:
                break
            key = subject_key(entry["title"])
            if not key or key in people:
                continue
            try:
                data = requests.get(
                    entry["url"], headers=HEADERS, timeout=DOWNLOAD_TIMEOUT_SECONDS
                ).content
                faces = encode_faces(data)
            except (requests.RequestException, PipelineError):
                continue
            if len(faces) != REQUIRED_FACES:
                continue
            people[key] = faces[0]
            print(f"  [{len(people):>3}] {key}")
            time.sleep(0.15)

    return list(people.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people", type=int, default=40)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    people = collect(args.people)
    if len(people) < 5:
        print(f"Only found {len(people)} usable faces; need more to measure a tail.")
        return 1

    scores = [
        cosine_similarity(a.embedding, b.embedding)  # type: ignore[attr-defined]
        for (_, a), (_, b) in itertools.combinations(people, 2)
    ]
    scores.sort()

    def percentile(fraction: float) -> float:
        return scores[min(int(len(scores) * fraction), len(scores) - 1)]

    print(f"\n{len(people)} distinct people -> {len(scores)} impostor pairs\n")
    print(f"  mean            {statistics.fmean(scores):+.4f}")
    print(f"  median          {percentile(0.50):+.4f}")
    print(f"  95th percentile {percentile(0.95):+.4f}")
    print(f"  99th percentile {percentile(0.99):+.4f}")
    print(f"  maximum         {max(scores):+.4f}")

    above = sum(1 for s in scores if s >= FACE_MATCH_THRESHOLD)
    print(
        f"\n  pairs at or above the configured {FACE_MATCH_THRESHOLD} threshold: "
        f"{above} ({100 * above / len(scores):.2f}%)"
    )
    print(
        "\n  Every one of those is two different people the pipeline would call\n"
        "  a match. The threshold has to sit above the impostor maximum."
    )

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "people": len(people),
                    "pairs": len(scores),
                    "mean": statistics.fmean(scores),
                    "p95": percentile(0.95),
                    "p99": percentile(0.99),
                    "max": max(scores),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
