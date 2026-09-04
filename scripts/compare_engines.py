"""Measure how well each search engine finds a given face.

The pipeline's reach depends almost entirely on who the subject is. A public
figure is found by anything; an ordinary person with a public account is the
hard case, and the engines differ sharply there. This script measures that
difference for one photograph instead of guessing at it.

It runs the visual engines, downloads every lead and scores it against the
probe face, then reports per engine: how many leads, how many faces verified,
and how many of those were social media posts.

    python scripts/compare_engines.py --image inputs/probe.jpg

Costs two SerpApi searches and no gas. Nothing is anchored and no evidence
folder is written — this measures reach, it does not produce a record.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FACE_MATCH_THRESHOLD, load_settings  # noqa: E402
from src.errors import PipelineError  # noqa: E402
from src.face import encode_primary_face  # noqa: E402
from src.search.filter import score_candidates  # noqa: E402
from src.search.imghost import public_url_for  # noqa: E402
from src.search.provider import Candidate  # noqa: E402
from src.search.serpapi_lens import SerpApiLensProvider  # noqa: E402
from src.search.yandex import YandexReverseImageProvider  # noqa: E402


@dataclass
class EngineReport:
    """What one engine achieved for this face."""

    engine: str
    leads: int
    verified: int
    social_verified: int
    best_score: float | None
    identified: str | None
    platforms: Counter[str]

    def line(self) -> str:
        best = f"{self.best_score:.4f}" if self.best_score is not None else "   -  "
        return (
            f"{self.engine:<22}{self.leads:<8}{self.verified:<11}"
            f"{self.social_verified:<9}{best}"
        )


def measure(
    engine_name: str,
    candidates: list[Candidate],
    reference: object,
    images_dir: Path,
    threshold: float,
    identified: str | None,
) -> EngineReport:
    """Score every lead from one engine and summarise the outcome."""
    scored = score_candidates(reference, candidates, images_dir, threshold=threshold)  # type: ignore[arg-type]
    verified = [item for item in scored if item.is_match]
    social = [item for item in verified if item.is_social_match]

    return EngineReport(
        engine=engine_name,
        leads=len(scored),
        verified=len(verified),
        social_verified=len(social),
        best_score=max((item.similarity or 0.0 for item in verified), default=None),
        identified=identified,
        platforms=Counter(item.platform.value for item in social),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "inputs" / "probe.jpg")
    parser.add_argument("--threshold", type=float, default=FACE_MATCH_THRESHOLD)
    args = parser.parse_args()

    if not args.image.exists():
        print(f"No such image: {args.image}")
        return 1

    settings = load_settings()
    if not settings.serpapi_key:
        print("SERPAPI_KEY is not set; see .env.example")
        return 1

    print(f"Probe: {args.image}")
    encoding = encode_primary_face(args.image)
    print(f"Face detected at confidence {encoding.confidence:.4f}\n")

    url, how = public_url_for(args.image, settings.github_raw_base)
    print(f"Hosted: {how}\n")

    reports: list[EngineReport] = []
    with tempfile.TemporaryDirectory() as workspace:
        for name, provider in (
            ("Google Lens", SerpApiLensProvider(settings.serpapi_key)),
            ("Yandex", YandexReverseImageProvider(settings.serpapi_key)),
        ):
            print(f"Querying {name}...")
            try:
                response = provider.search(url)
            except PipelineError as exc:
                print(f"  failed: {exc}")
                continue

            reports.append(
                measure(
                    name,
                    response.candidates,
                    encoding.embedding,
                    Path(workspace) / name.replace(" ", "_"),
                    args.threshold,
                    response.identity.name if response.identity else None,
                )
            )

    print(f"\n{'engine':<22}{'leads':<8}{'verified':<11}{'social':<9}best")
    print("-" * 58)
    for report in reports:
        print(report.line())

    print("\nIdentity resolved by:")
    for report in reports:
        answer = report.identified or "(none — no knowledge-graph entity)"
        print(f"  {report.engine:<16}{answer}")

    print("\nSocial platforms verified:")
    for report in reports:
        found = ", ".join(f"{k} x{v}" for k, v in report.platforms.most_common())
        print(f"  {report.engine:<16}{found or '(none)'}")

    total_social = sum(report.social_verified for report in reports)
    print()
    if total_social:
        print(f"{total_social} verified social post(s) found across both engines.")
    else:
        print(
            "No verified social post. For a subject with no public indexed\n"
            "photographs this is the correct answer, not a failure."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as error:
        print(f"error [{error.code}]: {error}", file=sys.stderr)
        raise SystemExit(1) from error
