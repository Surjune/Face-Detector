"""Derive the face-match threshold from labelled example images.

`FACE_MATCH_THRESHOLD` in src/config.py must be a measured value, not a guess.
Point this at a directory holding one sub-directory per person:

    dataset/
      alice/  a1.jpg a2.jpg a3.jpg
      bob/    b1.jpg b2.jpg

Every within-folder pair is a genuine match, every cross-folder pair an impostor.
The script reports both distributions and recommends a cut-off.

    python scripts/calibrate_threshold.py --dataset dataset/
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FACE_MATCH_THRESHOLD  # noqa: E402
from src.errors import PipelineError  # noqa: E402
from src.face import cosine_similarity, encode_primary_face  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def encode_dataset(root: Path) -> dict[str, list[tuple[Path, object]]]:
    """Encode the largest face in every image, grouped by person directory."""
    people: dict[str, list[tuple[Path, object]]] = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        encodings: list[tuple[Path, object]] = []
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                encodings.append((image_path, encode_primary_face(image_path)))
            except PipelineError as exc:
                print(f"  skipped {image_path.name}: {exc}")
        if encodings:
            people[person_dir.name] = encodings
            print(f"  {person_dir.name}: {len(encodings)} face(s)")
    return people


def score_pairs(people: dict[str, list[tuple[Path, object]]]) -> tuple[list[float], list[float]]:
    """Return (genuine scores, impostor scores)."""
    genuine: list[float] = []
    impostor: list[float] = []

    for encodings in people.values():
        for (_, left), (_, right) in itertools.combinations(encodings, 2):
            genuine.append(cosine_similarity(left.embedding, right.embedding))  # type: ignore[attr-defined]

    for left_name, right_name in itertools.combinations(people, 2):
        for _, left in people[left_name]:
            for _, right in people[right_name]:
                impostor.append(cosine_similarity(left.embedding, right.embedding))  # type: ignore[attr-defined]

    return genuine, impostor


def describe(label: str, scores: list[float]) -> None:
    if not scores:
        print(f"{label:<10} none")
        return
    mean = statistics.fmean(scores)
    print(
        f"{label:<10} n={len(scores):<4} min={min(scores):+.4f} "
        f"mean={mean:+.4f} max={max(scores):+.4f}"
    )


def recommend(genuine: list[float], impostor: list[float]) -> None:
    if not genuine or not impostor:
        print("\nNeed at least two people with two images each to recommend a threshold.")
        return

    lowest_genuine = min(genuine)
    highest_impostor = max(impostor)

    if lowest_genuine > highest_impostor:
        midpoint = (lowest_genuine + highest_impostor) / 2
        print(
            f"\nDistributions separate cleanly by {lowest_genuine - highest_impostor:.4f}.\n"
            f"Midpoint threshold: {midpoint:.4f}"
        )
    else:
        print(
            "\nDistributions overlap: no threshold separates them perfectly.\n"
            f"Worst genuine {lowest_genuine:.4f} sits below best impostor {highest_impostor:.4f}.\n"
            "Prefer a value above the impostor maximum — a false match anchored\n"
            "on-chain is worse than a missed one."
        )
        print(f"Impostor-safe threshold: {highest_impostor:.4f}")

    print(f"Currently configured FACE_MATCH_THRESHOLD: {FACE_MATCH_THRESHOLD}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="directory containing one sub-directory of images per person",
    )
    args = parser.parse_args()

    if not args.dataset.is_dir():
        print(f"Not a directory: {args.dataset}")
        return 1

    print(f"Encoding {args.dataset}...")
    people = encode_dataset(args.dataset)
    if len(people) < 2:
        print("Need at least two person directories.")
        return 1

    genuine, impostor = score_pairs(people)
    print()
    describe("genuine", genuine)
    describe("impostor", impostor)
    recommend(genuine, impostor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
