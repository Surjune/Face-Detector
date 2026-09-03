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
from src.face import cosine_similarity, encode_faces  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# An image is dropped when it does not agree with ANY other image in its folder
# at least this strongly. Commons is searched by text, so a file captioned "Meet
# Google CEO Sundar Pichai" frequently shows somebody else at the same event, and
# one such mislabel drags the genuine distribution down and corrupts the
# threshold derived from it.
#
# The test is the maximum, not the mean or median: a genuine photograph agrees
# strongly with at least one other photograph of the same person, while a
# stranger agrees with none. Averaging instead would fail on a small folder,
# where every genuine image is averaged against the very impostor being looked
# for. The floor sits in the empty band between the two measured distributions.
LABEL_COHESION_FLOOR = 0.45

# Below this, a folder has too few images for one to be checked against the rest.
MIN_IMAGES_FOR_COHESION = 3


def encode_dataset(root: Path) -> dict[str, list[tuple[Path, object]]]:
    """Encode one face per image, grouped by person directory.

    Images holding more than one face are skipped rather than resolved by
    picking the largest. A group photograph filed under someone's name will
    often show a different person largest, and one mislabelled face is enough
    to corrupt both distributions and the threshold derived from them.
    """
    people: dict[str, list[tuple[Path, object]]] = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        encodings: list[tuple[Path, object]] = []
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                faces = encode_faces(image_path.read_bytes())
            except PipelineError as exc:
                print(f"  skipped {image_path.name}: {exc}")
                continue
            if len(faces) != 1:
                print(f"  skipped {image_path.name}: {len(faces)} faces, label is ambiguous")
                continue
            encodings.append((image_path, faces[0]))
        if encodings:
            people[person_dir.name] = encodings
            print(f"  {person_dir.name}: {len(encodings)} usable image(s)")
    return people


def drop_mislabelled(people: dict[str, list[tuple[Path, object]]]) -> dict[str, list[tuple[Path, object]]]:
    """Remove images that do not agree with the rest of their own folder.

    A face that matches nobody else filed under the same name is far more
    likely to be a different person than a hard example, and keeping it would
    put an impostor score into the genuine distribution.
    """
    cleaned: dict[str, list[tuple[Path, object]]] = {}
    for name, encodings in people.items():
        if len(encodings) < MIN_IMAGES_FOR_COHESION:
            cleaned[name] = encodings
            if len(encodings) > 1:
                print(f"  {name}: too few images to check labels; keeping all")
            continue

        kept: list[tuple[Path, object]] = []
        for index, (path, encoding) in enumerate(encodings):
            others = [
                cosine_similarity(encoding.embedding, other.embedding)  # type: ignore[attr-defined]
                for position, (_, other) in enumerate(encodings)
                if position != index
            ]
            cohesion = max(others)
            if cohesion < LABEL_COHESION_FLOOR:
                print(
                    f"  dropped {name}/{path.name}: best agreement with its own "
                    f"folder is only {cohesion:+.4f}"
                )
            else:
                kept.append((path, encoding))
        cleaned[name] = kept
    return cleaned


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
    print()
    print("Checking labels...")
    people = {name: items for name, items in drop_mislabelled(people).items() if len(items) > 1}
    if len(people) < 2:
        print("Need at least two people with two agreeing images each.")
        return 1
    for name, items in people.items():
        print(f"  {name}: {len(items)} image(s) kept")

    genuine, impostor = score_pairs(people)
    print()
    describe("genuine", genuine)
    describe("impostor", impostor)
    recommend(genuine, impostor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
