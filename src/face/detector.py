"""Face detection and alignment.

Wraps MTCNN. The detector is loaded once per process: instantiating it downloads
and deserialises three networks, which is far too expensive to repeat per image.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.config import FACE_CROP_MARGIN, FACE_IMAGE_SIZE, MIN_FACE_CONFIDENCE
from src.errors import ImageLoadError, NoFaceFound

# CPU only. One image at a time does not justify a CUDA dependency, and the
# published requirements deliberately install the CPU-only torch wheel.
_DEVICE = torch.device("cpu")

_detector_lock = threading.Lock()
_detector: object | None = None


@dataclass(frozen=True)
class DetectedFace:
    """One aligned face crop, ready to embed."""

    box: tuple[float, float, float, float]
    confidence: float
    crop: torch.Tensor

    @property
    def area(self) -> float:
        """Pixel area of the detection box, used to pick the subject of a photo."""
        left, top, right, bottom = self.box
        return max(right - left, 0.0) * max(bottom - top, 0.0)


def _get_detector() -> object:
    """Return the process-wide MTCNN instance, constructing it on first use."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                from facenet_pytorch import MTCNN

                _detector = MTCNN(
                    image_size=FACE_IMAGE_SIZE,
                    margin=FACE_CROP_MARGIN,
                    keep_all=True,
                    device=_DEVICE,
                )
    return _detector


def load_image(source: Path | bytes) -> Image.Image:
    """Load an image from a path or raw bytes as RGB.

    Raises:
        ImageLoadError: the file is missing, truncated, or not a decodable image.
    """
    try:
        if isinstance(source, bytes):
            from io import BytesIO

            image = Image.open(BytesIO(source))
        else:
            image = Image.open(source)
        image.load()
        return image.convert("RGB")
    except FileNotFoundError as exc:
        raise ImageLoadError("Image file not found", path=str(source)) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        origin = "<bytes>" if isinstance(source, bytes) else str(source)
        raise ImageLoadError(f"Could not decode image: {exc}", path=origin) from exc


def detect_faces(image: Image.Image) -> list[DetectedFace]:
    """Detect every face in an image, ordered largest box first.

    Returns an empty list when the image contains no face above the confidence
    floor; callers decide whether that is fatal.
    """
    from facenet_pytorch import extract_face
    from facenet_pytorch.models.mtcnn import fixed_image_standardization

    detector = _get_detector()
    boxes, probabilities = detector.detect(image)  # type: ignore[attr-defined]
    if boxes is None:
        return []

    faces: list[DetectedFace] = []
    for box, probability in zip(boxes, probabilities):
        if box is None or probability is None or probability < MIN_FACE_CONFIDENCE:
            continue
        crop = extract_face(
            image,
            box,
            image_size=FACE_IMAGE_SIZE,
            margin=FACE_CROP_MARGIN,
        )
        faces.append(
            DetectedFace(
                box=tuple(float(value) for value in np.asarray(box, dtype=np.float64)),  # type: ignore[arg-type]
                confidence=float(probability),
                crop=fixed_image_standardization(crop),
            )
        )

    faces.sort(key=lambda face: face.area, reverse=True)
    return faces


def detect_primary_face(image: Image.Image, *, origin: str) -> DetectedFace:
    """Detect the largest face in an image.

    The largest box is the subject of a portrait; smaller boxes are bystanders.

    Raises:
        NoFaceFound: no detection cleared the confidence floor.
    """
    faces = detect_faces(image)
    if not faces:
        raise NoFaceFound(
            "No face detected above the confidence threshold",
            origin=origin,
            min_confidence=MIN_FACE_CONFIDENCE,
        )
    return faces[0]
