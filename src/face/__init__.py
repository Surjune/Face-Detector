"""Face stage: detect a face in an image and reduce it to a comparable vector.

The pipeline uses this twice — once on the probe image, then again on every
candidate image the web search returns. Re-running recognition on the candidates
is what makes the search a *face* match rather than an image-hash lookup: it
matches a different photograph of the same person.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.face.detector import DetectedFace, detect_faces, detect_primary_face, load_image
from src.face.embed import (
    Embedding,
    cosine_similarity,
    embed_face,
    embed_faces,
    embedding_digest,
)

__all__ = [
    "DetectedFace",
    "Embedding",
    "FaceEncoding",
    "cosine_similarity",
    "embedding_digest",
    "encode_best_match",
    "encode_faces",
    "encode_primary_face",
    "load_image",
]


@dataclass(frozen=True)
class FaceEncoding:
    """A detected face together with its embedding."""

    box: tuple[float, float, float, float]
    confidence: float
    embedding: Embedding

    @property
    def digest(self) -> str:
        return embedding_digest(self.embedding)


def encode_primary_face(image_path: Path) -> FaceEncoding:
    """Detect and embed the largest face in an image file.

    Raises:
        ImageLoadError: the file could not be decoded.
        NoFaceFound: no face cleared the confidence floor.
    """
    image = load_image(image_path)
    face = detect_primary_face(image, origin=str(image_path))
    return FaceEncoding(box=face.box, confidence=face.confidence, embedding=embed_face(face))


def encode_faces(image_bytes: bytes) -> list[FaceEncoding]:
    """Detect and embed every face in raw image bytes.

    Used for candidate images pulled off the web, which routinely contain group
    shots — every face has to be scored, not just the largest.
    """
    image = load_image(image_bytes)
    faces = detect_faces(image)
    embeddings = embed_faces(faces)
    return [
        FaceEncoding(box=face.box, confidence=face.confidence, embedding=embedding)
        for face, embedding in zip(faces, embeddings)
    ]


def encode_best_match(
    reference: Embedding, image_bytes: bytes
) -> tuple[float, FaceEncoding] | None:
    """Score every face in an image against a reference, returning the best.

    Returns None when the image contains no detectable face.
    """
    encodings = encode_faces(image_bytes)
    if not encodings:
        return None
    scored = [(cosine_similarity(reference, item.embedding), item) for item in encodings]
    return max(scored, key=lambda pair: pair[0])
