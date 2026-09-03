"""Face embedding and comparison.

InceptionResnetV1 with vggface2 weights maps an aligned 160x160 crop to an
L2-normalised 512-d vector, so cosine similarity between two embeddings is a
plain dot product.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch

from src.config import EMBEDDING_DECIMALS, EMBEDDING_DIM
from src.face.detector import DetectedFace

Embedding = npt.NDArray[np.float32]

_DEVICE = torch.device("cpu")

_encoder_lock = threading.Lock()
_encoder: object | None = None


def _get_encoder() -> object:
    """Return the process-wide embedding network, constructing it on first use."""
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                from facenet_pytorch import InceptionResnetV1

                _encoder = InceptionResnetV1(pretrained="vggface2").eval().to(_DEVICE)
    return _encoder


def embed_faces(faces: Sequence[DetectedFace]) -> list[Embedding]:
    """Embed a batch of aligned crops in a single forward pass."""
    if not faces:
        return []

    encoder = _get_encoder()
    batch = torch.stack([face.crop for face in faces]).to(_DEVICE)
    with torch.no_grad():
        vectors = encoder(batch)  # type: ignore[operator]
    return [np.asarray(vector, dtype=np.float32) for vector in vectors.cpu().numpy()]


def embed_face(face: DetectedFace) -> Embedding:
    """Embed a single aligned crop."""
    return embed_faces([face])[0]


def cosine_similarity(left: Embedding, right: Embedding) -> float:
    """Cosine similarity of two embeddings, in [-1, 1].

    The network already returns unit vectors, but re-normalising costs nothing
    and keeps the function correct for any embedding handed to it.
    """
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def embedding_digest(embedding: Embedding) -> str:
    """Stable sha256 of an embedding, for inclusion in the on-chain receipt.

    Values are rounded before hashing: an unrounded float32 array serialises
    differently across platforms and would make the receipt unverifiable
    elsewhere. The digest identifies which face was searched without publishing
    the biometric vector itself.
    """
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(f"Expected a {EMBEDDING_DIM}-d embedding, got {embedding.shape}")
    rounded = np.round(embedding.astype(np.float64), EMBEDDING_DECIMALS)
    payload = ",".join(f"{value:.{EMBEDDING_DECIMALS}f}" for value in rounded)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
