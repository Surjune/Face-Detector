"""Comparison and digest maths for the face stage.

None of these touch the network or the pretrained weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import EMBEDDING_DIM
from src.face.embed import cosine_similarity, embedding_digest


def unit_vector(seed: int) -> np.ndarray:
    """A deterministic pseudo-random unit vector of the model's embedding size."""
    generator = np.random.default_rng(seed)
    vector = generator.normal(size=EMBEDDING_DIM).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        vector = unit_vector(1)
        assert cosine_similarity(vector, vector) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_score_minus_one(self) -> None:
        vector = unit_vector(2)
        assert cosine_similarity(vector, -vector) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_vectors_score_zero(self) -> None:
        left = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        right = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        left[0] = 1.0
        right[1] = 1.0
        assert cosine_similarity(left, right) == pytest.approx(0.0, abs=1e-6)

    def test_is_symmetric(self) -> None:
        left, right = unit_vector(3), unit_vector(4)
        assert cosine_similarity(left, right) == pytest.approx(cosine_similarity(right, left))

    def test_ignores_magnitude(self) -> None:
        """Scaling an input must not change the score; the function normalises."""
        left, right = unit_vector(5), unit_vector(6)
        assert cosine_similarity(left * 7.0, right) == pytest.approx(
            cosine_similarity(left, right), abs=1e-6
        )

    def test_zero_vector_does_not_divide_by_zero(self) -> None:
        zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        assert cosine_similarity(zero, unit_vector(7)) == 0.0


class TestEmbeddingDigest:
    def test_is_deterministic(self) -> None:
        vector = unit_vector(8)
        assert embedding_digest(vector) == embedding_digest(vector.copy())

    def test_differs_between_faces(self) -> None:
        assert embedding_digest(unit_vector(9)) != embedding_digest(unit_vector(10))

    def test_is_stable_under_float_noise_below_the_rounding_floor(self) -> None:
        """A difference too small to survive rounding must not change the digest.

        This is what lets a receipt verify on a machine other than the one that
        produced it.
        """
        vector = unit_vector(11)
        nudged = (vector + 1e-9).astype(np.float32)
        assert embedding_digest(vector) == embedding_digest(nudged)

    def test_rejects_wrong_dimensions(self) -> None:
        with pytest.raises(ValueError, match=str(EMBEDDING_DIM)):
            embedding_digest(np.zeros(8, dtype=np.float32))
