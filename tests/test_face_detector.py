"""Image loading and detection bookkeeping."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.config import EMBEDDING_DIM, FACE_IMAGE_SIZE
from src.errors import ImageLoadError
from src.face.detector import DetectedFace, load_image


def png_bytes(mode: str = "RGB", size: tuple[int, int] = (32, 24)) -> bytes:
    buffer = BytesIO()
    Image.new(mode, size, color=128 if mode == "L" else (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def blank_crop() -> torch.Tensor:
    return torch.zeros(3, FACE_IMAGE_SIZE, FACE_IMAGE_SIZE)


class TestLoadImage:
    def test_loads_from_bytes(self) -> None:
        image = load_image(png_bytes())
        assert image.size == (32, 24)

    def test_loads_from_path(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.png"
        path.write_bytes(png_bytes())
        assert load_image(path).size == (32, 24)

    def test_converts_greyscale_to_rgb(self) -> None:
        """Downloaded candidates are not reliably RGB, but the network requires it."""
        assert load_image(png_bytes(mode="L")).mode == "RGB"

    def test_converts_rgba_to_rgb(self) -> None:
        assert load_image(png_bytes(mode="RGBA")).mode == "RGB"

    def test_missing_file_raises_typed_error(self, tmp_path: Path) -> None:
        with pytest.raises(ImageLoadError) as excinfo:
            load_image(tmp_path / "absent.png")
        assert excinfo.value.code == "image_load_failed"

    def test_undecodable_bytes_raise_typed_error(self) -> None:
        """A candidate URL can serve HTML or a truncated file under an image type."""
        with pytest.raises(ImageLoadError):
            load_image(b"<!doctype html><html>not an image</html>")

    def test_truncated_image_raises_typed_error(self) -> None:
        with pytest.raises(ImageLoadError):
            load_image(png_bytes()[:20])


class TestDetectedFace:
    def test_area_is_box_area(self) -> None:
        face = DetectedFace(box=(10.0, 20.0, 40.0, 60.0), confidence=0.99, crop=blank_crop())
        assert face.area == pytest.approx(30.0 * 40.0)

    def test_inverted_box_has_no_negative_area(self) -> None:
        """A degenerate box must not sort ahead of real detections."""
        face = DetectedFace(box=(40.0, 60.0, 10.0, 20.0), confidence=0.99, crop=blank_crop())
        assert face.area == 0.0


@pytest.mark.model
class TestPretrainedEncoder:
    def test_embedding_is_unit_length(self) -> None:
        """The receipt's cosine maths assumes the network returns unit vectors."""
        from src.face.embed import embed_faces

        face = DetectedFace(box=(0.0, 0.0, 1.0, 1.0), confidence=1.0, crop=torch.rand(3, 160, 160))
        embedding = embed_faces([face])[0]
        assert embedding.shape == (EMBEDDING_DIM,)
        assert float((embedding**2).sum()) == pytest.approx(1.0, abs=1e-4)
