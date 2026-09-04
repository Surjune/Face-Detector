"""The three commands, end to end.

The face encoder, the downloader and the candidate scorer are substituted, and
the chain is a real EVM running in this process. Everything between — evidence
layout, receipt construction, digest, anchoring, verification and the tamper
demonstration — is the genuine code path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

import src.pipeline as pipeline
import src.search.filter as filter_module
from src.chain.canonical import receipt_digest
from src.config import EMBEDDING_DIM
from src.errors import DownloadError
from src.face import FaceEncoding
from src.search.provider import SearchResponse
from src.search.replay import record_response
from src.search.serpapi_lens import parse_visual_matches
from tests.test_search import lens_payload, non_social_payload

runner = CliRunner()

CANDIDATE_SCORES = {0: 0.88, 1: 0.31, 2: 0.62}


@pytest.fixture
def probe_image(tmp_path: Path) -> Path:
    """Stand-in for a photograph; the encoder is stubbed, so bytes are enough."""
    path = tmp_path / "probe.jpg"
    path.write_bytes(b"probe-image-bytes")
    return path


@pytest.fixture
def recorded_search(tmp_path: Path) -> Path:
    """A directory holding a recorded provider response to replay."""
    source = tmp_path / "recorded"
    source.mkdir()
    payload = lens_payload(3)
    record_response(
        source,
        SearchResponse(
            provider="serpapi_google_lens",
            query_image_url="https://example.com/probe.jpg",
            retrieved_at="2026-09-03T15:40:00Z",
            candidates=parse_visual_matches(payload),
            raw=payload,
        ),
    )
    return source


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the model and the network; keep everything else real."""
    embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    embedding[0] = 1.0

    monkeypatch.setattr(
        pipeline,
        "encode_primary_face",
        lambda path: FaceEncoding(box=(10.0, 20.0, 90.0, 120.0), confidence=0.997, embedding=embedding),
    )

    def fake_download(urls: Any, *, referer: str | None = None) -> tuple[bytes, str]:
        url = urls[0]
        index = int(url.rsplit("/", 1)[-1].split(".")[0])
        return f"image-{index}".encode(), url

    def fake_best_match(reference: np.ndarray, image_bytes: bytes) -> Any:
        index = int(image_bytes.decode().rsplit("-", 1)[-1])
        return CANDIDATE_SCORES[index], object()

    monkeypatch.setattr(filter_module, "download_first_available", fake_download)
    monkeypatch.setattr(filter_module, "encode_best_match", fake_best_match)


def do_run(probe_image: Path, recorded_search: Path, out: Path) -> Any:
    return runner.invoke(
        pipeline.app,
        [
            "run",
            "--image", str(probe_image),
            "--chain", "local",
            "--replay", str(recorded_search),
            "--out", str(out),
        ],
        catch_exceptions=False,
    )


def latest_run_dir(out: Path) -> Path:
    return sorted(out.glob("run_*"))[-1]


def _recorded(directory: Path, payload: dict[str, Any]) -> Path:
    """Write a recorded search response for the pipeline to replay."""
    directory.mkdir(parents=True, exist_ok=True)
    record_response(
        directory,
        SearchResponse(
            provider="serpapi_google_lens",
            query_image_url="https://example.com/probe.jpg",
            retrieved_at="2026-09-03T15:40:00Z",
            candidates=parse_visual_matches(payload),
            raw=payload,
        ),
    )
    return directory


def _run(
    probe_image: Path, source: Path, out: Path, extra: list[str] | None = None
) -> Any:
    return runner.invoke(
        pipeline.app,
        [
            "run",
            "--image", str(probe_image),
            "--chain", "local",
            "--replay", str(source),
            "--out", str(out),
            *(extra or []),
        ],
    )


@pytest.mark.usefixtures("stubs")
class TestRun:
    def test_completes_and_reports_each_stage(
        self, probe_image: Path, recorded_search: Path, tmp_path: Path
    ) -> None:
        result = do_run(probe_image, recorded_search, tmp_path / "out")
        assert result.exit_code == 0, result.output
        assert "[1/3] FACE DETECTION & ENCODING" in result.output
        assert "[2/3] WEB / SOCIAL MEDIA SEARCH" in result.output
        assert "[3/3] BLOCKCHAIN ATTESTATION & RECORDING" in result.output

    def test_runs_the_verification_routine_after_anchoring(
        self, probe_image: Path, recorded_search: Path, tmp_path: Path
    ) -> None:
        """The round trip is proven at the point the receipt is created."""
        result = do_run(probe_image, recorded_search, tmp_path / "out")
        assert "[VERIFICATION ROUTINE]" in result.output
        assert "On-chain digest matches computed payload digest" in result.output
        assert "Tamper-evidence verified: MATCH" in result.output

    def test_prints_every_candidate_score_including_rejects(
        self, probe_image: Path, recorded_search: Path, tmp_path: Path
    ) -> None:
        """A result set with no rejects shown could not be told from a fake one."""
        result = do_run(probe_image, recorded_search, tmp_path / "out")
        assert "below_threshold" in result.output
        assert result.output.count("match") >= 2

    def test_writes_a_complete_evidence_folder(
        self, probe_image: Path, recorded_search: Path, tmp_path: Path
    ) -> None:
        do_run(probe_image, recorded_search, tmp_path / "out")
        run_dir = latest_run_dir(tmp_path / "out")

        for name in ("receipt.json", "candidates.json", "anchor.json", "report.html"):
            assert (run_dir / name).exists(), f"{name} missing"
        assert (run_dir / "input.jpg").read_bytes() == b"probe-image-bytes"
        assert len(list((run_dir / "candidates").iterdir())) == 3

    def test_anchors_the_digest_of_the_stored_receipt(
        self, probe_image: Path, recorded_search: Path, tmp_path: Path
    ) -> None:
        """The anchored value must be derivable from what was written to disk."""
        from src import evidence

        do_run(probe_image, recorded_search, tmp_path / "out")
        run_dir = latest_run_dir(tmp_path / "out")

        assert receipt_digest(evidence.read_receipt(run_dir)) == evidence.read_anchor(run_dir).digest

    def test_selects_the_highest_scoring_candidate(
        self, probe_image: Path, recorded_search: Path, tmp_path: Path
    ) -> None:
        from src import evidence

        do_run(probe_image, recorded_search, tmp_path / "out")
        receipt = evidence.read_receipt(latest_run_dir(tmp_path / "out"))
        assert receipt.match.similarity == pytest.approx(max(CANDIDATE_SCORES.values()))

    def test_records_every_candidate_including_rejects(
        self, probe_image: Path, recorded_search: Path, tmp_path: Path
    ) -> None:
        do_run(probe_image, recorded_search, tmp_path / "out")
        run_dir = latest_run_dir(tmp_path / "out")
        candidates = json.loads((run_dir / "candidates.json").read_text(encoding="utf-8"))

        assert len(candidates) == 3
        assert {item["status"] for item in candidates} == {"match", "below_threshold"}

    def test_labels_a_replayed_search_as_a_replay(
        self, probe_image: Path, recorded_search: Path, tmp_path: Path
    ) -> None:
        """A recording must never be presented as a live query."""
        from src import evidence

        do_run(probe_image, recorded_search, tmp_path / "out")
        receipt = evidence.read_receipt(latest_run_dir(tmp_path / "out"))
        assert "replay" in receipt.search.provider

    def test_fails_when_no_candidate_clears_the_threshold(
        self, probe_image: Path, recorded_search: Path, tmp_path: Path
    ) -> None:
        """No match must be an explicit failure, never an invented result."""
        result = runner.invoke(
            pipeline.app,
            [
                "run",
                "--image", str(probe_image),
                "--chain", "local",
                "--replay", str(recorded_search),
                "--out", str(tmp_path / "out"),
                "--threshold", "0.99",
            ],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, SystemExit) or result.exception is not None

    def test_rejects_a_missing_image(self, recorded_search: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            pipeline.app,
            [
                "run",
                "--image", str(tmp_path / "absent.jpg"),
                "--chain", "local",
                "--replay", str(recorded_search),
                "--out", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0


@pytest.mark.usefixtures("stubs")
class TestSocialSelection:
    """The brief asks for a social media post, so one is what gets anchored."""

    def test_anchors_a_social_post_over_a_higher_scoring_page(
        self, probe_image: Path, tmp_path: Path
    ) -> None:
        """The defect this whole change exists to fix.

        An exact-file copy on an encyclopaedia scores 1.0000 and would win on
        raw similarity, burying a genuine social post that scores lower.
        """
        from src import evidence

        payload = {
            "visual_matches": [
                {
                    "title": "Subject - Wikipedia",
                    "link": "https://en.wikipedia.org/wiki/Subject",
                    "image": "https://cdn.example.com/0.jpg",
                },
                {
                    # Scores lower than the Wikipedia copy but still clears the
                    # threshold: exactly the case that used to be buried.
                    "title": "A post",
                    "link": "https://www.instagram.com/p/POST1/",
                    "image": "https://cdn.example.com/2.jpg",
                },
            ]
        }
        source = _recorded(tmp_path / "recorded", payload)
        result = _run(probe_image, source, tmp_path / "out")

        assert result.exit_code == 0, result.output
        receipt = evidence.read_receipt(latest_run_dir(tmp_path / "out"))
        assert "instagram.com" in receipt.match.post_url
        assert "Social media post found" in result.output

    def test_refuses_to_anchor_when_nothing_social_matched(
        self, probe_image: Path, tmp_path: Path
    ) -> None:
        """Silently anchoring a non-social page is the bug, not the fallback."""
        source = _recorded(tmp_path / "recorded", non_social_payload(2))
        result = _run(probe_image, source, tmp_path / "out")

        assert result.exit_code != 0
        assert isinstance(result.exception, SystemExit) or result.exception is not None

    def test_allow_non_social_overrides_deliberately(
        self, probe_image: Path, tmp_path: Path
    ) -> None:
        from src import evidence

        source = _recorded(tmp_path / "recorded", non_social_payload(2))
        result = _run(
            probe_image, source, tmp_path / "out", extra=["--allow-non-social"]
        )

        assert result.exit_code == 0, result.output
        receipt = evidence.read_receipt(latest_run_dir(tmp_path / "out"))
        assert "wikipedia.org" in receipt.match.post_url

    def test_reports_coverage_for_every_target_platform(
        self, probe_image: Path, recorded_search: Path, tmp_path: Path
    ) -> None:
        """Searched-and-found-nothing is a more honest report than silence."""
        result = do_run(probe_image, recorded_search, tmp_path / "out")
        for label in ("Facebook", "X / Twitter", "Threads", "LinkedIn", "TikTok"):
            assert label in result.output


class TestPublishedRunStillVerifies:
    """Guards the decision to freeze the receipt schema.

    A run is already anchored on Sepolia. Any change to how a receipt
    canonicalises would change its digest and silently invalidate that record.
    """

    def test_the_committed_receipt_still_hashes_to_its_anchored_digest(self) -> None:
        from src import evidence
        from src.chain.canonical import receipt_digest
        from src.config import REPO_ROOT

        run_dir = REPO_ROOT / "evidence" / "run_2026-09-03T16-10-03Z"
        if not run_dir.exists():
            pytest.skip("the published demo run is not present")

        assert receipt_digest(evidence.read_receipt(run_dir)) == (
            evidence.read_anchor(run_dir).digest
        )


@pytest.mark.usefixtures("stubs")
class TestVerifyAndTamper:
    def test_verify_reports_a_local_run_as_unverifiable(
        self, probe_image: Path, recorded_search: Path, tmp_path: Path
    ) -> None:
        """Exit 2 distinguishes 'no ledger left' from 'the receipt was edited'."""
        do_run(probe_image, recorded_search, tmp_path / "out")
        run_dir = latest_run_dir(tmp_path / "out")

        result = runner.invoke(pipeline.app, ["verify", "--run", str(run_dir)])
        assert result.exit_code == 2
        assert "UNVERIFIABLE" in result.output

    def test_verify_detects_an_edited_receipt(
        self, probe_image: Path, recorded_search: Path, tmp_path: Path
    ) -> None:
        do_run(probe_image, recorded_search, tmp_path / "out")
        run_dir = latest_run_dir(tmp_path / "out")

        receipt_path = run_dir / "receipt.json"
        stored = json.loads(receipt_path.read_text(encoding="utf-8"))
        stored["match"]["page_title"] = stored["match"]["page_title"] + "!"
        receipt_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

        result = runner.invoke(pipeline.app, ["verify", "--run", str(run_dir)])
        assert result.exit_code == 1
        assert "TAMPERED" in result.output

    def test_tamper_demo_changes_the_digest(
        self, probe_image: Path, recorded_search: Path, tmp_path: Path
    ) -> None:
        do_run(probe_image, recorded_search, tmp_path / "out")
        run_dir = latest_run_dir(tmp_path / "out")

        result = runner.invoke(pipeline.app, ["tamper-demo", "--run", str(run_dir)])
        assert result.exit_code == 0
        digests = [line for line in result.output.splitlines() if "digest=" in line]
        assert len(digests) == 2
        assert digests[0] != digests[1]

    def test_tamper_demo_leaves_the_stored_receipt_untouched(
        self, probe_image: Path, recorded_search: Path, tmp_path: Path
    ) -> None:
        do_run(probe_image, recorded_search, tmp_path / "out")
        run_dir = latest_run_dir(tmp_path / "out")
        before = (run_dir / "receipt.json").read_bytes()

        runner.invoke(pipeline.app, ["tamper-demo", "--run", str(run_dir)])
        assert (run_dir / "receipt.json").read_bytes() == before

    def test_verify_on_a_directory_with_no_run(self, tmp_path: Path) -> None:
        result = runner.invoke(pipeline.app, ["verify", "--run", str(tmp_path)])
        assert result.exit_code != 0


@pytest.mark.usefixtures("stubs")
class TestUnreachableCandidates:
    def test_a_dead_link_does_not_stop_the_run(
        self,
        probe_image: Path,
        recorded_search: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def flaky_download(
            urls: Any, *, referer: str | None = None
        ) -> tuple[bytes, str]:
            url = urls[0]
            index = int(url.rsplit("/", 1)[-1].split(".")[0])
            if index == 0:
                raise DownloadError("HTTP 404", url=url)
            return f"image-{index}".encode(), url

        monkeypatch.setattr(
            filter_module, "download_first_available", flaky_download
        )

        result = do_run(probe_image, recorded_search, tmp_path / "out")
        assert result.exit_code == 0, result.output
        assert "unreachable" in result.output
