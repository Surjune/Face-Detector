"""Terminal output on a console that cannot represent every character.

Page titles come from social platforms and routinely contain emoji. On a
Windows console running a legacy code page, printing one raises
UnicodeEncodeError, which killed `tamper-demo` on a real anchored run whose
YouTube title ended in a heart. Output has to degrade, not abort.
"""

from __future__ import annotations

import io

import pytest

from src import ui

# Present in a real anchored page title, and absent from cp1252.
HEART = "♥️"

WESTERN_CODE_PAGE = "cp1252"


def legacy_console() -> io.TextIOWrapper:
    """A stream that behaves like a Windows console on a legacy code page."""
    return io.TextIOWrapper(io.BytesIO(), encoding=WESTERN_CODE_PAGE, newline="")


def written(stream: io.TextIOWrapper) -> str:
    stream.flush()
    buffer = stream.buffer
    assert isinstance(buffer, io.BytesIO)
    return buffer.getvalue().decode(WESTERN_CODE_PAGE)


class TestUnencodableOutput:
    def test_a_legacy_console_rejects_the_character_unprotected(self) -> None:
        """The failure being guarded against is real, not hypothetical."""
        stream = legacy_console()
        with pytest.raises(UnicodeEncodeError):
            stream.write(f"page_title={HEART}")
            stream.flush()

    def test_the_character_is_escaped_once_the_stream_is_reconfigured(self) -> None:
        stream = legacy_console()
        ui._survive_unencodable_output(stream)
        stream.write(f"Happy retirement {HEART} - YouTube")

        output = written(stream)
        assert "Happy retirement" in output
        # Escaped rather than dropped: a reader can still tell what was there.
        assert "\\u2665" in output

    def test_the_console_encoding_is_left_alone(self) -> None:
        """Forcing UTF-8 onto a cp1252 console would print mojibake instead.

        That is the worse failure, because corrupted output looks like corrupted
        data rather than a limitation of the terminal.
        """
        stream = legacy_console()
        ui._survive_unencodable_output(stream)
        assert stream.encoding == WESTERN_CODE_PAGE

    def test_ordinary_text_is_unaffected(self) -> None:
        stream = legacy_console()
        ui._survive_unencodable_output(stream)
        stream.write("Similarity: 0.9616 cosine")
        assert written(stream) == "Similarity: 0.9616 cosine"

    def test_a_stream_that_cannot_be_reconfigured_is_left_alone(self) -> None:
        """pytest's own capture object among others; there is no console to protect."""
        ui._survive_unencodable_output(io.StringIO())  # must not raise
