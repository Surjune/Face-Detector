"""Anchoring and verification against a real EVM.

These run on eth-tester, an in-process chain — the contract is genuinely
compiled and executed, but nothing leaves the machine and no gas is spent.
"""

from __future__ import annotations

import pytest

from src.chain.canonical import receipt_digest
from src.chain.registry import RegistryClient, _to_bytes32
from src.errors import AnchorFailed, ChainError, ChainNotConfigured
from tests.test_canonical import make_receipt, with_title

RECEIPT_URI = "evidence/demo_run/receipt.json"


@pytest.fixture(scope="module")
def registry() -> RegistryClient:
    """A freshly deployed registry on an in-process chain."""
    return RegistryClient.local_chain()


class TestDigestConversion:
    def test_accepts_a_prefixed_digest(self) -> None:
        assert len(_to_bytes32("0x" + "ab" * 32)) == 32

    def test_accepts_an_unprefixed_digest(self) -> None:
        assert len(_to_bytes32("ab" * 32)) == 32

    def test_rejects_a_short_digest(self) -> None:
        with pytest.raises(ChainError, match="32 bytes"):
            _to_bytes32("0xabcd")

    def test_rejects_a_non_hex_digest(self) -> None:
        with pytest.raises(ChainError, match="hexadecimal"):
            _to_bytes32("0x" + "zz" * 32)


class TestAnchorAndVerify:
    def test_an_unknown_digest_is_not_anchored(self, registry: RegistryClient) -> None:
        assert registry.verify("0x" + "11" * 32) is None

    def test_anchoring_makes_a_digest_verifiable(self, registry: RegistryClient) -> None:
        digest = receipt_digest(make_receipt())

        assert registry.verify(digest) is None
        result = registry.anchor(digest, RECEIPT_URI)

        record = registry.verify(digest)
        assert record is not None
        assert record.block_number == result.block_number
        assert record.timestamp > 0
        assert int(record.submitter, 16) != 0

    def test_re_anchoring_the_same_digest_is_rejected(self, registry: RegistryClient) -> None:
        """The first timestamp is the whole evidentiary value; it must be immutable."""
        digest = receipt_digest(with_title("A post that gets anchored twice"))
        registry.anchor(digest, RECEIPT_URI)

        with pytest.raises(AnchorFailed):
            registry.anchor(digest, RECEIPT_URI)

    def test_an_empty_digest_is_rejected(self, registry: RegistryClient) -> None:
        with pytest.raises(AnchorFailed):
            registry.anchor("0x" + "00" * 32, RECEIPT_URI)


class TestTamperDetection:
    def test_an_edited_receipt_no_longer_matches_the_chain(
        self, registry: RegistryClient
    ) -> None:
        """The demonstration the brief asks for, reduced to its essentials."""
        original = with_title("Original caption as published")
        registry.anchor(receipt_digest(original), RECEIPT_URI)
        assert registry.verify(receipt_digest(original)) is not None

        tampered = with_title("Original caption as publishee")
        assert registry.verify(receipt_digest(tampered)) is None

    def test_an_unedited_receipt_still_matches(self, registry: RegistryClient) -> None:
        """Re-deriving the digest from an untouched receipt must keep verifying."""
        receipt = with_title("A caption nobody touched")
        registry.anchor(receipt_digest(receipt), RECEIPT_URI)

        recomputed = receipt_digest(with_title("A caption nobody touched"))
        assert registry.verify(recomputed) is not None


class TestReadOnlyClient:
    def test_a_client_without_a_key_cannot_anchor(self, registry: RegistryClient) -> None:
        readonly = RegistryClient(registry._web3, registry._contract)
        with pytest.raises(ChainNotConfigured):
            readonly.anchor(receipt_digest(with_title("Never anchored")), RECEIPT_URI)
