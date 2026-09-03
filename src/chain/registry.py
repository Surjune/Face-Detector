"""Client for the PostRegistry contract.

Three ways to reach a registry, matching the three kinds of user:

* `connect_readonly` — no key, no wallet, no gas. This is how a reviewer
  verifies a shipped receipt against the public chain.
* `connect_signing` — a funded Sepolia key, for anchoring a real run.
* `local_chain` — an in-process eth-tester chain that deploys a fresh registry
  on the spot, so the full pipeline can be exercised end to end with no faucet
  and no external node.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import (
    ANCHOR_GAS_LIMIT,
    ARTIFACTS_ROOT,
    SEPOLIA_CHAIN_ID,
    TX_RECEIPT_TIMEOUT_SECONDS,
)
from src.chain.compiler import load_artifact
from src.errors import AnchorFailed, ChainError, ChainNotConfigured

DEPLOYMENT_PATH = ARTIFACTS_ROOT / "deployment.json"


@dataclass(frozen=True)
class AnchorRecord:
    """An existing on-chain record for a digest."""

    submitter: str
    timestamp: int
    block_number: int


@dataclass(frozen=True)
class AnchorResult:
    """The outcome of anchoring a digest."""

    tx_hash: str
    block_number: int
    chain_id: int
    contract_address: str
    gas_used: int


def load_deployment() -> dict[str, Any]:
    """Read the committed deployment record (address, chain, tx).

    Raises:
        ChainNotConfigured: no deployment has been recorded yet.
    """
    if not DEPLOYMENT_PATH.exists():
        raise ChainNotConfigured(
            "No deployment recorded. Run scripts/deploy.py, or pass --chain local.",
            expected=str(DEPLOYMENT_PATH),
        )
    try:
        return json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (OSError, json.JSONDecodeError) as exc:
        raise ChainError(f"Could not read {DEPLOYMENT_PATH.name}: {exc}") from exc


def save_deployment(record: dict[str, Any]) -> Path:
    """Persist a deployment record next to the contract artifact."""
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return DEPLOYMENT_PATH


class RegistryClient:
    """Thin wrapper over the deployed contract."""

    def __init__(self, web3: Any, contract: Any, *, account: Any | None = None) -> None:
        self._web3 = web3
        self._contract = contract
        self._account = account

    # -- construction ---------------------------------------------------

    @classmethod
    def connect_readonly(cls, rpc_url: str, address: str) -> RegistryClient:
        """Connect for lookups only. Needs no key and spends no gas."""
        web3 = _http_web3(rpc_url)
        return cls(web3, _contract_at(web3, address))

    @classmethod
    def connect_signing(cls, rpc_url: str, private_key: str, address: str) -> RegistryClient:
        """Connect with a signing key, for anchoring.

        Raises:
            ChainNotConfigured: the key is unusable.
        """
        web3 = _http_web3(rpc_url)
        try:
            account = web3.eth.account.from_key(private_key)
        except (ValueError, TypeError) as exc:
            raise ChainNotConfigured(f"PRIVATE_KEY is not a valid key: {exc}") from exc
        return cls(web3, _contract_at(web3, address), account=account)

    @classmethod
    def local_chain(cls) -> RegistryClient:
        """Spin up an in-process chain and deploy a fresh registry to it.

        Nothing is installed or downloaded: eth-tester runs the EVM inside this
        Python process. Used by `--chain local` and by the test-suite.
        """
        try:
            from web3 import EthereumTesterProvider, Web3
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ChainError("web3 is not installed") from exc

        try:
            web3 = Web3(EthereumTesterProvider())
        except Exception as exc:
            raise ChainError(f"Could not start the in-process chain: {exc}") from exc

        artifact = load_artifact()
        deployer = web3.eth.accounts[0]
        factory = web3.eth.contract(abi=artifact["abi"], bytecode=artifact["bytecode"])
        tx_hash = factory.constructor().transact({"from": deployer})
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        contract = web3.eth.contract(address=receipt["contractAddress"], abi=artifact["abi"])
        return cls(web3, contract, account=_LocalAccount(deployer))

    # -- properties -----------------------------------------------------

    @property
    def address(self) -> str:
        return str(self._contract.address)

    @property
    def chain_id(self) -> int:
        return int(self._web3.eth.chain_id)

    @property
    def is_local(self) -> bool:
        return isinstance(self._account, _LocalAccount)

    # -- operations -----------------------------------------------------

    def verify(self, digest: str) -> AnchorRecord | None:
        """Look up a digest. Returns None when it has never been anchored."""
        try:
            exists, submitter, timestamp, block_number = self._contract.functions.verify(
                _to_bytes32(digest)
            ).call()
        except Exception as exc:
            raise ChainError(f"Registry lookup failed: {exc}", digest=digest) from exc

        if not exists:
            return None
        return AnchorRecord(
            submitter=str(submitter),
            timestamp=int(timestamp),
            block_number=int(block_number),
        )

    def anchor(self, digest: str, uri: str) -> AnchorResult:
        """Write a digest to the registry and wait for it to be mined.

        Raises:
            ChainNotConfigured: the client has no signing account.
            AnchorFailed: the transaction reverted or never confirmed.
        """
        if self._account is None:
            raise ChainNotConfigured("This client is read-only; no signing key was supplied")

        digest_bytes = _to_bytes32(digest)
        call = self._contract.functions.anchor(digest_bytes, uri)

        try:
            if isinstance(self._account, _LocalAccount):
                tx_hash = call.transact({"from": self._account.address})
            else:
                tx_hash = self._send_signed(call)
            receipt = self._web3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=TX_RECEIPT_TIMEOUT_SECONDS
            )
        except Exception as exc:
            raise AnchorFailed(f"Anchoring failed: {exc}", digest=digest) from exc

        if receipt["status"] != 1:
            raise AnchorFailed("Anchor transaction reverted", digest=digest)

        return AnchorResult(
            tx_hash=receipt["transactionHash"].hex(),
            block_number=int(receipt["blockNumber"]),
            chain_id=self.chain_id,
            contract_address=self.address,
            gas_used=int(receipt["gasUsed"]),
        )

    def _send_signed(self, call: Any) -> Any:
        """Build, sign and broadcast a transaction from the configured key."""
        account = self._account
        transaction = call.build_transaction(
            {
                "from": account.address,
                "nonce": self._web3.eth.get_transaction_count(account.address),
                "gas": ANCHOR_GAS_LIMIT,
                "chainId": self.chain_id,
            }
        )
        signed = self._web3.eth.account.sign_transaction(transaction, account.key)
        # web3 v7 renamed this attribute; support both so the pin can float.
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        return self._web3.eth.send_raw_transaction(raw)


@dataclass(frozen=True)
class _LocalAccount:
    """Marker for an unlocked eth-tester account, which needs no signing."""

    address: str


def _http_web3(rpc_url: str) -> Any:
    """Open an HTTP connection to a node.

    Raises:
        ChainError: the endpoint is unreachable.
    """
    try:
        from web3 import Web3
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ChainError("web3 is not installed") from exc

    web3 = Web3(Web3.HTTPProvider(rpc_url))
    try:
        connected = web3.is_connected()
    except Exception as exc:
        raise ChainError(f"Could not reach {rpc_url}: {exc}") from exc
    if not connected:
        raise ChainError(f"Could not reach {rpc_url}")
    return web3


def _contract_at(web3: Any, address: str) -> Any:
    """Bind the committed ABI to a deployed address."""
    artifact = load_artifact()
    try:
        checksummed = web3.to_checksum_address(address)
    except (ValueError, TypeError) as exc:
        raise ChainNotConfigured(f"Not a valid contract address: {address!r}") from exc
    return web3.eth.contract(address=checksummed, abi=artifact["abi"])


def _to_bytes32(digest: str) -> bytes:
    """Convert a 0x-prefixed sha256 hex string to the contract's bytes32.

    Raises:
        ChainError: the digest is not 32 bytes of hex.
    """
    text = digest[2:] if digest.startswith("0x") else digest
    try:
        raw = bytes.fromhex(text)
    except ValueError as exc:
        raise ChainError(f"Digest is not hexadecimal: {digest!r}") from exc
    if len(raw) != 32:
        raise ChainError(f"Digest must be 32 bytes, got {len(raw)}", digest=digest)
    return raw


def describe_chain(chain_id: int) -> str:
    """Human-readable chain name for logs and reports."""
    if chain_id == SEPOLIA_CHAIN_ID:
        return "Ethereum Sepolia"
    return f"chain {chain_id}"
