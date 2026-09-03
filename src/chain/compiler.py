"""Solidity compilation.

py-solc-x fetches and runs a pinned solc binary, so the repository needs no
Node.js, Hardhat or Foundry toolchain — one `pip install` covers everything.

The compiled ABI and bytecode are written to `artifacts/PostRegistry.json` and
committed, so verifying a receipt never compiles anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import ARTIFACTS_ROOT, CONTRACTS_ROOT, SOLC_VERSION
from src.errors import ContractCompileError

CONTRACT_NAME = "PostRegistry"
CONTRACT_SOURCE = CONTRACTS_ROOT / f"{CONTRACT_NAME}.sol"
ARTIFACT_PATH = ARTIFACTS_ROOT / f"{CONTRACT_NAME}.json"


def compile_registry() -> dict[str, Any]:
    """Compile PostRegistry.sol and write its artifact.

    Raises:
        ContractCompileError: solc could not be installed or the source failed
            to compile.
    """
    try:
        import solcx
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ContractCompileError("py-solc-x is not installed") from exc

    try:
        if SOLC_VERSION not in {str(version) for version in solcx.get_installed_solc_versions()}:
            solcx.install_solc(SOLC_VERSION)

        compiled = solcx.compile_files(
            [str(CONTRACT_SOURCE)],
            output_values=["abi", "bin"],
            solc_version=SOLC_VERSION,
            optimize=True,
        )
    except Exception as exc:  # solcx raises a family of unrelated exception types
        raise ContractCompileError(f"Compiling {CONTRACT_SOURCE.name} failed: {exc}") from exc

    key = _find_contract_key(compiled)
    artifact = {
        "contract": CONTRACT_NAME,
        "solc_version": SOLC_VERSION,
        "abi": compiled[key]["abi"],
        "bytecode": "0x" + compiled[key]["bin"],
    }

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def load_artifact() -> dict[str, Any]:
    """Read the committed artifact, compiling it first if it is absent."""
    if not ARTIFACT_PATH.exists():
        return compile_registry()
    try:
        artifact: dict[str, Any] = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractCompileError(f"Could not read {ARTIFACT_PATH.name}: {exc}") from exc

    if "abi" not in artifact or "bytecode" not in artifact:
        raise ContractCompileError(f"{ARTIFACT_PATH.name} is missing abi or bytecode")
    return artifact


def _find_contract_key(compiled: dict[str, Any]) -> str:
    """Locate our contract in solc's output.

    solc keys results as `<path>:<ContractName>`, and the path form varies by
    platform, so the name is matched rather than the whole key.
    """
    for key in compiled:
        if key.rsplit(":", 1)[-1] == CONTRACT_NAME:
            return key
    raise ContractCompileError(
        f"{CONTRACT_NAME} not found in compiler output", produced=sorted(compiled)
    )
