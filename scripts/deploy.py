"""Deploy PostRegistry to Ethereum Sepolia.

Run once. The resulting address is written to artifacts/deployment.json and
committed, so nobody else needs to deploy anything to verify a receipt.

    python scripts/deploy.py

Needs PRIVATE_KEY in .env, funded from a faucet that asks for nothing in return:
https://sepolia-faucet.pk910.de
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chain.compiler import compile_registry  # noqa: E402
from src.chain.registry import save_deployment  # noqa: E402
from src.config import (  # noqa: E402
    SEPOLIA_ADDRESS_EXPLORER,
    SEPOLIA_CHAIN_ID,
    SEPOLIA_TX_EXPLORER,
    TX_RECEIPT_TIMEOUT_SECONDS,
    load_settings,
)
from src.errors import ChainError, ChainNotConfigured, PipelineError  # noqa: E402

# A deployment writes the whole contract, so it costs far more than an anchor.
# Measured at roughly 300k; the ceiling leaves room for a compiler change.
DEPLOY_GAS_LIMIT = 600_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation prompt",
    )
    args = parser.parse_args()

    settings = load_settings()
    if not settings.private_key:
        raise ChainNotConfigured("PRIVATE_KEY is not set; see .env.example")

    from web3 import Web3

    print("Compiling PostRegistry.sol...")
    artifact = compile_registry()

    web3 = Web3(Web3.HTTPProvider(settings.sepolia_rpc_url))
    if not web3.is_connected():
        raise ChainError(f"Could not reach {settings.sepolia_rpc_url}")

    chain_id = web3.eth.chain_id
    if chain_id != SEPOLIA_CHAIN_ID:
        raise ChainError(
            f"Connected to chain {chain_id}, expected Sepolia ({SEPOLIA_CHAIN_ID})"
        )

    account = web3.eth.account.from_key(settings.private_key)
    balance = web3.eth.get_balance(account.address)
    print(f"Deployer: {account.address}")
    print(f"Balance:  {web3.from_wei(balance, 'ether')} ETH")
    if balance == 0:
        raise ChainError(
            "Deployer has no Sepolia ETH. Fund it at https://sepolia-faucet.pk910.de",
            address=account.address,
        )

    if not args.yes:
        answer = input("Deploy to Sepolia? [y/N] ").strip().lower()
        if answer != "y":
            print("Cancelled.")
            return 1

    factory = web3.eth.contract(abi=artifact["abi"], bytecode=artifact["bytecode"])
    transaction = factory.constructor().build_transaction(
        {
            "from": account.address,
            "nonce": web3.eth.get_transaction_count(account.address),
            "gas": DEPLOY_GAS_LIMIT,
            "chainId": chain_id,
        }
    )
    signed = web3.eth.account.sign_transaction(transaction, account.key)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    tx_hash = web3.eth.send_raw_transaction(raw)
    print(f"Submitted {tx_hash.hex()}; waiting for it to be mined...")

    receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=TX_RECEIPT_TIMEOUT_SECONDS)
    if receipt["status"] != 1:
        raise ChainError("Deployment transaction reverted", tx_hash=tx_hash.hex())

    address = receipt["contractAddress"]
    record = {
        "contract": artifact["contract"],
        "address": address,
        "chain_id": chain_id,
        "network": "sepolia",
        "tx_hash": receipt["transactionHash"].hex(),
        "block_number": int(receipt["blockNumber"]),
        "deployed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = save_deployment(record)

    print(f"\nDeployed to {address}")
    print(f"  {SEPOLIA_ADDRESS_EXPLORER}{address}")
    print(f"  {SEPOLIA_TX_EXPLORER}{record['tx_hash']}")
    print(f"\nRecorded in {path.relative_to(REPO_ROOT)}. Commit it.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as error:
        print(f"error [{error.code}]: {error}", file=sys.stderr)
        raise SystemExit(1) from error
