// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title PostRegistry
/// @notice Tamper-evident anchor for face-search results.
/// @dev Only a digest is stored on chain. The receipt it summarises stays off
///      chain, so no image, embedding or personal data is ever published. A
///      verifier recomputes the digest from their copy of the receipt and looks
///      it up here: a match proves the receipt is byte-identical to the one
///      anchored at `timestamp`, and any edit changes the digest so the lookup
///      fails.
contract PostRegistry {
    struct Record {
        address submitter;
        uint64 timestamp;
        uint64 blockNumber;
    }

    /// @dev sha256 of the canonical receipt => anchoring record.
    mapping(bytes32 => Record) private _records;

    /// @notice Emitted once per anchored digest.
    /// @param uri Where the receipt can be found. Held in the log rather than in
    ///        storage: logs are far cheaper and this value is never read on chain.
    event Anchored(
        bytes32 indexed digest,
        address indexed submitter,
        uint64 timestamp,
        string uri
    );

    error EmptyDigest();
    error AlreadyAnchored(bytes32 digest, uint64 timestamp);

    /// @notice Record a receipt digest.
    /// @dev Re-anchoring is rejected so the first submission timestamp, which is
    ///      the only thing giving the record evidentiary value, can never be
    ///      overwritten by a later submitter.
    function anchor(bytes32 digest, string calldata uri) external {
        if (digest == bytes32(0)) {
            revert EmptyDigest();
        }

        Record storage existing = _records[digest];
        if (existing.timestamp != 0) {
            revert AlreadyAnchored(digest, existing.timestamp);
        }

        _records[digest] = Record({
            submitter: msg.sender,
            timestamp: uint64(block.timestamp),
            blockNumber: uint64(block.number)
        });

        emit Anchored(digest, msg.sender, uint64(block.timestamp), uri);
    }

    /// @notice Look up a digest.
    /// @return exists Whether this digest has been anchored.
    /// @return submitter Address that anchored it, or the zero address.
    /// @return timestamp Block timestamp of the anchoring, or zero.
    /// @return blockNumber Block the anchoring landed in, or zero.
    function verify(bytes32 digest)
        external
        view
        returns (bool exists, address submitter, uint64 timestamp, uint64 blockNumber)
    {
        Record memory record = _records[digest];
        return (
            record.timestamp != 0,
            record.submitter,
            record.timestamp,
            record.blockNumber
        );
    }
}
