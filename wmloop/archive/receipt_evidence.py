"""Read-only validation of the evidence declared by settled receipts."""

from __future__ import annotations

from typing import Any, Mapping

from wmloop.archive.store import ArchiveInvariantError, ContentAddressedStore
from wmloop.storage import load_json_bytes


class ReceiptEvidenceError(ValueError):
    """A required evidence object is unavailable, invalid, or inconsistent."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"RECEIPT_EVIDENCE_{reason}")


def verify_receipt_evidence(
    *, receipt: Mapping[str, Any], cas: ContentAddressedStore,
) -> None:
    """Read every declared CAS object and bind the separate verdict to its copy.

    Auto-experiment receipts require the complete evidence fields. Older,
    minimal diagnostic receipts may omit support/verdict/context fields, but
    every reference they declare must still resolve. In particular, missing
    fields on an auto-experiment receipt never activate legacy compatibility.
    Settlement and Archive identity remain the caller's responsibility.
    """
    complete = receipt.get("artifact_type") == "verdiwm-auto-experiment-receipt"
    refs: list[str] = []
    for name in ("artifact_refs", "support_refs"):
        default = {} if name == "support_refs" and not complete else None
        block = receipt.get(name, default)
        if not isinstance(block, Mapping) or not all(isinstance(ref, str) for ref in block.values()):
            raise ReceiptEvidenceError("INVALID")
        refs.extend(block.values())
    for name in ("failure_context_ref", "verdict_ref"):
        if name not in receipt and not complete:
            continue
        ref = receipt.get(name)
        if not isinstance(ref, str):
            raise ReceiptEvidenceError("INVALID")
        refs.append(ref)
    json_refs = {receipt[name] for name in ("failure_context_ref", "verdict_ref") if name in receipt}
    payloads: dict[str, bytes] = {}
    try:
        for ref in dict.fromkeys(refs):
            payload = cas.read_bytes(ref)
            if ref in json_refs:
                payloads[ref] = payload
    except (ArchiveInvariantError, OSError) as exc:
        raise ReceiptEvidenceError("INVALID") from exc
    for name in ("failure_context_ref", "verdict_ref"):
        if name not in receipt:
            continue
        try:
            document = load_json_bytes(payloads[receipt[name]])
        except (ValueError, UnicodeDecodeError) as exc:
            raise ReceiptEvidenceError("INVALID") from exc
        if not isinstance(document, dict):
            raise ReceiptEvidenceError("INVALID")
        if name == "verdict_ref" and document != receipt.get("verdict"):
            raise ReceiptEvidenceError("MISMATCH")
