"""Durable CAS logging for proposal LLM calls."""

from __future__ import annotations

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.propose.generator import GeneratedProposal


def seal_llm_call_log(
    generated: GeneratedProposal,
    *,
    cas: ContentAddressedStore,
    archive: ArchiveStore | None,
) -> dict[str, object]:
    """Store the full prompt and every raw response without truncation."""

    prompt_bytes = generated.prompt.encode("utf-8")
    prompt_ref = _put_bytes(cas, prompt_bytes, archive=archive, media_type="application/json")
    responses = []
    for index, response in enumerate(generated.raw_responses, start=1):
        payload = response.encode("utf-8")
        responses.append(
            {
                "ordinal": index,
                "cas_ref": _put_bytes(cas, payload, archive=archive, media_type="application/json"),
                "size_bytes": len(payload),
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-llm-call-log",
        "state": "ready",
        "prompt_ref": prompt_ref,
        "prompt_size_bytes": len(prompt_bytes),
        "raw_response_refs": responses,
        "raw_response_count": len(responses),
        "attempts": generated.attempts,
        "untruncated": True,
    }


def _put_bytes(
    cas: ContentAddressedStore,
    payload: bytes,
    *,
    archive: ArchiveStore | None,
    media_type: str,
) -> str:
    ref = cas.put_bytes(payload, media_type=media_type).uri
    if archive is not None:
        archive.record_artifact_reference(ref)
    return ref
