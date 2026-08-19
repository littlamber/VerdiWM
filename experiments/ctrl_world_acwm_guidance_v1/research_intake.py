"""Read-only external research intake for the Ctrl-World ACWM experiment.

External papers and projects are evidence inputs, never executable instructions.
The intake writes source-linked, falsifiable work orders.  A work order cannot
launch a GPU process until a separately frozen implementation and ACWM batch
bind it to the existing screen -> confirm -> frozen-verifier contract.
"""

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.acwm_campaign import canonical_json_bytes, sha256_bytes, sha256_file
from wmloop.control.acwm_dual_evaluation import validate_acwm_dual_evaluation_contract


class ACWMResearchIntakeError(RuntimeError):
    """The source-backed ACWM research intake cannot be resumed safely."""


FetchBytes = Callable[[str, float, int], bytes]

_ARXIV_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
_ARXIV_ID = re.compile(r"arxiv.org/abs/([^/?]+)")
_SAFE_ID = re.compile(r"[^a-z0-9_-]+")
_INSTRUCTION_MARKERS = (
    "ignore previous",
    "system prompt",
    "developer message",
    "<script",
    "```",
)
_CORE_RELEVANCE_TERMS = (
    "world model",
    "multimodal",
    "vision-language model",
    "vision language model",
    "video-language",
    "vlm",
    "action-conditioned",
    "action conditioned",
    "latent action",
    "rollout",
    "video prediction",
    "video world",
    "robot",
)
_MECHANISM_RELEVANCE_TERMS = (
    "long horizon",
    "long-horizon",
    "temporal",
    "memory",
    "autoregressive",
    "diffusion",
    "inference mismatch",
    "training inference",
    "drift",
)
_TRACKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("training_inference_alignment", ("self-forcing", "self forcing", "teacher forcing", "inference mismatch")),
    ("long_horizon_memory", ("memory", "long horizon", "long-horizon", "state space", "history")),
    ("action_conditioning", ("action conditioned", "action-conditioned", "multimodal action", "vision-language action", "vision language action", "control", "robot")),
    ("rollout_stability", ("rollout", "autoregressive", "diffusion forcing", "stability", "drift")),
)


@dataclass(frozen=True)
class ResearchSource:
    source_id: str
    source_type: str
    title: str
    summary: str
    source_url: str
    query: str
    source_digest: str

    def to_document(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "title": self.title,
            "summary": self.summary,
            "source_url": self.source_url,
            "query": self.query,
            "source_digest": self.source_digest,
        }


def run_research_intake(
    *,
    config_path: Path,
    contract_path: Path,
    output_root: Path,
    project_root: Path,
    failure_context: Sequence[str] = (),
    fetch_bytes: FetchBytes | None = None,
) -> dict[str, object]:
    """Collect external metadata and write non-executable ACWM research work orders."""

    root = Path(project_root).expanduser().resolve()
    config_file = _require_file(config_path, "ACWM_RESEARCH_CONFIG_INVALID")
    contract_file = _require_file(contract_path, "ACWM_RESEARCH_CONTRACT_INVALID")
    config = _load_mapping(config_file, "ACWM_RESEARCH_CONFIG_INVALID")
    contract = _load_mapping(contract_file, "ACWM_RESEARCH_CONTRACT_INVALID")
    _validate_config(config, contract, root=root)
    destination = Path(output_root).expanduser().resolve()
    if destination == root or root in destination.parents:
        raise ACWMResearchIntakeError("ACWM_RESEARCH_OUTPUT_INSIDE_SOURCE")

    normalized_context = tuple(sorted({value.strip() for value in failure_context if value.strip()}))
    input_lock = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-research-intake-lock",
        "policy_id": config["policy_id"],
        "policy_digest": config["policy_digest"],
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "contract_path": str(contract_file),
        "contract_sha256": sha256_file(contract_file),
        "intake_implementation_sha256": sha256_file(Path(__file__)),
        "failure_context": list(normalized_context),
    }
    input_lock["input_sha256"] = sha256_bytes(canonical_json_bytes(input_lock))
    existing = _resume_if_bound(destination, input_lock)
    if existing is not None:
        return existing

    fetch = fetch_bytes or _fetch_url
    sources, retrieval = _collect_sources(config, fetch=fetch)
    safe_sources, rejected = _safe_sources(sources)
    ideas = _synthesize_ideas(config, safe_sources, failure_context=normalized_context)
    state = "ready_for_materialization" if ideas else "no_admissible_external_idea"
    work_orders = [_work_order(config, contract, idea) for idea in ideas]

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if destination.exists() or destination.is_symlink() or temporary.exists():
        raise ACWMResearchIntakeError("ACWM_RESEARCH_OUTPUT_EXISTS")
    try:
        temporary.mkdir(mode=0o700, parents=True)
        _write_json(temporary / "input-lock.json", input_lock)
        _write_json(
            temporary / "sources.json",
            {
                "artifact_type": "verdiwm-acwm-research-sources",
                "accepted_sources": [source.to_document() for source in safe_sources],
                "rejected_sources": rejected,
                "retrieval": retrieval,
            },
        )
        for source in safe_sources:
            _write_json(temporary / "sources" / f"{source.source_id}.json", source.to_document())
        for idea, work_order in zip(ideas, work_orders):
            _write_json(temporary / "ideas" / f"{idea['idea_id']}.json", idea)
            _write_json(temporary / "work-orders" / f"{idea['idea_id']}.json", work_order)
        manifest = {
            "schema_version": 1,
            "artifact_type": "verdiwm-acwm-research-intake-manifest",
            "state": state,
            "policy_id": config["policy_id"],
            "policy_digest": config["policy_digest"],
            "input_sha256": input_lock["input_sha256"],
            "source_count": len(safe_sources),
            "rejected_source_count": len(rejected),
            "idea_count": len(ideas),
            "idea_paths": [str(destination / "ideas" / f"{idea['idea_id']}.json") for idea in ideas],
            "work_order_paths": [str(destination / "work-orders" / f"{idea['idea_id']}.json") for idea in ideas],
            "retrieval": retrieval,
            "side_effects": {
                "network_metadata_read": True,
                "source_code_mutated": False,
                "gpu_execution_started": False,
                "candidate_batch_created": False,
                "model_promoted": False,
            },
            "claim_boundary": (
                "External sources generate only source-linked, falsifiable materialization work orders. "
                "A work order has no GPU or promotion authority; a frozen implementation, immutable ACWM batch, "
                "screen, independent confirm, and frozen verifier remain mandatory."
            ),
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.exists() or temporary.is_symlink():
            _remove_tree(temporary)
        raise


def _validate_config(config: Mapping[str, object], contract: Mapping[str, object], *, root: Path) -> None:
    try:
        validate_document("acwm_research_intake", config, root=root)
    except ContractValidationError as exc:
        raise ACWMResearchIntakeError(f"ACWM_RESEARCH_SCHEMA_INVALID:{exc}") from exc
    validate_acwm_dual_evaluation_contract(contract, root=root)
    if config.get("policy_digest") != _digest_without(config, "policy_digest"):
        raise ACWMResearchIntakeError("ACWM_RESEARCH_POLICY_DIGEST_MISMATCH")
    if config.get("contract_id") != contract.get("contract_id"):
        raise ACWMResearchIntakeError("ACWM_RESEARCH_CONTRACT_ID_MISMATCH")
    if config.get("contract_digest") != contract.get("contract_digest"):
        raise ACWMResearchIntakeError("ACWM_RESEARCH_CONTRACT_DIGEST_MISMATCH")
    source_policy = _mapping(config, "source_policy")
    sources = source_policy["sources"]
    if not isinstance(sources, list) or len(sources) != len(set(sources)):
        raise ACWMResearchIntakeError("ACWM_RESEARCH_SOURCE_DUPLICATE")
    execution = _mapping(config, "execution_policy")
    if list(execution["required_stages"]) != ["screen", "confirm", "frozen_verifier"]:
        raise ACWMResearchIntakeError("ACWM_RESEARCH_STAGE_POLICY_INVALID")


def _collect_sources(
    config: Mapping[str, object], *, fetch: FetchBytes
) -> tuple[list[ResearchSource], list[dict[str, object]]]:
    source_policy = _mapping(config, "source_policy")
    queries = config["queries"]
    assert isinstance(queries, list)
    limit = int(source_policy["max_results_per_source"])
    timeout = float(source_policy["timeout_seconds"])
    byte_limit = int(source_policy["response_byte_limit"])
    retries = int(source_policy.get("max_retries", 2))
    backoff = float(source_policy.get("retry_backoff_seconds", 0.25))
    if retries < 0 or retries > 5 or backoff < 0 or backoff > 10:
        raise ACWMResearchIntakeError("ACWM_RESEARCH_RETRY_POLICY_INVALID")
    network_fetch = lambda url, request_timeout, limit: _fetch_with_retry(
        fetch,
        url,
        request_timeout,
        limit,
        retries=retries,
        backoff_seconds=backoff,
    )
    records: list[ResearchSource] = []
    retrieval: list[dict[str, object]] = []
    for query in queries:
        assert isinstance(query, str)
        for source_name in source_policy["sources"]:
            try:
                if source_name == "arxiv":
                    found = _search_arxiv(query, limit=limit, timeout=timeout, byte_limit=byte_limit, fetch=network_fetch)
                elif source_name == "github":
                    found = _search_github(query, limit=limit, timeout=timeout, byte_limit=byte_limit, fetch=network_fetch)
                elif source_name == "openalex":
                    found = _search_openalex(query, limit=limit, timeout=timeout, byte_limit=byte_limit, fetch=network_fetch)
                else:  # schema rejects this; retaining the check keeps the boundary closed under a bypass.
                    raise ACWMResearchIntakeError("ACWM_RESEARCH_SOURCE_UNSUPPORTED")
                records.extend(found)
                retrieval.append({"source": source_name, "query": query, "state": "fetched", "record_count": len(found)})
            except ACWMResearchIntakeError as exc:
                retrieval.append({"source": source_name, "query": query, "state": "unavailable", "reason": str(exc), "record_count": 0})
    deduplicated = {record.source_id: record for record in records}
    return [deduplicated[key] for key in sorted(deduplicated)], retrieval


def _search_arxiv(
    query: str, *, limit: int, timeout: float, byte_limit: int, fetch: FetchBytes
) -> list[ResearchSource]:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": limit, "sortBy": "relevance"}
    )
    try:
        root = ElementTree.fromstring(fetch(url, timeout, byte_limit))
    except (OSError, ElementTree.ParseError) as exc:
        raise ACWMResearchIntakeError("ACWM_RESEARCH_ARXIV_FETCH_FAILED") from exc
    records: list[ResearchSource] = []
    for entry in root.findall("atom:entry", _ARXIV_NAMESPACE):
        raw_id = _text(entry.find("atom:id", _ARXIV_NAMESPACE))
        matched = _ARXIV_ID.search(raw_id)
        arxiv_id = matched.group(1) if matched else raw_id.rsplit("/", 1)[-1]
        title = _compact(_text(entry.find("atom:title", _ARXIV_NAMESPACE)))
        summary = _compact(_text(entry.find("atom:summary", _ARXIV_NAMESPACE)))
        if not arxiv_id or not title or not summary:
            continue
        source_url = raw_id or f"https://arxiv.org/abs/{arxiv_id}"
        records.append(_source("arxiv", arxiv_id, title, summary, source_url, query))
    return records


def _search_github(
    query: str, *, limit: int, timeout: float, byte_limit: int, fetch: FetchBytes
) -> list[ResearchSource]:
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {"q": query, "per_page": limit, "sort": "updated", "order": "desc"}
    )
    try:
        payload = json.loads(fetch(url, timeout, byte_limit))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ACWMResearchIntakeError("ACWM_RESEARCH_GITHUB_FETCH_FAILED") from exc
    items = payload.get("items") if isinstance(payload, Mapping) else None
    if not isinstance(items, list):
        raise ACWMResearchIntakeError("ACWM_RESEARCH_GITHUB_RESPONSE_INVALID")
    records: list[ResearchSource] = []
    for item in items[:limit]:
        if not isinstance(item, Mapping):
            continue
        full_name = str(item.get("full_name") or "").strip()
        title = str(item.get("name") or full_name).strip()
        summary = _compact(str(item.get("description") or ""))
        source_url = str(item.get("html_url") or "").strip()
        if not full_name or not title or not summary or not source_url:
            continue
        records.append(_source("github", full_name, title, summary, source_url, query))
    return records


def _search_openalex(
    query: str, *, limit: int, timeout: float, byte_limit: int, fetch: FetchBytes
) -> list[ResearchSource]:
    """Read paper metadata and abstracts from OpenAlex without executing code."""

    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": query, "per-page": limit, "sort": "relevance_score:desc"}
    )
    try:
        payload = json.loads(fetch(url, timeout, byte_limit))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ACWMResearchIntakeError("ACWM_RESEARCH_OPENALEX_FETCH_FAILED") from exc
    items = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(items, list):
        raise ACWMResearchIntakeError("ACWM_RESEARCH_OPENALEX_RESPONSE_INVALID")
    records: list[ResearchSource] = []
    for item in items[:limit]:
        if not isinstance(item, Mapping):
            continue
        raw_id = str(item.get("id") or "").strip()
        identity = raw_id.rsplit("/", 1)[-1]
        title = str(item.get("display_name") or "").strip()
        abstract = _openalex_abstract(item.get("abstract_inverted_index"))
        source_url = str(
            item.get("landing_page_url") or item.get("doi") or raw_id
        ).strip()
        if not identity or not title or not abstract or not source_url:
            continue
        records.append(_source("openalex", identity, title, abstract, source_url, query))
    return records


def _openalex_abstract(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    indexed: dict[int, str] = {}
    for token, positions in value.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and position >= 0:
                indexed.setdefault(position, token)
    return _compact(" ".join(indexed[index] for index in sorted(indexed)))


def _source(source_type: str, identity: str, title: str, summary: str, source_url: str, query: str) -> ResearchSource:
    normalized_id = _safe_id(f"{source_type}-{identity}")
    payload = {
        "source_id": normalized_id,
        "source_type": source_type,
        "title": title,
        "summary": summary,
        "source_url": source_url,
        "query": query,
    }
    return ResearchSource(source_digest=sha256_bytes(canonical_json_bytes(payload)), **payload)


def _safe_sources(sources: Sequence[ResearchSource]) -> tuple[list[ResearchSource], list[dict[str, str]]]:
    accepted: list[ResearchSource] = []
    rejected: list[dict[str, str]] = []
    for source in sources:
        content = f"{source.title}\n{source.summary}".lower()
        if any(marker in content for marker in _INSTRUCTION_MARKERS):
            rejected.append({"source_id": source.source_id, "reason": "ACWM_RESEARCH_INSTRUCTION_CONTENT"})
        elif not _is_relevant(content):
            rejected.append({"source_id": source.source_id, "reason": "ACWM_RESEARCH_RELEVANCE_INSUFFICIENT"})
        else:
            accepted.append(source)
    return accepted, rejected


def _is_relevant(content: str) -> bool:
    """Require a target-domain signal and a mechanism signal before synthesis."""

    core_hits = {term for term in _CORE_RELEVANCE_TERMS if term in content}
    mechanism_hits = {term for term in _MECHANISM_RELEVANCE_TERMS if term in content}
    if "world model" in core_hits and (len(core_hits) >= 2 or bool(mechanism_hits)):
        return True
    return len(core_hits) >= 2 and bool(mechanism_hits)


def _synthesize_ideas(
    config: Mapping[str, object], sources: Sequence[ResearchSource], *, failure_context: Sequence[str]
) -> list[dict[str, object]]:
    idea_policy = _mapping(config, "idea_policy")
    if len(sources) < int(idea_policy["min_external_sources"]):
        return []
    allowed = {str(value) for value in idea_policy["allowed_tracks"]}
    ideas: list[dict[str, object]] = []
    for source in sources:
        track = _track_for(source)
        if track not in allowed:
            continue
        idea_id = _safe_id(f"idea-{source.source_id}-{track}")
        ideas.append(
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-acwm-research-idea",
                "idea_id": idea_id,
                "track": track,
                "source": {
                    "source_id": source.source_id,
                    "source_type": source.source_type,
                    "source_url": source.source_url,
                    "source_digest": source.source_digest,
                },
                "failure_context": list(failure_context),
                "objective": f"Test whether a materialized {track.replace('_', ' ')} intervention improves Ctrl-World ACWM prediction without protected regressions.",
                "hypothesis": _hypothesis(track, source),
                "falsification_criterion": "Reject the intervention unless it has strict paired_prediction improvement and zero giga_style_rollout protected regressions on independent confirmation.",
                "required_protocol": ["screen", "confirm", "frozen_verifier"],
                "synthesis_method": "source_grounded_keyword_routing_v1",
                "execution_state": "materialization_required",
                "claim_boundary": "The source motivates a testable idea only. It does not establish correctness, authorize a source change, or authorize a GPU launch.",
            }
        )
    return ideas[: int(idea_policy["max_ideas"])]


def _track_for(source: ResearchSource) -> str:
    content = f"{source.title} {source.summary}".lower()
    for track, terms in _TRACKS:
        if any(term in content for term in terms):
            return track
    return "rollout_stability"


def _hypothesis(track: str, source: ResearchSource) -> str:
    labels = {
        "action_conditioning": "more faithful action-conditioned state transitions",
        "long_horizon_memory": "lower long-horizon state drift through bounded history retention",
        "training_inference_alignment": "lower train-inference rollout mismatch",
        "rollout_stability": "more stable multi-step rollouts",
    }
    return (
        f"A separately materialized {track.replace('_', ' ')} mechanism inspired by '{source.title}' "
        f"produces {labels[track]} under the unchanged ACWM contract."
    )


def _work_order(config: Mapping[str, object], contract: Mapping[str, object], idea: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-research-materialization-work-order",
        "idea_id": idea["idea_id"],
        "idea_sha256": sha256_bytes(canonical_json_bytes(idea)),
        "policy_id": config["policy_id"],
        "contract": {"contract_id": contract["contract_id"], "contract_digest": contract["contract_digest"]},
        "required_before_gpu": [
            "isolated implementation receipt with source revision",
            "immutable candidate batch bound to this idea",
            "unchanged ACWM dual-evaluation contract",
            "independent confirm baseline",
        ],
        "required_execution_stages": ["screen", "confirm", "frozen_verifier"],
        "forbidden_actions": [
            "execute external source code",
            "mutate the frozen evaluator or data split",
            "reuse confirm measurements as a screen baseline",
            "promote a model from screen or confirm evidence",
        ],
        "execution_authority": "none_until_materialized_and_compiled",
        "claim_boundary": "This work order carries a falsifiable research idea but has no source-mutation, GPU, or promotion authority.",
    }


def _resume_if_bound(destination: Path, input_lock: Mapping[str, object]) -> dict[str, object] | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_symlink() or not destination.is_dir():
        raise ACWMResearchIntakeError("ACWM_RESEARCH_OUTPUT_INVALID")
    lock = _load_mapping(destination / "input-lock.json", "ACWM_RESEARCH_INPUT_LOCK_INVALID")
    if lock.get("input_sha256") != input_lock.get("input_sha256"):
        raise ACWMResearchIntakeError("ACWM_RESEARCH_INPUT_LOCK_MISMATCH")
    return _load_mapping(destination / "manifest.json", "ACWM_RESEARCH_MANIFEST_INVALID")


def _fetch_url(url: str, timeout: float, byte_limit: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "verdiwm-acwm-research-intake/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(byte_limit + 1)
    except OSError as exc:
        raise ACWMResearchIntakeError("ACWM_RESEARCH_NETWORK_UNAVAILABLE") from exc
    if len(payload) > byte_limit:
        raise ACWMResearchIntakeError("ACWM_RESEARCH_RESPONSE_TOO_LARGE")
    return payload


def _fetch_with_retry(
    fetch: FetchBytes,
    url: str,
    timeout: float,
    byte_limit: int,
    *,
    retries: int,
    backoff_seconds: float,
) -> bytes:
    attempts = retries + 1
    for attempt in range(attempts):
        try:
            return fetch(url, timeout, byte_limit)
        except ACWMResearchIntakeError as exc:
            if "NETWORK_UNAVAILABLE" not in str(exc):
                raise
            error: Exception = exc
        except OSError as exc:
            error = exc
        if attempt + 1 < attempts and backoff_seconds:
            time.sleep(backoff_seconds * (2**attempt))
    raise ACWMResearchIntakeError("ACWM_RESEARCH_NETWORK_UNAVAILABLE") from error


def _load_mapping(path: Path, code: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACWMResearchIntakeError(code) from exc
    if not isinstance(payload, dict):
        raise ACWMResearchIntakeError(code)
    return payload


def _require_file(path: Path, code: str) -> Path:
    value = Path(path).expanduser().resolve()
    if value.is_symlink() or not value.is_file():
        raise ACWMResearchIntakeError(code)
    return value


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise ACWMResearchIntakeError("ACWM_RESEARCH_CONFIG_INVALID")
    return child


def _text(node: ElementTree.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


def _compact(value: str) -> str:
    return " ".join(value.split())[:4000]


def _safe_id(value: str) -> str:
    result = _SAFE_ID.sub("-", value.lower()).strip("-")[:110]
    return result or "source"


def _digest_without(document: Mapping[str, object], field: str) -> str:
    return sha256_bytes(canonical_json_bytes({key: value for key, value in document.items() if key != field}))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _remove_tree(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()
