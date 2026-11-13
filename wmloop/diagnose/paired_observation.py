"""Produce and recheck signed, ranking-only Wan history observations.

The enclosing experiment archives the result and every evidence file. This
module does not execute a model, settle a trial, or pronounce a formal effect.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.diagnose.wan_native_evidence import wan_invocation_matches, wan_manifest_errors
from wmloop.evaluate.future_video_metrics import (
    METRIC_DIRECTIONS, FutureVideoMetricError, recompute_future_video_metrics, supported_frozen_algorithm,
)
from wmloop.storage import atomic_write, canonical_bytes, checked_path, exclusive_file_lock, sha256_bytes


class PairedObservationError(ValueError):
    """The pair cannot be reconstructed from its declared execution evidence."""


_BOUNDARY = (
    "A matched single-context diagnostic response is a ranking prior only. "
    "It grants no formal effect, conformance, promotion, or transfer authority."
)
_CONTEXT_FIELDS = (
    "checkpoint_sha256", "model_asset_sha256", "selection_sha256", "data_sha256",
    "condition_override_sha256", "entrypoint_sha256", "source_tree_revision",
    "split", "inference_steps", "max_chunks", "limit", "fps",
)


def build_wan_history_observations(
    *, evidence_root: Path, specification: Mapping[str, object],
) -> list[dict[str, object]]:
    """Build observations from frozen measurements and actual native outputs.

    All specification paths are relative to ``evidence_root``. Every returned
    evidence path must also appear in the enclosing experiment's artifacts.
    """
    root = checked_path(evidence_root, code="PAIRED_EVIDENCE_ROOT_INVALID", error=PairedObservationError)
    if not root.is_dir():
        raise PairedObservationError("PAIRED_EVIDENCE_ROOT_INVALID")

    def read(path: str) -> bytes:
        source = checked_path(root / path, code="PAIRED_EVIDENCE_PATH_INVALID", error=PairedObservationError)
        if not source.is_file() or root not in source.parents:
            raise PairedObservationError("PAIRED_EVIDENCE_PATH_INVALID")
        return source.read_bytes()

    return _build(specification, read=read)


def verify_paired_observation(
    observation: Mapping[str, object], *, read_artifact: Callable[[str], bytes],
) -> None:
    """Reconstruct an observation from receipt-bound CAS artifacts on read."""
    verify_paired_observations([observation], read_artifact=read_artifact)


def verify_paired_observations(
    observations: Sequence[Mapping[str, object]], *, read_artifact: Callable[[str], bytes],
) -> None:
    """Replay each distinct proof once, including when it reports many metrics."""
    rebuilt: dict[bytes, set[bytes]] = {}
    for observation in observations:
        validate_document("diagnostic_paired_observation", observation)
        proof = observation["evidence"]
        identity = canonical_bytes(proof)
        if identity not in rebuilt:
            files = proof["files"]

            def read(path: str) -> bytes:
                payload = read_artifact(path)
                if sha256_bytes(payload) != files.get(path):
                    raise PairedObservationError("PAIRED_EVIDENCE_HASH_MISMATCH")
                return payload

            rebuilt[identity] = {canonical_bytes(row) for row in _build(proof["specification"], read=read)}
        if canonical_bytes(observation) not in rebuilt[identity]:
            raise PairedObservationError("PAIRED_OBSERVATION_CONTENT_MISMATCH")


def _object(payload: bytes, code: str) -> dict:
    try:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("object required")
        canonical_bytes(value)
        return value
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise PairedObservationError(code) from exc


def _path(value: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise PairedObservationError("PAIRED_EVIDENCE_PATH_INVALID")


def _build(specification: Mapping[str, object], *, read: Callable[[str], bytes]) -> list[dict[str, object]]:
    try:
        return _build_validated(specification, read=read)
    except (AttributeError, KeyError, TypeError, OverflowError) as exc:
        raise PairedObservationError("PAIRED_EVIDENCE_STRUCTURE_INVALID") from exc


def _build_validated(specification: Mapping[str, object], *, read: Callable[[str], bytes]) -> list[dict[str, object]]:
    validate_document("diagnostic_paired_specification", specification)
    spec = deepcopy(dict(specification))
    names = {spec["protocol"], spec["freeze_record"], *spec["evaluator_files"].values()}
    for side in ("control", "candidate"):
        names.update(spec[side].values())
    for name in names:
        _path(name)
    payloads = {name: read(name) for name in sorted(names)}
    digests = {name: sha256_bytes(payload) for name, payload in payloads.items()}
    protocol = _object(payloads[spec["protocol"]], "PAIRED_PROTOCOL_INVALID")
    validate_document("diagnostic_paired_protocol", protocol)
    freeze = _object(payloads[spec["freeze_record"]], "PAIRED_FREEZE_INVALID")
    if freeze.get("artifact_type") == "verdiwm-frozen-future-video-recipe":
        validate_document("frozen_future_video_recipe", freeze)
    frozen_files = {Path(spec["protocol"]).name: digests[spec["protocol"]]}
    for filename, path in spec["evaluator_files"].items():
        if Path(filename).name != filename or filename in frozen_files:
            raise PairedObservationError("PAIRED_FREEZE_FILES_INVALID")
        frozen_files[filename] = digests[path]
    if not supported_frozen_algorithm({name: frozen_files[name] for name in spec["evaluator_files"]}, freeze):
        raise PairedObservationError("PAIRED_NUMERIC_RECIPE_UNSUPPORTED")
    if (freeze.get("files") != frozen_files or freeze.get("diagnostic_only") is not True
        or freeze.get("identity") != protocol.get("evaluator_id")
        or protocol.get("formal_acceptance") is not False):
        raise PairedObservationError("PAIRED_FREEZE_BINDING_MISMATCH")
    comparison = protocol.get("comparison")
    if not isinstance(comparison, dict) or comparison.get("primitive") != spec["primitive"]:
        raise PairedObservationError("PAIRED_PROTOCOL_ACTION_MISSING")
    if protocol.get("metric_tie_epsilon") != spec["null_tolerance"]:
        raise PairedObservationError("PAIRED_PROTOCOL_TOLERANCE_MISMATCH")
    for metric, direction in spec["metrics"].items():
        if (METRIC_DIRECTIONS.get(metric) != direction
            or protocol.get("metrics", {}).get(metric, {}).get("direction") != direction):
            raise PairedObservationError("PAIRED_PROTOCOL_METRIC_MISMATCH")

    sources = {}
    configs = {}
    measurements = {}
    doses = {}
    for side in ("control", "candidate"):
        paths = spec[side]
        source = _object(payloads[paths.get("receipt", paths.get("execution"))], "PAIRED_SOURCE_INVALID")
        expected_type = ("verdiwm-wan-arm-slot-inference-receipt" if "receipt" in paths
                         else "verdiwm-wan-arm-slot-native-execution")
        validate_document("wan_arm_slot_inference_receipt" if "receipt" in paths else "wan_arm_slot_native_execution", source)
        process = source.get("process", {})
        validation = source.get("validation", {})
        if (source.get("artifact_type") != expected_type or source.get("state") != "passed"
            or validation.get("input_unchanged") is not True or validation.get("errors") != []
            or process.get("state") != "passed" or type(process.get("returncode")) is not int
            or process["returncode"] != 0
            or source.get("authority") != {"diagnostic_only": True, "promotion": False}):
            raise PairedObservationError("PAIRED_SOURCE_NOT_VALIDATED")
        binding = source.get("input_binding")
        if not isinstance(binding, dict) or any(field not in binding for field in (*_CONTEXT_FIELDS, "seed", "config_sha256")):
            raise PairedObservationError("PAIRED_INPUT_BINDING_INCOMPLETE")
        if digests[paths["config"]] != binding["config_sha256"]:
            raise PairedObservationError("PAIRED_CONFIG_BINDING_MISMATCH")
        selection = _object(payloads[paths["selection"]], "PAIRED_SELECTION_INVALID").get("selected")
        manifest = _object(payloads[paths["manifest"]], "PAIRED_MANIFEST_INVALID")
        if (digests[paths["selection"]] != binding["selection_sha256"]
            or not isinstance(selection, list) or not all(isinstance(row, dict) for row in selection)
            or wan_manifest_errors(manifest, binding, selection)
            or not wan_invocation_matches(process, binding)):
            raise PairedObservationError("PAIRED_NATIVE_BINDING_MISMATCH")
        # This recipe describes one context. It cannot silently pool samples.
        if (binding["limit"] != 1 or len(manifest["samples"]) != 1
            or any(manifest["samples"][0].get(key) != comparison.get(key) for key in ("episode_id", "frame_now"))
            or manifest["samples"][0]["frames"] != 13):
            raise PairedObservationError("PAIRED_PROTOCOL_SAMPLE_MISMATCH")
        try:
            config = yaml.safe_load(payloads[paths["config"]])
            world = config["world_model"]
            training = config["wm_training"]
            dose = world.pop("history_rollout_dilation")
            other_dose = training.pop("history_rollout_dilation")
        except (ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            raise PairedObservationError("PAIRED_CONFIG_INVALID") from exc
        if type(dose) is not int or dose < 1 or type(other_dose) is not int or dose != other_dose:
            raise PairedObservationError("PAIRED_CONFIG_ACTION_MISMATCH")
        canonical_bytes(config)
        measurement = _object(payloads[paths["measurement"]], "PAIRED_MEASUREMENT_INVALID")
        if measurement.get("artifact_type") == "verdiwm-future-frame-measurement":
            validate_document("future_frame_measurement", measurement)
        video_hashes = measurement.get("input_videos_sha256")
        expected_videos = sorted((digests[paths["prediction"]], digests[paths["ground_truth"]]))
        if (measurement.get("evaluator_id") != protocol.get("evaluator_id")
            or measurement.get("freeze_record_sha256") != digests[spec["freeze_record"]]
            or measurement.get("diagnostic_only") is not True
            or measurement.get("promotion_allowed") is not False
            or not isinstance(video_hashes, dict) or sorted(video_hashes.values()) != expected_videos):
            raise PairedObservationError("PAIRED_MEASUREMENT_BINDING_MISMATCH")
        artifact_refs = source.get("artifacts")
        sample = manifest["samples"][0]
        expected_refs = {
            "rollouts/manifest.json": digests[paths["manifest"]],
            "rollouts/" + sample["dir"] + "/" + sample["sa_wm_video_file"]: digests[paths["prediction"]],
            "rollouts/" + sample["dir"] + "/" + sample["gt_video_file"]: digests[paths["ground_truth"]],
        }
        if not isinstance(artifact_refs, dict) or any(artifact_refs.get(name) != "cas://sha256/" + digest for name, digest in expected_refs.items()):
            raise PairedObservationError("PAIRED_VIDEO_EXECUTION_MISMATCH")
        try:
            replay = recompute_future_video_metrics(
                prediction=payloads[paths["prediction"]], ground_truth=payloads[paths["ground_truth"]],
            )
        except FutureVideoMetricError as exc:
            raise PairedObservationError("PAIRED_NUMERIC_REPLAY_FAILED") from exc
        if measurement.get("metrics") != replay["metrics"] or measurement.get("per_frame") != replay["per_frame"]:
            raise PairedObservationError("PAIRED_MEASUREMENT_NUMERIC_MISMATCH")
        sources[side] = binding
        configs[side] = config
        doses[side] = dose
        measurements[side] = measurement
    if (configs["control"] != configs["candidate"]
        or any(sources["control"][field] != sources["candidate"][field] for field in (*_CONTEXT_FIELDS, "seed"))
        or digests[spec["control"]["ground_truth"]] != digests[spec["candidate"]["ground_truth"]]):
        raise PairedObservationError("PAIRED_CONTEXT_MISMATCH")
    if (doses["control"] != comparison.get("baseline_history_rollout_dilation")
        or doses["candidate"] not in comparison.get("allowed_candidate_history_rollout_dilations", [])
        or sources["control"]["seed"] != comparison.get("seed")
        or any(sources["control"][field] != comparison.get(field) for field in ("split", "inference_steps", "max_chunks"))):
        raise PairedObservationError("PAIRED_PROTOCOL_CONTEXT_MISMATCH")
    # File names and checkout paths are evidence locations, not query identity.
    context = {field: sources["control"][field] for field in _CONTEXT_FIELDS}
    context["source_tree_revision"] = {key: context["source_tree_revision"][key] for key in ("kind", "revision")}
    context["model_asset_sha256"] = sorted(context["model_asset_sha256"].values())
    context["config_without_intervention_sha256"] = sha256_bytes(canonical_bytes(
        _portable_config(configs["control"], sources["control"]),
    ))
    context_sha256 = sha256_bytes(canonical_bytes(context))
    rows = []
    for metric, direction in spec["metrics"].items():
        values = []
        for side in ("control", "candidate"):
            value = measurements[side].get("metrics", {}).get(metric)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise PairedObservationError("PAIRED_METRIC_NOT_FINITE")
            values.append(float(value))
        delta = (values[1] - values[0]) * (1 if direction == "higher" else -1)
        if not math.isfinite(delta):
            raise PairedObservationError("PAIRED_DELTA_NOT_FINITE")
        tolerance = spec["null_tolerance"]
        outcome = "null" if abs(delta) <= tolerance else "positive" if delta > 0 else "negative"
        row = {
            "schema_version": 1, "artifact_type": "verdiwm-diagnostic-paired-observation",
            "authority": "ranking_only",
            "query": {"primitive": spec["primitive"], "metric_name": metric, "metric_direction": direction,
                      "protocol_sha256": digests[spec["freeze_record"]], "context_sha256": context_sha256,
                      "seed": sources["control"]["seed"],
                      "control_parameters": {"history_rollout_dilation": doses["control"]},
                      "candidate_parameters": {"history_rollout_dilation": doses["candidate"]}},
            "control_value": values[0], "candidate_value": values[1], "signed_delta": delta,
            "outcome": outcome, "evidence": {"specification": spec, "files": digests}, "claim_boundary": _BOUNDARY,
        }
        validate_document("diagnostic_paired_observation", row)
        rows.append(row)
    return rows


def _portable_config(config: dict, binding: dict) -> dict:
    """Replace declared input locations with content identities for matching.

    Only bound locations are normalized; other configuration semantics remain
    part of the identity. Relocating all declared assets preserves this digest.
    """
    locations = {}
    for field in ("source_root", "checkpoint", "data_root", "condition_override_root"):
        if binding[field]:
            locations[binding[field]] = "bound-input:" + field
    for path, digest in binding["model_asset_sha256"].items():
        locations[path] = "sha256:" + digest
    for field in ("checkpoint", "vae_checkpoint"):
        value = config.get("world_model", {}).get(field)
        if isinstance(value, str):
            resolved = Path(value) if Path(value).is_absolute() else Path(binding["source_root"]) / value
            assets = sorted(digest for path, digest in binding["model_asset_sha256"].items()
                            if Path(path) == resolved or resolved in Path(path).parents)
            if assets:
                locations[value] = "model-assets:" + sha256_bytes(canonical_bytes(assets))

    def normalize(value: object) -> object:
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, str):
            return locations.get(value, value)
        return value

    return normalize(config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--diagnostic-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = _object(args.diagnostic_result.read_bytes(), "PAIRED_DIAGNOSTIC_RESULT_INVALID")
    result["paired_observations"] = build_wan_history_observations(
        evidence_root=args.evidence_root, specification=_object(args.specification.read_bytes(), "PAIRED_SPECIFICATION_INVALID"),
    )
    result["primitive"] = result["paired_observations"][0]["query"]["primitive"]
    validate_document("diagnostic_probe_result", result)
    output = checked_path(args.output, code="PAIRED_OUTPUT_INVALID", error=PairedObservationError)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with exclusive_file_lock(output.parent / f".{output.name}.lock"):
        if output.exists():
            raise PairedObservationError("PAIRED_OUTPUT_EXISTS")
        atomic_write(output, canonical_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
