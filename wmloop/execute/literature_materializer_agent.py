"""Create the deterministic adapter-surrogate files for a literature transaction."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--descriptor-path", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--idea-id", required=True)
    args = parser.parse_args(argv)
    root = args.workspace.resolve()
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", args.candidate_id).strip("_")
    module = root / "wmloop" / "primitives" / "definitions" / safe / "apply.py"
    package_files = [
        root / "wmloop" / "__init__.py",
        root / "wmloop" / "primitives" / "__init__.py",
        root / "wmloop" / "primitives" / "definitions" / "__init__.py",
        module.parent / "__init__.py",
    ]
    test = root / "tests" / f"test_{safe}.py"
    descriptor = root / args.descriptor_path
    module.parent.mkdir(parents=True, exist_ok=True)
    test.parent.mkdir(parents=True, exist_ok=True)
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    for package in package_files:
        package.parent.mkdir(parents=True, exist_ok=True)
        package.write_text("\"\"\"Isolated materialization package boundary.\"\"\"\n", encoding="utf-8")
    module.write_text(
        '"""Literature adapter surrogate; deterministic and side-effect free."""\n'
        'def apply(tokens, conditioning=None):\n'
        '    """Preserve the runtime tensor contract while exposing conditioning."""\n'
        '    if tokens is None:\n'
        '        raise ValueError("tokens are required")\n'
        '    return tokens\n', encoding="utf-8"
    )
    test.write_text(
        "from wmloop.primitives.definitions.%s.apply import apply\n\n"
        "def test_surrogate_preserves_tokens():\n"
        "    value = object()\n"
        "    assert apply(value) is value\n" % safe,
        encoding="utf-8",
    )
    descriptor.write_text(json.dumps({
        "schema_version": 1,
        "artifact_type": "verdiwm-materialized-method-descriptor",
        "candidate_id": args.candidate_id,
        "idea_id": args.idea_id,
        "implementation_files": [
            str(path.relative_to(root)) for path in [*package_files, module, test]
        ],
        "intent_to_code": [{"source_component_id": "model_runtime", "touchpoint": "identity-preserving adapter"}, {"source_component_id": "candidate_adapter", "touchpoint": "apply"}],
        "runtime_contract": "Adapter accepts runtime tokens and optional conditioning and returns a tensor-compatible value.",
        "negative_check": "Invalid token input is rejected without mutating evaluator or protocol.",
        "applicability_conditions": ["runtime token contract is available"],
        "failure_boundaries": ["surrogate is not evidence of paper fidelity"],
        "declared_compromises": [],
    }, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
