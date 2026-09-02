#!/usr/bin/env python3
"""Fetch and verify WorldArena evaluator assets from a declarative manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "verdiwm-worldarena-assets/1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _verify(path: Path, asset: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        return {"state": "missing", "path": str(path)}
    digest = _sha256(path)
    expected = str(asset["sha256"])
    size = path.stat().st_size
    return {
        "state": "verified" if digest == expected and size == int(asset["bytes"]) else "mismatch",
        "path": str(path),
        "bytes": size,
        "sha256": digest,
        "expected_sha256": expected,
        "expected_bytes": int(asset["bytes"]),
    }


def _convert_sea_raft(source: Path, destination: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    state = load_file(str(source), device="cpu")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": _sha256(destination)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/evaluators/worldarena_assets_v1.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--receipt", type=Path, default=None, help="Write the verification receipt to this path")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.expanduser().read_text(encoding="utf-8"))
    root = args.output_root.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    failed = False
    for asset in manifest.get("assets", []):
        path = root / str(asset["local_path"])
        before = _verify(path, asset)
        if before["state"] != "verified" and not args.verify_only:
            _download(str(asset["source"]), path)
            before = _verify(path, asset)
        row = {"id": asset["id"], "asset": before}
        if asset.get("derived_torch_path") and before["state"] == "verified":
            derived = root / str(asset["derived_torch_path"])
            if not derived.is_file() or _sha256(derived) != str(asset["derived_sha256"]):
                if not args.verify_only:
                    row["derived"] = _convert_sea_raft(path, derived)
                else:
                    row["derived"] = {"state": "missing_or_mismatch", "path": str(derived)}
            else:
                row["derived"] = {"state": "verified", "path": str(derived), "bytes": derived.stat().st_size, "sha256": _sha256(derived)}
        rows.append(row)
        failed = failed or before["state"] != "verified" or ("derived" in row and row["derived"].get("sha256") != asset.get("derived_sha256"))
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-worldarena-asset-receipt",
        "state": "blocked" if failed else "verified",
        "manifest": str(args.manifest.expanduser().resolve()),
        "output_root": str(root),
        "assets": rows,
        "claim_boundary": manifest.get("claim_boundary"),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    receipt_path = args.receipt.expanduser().resolve() if args.receipt else root / "worldarena_asset_receipt.json"
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
