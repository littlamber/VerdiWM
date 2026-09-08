from __future__ import annotations

import json
import hashlib
import base64
from pathlib import Path
import tempfile
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from wmloop.experiments.community_bundle import (
    CommunityBundleError,
    publish_community_bundle,
    verify_community_bundle,
)
from wmloop.control.intermediate_ir import build_model_capability_ir
from wmloop.geometry.community_knowledge import build_knowledge_lifecycle_record


class CommunityBundleTests(unittest.TestCase):
    def _document(self) -> dict[str, object]:
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-model-onboarding-report",
            "repo_name": "fixture",
            "source_revision": {"kind": "source_tree_sha256", "revision": "a" * 64},
            "capabilities": [{"capability": "inference", "state": "discovered", "evidence": ["fixture"]}],
            "connector": {
                "entrypoints_by_kind": {"inference": ["infer"]},
                "asset_bindings": [],
            },
            "evaluator_contract": {
                "state": "ready",
                "evaluator_id": "fixture-evaluator",
                "contract_sha256": "b" * 64,
                "verifier": "fixture-verifier",
            },
        }
        return build_model_capability_ir(report, model_family="fixture")

    def _keypair(self, root: Path) -> tuple[Path, Path]:
        private = Ed25519PrivateKey.generate()
        private_path = root / "publisher-private.pem"
        public_path = root / "publisher-public.pem"
        private_path.write_bytes(
            private.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        public_path.write_bytes(
            private.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        return private_path, public_path

    def _execution(self, root: Path) -> Path:
        body = {
            "schema_version": 1,
            "artifact_type": "verdiwm-model-batch-execution",
            "state": "completed",
            "batch_id": "fixture-batch",
            "plan_sha256": "c" * 64,
            "shared_budget": {"path": str(root / "private-budget.db"), "total_gpu_hours": 2.0},
            "campaigns_root": str(root / "private-campaigns"),
            "rows": [
                {"model_id": "fixture-model", "state": "completed", "campaign_id": "fixture-campaign"}
            ],
            "summary": {"model_count": 1, "campaign_count": 1, "blocked_count": 0},
            "dispatcher": None,
            "claim_boundary": "This manifest records campaign orchestration and execution status only.",
        }
        canonical = json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        body["execution_sha256"] = hashlib.sha256(canonical).hexdigest()
        path = root / "execution.json"
        path.write_text(json.dumps(body, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return path

    def test_publish_and_verify_signed_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private, public = self._keypair(root)
            bundle = publish_community_bundle(
                documents=[self._document()],
                output_root=root / "bundle",
                publisher_id="community/fixture",
                signing_key=private,
                trust_state="locally_validated",
            )
            self.assertEqual(bundle["trust_state"], "locally_validated")
            result = verify_community_bundle(root / "bundle", public_key=public)
            self.assertEqual(result["state"], "verified")
            self.assertEqual(result["record_count"], 1)

    def test_member_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private, public = self._keypair(root)
            publish_community_bundle(
                documents=[self._document()],
                output_root=root / "bundle",
                publisher_id="community/fixture",
                signing_key=private,
            )
            graph = root / "bundle" / "graph.json"
            graph.write_text(graph.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CommunityBundleError, "MEMBER_DIGEST_MISMATCH"):
                verify_community_bundle(root / "bundle", public_key=public)

    def test_revoked_state_is_integrity_verified_but_not_usable_as_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private, public = self._keypair(root)
            publish_community_bundle(
                documents=[self._document()],
                output_root=root / "bundle",
                publisher_id="community/fixture",
                signing_key=private,
                trust_state="revoked",
                community_review_state="withdrawn",
            )
            result = verify_community_bundle(root / "bundle", public_key=public)
            self.assertEqual(result["state"], "revoked")

    def test_unlisted_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private, public = self._keypair(root)
            publish_community_bundle(
                documents=[self._document()],
                output_root=root / "bundle",
                publisher_id="community/fixture",
                signing_key=private,
            )
            (root / "bundle" / "README.txt").write_text("metadata", encoding="utf-8")
            with self.assertRaisesRegex(CommunityBundleError, "DIRECTORY_CONTENT_MISMATCH"):
                verify_community_bundle(root / "bundle", public_key=public)

    def test_invalid_signature_is_reported_as_bundle_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private, public = self._keypair(root)
            publish_community_bundle(
                documents=[self._document()],
                output_root=root / "bundle",
                publisher_id="community/fixture",
                signing_key=private,
            )
            signature_path = root / "bundle" / "signature.json"
            signature = json.loads(signature_path.read_text(encoding="utf-8"))
            original = signature["signature"]
            signature["signature"] = ("A" if original[0] != "A" else "B") + original[1:]
            signature_path.write_text(
                json.dumps(signature, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CommunityBundleError, "SIGNATURE_INVALID"):
                verify_community_bundle(root / "bundle", public_key=public)

    def test_bundle_id_is_bound_to_manifest_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private, public = self._keypair(root)
            publish_community_bundle(
                documents=[self._document()],
                output_root=root / "bundle",
                publisher_id="community/fixture",
                signing_key=private,
            )
            manifest_path = root / "bundle" / "bundle.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["bundle_id"] = "verdiwm-bundle-" + "0" * 24
            manifest_bytes = json.dumps(
                manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode() + b"\n"
            manifest_path.write_text(
                manifest_bytes.decode(),
                encoding="utf-8",
            )
            signature_path = root / "bundle" / "signature.json"
            signature = json.loads(signature_path.read_text(encoding="utf-8"))
            loaded_private = serialization.load_pem_private_key(
                private.read_bytes(), password=None
            )
            signature["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
            signature["signature"] = base64.b64encode(
                loaded_private.sign(manifest_bytes)
            ).decode("ascii")
            signature_path.write_text(
                json.dumps(signature, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CommunityBundleError, "ID_MISMATCH"):
                verify_community_bundle(root / "bundle", public_key=public)

    def test_batch_execution_binding_is_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private, public = self._keypair(root)
            execution = self._execution(root)
            manifest = publish_community_bundle(
                documents=[self._document()],
                output_root=root / "bundle",
                publisher_id="community/fixture",
                signing_key=private,
                execution_manifest=execution,
            )
            source = manifest["source_execution"]
            self.assertEqual(source["batch_id"], "fixture-batch")
            self.assertEqual(source["model_ids"], ["fixture-model"])
            bundle_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (root / "bundle").rglob("*")
                if path.is_file()
            )
            self.assertNotIn(str(execution), bundle_text)
            result = verify_community_bundle(root / "bundle", public_key=public)
            self.assertEqual(result["state"], "verified")

    def test_batch_execution_digest_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private, _ = self._keypair(root)
            execution = self._execution(root)
            payload = json.loads(execution.read_text(encoding="utf-8"))
            payload["rows"][0]["state"] = "failed"
            execution.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CommunityBundleError, "EXECUTION_DIGEST_MISMATCH"):
                publish_community_bundle(
                    documents=[self._document()],
                    output_root=root / "bundle",
                    publisher_id="community/fixture",
                    signing_key=private,
                    execution_manifest=execution,
                )

    def test_republishing_over_a_tampered_same_id_bundle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private, _ = self._keypair(root)
            publish_community_bundle(
                documents=[self._document()],
                output_root=root / "bundle",
                publisher_id="community/fixture",
                signing_key=private,
            )
            (root / "bundle" / "graph.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(CommunityBundleError, "OUTPUT_CONFLICT"):
                publish_community_bundle(
                    documents=[self._document()],
                    output_root=root / "bundle",
                    publisher_id="community/fixture",
                    signing_key=private,
                )

    def test_empty_destination_directory_is_claimed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private, public = self._keypair(root)
            destination = root / "bundle"
            destination.mkdir()
            publish_community_bundle(
                documents=[self._document()],
                output_root=destination,
                publisher_id="community/fixture",
                signing_key=private,
            )
            self.assertEqual(verify_community_bundle(destination, public_key=public)["state"], "verified")

    def test_lifecycle_record_is_path_free_and_deterministic(self) -> None:
        cas_ref = "cas://sha256/" + "d" * 64
        first = build_knowledge_lifecycle_record(
            action="revocation",
            subject_kind="community_bundle",
            subject_id="verdiwm-bundle-fixture",
            reason="target-side verifier found an invalid claim",
            authority_ref=cas_ref,
            evidence_refs=[cas_ref],
        )
        second = build_knowledge_lifecycle_record(
            action="revocation",
            subject_kind="community_bundle",
            subject_id="verdiwm-bundle-fixture",
            reason="target-side verifier found an invalid claim",
            authority_ref=cas_ref,
            evidence_refs=[cas_ref],
        )
        self.assertEqual(first, second)
        self.assertNotIn("/tmp/", json.dumps(first, ensure_ascii=True))


if __name__ == "__main__":
    unittest.main()
