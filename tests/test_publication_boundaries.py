from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wmloop.contracts import ContractValidationError, validate_instance, validate_document, validate_bootstrap_instance
from wmloop.control.model_batch import build_model_batch_execution_binding, ModelBatchError, _digest
from wmloop.experiments.community_bundle import publish_community_bundle, verify_community_bundle, CommunityBundleError
from wmloop.experiments.community_export import export_community_knowledge, CommunityExportError
from wmloop.experiments.community_files import semantic_files
import tests.test_community_bundle as bundle_fixtures


class PublicationBoundaryTests(unittest.TestCase):
    def test_map_values_checked_in_production_and_bootstrap(self):
        schema = {"type": "object", "additionalProperties": {"type": "integer", "minimum": 1}}
        for validator in (validate_instance, validate_bootstrap_instance):
            for invalid in (0, "one", -1, False):
                with self.subTest(validator=validator.__name__, invalid=invalid):
                    with self.assertRaises(ContractValidationError):
                        validator(schema, {"count": invalid})
            validator(schema, {"count": 1})

    def test_document_uses_production_validator_and_missing_dependency_fails(self):
        with patch("wmloop.contracts.Draft202012Validator", None):
            with self.assertRaisesRegex(ContractValidationError, "SCHEMA_VALIDATOR_UNAVAILABLE"):
                validate_document("community_export", {})

    def test_extra_file_and_directory_rejected_when_republishing(self):
        fixture = bundle_fixtures.CommunityBundleTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key, public = fixture._keypair(root)
            options = dict(documents=[fixture._document()], output_root=root / "bundle", publisher_id="community/test", signing_key=key)
            publish_community_bundle(**options)
            for path, directory in ((root / "bundle/extra.txt", False), (root / "bundle/extra", True)):
                if directory:
                    path.mkdir()
                else:
                    path.write_text("unlisted")
                with self.assertRaisesRegex(CommunityBundleError, "OUTPUT_CONFLICT"):
                    publish_community_bundle(**options)
                if directory:
                    path.rmdir()
                else:
                    path.unlink()
            self.assertEqual(verify_community_bundle(root / "bundle", public_key=public)["state"], "verified")

    def test_linked_source_output_and_keys_rejected(self):
        fixture = bundle_fixtures.CommunityBundleTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "one.json").write_text(json.dumps(fixture._document()))
            link = root / "link"
            link.symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(CommunityExportError, "SOURCE_ROOT_INVALID"):
                export_community_knowledge(source_roots=[link], output_root=root / "output")
            with self.assertRaisesRegex(CommunityExportError, "OUTPUT_INVALID"):
                export_community_knowledge(source_roots=[source], output_root=link / "nested")
            key, _ = fixture._keypair(root)
            key_link = root / "linked.pem"
            key_link.symlink_to(key)
            with self.assertRaisesRegex(CommunityBundleError, "SIGNING_KEY_INVALID"):
                publish_community_bundle(documents=[fixture._document()], output_root=root / "bundle", publisher_id="community/test", signing_key=key_link)

    def test_model_id_is_checked_before_signing(self):
        fixture = bundle_fixtures.CommunityBundleTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = fixture._execution(root)
            for value in ("/private/model", "C:\\model", "../model", "model.ckpt", "model/data"):
                document = json.loads(path.read_text())
                document["rows"][0]["model_id"] = value
                document.pop("execution_sha256")
                document["execution_sha256"] = _digest(document)
                path.write_text(json.dumps(document))
                with self.assertRaisesRegex(ModelBatchError, "MODEL_IDS_INVALID"):
                    build_model_batch_execution_binding(path)

    def test_scan_stops_before_visiting_the_whole_tree_and_accepts_jsonl(self):
        fixture = bundle_fixtures.CommunityBundleTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(20):
                (root / f"{index}.json").write_text("{}")
            paths = semantic_files([root], root / "output", max_files=1)
            self.assertTrue(next(paths).is_file())
            with self.assertRaisesRegex(CommunityExportError, "FILE_LIMIT_EXCEEDED"):
                next(paths)
            with self.assertRaisesRegex(CommunityExportError, "ENTRY_LIMIT_EXCEEDED"):
                list(semantic_files([root], root / "output", max_files=100, max_entries=2))
            source = root / "jsonl"
            source.mkdir()
            (source / "memory.jsonl").write_text(json.dumps(fixture._document()) + "\n")
            report = export_community_knowledge(source_roots=[source], output_root=root / "output")
            self.assertEqual(report["record_count"], 1)
