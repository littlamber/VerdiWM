from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from wmloop.archive.store import ArchiveStore, ContentAddressedStore, SettledTrialRecord
from wmloop.archive.community_projection import project_archive_evidence, EvidenceProjectionError
from wmloop.experiments.community_bundle import publish_community_bundle, verify_community_bundle
import tests.test_community_bundle as bundle_fixtures


class ArchiveCommunityProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.database = self.root / "archive.db"
        self.cas = ContentAddressedStore(self.root)
        self.archive = ArchiveStore(self.database)
        self.verdicts = []
        for index, decision in enumerate(("ACCEPT", "REJECT", "INCONCLUSIVE", "VOID")):
            proposal = f"proposal-{index}"
            verdict = self.cas.put_bytes(json.dumps({"proposal_id": proposal, "gates": {"G1_readonly": True, "G2_heldout": True, "G3_audit": True, "G4_replication": True}, "action_following_gate": {"enabled": False, "threshold": 0, "observed": 0, "pass": True}, "delta_m_ver": {}, "negative_metrics": {}, "verdict": decision, "violation": None}).encode(), media_type="application/json")
            receipt = self.cas.put_bytes(f"settled receipt {index}".encode(), media_type="text/plain")
            context = self.cas.put_bytes(b"private raw context", media_type="text/plain")
            self.verdicts.append(verdict)
            self.archive.record_settled_trial(SettledTrialRecord(trial_id=f"trial-{index}", proposal_id=proposal, goal_id="goal", library_version="v1", failure_context_ref=context.uri, verdict_ref=verdict.uri, receipt_ref=receipt.uri, gpu_hours=0.1, hypothesis_hash="a" * 64, impl_diff_hash="b" * 64, evaluator_hash="c" * 64, settlement_state="settled", receipt_hash=receipt.sha256))

    def project(self):
        return project_archive_evidence(archive=self.database, cas=self.root, output_root=self.root / "export")

    def test_all_verdicts_project_without_raw_data_or_transfer_claims(self):
        before = hashlib.sha256(self.database.read_bytes()).hexdigest()
        manifest = self.project()
        self.assertEqual(manifest["record_count"], 4)
        self.assertEqual(before, hashlib.sha256(self.database.read_bytes()).hexdigest())
        records = [json.loads(p.read_text()) for p in (self.root / "export/records").glob("*.json")]
        self.assertEqual({r["decision"] for r in records}, {"ACCEPT", "REJECT", "INCONCLUSIVE", "VOID"})
        self.assertTrue(all(r["claim_scope"] == "settlement_only" for r in records))
        self.assertNotIn(str(self.root), json.dumps(records))
        self.assertNotIn("private raw context", json.dumps(records))
        key, public = bundle_fixtures.CommunityBundleTests()._keypair(self.root)
        publish_community_bundle(documents=records, output_root=self.root / "bundle", publisher_id="test/community", signing_key=key)
        self.assertEqual(verify_community_bundle(self.root / "bundle", public_key=public)["record_count"], 4)
        self.assertEqual(self.project()["export_id"], manifest["export_id"])

    def test_tampered_cas_and_partial_settlement_fail_before_export(self):
        path = self.verdicts[0].path
        original = path.read_bytes()
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(EvidenceProjectionError, "DIGEST_MISMATCH"):
            self.project()
        self.assertFalse((self.root / "export").exists())
        path.write_bytes(original)
        with sqlite3.connect(self.database) as connection:
            connection.execute("DELETE FROM receipts WHERE trial_id='trial-0'")
        with self.assertRaisesRegex(EvidenceProjectionError, "SETTLEMENT_BINDING_INVALID"):
            self.project()

    def test_trial_limit_is_enforced(self):
        with self.assertRaisesRegex(EvidenceProjectionError, "TRIAL_LIMIT_EXCEEDED"):
            project_archive_evidence(archive=self.database, cas=self.root, output_root=self.root / "export", max_trials=1)
