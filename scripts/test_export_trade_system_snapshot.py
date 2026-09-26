import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("export_snapshot", Path(__file__).with_name("export-trade-system-snapshot.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SnapshotExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.journal = self.root / "state" / "journal.jsonl"
        self.journal.parent.mkdir()
        self.row = {
            "schema_version": 2, "date": "2020-01-02", "run_at_utc": "2020-01-02T21:00:00+00:00",
            "realized_pnl": -10, "unrealized_pnl": 2, "open_positions": 1, "closed_trades": 1,
            "trading_authority": {"paper_entries_enabled": False, "live_execution_enabled": False},
            "recommendations": [], "position_observations": [],
            "private_debug_payload": "must-not-be-published",
        }

    def save(self):
        self.journal.write_text(json.dumps(self.row) + "\n")

    def test_read_only_whitelist_and_observation_clock(self):
        self.save()
        original = self.journal.read_bytes()
        out = self.root / "public" / "snapshot.json"
        result = module.export(self.journal, out)
        self.assertEqual(self.journal.read_bytes(), original)
        self.assertEqual(result["observed_at"], self.row["run_at_utc"])
        self.assertNotEqual(result["observed_at"], result["generated_at"])
        self.assertNotIn("must-not-be-published", out.read_text())
        self.assertNotIn(str(self.root), out.read_text())
        self.assertEqual(result["source"]["journal_sha256"], hashlib.sha256(original).hexdigest())

    def test_nonfinite_or_missing_metrics_do_not_replace_existing_output(self):
        out = self.root / "public.json"
        out.write_text("previous valid artifact")
        for value in [float("nan"), float("inf"), None]:
            self.row["realized_pnl"] = value
            self.save()
            with self.assertRaises(ValueError):
                module.export(self.journal, out)
            self.assertEqual(out.read_text(), "previous valid artifact")

    def test_missing_quote_clock_is_not_inferred_from_contract_time(self):
        row = {"ticker": "SPY", "option_type": "call", "side": "buy", "strike": 100,
               "expiry": "2020-02-01", "bid": 1, "ask": 2,
               "contract_as_of": "2020-01-02T21:00:00+00:00"}
        self.assertIsNone(module.leg(row)["source_quote_at"])

    def test_input_hash_is_checked_and_corruption_rejected(self):
        content = b'{"selected_inputs":true}'
        digest = hashlib.sha256(content).hexdigest()
        relative = f"input_snapshots/{digest}.json"
        source = self.journal.parent / relative
        source.parent.mkdir()
        source.write_bytes(content)
        self.row["input_snapshot"] = {"path": relative, "sha256": digest}
        self.save()
        self.assertTrue(module.build_snapshot(self.journal)["source"]["input_snapshot_verified"])
        source.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "content does not match"):
            module.build_snapshot(self.journal)

    def test_incomplete_journal_or_missing_authority_does_not_export(self):
        self.save()
        self.journal.write_text(self.journal.read_text() + '{"partial":')
        with self.assertRaises(json.JSONDecodeError):
            module.build_snapshot(self.journal)
        del self.row["trading_authority"]
        self.save()
        with self.assertRaisesRegex(ValueError, "authority"):
            module.build_snapshot(self.journal)

    def test_export_cannot_overwrite_source_state(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "outside the source"):
            module.export(self.journal, self.journal.parent / "portfolio.json")


if __name__ == "__main__":
    unittest.main()
