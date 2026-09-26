#!/usr/bin/env python3
"""Export a public, read-only view of the MVP paper journal; never run a scan.

Usage: python3 scripts/export-trade-system-snapshot.py --journal /path/to/
       optionAgents/logs/mvp_control/journal.jsonl --output public/data/trade-system-snapshot.json
Only explicitly selected fields are published. Source files are never modified.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile


def numeric(value, *, nullable=False, integer=False, minimum=None, maximum=None):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite number")
    if integer and not isinstance(value, int):
        raise ValueError("Expected an integer")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError("Number is outside the supported range")
    return value


def label(value):
    # These fields are identifiers/reason codes, never free-form log messages.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_. -]{1,160}", value):
        raise ValueError("Invalid public identifier")
    return value


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Missing timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include its timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def day(value):
    return date.fromisoformat(value).isoformat()


def flag(value):
    if type(value) is not bool:
        raise ValueError("Missing authority flag")
    return value


def leg(row):
    sources = sorted({label(value["source"]) for value in row.get("source_provenance", {}).values()
                      if isinstance(value, dict) and value.get("source")})
    return {
        **{key: label(row[key]) for key in ("ticker", "option_type", "side")},
        "strike": numeric(row["strike"], minimum=0), "expiry": day(row["expiry"]),
        "bid": numeric(row.get("bid"), nullable=True, minimum=0),
        "ask": numeric(row.get("ask"), nullable=True, minimum=0),
        # Never substitute contract_as_of or the export/run time for quote time.
        "source_quote_at": timestamp(row["source_quote_at"]) if row.get("source_quote_at") else None,
        "quote_sources": sources,
    }


def recommendation(row):
    return {
        **{key: label(row[key]) for key in ("ticker", "strategy", "agent")},
        "rank": numeric(row["rank"], integer=True, minimum=1),
        "confidence": numeric(row["confidence"], minimum=0, maximum=1),
        "score": numeric(row["score"]), "expiry": day(row["expiry"]),
        "entry_net_debit": numeric(row.get("entry_net_debit"), nullable=True, minimum=0),
        "entry_net_credit": numeric(row.get("entry_net_credit"), nullable=True, minimum=0),
        "legs": [leg(item) for item in row.get("legs", [])],
    }


def position(row):
    return {
        **{key: label(row[key]) for key in ("ticker", "strategy", "action", "reason")},
        "contracts": numeric(row["contracts"], integer=True, minimum=0),
        "expiry": day(row["expiry"]), "entry_price": numeric(row["entry_price"], minimum=0),
        "liquidation_mark": numeric(row.get("liquidation_mark"), nullable=True),
        "unrealized_pnl": numeric(row.get("unrealized_pnl"), nullable=True),
        "legs": [leg(item) for item in row.get("legs", [])],
    }


def reject_constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


def build_snapshot(journal: Path, *, now: datetime | None = None):
    raw = journal.read_bytes()
    rows = [json.loads(line, parse_constant=reject_constant) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError("The journal has no observations")
    latest = rows[-1]
    if latest.get("schema_version") != 2:
        raise ValueError("The latest observation does not have the required schema version 2")
    observed = timestamp(latest["run_at_utc"])
    now = now or datetime.now(timezone.utc)
    if datetime.fromisoformat(observed) > now:
        raise ValueError("The latest observation is in the future")
    reference = latest.get("input_snapshot") or {}
    digest = reference.get("sha256")
    verified = False
    if digest is not None:
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid decision-input hash")
        relative = reference.get("path", "")
        if relative != f"input_snapshots/{digest}.json":
            raise ValueError("Unexpected decision-input path")
        source = journal.parent / relative
        if source.is_file():
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError("Decision-input content does not match its hash")
            verified = True
    authority = latest.get("trading_authority") or {}
    return {
        "schema_version": 1, "mode": "paper_observation",
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "observed_at": observed, "session_date": day(latest["date"]),
        "source": {
            "journal": "logs/mvp_control/journal.jsonl",
            "journal_sha256": hashlib.sha256(raw).hexdigest(),
            "record_count": len(rows), "input_snapshot_sha256": digest,
            "input_snapshot_verified": verified, "raw_lineage": "unavailable",
        },
        "authority": {key: flag(authority.get(key)) for key in ("paper_entries_enabled", "live_execution_enabled")},
        "metrics": {
            **{key: numeric(latest[key]) for key in ("realized_pnl", "unrealized_pnl")},
            **{key: numeric(latest[key], integer=True, minimum=0) for key in ("open_positions", "closed_trades")},
        },
        "recommendations": [recommendation(row) for row in latest.get("recommendations", [])],
        "position_observations": [position(row) for row in latest.get("position_observations", [])],
    }


def export(journal: Path, output: Path):
    if output.resolve().is_relative_to(journal.parent.resolve()):
        raise ValueError("Output must be outside the source state directory")
    snapshot = build_snapshot(journal)
    content = json.dumps(snapshot, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=".paper-snapshot-", delete=False) as file:
        file.write(content)
        temporary = Path(file.name)
    temporary.replace(output)
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("public/data/trade-system-snapshot.json"))
    args = parser.parse_args()
    snapshot = export(args.journal, args.output)
    print(f"Exported {snapshot['source']['record_count']} observations; latest {snapshot['observed_at']} to {args.output}")
