"""Structured JSONL observability and live telemetry summary for Atlas.

Every request passing through app/server.py is recorded as one JSON line with:
- transport endpoint, session id, turn index, active defence level, model name
- prompt, reply, latency, HTTP status, token counts
- input_guard / output_guard triggers
- structured tool calls executed during that turn
- ground-truth oracle flags fired on that turn (delta) and cumulatively in session

The resulting JSONL joins 1:1 with sixi-scanner's --log-jsonl trace so
precision, recall, false-positive attribution, and defence-ladder comparisons
can be computed directly from raw wire logs.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ORACLE_KEYS: tuple[str, ...] = (
    "leaked",
    "privileged_action",
    "exfiltrated",
    "unauthorized_recipient",
    "unauthorized_account",
    "taint_relayed",
    "cap_split",
)


@dataclass
class TurnRecord:
    ts: str
    run_id: str
    endpoint: str
    session_id: str
    turn_index: int
    defence: str
    model: str
    prompt: str
    reply: str
    reply_chars: int
    latency_ms: int
    status: int
    input_guard_blocked: bool = False
    output_guard_withheld: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    oracles_fired: list[str] = field(default_factory=list)
    session_oracles: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str = ""


def _percentile(sorted_vals: list[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    idx = round((pct / 100.0) * (len(sorted_vals) - 1))
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return sorted_vals[idx]


class TurnLogger:
    """Thread-safe JSONL writer and live in-memory telemetry aggregator."""

    def __init__(self, log_path: str | Path | None = None, run_id: str | None = None):
        self.run_id = run_id or f"atlas-{uuid.uuid4().hex[:8]}"
        self.log_path = Path(log_path) if log_path else None
        self.started_at = time.monotonic()
        self._lock = threading.Lock()
        self._records: list[TurnRecord] = []

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Touch with owner-only permissions (0600) because logs hold prompts & replies.
            fd = os.open(
                self.log_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            os.close(fd)

    def record(self, rec: TurnRecord) -> None:
        line = json.dumps(asdict(rec), ensure_ascii=False)
        with self._lock:
            self._records.append(rec)
            if self.log_path is not None:
                with open(self.log_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self.started_at = time.monotonic()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            recs = list(self._records)

        latencies = sorted(r.latency_ms for r in recs)
        by_endpoint: Counter[str] = Counter()
        by_defence: Counter[str] = Counter()
        tool_counts: Counter[str] = Counter()
        oracle_turn_counts: Counter[str] = Counter()
        oracle_session_sets: dict[str, set[str]] = {k: set() for k in ORACLE_KEYS}
        sessions: set[str] = set()

        errors = 0
        input_blocked = 0
        output_withheld = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        turns_with_oracle = 0

        for r in recs:
            sessions.add(r.session_id)
            by_endpoint[r.endpoint] += 1
            by_defence[r.defence] += 1
            if r.error or r.status >= 400:
                errors += 1
            if r.input_guard_blocked:
                input_blocked += 1
            if r.output_guard_withheld:
                output_withheld += 1
            prompt_tokens += r.prompt_tokens
            completion_tokens += r.completion_tokens
            total_tokens += r.total_tokens
            for tc in r.tool_calls:
                name = str(tc.get("name") or "unknown")
                tool_counts[name] += 1
            if r.oracles_fired:
                turns_with_oracle += 1
                for o in r.oracles_fired:
                    oracle_turn_counts[o] += 1
            for o in r.session_oracles:
                if o in oracle_session_sets:
                    oracle_session_sets[o].add(r.session_id)

        return {
            "run_id": self.run_id,
            "uptime_s": round(time.monotonic() - self.started_at, 2),
            "log_file": str(self.log_path) if self.log_path else "",
            "turns_total": len(recs),
            "unique_sessions": len(sessions),
            "errors": errors,
            "latency_ms": {
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
                "max": latencies[-1] if latencies else 0,
            },
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens,
            },
            "guards": {
                "input_blocked": input_blocked,
                "output_withheld": output_withheld,
            },
            "tool_calls": dict(tool_counts),
            "turns_with_oracle": turns_with_oracle,
            "oracles_by_turn": dict(oracle_turn_counts),
            "oracles_by_session": {
                k: len(v) for k, v in oracle_session_sets.items() if v
            },
            "by_endpoint": dict(by_endpoint),
            "by_defence": dict(by_defence),
        }


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
