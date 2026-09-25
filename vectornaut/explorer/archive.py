# -*- coding: utf-8 -*-
"""
Quality-diversity archive (MAP-Elites style) kept as one JSON file per profile.

Layout of ``archive.json``::

    version, profile, axes, queries
    entries      {entry_id: entry}      every candidate ever proposed, including failures
    elites       {cell_key: entry_id}   best evaluated entry per full cell
    proposals    {cell_key: count}      where candidates landed (all statuses): the
                                        "proposal density" map
    targets      {cell_key: count}      how often a search order aimed at a cell/pattern
    infeasible   {pattern_key: {...}}   cells or patterns reported as impossible
    rounds       [round log]            orders, outcomes and strategy yield per round
    round_counter, next_entry

Writes are atomic (temp file + ``os.replace``). Entry ids are sequential and the file is
written with sorted keys, so the same inputs give the same file except for timestamps
(pass ``clock`` to freeze those in tests).
"""
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional

from vectornaut.explorer.descriptors import DescriptorSpace
from vectornaut.storage import data_path

ARCHIVE_VERSION = 1

# Entry statuses.
EVALUATED = "evaluated"          # scored; may be an elite
FAILED = "failed"                # evaluator error / pipeline failure
INFEASIBLE = "infeasible"        # generator or evaluator said the cell is impossible
INVALID = "invalid"              # descriptors did not fit the axes (not placed)
REJECTED = "rejected"            # failed sanity checks before evaluation

# Outcomes of adding an entry (strategy yield statistics).
NEW_ELITE = "new_elite"
IMPROVED = "improved"
NOT_BETTER = "not_better"


def default_archive_path(profile: str, name: Optional[str] = None) -> str:
    if name:
        return data_path("explorer", profile, name, "archive.json")
    return data_path("explorer", profile, "archive.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class Archive:
    def __init__(
        self,
        profile: str,
        space: DescriptorSpace,
        path: Optional[str] = None,
        clock: Optional[Callable[[], str]] = None,
    ):
        self.profile = profile
        self.space = space
        self.path = os.path.abspath(path or default_archive_path(profile))
        self.clock = clock or _utc_now
        self.data: Dict[str, Any] = self._empty()

    def _empty(self) -> Dict[str, Any]:
        return {
            "version": ARCHIVE_VERSION,
            "profile": self.profile,
            "axes": self.space.to_dict(),
            "queries": [],
            "entries": {},
            "elites": {},
            "proposals": {},
            "targets": {},
            "infeasible": {},
            "rounds": [],
            "round_counter": 0,
            "next_entry": 1,
        }

    # ---- persistence --------------------------------------------------
    @property
    def directory(self) -> str:
        return os.path.dirname(self.path)

    @property
    def results_dir(self) -> str:
        return os.path.join(self.directory, "results")

    @classmethod
    def load(cls, profile: str, space: DescriptorSpace, path: Optional[str] = None,
             clock: Optional[Callable[[], str]] = None) -> "Archive":
        archive = cls(profile, space, path=path, clock=clock)
        if os.path.exists(archive.path):
            with open(archive.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("profile") != profile:
                raise ValueError(f"Archive {archive.path} belongs to profile '{data.get('profile')}', not '{profile}'.")
            stored_axes = [axis.get("name") for axis in data.get("axes", [])]
            if stored_axes != space.names:
                raise ValueError(
                    f"Archive {archive.path} was built with axes {stored_axes}; the profile now has {space.names}. "
                    "Use a new archive name."
                )
            base = archive._empty()
            base.update(data)
            archive.data = base
        return archive

    def save(self) -> str:
        os.makedirs(self.directory, exist_ok=True)
        payload = json.dumps(self.data, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        fd, tmp_path = tempfile.mkstemp(prefix=".archive-", suffix=".json", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        return self.path

    def write_result(self, entry_id: str, raw: Any) -> str:
        """Stores a raw evaluation result next to the archive and returns its absolute path."""
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, f"{entry_id}.json")
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        os.replace(tmp_path, path)
        return path

    # ---- queries ------------------------------------------------------
    @property
    def entries(self) -> Dict[str, Dict[str, Any]]:
        return self.data["entries"]

    @property
    def elites(self) -> Dict[str, str]:
        return self.data["elites"]

    @property
    def round_counter(self) -> int:
        return int(self.data.get("round_counter", 0))

    def elite_entries(self) -> List[Dict[str, Any]]:
        """Elites sorted by cell key (deterministic order)."""
        return [self.entries[self.elites[key]] for key in sorted(self.elites)]

    def elite_for(self, cell: Mapping[str, str]) -> Optional[Dict[str, Any]]:
        entry_id = self.elites.get(self.space.cell_key(cell))
        return self.entries.get(entry_id) if entry_id else None

    def proposal_count(self, cell: Mapping[str, str]) -> int:
        return int(self.data["proposals"].get(self.space.cell_key(cell), 0))

    def target_count(self, cell: Mapping[str, str]) -> int:
        return int(self.data["targets"].get(self.space.cell_key(cell), 0))

    def infeasible_reason(self, cell: Mapping[str, str]) -> Optional[str]:
        """Reason if the full cell matches any infeasible cell or pattern."""
        for key in sorted(self.data["infeasible"]):
            pattern = self.space.parse_key(key)
            if self.space.matches(cell, pattern):
                return self.data["infeasible"][key].get("reason") or "reported infeasible"
        return None

    def is_infeasible(self, cell: Mapping[str, str]) -> bool:
        return self.infeasible_reason(cell) is not None

    def titles(self) -> List[str]:
        return [entry.get("title") or "" for _, entry in sorted(self.entries.items())]

    def score_range(self) -> Optional[tuple]:
        scores = [entry["score"] for entry in self.elite_entries() if entry.get("score") is not None]
        return (min(scores), max(scores)) if scores else None

    # ---- updates ------------------------------------------------------
    def register_query(self, query: str) -> None:
        if query and query not in self.data["queries"]:
            self.data["queries"].append(query)

    def next_round(self) -> int:
        self.data["round_counter"] = self.round_counter + 1
        return self.data["round_counter"]

    def count_target(self, target: Mapping[str, str]) -> None:
        key = self.space.cell_key(target)
        self.data["targets"][key] = int(self.data["targets"].get(key, 0)) + 1

    def mark_infeasible(self, pattern: Mapping[str, str], reason: str, *, entry_id: Optional[str],
                        round_no: int, source: str) -> str:
        key = self.space.cell_key(pattern)
        record = self.data["infeasible"].get(key)
        if record is None:
            record = {"reason": reason, "entry_ids": [], "round": round_no, "source": source, "count": 0}
            self.data["infeasible"][key] = record
        record["count"] = int(record.get("count", 0)) + 1
        if entry_id and entry_id not in record["entry_ids"]:
            record["entry_ids"].append(entry_id)
        return key

    def add_entry(
        self,
        *,
        round_no: int,
        run_id: str,
        order: Mapping[str, Any],
        title: str,
        concept: Mapping[str, Any],
        descriptors: Optional[Mapping[str, str]],
        status: str,
        score: Optional[float] = None,
        score_breakdown: Optional[Mapping[str, Any]] = None,
        reason: str = "",
        raw_result: Any = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Records one candidate. Returns the stored entry; ``entry["outcome"]`` says
        whether it became a new elite, improved a cell, or neither. An elite is only
        replaced by a strictly higher score (ties keep the older entry).
        """
        entry_id = f"e{int(self.data['next_entry']):05d}"
        self.data["next_entry"] = int(self.data["next_entry"]) + 1
        cell_key = self.space.cell_key(descriptors) if descriptors else None
        target = order.get("target") or {}
        on_target = None
        if descriptors is not None:
            on_target = self.space.matches(descriptors, target)
        now = self.clock()
        entry: Dict[str, Any] = {
            "id": entry_id,
            "profile": self.profile,
            "run_id": run_id,
            "round": round_no,
            "order_id": order.get("order_id"),
            "strategy": order.get("strategy"),
            "target_key": self.space.cell_key(target),
            "on_target": on_target,
            "parent_ids": list(order.get("parent_ids") or []),
            "title": title,
            "concept": dict(concept or {}),
            "descriptors": dict(descriptors) if descriptors else None,
            "cell": cell_key,
            "status": status,
            "score": _finite(score) if status == EVALUATED else None,
            "score_breakdown": dict(score_breakdown or {}),
            "reason": reason,
            "outcome": None,
            "created_at": now,
            "updated_at": now,
            "raw_result_path": None,
        }
        if extra:
            entry.update(dict(extra))
        if raw_result is not None:
            # Relative to the archive folder, so the archive stays portable.
            entry["raw_result_path"] = os.path.relpath(self.write_result(entry_id, raw_result), self.directory)

        if cell_key is not None:
            self.data["proposals"][cell_key] = int(self.data["proposals"].get(cell_key, 0)) + 1

        if status == EVALUATED and entry["score"] is not None and cell_key is not None:
            current_id = self.elites.get(cell_key)
            current = self.entries.get(current_id) if current_id else None
            if current is None:
                entry["outcome"] = NEW_ELITE
                self.elites[cell_key] = entry_id
            elif entry["score"] > float(current.get("score") or 0.0):
                entry["outcome"] = IMPROVED
                entry["replaced"] = current_id
                self.elites[cell_key] = entry_id
            else:
                entry["outcome"] = NOT_BETTER
        self.entries[entry_id] = entry
        return entry

    def log_round(self, record: Mapping[str, Any]) -> None:
        self.data["rounds"].append(dict(record))
