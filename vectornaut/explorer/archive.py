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
    request_analysis                    requirements extracted from the query (fixed once set),
                                        the objective and baseline statements (fixed once set)
                                        and the relevant values of "relevance axes"
    rounds       [round log]            orders, outcomes and strategy yield per round
    round_counter, next_entry

Elites are compared by evidence first, then score: an entry's ``score_breakdown`` may carry
an ``evidence_rank`` (materials: 2 = simulated, 1 = estimated, 0 = estimated after an
implausible simulation). A lower-ranked entry never replaces a higher-ranked elite, and a
higher-ranked entry replaces a lower-ranked one regardless of score. Entries without a rank
(business) all rank equal, so only the score decides. Scores within TIE_EPSILON points are a
tie when both entries carry a ``tiebreak`` key (materials: critic objective gain, fewer killer
risks, simulated benefit used): the better key wins (``tiebreak: improved_on_tiebreak`` on the
new entry, or ``kept_on_tiebreak`` when a slightly higher score loses); equal keys fall back to
"strictly higher score".

Versioning: ``version`` is ARCHIVE_VERSION. An older archive whose axes (names *and* values)
equal the profile's is migrated on load (the version is raised and ``migrated_from`` noted;
the profile may re-score entries on the way, see ``Archive.load(migrate=...)``); an archive
built with a different vocabulary is refused with an explanation, because its cells and scores
are not comparable (version 3 added ``governing_quantity`` values for fouling control and changed
the materials score; version 4 changed the materials score again, re-scores version-3 entries
from their stored breakdowns and adds preferred axis values). A newer archive is refused as well.

Writes are atomic (temp file + ``os.replace``). Entry ids are sequential and the file is
written with sorted keys, so the same inputs give the same file except for timestamps
(pass ``clock`` to freeze those in tests).
"""
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from vectornaut.explorer.descriptors import DescriptorSpace
from vectornaut.storage import data_path

ARCHIVE_VERSION = 4
# Two scores within this many points are a tie within a cell (decided by the tiebreak key).
TIE_EPSILON = 1.0


class ArchiveCompatibilityError(ValueError):
    """The stored archive cannot be used with the current profile (version or vocabulary)."""

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


def evidence_rank(entry: Mapping[str, Any]) -> int:
    """Evidence rank of an entry (0 when the profile does not rank evidence)."""
    try:
        return int((entry.get("score_breakdown") or {}).get("evidence_rank") or 0)
    except (TypeError, ValueError):
        return 0


def rank_key(entry: Mapping[str, Any]) -> tuple:
    """(evidence rank, score): the larger key is the better entry."""
    return (evidence_rank(entry), float(entry.get("score") or 0.0))


# Outcomes of a tie decided by the tiebreak key (stored as entry["tiebreak"]).
IMPROVED_ON_TIEBREAK = "improved_on_tiebreak"
KEPT_ON_TIEBREAK = "kept_on_tiebreak"


def tiebreak_key(entry: Mapping[str, Any]) -> Optional[tuple]:
    """The entry's tiebreak key (larger is better; missing numbers rank lowest), or None."""
    values = (entry.get("score_breakdown") or {}).get("tiebreak")
    if not isinstance(values, (list, tuple)) or not values:
        return None
    return tuple(_finite(v) if _finite(v) is not None else -math.inf for v in values)


def compare_entries(new: Mapping[str, Any], current: Mapping[str, Any],
                    epsilon: float = TIE_EPSILON) -> Tuple[bool, bool]:
    """
    (new is better, decided by the tiebreak?). Evidence rank first; at equal rank, scores within
    ``epsilon`` points are a tie decided by the tiebreak keys when both entries have one and the
    keys differ; otherwise the strictly higher score wins (ties keep the current entry).
    """
    rank_new, rank_cur = evidence_rank(new), evidence_rank(current)
    if rank_new != rank_cur:
        return rank_new > rank_cur, False
    score_new, score_cur = float(new.get("score") or 0.0), float(current.get("score") or 0.0)
    if abs(score_new - score_cur) <= epsilon:
        key_new, key_cur = tiebreak_key(new), tiebreak_key(current)
        if key_new is not None and key_cur is not None and key_new != key_cur:
            return key_new > key_cur, True
    return score_new > score_cur, False


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
            "request_analysis": {"requirements": [], "requirements_round": None, "relevant": {},
                                 "objective_statement": "", "baseline_statement": "", "framing_round": None,
                                 "preferred": {}, "preferred_reason": {}},
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
             clock: Optional[Callable[[], str]] = None,
             migrate: Optional[Callable[["Archive", int], Dict[str, Any]]] = None) -> "Archive":
        """
        Loads (or starts) an archive. ``migrate(archive, from_version)`` is called after an older,
        vocabulary-compatible archive was loaded (the profile's re-scoring); a summary it returns
        is appended to ``migrations``.
        """
        archive = cls(profile, space, path=path, clock=clock)
        if os.path.exists(archive.path):
            with open(archive.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("profile") != profile:
                raise ValueError(f"Archive {archive.path} belongs to profile '{data.get('profile')}', not '{profile}'.")
            stored_axes = [axis.get("name") for axis in data.get("axes", [])]
            if stored_axes != space.names:
                raise ArchiveCompatibilityError(
                    f"Archive {archive.path} was built with axes {stored_axes}; the profile now has {space.names}. "
                    "Use a new archive name."
                )
            data = archive._check_version(data)
            base = archive._empty()
            analysis = dict(base["request_analysis"])
            base.update(data)
            analysis.update(base.get("request_analysis") or {})
            base["request_analysis"] = analysis
            archive.data = base
            old_version = data.get("migrated_now")
            base.pop("migrated_now", None)
            if old_version is not None and migrate is not None:
                summary = migrate(archive, int(old_version))
                if summary:
                    base.setdefault("migrations", []).append(dict(summary, from_version=int(old_version),
                                                                  to_version=ARCHIVE_VERSION))
        return archive

    def recompute_elites(self) -> Dict[str, Dict[str, Any]]:
        """
        Rebuilds the elite of every cell from the evaluated entries in id order with the current
        comparison rule (after a re-scoring). Returns the changed cells as
        {cell: {"from": old id, "to": new id, "tiebreak": decided by the tiebreak key?}}.
        Historical ``outcome`` fields of the entries are left as recorded.
        """
        old = dict(self.elites)
        elites: Dict[str, str] = {}
        by_tiebreak: Dict[str, bool] = {}
        for entry_id in sorted(self.entries):
            entry = self.entries[entry_id]
            cell = entry.get("cell")
            if entry.get("status") != EVALUATED or entry.get("score") is None or not cell:
                continue
            current = self.entries.get(elites.get(cell, ""))
            if current is None:
                elites[cell], by_tiebreak[cell] = entry_id, False
                continue
            better, tiebreak = compare_entries(entry, current)
            if better:
                elites[cell], by_tiebreak[cell] = entry_id, tiebreak
        self.data["elites"] = elites
        return {cell: {"from": old.get(cell), "to": new, "tiebreak": by_tiebreak[cell]}
                for cell, new in sorted(elites.items()) if old.get(cell) != new}

    def _check_version(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Refuses newer archives and archives with another vocabulary; migrates older compatible ones."""
        try:
            version = int(data.get("version") or 1)
        except (TypeError, ValueError):
            version = 1
        if version > ARCHIVE_VERSION:
            raise ArchiveCompatibilityError(
                f"Archive {self.path} has version {version}, newer than this explorer (version {ARCHIVE_VERSION}). "
                "Update Vectornaut or use another archive (--archive NAME)."
            )
        current = {axis["name"]: list(axis["values"]) for axis in self.space.to_dict()}
        changes = []
        for axis in data.get("axes", []):
            name, stored = axis.get("name"), list(axis.get("values") or [])
            wanted = current.get(name, [])
            if stored == wanted:
                continue
            added = [v for v in wanted if v not in stored]
            removed = [v for v in stored if v not in wanted]
            detail = (["new values " + ", ".join(added)] if added else []) + \
                     (["removed values " + ", ".join(removed)] if removed else [])
            changes.append(f"{name}: " + ("; ".join(detail) or "values reordered"))
        if changes:
            raise ArchiveCompatibilityError(
                f"Archive {self.path} (version {version}) was built with a different descriptor vocabulary "
                f"({'; '.join(changes)}). Its cells and scores are not comparable with the current profile "
                f"(archive version {ARCHIVE_VERSION}), so it is not migrated. Start a new map with --archive NAME, "
                "or move the old archive folder away."
            )
        if version < ARCHIVE_VERSION:
            data = dict(data)
            data["migrated_from"] = version
            data["migrated_now"] = version
            data["version"] = ARCHIVE_VERSION
        return data

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

    def ranked_elites(self) -> List[Dict[str, Any]]:
        """Elites best first: evidence rank, then score, then id."""
        return sorted(self.elite_entries(), key=lambda e: (-evidence_rank(e), -float(e.get("score") or 0.0), e["id"]))

    # ---- request analysis ----------------------------------------------
    @property
    def request_analysis(self) -> Dict[str, Any]:
        return self.data["request_analysis"]

    @property
    def requirements(self) -> List[Dict[str, str]]:
        return list(self.request_analysis.get("requirements") or [])

    def set_requirements(self, requirements: Sequence[Mapping[str, Any]], round_no: int) -> bool:
        """
        Stores the requirement list once (the first non-empty one), so the scores of later
        rounds stay comparable. Returns True if the list was stored now.
        """
        if self.requirements:
            return False
        cleaned, seen = [], set()
        for item in requirements or []:
            name = str(item.get("name") or "").strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            cleaned.append({"name": name, "criterion": str(item.get("criterion") or "").strip()})
        if not cleaned:
            return False
        self.request_analysis["requirements"] = cleaned
        self.request_analysis["requirements_round"] = round_no
        return True

    @property
    def objective_statement(self) -> str:
        return str(self.request_analysis.get("objective_statement") or "")

    @property
    def baseline_statement(self) -> str:
        return str(self.request_analysis.get("baseline_statement") or "")

    def set_framing(self, objective: str, baseline: str, round_no: int) -> Dict[str, bool]:
        """
        Stores the objective statement (the request's main benefit as it applies over the stated
        service life and conditions) and the baseline statement (the conventional solution in the
        same condition). Each is stored once, the first non-empty one, like the requirements, so
        the scores of later rounds stay comparable. Returns which of the two were stored now.
        """
        stored = {"objective": False, "baseline": False}
        for key, value, flag in (("objective_statement", objective, "objective"),
                                 ("baseline_statement", baseline, "baseline")):
            text = " ".join(str(value or "").split())
            if text and not self.request_analysis.get(key):
                self.request_analysis[key] = text
                stored[flag] = True
        if any(stored.values()) and self.request_analysis.get("framing_round") is None:
            self.request_analysis["framing_round"] = round_no
        return stored

    def stated_relevant_values(self, axis: str) -> List[str]:
        """Relevant values named by the function analysis only (without the values of elites)."""
        return list((self.request_analysis.get("relevant") or {}).get(axis) or [])

    def source_uses(self, exclude_strategies: Sequence[str] = ("refine",)) -> Dict[str, int]:
        """
        How often each entry served as the source of a later order: the first parent of every
        entry (fill_gap/diversify/extrapolate list further neighbours only as context), both
        parents of a combine entry. Refinements are counted separately by the refine strategy.
        """
        uses: Dict[str, int] = {}
        for entry in self.entries.values():
            strategy = entry.get("strategy")
            if strategy in exclude_strategies:
                continue
            parents = list(entry.get("parent_ids") or [])
            for parent in parents[:2] if strategy == "combine" else parents[:1]:
                uses[parent] = uses.get(parent, 0) + 1
        return uses

    def declare_relevance_axes(self, axes: Sequence[str]) -> None:
        """Axes whose values must be relevant to the request (materials: governing_quantity)."""
        relevant = self.request_analysis.setdefault("relevant", {})
        for name in axes or ():
            relevant.setdefault(name, [])

    def relevance_axes(self) -> List[str]:
        relevant = self.request_analysis.get("relevant") or {}
        return [name for name in self.space.names if name in relevant]

    def add_relevant_values(self, axis: str, values: Sequence[str]) -> List[str]:
        """Adds already validated values to an axis' relevant set; returns the new ones."""
        stored = self.request_analysis.setdefault("relevant", {}).setdefault(axis, [])
        added = []
        for value in values or ():
            if value not in stored:
                stored.append(value)
                added.append(value)
        stored.sort(key=self.space.axis(axis).index)
        return added

    def relevant_values(self, axis: str) -> Optional[List[str]]:
        """
        Relevant values of a relevance axis: those named by the generator's function analysis
        plus those held by elites. None if the axis is not a relevance axis; an empty list if
        nothing is known yet.
        """
        relevant = self.request_analysis.get("relevant") or {}
        if axis not in relevant:
            return None
        values = set(relevant.get(axis) or [])
        values.update(e["descriptors"][axis] for e in self.elite_entries())
        return sorted(values, key=self.space.axis(axis).index)

    def is_relevant(self, cell: Mapping[str, str]) -> bool:
        """False if the cell uses a value of a relevance axis that is known to be irrelevant.
        While nothing is known about an axis, every value counts as relevant here."""
        for axis in self.relevance_axes():
            if axis not in cell:
                continue
            values = self.relevant_values(axis)
            if values and cell[axis] not in values:
                return False
        return True

    # ---- preferred values (e.g. biological origins for a bio-inspired request) ----
    def set_preferred_values(self, axis: str, values: Sequence[str], reason: str) -> bool:
        """
        Stores the preferred values of an axis (validated tokens, in axis order) and why. Unlike
        relevant values, elites do not widen them: fill_gap, diversify, combine and extrapolate only
        target preferred values, explore reaches the others at a low weight. Returns True if changed.
        """
        axis_obj = self.space.axis(axis)
        cleaned = sorted({v for v in values or () if v in axis_obj.values}, key=axis_obj.index)
        preferred = self.request_analysis.setdefault("preferred", {})
        reasons = self.request_analysis.setdefault("preferred_reason", {})
        if not cleaned or (preferred.get(axis) == cleaned and reasons.get(axis) == reason):
            return False
        preferred[axis] = cleaned
        reasons[axis] = reason
        return True

    def preferred_values(self, axis: str) -> Optional[List[str]]:
        """Preferred values of an axis, or None if the axis has no preference."""
        values = (self.request_analysis.get("preferred") or {}).get(axis)
        return list(values) if values else None

    def preferred_axes(self) -> List[str]:
        return [name for name in self.space.names if self.preferred_values(name)]

    def is_preferred(self, cell: Mapping[str, str]) -> bool:
        """False if the (full or partial) cell uses a non-preferred value of an axis with preferences."""
        for axis in self.preferred_axes():
            if axis in cell and cell[axis] not in self.preferred_values(axis):
                return False
        return True

    # ---- proposal statistics per axis value -----------------------------
    def value_counts(self, axis: str) -> Dict[str, int]:
        """Proposals (placed candidates of any status) per value of one axis."""
        counts = {value: 0 for value in self.space.axis(axis).values}
        for key, count in self.data["proposals"].items():
            value = self.space.parse_key(key).get(axis)
            if value in counts:
                counts[value] += int(count)
        return counts

    def axis_concentration(self, axis: str) -> float:
        """Share of all proposals held by the most common value of an axis (0 if none yet)."""
        counts = self.value_counts(axis)
        total = sum(counts.values())
        return max(counts.values()) / total if total else 0.0

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
        replaced by stronger evidence or, at equal evidence rank, by a strictly higher score
        (ties keep the older entry); scores within TIE_EPSILON with differing tiebreak keys are
        decided by the key (``entry["tiebreak"]`` records it, see ``compare_entries``).
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
            else:
                better, by_tiebreak = compare_entries(entry, current)
                if better:
                    entry["outcome"] = IMPROVED
                    entry["replaced"] = current_id
                    self.elites[cell_key] = entry_id
                else:
                    entry["outcome"] = NOT_BETTER
                if by_tiebreak:
                    entry["tiebreak"] = IMPROVED_ON_TIEBREAK if better else KEPT_ON_TIEBREAK
        self.entries[entry_id] = entry
        return entry

    def log_round(self, record: Mapping[str, Any]) -> None:
        self.data["rounds"].append(dict(record))
