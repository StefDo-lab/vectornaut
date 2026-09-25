# -*- coding: utf-8 -*-
"""
Descriptor axes: the fixed coordinate system of an explorer map.

Every concept is placed in exactly one cell: one allowed value per axis. Axes are
either *nominal* (unordered categories, e.g. a mechanism class) or *ordinal* (ordered
steps, e.g. a length scale from nm to cm). Only ordinal axes support trends and
extrapolation.

The model assigns descriptors from a closed vocabulary. Normalisation is tolerant of
case, spaces and hyphens ("10 um", "10-UM" -> "10_um"), but an unknown value is never
guessed: the candidate is rejected with a reason.

Cell keys are strings like ``"mechanism_class=phase_change|length_scale=mm"`` in axis
order. A *pattern* is a partial assignment; unspecified axes are written as ``*``.
"""
import itertools
import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

NOMINAL = "nominal"
ORDINAL = "ordinal"
WILDCARD = "*"


class DescriptorError(ValueError):
    """Raised when model-assigned descriptors do not fit the axes. ``problems`` lists every issue."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def normalize_token(raw: object) -> str:
    """Lower-case, map 'µ' to 'u', and join words with single underscores."""
    text = str(raw if raw is not None else "").strip().lower()
    text = text.replace("µ", "u").replace("μ", "u")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


@dataclass(frozen=True)
class Axis:
    name: str
    kind: str
    values: Tuple[str, ...]
    description: str = ""
    # Optional one-line explanation per value, shown to the model.
    value_help: Mapping[str, str] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        if self.kind not in (NOMINAL, ORDINAL):
            raise ValueError(f"Axis '{self.name}': kind must be '{NOMINAL}' or '{ORDINAL}', got '{self.kind}'.")
        if len(self.values) < 2:
            raise ValueError(f"Axis '{self.name}' needs at least two values.")
        normalized = [normalize_token(v) for v in self.values]
        if normalized != list(self.values):
            raise ValueError(f"Axis '{self.name}': values must already be normalised tokens: {self.values}")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"Axis '{self.name}' has duplicate values.")

    @property
    def is_ordinal(self) -> bool:
        return self.kind == ORDINAL

    def normalize(self, raw: object) -> str:
        token = normalize_token(raw)
        if token not in self.values:
            raise DescriptorError([
                f"{self.name}: '{raw}' is not an allowed value (allowed: {', '.join(self.values)})"
            ])
        return token

    def index(self, value: str) -> int:
        return self.values.index(value)


class DescriptorSpace:
    """The ordered set of axes of one profile."""

    def __init__(self, axes: Sequence[Axis]):
        names = [axis.name for axis in axes]
        if len(set(names)) != len(names):
            raise ValueError("Axis names must be unique.")
        self.axes: Tuple[Axis, ...] = tuple(axes)
        self._by_name = {axis.name: axis for axis in self.axes}

    # ---- basics -------------------------------------------------------
    @property
    def names(self) -> List[str]:
        return [axis.name for axis in self.axes]

    def axis(self, name: str) -> Axis:
        return self._by_name[name]

    @property
    def ordinal_axes(self) -> List[Axis]:
        return [axis for axis in self.axes if axis.is_ordinal]

    @property
    def size(self) -> int:
        total = 1
        for axis in self.axes:
            total *= len(axis.values)
        return total

    def all_cells(self) -> Iterator[Dict[str, str]]:
        for combo in itertools.product(*(axis.values for axis in self.axes)):
            yield dict(zip(self.names, combo))

    # ---- validation ---------------------------------------------------
    def validate(self, raw: Mapping[str, object], partial: bool = False) -> Dict[str, str]:
        """
        Normalises a model-assigned descriptor mapping. Keys are matched tolerantly as
        well. Every problem is collected; any problem rejects the whole assignment.
        Unknown extra axes are a problem too (they usually mean the model invented one).
        """
        problems: List[str] = []
        result: Dict[str, str] = {}
        by_token = {normalize_token(name): name for name in self.names}
        for raw_key, raw_value in (raw or {}).items():
            key = by_token.get(normalize_token(raw_key))
            if key is None:
                problems.append(f"unknown axis '{raw_key}' (axes: {', '.join(self.names)})")
                continue
            if key in result:
                problems.append(f"axis '{key}' assigned twice")
                continue
            if partial and normalize_token(raw_value) in ("", WILDCARD):
                continue
            try:
                result[key] = self._by_name[key].normalize(raw_value)
            except DescriptorError as err:
                problems.extend(err.problems)
        if not partial:
            for name in self.names:
                if name not in result and not any(p.startswith(f"{name}:") for p in problems):
                    problems.append(f"{name}: missing")
        if problems:
            raise DescriptorError(problems)
        return {name: result[name] for name in self.names if name in result}

    def validate_pairs(self, pairs: Sequence[object]) -> Dict[str, str]:
        """Validates a list of ``{axis, value}`` objects (the response-schema form)."""
        mapping: Dict[str, object] = {}
        problems: List[str] = []
        for pair in pairs or []:
            axis = getattr(pair, "axis", None) if not isinstance(pair, Mapping) else pair.get("axis")
            value = getattr(pair, "value", None) if not isinstance(pair, Mapping) else pair.get("value")
            if axis in mapping:
                problems.append(f"axis '{axis}' assigned twice")
                continue
            mapping[axis] = value
        try:
            result = self.validate(mapping)
        except DescriptorError as err:
            problems.extend(err.problems)
            raise DescriptorError(problems)
        if problems:
            raise DescriptorError(problems)
        return result

    # ---- cell keys ----------------------------------------------------
    def cell_key(self, descriptors: Mapping[str, str]) -> str:
        """Full or partial key; missing axes become ``*``."""
        return "|".join(f"{name}={descriptors.get(name, WILDCARD)}" for name in self.names)

    def parse_key(self, key: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for part in (key or "").split("|"):
            if not part:
                continue
            name, _, value = part.partition("=")
            if value and value != WILDCARD:
                result[name] = value
        return result

    def is_full(self, descriptors: Mapping[str, str]) -> bool:
        return all(name in descriptors for name in self.names)

    @staticmethod
    def matches(cell: Mapping[str, str], pattern: Mapping[str, str]) -> bool:
        """True if the full cell satisfies every axis the pattern fixes."""
        return all(cell.get(name) == value for name, value in pattern.items())

    # ---- geometry -----------------------------------------------------
    def distance(self, a: Mapping[str, str], b: Mapping[str, str]) -> int:
        """
        Steps between two full cells: a differing nominal axis counts 1, an ordinal
        axis counts the number of ordinal steps. Distance 1 = one nominal change or one
        ordinal step (the "Hamming distance 1" neighbourhood on this grid).
        """
        total = 0
        for axis in self.axes:
            va, vb = a[axis.name], b[axis.name]
            if va == vb:
                continue
            total += abs(axis.index(va) - axis.index(vb)) if axis.is_ordinal else 1
        return total

    def differing_axes(self, a: Mapping[str, str], b: Mapping[str, str]) -> List[str]:
        return [name for name in self.names if a.get(name) != b.get(name)]

    def neighbours(self, cell: Mapping[str, str]) -> List[Dict[str, str]]:
        """All cells at distance 1, in deterministic axis/value order."""
        result = []
        for axis in self.axes:
            current = cell[axis.name]
            if axis.is_ordinal:
                idx = axis.index(current)
                options = [axis.values[i] for i in (idx - 1, idx + 1) if 0 <= i < len(axis.values)]
            else:
                options = [value for value in axis.values if value != current]
            for value in options:
                neighbour = dict(cell)
                neighbour[axis.name] = value
                result.append(neighbour)
        return result

    def describe(self, descriptors: Mapping[str, str]) -> str:
        """Readable form for prompts and reports; partial assignments show only fixed axes."""
        parts = [f"{name}={descriptors[name]}" for name in self.names if name in descriptors]
        return ", ".join(parts) if parts else "(any cell)"

    def vocabulary_text(self) -> str:
        lines = []
        for axis in self.axes:
            order = " < ".join(axis.values) if axis.is_ordinal else ", ".join(axis.values)
            kind = "ordinal, ordered" if axis.is_ordinal else "nominal"
            lines.append(f"- {axis.name} ({kind}): {order}")
            if axis.description:
                lines.append(f"    {axis.description}")
            for value in axis.values:
                help_text = axis.value_help.get(value)
                if help_text:
                    lines.append(f"    * {value}: {help_text}")
        return "\n".join(lines)

    def to_dict(self) -> List[Dict[str, object]]:
        return [{"name": axis.name, "kind": axis.kind, "values": list(axis.values)} for axis in self.axes]


def as_pattern(space: DescriptorSpace, target: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """Normalises a (possibly partial) target assignment in axis order."""
    return {name: target[name] for name in space.names if target and name in target}
