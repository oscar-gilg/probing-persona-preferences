r"""Claims: the unified registry behind paper numbers, plot sidecars, and LaTeX macros.

Every empirical number rendered into the paper (plot bar, annotation, caption,
prose numeral) is a `Claim`. Producers register claims at the point they
compute the value; the same value flows into the plot and into the registry.

Downstream, `build_claims.py` merges all per-producer sidecars into:
  - `paper/numbers.tex`  (LaTeX macros)
  - `paper/claims.md`    (human/agent audit surface)

The claim `name` is the human key ("Gemma probe heldout r within-topic"). The
LaTeX macro name is derived by slugifying (`gemmaProbeHeldoutRWithinTopic`).
The `statement` is a declarative sentence describing what the number asserts —
the primary read for anyone auditing the registry.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Union


ClaimValue = Union[float, int, str]


@dataclass(frozen=True)
class Claim:
    name: str
    value: ClaimValue
    statement: str
    source: str
    used_in: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["used_in"] = list(self.used_in)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Claim":
        return Claim(
            name=d["name"],
            value=d["value"],
            statement=d["statement"],
            source=d["source"],
            used_in=tuple(d.get("used_in", [])),
        )


@dataclass
class ClaimSet:
    """Collector used by producer scripts. Registers claims and writes a sidecar."""

    source: str
    claims: list[Claim] = field(default_factory=list)

    def register(
        self,
        name: str,
        value: ClaimValue,
        statement: str,
        used_in: list[str] | tuple[str, ...] = (),
        source: str | None = None,
    ) -> ClaimValue:
        """Record a claim and return its value so it can be used inline.

        `source` defaults to the ClaimSet's source. Pass a custom `source` (e.g.
        ``"manual: ..."``) when a single claim within the producer is not
        machine-derived, so audit_claims flags it.
        """
        if any(c.name == name for c in self.claims):
            raise ValueError(f"Duplicate claim name within producer: {name!r}")
        self.claims.append(
            Claim(
                name=name,
                value=value,
                statement=statement,
                source=source if source is not None else self.source,
                used_in=tuple(used_in),
            )
        )
        return value

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": self.source,
            "claims": [c.to_dict() for c in self.claims],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def load_all(claims_dir: Path | str) -> list[Claim]:
    """Load and merge every sidecar in `claims_dir`. Raises on name collision."""
    claims_dir = Path(claims_dir)
    claims: list[Claim] = []
    seen: dict[str, str] = {}
    for sidecar in sorted(claims_dir.glob("*.json")):
        payload = json.loads(sidecar.read_text())
        for raw in payload["claims"]:
            c = Claim.from_dict(raw)
            if c.name in seen:
                raise ValueError(
                    f"Claim name collision: {c.name!r} in {sidecar.name} "
                    f"and {seen[c.name]}"
                )
            seen[c.name] = sidecar.name
            claims.append(c)
    return claims


_MACRO_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _spell_digits(s: str) -> str:
    return "".join(_DIGIT_WORDS.get(c, c) for c in s)


def name_to_macro(name: str) -> str:
    """Slugify a claim name into a LaTeX macro identifier.

    LaTeX control sequences accept only letters, so digits are spelled out.
    Rules:
      - Strip apostrophes so "Cohen's" stays one token.
      - Split on any non-alphanumeric character.
      - Lowercase each token.
      - Spell out every digit (0 -> "zero", 1 -> "one", ...). This keeps the
        macro valid and readable: "phase1" -> "phaseone", "0.03" -> two
        tokens "0", "03" -> "zero", "zerothree".
      - First token stays lowercase; subsequent tokens have their first
        letter capitalized (camelCase). All-caps acronyms normalise: "CREAK"
        -> "creak", not "cREAK".
    """
    stripped = name.replace("'", "").replace("’", "")
    tokens = _MACRO_TOKEN_RE.findall(stripped)
    if not tokens:
        raise ValueError(f"Claim name has no alphanumeric content: {name!r}")
    pieces: list[str] = []
    for i, tok in enumerate(tokens):
        low = _spell_digits(tok.lower())
        if i == 0:
            piece = low
        else:
            piece = low[0].upper() + low[1:]
        pieces.append(piece)
    return "".join(pieces)


def _format_value_for_tex(value: ClaimValue) -> str:
    """Render a claim value for a LaTeX macro body."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Trim trailing zeros but keep at least 2 dp for sub-unit values;
        # let producers control precision by passing pre-rounded floats.
        s = f"{value:g}"
        return s
    return str(value)


def _tex_escape(s: str) -> str:
    """Minimal escape for claim names shown in comments."""
    return s.replace("\\", "\\textbackslash{}").replace("%", "\\%")


def write_numbers_tex(claims: list[Claim], path: Path | str) -> None:
    r"""Emit ``\newcommand{\macroName}{value}`` for every claim."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by scripts/paper/build_claims.py. Do not edit by hand.",
        "% Each macro's value comes from a producer registered in paper/claims/*.json.",
        "",
    ]
    macros_seen: dict[str, str] = {}
    for c in claims:
        macro = name_to_macro(c.name)
        if macro in macros_seen:
            raise ValueError(
                f"Macro collision: {macro!r} from {c.name!r} conflicts with "
                f"{macros_seen[macro]!r}"
            )
        macros_seen[macro] = c.name
        body = _format_value_for_tex(c.value)
        lines.append(f"% {_tex_escape(c.name)}")
        lines.append(f"\\newcommand{{\\{macro}}}{{{body}}}")
    path.write_text("\n".join(lines) + "\n")


def write_claims_md(claims: list[Claim], path: Path | str) -> None:
    """Emit the human/agent audit surface."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper claims registry",
        "",
        "Auto-generated by `scripts/paper/build_claims.py`. Each row is a single",
        "empirical number used in `paper/main.tex`. The `statement` column is the",
        "claim the paper is making; `source` is the producer that computed the",
        "value from data.",
        "",
        f"Total claims: **{len(claims)}**",
        "",
        "| Name | Value | Statement | Source | Used in |",
        "|---|---|---|---|---|",
    ]
    for c in sorted(claims, key=lambda c: c.name.lower()):
        used = ", ".join(c.used_in) if c.used_in else ""
        value_str = _format_value_for_tex(c.value)
        statement = c.statement.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{c.name}` | {value_str} | {statement} | `{c.source}` | {used} |"
        )
    path.write_text("\n".join(lines) + "\n")
