"""New-tool proposal loop — safe registration of IR primitives (IR.md).

When the Synthesizer Agent spots tools in the trader's concept that have no
registered mathematical primitive (e.g. "OBV Divergence", "Session VWAP"), it
returns ``new_primitives`` proposals: name, description, parameters (types +
defaults) and a deterministic formula. This module is the approval gate and
safe generator:

* :func:`validate_proposal` — deterministic vetting (identifier shape, param
  types, type-checked defaults, registry collisions, caps). LLM output is never
  trusted blindly.
* :func:`generate_primitive_code` — emits the canonical
  ``@registry.register_primitive`` function block (exactly the shape a human
  engineer would write by hand).
* :func:`write_and_reload` — atomic append into ``ir_primitives.py`` + live
  reload, so the Tool Schema Menu and the compiler vocabulary update instantly
  for the running process (hot-swappable primitives, IR.md). The file is rolled
  back if the reload fails, so a bad write never leaves a half-applied registry.

Runtime LLM output only lands in the codebase when the user explicitly approves
it via ``POST /strategies/primitives/register``.
"""

from __future__ import annotations

import importlib
import json
import keyword
import re
import threading
from pathlib import Path

import app.services.ir_primitives as _ir_primitives

# Serialize read-modify-replace so two rapid register calls can't clobber each
# other (matches the _LOCK pattern used by the strategy pipeline).
_WRITE_LOCK = threading.Lock()

_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_PARAM_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
VALID_TYPES = ("int", "float", "str", "bool")
MAX_PARAMS = 8
MAX_BATCH = 5
_DEFAULT_BY_TYPE = {"int": 0, "float": 0.0, "str": "", "bool": False}


def _default_ok(value: object, ptype: str) -> bool:
    if ptype == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if ptype == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if ptype == "str":
        return isinstance(value, str)
    if ptype == "bool":
        return isinstance(value, bool)
    return False


def registry_snapshot(registry=None) -> list[dict]:
    """The active registry as plain JSON: name / description / parameters."""
    registry = registry or _ir_primitives.DEFAULT_REGISTRY
    out: list[dict] = []
    for name in registry.primitives:
        data = registry.get(name)
        out.append({
            "name": name,
            "description": data["description"],
            "parameters": {
                k: {
                    "type": v["type"],
                    "default": None if v["default"] == "REQUIRED" else v["default"],
                }
                for k, v in data["parameters"].items()
            },
        })
    return out


def validate_proposal(proposal: object,
                      registry=None) -> tuple[dict | None, str]:
    """Deterministic vetting of one LLM-produced primitive proposal.

    Returns ``(normalized_proposal, error)``. An empty error string means the
    proposal is safe to generate; a ``None`` proposal means it was rejected.
    """
    registry = registry or _ir_primitives.DEFAULT_REGISTRY
    if not isinstance(proposal, dict):
        return None, "proposal must be an object"
    name = proposal.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        return None, (
            f"Invalid primitive name {name!r}: use UPPER_SNAKE_CASE "
            "(e.g. OBV_DIVERGENCE).")
    if name in registry:
        return None, f"{name} is already in the registry."
    fn_name = name.lower()
    if not fn_name.isidentifier() or keyword.iskeyword(fn_name):
        return None, (
            f"{name} cannot be used: it would generate an invalid Python "
            "function name.")
    description = proposal.get("description")
    if not isinstance(description, str) or not description.strip():
        return None, f"{name}: a description of the OHLCV contract is required."
    raw_params = proposal.get("parameters")
    if not isinstance(raw_params, dict) or not raw_params:
        return None, f"{name}: parameters must be a non-empty object."
    if len(raw_params) > MAX_PARAMS:
        return None, f"{name}: too many parameters (max {MAX_PARAMS})."
    params: dict[str, dict] = {}
    for pname, meta in raw_params.items():
        if not isinstance(pname, str) or not _PARAM_RE.match(pname):
            return None, (
                f"{name}: invalid parameter name {pname!r} "
                "(lowercase snake_case).")
        if isinstance(meta, str):
            ptype, default = meta, None  # tolerate "int" shorthand
        elif isinstance(meta, dict):
            ptype = meta.get("type")
            default = meta.get("default")
        else:
            return None, f"{name}: parameter {pname} metadata must be an object."
        if ptype not in VALID_TYPES:
            return None, (
                f"{name}: parameter {pname} has invalid type {ptype!r} "
                "(int|float|str|bool).")
        if default is not None and not _default_ok(default, ptype):
            return None, (
                f"{name}: parameter {pname} default {default!r} does not "
                f"match type {ptype}.")
        if default is None:
            default = _DEFAULT_BY_TYPE[ptype]
        params[pname] = {"type": ptype, "default": default}
    formula = proposal.get("formula")
    if not isinstance(formula, str) or not formula.strip():
        return None, f"{name}: a formula (deterministic math in words) is required."
    return {
        "name": name,
        "description": description.strip(),
        "parameters": params,
        "formula": formula.strip(),
    }, ""


def validate_batch(proposals: list,
                   registry=None,
                   cap: int = MAX_BATCH) -> tuple[list[dict], list[str]]:
    """Validate a batch, dedupe, and cap the number of writes per call."""
    registry = registry or _ir_primitives.DEFAULT_REGISTRY
    valid: list[dict] = []
    errors: list[str] = []
    seen: set[str] = set()
    for p in (proposals or [])[:cap]:
        if not isinstance(p, dict):
            errors.append("proposal must be an object")
            continue
        name = p.get("name")
        if name in seen:
            errors.append(f"Duplicate proposal: {name}.")
            continue
        ok, err = validate_proposal(p, registry=registry)
        if ok is None:
            errors.append(err)
        else:
            seen.add(ok["name"])
            valid.append(ok)
    return valid, errors


def _default_repr(value: object, ptype: str) -> str:
    if ptype == "str":
        return json.dumps(value)
    return repr(value)


def _wrap_description(desc: str, indent: str = "        ") -> str:
    """Wrap a description as implicit string concatenation (~72 cols) so the
    generated decorator matches the hand-written style of the registry."""
    words = desc.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > 72:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    if len(lines) == 1:
        return f'{indent}"{lines[0]}")'
    parts = [f'{indent}"{lines[0]} "']
    for ln in lines[1:-1]:
        parts.append(f'{indent}"{ln} "')
    parts.append(f'{indent}"{lines[-1]}")')
    return "\n".join(parts)


def generate_primitive_code(proposal: dict) -> str:
    """Emit the canonical ``@registry.register_primitive`` block for a validated
    proposal — exactly the shape the pipeline's other primitives use."""
    name = proposal["name"]
    fn_name = name.lower()
    params = proposal["parameters"]
    sig = ", ".join(
        f"{p}: {meta['type']} = {_default_repr(meta['default'], meta['type'])}"
        for p, meta in params.items())
    ret = ", ".join(f'"{p}": {p}' for p in params)
    return (
        "    @registry.register_primitive(\n"
        f'        "{name}",\n'
        f"{_wrap_description(proposal['description'])}\n"
        f"    def {fn_name}({sig}) -> dict:\n"
        f"        return {{{ret}}}\n"
    )


def append_to_file(path: Path, code_blocks: list[str]) -> str:
    """Insert generated blocks into ``build_default_registry`` (before its
    ``return registry``) and return the new module text. Pure — nothing is
    written to disk here; the caller owns the atomic write."""
    text = path.read_text(encoding="utf-8")
    marker = "\n    return registry\n"
    if marker not in text:
        raise ValueError(
            "ir_primitives.py is missing the 'return registry' marker — "
            "cannot append new primitives.")
    joined = "\n".join(code_blocks)
    new_text = text.replace(marker, "\n" + joined + marker, 1)
    compile(new_text, str(path), "exec")  # syntax gate before touching disk
    return new_text


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)  # atomic (os.replace semantics) on POSIX + Windows


def write_and_reload(code_blocks: list[str]) -> list[str]:
    """Atomically append the generated blocks to ``ir_primitives.py`` and
    live-reload the module. Returns the registered primitive names.

    On any failure the file is rolled back to its previous contents and the
    reload retried, so a bad write never leaves the running process with a
    half-applied registry.
    """
    path = Path(_ir_primitives.__file__)
    with _WRITE_LOCK:
        original = path.read_text(encoding="utf-8")
        names: list[str] = []
        for block in code_blocks:
            m = re.search(r'register_primitive\(\s*"([A-Z][A-Z0-9_]+)"', block)
            if m:
                names.append(m.group(1))
        try:
            _write_atomic(path, append_to_file(path, code_blocks))
            importlib.reload(_ir_primitives)
        except Exception as exc:
            # Roll the file back so the running process stays consistent; the
            # in-memory registry may still be stale until the next reload.
            path.write_text(original, encoding="utf-8")
            try:
                importlib.reload(_ir_primitives)
            except Exception:  # noqa: BLE001, S110 - rollback reload is best-effort
                pass
            raise ValueError(
                f"Could not register {names or 'primitive'}: {exc}") from exc
    missing = [n for n in names if n not in _ir_primitives.DEFAULT_REGISTRY]
    if missing:
        raise ValueError(f"Reload did not register: {missing}.")
    return names


def remove_primitives(names: list[str]) -> None:
    """Remove previously auto-generated blocks (cleanup for tests / rollback).

    Iterates in reverse so each removal only ever matches the block directly
    in front of ``return registry``.
    """
    path = Path(_ir_primitives.__file__)
    with _WRITE_LOCK:
        text = path.read_text(encoding="utf-8")
        for name in reversed(names):
            pat = re.compile(
                r"\n\n    @registry\.register_primitive\(\n        \""
                + re.escape(name)
                + r'".*?\n    return registry',
                re.DOTALL,
            )
            text, n = pat.subn("\n\n    return registry", text, count=1)
            if n == 0:
                raise ValueError(f"{name} block not found for removal.")
        compile(text, str(path), "exec")
        _write_atomic(path, text)
        importlib.reload(_ir_primitives)


def register_primitives(proposals: list) -> dict:
    """Validate + write + live-reload a batch of approved proposals."""
    valid, errors = validate_batch(proposals)
    if not valid:
        raise ValueError("; ".join(errors) or "No valid primitives to register.")
    code_blocks = [generate_primitive_code(p) for p in valid]
    names = write_and_reload(code_blocks)
    return {"registered": names, "errors": errors}
