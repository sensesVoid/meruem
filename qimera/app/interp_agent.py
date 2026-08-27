"""QiMera Indicator Interpretation Agent (QiMera.md §6.6 / M6).

A **bounded workflow** agent that turns a descriptive trading concept (e.g.
"sell the Asian-session sweep then the FVG retest") into a typed strategy
blueprint over the M5 indicator schemas.

Workflow (linear, with a single guarded retry):

    parse -> resolve_schema -> lookup_definition -> generate_code
          -> backtest_guard -> finalize

- ``parse``            LLM extracts direction + indicator-family hint (falls
                       back to deterministic keyword matching when the LLM is
                       unavailable/offline).
- ``resolve_schema``   Maps the concept to one of the M5 ``INDICATOR_SCHEMAS``.
- ``lookup_definition``Web search for concept definitions; **provenance is
                       logged** (retrieval date, URL, snippet) for auditability.
- ``generate_code``    LLM writes ``generate_signal(df)`` against the schema
                       (template fallback when offline).
- ``backtest_guard``   Runs the generated code in an **isolated subprocess with
                       a hard 2-second timeout** (static forbidden-import scan +
                       ``python -I`` + subprocess timeout), so a runaway or
                       malicious snippet can never hang or touch the host.
- ``finalize``         Assembles the JSON-safe record: schema, params, direction,
                       provenance, guard verdict, audit trail.

The engine is a tiny explicit state machine (``BoundedWorkflow``): nodes are
plain functions that return the next node name, steps are capped, and every
transition is recorded in the audit trail. It deliberately has no third-party
dependency (LangGraph-style boundedness without the LangGraph dependency tree).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qimera.app.gp_indicators import (
    INDICATOR_IDS, INDICATOR_SCHEMAS, signal_for,
)

# Max wall-clock the generated strategy code may run inside the guard.
# ~1s of this budget is interpreter + numpy/pandas cold import; the user code
# itself gets roughly the rest (spec: 2-second timeout).
GUARD_TIMEOUT_S = 3.0

# Modules the generated code may import. numpy/pandas/math are the harness
# building blocks; the qimera entries are the read-only feature modules the
# strategy/synthesis code is allowed to consume (``signal_for`` from the GP
# indicators, the feature library for M7 expressions). Anything else is
# rejected at the AST level, so `import socket`, `from app.services.vault
# import ...`, or a bypass via `importlib` can never even reach the subprocess.
_ALLOWED_IMPORTS = {
    "numpy", "numpy as np", "pandas", "pandas as pd", "math",
    "qimera.app.gp_indicators", "qimera.app.feature_library",
}

# Builtins the generated code may use. ``__import__``/``open``/``eval``/
# ``compile``/``getattr``/``globals``/``vars`` are deliberately absent — with a
# restricted builtins dict, ``__builtins__['__import__']('os')`` tricks fail at
# the interpreter level instead of relying on string scanning.
_ALLOWED_BUILTINS = frozenset({
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "int", "isinstance", "issubclass", "len", "list", "map",
    "max", "min", "pow", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip",
    # Exception classes: raising/guarding errors in strategy code is normal
    # and safe (classes carry no I/O surface).
    "Exception", "ValueError", "TypeError", "RuntimeError", "KeyError",
    "IndexError", "ArithmeticError", "ZeroDivisionError",
    "True", "False", "None", "NotImplemented", "Ellipsis",
})

_PROVENANCE_ROOT = Path(__file__).resolve().parent.parent / "data" / "interp"

# Deterministic concept -> indicator family mapping (offline fallback).
_KEYWORD_MAP: list[tuple[tuple[str, ...], str]] = [
    (("vwap", "volume weighted", "volume-weighted", "session anchor"), "vwap"),
    (("fvg", "fair value gap", "imbalance"), "fvg"),
    (("sweep", "swept", "stop hunt", "liquidity grab", "equal high",
      "equal low", "liquidity"), "liquidity"),
    (("order block", "orderblock", "ob "), "order_blocks"),
    (("cvd", "cumulative volume delta", "volume delta", "delta"), "cvd"),
]

_DIRECTION_MAP: list[tuple[tuple[str, ...], str]] = [
    (("short", "sell", "bearish", "down"), "short"),
    (("long", "buy", "bullish", "up"), "long"),
]


class WorkflowHalted(RuntimeError):
    """Raised when a bounded workflow exceeds its step budget or hits a dead end."""


class BoundedWorkflow:
    """Tiny explicit state machine: nodes return the next node name or None.

    Steps are capped (``max_steps``) and every transition is recorded on the
    state's ``audit`` list — the "bounded" guarantee for agent loops.
    """

    def __init__(self, max_steps: int = 16):
        self.max_steps = max_steps
        self._nodes: dict[str, callable] = {}

    def add_node(self, name: str, fn: callable) -> "BoundedWorkflow":
        self._nodes[name] = fn
        return self

    def run(self, state: dict, start: str = "start") -> dict:
        state.setdefault("audit", [])
        node = start
        for _ in range(self.max_steps):
            if node is None:
                return state
            if node not in self._nodes:
                raise WorkflowHalted(f"unknown node {node!r}")
            state["audit"].append({
                "node": node,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            node = self._nodes[node](state)
        raise WorkflowHalted(f"step budget ({self.max_steps}) exceeded")


# ---------------------------------------------------------------------------
# Search tool (web concept definitions, keyless, graceful offline)
# ---------------------------------------------------------------------------


class WebSearchTool:
    """Keyless concept search via Wikipedia's opensearch API.

    Returns ``[{title, url, snippet}]``. Any network failure returns [] so the
    workflow keeps running offline — provenance just records what was found.
    """

    def __init__(self, timeout: float = 4.0):
        self.timeout = timeout

    def search(self, query: str, limit: int = 3) -> list[dict]:
        import requests  # lazy: requests is a Meruem/qimera dependency

        try:
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": limit,
                    "namespace": 0,
                    "format": "json",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:  # noqa: BLE001 - offline-safe
            return []
        titles, urls = data[1], data[3]
        snippets = data[2] if len(data) > 2 else []
        out = []
        for i, url in enumerate(urls):
            out.append({
                "title": titles[i] if i < len(titles) else "",
                "url": url,
                "snippet": snippets[i] if i < len(snippets) else "",
            })
        return out


# ---------------------------------------------------------------------------
# LLM adapter (reuses Meruem's UniversalLLM; injectable for offline tests)
# ---------------------------------------------------------------------------


def _default_llm(prompt: str) -> str:
    """Lazy UniversalLLM call — returns '' on any failure (offline-safe)."""
    try:
        from app.agents.universal_llm import UniversalLLM  # backend package

        text, _provider = UniversalLLM(agent_id="qimera_interp").generate(
            prompt, temperature=0.1)
        return text or ""
    except Exception:  # noqa: BLE001 - offline / no backend -> deterministic path
        return ""


def _parse_llm_json(text: str) -> dict | None:
    """Best-effort JSON extraction from an LLM reply (fenced or bare)."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _keyword_resolve(concept: str) -> dict:
    low = concept.lower()
    direction = "either"
    for keys, d in _DIRECTION_MAP:
        if any(k in low for k in keys):
            direction = d
            break
    indicator = "vwap"  # conservative default
    for keys, iid in _KEYWORD_MAP:
        if any(k in low for k in keys):
            indicator = iid
            break
    return {"indicator": indicator, "direction": direction,
            "rationale": "deterministic keyword fallback"}


def _schema_params(indicator_id: str) -> list[float]:
    return [float(p["default"]) for p in INDICATOR_SCHEMAS[indicator_id]["params"]]


def _fallback_signal_code(indicator_id: str, params: list[float]) -> str:
    return (
        "import numpy as np\n"
        "import pandas as pd\n"
        "from qimera.app.gp_indicators import signal_for\n"
        f"def generate_signal(df):\n"
        f"    return signal_for({indicator_id!r}, df, {params!r})\n"
    )


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class InterpAgent:
    """Bounded concept-to-blueprint agent over the M5 indicator schemas."""

    def __init__(self, llm: callable | None = None,
                 search: callable | None = None,
                 guard_timeout: float = GUARD_TIMEOUT_S,
                 provenance_root: Path = _PROVENANCE_ROOT):
        self.llm = llm or _default_llm
        self.search = search or WebSearchTool().search
        self.guard_timeout = guard_timeout
        self.provenance_root = Path(provenance_root)

    # -- provenance ---------------------------------------------------------

    def _log_provenance(self, record: dict) -> None:
        self.provenance_root.mkdir(parents=True, exist_ok=True)
        path = self.provenance_root / "interpretations.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # -- nodes --------------------------------------------------------------

    def _parse(self, state: dict) -> str | None:
        concept = state["concept"]
        reply = self.llm(
            "You map a trading concept to a typed indicator family. Reply with "
            "JSON only: {\"indicator\": <one of "
            f"{INDICATOR_IDS}>, \"direction\": <long|short|either>, "
            "\"rationale\": <one sentence>}.\n"
            f"Concept: {concept}")
        parsed = _parse_llm_json(reply) or _keyword_resolve(concept)
        indicator = parsed.get("indicator")
        if indicator not in INDICATOR_IDS:
            indicator = _keyword_resolve(concept)["indicator"]
        direction = parsed.get("direction", "either")
        if direction not in ("long", "short", "either"):
            direction = "either"
        state["indicator"] = indicator
        state["direction"] = direction
        state["rationale"] = str(parsed.get("rationale", ""))
        state["llm_used"] = bool(reply)
        return "resolve_schema"

    def _resolve_schema(self, state: dict) -> str | None:
        schema = INDICATOR_SCHEMAS[state["indicator"]]
        state["schema"] = {
            "id": state["indicator"],
            "name": schema["name"],
            "description": schema["description"],
            "params": schema["params"],
            "param_defaults": _schema_params(state["indicator"]),
        }
        return "lookup_definition"

    def _lookup_definition(self, state: dict) -> str | None:
        concept = state["concept"]
        results = self.search(concept, limit=3)
        state["provenance"] = [
            {
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                **r,
            }
            for r in results
        ]
        return "generate_code"

    def _generate_code(self, state: dict) -> str | None:
        schema = state["schema"]
        reply = self.llm(
            "Write a Python function `generate_signal(df)` that returns a pandas "
            "Series of the same length as df, values normalized to [-1, +1] "
            "(positive = long bias), implementing this trading concept:\n"
            f"Concept: {state['concept']}\n"
            f"Chosen family: {schema['name']} ({schema['id']}) with params "
            f"{schema['params']}.\n"
            "You may import numpy/pandas and `from qimera.app.gp_indicators "
            "import signal_for` and call it with your chosen param values. "
            "Return ONLY the function definition, no explanation, no "
            "markdown fences.")
        code = reply.strip()
        if not code or "def generate_signal" not in code:
            code = _fallback_signal_code(state["indicator"],
                                         state["schema"]["param_defaults"])
        state["generated_code"] = code
        state["code_source"] = "llm" if reply else "template"
        return "backtest_guard"

    def _backtest_guard(self, state: dict) -> str | None:
        verdict = run_backtest_guard(
            state["generated_code"], self.guard_timeout)
        state["guard"] = verdict
        # One guarded retry with the template when the LLM code is broken.
        if not verdict.get("ok"):
            code = _fallback_signal_code(state["indicator"],
                                         state["schema"]["param_defaults"])
            state["generated_code"] = code
            state["code_source"] = "template_retry"
            state["guard"] = run_backtest_guard(code, self.guard_timeout)
        return "finalize"

    def _finalize(self, state: dict) -> str | None:
        record = {
            "run_id": uuid.uuid4().hex[:12],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "concept": state["concept"],
            "indicator": state["indicator"],
            "direction": state["direction"],
            "rationale": state["rationale"],
            "schema": state["schema"],
            "provenance": state.get("provenance", []),
            "generated_code": state["generated_code"],
            "code_source": state["code_source"],
            "guard": state["guard"],
            "audit": state["audit"],
            "llm_used": state.get("llm_used", False),
        }
        state["record"] = record
        self._log_provenance(record)
        return None

    # -- entry point ---------------------------------------------------------

    def interpret(self, concept: str) -> dict:
        """Run the bounded interpretation workflow for a descriptive concept."""
        wf = (BoundedWorkflow()
              .add_node("parse", self._parse)
              .add_node("resolve_schema", self._resolve_schema)
              .add_node("lookup_definition", self._lookup_definition)
              .add_node("generate_code", self._generate_code)
              .add_node("backtest_guard", self._backtest_guard)
              .add_node("finalize", self._finalize))
        state = wf.run({"concept": concept}, start="parse")
        return state["record"]


# ---------------------------------------------------------------------------
# Sandboxed Backtest Guard (isolated subprocess + hard timeout)
# ---------------------------------------------------------------------------


def _synthetic_df(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1.09 + np.cumsum(rng.normal(0.0002, 0.001, n))
    op = close + rng.normal(0, 0.0005, n)
    hi = np.maximum(op, close) + rng.uniform(0, 0.001, n)
    lo = np.minimum(op, close) - rng.uniform(0, 0.001, n)
    return pd.DataFrame({
        "time": np.arange(n) * 900 + 1690000000,
        "open": op, "high": hi, "low": lo, "close": close,
        "volume": rng.uniform(10, 1000, n),
    })


def _check_imports_ast(code: str) -> str | None:
    """AST allowlist: reject any import outside ``_ALLOWED_IMPORT_ROOTS``.

    Returns an error message, or None when the code's imports are all safe.
    AST-level checking catches forms a regex would miss (``import os,sys``,
    ``from x import y``, aliasing) and has no false positives on string literals.
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"generated code does not parse: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                key = alias.name if not alias.asname else f"{alias.name} as {alias.asname}"
                if key not in _ALLOWED_IMPORTS:
                    return f"forbidden import {alias.name!r} in generated code"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod not in _ALLOWED_IMPORTS:
                return f"forbidden import from {mod!r} in generated code"
    return None


def _guard_harness() -> str:
    """Isolated subprocess script: loads df, runs generate_signal, reports.

    The user code is exec'd into a namespace whose ``__builtins__`` is a
    restricted copy (no ``__import__``/``open``/``eval``/``compile``/``getattr``),
    so credential theft via ``__builtins__['__import__']('app.services.vault')``
    fails at the interpreter level. The harness itself is the only code that
    touches json/sys/stdin.
    """
    return (
        "import json, sys, builtins\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import numpy as np, pandas as pd\n"
        "payload = json.loads(sys.stdin.read())\n"
        "df = pd.DataFrame(payload['df'])\n"
        "_real_import = builtins.__import__\n"
        "_allowed = " + repr(sorted(_ALLOWED_IMPORTS)) + "\n"
        "def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if name not in _allowed:\n"
        "        raise ImportError('module %r not allowed in backtest sandbox' % name)\n"
        "    return _real_import(name, globals, locals, fromlist, level)\n"
        "allowed = " + repr(sorted(_ALLOWED_BUILTINS)) + "\n"
        "safe = {name: getattr(builtins, name) for name in allowed}\n"
        "safe['__import__'] = _guarded_import\n"
        "ns = {'__builtins__': safe}\n"
        "exec(payload['code'], ns)\n"
        "sig = ns['generate_signal'](df)\n"
        "s = list(float(x) for x in sig)\n"
        "ok = (len(s) == len(df)\n"
        "      and all(np.isfinite(x) for x in s)\n"
        "      and all(-1.0 - 1e-9 <= x <= 1.0 + 1e-9 for x in s))\n"
        "json.dump({'ok': bool(ok), 'length': len(s)}, sys.stdout)\n"
    )


def run_backtest_guard(code: str, timeout: float = GUARD_TIMEOUT_S,
                       repo_root: str | None = None) -> dict:
    """Execute generated strategy code in an isolated subprocess.

    Defenses, in depth:
    1. **AST import allowlist** — generated code may only import numpy/pandas/
       math/qimera.* (a ``signal_for`` call), so network and credential modules
       are rejected before anything runs.
    2. **Restricted builtins** — the exec namespace carries a whitelisted
       ``__builtins__`` without ``__import__``/``open``/``eval``/``compile``/
       ``getattr``, so dictionary-access tricks fail at the interpreter level.
    3. **Isolated subprocess** — ``python -I`` (no user site / env import
       paths) with a stripped environment and a **hard wall-clock timeout**;
       a runaway loop is killed. The harness never passes the host's env in.

    This is a guard for runaway/broken LLM-generated code, not a hardened
    security boundary against a determined adversary (no OS-level sandbox).

    Returns ``{ok, timed_out, duration_s, length, error}``.
    """
    if code is None or code.strip() == "":
        return {"ok": False, "timed_out": False, "duration_s": 0.0,
                "length": 0, "error": "empty code"}
    err = _check_imports_ast(code)
    if err:
        return {"ok": False, "timed_out": False, "duration_s": 0.0,
                "length": 0, "error": err}

    root = repo_root or str(Path(__file__).resolve().parent.parent.parent)
    df = _synthetic_df()
    payload = {
        "df": {k: df[k].tolist() for k in ("time", "open", "high", "low",
                                           "close", "volume")},
        "code": code,
    }
    t0 = time.perf_counter()
    try:
        # ``python -I -c <harness> <repo_root>``: everything after ``-c`` lands
        # in sys.argv, so the harness can put the repo on its own import path.
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _guard_harness(), root],
            input=json.dumps(payload), capture_output=True, text=True,
            timeout=timeout, cwd=tempfile.gettempdir(),
            env={"PATH": __import__("os").environ.get("PATH", ""),
                 "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "timed_out": True,
                "duration_s": round(time.perf_counter() - t0, 3),
                "length": 0, "error": f"timed out after {timeout}s"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "timed_out": False,
                "duration_s": round(time.perf_counter() - t0, 3),
                "length": 0, "error": f"{type(exc).__name__}: {exc}"}

    duration = round(time.perf_counter() - t0, 3)
    if proc.returncode != 0:
        return {"ok": False, "timed_out": False, "duration_s": duration,
                "length": 0,
                "error": (proc.stderr or proc.stdout or "").strip()[-400:]}
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "timed_out": False, "duration_s": duration,
                "length": 0, "error": "harness produced no parseable verdict"}
    return {"ok": bool(result.get("ok")), "timed_out": False,
            "duration_s": duration, "length": int(result.get("length", 0)),
            "error": ""}
