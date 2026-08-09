"""Strict sandbox for executing LLM-generated strategy code.

The codegen agent produces free-form Python. Before it is ever run we:
  1. statically reject dangerous imports / IO builtins via AST,
  2. compile it,
  3. import it into a clean module namespace (only backtesting/numpy/pandas/math),
  4. dry-run it on a real slice of OHLCV to prove it executes and trades.

If any step fails the code is rejected — never executed.
"""

import ast
import sys
import types

ALLOWED_IMPORTS = {"backtesting", "numpy", "pandas", "math"}


class UnsafeCodeError(Exception):
    pass


def _check_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                base = (a.name or "").split(".")[0]
                if base not in ALLOWED_IMPORTS:
                    raise UnsafeCodeError(f"forbidden import: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            base = (node.module or "").split(".")[0]
            if base not in ALLOWED_IMPORTS:
                raise UnsafeCodeError(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                name = func.attr
            if name in {"eval", "exec", "compile", "open", "__import__", "input"}:
                raise UnsafeCodeError(f"forbidden builtin: {name}")
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in {
                "os", "sys", "subprocess", "socket", "pathlib", "shutil",
                "requests", "urllib", "builtins", "glob", "locals", "vars",
            }:
                raise UnsafeCodeError(f"forbidden attribute access: {node.value.id}.{node.attr}")
        elif isinstance(node, ast.Name) and node.id in {"open", "__import__", "eval", "exec"}:
            raise UnsafeCodeError(f"forbidden name: {node.id}")


def validate_code(code: str) -> tuple[bool, str]:
    """Return (ok, error). ``ok`` is True only if the code is safe and compiles."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"
    try:
        _check_ast(tree)
    except UnsafeCodeError as exc:
        return False, str(exc)
    try:
        compile(tree, "<generated>", "exec")
    except Exception as exc:  # noqa: BLE001
        return False, f"compile error: {exc}"
    return True, ""


def _normalize_order_size(size):
    """Backtesting requires a `size` that is a fraction of equity (0<size<1) or a
    whole number of units (round(size)==size). LLM code frequently computes
    `risk_pct*equity/stop_distance`, which lands between 1 and an integer (e.g.
    312.5) — impossible for the broker. Map those to whole units."""
    if size is None:
        return size
    try:
        s = float(size)
    except (TypeError, ValueError):
        return size
    if not (s > 0):
        return size
    if s < 1:
        return s  # fractional equity allocation
    return float(int(round(s)) or 1)  # whole number of units


def build_strategy_class(code: str) -> type:
    """Compile + exec the generated code into a fresh module and return the
    GeneratedStrategy class. Compilation happens only after validate_code()
    passed. Order sizes are normalized through the buy/sell call sites so that
    valid-but-fractional unit counts can never crash the backtest."""
    ok, err = validate_code(code)
    if not ok:
        raise UnsafeCodeError(err)

    mod = types.ModuleType("_generated_strategy")
    sys.modules["_generated_strategy"] = mod
    exec(compile(code, "<generated>", "exec"), mod.__dict__)

    cls = None
    for name, obj in vars(mod).items():
        if isinstance(obj, type) and name == "GeneratedStrategy":
            cls = obj
            break
    if cls is None:
        raise UnsafeCodeError("generated code did not define 'GeneratedStrategy'")

    for method in ("buy", "sell"):
        base = getattr(cls, method)
        def wrap(size_sanitizer=_normalize_order_size, base=base):
            def inner(self, *args, **kwargs):
                if "size" in kwargs:
                    kwargs["size"] = size_sanitizer(kwargs["size"])
                return base(self, *args, **kwargs)
            return inner
        setattr(cls, method, wrap())

    # Defensive next(): generated code checks SL/TP geometry against the CURRENT
    # close, but the broker fills at the NEXT bar's open. A gap can make a
    # single bar's order invalid (e.g. TP >= entry on a short). That one bar
    # must be skipped, not allowed to crash the whole backtest run.
    base_next = getattr(cls, "next")
    def wrap_next(base=base_next):
        def inner(self, *args, **kwargs):
            try:
                return base(self, *args, **kwargs)
            except ValueError:
                return None
        return inner
    setattr(cls, "next", wrap_next())
    return cls


def dry_run_strategy(cls: type, df, bars: int = 300) -> dict:
    """Run the strategy class on the first `bars` rows to prove it executes and
    produces at least one trade. Returns {ok, trades, error}."""
    from backtesting import Backtest

    sub = df.iloc[:bars].copy()
    try:
        bt = Backtest(sub, cls, cash=100_000)
        stats = bt.run()
        trades = stats.get("_trades")
        count = len(trades) if trades is not None else 0
        return {
            "ok": True,
            "trades": int(count),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "trades": 0, "error": str(exc)}
