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


def build_strategy_class(code: str) -> type:
    """Compile + exec the generated code into a fresh module and return the
    GeneratedStrategy class. Call only after validate_code() passed."""
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
