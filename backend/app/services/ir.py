"""Intermediate Representation (IR) Compiler Layer + Dynamic Tool Registry (IR.md).

Sits between the Synthesizing Agent (text/concepts) and the Backtest / Refine Loop
(code execution). Two pieces:

* :class:`DynamicToolRegistry`: a self-documenting catalog of the mathematical
  primitives the backtest layer can actually compute from OHLCV. It introspects
  registered function signatures and emits a "Tool Schema Menu" JSON that is
  injected straight into the Synthesizer Agent prompt (and the code generator),
  so the agent only references primitives that exist in the codebase.

* :class:`IRCompilerEngine`: a syntax-checking guardrail that intercepts the
  synthesizer's ``ir_blueprint`` output, verifies every referenced primitive and
  its arguments against the registry, and fails forward safely:
  CRITICAL_ERROR (bad JSON) / VALIDATION_FAILED (hallucinated primitive, missing
  or mistyped argument) / SUCCESS. Defective rules are dropped and the defect
  list is returned so the closed self-correction loop can fix the Agent output
  instead of wasting backtest cycles on uncallable code.
"""

import inspect
import json
from collections.abc import Callable
from typing import Any


def _type_name(annotation: Any) -> str:
    if annotation == inspect.Parameter.empty:
        return "any"
    return getattr(annotation, "__name__", str(annotation))


class DynamicToolRegistry:
    """Self-documenting catalog of Python math primitives (OHLCV computations).

    A primitive is any function whose signature declares the parameters of the
    corresponding mathematical module. Registering one makes it part of the
    schema menu given to the AI agents AND the validation vocabulary of the
    compiler — no prompt-editing needed.
    """

    def __init__(self) -> None:
        self._registry: dict[str, dict[str, Any]] = {}

    def register_primitive(self, name: str, description: str) -> Callable:
        """Decorator: register a math primitive function under ``name``.

        Example:
            @registry.register_primitive("ATR_BAND",
                                         "Volatility-consistent stop/target band.")
            def atr_band(period: int = 14, sl_mult: float = 1.5,
                         tp_mult: float = 2.0) -> dict:
                ...
        """

        def decorator(func: Callable) -> Callable:
            sig = inspect.signature(func)
            parameters: dict[str, dict[str, Any]] = {}
            for param in sig.parameters.values():
                if param.name in ("self", "cls"):
                    continue
                is_required = param.default == inspect.Parameter.empty
                parameters[param.name] = {
                    "type": _type_name(param.annotation),
                    "default": "REQUIRED" if is_required else param.default,
                }
            self._registry[name] = {
                "primitive_name": name,
                "description": description,
                "parameters": parameters,
                "callable": func,
            }
            return func

        return decorator

    @property
    def primitives(self) -> list[str]:
        return sorted(self._registry.keys())

    def get(self, name: str) -> dict[str, Any] | None:
        return self._registry.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def generate_agent_prompt_schema(self) -> str:
        """Clean JSON menu of tools to inject into the Synthesizer system prompt."""
        schema = []
        for data in self._registry.values():
            schema.append({
                "primitive": data["primitive_name"],
                "functional_purpose": data["description"],
                "required_arguments": data["parameters"],
            })
        return json.dumps(schema, indent=2)

    def generate_markdown_menu(self) -> str:
        """Compact one-line-per-primitive menu (used by the code generator)."""
        lines = []
        for name in self.primitives:
            data = self._registry[name]
            params = ", ".join(
                f"{p}={m['type']}" for p, m in data["parameters"].items())
            lines.append(f"- {name}({params}): {data['description']}")
        return "\n".join(lines)


class IRCompilerEngine:
    """Syntax-checking guardrail between the synthesizer and the backtest loop.

    Validates an ``ir_blueprint`` JSON (a list of ``{"primitive": ...,
    "args": {...}}`` rules) against a :class:`DynamicToolRegistry` and fails
    forward safely, returning a machine-readable defect report the closed
    self-correction loop uses to repair the Agent's output.
    """

    STATUS_SUCCESS = "SUCCESS"
    STATUS_VALIDATION_FAILED = "VALIDATION_FAILED"
    STATUS_CRITICAL_ERROR = "CRITICAL_ERROR"

    def __init__(self, registry: DynamicToolRegistry) -> None:
        self.registry = registry

    def compile_dict(self, blueprint: Any) -> dict:
        """Compile an already-parsed blueprint structure (dict or list)."""
        if isinstance(blueprint, dict):
            rules = blueprint.get("rules", [])
        elif isinstance(blueprint, list):
            rules = blueprint
        else:
            return {
                "status": self.STATUS_CRITICAL_ERROR,
                "errors": ["ir_blueprint must be a JSON {\"rules\": [...]} object."],
                "fix_instruction": (
                    "Output the complete strategy ONLY as {'rules': "
                    "[{'primitive': ..., 'args': {...}}, ...]} using registered "
                    "primitives."),
            }
        if not isinstance(rules, list):
            return {
                "status": self.STATUS_CRITICAL_ERROR,
                "errors": ["'rules' in ir_blueprint must be a JSON array."],
                "fix_instruction": "Return 'rules' as a JSON array of primitives.",
            }

        validated_rules: list[dict] = []
        rejected_rules: list[dict] = []
        errors: list[str] = []

        for rule in rules:
            if not isinstance(rule, dict):
                errors.append(f"Rule is not an object: {rule!r}")
                rejected_rules.append({
                    "rule": rule, "reason": "rule is not an object"})
                continue
            primitive = rule.get("primitive")
            args = rule.get("args") or {}
            if not isinstance(primitive, str) or not primitive.strip():
                errors.append(f"Rule is missing a 'primitive' name: {rule!r}")
                rejected_rules.append({"rule": rule,
                                       "reason": "missing primitive name"})
                continue
            if primitive not in self.registry:
                errors.append(
                    f"Hallucinated Primitive: '{primitive}' does not exist in "
                    "the code registry.")
                rejected_rules.append({
                    "rule": rule,
                    "reason": f"unknown primitive '{primitive}'"})
                continue
            if not isinstance(args, dict):
                errors.append(f"Tool '{primitive}': 'args' must be a JSON object.")
                rejected_rules.append({"rule": rule, "reason": "args not an object"})
                continue

            expected = self.registry.get(primitive)["parameters"]
            rule_conflicts: bool = False
            for arg_name, arg_meta in expected.items():
                if arg_meta["default"] == "REQUIRED" and arg_name not in args:
                    errors.append(f"Missing Parameter: Tool '{primitive}' requires "
                                  f"argument '{arg_name}'.")
                    rule_conflicts = True
                    continue
                if arg_name not in args:
                    continue
                value = args[arg_name]
                expected_type = arg_meta["type"]
                if expected_type == "int":
                    valid_type = isinstance(value, int) and not isinstance(
                        value, bool)
                elif expected_type == "float":
                    valid_type = isinstance(value, (int, float)) and not isinstance(
                        value, bool)
                elif expected_type == "str":
                    valid_type = isinstance(value, str)
                elif expected_type == "bool":
                    valid_type = isinstance(value, bool)
                elif expected_type == "any":
                    valid_type = True
                else:
                    valid_type = True
                if not valid_type:
                    errors.append(f"Wrong Type: Tool '{primitive}' argument "
                                  f"'{arg_name}' must be {expected_type}, got "
                                  f"{type(value).__name__} ({value!r}).")
                    rule_conflicts = True

            if rule_conflicts:
                rejected_rules.append({
                    "rule": rule,
                    "reason": "argument validation failed (see errors)"})
            else:
                validated_rules.append(rule)

        if errors:
            return {
                "status": self.STATUS_VALIDATION_FAILED,
                "errors": errors,
                "validated_rules": validated_rules,
                "rejected_rules": rejected_rules,
                "fix_instruction": (
                    "Feed this defect block back to the Synthesizing Agent so it "
                    "corrects ONLY the rejected rules: replace hallucinated "
                    "primitives with registered ones and supply the missing "
                    "required arguments. Keep the accepted rules unchanged."),
            }

        return {
            "status": self.STATUS_SUCCESS,
            "compiled_blueprint": {"rules": [
                {"primitive": rule.get("primitive"), "args": rule.get("args") or {}}
                for rule in validated_rules]},
        }

    def compile_and_validate(self, agent_json_output: str) -> dict:
        """Compile a synthesized ``ir_blueprint`` JSON string output by an agent."""
        try:
            blueprint = json.loads(agent_json_output)
        except (json.JSONDecodeError, TypeError):
            return {
                "status": self.STATUS_CRITICAL_ERROR,
                "errors": [
                    "AI failed to output clean structural JSON for ir_blueprint."
                ],
                "fix_instruction": "Output the strategy ONLY as strict JSON.",
            }
        return self.compile_dict(blueprint)