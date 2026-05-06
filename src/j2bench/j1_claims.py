from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .io import load_yaml


DEFAULT_J1_CLAIMS_PATH = Path(__file__).resolve().parents[2] / "config" / "j1_claims.yaml"

_REQUIRED_CONFIG_KEYS = ("version", "program_id", "claim_surface", "claim_classes", "gates")
_REQUIRED_GATE_KEYS = ("primary_split_success", "comparator_support", "robustness_direction")
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "ci_lower": ("ci_lower", "ci_low", "lower_ci", "ci_lb"),
    "delta": ("delta", "value"),
    "abs": ("abs",),
}


def default_j1_claims_path() -> Path:
    return DEFAULT_J1_CLAIMS_PATH


def load_j1_claims(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else default_j1_claims_path()
    payload = load_yaml(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"J1 claims config must be a YAML mapping: {config_path}")
    _validate_config(payload)
    return payload


def evaluate_j1_claim_governance(
    evidence: Mapping[str, Any],
    *,
    requested_claim_classes: Iterable[str] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")

    governance = dict(config) if config is not None else load_j1_claims()
    _validate_config(governance)

    primary_splits = _normalize_named_rows(evidence.get("primary_splits"), key_field="split_family")
    supporting_datasets = _normalize_named_rows(
        evidence.get("supporting_datasets"),
        key_field="dataset_id",
    )

    gates = governance["gates"]
    gate_results = {
        "primary_split_success": _evaluate_primary_split_success(
            primary_splits,
            gates["primary_split_success"],
        ),
        "comparator_support": _evaluate_comparator_support(
            primary_splits,
            gates["comparator_support"],
        ),
        "robustness_direction": _evaluate_robustness_direction(
            supporting_datasets,
            gates["robustness_direction"],
        ),
    }

    claim_classes = governance["claim_classes"]
    requested = list(requested_claim_classes or _default_requested_claims(claim_classes))
    claim_results: dict[str, dict[str, Any]] = {}
    allowed: list[str] = []
    blocked: list[str] = []
    disallowed: list[str] = []

    for claim_class in requested:
        if claim_class not in claim_classes:
            raise KeyError(f"unknown J1 claim class: {claim_class}")
        spec = claim_classes[claim_class]
        if not bool(spec["allowed"]):
            disallowed.append(claim_class)
            claim_results[claim_class] = {
                "status": "disallowed",
                "reason": str(spec.get("reason", "")),
                "disallowed_terms": list(spec.get("disallowed_terms", [])),
            }
            continue

        required_gates = [str(gate_name) for gate_name in spec.get("requires", [])]
        failed_gates = [
            gate_name
            for gate_name in required_gates
            if gate_results[gate_name]["status"] != "pass"
        ]
        if failed_gates:
            blocked.append(claim_class)
            claim_results[claim_class] = {
                "status": "blocked",
                "required_gates": required_gates,
                "failed_gates": failed_gates,
            }
            continue

        allowed.append(claim_class)
        claim_results[claim_class] = {
            "status": "allowed",
            "required_gates": required_gates,
            "failed_gates": [],
        }

    overall_status = "NO_GO" if disallowed else "HOLD" if blocked else "GO"
    return {
        "program_id": str(governance["program_id"]),
        "claim_surface": str(governance["claim_surface"]),
        "overall_status": overall_status,
        "requested_claim_classes": requested,
        "allowed_claim_classes": [
            claim_id
            for claim_id, spec in claim_classes.items()
            if bool(spec["allowed"])
        ],
        "disallowed_claim_classes": [
            claim_id
            for claim_id, spec in claim_classes.items()
            if not bool(spec["allowed"])
        ],
        "gate_results": gate_results,
        "claim_results": claim_results,
        "supported_claims": allowed,
        "blocked_claims": blocked,
        "rejected_claims": disallowed,
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    missing = [key for key in _REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"J1 claims config missing required keys: {', '.join(missing)}")

    claim_classes = config["claim_classes"]
    gates = config["gates"]
    if not isinstance(claim_classes, Mapping) or not claim_classes:
        raise ValueError("claim_classes must be a non-empty mapping")
    if not isinstance(gates, Mapping):
        raise ValueError("gates must be a mapping")

    missing_gates = [gate_name for gate_name in _REQUIRED_GATE_KEYS if gate_name not in gates]
    if missing_gates:
        raise ValueError(
            f"J1 claims config missing required gates: {', '.join(missing_gates)}"
        )

    for claim_class, spec in claim_classes.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"claim class {claim_class!r} must be a mapping")
        if "allowed" not in spec:
            raise ValueError(f"claim class {claim_class!r} must declare allowed")
        if not isinstance(spec["allowed"], bool):
            raise ValueError(f"claim class {claim_class!r} allowed must be boolean")
        if bool(spec["allowed"]):
            requires = spec.get("requires", [])
            if not isinstance(requires, list):
                raise ValueError(f"claim class {claim_class!r} requires must be a list")
            unknown = [gate_name for gate_name in requires if gate_name not in gates]
            if unknown:
                raise ValueError(
                    f"claim class {claim_class!r} references unknown gates: {', '.join(unknown)}"
                )
        else:
            disallowed_terms = spec.get("disallowed_terms", [])
            if not isinstance(disallowed_terms, list):
                raise ValueError(
                    f"claim class {claim_class!r} disallowed_terms must be a list"
                )


def _default_requested_claims(claim_classes: Mapping[str, Any]) -> list[str]:
    return [
        str(claim_id)
        for claim_id, spec in claim_classes.items()
        if bool(spec.get("allowed", False))
    ]


def _normalize_named_rows(payload: Any, *, key_field: str) -> dict[str, dict[str, Any]]:
    if payload is None:
        return {}

    if isinstance(payload, Mapping):
        normalized: dict[str, dict[str, Any]] = {}
        for key, row in payload.items():
            if not isinstance(row, Mapping):
                raise TypeError(f"{key_field} payload entries must be mappings")
            record = dict(row)
            record.setdefault(key_field, key)
            normalized[str(key)] = record
        return normalized

    if not isinstance(payload, Iterable) or isinstance(payload, (str, bytes)):
        raise TypeError(f"{key_field} payload must be a mapping or iterable of mappings")

    normalized = {}
    for row in payload:
        if not isinstance(row, Mapping):
            raise TypeError(f"{key_field} payload entries must be mappings")
        if key_field not in row:
            raise ValueError(f"{key_field} payload rows must include {key_field}")
        key = str(row[key_field])
        normalized[key] = dict(row)
    return normalized


def _evaluate_primary_split_success(
    primary_splits: Mapping[str, dict[str, Any]],
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    required_splits = [str(split) for split in gate_config.get("required_split_families", [])]
    min_passing_splits = int(gate_config.get("min_passing_splits", len(required_splits)))
    primary_metric = gate_config["primary_metric"]
    guardrails = gate_config.get("guardrails", {})

    passing_splits: list[str] = []
    missing_splits: list[str] = []
    failures: dict[str, list[str]] = {}

    for split_family in required_splits:
        row = primary_splits.get(split_family)
        if row is None:
            missing_splits.append(split_family)
            continue

        split_failures: list[str] = []
        if not _metric_rule_passes(row, metric_name=str(primary_metric["name"]), rule=primary_metric):
            observed = _extract_metric_value(row, str(primary_metric["name"]), str(primary_metric["field"]))
            split_failures.append(
                f"{primary_metric['name']}:{primary_metric['field']}={observed!r}"
            )

        for metric_name, rule in guardrails.items():
            if not _metric_rule_passes(row, metric_name=str(metric_name), rule=rule):
                observed = _extract_metric_value(row, str(metric_name), str(rule["field"]))
                split_failures.append(f"{metric_name}:{rule['field']}={observed!r}")

        if split_failures:
            failures[split_family] = split_failures
        else:
            passing_splits.append(split_family)

    status = (
        "pass"
        if not missing_splits and not failures and len(passing_splits) >= min_passing_splits
        else "fail"
    )
    return {
        "status": status,
        "required_split_families": required_splits,
        "min_passing_splits": min_passing_splits,
        "passing_splits": passing_splits,
        "missing_splits": missing_splits,
        "failing_splits": failures,
    }


def _evaluate_comparator_support(
    primary_splits: Mapping[str, dict[str, Any]],
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    required_splits = [str(split) for split in gate_config.get("required_split_families", [])]
    min_supported_splits = int(gate_config.get("min_supported_splits", len(required_splits)))
    required_family = str(gate_config.get("comparison_family", ""))
    baseline_prefix = str(gate_config.get("baseline_policy_prefix", ""))

    supported_splits: list[str] = []
    missing_splits: list[str] = []
    failing_splits: dict[str, dict[str, Any]] = {}

    for split_family in required_splits:
        row = primary_splits.get(split_family)
        if row is None:
            missing_splits.append(split_family)
            continue

        family = str(row.get("comparison_family", ""))
        baseline = str(row.get("baseline_policy", ""))
        family_ok = family == required_family
        baseline_ok = baseline.startswith(baseline_prefix)

        if family_ok and baseline_ok:
            supported_splits.append(split_family)
        else:
            failing_splits[split_family] = {
                "comparison_family": family,
                "baseline_policy": baseline,
                "expected_comparison_family": required_family,
                "expected_baseline_policy_prefix": baseline_prefix,
            }

    status = (
        "pass"
        if not missing_splits
        and not failing_splits
        and len(supported_splits) >= min_supported_splits
        else "fail"
    )
    return {
        "status": status,
        "required_split_families": required_splits,
        "min_supported_splits": min_supported_splits,
        "supported_splits": supported_splits,
        "missing_splits": missing_splits,
        "failing_splits": failing_splits,
    }


def _evaluate_robustness_direction(
    supporting_datasets: Mapping[str, dict[str, Any]],
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    eligible_datasets = [str(dataset_id) for dataset_id in gate_config.get("eligible_datasets", [])]
    min_supporting_datasets = int(
        gate_config.get("min_supporting_datasets", len(eligible_datasets))
    )
    metric_rules = gate_config.get("metrics", {})

    considered = [
        dataset_id
        for dataset_id in eligible_datasets
        if dataset_id in supporting_datasets
    ]
    ignored = sorted(
        dataset_id
        for dataset_id in supporting_datasets
        if dataset_id not in set(eligible_datasets)
    )

    passing_datasets: list[str] = []
    failures: dict[str, list[str]] = {}

    for dataset_id in considered:
        row = supporting_datasets[dataset_id]
        dataset_failures: list[str] = []
        for metric_name, rule in metric_rules.items():
            if not _metric_rule_passes(row, metric_name=str(metric_name), rule=rule):
                observed = _extract_metric_value(row, str(metric_name), str(rule["field"]))
                dataset_failures.append(f"{metric_name}:{rule['field']}={observed!r}")
        if dataset_failures:
            failures[dataset_id] = dataset_failures
        else:
            passing_datasets.append(dataset_id)

    status = "pass" if len(passing_datasets) >= min_supporting_datasets else "fail"
    return {
        "status": status,
        "eligible_datasets": eligible_datasets,
        "min_supporting_datasets": min_supporting_datasets,
        "considered_datasets": considered,
        "passing_datasets": passing_datasets,
        "failing_datasets": failures,
        "ignored_datasets": ignored,
    }


def _metric_rule_passes(
    row: Mapping[str, Any],
    *,
    metric_name: str,
    rule: Mapping[str, Any],
) -> bool:
    field = str(rule["field"])
    operator = str(rule["operator"])
    threshold = float(rule["value"])
    observed = _extract_metric_value(row, metric_name, field)
    if observed is None:
        return False
    return _compare(observed, operator, threshold)


def _extract_metric_value(
    row: Mapping[str, Any],
    metric_name: str,
    field: str,
) -> float | None:
    metrics = row.get("metrics", {})
    metric_payload = metrics.get(metric_name) if isinstance(metrics, Mapping) else None

    if isinstance(metric_payload, Mapping):
        aliases = _FIELD_ALIASES.get(field, (field,))
        for alias in aliases:
            if alias in metric_payload:
                return _coerce_float(metric_payload[alias])
        if field == "abs":
            for alias in ("delta", "value"):
                if alias in metric_payload:
                    value = _coerce_float(metric_payload[alias])
                    return None if value is None else abs(value)
        return None

    if isinstance(metric_payload, (int, float)) and field in {"delta", "value"}:
        return float(metric_payload)

    direct_value = row.get(metric_name)
    if isinstance(direct_value, (int, float)) and field in {"delta", "value"}:
        return float(direct_value)
    return None


def _compare(observed: float, operator: str, threshold: float) -> bool:
    if operator == "gt":
        return observed > threshold
    if operator == "gte":
        return observed >= threshold
    if operator == "lt":
        return observed < threshold
    if operator == "lte":
        return observed <= threshold
    raise ValueError(f"unsupported comparison operator: {operator}")


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

