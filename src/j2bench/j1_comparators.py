"""Governance helpers for freezing the J1-open comparator lock."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import isclose
from types import MappingProxyType
import re
from typing import Literal, Mapping, Sequence


J1_PRIMARY_COMPARATOR = "matched_confidence_threshold"
J1_REQUIRED_COMPARATORS = (
    "user_only",
    "assist_only",
    "confidence_blend",
    "selective_prediction",
    J1_PRIMARY_COMPARATOR,
    "set_acsa",
    "agency_margin",
)
J1_MATCHED_CONFIDENCE_POLICY_PREFIX = "plain_conf_threshold_matched_tau_"


@dataclass(frozen=True)
class J1MatchedComparatorSpec:
    tau: str
    policy_name: str
    threshold: float
    target_intervention_rate: float
    matched_intervention_rate: float
    intervention_rate_gap: float | None = None

    def __post_init__(self) -> None:
        tau = _normalize_tau(self.tau)
        expected_policy_name = _matched_policy_name(tau)
        provided_policy = str(self.policy_name).strip()
        if provided_policy and provided_policy != expected_policy_name:
            raise ValueError(
                f"matched comparator policy '{provided_policy}' does not match tau {tau}"
            )
        threshold = _normalize_probability(
            self.threshold,
            field_name="threshold",
            upper=1.01,
        )
        target_rate = _normalize_probability(
            self.target_intervention_rate,
            field_name="target_intervention_rate",
        )
        matched_rate = _normalize_probability(
            self.matched_intervention_rate,
            field_name="matched_intervention_rate",
        )
        computed_gap = abs(matched_rate - target_rate)
        declared_gap = self.intervention_rate_gap
        if declared_gap is None:
            frozen_gap = computed_gap
        else:
            frozen_gap = _normalize_probability(
                declared_gap,
                field_name="intervention_rate_gap",
            )
            if not isclose(frozen_gap, computed_gap, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "intervention_rate_gap must equal "
                    "abs(matched_intervention_rate - target_intervention_rate)"
                )
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "policy_name", expected_policy_name)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "target_intervention_rate", target_rate)
        object.__setattr__(self, "matched_intervention_rate", matched_rate)
        object.__setattr__(self, "intervention_rate_gap", frozen_gap)

    def within_tolerance(self, tolerance: float) -> bool:
        tolerance_value = _normalize_probability(
            tolerance,
            field_name="intervention_rate_tolerance",
        )
        return bool(self.intervention_rate_gap <= tolerance_value + 1e-12)

    def to_dict(self) -> dict[str, object]:
        return {
            "tau": self.tau,
            "policy_name": self.policy_name,
            "threshold": self.threshold,
            "target_intervention_rate": self.target_intervention_rate,
            "matched_intervention_rate": self.matched_intervention_rate,
            "intervention_rate_gap": self.intervention_rate_gap,
        }


@dataclass(frozen=True)
class FrozenJ1ComparatorLock:
    lock_name: str
    dataset_id: str
    comparator_policies: tuple[str, ...]
    primary_comparator: str
    intervention_rate_tolerance: float
    primary_tau_order: tuple[str, ...]
    primary_tau_manifest: Mapping[str, J1MatchedComparatorSpec]

    def __post_init__(self) -> None:
        lock_name = str(self.lock_name).strip() or "J1-open"
        dataset_id = str(self.dataset_id).strip() or "j1"
        comparator_policies = _freeze_policy_tuple(
            self.comparator_policies,
            field_name="comparator_policies",
        )
        primary_comparator = _normalize_policy_name(self.primary_comparator)
        if primary_comparator != J1_PRIMARY_COMPARATOR:
            raise ValueError(
                "J1-open primary comparator must be the matched confidence-threshold gate"
            )
        if primary_comparator not in comparator_policies:
            raise ValueError(
                f"primary comparator '{primary_comparator}' must be present in comparator_policies"
            )
        intervention_rate_tolerance = _normalize_probability(
            self.intervention_rate_tolerance,
            field_name="intervention_rate_tolerance",
        )
        primary_tau_order = _freeze_tau_tuple(
            self.primary_tau_order,
            field_name="primary_tau_order",
        )
        manifest_map = dict(self.primary_tau_manifest)
        if not manifest_map:
            raise ValueError(
                "matched confidence-threshold primary comparator requires an explicit tau manifest"
            )
        if len(manifest_map) != len(primary_tau_order):
            raise ValueError("primary_tau_order must list every frozen tau exactly once")
        if set(manifest_map) != set(primary_tau_order):
            raise ValueError("primary_tau_order and primary_tau_manifest must reference the same taus")
        for tau, spec in manifest_map.items():
            if spec.tau != tau:
                raise ValueError(
                    f"tau manifest key '{tau}' does not match comparator spec tau '{spec.tau}'"
                )
        object.__setattr__(self, "lock_name", lock_name)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "comparator_policies", comparator_policies)
        object.__setattr__(self, "primary_comparator", primary_comparator)
        object.__setattr__(
            self,
            "intervention_rate_tolerance",
            intervention_rate_tolerance,
        )
        object.__setattr__(self, "primary_tau_order", primary_tau_order)
        object.__setattr__(self, "primary_tau_manifest", MappingProxyType(manifest_map))

    def get_primary_tau_spec(self, tau: str) -> J1MatchedComparatorSpec | None:
        return self.primary_tau_manifest.get(_normalize_tau(tau))

    def to_dict(self) -> dict[str, object]:
        return {
            "lock_name": self.lock_name,
            "dataset_id": self.dataset_id,
            "comparator_policies": list(self.comparator_policies),
            "primary_comparator": self.primary_comparator,
            "intervention_rate_tolerance": self.intervention_rate_tolerance,
            "primary_tau_manifest": [
                self.primary_tau_manifest[tau].to_dict() for tau in self.primary_tau_order
            ],
        }


@dataclass(frozen=True)
class J1PrimaryComparatorSelection:
    status: Literal["selected", "no_selection"]
    comparator_policy: str | None
    tau: str | None
    threshold: float | None
    target_intervention_rate: float | None
    matched_intervention_rate: float | None
    intervention_rate_gap: float | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "comparator_policy": self.comparator_policy,
            "tau": self.tau,
            "threshold": self.threshold,
            "target_intervention_rate": self.target_intervention_rate,
            "matched_intervention_rate": self.matched_intervention_rate,
            "intervention_rate_gap": self.intervention_rate_gap,
            "reason": self.reason,
        }


def freeze_matched_confidence_tau_manifest(
    rows: Sequence[J1MatchedComparatorSpec | Mapping[str, object]],
) -> tuple[tuple[str, ...], Mapping[str, J1MatchedComparatorSpec]]:
    if not rows:
        raise ValueError(
            "matched confidence-threshold primary comparator requires an explicit tau manifest"
        )
    tau_order: list[str] = []
    manifest: dict[str, J1MatchedComparatorSpec] = {}
    for row in rows:
        spec = _coerce_matched_spec(row)
        if spec.tau in manifest:
            raise ValueError(f"duplicate matched comparator tau in manifest: {spec.tau}")
        tau_order.append(spec.tau)
        manifest[spec.tau] = spec
    return tuple(tau_order), MappingProxyType(manifest)


def freeze_j1_primary_comparator_set(
    primary_tau_manifest: Sequence[J1MatchedComparatorSpec | Mapping[str, object]],
    *,
    comparator_policies: Sequence[str] = J1_REQUIRED_COMPARATORS,
    lock_name: str = "J1-open",
    dataset_id: str = "j1",
    primary_comparator: str = J1_PRIMARY_COMPARATOR,
    intervention_rate_tolerance: float,
) -> FrozenJ1ComparatorLock:
    tau_order, manifest = freeze_matched_confidence_tau_manifest(primary_tau_manifest)
    return FrozenJ1ComparatorLock(
        lock_name=lock_name,
        dataset_id=dataset_id,
        comparator_policies=tuple(comparator_policies),
        primary_comparator=primary_comparator,
        intervention_rate_tolerance=intervention_rate_tolerance,
        primary_tau_order=tau_order,
        primary_tau_manifest=manifest,
    )


def matches_frozen_j1_comparator_set(
    comparator_lock: FrozenJ1ComparatorLock,
    comparator_policies: Sequence[str],
) -> bool:
    try:
        observed = _freeze_policy_tuple(
            comparator_policies,
            field_name="comparator_policies",
        )
    except ValueError:
        return False
    return set(comparator_lock.comparator_policies) == set(observed)


def select_j1_primary_comparator(
    comparator_lock: FrozenJ1ComparatorLock,
    *,
    tau: str,
    comparator_policies: Sequence[str],
) -> J1PrimaryComparatorSelection:
    normalized_tau = _normalize_tau(tau)
    if not matches_frozen_j1_comparator_set(comparator_lock, comparator_policies):
        return J1PrimaryComparatorSelection(
            status="no_selection",
            comparator_policy=None,
            tau=normalized_tau,
            threshold=None,
            target_intervention_rate=None,
            matched_intervention_rate=None,
            intervention_rate_gap=None,
            reason="observed comparator set does not match the frozen J1 comparator lock",
        )
    spec = comparator_lock.get_primary_tau_spec(normalized_tau)
    if spec is None:
        return J1PrimaryComparatorSelection(
            status="no_selection",
            comparator_policy=None,
            tau=normalized_tau,
            threshold=None,
            target_intervention_rate=None,
            matched_intervention_rate=None,
            intervention_rate_gap=None,
            reason="requested tau is absent from the frozen primary comparator manifest",
        )
    if not spec.within_tolerance(comparator_lock.intervention_rate_tolerance):
        return J1PrimaryComparatorSelection(
            status="no_selection",
            comparator_policy=spec.policy_name,
            tau=spec.tau,
            threshold=spec.threshold,
            target_intervention_rate=spec.target_intervention_rate,
            matched_intervention_rate=spec.matched_intervention_rate,
            intervention_rate_gap=spec.intervention_rate_gap,
            reason=(
                "frozen matched confidence-threshold comparator exceeds the "
                "declared intervention-rate tolerance"
            ),
        )
    return J1PrimaryComparatorSelection(
        status="selected",
        comparator_policy=spec.policy_name,
        tau=spec.tau,
        threshold=spec.threshold,
        target_intervention_rate=spec.target_intervention_rate,
        matched_intervention_rate=spec.matched_intervention_rate,
        intervention_rate_gap=spec.intervention_rate_gap,
        reason=(
            "selected the frozen matched confidence-threshold comparator for the "
            "requested tau"
        ),
    )


def _coerce_matched_spec(
    row: J1MatchedComparatorSpec | Mapping[str, object],
) -> J1MatchedComparatorSpec:
    if isinstance(row, J1MatchedComparatorSpec):
        return row
    payload = dict(row)
    tau = payload.get("tau")
    if tau is None:
        raise ValueError("matched comparator rows must declare tau")
    return J1MatchedComparatorSpec(
        tau=str(tau),
        policy_name=str(payload.get("policy_name") or payload.get("policy") or "").strip(),
        threshold=payload["threshold"],
        target_intervention_rate=payload["target_intervention_rate"],
        matched_intervention_rate=payload["matched_intervention_rate"],
        intervention_rate_gap=payload.get("intervention_rate_gap"),
    )


def _matched_policy_name(tau: str) -> str:
    return f"{J1_MATCHED_CONFIDENCE_POLICY_PREFIX}{_normalize_tau(tau)}"


def _freeze_policy_tuple(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    policies: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = _normalize_policy_name(value)
        if token in seen:
            continue
        seen.add(token)
        policies.append(token)
    if not policies:
        raise ValueError(f"{field_name} must include at least one comparator policy")
    return tuple(policies)


def _freeze_tau_tuple(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    taus: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = _normalize_tau(value)
        if token in seen:
            continue
        seen.add(token)
        taus.append(token)
    if not taus:
        raise ValueError(f"{field_name} must include at least one tau")
    return tuple(taus)


def _normalize_policy_name(value: object) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if not token:
        raise ValueError("comparator policy names must be non-empty")
    return token


def _normalize_tau(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("tau values must be non-empty")
    if text.lower().startswith("tau_"):
        text = text[4:]
    try:
        tau = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"invalid tau value: {value}") from exc
    if tau < 0:
        raise ValueError(f"tau values must be non-negative: {value}")
    scale = max(-tau.as_tuple().exponent, 0)
    if scale <= 2:
        return format(tau, ".2f")
    return format(tau.normalize(), "f")


def _normalize_probability(
    value: object,
    *,
    field_name: str,
    upper: float = 1.0,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite numeric value") from exc
    if numeric < 0.0 or numeric > upper:
        raise ValueError(f"{field_name} must be in [0, {upper}]")
    return numeric


__all__ = [
    "FrozenJ1ComparatorLock",
    "J1MatchedComparatorSpec",
    "J1PrimaryComparatorSelection",
    "J1_MATCHED_CONFIDENCE_POLICY_PREFIX",
    "J1_PRIMARY_COMPARATOR",
    "J1_REQUIRED_COMPARATORS",
    "freeze_j1_primary_comparator_set",
    "freeze_matched_confidence_tau_manifest",
    "matches_frozen_j1_comparator_set",
    "select_j1_primary_comparator",
]

