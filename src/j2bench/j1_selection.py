"""Governance helpers for freezing and promoting J1-open candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import re
from types import MappingProxyType
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
J1_LANES = ("canonical", "challenger")


@dataclass(frozen=True)
class J1Candidate:
    candidate_id: str
    user_model: str
    assist_model: str
    lane: Literal["canonical", "challenger"]
    allowed_taus: tuple[str, ...]
    evidence_tier: str = ""
    sensor_budget: str = "matched"
    priority: int | None = None
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lane = _normalize_lane(self.lane)
        object.__setattr__(self, "candidate_id", _normalize_nonempty_text(self.candidate_id, field_name="candidate_id"))
        object.__setattr__(self, "user_model", _normalize_nonempty_text(self.user_model, field_name="user_model"))
        object.__setattr__(self, "assist_model", _normalize_nonempty_text(self.assist_model, field_name="assist_model"))
        object.__setattr__(self, "lane", lane)
        object.__setattr__(self, "allowed_taus", _freeze_tau_tuple(self.allowed_taus, field_name="allowed_taus"))
        object.__setattr__(self, "evidence_tier", _normalize_evidence_tier(self.evidence_tier, lane=lane))
        object.__setattr__(self, "sensor_budget", _normalize_nonempty_text(self.sensor_budget, field_name="sensor_budget"))
        object.__setattr__(self, "notes", _freeze_text_tuple(self.notes))
        object.__setattr__(self, "metadata", _freeze_string_mapping(self.metadata))
        if self.priority is not None and int(self.priority) < 0:
            raise ValueError("priority must be non-negative when provided")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "user_model": self.user_model,
            "assist_model": self.assist_model,
            "lane": self.lane,
            "allowed_taus": list(self.allowed_taus),
            "evidence_tier": self.evidence_tier,
            "sensor_budget": self.sensor_budget,
            "priority": self.priority,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FrozenJ1CandidateRegistry:
    registry_name: str
    dataset_id: str
    comparator_policies: tuple[str, ...]
    primary_comparator: str
    candidate_order: tuple[str, ...]
    candidates: Mapping[str, J1Candidate]

    def __post_init__(self) -> None:
        registry_name = str(self.registry_name).strip() or "J1-open"
        dataset_id = str(self.dataset_id).strip() or "j1"
        primary_comparator = _normalize_policy_name(self.primary_comparator)
        comparator_policies = _freeze_policy_tuple(self.comparator_policies, field_name="comparator_policies")
        if primary_comparator not in comparator_policies:
            raise ValueError(
                f"primary comparator '{primary_comparator}' must be present in comparator_policies"
            )
        candidate_order = _freeze_text_tuple(self.candidate_order)
        candidate_map = dict(self.candidates)
        if not candidate_map:
            raise ValueError("at least one J1 candidate must be frozen in the registry")
        if len(candidate_map) != len(candidate_order):
            raise ValueError("candidate_order must list every frozen candidate exactly once")
        if set(candidate_map) != set(candidate_order):
            raise ValueError("candidate_order and candidates must reference the same candidate IDs")
        for candidate_id, candidate in candidate_map.items():
            if candidate.candidate_id != candidate_id:
                raise ValueError(f"registry key '{candidate_id}' does not match candidate_id '{candidate.candidate_id}'")
        object.__setattr__(self, "registry_name", registry_name)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "primary_comparator", primary_comparator)
        object.__setattr__(self, "comparator_policies", comparator_policies)
        object.__setattr__(self, "candidate_order", candidate_order)
        object.__setattr__(self, "candidates", MappingProxyType(candidate_map))

    def candidate_ids_for_lane(self, lane: str) -> tuple[str, ...]:
        normalized_lane = _normalize_lane(lane)
        return tuple(
            candidate_id
            for candidate_id in self.candidate_order
            if self.candidates[candidate_id].lane == normalized_lane
        )

    def get_candidate(self, candidate_id: str) -> J1Candidate:
        normalized_candidate_id = _normalize_nonempty_text(candidate_id, field_name="candidate_id")
        try:
            return self.candidates[normalized_candidate_id]
        except KeyError as exc:
            raise KeyError(f"candidate '{normalized_candidate_id}' is not present in the frozen J1 registry") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "registry_name": self.registry_name,
            "dataset_id": self.dataset_id,
            "comparator_policies": list(self.comparator_policies),
            "primary_comparator": self.primary_comparator,
            "candidate_order": list(self.candidate_order),
            "candidates": [self.candidates[candidate_id].to_dict() for candidate_id in self.candidate_order],
        }


@dataclass(frozen=True)
class J1CandidateOutcome:
    candidate_id: str
    comparator_policies: tuple[str, ...]
    qualifying_taus: tuple[str, ...]
    primary_gate_passed: bool
    safety_gate_passed: bool
    support_gate_passed: bool = True
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _normalize_nonempty_text(self.candidate_id, field_name="candidate_id"))
        object.__setattr__(self, "comparator_policies", _freeze_policy_tuple(self.comparator_policies, field_name="comparator_policies"))
        object.__setattr__(self, "qualifying_taus", _freeze_tau_tuple(self.qualifying_taus, field_name="qualifying_taus", allow_empty=True))
        object.__setattr__(self, "notes", _freeze_text_tuple(self.notes))

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "comparator_policies": list(self.comparator_policies),
            "qualifying_taus": list(self.qualifying_taus),
            "primary_gate_passed": self.primary_gate_passed,
            "safety_gate_passed": self.safety_gate_passed,
            "support_gate_passed": self.support_gate_passed,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class J1TauSelection:
    candidate_id: str
    status: Literal["selected", "no_selection"]
    selected_tau: str | None
    eligible_taus: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "selected_tau": self.selected_tau,
            "eligible_taus": list(self.eligible_taus),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class J1CandidateAssessment:
    candidate_id: str
    lane: Literal["canonical", "challenger"]
    status: Literal[
        "promotable",
        "missing_outcome",
        "comparator_mismatch",
        "no_qualifying_tau",
        "primary_gate_failed",
        "safety_gate_failed",
        "support_gate_failed",
    ]
    promotable: bool
    selected_tau: str | None
    eligible_taus: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "lane": self.lane,
            "status": self.status,
            "promotable": self.promotable,
            "selected_tau": self.selected_tau,
            "eligible_taus": list(self.eligible_taus),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class J1PromotionDecision:
    lane: Literal["canonical", "challenger"]
    outcome: Literal["promoted", "no_winner"]
    candidate_id: str | None
    selected_tau: str | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "lane": self.lane,
            "outcome": self.outcome,
            "candidate_id": self.candidate_id,
            "selected_tau": self.selected_tau,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class J1PromotionReport:
    registry_name: str
    dataset_id: str
    comparator_policies: tuple[str, ...]
    canonical: J1PromotionDecision
    challenger: J1PromotionDecision

    def to_dict(self) -> dict[str, object]:
        return {
            "registry_name": self.registry_name,
            "dataset_id": self.dataset_id,
            "comparator_policies": list(self.comparator_policies),
            "canonical": self.canonical.to_dict(),
            "challenger": self.challenger.to_dict(),
        }


def freeze_j1_candidate_registry(
    candidates: Sequence[J1Candidate],
    *,
    comparator_policies: Sequence[str] = J1_REQUIRED_COMPARATORS,
    registry_name: str = "J1-open",
    dataset_id: str = "j1",
    primary_comparator: str = J1_PRIMARY_COMPARATOR,
) -> FrozenJ1CandidateRegistry:
    if not candidates:
        raise ValueError("at least one J1 candidate must be provided")
    candidate_map: dict[str, J1Candidate] = {}
    candidate_order: list[str] = []
    for candidate in candidates:
        if candidate.candidate_id in candidate_map:
            raise ValueError(f"duplicate candidate_id in registry: {candidate.candidate_id}")
        candidate_map[candidate.candidate_id] = candidate
        candidate_order.append(candidate.candidate_id)
    return FrozenJ1CandidateRegistry(
        registry_name=registry_name,
        dataset_id=dataset_id,
        comparator_policies=tuple(comparator_policies),
        primary_comparator=primary_comparator,
        candidate_order=tuple(candidate_order),
        candidates=MappingProxyType(candidate_map),
    )


def select_j1_tau(candidate: J1Candidate, qualifying_taus: Sequence[str]) -> J1TauSelection:
    observed_taus = _freeze_tau_tuple(qualifying_taus, field_name="qualifying_taus", allow_empty=True)
    observed_tau_set = set(observed_taus)
    eligible_taus = tuple(tau for tau in candidate.allowed_taus if tau in observed_tau_set)
    if not eligible_taus:
        return J1TauSelection(
            candidate_id=candidate.candidate_id,
            status="no_selection",
            selected_tau=None,
            eligible_taus=(),
            reason="no frozen tau survived the candidate outcome",
        )
    return J1TauSelection(
        candidate_id=candidate.candidate_id,
        status="selected",
        selected_tau=eligible_taus[0],
        eligible_taus=eligible_taus,
        reason="selected the earliest qualifying tau from the frozen candidate order",
    )


def assess_j1_candidate_promotion(
    registry: FrozenJ1CandidateRegistry,
    outcome: J1CandidateOutcome,
) -> J1CandidateAssessment:
    candidate = registry.get_candidate(outcome.candidate_id)
    if not _same_policy_set(registry.comparator_policies, outcome.comparator_policies):
        return J1CandidateAssessment(
            candidate_id=candidate.candidate_id,
            lane=candidate.lane,
            status="comparator_mismatch",
            promotable=False,
            selected_tau=None,
            eligible_taus=(),
            reason="candidate outcome was evaluated against a comparator set outside the frozen J1 registry",
        )
    tau_selection = select_j1_tau(candidate, outcome.qualifying_taus)
    if tau_selection.status != "selected":
        return J1CandidateAssessment(
            candidate_id=candidate.candidate_id,
            lane=candidate.lane,
            status="no_qualifying_tau",
            promotable=False,
            selected_tau=None,
            eligible_taus=tau_selection.eligible_taus,
            reason=tau_selection.reason,
        )
    if not outcome.primary_gate_passed:
        return J1CandidateAssessment(
            candidate_id=candidate.candidate_id,
            lane=candidate.lane,
            status="primary_gate_failed",
            promotable=False,
            selected_tau=None,
            eligible_taus=tau_selection.eligible_taus,
            reason="candidate did not clear the primary J1 go/no-go gate",
        )
    if not outcome.safety_gate_passed:
        return J1CandidateAssessment(
            candidate_id=candidate.candidate_id,
            lane=candidate.lane,
            status="safety_gate_failed",
            promotable=False,
            selected_tau=None,
            eligible_taus=tau_selection.eligible_taus,
            reason="candidate regressed on a required safety or timing gate",
        )
    if not outcome.support_gate_passed:
        return J1CandidateAssessment(
            candidate_id=candidate.candidate_id,
            lane=candidate.lane,
            status="support_gate_failed",
            promotable=False,
            selected_tau=None,
            eligible_taus=tau_selection.eligible_taus,
            reason="candidate failed a required supporting-dataset gate",
        )
    return J1CandidateAssessment(
        candidate_id=candidate.candidate_id,
        lane=candidate.lane,
        status="promotable",
        promotable=True,
        selected_tau=tau_selection.selected_tau,
        eligible_taus=tau_selection.eligible_taus,
        reason="candidate satisfied the frozen comparator, tau, primary, safety, and support gates",
    )


def promote_j1_winners(
    registry: FrozenJ1CandidateRegistry,
    outcomes: Sequence[J1CandidateOutcome],
) -> J1PromotionReport:
    outcome_map, unregistered_candidate_ids = _index_outcomes(outcomes, registry=registry)
    if unregistered_candidate_ids:
        reason = (
            "promotion input referenced unregistered candidate outcomes: "
            + ", ".join(unregistered_candidate_ids)
        )
        return J1PromotionReport(
            registry_name=registry.registry_name,
            dataset_id=registry.dataset_id,
            comparator_policies=registry.comparator_policies,
            canonical=_no_winner(lane="canonical", reason=reason),
            challenger=_no_winner(lane="challenger", reason=reason),
        )
    return J1PromotionReport(
        registry_name=registry.registry_name,
        dataset_id=registry.dataset_id,
        comparator_policies=registry.comparator_policies,
        canonical=_promote_lane(registry, lane="canonical", outcome_map=outcome_map),
        challenger=_promote_lane(registry, lane="challenger", outcome_map=outcome_map),
    )


def generate_j1_challenger_manifest(
    registry: FrozenJ1CandidateRegistry,
    outcomes: Sequence[J1CandidateOutcome],
    *,
    promotion_report: J1PromotionReport | None = None,
) -> dict[str, object]:
    report = promotion_report or promote_j1_winners(registry, outcomes)
    outcome_map, unregistered_candidate_ids = _index_outcomes(outcomes, registry=registry)
    challenger_rows: list[dict[str, object]] = []
    for candidate_id in registry.candidate_ids_for_lane("challenger"):
        candidate = registry.get_candidate(candidate_id)
        outcome = outcome_map.get(candidate_id)
        if outcome is None:
            assessment = J1CandidateAssessment(
                candidate_id=candidate.candidate_id,
                lane=candidate.lane,
                status="missing_outcome",
                promotable=False,
                selected_tau=None,
                eligible_taus=(),
                reason="no challenger outcome was supplied for this frozen candidate",
            )
            observed_taus: tuple[str, ...] = ()
            primary_gate_passed: bool | None = None
            safety_gate_passed: bool | None = None
            support_gate_passed: bool | None = None
        else:
            assessment = assess_j1_candidate_promotion(registry, outcome)
            observed_taus = outcome.qualifying_taus
            primary_gate_passed = outcome.primary_gate_passed
            safety_gate_passed = outcome.safety_gate_passed
            support_gate_passed = outcome.support_gate_passed
        row = candidate.to_dict()
        row.update(
            {
                "assessment": assessment.to_dict(),
                "observed_qualifying_taus": list(observed_taus),
                "primary_gate_passed": primary_gate_passed,
                "safety_gate_passed": safety_gate_passed,
                "support_gate_passed": support_gate_passed,
            }
        )
        challenger_rows.append(row)
    winner = report.challenger.to_dict() if report.challenger.outcome == "promoted" else None
    no_winner_reason = report.challenger.reason if report.challenger.outcome == "no_winner" else None
    return {
        "registry_name": registry.registry_name,
        "dataset_id": registry.dataset_id,
        "lane": "challenger",
        "comparator_policies": list(registry.comparator_policies),
        "primary_comparator": registry.primary_comparator,
        "winner": winner,
        "no_winner_reason": no_winner_reason,
        "candidates": challenger_rows,
        "unregistered_candidate_outcomes": list(unregistered_candidate_ids),
    }


def _promote_lane(
    registry: FrozenJ1CandidateRegistry,
    *,
    lane: str,
    outcome_map: Mapping[str, J1CandidateOutcome],
) -> J1PromotionDecision:
    normalized_lane = _normalize_lane(lane)
    assessments: list[J1CandidateAssessment] = []
    for candidate_id in registry.candidate_ids_for_lane(normalized_lane):
        outcome = outcome_map.get(candidate_id)
        if outcome is None:
            continue
        assessments.append(assess_j1_candidate_promotion(registry, outcome))
    promotable = [assessment for assessment in assessments if assessment.promotable]
    if not promotable:
        return _no_winner(
            lane=normalized_lane,
            reason=f"no registered {normalized_lane} candidate satisfied every frozen promotion gate",
        )
    if len(promotable) == 1:
        winner = promotable[0]
        return J1PromotionDecision(
            lane=normalized_lane,
            outcome="promoted",
            candidate_id=winner.candidate_id,
            selected_tau=winner.selected_tau,
            reason="candidate cleared every frozen promotion gate",
        )

    # Lower explicit priority wins when more than one candidate survives the gates.
    prioritized = [
        (registry.get_candidate(assessment.candidate_id).priority, assessment)
        for assessment in promotable
    ]
    if all(priority is not None for priority, _ in prioritized):
        best_priority = min(int(priority) for priority, _ in prioritized if priority is not None)
        best = [assessment for priority, assessment in prioritized if priority == best_priority]
        if len(best) == 1:
            winner = best[0]
            return J1PromotionDecision(
                lane=normalized_lane,
                outcome="promoted",
                candidate_id=winner.candidate_id,
                selected_tau=winner.selected_tau,
                reason="candidate cleared every frozen promotion gate and won the explicit lane priority",
            )

    return _no_winner(
        lane=normalized_lane,
        reason=f"multiple {normalized_lane} candidates satisfied the promotion gates without a unique frozen priority",
    )


def _no_winner(*, lane: str, reason: str) -> J1PromotionDecision:
    return J1PromotionDecision(
        lane=_normalize_lane(lane),
        outcome="no_winner",
        candidate_id=None,
        selected_tau=None,
        reason=str(reason).strip(),
    )


def _index_outcomes(
    outcomes: Sequence[J1CandidateOutcome],
    *,
    registry: FrozenJ1CandidateRegistry | None = None,
) -> tuple[dict[str, J1CandidateOutcome], tuple[str, ...]]:
    indexed: dict[str, J1CandidateOutcome] = {}
    unregistered: list[str] = []
    for outcome in outcomes:
        if outcome.candidate_id in indexed:
            raise ValueError(f"duplicate outcome supplied for candidate_id '{outcome.candidate_id}'")
        if registry is not None and outcome.candidate_id not in registry.candidates:
            unregistered.append(outcome.candidate_id)
            continue
        indexed[outcome.candidate_id] = outcome
    return indexed, tuple(unregistered)


def _same_policy_set(left: Sequence[str], right: Sequence[str]) -> bool:
    return set(_freeze_policy_tuple(left, field_name="left")) == set(
        _freeze_policy_tuple(right, field_name="right")
    )


def _normalize_lane(value: str) -> Literal["canonical", "challenger"]:
    token = str(value).strip().lower()
    if token not in J1_LANES:
        raise ValueError(f"unsupported J1 lane: {value}")
    return token  # type: ignore[return-value]


def _normalize_evidence_tier(value: str, *, lane: str) -> str:
    token = str(value).strip()
    if token:
        return token
    if lane == "canonical":
        return "primary"
    return "challenger"


def _normalize_nonempty_text(value: object, *, field_name: str) -> str:
    token = str(value).strip()
    if not token:
        raise ValueError(f"{field_name} must be a non-empty string")
    return token


def _freeze_text_tuple(values: Sequence[str]) -> tuple[str, ...]:
    frozen: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        frozen.append(token)
    return tuple(frozen)


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


def _normalize_policy_name(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if not token:
        raise ValueError("comparator policy names must be non-empty")
    return token


def _freeze_tau_tuple(
    values: Sequence[str],
    *,
    field_name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    taus: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = _normalize_tau(value)
        if token in seen:
            continue
        seen.add(token)
        taus.append(token)
    if not taus and not allow_empty:
        raise ValueError(f"{field_name} must include at least one tau")
    return tuple(taus)


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


def _freeze_string_mapping(values: Mapping[str, str]) -> Mapping[str, str]:
    normalized = {
        _normalize_nonempty_text(key, field_name="metadata key"): str(value).strip()
        for key, value in dict(values).items()
    }
    return MappingProxyType(normalized)


__all__ = [
    "FrozenJ1CandidateRegistry",
    "J1Candidate",
    "J1CandidateAssessment",
    "J1CandidateOutcome",
    "J1PromotionDecision",
    "J1PromotionReport",
    "J1TauSelection",
    "J1_LANES",
    "J1_PRIMARY_COMPARATOR",
    "J1_REQUIRED_COMPARATORS",
    "assess_j1_candidate_promotion",
    "freeze_j1_candidate_registry",
    "generate_j1_challenger_manifest",
    "promote_j1_winners",
    "select_j1_tau",
]

