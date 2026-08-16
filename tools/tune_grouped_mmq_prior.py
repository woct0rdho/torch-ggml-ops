#!/usr/bin/env python3
"""Generate deterministic fitted-prior profiles for grouped MMQ HIP tuning.

This tool owns benchmark-only route generation and medoid reduction. It does not
read route captures, model metadata, temporary artifacts, or Markdown.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PRIOR_VERSION = "grouped-mmq-route-prior-v1"
REFERENCE_TOKENS = 2048
EXPERTS = 256
SEARCH_SEED = 8_314_159
CONFIRMATION_OFFSET = 1_000_003
HASH_OFFSET = 31_000
DEFAULT_DRAWS = 512
DEFAULT_MEDOIDS = 5
BATCHES = (1, 4, 16)


@dataclass(frozen=True)
class Residual:
    degrees_of_freedom: float
    location: float
    scale: float
    lower: float
    upper: float

    def sample(self, rng: np.random.Generator) -> float:
        value = self.location + self.scale * float(
            rng.standard_t(self.degrees_of_freedom)
        )
        return min(self.upper, max(self.lower, value))


@dataclass(frozen=True)
class LearnedPrior:
    name: str
    top_k: int
    shift: float
    head_multiplier: float
    active_a0: float
    active_log_tokens: float
    alpha_b0: float
    alpha_log_tokens: float
    alpha_active_residual: float
    active_residual: Residual
    alpha_residual: Residual


QWEN_PRIOR = LearnedPrior(
    name="qwen",
    top_k=8,
    shift=11.5465232873,
    head_multiplier=1.1255020052,
    active_a0=1.3568429238,
    active_log_tokens=1.1367804236,
    alpha_b0=0.7376182182,
    alpha_log_tokens=-0.0215993943,
    alpha_active_residual=-0.1730074363,
    active_residual=Residual(
        8.3978226526,
        -0.0335118652,
        0.9830493404,
        -2.5036492445,
        3.7809143778,
    ),
    alpha_residual=Residual(
        5.2038953632,
        0.0054814884,
        0.1025706842,
        -0.4463245132,
        0.3833201702,
    ),
)

DEEPSEEK_PRIOR = LearnedPrior(
    name="deepseek",
    top_k=6,
    shift=6.3223820835,
    head_multiplier=1.0643729189,
    active_a0=2.2468973539,
    active_log_tokens=0.8906164814,
    alpha_b0=0.4081577973,
    alpha_log_tokens=-0.0211298377,
    alpha_active_residual=-0.0754167931,
    active_residual=Residual(
        33.5987235964,
        -0.0013194870,
        0.8328268568,
        -1.6829685257,
        2.7587218851,
    ),
    alpha_residual=Residual(
        5.1551932778,
        -0.0071110386,
        0.0923527684,
        -0.3180690433,
        0.3604795507,
    ),
)

HASH_COEFFICIENTS = {
    "rho_mean": 0.076568603515625,
    "rho_beta_concentration": 111.40091020461985,
    "body_a0": 6.660030508273053,
    "body_log_tokens_exponent": 0.4771750368253468,
}


def _active_curve(prior: LearnedPrior, active: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, active + 1, dtype=np.float64)
    values = np.power(ranks + prior.shift, -alpha)
    low = 0.0
    high = 1.0 / values[-1]
    for _ in range(64):
        middle = (low + high) / 2.0
        if np.minimum(1.0, middle * values).sum() < prior.top_k:
            low = middle
        else:
            high = middle
    inclusion = np.minimum(1.0, ((low + high) / 2.0) * values)
    if active > 1:
        old_head = float(inclusion[0])
        new_head = min(1.0, old_head * prior.head_multiplier)
        inclusion[1:] *= (prior.top_k - new_head) / (prior.top_k - old_head)
        inclusion[0] = new_head
    return inclusion


def _largest_remainder(
    values: np.ndarray,
    total: int,
    *,
    lower: int = 0,
    upper: int,
) -> np.ndarray:
    raw = values * total
    rows = np.clip(np.floor(raw).astype(np.int64), lower, upper)
    while int(rows.sum()) > total:
        candidates = np.flatnonzero(rows > lower)
        if not len(candidates):
            raise ValueError("cannot remove constrained-rounding excess")
        errors = rows[candidates] - raw[candidates]
        order = candidates[np.lexsort((candidates, -errors))]
        take = min(int(rows.sum()) - total, len(order))
        rows[order[:take]] -= 1
    while int(rows.sum()) < total:
        candidates = np.flatnonzero(rows < upper)
        if not len(candidates):
            raise ValueError("cannot distribute constrained-rounding deficit")
        errors = raw[candidates] - rows[candidates]
        order = candidates[np.lexsort((candidates, -errors))]
        take = min(total - int(rows.sum()), len(order))
        rows[order[:take]] += 1
    if int(rows.sum()) != total or np.any(rows < lower) or np.any(rows > upper):
        raise ValueError("constrained rounding violated the route contract")
    return rows


def sample_learned_rows(
    prior: LearnedPrior,
    tokens: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = math.log(tokens / REFERENCE_TOKENS)
    active_residual = prior.active_residual.sample(rng)
    active_logit = prior.active_a0 + prior.active_log_tokens * x + active_residual
    active = int(np.rint(257.0 / (1.0 + math.exp(-active_logit)) - 0.5))
    active = min(EXPERTS, max(prior.top_k, active))
    alpha_residual = prior.alpha_residual.sample(rng)
    alpha = math.exp(
        prior.alpha_b0
        + prior.alpha_log_tokens * x
        + prior.alpha_active_residual * active_residual
        + alpha_residual
    )
    inclusion = _active_curve(prior, active, alpha)
    ranked = _largest_remainder(
        inclusion / prior.top_k,
        prior.top_k * tokens,
        upper=tokens,
    )
    rows = np.zeros(EXPERTS, dtype=np.int64)
    rows[rng.permutation(EXPERTS)[:active]] = ranked
    _validate_rows(rows, tokens, prior.top_k, require_all=False)
    return rows


def sample_hash_rows(tokens: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    batch_ratio = tokens / REFERENCE_TOKENS
    rho_mean = HASH_COEFFICIENTS["rho_mean"]
    base_kappa = HASH_COEFFICIENTS["rho_beta_concentration"]
    beta_kappa = batch_ratio * (base_kappa + 1.0) - 1.0
    rho = float(rng.beta(rho_mean * beta_kappa, (1.0 - rho_mean) * beta_kappa))
    body_a = (
        HASH_COEFFICIENTS["body_a0"]
        * batch_ratio ** HASH_COEFFICIENTS["body_log_tokens_exponent"]
    )
    body = rng.dirichlet(np.full(EXPERTS, body_a))
    head = rng.permutation(EXPERTS)[:6]
    probabilities = (1.0 - rho) * body
    probabilities[head] += rho / 6.0
    rows = _largest_remainder(
        probabilities,
        6 * tokens,
        lower=1,
        upper=tokens,
    )
    _validate_rows(rows, tokens, 6, require_all=True)
    return rows


def _validate_rows(
    rows: np.ndarray,
    tokens: int,
    top_k: int,
    *,
    require_all: bool,
) -> None:
    if rows.shape != (EXPERTS,):
        raise ValueError(f"expected {EXPERTS} physical experts")
    if int(rows.sum()) != tokens * top_k:
        raise ValueError("route rows do not preserve top-k mass")
    if np.any(rows < 0) or np.any(rows > tokens):
        raise ValueError("route rows violate physical token bounds")
    if require_all and np.any(rows == 0):
        raise ValueError("hash prior must activate every expert")


def _assign(distances: np.ndarray, medoids: list[int]) -> np.ndarray:
    ordered = np.asarray(sorted(medoids), dtype=np.int64)
    return ordered[np.argmin(distances[:, ordered], axis=1)]


def medoid_reduction(
    rows: np.ndarray,
    count: int,
) -> tuple[list[int], dict[int, list[int]]]:
    if rows.ndim != 2 or rows.shape[1] != EXPERTS:
        raise ValueError("medoid rows must have shape [draws, 256]")
    if count <= 0 or count > rows.shape[0]:
        raise ValueError("invalid medoid count")
    distances = np.empty((rows.shape[0], rows.shape[0]), dtype=np.int64)
    for index in range(rows.shape[0]):
        distances[index] = np.abs(rows - rows[index]).sum(axis=1)

    selected = [0]
    while len(selected) < count:
        nearest = distances[:, selected].min(axis=1)
        nearest[np.asarray(selected)] = -1
        selected.append(int(np.argmax(nearest)))
        selected.sort()

    for _ in range(100):
        assigned_sources = _assign(distances, selected)
        updated: list[int] = []
        for medoid in sorted(selected):
            members = np.flatnonzero(assigned_sources == medoid)
            costs = distances[np.ix_(members, members)].sum(axis=1)
            updated.append(int(members[int(np.argmin(costs))]))
        updated.sort()
        if updated == selected:
            break
        selected = updated
    else:
        raise RuntimeError("k-medoids did not converge")

    assigned_sources = _assign(distances, selected)
    clusters = {
        medoid: np.flatnonzero(assigned_sources == medoid).astype(int).tolist()
        for medoid in selected
    }
    if sum(len(members) for members in clusters.values()) != rows.shape[0]:
        raise RuntimeError("medoid clusters do not cover the draw bank")
    return selected, clusters


def _profile_metrics(rows: np.ndarray, tokens: int, top_k: int) -> dict[str, Any]:
    active = rows[rows > 0]
    shares = rows / float(rows.sum())
    return {
        "active_experts": int(active.size),
        "minimum_positive_rows": int(active.min()),
        "maximum_rows": int(active.max()),
        "median_positive_rows": float(np.median(active)),
        "maximum_inclusion": float(active.max() / tokens),
        "effective_experts": float(1.0 / np.square(shares).sum()),
        "pad_to_max_inflation": float(active.size * active.max() / rows.sum()),
    }


def _serialize_profile(
    *,
    family: str,
    kind: str,
    batch: int,
    bank: str,
    source_index: int,
    seed: int,
    rows: np.ndarray,
    cluster_members: list[int],
    draws: int,
    top_k: int,
) -> dict[str, Any]:
    tokens = batch * REFERENCE_TOKENS
    expert_ids = np.flatnonzero(rows > 0).astype(int).tolist()
    group_sizes = rows[rows > 0].astype(int).tolist()
    return {
        "profile_id": f"{family}_{kind}_b{batch}_{bank}_medoid_{source_index}",
        "prior_version": PRIOR_VERSION,
        "bank": bank,
        "source_index": source_index,
        "seed": seed,
        "physical_batch": batch,
        "tokens": tokens,
        "top_k": top_k,
        "row_sum": int(rows.sum()),
        "maximum_rows": int(rows.max()),
        "expert_ids": expert_ids,
        "group_sizes": group_sizes,
        "rows_per_expert": rows.astype(int).tolist(),
        "cluster_members": cluster_members,
        "medoid_weight": len(cluster_members) / draws,
        "metrics": _profile_metrics(rows, tokens, top_k),
    }


def build_profile_family(
    family: str,
    kind: str,
    batch: int,
    bank: str,
    draws: int,
    medoids: int,
) -> dict[str, Any]:
    if family not in {"qwen", "deepseek"}:
        raise ValueError(f"unsupported family {family}")
    if kind not in {"learned", "hash"}:
        raise ValueError(f"unsupported router kind {kind}")
    if kind == "hash" and family != "deepseek":
        raise ValueError("only DeepSeek has a hash prior")
    offset = CONFIRMATION_OFFSET if bank == "confirmation" else 0
    if bank not in {"search", "confirmation"}:
        raise ValueError(f"unsupported profile bank {bank}")
    if kind == "hash":
        offset += HASH_OFFSET
    tokens = batch * REFERENCE_TOKENS
    top_k = 8 if family == "qwen" else 6
    prior = QWEN_PRIOR if family == "qwen" else DEEPSEEK_PRIOR
    seeds = [SEARCH_SEED + batch * 104_729 + index + offset for index in range(draws)]
    rows = np.asarray(
        [
            sample_hash_rows(tokens, seed)
            if kind == "hash"
            else sample_learned_rows(prior, tokens, seed)
            for seed in seeds
        ],
        dtype=np.int64,
    )
    selected, clusters = medoid_reduction(rows, medoids)
    profiles = [
        _serialize_profile(
            family=family,
            kind=kind,
            batch=batch,
            bank=bank,
            source_index=source_index,
            seed=seeds[source_index],
            rows=rows[source_index],
            cluster_members=clusters[source_index],
            draws=draws,
            top_k=top_k,
        )
        for source_index in selected
    ]
    return {
        "family": family,
        "router_kind": kind,
        "physical_batch": batch,
        "tokens": tokens,
        "top_k": top_k,
        "draws": draws,
        "medoids": medoids,
        "distance": "normalized L1 over the full physical 256-expert row vector",
        "initialization": "source index zero, then farthest-first; lowest-index ties",
        "profiles": profiles,
    }


def _adjust_positive(values: list[int], total: int) -> tuple[int, ...]:
    if total < len(values):
        raise ValueError("cannot construct a positive route distribution")
    result = [max(1, int(value)) for value in values]
    difference = total - sum(result)
    index = 0
    while difference:
        slot = index % len(result)
        if difference > 0:
            result[slot] += 1
            difference -= 1
        elif result[slot] > 1:
            result[slot] -= 1
            difference += 1
        index += 1
    return tuple(result)


def _centered_sizes(total: int, groups: int, amplitude: int) -> tuple[int, ...]:
    center = total / groups
    return _adjust_positive(
        [
            round(center + (((index * 37) % 129) - 64) * amplitude / 64)
            for index in range(groups)
        ],
        total,
    )


def control_profiles(family: str, batch: int) -> list[dict[str, Any]]:
    top_k = 8 if family == "qwen" else 6
    rows = batch * REFERENCE_TOKENS * top_k
    all_experts = tuple(range(EXPERTS))
    uniform = _adjust_positive([rows // EXPERTS] * EXPERTS, rows)
    amplitude = {1: 22, 4: 64, 16: 256}[batch]
    skewed = _centered_sizes(rows, EXPERTS, amplitude)
    sparse_groups = {1: 192, 4: 224, 16: 240}[batch]
    sparse_experts = tuple(
        int(value)
        for value in np.linspace(0, EXPERTS - 1, sparse_groups, dtype=np.int64)
    )
    sparse_sizes = _centered_sizes(
        rows,
        sparse_groups,
        max(1, round((rows / sparse_groups) * 0.25)),
    )
    boundary_prefix = (1, 15, 16, 17, 63, 64, 65, 127, 128, 129)
    tail_groups = EXPERTS - len(boundary_prefix)
    tail_total = rows - sum(boundary_prefix)
    boundary = boundary_prefix + _centered_sizes(
        tail_total,
        tail_groups,
        max(1, round((tail_total / tail_groups) * 0.10)),
    )
    definitions = (
        ("uniform", all_experts, uniform),
        ("skewed", all_experts, skewed),
        ("sparse", sparse_experts, sparse_sizes),
        ("boundary", all_experts, boundary),
    )
    profiles = []
    for name, expert_ids, group_sizes in definitions:
        physical = np.zeros(EXPERTS, dtype=np.int64)
        physical[np.asarray(expert_ids)] = np.asarray(group_sizes)
        _validate_rows(
            physical,
            batch * REFERENCE_TOKENS,
            top_k,
            require_all=False,
        )
        profiles.append(
            {
                "profile_id": f"{family}_b{batch}_control_{name}",
                "prior_version": PRIOR_VERSION,
                "bank": "control",
                "distribution": name,
                "physical_batch": batch,
                "tokens": batch * REFERENCE_TOKENS,
                "top_k": top_k,
                "row_sum": int(physical.sum()),
                "maximum_rows": max(group_sizes),
                "expert_ids": list(expert_ids),
                "group_sizes": list(group_sizes),
                "rows_per_expert": physical.astype(int).tolist(),
                "medoid_weight": 0.0,
                "metrics": _profile_metrics(
                    physical,
                    batch * REFERENCE_TOKENS,
                    top_k,
                ),
            }
        )
    return profiles


def generate_document(draws: int, medoids: int) -> dict[str, Any]:
    banks: dict[str, dict[str, Any]] = {}
    for bank in ("search", "confirmation"):
        families: dict[str, Any] = {}
        for family, kinds in (
            ("qwen", ("learned",)),
            ("deepseek", ("learned", "hash")),
        ):
            for kind in kinds:
                for batch in BATCHES:
                    key = f"{family}_{kind}_b{batch}"
                    families[key] = build_profile_family(
                        family,
                        kind,
                        batch,
                        bank,
                        draws,
                        medoids,
                    )
        banks[bank] = families
    controls = {
        f"{family}_b{batch}": control_profiles(family, batch)
        for family in ("qwen", "deepseek")
        for batch in BATCHES
    }
    return {
        "prior_version": PRIOR_VERSION,
        "purpose": "coefficient-only grouped MMQ HIP tuning; not production routing evidence",
        "reference_tokens": REFERENCE_TOKENS,
        "seed_contract": {
            "search": "8314159 + physical_batch * 104729 + source_index",
            "confirmation_offset": CONFIRMATION_OFFSET,
            "deepseek_hash_offset": HASH_OFFSET,
        },
        "draws_per_family_batch_component": draws,
        "medoids_per_family_batch_component": medoids,
        "deepseek_reporting_weights": {"learned": 40 / 43, "hash": 3 / 43},
        "banks": banks,
        "controls": controls,
        "fixed_output_a": [
            {
                "physical_batch": batch,
                "tokens": batch * REFERENCE_TOKENS,
                "rows_per_group": batch * REFERENCE_TOKENS,
                "groups": 8,
                "prior_weight": 1.0,
            }
            for batch in BATCHES
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--medoids", type=int, default=DEFAULT_MEDOIDS)
    args = parser.parse_args()
    if args.draws <= 0:
        parser.error("--draws must be positive")
    if args.medoids <= 0 or args.medoids > args.draws:
        parser.error("--medoids must be in [1, draws]")
    return args


def main() -> None:
    args = parse_args()
    document = generate_document(args.draws, args.medoids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: "
        f"{len(document['banks']['search'])} search families, "
        f"{len(document['banks']['confirmation'])} confirmation families"
    )


if __name__ == "__main__":
    main()
