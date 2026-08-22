#!/usr/bin/env python3
"""Materialize one deterministic profile from one fitted grouped-MMQ law."""

import argparse
import json
import statistics
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from workload_prior import (
    EXPERT_PRIOR_NAMES,
    ExpertProfile,
    expert_prior_metadata,
    sample_expert_profile,
)


def _positive_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("batches must be positive integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("batches must be positive integers")
    return result


def _profile_mapping(profile: ExpertProfile, physical_batch: int) -> dict[str, object]:
    active = [rows for rows in profile.rows_per_expert if rows]
    shares = [rows / profile.aggregate_rows for rows in profile.rows_per_expert]
    return {
        **profile.to_mapping(),
        "profile_id": profile.profile_id,
        "physical_batch": physical_batch,
        "expert_indices": [
            expert for expert, rows in enumerate(profile.rows_per_expert) if rows
        ],
        "group_sizes": active,
        "metrics": {
            "active_experts": len(active),
            "minimum_positive_rows": min(active),
            "maximum_rows": max(active),
            "median_positive_rows": statistics.median(active),
            "maximum_inclusion": max(active) / profile.tokens,
            "effective_experts": 1.0 / sum(share * share for share in shares),
            "pad_to_max_inflation": (
                len(active) * max(active) / profile.aggregate_rows
            ),
        },
    }


def generate_document(
    expert_prior: str,
    batches: tuple[int, ...],
    sequence_length: int,
) -> dict[str, object]:
    profiles = [
        _profile_mapping(
            sample_expert_profile(expert_prior, batch * sequence_length),
            batch,
        )
        for batch in batches
    ]
    return {
        "purpose": "one-law grouped MMQ performance benchmark profiles",
        "fit": expert_prior_metadata(expert_prior),
        "profile_policy": "one deterministic profile per physical token count",
        "seed_contract": (
            "8314159 + floor(tokens * 104729 / 2048), plus 31000 only for deepseek-hash"
        ),
        "sequence_length": sequence_length,
        "profiles": profiles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expert-prior",
        choices=EXPERT_PRIOR_NAMES,
        required=True,
        help="the sole fitted law represented by the output",
    )
    parser.add_argument("--batches", type=_positive_int_list, default=(1, 4, 16))
    parser.add_argument("--sequence-length", type=int, default=2048)
    args = parser.parse_args()
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    return args


def main() -> None:
    args = parse_args()
    document = generate_document(
        args.expert_prior,
        args.batches,
        args.sequence_length,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    profiles = document["profiles"]
    if not isinstance(profiles, list):
        raise TypeError("profile document has an invalid profiles field")
    print(f"wrote {args.output}: law={args.expert_prior} profiles={len(profiles)}")


if __name__ == "__main__":
    main()
