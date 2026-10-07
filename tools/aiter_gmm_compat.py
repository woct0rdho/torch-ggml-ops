"""No-prior compatibility dispatch for the current MMQ deployment cases.

The copied AITER inventory remains explicit-prior keyed. This module is the
small boundary for consumers whose current inputs identify the tuned family by
quantization type and matrix geometry but do not expose an expert prior.

Every key here is AITER's, in the computed GEMM's letters: `(m, k, n)` with `k`
the contracted width and `n` the width the kernel writes. A forward problem
feeds `(rows, in_features, out_features)` and a backward problem feeds
`(rows, out_features, in_features)`. `ptgmm` uses its own letters again, see
`ptgmm_config`.

The inferred law is always a learned one. The public routed deployment cannot
tell learned routing from hash routing in its inputs, so DeepSeek shapes assume
the learned router. Hash-tuned kernels stay benchmark-opt-in and are never
inferred here.
"""

from tools.aiter_gmm_heuristics import (
    _GMM_CONFIGS,
    _PTGMM_CONFIGS,
)
from tools.aiter_gmm_heuristics import (
    gmm_config as _gmm_config,
)
from tools.aiter_gmm_heuristics import (
    ptgmm_config as _ptgmm_config,
)

_QWEN_QUANT_TYPES = frozenset({"Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_S"})
_QWEN38_QUANT_TYPES = frozenset({"Q2_0", "IQ4_NL", "IQ4_XS"})
_DEEPSEEK_QUANT_TYPES = frozenset({"Q2_K", "IQ2_XXS"})
_LEARNED_PRIORS = ("qwen-learned", "qwen3.8-learned", "deepseek-learned")


def _quant_type_name(quant_type: object | None) -> str | None:
    if quant_type is None:
        return None
    if isinstance(quant_type, str):
        return quant_type
    name = getattr(quant_type, "name", None)
    return name if isinstance(name, str) else str(quant_type)


def _has_entry(prior: str, m: int, k: int, n: int, transposed_rhs: bool | None) -> bool:
    if transposed_rhs is None:
        return (prior, m, k, n) in _PTGMM_CONFIGS
    return (prior, m, k, n, transposed_rhs) in _GMM_CONFIGS


def infer_expert_prior(
    m: int,
    k: int,
    n: int,
    *,
    quant_type: object | None = None,
    transposed_rhs: bool | None,
) -> str:
    """Infer the benchmark default prior for one matrix key.

    `(m, k, n)` are AITER's letters of the computed GEMM: `k` contracts and `n`
    is written, so a backward problem passes its two problem letters swapped.
    The quant type selects the routed family. Without one the learned inventory
    has to be unambiguous. The result is validated against the tuned inventory,
    so an unmeasured key fails instead of borrowing another family's law. A
    quantization type that selects a family with no entry for the key fails as
    well: a Qwen3.5-family type must not silently resolve a Qwen3.8 geometry.
    """

    quant_name = _quant_type_name(quant_type)
    if quant_name in _QWEN38_QUANT_TYPES:
        family = "qwen3.8"
    elif quant_name in _QWEN_QUANT_TYPES:
        family = "qwen"
    elif quant_name in _DEEPSEEK_QUANT_TYPES:
        family = "deepseek"
    elif quant_name is not None:
        raise ValueError(
            f"quant_type={quant_name} has no public routed expert-prior family."
        )
    else:
        candidates = tuple(
            prior
            for prior in _LEARNED_PRIORS
            if _has_entry(prior, m, k, n, transposed_rhs)
        )
        families = {prior.split("-")[0] for prior in candidates}
        if not families:
            raise ValueError(
                f"quant_type is required. No tuned learned config exists for "
                f"M={m}, K={k}, N={n}."
            )
        if len(families) > 1:
            raise ValueError(
                "quant_type is required to distinguish the learned configs for "
                f"M={m}, K={k}, N={n}."
            )
        family = families.pop()

    prior = f"{family}-learned"
    if not _has_entry(prior, m, k, n, transposed_rhs):
        raise ValueError(f"No tuned {prior} AITER config for M={m}, K={k}, N={n}.")
    return prior


def gmm_config(
    m: int,
    k: int,
    n: int,
    transposed_rhs: bool,
    *,
    quant_type: object | None = None,
) -> dict[str, int]:
    """Return a GMM config without exposing `expert_prior`.

    `(m, k, n)` are AITER's letters: `k` contracts and `n` is written.
    """

    prior = infer_expert_prior(
        m,
        k,
        n,
        quant_type=quant_type,
        transposed_rhs=transposed_rhs,
    )
    return _gmm_config(m, k, n, transposed_rhs, prior)


def ptgmm_config(
    m: int,
    k: int,
    n: int,
    *,
    quant_type: object | None = None,
) -> dict[str, int]:
    """Return a PTGMM config without exposing `expert_prior`.

    PTGMM's letters are its own and are not the problem's or `gmm_config`'s: `m`
    is the contracted length (the `lhs` columns and `rhs` rows), and `k` and `n`
    are the rows and columns of one group's output.
    """

    prior = infer_expert_prior(
        m,
        k,
        n,
        quant_type=quant_type,
        transposed_rhs=None,
    )
    return _ptgmm_config(m, k, n, prior)
