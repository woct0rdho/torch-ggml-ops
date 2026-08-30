"""No-prior compatibility dispatch for the current MMQ deployment cases.

The copied AITER inventory remains explicit-prior keyed. This module is the
small boundary for consumers whose current inputs identify the tuned family by
quantization type and matrix geometry but do not expose an expert prior.
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
_DEEPSEEK_QUANT_TYPES = frozenset({"Q2_K", "IQ2_XXS"})
_PRIORS = ("qwen-learned", "deepseek-learned", "deepseek-hash")


def _quant_type_name(quant_type: object | None) -> str | None:
    if quant_type is None:
        return None
    if isinstance(quant_type, str):
        return quant_type
    name = getattr(quant_type, "name", None)
    return name if isinstance(name, str) else str(quant_type)


def infer_expert_prior(
    m: int,
    k: int,
    n: int,
    *,
    quant_type: object | None = None,
    transposed_rhs: bool | None,
) -> str:
    """Infer the current public representative prior for one matrix key."""

    def has_entry(prior: str) -> bool:
        if transposed_rhs is None:
            return (prior, m, k, n) in _PTGMM_CONFIGS
        return (prior, m, k, n, transposed_rhs) in _GMM_CONFIGS

    candidates = tuple(prior for prior in _PRIORS if has_entry(prior))
    families = {
        "qwen" if prior == "qwen-learned" else "deepseek" for prior in candidates
    }
    quant_name = _quant_type_name(quant_type)
    if quant_name in _QWEN_QUANT_TYPES:
        preferred = "qwen-learned"
    elif quant_name in _DEEPSEEK_QUANT_TYPES:
        # The current public DeepSeek tensors are blk.0 representatives.
        preferred = "deepseek-hash"
    elif len(families) == 1:
        preferred = "qwen-learned" if "qwen" in families else "deepseek-hash"
    elif not families:
        raise ValueError(
            f"No tuned AITER config for M={m}, K={k}, N={n}, quant_type={quant_name}."
        )
    else:
        raise ValueError(
            "quant_type is required to distinguish the current AITER "
            f"configs for M={m}, K={k}, N={n}."
        )
    if preferred not in candidates:
        raise ValueError(f"quant_type={quant_name} does not match M={m}, K={k}, N={n}.")
    return preferred


def gmm_config(
    m: int,
    k: int,
    n: int,
    transposed_rhs: bool,
    *,
    quant_type: object | None = None,
) -> dict[str, int]:
    """Return a GMM config without exposing ``expert_prior``."""

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
    """Return a PTGMM config without exposing ``expert_prior``."""

    prior = infer_expert_prior(
        m,
        k,
        n,
        quant_type=quant_type,
        transposed_rhs=None,
    )
    return _ptgmm_config(m, k, n, prior)
