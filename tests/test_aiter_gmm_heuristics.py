import pytest

from torch_ggml_ops.aiter_gmm_heuristics import gmm_config, ptgmm_config


def _config(m: int, k: int, n: int, warps: int) -> dict[str, int]:
    return {
        "BLOCK_SIZE_M": m,
        "BLOCK_SIZE_K": k,
        "BLOCK_SIZE_N": n,
        "GROUP_SIZE": 1,
        "GRID_DIM": 256,
        "num_warps": warps,
        "num_stages": 1,
    }


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    (
        (2048, 16, _config(32, 128, 16, 4)),
        (2047, 16, _config(32, 128, 32, 4)),
        (16, 2048, _config(32, 32, 256, 4)),
        (16, 2047, _config(32, 16, 64, 4)),
        (2048, 512, _config(64, 64, 128, 8)),
        (2048, 513, _config(128, 64, 128, 8)),
        (17, 17, _config(128, 64, 128, 8)),
    ),
)
def test_gmm_config(k: int, n: int, expected: dict[str, int]) -> None:
    assert gmm_config(k, n) == expected


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    (
        (1024, 16, _config(32, 256, 16, 4)),
        (1023, 16, _config(32, 128, 16, 8)),
        (16, 2048, _config(32, 16, 256, 4)),
        (16, 2047, _config(32, 16, 128, 4)),
        (17, 17, _config(64, 256, 256, 8)),
    ),
)
def test_ptgmm_config(k: int, n: int, expected: dict[str, int]) -> None:
    assert ptgmm_config(k, n) == expected
