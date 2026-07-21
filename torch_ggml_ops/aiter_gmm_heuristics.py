"""Project-owned gfx1151 launch heuristics for AITER grouped GEMM."""

__all__ = ["gmm_config", "ptgmm_config"]


def gmm_config(k: int, n: int) -> dict[str, int]:
    """Return the production gfx1151 AITER GMM configuration."""

    if n <= 16:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_K": 128,
            "BLOCK_SIZE_N": 16 if k >= 2048 else 32,
            "GROUP_SIZE": 1,
            "GRID_DIM": 256,
            "num_warps": 4,
            "num_stages": 1,
        }
    if k <= 16:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_K": 32 if n >= 2048 else 16,
            "BLOCK_SIZE_N": 256 if n >= 2048 else 64,
            "GROUP_SIZE": 1,
            "GRID_DIM": 256,
            "num_warps": 4,
            "num_stages": 1,
        }
    if k >= 2048 and n <= 512:
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_K": 64,
            "BLOCK_SIZE_N": 128,
            "GROUP_SIZE": 1,
            "GRID_DIM": 256,
            "num_warps": 8,
            "num_stages": 1,
        }
    return {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 128,
        "GROUP_SIZE": 1,
        "GRID_DIM": 256,
        "num_warps": 8,
        "num_stages": 1,
    }


def ptgmm_config(k: int, n: int) -> dict[str, int]:
    """Return the production gfx1151 AITER PTGMM configuration."""

    if n <= 16:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_K": 256 if k >= 1024 else 128,
            "BLOCK_SIZE_N": 16,
            "GROUP_SIZE": 1,
            "GRID_DIM": 256,
            "num_warps": 4 if k >= 1024 else 8,
            "num_stages": 1,
        }
    if k <= 16:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_K": 16,
            "BLOCK_SIZE_N": 256 if n >= 2048 else 128,
            "GROUP_SIZE": 1,
            "GRID_DIM": 256,
            "num_warps": 4,
            "num_stages": 1,
        }
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_K": 256,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE": 1,
        "GRID_DIM": 256,
        "num_warps": 8,
        "num_stages": 1,
    }
