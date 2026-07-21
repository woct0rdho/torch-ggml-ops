import torch  # noqa: F401,I001 Import torch before loading extension.

from . import _C  # noqa: F401 Register native operators before torch.ops use. # ty: ignore[unresolved-import]
from .grouped_mmq import grouped_mmq, grouped_mmq_pair
from .mmq import mmq

__all__ = ["grouped_mmq", "grouped_mmq_pair", "mmq"]
