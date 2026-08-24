import torch  # noqa: F401,I001 Import torch before loading extension.

from . import _C  # noqa: F401 Register native operators before torch.ops use. # ty: ignore[unresolved-import]
from ._mmq_autograd import (
    FixedGroupedMMQFunction,
    GroupedMMQFunction,
    GroupedMMQPairFunction,
    MMQFunction,
)
from ._mmq_cuda import mmq_grad_input_inplace, mmq_inplace

fixed_grouped_mmq = FixedGroupedMMQFunction.apply
grouped_mmq = GroupedMMQFunction.apply
grouped_mmq_pair = GroupedMMQPairFunction.apply
mmq = MMQFunction.apply

__all__ = [
    "fixed_grouped_mmq",
    "grouped_mmq",
    "grouped_mmq_pair",
    "mmq",
    "mmq_grad_input_inplace",
    "mmq_inplace",
]
