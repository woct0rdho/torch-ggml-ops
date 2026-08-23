import pytest
import torch

import torch_ggml_ops
from tests.mmq_test_support import find_tensor, load_packed_rows, random_bf16
from tests.model_test_cases import QWEN_Q4_DENSE_MMQ_TEST_CASE
from tests.model_test_support import model_reader

_ROWS = 2048
_OUT_FEATURES = 512


def _q4_weight() -> tuple[torch.Tensor, int]:
    case = QWEN_Q4_DENSE_MMQ_TEST_CASE
    tensor = find_tensor(model_reader(case.model), case.tensor_name)
    packed = load_packed_rows(tensor, _OUT_FEATURES)
    return packed, int(tensor.tensor_type)


def test_exact_dense_compiles_with_visible_allocations() -> None:
    packed, quant_type = _q4_weight()
    input = random_bf16(_ROWS, 2048, seed=10003)

    @torch.compile(fullgraph=True)
    def compiled(input: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        return torch_ggml_ops.mmq(input, packed, quant_type, _OUT_FEATURES)

    expected = torch_ggml_ops.mmq(input, packed, quant_type, _OUT_FEATURES)
    actual = compiled(input, packed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_unsupported_dense_key_fails_at_native_launch() -> None:
    packed, quant_type = _q4_weight()
    input = random_bf16(129, 2048, seed=10004)
    with pytest.raises(RuntimeError, match="unsupported exact deployment key"):
        torch_ggml_ops.mmq(input, packed, quant_type, _OUT_FEATURES)


def test_dense_launch_validates_explicit_buffers() -> None:
    packed, quant_type = _q4_weight()
    input = random_bf16(_ROWS, 2048, seed=10005)
    output = torch.empty(
        (_ROWS, _OUT_FEATURES - 1), dtype=torch.bfloat16, device="cuda"
    )
    workspace = torch.empty(
        input.numel() // 128 * 144, dtype=torch.uint8, device="cuda"
    )
    with pytest.raises(RuntimeError, match="output"):
        torch.ops.torch_ggml_ops._mmq_launch.default(
            input, packed, quant_type, _OUT_FEATURES, output, workspace
        )


def test_dense_launch_validates_native_operand_contracts() -> None:
    packed, quant_type = _q4_weight()
    input = random_bf16(_ROWS, 2048, seed=10006)
    output = torch.empty((_ROWS, _OUT_FEATURES), dtype=torch.bfloat16, device="cuda")
    workspace = torch.empty(
        input.numel() // 128 * 144, dtype=torch.uint8, device="cuda"
    )

    def launch(candidate: torch.Tensor, weight: torch.Tensor = packed) -> None:
        torch.ops.torch_ggml_ops._mmq_launch.default(
            candidate,
            weight,
            quant_type,
            _OUT_FEATURES,
            output,
            workspace,
        )

    with pytest.raises(RuntimeError, match="BF16"):
        launch(input.float())
    with pytest.raises(RuntimeError, match="CUDA/HIP"):
        launch(input.cpu())
    with pytest.raises(RuntimeError, match="contiguous"):
        launch(input.T)

    offset_input = torch.empty(input.numel() + 1, dtype=torch.bfloat16, device="cuda")[
        1:
    ].view_as(input)
    with pytest.raises(RuntimeError, match="zero storage offset"):
        launch(offset_input)
    with pytest.raises(RuntimeError, match="packed_weight must have shape"):
        launch(input, packed.view(-1))
    with pytest.raises(RuntimeError, match="uint8"):
        launch(input, packed.to(torch.int8))


def test_dense_launch_validates_every_explicit_buffer_property() -> None:
    packed, quant_type = _q4_weight()
    input = random_bf16(_ROWS, 2048, seed=10007)
    output = torch.empty((_ROWS, _OUT_FEATURES), dtype=torch.bfloat16, device="cuda")
    workspace_elements = input.numel() // 128 * 144
    workspace = torch.empty(workspace_elements, dtype=torch.uint8, device="cuda")

    def launch(destination: torch.Tensor, scratch: torch.Tensor) -> None:
        torch.ops.torch_ggml_ops._mmq_launch.default(
            input,
            packed,
            quant_type,
            _OUT_FEATURES,
            destination,
            scratch,
        )

    with pytest.raises(RuntimeError, match="output has an invalid dtype"):
        launch(output.float(), workspace)
    with pytest.raises(RuntimeError, match="output must be a CUDA/HIP tensor"):
        launch(output.cpu(), workspace)
    with pytest.raises(RuntimeError, match="output must be contiguous"):
        launch(
            torch.empty(
                (_OUT_FEATURES, _ROWS),
                dtype=torch.bfloat16,
                device="cuda",
            ).T,
            workspace,
        )

    offset_output = torch.empty(
        output.numel() + 1, dtype=torch.bfloat16, device="cuda"
    )[1:].view_as(output)
    with pytest.raises(RuntimeError, match="output must have zero storage offset"):
        launch(offset_output, workspace)
    with pytest.raises(RuntimeError, match="output has an invalid element count"):
        launch(output[:, :-1].contiguous(), workspace)
    with pytest.raises(RuntimeError, match="workspace has an invalid dtype"):
        launch(output, workspace.to(torch.int8))
    with pytest.raises(RuntimeError, match="workspace must be a CUDA/HIP tensor"):
        launch(output, workspace.cpu())
    with pytest.raises(RuntimeError, match="exact one-dimensional shape"):
        launch(output, workspace.view(2, -1))

    offset_workspace = torch.empty(
        workspace_elements + 1, dtype=torch.uint8, device="cuda"
    )[1:]
    with pytest.raises(RuntimeError, match="workspace must have zero storage offset"):
        launch(output, offset_workspace)
    with pytest.raises(RuntimeError, match="workspace has an invalid element count"):
        launch(output, workspace[:-1].clone())


def test_obsolete_public_dispatcher_ops_are_absent() -> None:
    for name in ("mmq", "mmq_grad_input"):
        assert not hasattr(torch.ops.torch_ggml_ops, name)
