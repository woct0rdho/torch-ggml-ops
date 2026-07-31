import os
from dataclasses import dataclass
from pathlib import Path

from tests.deepseek_dense_cases import DEEPSEEK_DENSE_TENSOR_CASES


@dataclass(frozen=True)
class ModelTestCase:
    family: str
    model_env: str
    default_model: str

    @property
    def model_path(self) -> Path:
        return Path(
            os.environ.get(self.model_env, os.path.expanduser(self.default_model))
        )


QWEN_MODEL = ModelTestCase(
    "qwen",
    "GGUF_MMQ_TEST_MODEL",
    "~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf",
)
DEEPSEEK_MODEL = ModelTestCase(
    "deepseek",
    "GGUF_DEEPSEEK_MMQ_TEST_MODEL",
    "~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf",
)


@dataclass(frozen=True)
class DenseMMQTestCase:
    model: ModelTestCase
    name: str
    tensor_name: str
    quant_type: str
    out_features: int
    in_features: int

    @property
    def id(self) -> str:
        return f"{self.model.family}-{self.name}"


QWEN_DENSE_MMQ_TEST_CASES = (
    DenseMMQTestCase(QWEN_MODEL, "q3_k", "blk.3.attn_q.weight", "Q3_K", 37, 2048),
    DenseMMQTestCase(QWEN_MODEL, "q4_k", "blk.0.attn_gate.weight", "Q4_K", 37, 2048),
    DenseMMQTestCase(QWEN_MODEL, "q5_k", "blk.4.attn_qkv.weight", "Q5_K", 37, 2048),
    DenseMMQTestCase(QWEN_MODEL, "q6_k", "output.weight", "Q6_K", 37, 2048),
)
DEEPSEEK_DENSE_MMQ_TEST_CASES = tuple(
    DenseMMQTestCase(
        DEEPSEEK_MODEL,
        case.name,
        case.tensor_name,
        case.quant_type,
        129,
        case.in_features,
    )
    for case in DEEPSEEK_DENSE_TENSOR_CASES
)
DENSE_MMQ_TEST_CASES = QWEN_DENSE_MMQ_TEST_CASES + DEEPSEEK_DENSE_MMQ_TEST_CASES
DENSE_MMQ_TEST_CASE_IDS = tuple(case.id for case in DENSE_MMQ_TEST_CASES)

QWEN_Q4_DENSE_MMQ_TEST_CASE = next(
    case for case in QWEN_DENSE_MMQ_TEST_CASES if case.quant_type == "Q4_K"
)


@dataclass(frozen=True)
class RoutedMMQTestCase:
    model: ModelTestCase
    name: str
    tensor_name: str
    quant_type: str

    @property
    def id(self) -> str:
        return f"{self.model.family}-{self.name}"


ROUTED_MMQ_TEST_CASES = (
    RoutedMMQTestCase(QWEN_MODEL, "q3_k", "blk.0.ffn_gate_exps.weight", "Q3_K"),
    RoutedMMQTestCase(QWEN_MODEL, "q4_k", "blk.2.ffn_down_exps.weight", "Q4_K"),
    RoutedMMQTestCase(QWEN_MODEL, "q5_k", "blk.0.ffn_down_exps.weight", "Q5_K"),
    RoutedMMQTestCase(QWEN_MODEL, "q6_k", "output.weight", "Q6_K"),
    RoutedMMQTestCase(QWEN_MODEL, "iq2_s", "blk.10.ffn_gate_exps.weight", "IQ2_S"),
    RoutedMMQTestCase(
        DEEPSEEK_MODEL, "iq2_xxs", "blk.0.ffn_gate_exps.weight", "IQ2_XXS"
    ),
    RoutedMMQTestCase(DEEPSEEK_MODEL, "q2_k", "blk.0.ffn_down_exps.weight", "Q2_K"),
)
ROUTED_MMQ_TEST_CASE_IDS = tuple(case.id for case in ROUTED_MMQ_TEST_CASES)


@dataclass(frozen=True)
class PairedMMQTestCase:
    model: ModelTestCase
    name: str
    first_tensor_name: str
    second_tensor_name: str
    quant_type: str

    @property
    def id(self) -> str:
        return f"{self.model.family}-{self.name}"


PAIRED_MMQ_TEST_CASES = (
    PairedMMQTestCase(
        QWEN_MODEL,
        "q3_k",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "Q3_K",
    ),
    PairedMMQTestCase(
        QWEN_MODEL,
        "q4_k",
        "blk.0.attn_gate.weight",
        "blk.0.attn_qkv.weight",
        "Q4_K",
    ),
    PairedMMQTestCase(
        QWEN_MODEL,
        "q5_k",
        "blk.0.ffn_down_exps.weight",
        "blk.1.ffn_down_exps.weight",
        "Q5_K",
    ),
    PairedMMQTestCase(QWEN_MODEL, "q6_k", "output.weight", "output.weight", "Q6_K"),
    PairedMMQTestCase(
        QWEN_MODEL,
        "iq2_s",
        "blk.10.ffn_gate_exps.weight",
        "blk.10.ffn_up_exps.weight",
        "IQ2_S",
    ),
    PairedMMQTestCase(
        DEEPSEEK_MODEL,
        "iq2_xxs",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "IQ2_XXS",
    ),
)
PAIRED_MMQ_TEST_CASE_IDS = tuple(case.id for case in PAIRED_MMQ_TEST_CASES)
