"""Builders of real config and model objects for tests (spec 0010).

Every config is an instance of the pydantic schemas in ``anyprec.config.schemas``, so a fixture
can never drift from the schema it stands in for.
"""

from pathlib import Path

import torch
from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

from anyprec.config.schemas import (
    CalibrationConfig,
    EvalConfig,
    EvalDatasetConfig,
    EvaluateRunConfig,
    KLConfig,
    ModelConfig,
    OutputConfig,
    QuantizableModules,
    QuantizerConfig,
    QuantizerMode,
    QuantizeRunConfig,
    RotationNone,
)

TINY_PATTERN: str = (
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|shared_mlp\.(input|output)_linear)$"
)
TINY_QUANTIZABLE_COUNT: int = 12


def tiny_granite_config() -> GraniteMoeHybridConfig:
    """Return the verified two-layer Granite configuration of spec 0010.

    :return: A config with 12 quantizable linears and 90,432 parameters.
    """
    return GraniteMoeHybridConfig(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        shared_intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        num_local_experts=0,
        layer_types=["attention", "attention"],
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )


def tiny_model(seed: int = 0) -> GraniteMoeHybridForCausalLM:
    """Build a freshly initialized float32 tiny Granite model in eval mode.

    :param seed: Seed of the weight initialization.
    :return: The model on the CPU.
    """
    torch.default_generator.manual_seed(seed)
    return GraniteMoeHybridForCausalLM(tiny_granite_config()).eval()


def model_config() -> ModelConfig:
    """Model settings matching the tiny Granite model.

    :return: A ``ModelConfig`` whose pattern selects the 12 tiny linears.
    """
    return ModelConfig(
        model_id="tiny/granite",
        revision="0" * 40,
        dtype="float32",
        eval_dtype="float32",
        quantizable_modules=QuantizableModules(
            pattern=TINY_PATTERN, expected_count=TINY_QUANTIZABLE_COUNT
        ),
    )


def calibration_config() -> CalibrationConfig:
    """Small calibration settings.

    :return: A ``CalibrationConfig`` for 4 sequences of 32 tokens.
    """
    return CalibrationConfig(
        path="in-memory",
        split="train",
        text_field="text",
        num_sequences=4,
        seq_len=32,
        seed=0,
    )


def quantizer_config(mode: QuantizerMode = "incremental", row_chunk: int = 16) -> QuantizerConfig:
    """Quantizer settings sized for the tiny model (2 to 4 bits, so ``2**b <= n``).

    :param mode: Quantizer mode.
    :param row_chunk: Rows per kernel call.
    :return: A ``QuantizerConfig``.
    """
    return QuantizerConfig(
        mode=mode,
        seed_bits=2,
        parent_bits=4,
        seed=0,
        lloyd_max_iter=50,
        empty_eps=1e-12,
        row_chunk=row_chunk,
    )


def output_config(root: Path) -> OutputConfig:
    """Output locations under a temporary directory.

    :param root: Temporary directory.
    :return: An ``OutputConfig``.
    """
    return OutputConfig(base_dir=root, cache_dir=root / "cache")


def eval_config() -> EvalConfig:
    """A small evaluation protocol.

    :return: An ``EvalConfig`` with two datasets, 32-token chunks, and bit-widths 2 to 4.
    """
    dataset = EvalDatasetConfig(path="in-memory", split="test", text_field="text", joiner="\n\n")
    return EvalConfig(
        chunk_len=32,
        max_chunks=2,
        lm_head_chunk_tokens=8,
        datasets={"wikitext2": dataset, "c4": dataset},
        kl=KLConfig(dataset="wikitext2", quantile=0.99),
        bits=[2, 3, 4],
    )


def quantize_run_config(root: Path) -> QuantizeRunConfig:
    """A complete quantization run config for the tiny model on the CPU.

    :param root: Temporary output directory.
    :return: A ``QuantizeRunConfig``.
    """
    return QuantizeRunConfig(
        seed=0,
        device="cpu",
        model=model_config(),
        calibration=calibration_config(),
        quantizer=quantizer_config(),
        rotation=RotationNone(kind="none"),
        output=output_config(root),
    )


def evaluate_run_config(root: Path) -> EvaluateRunConfig:
    """A complete evaluation run config for the tiny model on the CPU.

    :param root: Temporary output directory.
    :return: An ``EvaluateRunConfig``.
    """
    return EvaluateRunConfig(
        seed=0,
        device="cpu",
        modes=["incremental", "standalone"],
        model=model_config(),
        calibration=calibration_config(),
        quantizer=quantizer_config(),
        rotation=RotationNone(kind="none"),
        eval=eval_config(),
        output=output_config(root),
    )
