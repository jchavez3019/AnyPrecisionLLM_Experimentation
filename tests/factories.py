"""Builders of real config and model objects for tests (spec 0010).

Every config is an instance of the pydantic schemas in ``anyprec.config.schemas``, so a fixture
can never drift from the schema it stands in for.
"""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

from anyprec.artifacts.keys import fisher_key, quantized_snapshot
from anyprec.artifacts.manifest import module_entries
from anyprec.artifacts.store import ArtifactStore, QuantizedArtifact, QuantizedMeta
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
from anyprec.models.discovery import find_quantizable_linears
from anyprec.quantization.model import quantize_model
from anyprec.sensitivity.fisher import FisherResult
from anyprec.utils.hashing import sha256_key

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
    # Hugging Face initializes weights from the global generator, so it must be seeded. Restoring
    # the saved state afterwards means building a model never changes another test's draws.

    saved = torch.default_generator.get_state()
    try:
        torch.default_generator.manual_seed(seed)
        return GraniteMoeHybridForCausalLM(tiny_granite_config()).eval()
    finally:
        torch.default_generator.set_state(saved)


def char_encode(text: str) -> list[int]:
    """Encode each UTF-8 byte as one token id in ``[0, 256)``, the tiny model's vocabulary.

    :param text: Raw text.
    :return: One token id per byte.
    """
    return list(text.encode("utf-8"))


def corpus(num_documents: int = 12) -> list[str]:
    """Deterministic documents whose byte lengths cycle through 7, 23, 39, 55, 71, and 87.

    A third of them are shorter than the 32-token calibration length, so sampling must skip them.

    :param num_documents: Number of documents.
    :return: The documents.
    """
    return [f"doc{i:03d} " + "abcdefgh" * (i % 6) * 2 for i in range(num_documents)]


def target_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """Select the tiny model's quantizable weight matrices through real module discovery.

    :param model: The tiny Granite model.
    :return: Qualified module name to its live ``[m, n]`` weight parameter, in discovery order.
    """
    targets = find_quantizable_linears(model, model_config().quantizable_modules)
    return {name: linear.weight for name, linear in targets.items()}


def random_fisher(weights: Mapping[str, torch.Tensor], seed: int = 0) -> dict[str, torch.Tensor]:
    """Draw a positive float32 Fisher diagonal per weight from a local generator.

    :param weights: Name to weight matrix.
    :param seed: Generator seed.
    :return: Name to ``[m, n]`` Fisher tensor on the CPU.
    """
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.rand(tuple(weight.shape), generator=generator) + 1e-3
        for name, weight in weights.items()
    }


def fisher_result(weights: Mapping[str, torch.Tensor], num_losses: int) -> FisherResult:
    """A Fisher result with random diagonals and ``num_losses`` plausible per-sequence losses.

    :param weights: Name to weight matrix.
    :param num_losses: Length of the losses vector.
    :return: The result, with CPU tensors.
    """
    losses = torch.linspace(3.0, 3.5, num_losses)
    return FisherResult(diagonals=random_fisher(weights), losses=losses, seconds=1.5)


def stored_tiny_artifact(
    model: nn.Module, config: QuantizeRunConfig, store: ArtifactStore
) -> QuantizedArtifact:
    """Quantize the tiny model with a random Fisher, save it, and load it back through the store.

    Keys and snapshots come from the real key functions, so the artifact is exactly what the
    quantization pipeline would write for ``config``.

    :param model: The tiny Granite model; its weights are only read.
    :param config: The run config, whose quantizer settings are used.
    :param store: The store to write into.
    :return: The reloaded artifact, with CPU tensors.
    """
    weights = target_weights(model)
    quantization = quantize_model(
        weights, random_fisher(weights), config.quantizer, torch.device("cpu")
    )

    # Real snapshots and keys, so the round trip exercises the same identity checks as a run.

    f_key = fisher_key(config.model, config.calibration, config.rotation)
    snapshot = quantized_snapshot(f_key, config.quantizer)
    key = sha256_key(snapshot)
    store.save_quantized(
        key, snapshot, quantization, QuantizedMeta.from_config(config, torch.device("cpu"))
    )
    return store.load_quantized(key, snapshot, module_entries(weights))


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


def quantize_run_config(root: Path, mode: QuantizerMode = "incremental") -> QuantizeRunConfig:
    """A complete quantization run config for the tiny model on the CPU.

    :param root: Temporary output directory.
    :param mode: Quantizer mode.
    :return: A ``QuantizeRunConfig``.
    """
    return QuantizeRunConfig(
        seed=0,
        device="cpu",
        model=model_config(),
        calibration=calibration_config(),
        quantizer=quantizer_config(mode=mode),
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
