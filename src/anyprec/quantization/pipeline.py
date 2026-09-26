"""One quantization run: the Fisher cache, the k-means drivers, and the artifact store (spec 0009).

This module and ``anyprec.evaluation.pipeline`` are the only ones that combine models, data, and
artifacts (spec 0001, layer 5). The entry script only composes the config and calls
``run_quantization``.
"""

import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from anyprec.artifacts.keys import fisher_snapshot, quantized_snapshot
from anyprec.artifacts.manifest import module_entries
from anyprec.artifacts.store import ArtifactStore, FisherMeta, QuantizedMeta
from anyprec.config.schemas import QuantizeRunConfig
from anyprec.data.calibration import make_encoder, sample_calibration
from anyprec.data.hub import load_texts
from anyprec.models.discovery import find_quantizable_linears
from anyprec.models.loading import load_model, load_tokenizer
from anyprec.quantization.model import ModelQuantization, quantize_model
from anyprec.rotation.resolve import resolve_rotation
from anyprec.sensitivity.fisher import estimate_fisher
from anyprec.utils.devices import resolve_device
from anyprec.utils.dtypes import torch_dtype
from anyprec.utils.hashing import JsonValue, sha256_key
from anyprec.utils.seeding import seed_everything

SUMMARY_FILE: str = "quantize_summary.json"


@dataclass(frozen=True)
class QuantizationOutcome:
    """What one quantization run produced or reused.

    :param fisher_key: Full key of the Fisher artifact.
    :param quantized_key: Full key of the quantized artifact.
    :param fisher_dir: Directory of the Fisher artifact.
    :param quantized_dir: Directory of the quantized artifact.
    :param fisher_reused: Whether the Fisher came from the cache rather than calibration data.
    :param quantized_reused: Whether the quantized artifact already existed.
    """

    fisher_key: str
    quantized_key: str
    fisher_dir: Path
    quantized_dir: Path
    fisher_reused: bool
    quantized_reused: bool


def run_quantization(cfg: QuantizeRunConfig, run_dir: Path) -> QuantizationOutcome:
    """Produce the quantized artifact of ``cfg.quantizer.mode``, reusing a cached Fisher.

    Running again with only ``quantizer.mode`` changed reuses the Fisher and adds the other
    mode's artifact. An existing artifact is validated against the discovered modules and
    reused; nothing is recomputed.

    :param cfg: The validated run config.
    :param run_dir: Hydra's run directory, which receives ``quantize_summary.json``.
    :return: Both keys, both directories, and what was reused.
    :raises NotImplementedError: For ``rotation=hadamard``, before any model or data is loaded.
    :raises RuntimeError: If CUDA is requested but not available.
    """
    # Fail fast on configuration the wave cannot run, before any download or GPU allocation.

    resolve_rotation(cfg.rotation)
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)
    store = ArtifactStore(cfg.output.cache_dir)

    # Keys depend only on configuration, so both are known before any model is loaded.

    f_snapshot = fisher_snapshot(cfg.model, cfg.calibration, cfg.rotation)
    f_key = sha256_key(f_snapshot)
    q_snapshot = quantized_snapshot(f_key, cfg.quantizer)
    q_key = sha256_key(q_snapshot)

    # The model is needed even when both artifacts exist, because discovery produces the module
    # list that every load checks against. Weights are [m, n], in model.dtype (ADR 0003).

    model = load_model(cfg.model, torch_dtype(cfg.model.dtype), device)
    targets = find_quantizable_linears(model, cfg.model.quantizable_modules)
    weights = {name: linear.weight for name, linear in targets.items()}
    modules = module_entries(weights)

    if store.has_quantized(q_key):
        store.load_quantized(q_key, q_snapshot, modules)
        return _outcome(store, f_key, q_key, store.has_fisher(f_key), quantized_reused=True)

    # Phase 1: the Fisher diagonal, from the cache or computed on the calibration set.

    fisher_reused = store.has_fisher(f_key)
    if fisher_reused:
        fisher = store.load_fisher(f_key, f_snapshot, modules)
    else:
        cal = cfg.calibration
        texts = load_texts(cal.path, cal.name, cal.data_files, cal.split, cal.text_field)
        encode = make_encoder(load_tokenizer(cfg.model))

        # [N, T] int64 calibration tokens; the Fisher pass moves one sequence at a time.

        calibration = sample_calibration(texts, encode, cal)
        with tqdm(total=cal.num_sequences, desc="fisher", unit="seq") as bar:
            result = estimate_fisher(
                model, calibration, targets, progress=_fisher_progress(bar.update)
            )
        store.save_fisher(f_key, f_snapshot, result, FisherMeta.from_config(cfg, device))
        fisher = result.diagonals
        del result, calibration

    # Phase 2: k-means over every module. Free the Fisher pass's cached blocks first, because
    # the k-means++ trial tensors are the next largest allocation. This mapping is where
    # ADR 0006's rotation will apply once it is implemented.

    if device.type == "cuda":
        torch.cuda.empty_cache()
    with tqdm(total=len(weights), desc="k-means", unit="module") as bar:
        quantization = quantize_model(
            weights, fisher, cfg.quantizer, device, progress=_module_progress(bar.update)
        )

    # Phase 3: persist atomically, and leave a human-readable summary in the run directory.

    meta = QuantizedMeta.from_config(cfg, device)
    store.save_quantized(q_key, q_snapshot, quantization, meta)
    outcome = _outcome(store, f_key, q_key, fisher_reused, quantized_reused=False)
    _write_summary(run_dir / SUMMARY_FILE, outcome, quantization, cfg)
    return outcome


def _outcome(
    store: ArtifactStore, f_key: str, q_key: str, fisher_reused: bool, quantized_reused: bool
) -> QuantizationOutcome:
    """Assemble the outcome of a run from its two keys."""
    return QuantizationOutcome(
        fisher_key=f_key,
        quantized_key=q_key,
        fisher_dir=store.fisher_dir(f_key),
        quantized_dir=store.quantized_dir(q_key),
        fisher_reused=fisher_reused,
        quantized_reused=quantized_reused,
    )


def _fisher_progress(update: Callable[[float], object]) -> Callable[[int, int], None]:
    """Adapt a progress bar's ``update`` to ``estimate_fisher``'s ``(done, total)`` callback.

    :param update: The bar's ``update`` method, advancing it by a number of steps.
    :return: The callback, which advances the bar by one sequence per call.
    """

    def advance(done: int, total: int) -> None:
        """Advance the bar by the one sequence just finished."""
        update(1)

    return advance


def _module_progress(update: Callable[[float], object]) -> Callable[[str], None]:
    """Adapt a progress bar's ``update`` to ``quantize_model``'s per-module callback.

    :param update: The bar's ``update`` method, advancing it by a number of steps.
    :return: The callback, which advances the bar by one module per call.
    """

    def advance(name: str) -> None:
        """Advance the bar by the one module just quantized."""
        update(1)

    return advance


def _write_summary(
    path: Path,
    outcome: QuantizationOutcome,
    quantization: ModelQuantization,
    cfg: QuantizeRunConfig,
) -> None:
    """Write the run summary: keys, directories, timing, and the median error per bit-width.

    Nothing reads this file back; the artifacts' manifests and ``stats.json`` are the record.

    :param path: Destination, inside the run directory.
    :param outcome: The run's keys and directories.
    :param quantization: The k-means results.
    :param cfg: The run config, for the bit range.
    """
    # The median over modules summarizes the error curve without one outlier module dominating.

    layers = quantization.layers.values()
    bit_widths = range(cfg.quantizer.seed_bits, cfg.quantizer.parent_bits + 1)
    median_error: dict[str, JsonValue] = {
        str(bits): statistics.median(layer.relative_error[bits] for layer in layers)
        for bits in bit_widths
    }
    summary: dict[str, JsonValue] = {
        "mode": cfg.quantizer.mode,
        "fisher_key": outcome.fisher_key,
        "quantized_key": outcome.quantized_key,
        "fisher_dir": str(outcome.fisher_dir),
        "quantized_dir": str(outcome.quantized_dir),
        "fisher_reused": outcome.fisher_reused,
        "kmeans_seconds": quantization.seconds,
        "median_relative_error": median_error,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
