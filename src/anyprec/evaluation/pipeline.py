"""One evaluation run: every requested (mode, bits) against an unmodified reference (spec 0009).

The metric loop follows spec 0008: the reference is measured once per dataset, and each
(mode, bits) pair is measured on every dataset, with KL on the configured KL dataset only.
Evaluation never quantizes; a missing artifact is an error that names the command creating it.
"""

import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import torch
from tqdm import tqdm

from anyprec.artifacts.keys import fisher_key, quantized_snapshot
from anyprec.artifacts.manifest import ModuleEntry, module_entries
from anyprec.artifacts.store import ArtifactNotFoundError, ArtifactStore, QuantizedArtifact
from anyprec.config.schemas import EvalConfig, EvalDatasetName, EvaluateRunConfig, QuantizerMode
from anyprec.data.calibration import make_encoder
from anyprec.data.evaluation_text import iter_chunks, load_eval_tokens
from anyprec.data.hub import load_texts
from anyprec.evaluation.bits import BitsReport, bits_report
from anyprec.evaluation.metrics import ChunkOutputs, MetricSummary, StreamingMetrics, chunk_metrics
from anyprec.evaluation.results import (
    RESULTS_SCHEMA_VERSION,
    QuantizedEntry,
    Results,
    make_quantized_entry,
    make_reference_entry,
)
from anyprec.inference.precision import set_precision
from anyprec.models.discovery import find_quantizable_linears
from anyprec.models.heads import body_hidden_states, check_sliced_logits, logit_head
from anyprec.models.loading import CausalLM, load_model, load_tokenizer
from anyprec.rotation.resolve import resolve_rotation
from anyprec.utils.devices import resolve_device
from anyprec.utils.dtypes import torch_dtype
from anyprec.utils.hashing import JsonValue, sha256_key
from anyprec.utils.seeding import seed_everything
from anyprec.utils.versions import library_versions

RESULTS_FILE: str = "results.json"
QUANTIZE_SCRIPT: str = "quantization/quantize_any_precision.py"
PROBE_TOKENS: int = 512
SAME_WEIGHTS_MAX_KL: float = 1e-6


class SameWeightsError(RuntimeError):
    """Two freshly loaded model instances disagree before any weight is quantized."""


def run_evaluation(cfg: EvaluateRunConfig, run_dir: Path) -> Results:
    """Measure every requested (mode, bits) pair and write ``results.json`` atomically.

    :param cfg: The validated run config.
    :param run_dir: Hydra's run directory, which receives ``results.json``.
    :return: The results that were written.
    :raises NotImplementedError: For ``rotation=hadamard``, before any model or data is loaded.
    :raises RuntimeError: If CUDA is requested but not available.
    :raises ArtifactNotFoundError: If a requested artifact is missing, before any dataset is
        tokenized; the message names the quantization command that creates it.
    :raises SameWeightsError: If the reference and the model to quantize disagree at start-up.
    """
    started = time.perf_counter()
    resolve_rotation(cfg.rotation)
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)
    store = ArtifactStore(cfg.output.cache_dir)

    # Recompute each requested mode's artifact key; only quantizer.mode differs between them.

    f_key = fisher_key(cfg.model, cfg.calibration, cfg.rotation)
    requested: dict[QuantizerMode, dict[str, JsonValue]] = {
        mode: quantized_snapshot(f_key, cfg.quantizer.with_mode(mode)) for mode in cfg.modes
    }

    # Two independent copies in eval_dtype: the reference is never modified (ADR 0004). The
    # discovered [m, n] modules are what every artifact is checked against.

    dtype = torch_dtype(cfg.model.eval_dtype)
    reference = load_model(cfg.model, dtype, device)
    quantized = load_model(cfg.model, dtype, device)
    targets = find_quantizable_linears(quantized, cfg.model.quantizable_modules)
    modules = module_entries({name: linear.weight for name, linear in targets.items()})

    # Load every artifact up front, so a missing one fails before hours of evaluation.

    artifacts: dict[QuantizerMode, QuantizedArtifact] = {
        mode: _load_or_explain(store, snapshot, modules, mode)
        for mode, snapshot in requested.items()
    }

    # Tokenize each evaluation dataset once; the [L] int64 streams are reused for every pair.

    encode = make_encoder(load_tokenizer(cfg.model))
    eval_tokens: dict[EvalDatasetName, torch.Tensor] = {
        name: load_eval_tokens(
            load_texts(d.path, d.name, d.data_files, d.split, d.text_field), encode, d
        )
        for name, d in cfg.eval.datasets.items()
    }

    # Prove the metric path and the two instances before any quantized weight is written.

    _startup_checks(reference, quantized, eval_tokens[cfg.eval.kl.dataset], cfg.eval, device)
    report = bits_report(
        [m.shape for m in modules],
        sum(p.numel() for p in reference.parameters()),
        cfg.quantizer.seed_bits,
        cfg.quantizer.parent_bits,
    )

    # The metric loop of spec 0008: reference metrics once, then every (mode, bits).

    reference_entry = make_reference_entry(
        {
            name: _evaluate_dataset(reference, None, tokens, cfg.eval, device, cfg.eval.max_chunks)
            for name, tokens in eval_tokens.items()
        }
    )
    entries = _sweep(reference, quantized, artifacts, eval_tokens, cfg.eval, report, device)

    # Persist the self-contained results: config, keys, versions, and every metric.

    results = Results(
        schema_version=RESULTS_SCHEMA_VERSION,
        config=cfg,
        model_id=cfg.model.model_id,
        revision=cfg.model.revision,
        fisher_key=f_key,
        versions=library_versions(),
        created_at=datetime.now(UTC),
        bits=report,
        reference=reference_entry,
        entries=entries,
        seconds=time.perf_counter() - started,
    )
    _write_text_atomic(run_dir / RESULTS_FILE, results.model_dump_json(indent=2))
    return results


def _load_or_explain(
    store: ArtifactStore,
    snapshot: dict[str, JsonValue],
    modules: Sequence[ModuleEntry],
    mode: QuantizerMode,
) -> QuantizedArtifact:
    """Load one mode's artifact, turning a missing one into the command that creates it.

    ``ArtifactMismatchError`` propagates unchanged: an artifact that exists but differs is a
    bug or a stale cache, and no command would fix it silently.

    :param store: The artifact store.
    :param snapshot: The mode's quantized snapshot.
    :param modules: The discovered modules, in order.
    :param mode: The quantizer mode, named in the error message.
    :return: The artifact, with CPU tensors.
    :raises ArtifactNotFoundError: If no artifact exists for the snapshot's key.
    """
    try:
        return store.load_quantized(sha256_key(snapshot), snapshot, modules)
    except ArtifactNotFoundError as error:
        raise ArtifactNotFoundError(
            f"no {mode} artifact for this config ({error}). Create it with "
            f"`python {QUANTIZE_SCRIPT} quantizer.mode={mode}` plus this run's model, "
            "calibration, quantizer, and rotation overrides."
        ) from error


def _startup_checks(
    reference: CausalLM,
    quantized: CausalLM,
    kl_tokens: torch.Tensor,
    cfg: EvalConfig,
    device: torch.device,
) -> None:
    """Run the two start-up checks of spec 0008 on the first KL chunk.

    The first proves that body plus sliced head reproduces ``forward()`` for both models. The
    second proves the two instances and the metric code agree while both hold the original
    weights. KL uses a tolerance because CUDA kernels need not be bitwise reproducible.

    :param reference: The reference model.
    :param quantized: The model that will be quantized; still unmodified here.
    :param kl_tokens: int64 ``[L]`` stream of the KL dataset.
    :param cfg: The evaluation protocol.
    :param device: The models' device.
    :raises ValueError: If the KL stream is shorter than one chunk.
    :raises SlicedLogitsError: If a sliced head does not reproduce ``forward()``.
    :raises SameWeightsError: If the two instances disagree.
    """
    first = next(iter_chunks(kl_tokens, cfg.chunk_len, 1), None)
    if first is None:
        raise ValueError(
            f"the {cfg.kl.dataset} stream is shorter than one {cfg.chunk_len}-token chunk"
        )

    # [1, T] -> [1, min(T, 512)]: the full-vocabulary forward in the check is the largest
    # allocation of the run, so the probe is capped.

    probe = first[:, :PROBE_TOKENS].to(device)
    check_sliced_logits(reference, probe, cfg.lm_head_chunk_tokens)
    check_sliced_logits(quantized, probe, cfg.lm_head_chunk_tokens)
    same = _evaluate_dataset(quantized, reference, kl_tokens, cfg, device, max_chunks=1)
    if same.kl_mean is None or same.kl_mean > SAME_WEIGHTS_MAX_KL or same.top1_agreement != 1.0:
        raise SameWeightsError(
            f"before quantization, KL is {same.kl_mean} and agreement {same.top1_agreement}; "
            f"expected KL <= {SAME_WEIGHTS_MAX_KL} and agreement 1"
        )


def _sweep(
    reference: CausalLM,
    quantized: CausalLM,
    artifacts: dict[QuantizerMode, QuantizedArtifact],
    eval_tokens: dict[EvalDatasetName, torch.Tensor],
    cfg: EvalConfig,
    report: BitsReport,
    device: torch.device,
) -> list[QuantizedEntry]:
    """Evaluate every (mode, bits) pair, ordered by mode, then bits.

    ``set_precision`` rewrites every target weight, so no restore is needed between pairs.

    :param reference: The unmodified reference model.
    :param quantized: The model whose weights are set to each precision in turn.
    :param artifacts: Mode to its loaded artifact, in the config's mode order.
    :param eval_tokens: Dataset name to its int64 ``[L]`` stream.
    :param cfg: The evaluation protocol.
    :param report: Bits per weight of the model.
    :param device: The models' device.
    :return: One entry per pair.
    """
    entries: list[QuantizedEntry] = []
    with tqdm(total=len(artifacts) * len(cfg.bits), desc="evaluate", unit="pair") as bar:
        for mode, artifact in artifacts.items():
            for bits in cfg.bits:
                # KL needs the reference distribution of the same chunk, so the reference body
                # runs again beside the quantized one on the KL dataset only (ADR 0004).

                set_precision(quantized, artifact, bits)
                summaries: dict[EvalDatasetName, MetricSummary] = {
                    name: _evaluate_dataset(
                        quantized,
                        reference if name == cfg.kl.dataset else None,
                        tokens,
                        cfg,
                        device,
                        cfg.max_chunks,
                    )
                    for name, tokens in eval_tokens.items()
                }
                entry = make_quantized_entry(
                    mode, bits, artifact.manifest.key, summaries, cfg.kl.dataset, report
                )
                entries.append(entry)
                bar.update(1)
    return entries


def _evaluate_dataset(
    model: CausalLM,
    reference: CausalLM | None,
    tokens: torch.Tensor,
    cfg: EvalConfig,
    device: torch.device,
    max_chunks: int | None,
) -> MetricSummary:
    """Stream one dataset through ``model``, with KL against ``reference`` if one is given.

    :param model: The evaluated model.
    :param reference: The reference model, or ``None`` for perplexity only.
    :param tokens: int64 ``[L]`` token stream on the CPU.
    :param cfg: The evaluation protocol.
    :param device: The models' device.
    :param max_chunks: Cap on chunks, or ``None`` for the whole stream.
    :return: The dataset's summary.
    """
    # The heads never change: set_precision touches only the decoder linears, never the tied
    # LM head, so each model's head is built once per dataset.

    head = logit_head(model)
    ref_head = logit_head(reference) if reference is not None else None
    stats = StreamingMetrics(cfg.kl.quantile)
    with torch.inference_mode():
        for chunk in iter_chunks(tokens, cfg.chunk_len, max_chunks):
            # [1, T] chunk on the device -> [T, H] final-norm hidden states per model; the
            # heads are applied slice by slice inside chunk_metrics.

            x = chunk.to(device)
            q_out = ChunkOutputs(body_hidden_states(model, x), head)
            ref_out = None
            if reference is not None and ref_head is not None:
                ref_out = ChunkOutputs(body_hidden_states(reference, x), ref_head)
            stats.update(chunk_metrics(q_out, ref_out, x[0], cfg.lm_head_chunk_tokens))
    return stats.summary()


def _write_text_atomic(path: Path, text: str) -> None:
    """Write through a temporary sibling and rename it, so no reader sees a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
