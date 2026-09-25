"""End-to-end path on the tiny model: text to ``results.json`` (wave index, steps 4 to 6).

This is the one test that crosses every milestone boundary with real objects: pydantic configs,
calibration sampling, module discovery, the empirical Fisher, real cache keys, the k-means
kernels, the on-disk layout, a real Granite module tree, the sliced metrics, and the results
schema. It follows the order of the two pipelines of spec 0009 without their I/O wrappers.
"""

from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import GraniteMoeHybridForCausalLM

from anyprec.artifacts.keys import fisher_key, quantized_snapshot
from anyprec.artifacts.manifest import module_entries
from anyprec.artifacts.store import ArtifactStore, QuantizedArtifact, QuantizedMeta
from anyprec.config.schemas import EvalDatasetName, EvaluateRunConfig, QuantizerMode
from anyprec.data.calibration import Encoder, sample_calibration
from anyprec.data.evaluation_text import iter_chunks, load_eval_tokens
from anyprec.evaluation.bits import bits_report
from anyprec.evaluation.metrics import ChunkOutputs, MetricSummary, StreamingMetrics, chunk_metrics
from anyprec.evaluation.results import (
    RESULTS_SCHEMA_VERSION,
    QuantizedEntry,
    Results,
    make_quantized_entry,
    make_reference_entry,
)
from anyprec.inference.precision import restore_weights, set_precision, snapshot_weights
from anyprec.models.discovery import find_quantizable_linears
from anyprec.models.heads import body_hidden_states, check_sliced_logits, logit_head
from anyprec.quantization.model import quantize_model
from anyprec.sensitivity.fisher import estimate_fisher
from anyprec.utils.hashing import sha256_key
from anyprec.utils.versions import library_versions
from tests import factories

_CPU: torch.device = torch.device("cpu")


def _summarize(
    quantized: GraniteMoeHybridForCausalLM,
    reference: GraniteMoeHybridForCausalLM | None,
    tokens: torch.Tensor,
    cfg: EvaluateRunConfig,
) -> MetricSummary:
    """Stream one dataset through the evaluated model, with KL against ``reference`` if given.

    :param quantized: The evaluated model.
    :param reference: The unquantized model, or ``None`` for perplexity only.
    :param tokens: int64 ``[L]`` token stream.
    :param cfg: The evaluation config.
    :return: The dataset's summary.
    """
    stats = StreamingMetrics(cfg.eval.kl.quantile)
    with torch.inference_mode():
        for chunk in iter_chunks(tokens, cfg.eval.chunk_len, cfg.eval.max_chunks):
            # [1, T] chunk -> [T, H] hidden states per model; the heads are sliced inside.

            q_out = ChunkOutputs(body_hidden_states(quantized, chunk), logit_head(quantized))
            ref_out = None
            if reference is not None:
                ref_out = ChunkOutputs(body_hidden_states(reference, chunk), logit_head(reference))
            stats.update(chunk_metrics(q_out, ref_out, chunk[0], cfg.eval.lm_head_chunk_tokens))
    return stats.summary()


def _check_every_precision(
    model: GraniteMoeHybridForCausalLM,
    artifact: QuantizedArtifact,
    untouched: dict[str, torch.Tensor],
) -> None:
    """Set every bit-width and recompute each target weight independently from the artifact.

    Also checks that the model still produces finite logits at every width, and that the
    recorded Fisher-weighted error is lower at the parent width than at the seed width.

    :param model: The model being quantized.
    :param artifact: The reloaded artifact.
    :param untouched: Clones of every non-target parameter, which must never change.
    """
    manifest = artifact.manifest
    parent = manifest.parent_bits
    for bits in range(manifest.seed_bits, parent + 1):
        set_precision(model, artifact, bits)

        # Recompute each weight from the reloaded tensors; nested mode shifts the parent.

        for name, weight in factories.target_weights(model).items():
            if bits in artifact.indices:
                idx = artifact.indices[bits][name]
            else:
                idx = artifact.indices[parent][name] >> (parent - bits)
            expected = artifact.luts[bits][name].float().gather(1, idx.long())
            assert torch.equal(weight, expected)

        # Non-target parameters are never written, and the quantized model still runs.

        for name, p in model.named_parameters():
            if name in untouched:
                assert torch.equal(p, untouched[name])
        with torch.inference_mode():
            logits = model(input_ids=torch.arange(16)[None, :]).logits
            assert bool(torch.isfinite(logits).all())

    # More bits must buy a lower Fisher-weighted error, on average over the modules.

    assert artifact.stats is not None
    errors = artifact.stats.relative_error.values()
    assert sum(e[parent] for e in errors) < sum(e[manifest.seed_bits] for e in errors)


def test_tiny_model_goes_from_calibration_text_to_a_valid_results_file(
    tmp_path: Path, tiny_model: GraniteMoeHybridForCausalLM, char_encoder: Encoder
) -> None:
    """
    Given: the tiny model, an in-memory corpus, and matching quantization and evaluation configs.
    When: one Fisher is estimated and cached, both modes are quantized from it and cached, every
        precision is set, and both modes are evaluated against an unmodified reference.
    Then: cached tensors reload exactly; weights equal their codebooks at every width; the
        same-weights check gives zero KL; 2-bit KL exceeds 4-bit KL in each mode; and the
        written results.json re-validates to the same object.
    """
    ecfg = factories.evaluate_run_config(tmp_path)
    store = ArtifactStore(ecfg.output.cache_dir)
    texts = factories.corpus()

    # Phase 1: calibration tokens and the empirical Fisher, through the Fisher cache. Both
    # pipelines must derive the same Fisher key from their own configs.

    qcfg = factories.quantize_run_config(tmp_path)
    targets = find_quantizable_linears(tiny_model, ecfg.model.quantizable_modules)
    weights = {name: linear.weight for name, linear in targets.items()}
    modules = module_entries(weights)
    original = snapshot_weights(tiny_model, list(targets))
    calibration = sample_calibration(texts, char_encoder, qcfg.calibration)
    fisher = estimate_fisher(tiny_model, calibration, targets)
    f_snapshot, f_key = factories.fisher_snapshot_and_key(qcfg)
    assert f_key == fisher_key(ecfg.model, ecfg.calibration, ecfg.rotation)
    store.save_fisher(f_key, f_snapshot, fisher, factories.fisher_meta(qcfg))
    diagonals = store.load_fisher(f_key, f_snapshot, modules)
    assert all(torch.equal(diagonals[n], fisher.diagonals[n]) for n in weights)

    # Phase 2: both modes from the one reloaded Fisher, keyed as the evaluation pipeline will.

    untouched = {
        name: p.detach().clone()
        for name, p in tiny_model.named_parameters()
        if name.removesuffix(".weight") not in targets
    }
    artifacts: dict[QuantizerMode, QuantizedArtifact] = {}
    for mode in ecfg.modes:
        quantizer = ecfg.quantizer.model_copy(update={"mode": mode})
        quantization = quantize_model(weights, diagonals, quantizer, _CPU)
        q_snapshot = quantized_snapshot(f_key, quantizer)
        q_key = sha256_key(q_snapshot)
        meta = QuantizedMeta.from_config(factories.quantize_run_config(tmp_path, mode), _CPU)
        store.save_quantized(q_key, q_snapshot, quantization, meta)
        artifacts[mode] = store.load_quantized(q_key, q_snapshot, modules)
        _check_every_precision(tiny_model, artifacts[mode], untouched)

    # Phase 3: the start-up checks of spec 0008, on restored weights and a fresh reference.

    restore_weights(tiny_model, original)
    reference = factories.tiny_model()
    eval_tokens: dict[EvalDatasetName, torch.Tensor] = {
        name: load_eval_tokens(texts, char_encoder, d) for name, d in ecfg.eval.datasets.items()
    }
    kl_name = ecfg.eval.kl.dataset
    probe = next(iter_chunks(eval_tokens[kl_name], ecfg.eval.chunk_len, 1))
    for model in (reference, tiny_model):
        check_sliced_logits(model, probe, ecfg.eval.lm_head_chunk_tokens)
    same = _summarize(tiny_model, reference, eval_tokens[kl_name], ecfg)
    assert same.kl_mean is not None and abs(same.kl_mean) <= 1e-6
    assert same.top1_agreement == 1.0

    # Phase 4: the sweep. The reference is evaluated once; every (mode, bits) gets one entry.

    report = bits_report(
        [m.shape for m in modules],
        sum(p.numel() for p in tiny_model.parameters()),
        ecfg.quantizer.seed_bits,
        ecfg.quantizer.parent_bits,
    )
    reference_entry = make_reference_entry(
        {name: _summarize(reference, None, tokens, ecfg) for name, tokens in eval_tokens.items()}
    )
    entries: list[QuantizedEntry] = []
    for mode, artifact in artifacts.items():
        for bits in ecfg.eval.bits:
            set_precision(tiny_model, artifact, bits)
            summaries: dict[EvalDatasetName, MetricSummary] = {
                name: _summarize(tiny_model, reference if name == kl_name else None, tokens, ecfg)
                for name, tokens in eval_tokens.items()
            }
            entry = make_quantized_entry(
                mode, bits, artifact.manifest.key, summaries, kl_name, report
            )
            entries.append(entry)

    # Phase 5: results.json, then the claims a reader would check first.

    results = Results(
        schema_version=RESULTS_SCHEMA_VERSION,
        config=ecfg,
        model_id=ecfg.model.model_id,
        revision=ecfg.model.revision,
        fisher_key=f_key,
        versions=library_versions(),
        created_at=datetime.now(UTC),
        bits=report,
        reference=reference_entry,
        entries=entries,
        seconds=0.0,
    )
    path = tmp_path / "results.json"
    path.write_text(results.model_dump_json(indent=2))
    assert Results.model_validate_json(path.read_text()) == results
    for mode in ecfg.modes:
        by_bits = {e.bits: e for e in entries if e.mode == mode}
        assert by_bits[2].kl_mean > by_bits[4].kl_mean
        assert by_bits[2].bits_per_weight < by_bits[4].bits_per_weight
