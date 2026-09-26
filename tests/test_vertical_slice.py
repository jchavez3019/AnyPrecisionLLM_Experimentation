"""End-to-end path on the tiny model: calibration text to ``results.json`` (wave index, steps 4 to 7).

This is the one test that crosses every stage with real objects, through the two pipelines of
spec 0009: pydantic configs, calibration sampling, module discovery, the empirical Fisher, real
cache keys, the k-means kernels, the on-disk layout, a real Granite module tree, the sliced
metrics, and the results schema. Only the Hub loaders are replaced (``offline_loaders``).
"""

from pathlib import Path

import torch
from transformers import GraniteMoeHybridForCausalLM

from anyprec.artifacts.keys import quantized_snapshot
from anyprec.artifacts.manifest import module_entries
from anyprec.artifacts.store import ArtifactStore, QuantizedArtifact
from anyprec.config.schemas import QuantizerMode
from anyprec.evaluation.pipeline import run_evaluation
from anyprec.inference.precision import set_precision
from anyprec.quantization.pipeline import run_quantization
from tests import factories
from tests.offline import OfflineLoaders


def _check_every_precision(model: GraniteMoeHybridForCausalLM, artifact: QuantizedArtifact) -> None:
    """Set every bit-width and recompute each target weight independently from the artifact.

    Also checks that non-target parameters never change, that the model still produces finite
    logits at every width, and that the recorded Fisher-weighted error is lower at the parent
    width than at the seed width.

    :param model: A fresh tiny model.
    :param artifact: The reloaded artifact.
    """
    manifest = artifact.manifest
    parent = manifest.parent_bits
    targets = factories.target_weights(model)
    untouched = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if name.removesuffix(".weight") not in targets
    }
    for bits in range(manifest.seed_bits, parent + 1):
        set_precision(model, artifact, bits)

        # Recompute each weight from the reloaded tensors; nested mode shifts the parent.

        for name, weight in targets.items():
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
    tmp_path: Path, offline_loaders: OfflineLoaders
) -> None:
    """
    Given: the tiny model behind offline loaders, and matching quantization and evaluation configs.
    When: both modes are quantized by run_quantization, then evaluated by run_evaluation.
    Then: the second quantization reuses the first one's Fisher; each artifact reproduces its
        codebooks at every width; both pipelines agree on the Fisher key; and in each mode
        2-bit KL exceeds 4-bit KL.
    """
    # Phase 1: one Fisher, both modes. Only the first run reads calibration text.

    outcomes = {
        mode: run_quantization(factories.quantize_run_config(tmp_path, mode), tmp_path / mode)
        for mode in ("incremental", "standalone")
    }
    assert not outcomes["incremental"].fisher_reused
    assert outcomes["standalone"].fisher_reused
    assert offline_loaders.calls.count("load_texts:in-memory:train") == 1

    # Phase 2: each stored artifact, reloaded independently, drives set_precision exactly.

    ecfg = factories.evaluate_run_config(tmp_path)
    store = ArtifactStore(ecfg.output.cache_dir)
    modules = module_entries(factories.target_weights(factories.tiny_model()))
    modes: list[QuantizerMode] = ["incremental", "standalone"]
    for mode in modes:
        snapshot = quantized_snapshot(outcomes[mode].fisher_key, ecfg.quantizer.with_mode(mode))
        artifact = store.load_quantized(outcomes[mode].quantized_key, snapshot, modules)
        _check_every_precision(factories.tiny_model(), artifact)

    # Phase 3: the evaluation pipeline finds both artifacts from its own config alone.

    results = run_evaluation(ecfg, tmp_path / "evaluate")

    assert results.fisher_key == outcomes["incremental"].fisher_key
    for mode in modes:
        by_bits = {e.bits: e for e in results.entries if e.mode == mode}
        assert by_bits[2].kl_mean > by_bits[4].kl_mean
