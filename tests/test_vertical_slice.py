"""End-to-end path on the tiny model: text to every precision (wave index, steps 4 and 5).

This is the one test that crosses every milestone boundary with real objects: pydantic configs,
calibration sampling, module discovery, the empirical Fisher, real cache keys, the k-means
kernels, the on-disk layout, and a real Granite module tree.
"""

from pathlib import Path

import pytest
import torch
from transformers import GraniteMoeHybridForCausalLM

from anyprec.artifacts.keys import quantized_snapshot
from anyprec.artifacts.manifest import module_entries
from anyprec.artifacts.store import ArtifactStore, QuantizedMeta
from anyprec.config.schemas import QuantizerMode
from anyprec.data.calibration import Encoder, sample_calibration
from anyprec.inference.precision import set_precision
from anyprec.models.discovery import find_quantizable_linears
from anyprec.quantization.model import quantize_model
from anyprec.sensitivity.fisher import estimate_fisher
from anyprec.utils.hashing import sha256_key
from tests import factories


@pytest.mark.parametrize("mode", ["incremental", "standalone"])
def test_tiny_model_goes_from_calibration_text_to_every_precision(
    tmp_path: Path,
    tiny_model: GraniteMoeHybridForCausalLM,
    char_encoder: Encoder,
    mode: QuantizerMode,
) -> None:
    """
    Given: the tiny model, an in-memory corpus, and a quantization run config for one mode.
    When: calibration is sampled, the Fisher is estimated, cached, and reloaded, the model is
        quantized with the reloaded Fisher, cached, reloaded, and set to every bit-width 2..4.
    Then: the reloaded Fisher equals the estimate; each target weight equals its codebook
        gathered at that width's indices; other parameters never change; the model still
        produces finite logits; and the mean recorded error falls from 2 to 4 bits.
    """
    config = factories.quantize_run_config(tmp_path, mode)
    store = ArtifactStore(config.output.cache_dir)
    cpu = torch.device("cpu")

    # Phase 1: calibration tokens and the empirical Fisher, through the Fisher cache.

    calibration = sample_calibration(factories.corpus(), char_encoder, config.calibration)
    targets = find_quantizable_linears(tiny_model, config.model.quantizable_modules)
    weights = {name: linear.weight for name, linear in targets.items()}
    modules = module_entries(weights)
    fisher = estimate_fisher(tiny_model, calibration, targets)
    f_snapshot, f_key = factories.fisher_snapshot_and_key(config)
    store.save_fisher(f_key, f_snapshot, fisher, factories.fisher_meta(config))
    diagonals = store.load_fisher(f_key, f_snapshot, modules)
    assert all(torch.equal(diagonals[n], fisher.diagonals[n]) for n in weights)

    # Phase 2: quantize with the reloaded Fisher, through the quantized cache.

    quantization = quantize_model(weights, diagonals, config.quantizer, cpu)
    q_snapshot = quantized_snapshot(f_key, config.quantizer)
    q_key = sha256_key(q_snapshot)
    store.save_quantized(q_key, q_snapshot, quantization, QuantizedMeta.from_config(config, cpu))
    artifact = store.load_quantized(q_key, q_snapshot, modules)

    # Phase 3: every precision, recomputed independently from the reloaded tensors.

    untouched = {
        name: p.detach().clone()
        for name, p in tiny_model.named_parameters()
        if name.removesuffix(".weight") not in targets
    }
    parent = artifact.manifest.parent_bits
    for bits in range(config.quantizer.seed_bits, parent + 1):
        set_precision(tiny_model, artifact, bits)

        # Recompute each weight from the reloaded tensors; nested mode shifts the parent.

        for name, weight in weights.items():
            if bits in artifact.indices:
                idx = artifact.indices[bits][name]
            else:
                idx = artifact.indices[parent][name] >> (parent - bits)
            expected = artifact.luts[bits][name].float().gather(1, idx.long())
            assert torch.equal(weight, expected)

        # Non-target parameters are never written, and the quantized model still runs.

        for name, p in tiny_model.named_parameters():
            if name in untouched:
                assert torch.equal(p, untouched[name])
        with torch.inference_mode():
            logits = tiny_model(input_ids=calibration[:1]).logits
            assert bool(torch.isfinite(logits).all())

    # More bits must buy a lower Fisher-weighted error, on average over the modules.

    assert artifact.stats is not None
    errors = artifact.stats.relative_error.values()
    assert sum(e[4] for e in errors) < sum(e[2] for e in errors)
