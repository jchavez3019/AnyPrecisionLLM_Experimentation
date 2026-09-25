"""End-to-end path on the tiny model: quantize, store round trip, set_precision (wave index, step 4).

This is the one test that crosses every milestone boundary with real objects: pydantic configs,
real cache keys, the kernels, the on-disk layout, and a real Granite module tree.
"""

from pathlib import Path

import pytest
import torch
from transformers import GraniteMoeHybridForCausalLM

from anyprec.artifacts.manifest import ModuleEntry
from anyprec.artifacts.store import ArtifactStore
from anyprec.config.schemas import QuantizerMode
from anyprec.inference.precision import set_precision
from tests import factories


@pytest.mark.parametrize("mode", ["incremental", "standalone"])
def test_quantized_tiny_model_round_trips_through_store_into_every_precision(
    tmp_path: Path, tiny_model: GraniteMoeHybridForCausalLM, mode: QuantizerMode
) -> None:
    """
    Given: the tiny model, a random Fisher, and a quantization run config for one mode.
    When: the model is quantized, saved, reloaded, and set to every bit-width from 2 to 4.
    Then: each target weight equals its reloaded codebook gathered at that width's indices,
        every other parameter is unchanged, and the model still produces finite logits.
    """
    config = factories.quantize_run_config(tmp_path, mode)
    artifact = factories.stored_tiny_artifact(
        tiny_model, config, ArtifactStore(config.output.cache_dir)
    )
    untouched = {
        name: p.detach().clone()
        for name, p in tiny_model.named_parameters()
        if name.removesuffix(".weight") not in artifact.module_names
    }
    tokens = torch.arange(16)[None, :]

    for bits in range(config.quantizer.seed_bits, config.quantizer.parent_bits + 1):
        set_precision(tiny_model, artifact, bits)

        # Recompute each weight from the reloaded tensors; nested mode shifts the parent.

        parent = artifact.manifest.parent_bits
        for name, weight in factories.target_weights(tiny_model).items():
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
            assert bool(torch.isfinite(tiny_model(input_ids=tokens).logits).all())


def test_fisher_round_trips_through_store_with_its_manifest(tmp_path: Path) -> None:
    """
    Given: a Fisher result for the tiny model's 12 targets and the run config's real snapshot.
    When: it is saved and loaded back with the discovered module list.
    Then: every diagonal is bitwise equal, and the manifest records the run's model and counts.
    """
    config = factories.quantize_run_config(tmp_path)
    weights = factories.target_weights(factories.tiny_model())
    result = factories.fisher_result(weights, config.calibration.num_sequences)
    snapshot, key = factories.fisher_snapshot_and_key(config)
    store = ArtifactStore(config.output.cache_dir)

    store.save_fisher(key, snapshot, result, factories.fisher_meta(config))
    modules = [ModuleEntry(name=n, shape=(w.shape[0], w.shape[1])) for n, w in weights.items()]
    loaded = store.load_fisher(key, snapshot, modules)

    # Tensors, order, and the manifest fields a later acceptance check reads back.

    assert list(loaded) == list(weights)
    assert all(torch.equal(loaded[n], result.diagonals[n]) for n in weights)
    manifest_text = (store.fisher_dir(key) / "manifest.json").read_text()
    assert config.model.model_id in manifest_text
    assert store.has_fisher(key)
