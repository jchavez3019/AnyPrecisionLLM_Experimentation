"""Tests for empirical Fisher estimation (spec 0004)."""

import pytest
import torch
from torch import nn
from transformers import GraniteMoeHybridForCausalLM

from anyprec.config.schemas import ModelConfig
from anyprec.models.discovery import find_quantizable_linears
from anyprec.models.heads import causal_lm_loss
from anyprec.sensitivity.fisher import estimate_fisher

_CALIBRATION: torch.Tensor = torch.randint(
    0, 256, (3, 16), generator=torch.Generator().manual_seed(0)
)


@pytest.fixture
def targets(
    tiny_model: GraniteMoeHybridForCausalLM, tiny_model_config: ModelConfig
) -> dict[str, nn.Linear]:
    """The tiny model's 12 quantizable linears, in discovery order."""
    return find_quantizable_linears(tiny_model, tiny_model_config.quantizable_modules)


def _per_sequence_gradients(
    model: GraniteMoeHybridForCausalLM, targets: dict[str, nn.Linear]
) -> list[dict[str, torch.Tensor]]:
    """Reference gradients: one ``torch.autograd.grad`` call per calibration sequence."""
    weights = [linear.weight for linear in targets.values()]
    gradients: list[dict[str, torch.Tensor]] = []
    for i in range(_CALIBRATION.shape[0]):
        loss = causal_lm_loss(model, _CALIBRATION[i : i + 1])
        grads = torch.autograd.grad(loss, weights)
        gradients.append(dict(zip(targets, grads, strict=True)))
    return gradients


def _state(model: nn.Module) -> dict[str, tuple[bool, bool]]:
    """Each parameter's ``(requires_grad, grad is None)``."""
    return {name: (p.requires_grad, p.grad is None) for name, p in model.named_parameters()}


def test_fisher_is_the_sum_of_per_sequence_squared_gradients(
    tiny_model: GraniteMoeHybridForCausalLM, targets: dict[str, nn.Linear]
) -> None:
    """
    Given: three random 16-token sequences and the float32 tiny model.
    When: the Fisher is estimated.
    Then: every diagonal equals sum_i g_i**2 from per-sequence autograd.grad (rtol 1e-5), and
        differs from (sum_i g_i)**2, which is what batching the sequences would compute.
    """
    gradients = _per_sequence_gradients(tiny_model, targets)

    result = estimate_fisher(tiny_model, _CALIBRATION, targets)

    # Squared per sequence, then summed: 3 x [m, n] -> [m, n].

    for name in targets:
        expected = torch.stack([g[name].square() for g in gradients]).sum(dim=0)
        batched = torch.stack([g[name] for g in gradients]).sum(dim=0).square()
        torch.testing.assert_close(result.diagonals[name], expected, rtol=1e-5, atol=1e-12)
        assert not torch.allclose(result.diagonals[name], batched, rtol=1e-3)


def test_output_contract_and_progress_follow_the_calibration_order(
    tiny_model: GraniteMoeHybridForCausalLM, targets: dict[str, nn.Linear]
) -> None:
    """
    Given: three calibration sequences and a progress recorder.
    When: the Fisher is estimated.
    Then: diagonals are float32, CPU, shaped like their weights, non-negative, and keyed in
        targets order; losses are the per-sequence NLLs [3]; progress reports 1/3, 2/3, 3/3.
    """
    calls: list[tuple[int, int]] = []

    result = estimate_fisher(
        tiny_model, _CALIBRATION, targets, lambda done, total: calls.append((done, total))
    )

    assert list(result.diagonals) == list(targets)
    for name, diagonal in result.diagonals.items():
        assert diagonal.dtype == torch.float32 and diagonal.device.type == "cpu"
        assert diagonal.shape == targets[name].weight.shape
        assert bool((diagonal >= 0).all())
    with torch.no_grad():
        expected = torch.stack(
            [causal_lm_loss(tiny_model, _CALIBRATION[i : i + 1]) for i in range(3)]
        )
    torch.testing.assert_close(result.losses, expected)
    assert calls == [(1, 3), (2, 3), (3, 3)]
    assert result.seconds >= 0.0


def test_model_state_is_restored_and_no_hook_remains(
    tiny_model: GraniteMoeHybridForCausalLM, targets: dict[str, nn.Linear]
) -> None:
    """
    Given: the tiny model with its embedding frozen beforehand.
    When: the Fisher is estimated, and then an ordinary backward pass is run.
    Then: requires_grad flags and empty grads match the state before the call, the backward
        leaves the returned diagonals unchanged, and it populates target grads normally.
    """
    tiny_model.get_input_embeddings().weight.requires_grad_(False)
    before = _state(tiny_model)

    result = estimate_fisher(tiny_model, _CALIBRATION, targets)
    after = _state(tiny_model)
    snapshot = {name: d.clone() for name, d in result.diagonals.items()}
    torch.autograd.backward(causal_lm_loss(tiny_model, _CALIBRATION[:1]))

    # A leftover hook would have squared this gradient into the CPU diagonals and freed .grad.

    assert after == before
    assert all(torch.equal(snapshot[n], d) for n, d in result.diagonals.items())
    assert all(linear.weight.grad is not None for linear in targets.values())


def test_failure_midway_restores_the_model_and_propagates(
    tiny_model: GraniteMoeHybridForCausalLM, targets: dict[str, nn.Linear]
) -> None:
    """
    Given: a model whose forward raises on the second sequence.
    When: the Fisher is estimated.
    Then: the error propagates, and flags, grads, and hooks are restored as after a success.
    """
    before = _state(tiny_model)
    calls: list[int] = []

    def fail_on_second_call(module: nn.Module, args: tuple[object, ...]) -> None:
        """Forward pre-hook: let the first sequence through, then fail like an OOM would."""
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("out of memory")

    handle = tiny_model.register_forward_pre_hook(fail_on_second_call)
    try:
        with pytest.raises(RuntimeError, match="out of memory"):
            estimate_fisher(tiny_model, _CALIBRATION, targets)
    finally:
        handle.remove()

    # With the model's own state restored, an ordinary backward fills target grads again.

    assert _state(tiny_model) == before
    torch.autograd.backward(causal_lm_loss(tiny_model, _CALIBRATION[:1]))
    assert all(linear.weight.grad is not None for linear in targets.values())


def test_refuses_to_run_with_gradients_disabled(
    tiny_model: GraniteMoeHybridForCausalLM, targets: dict[str, nn.Linear]
) -> None:
    """
    Given: a caller inside torch.no_grad().
    When: the Fisher is estimated.
    Then: RuntimeError is raised instead of returning all-zero diagonals.
    """
    with torch.no_grad(), pytest.raises(RuntimeError, match="grad mode"):
        estimate_fisher(tiny_model, _CALIBRATION, targets)
