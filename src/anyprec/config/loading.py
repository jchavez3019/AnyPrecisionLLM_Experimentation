"""Conversion of composed Hydra configs into validated run configs (spec 0002).

This is the only library module that sees an ``omegaconf.DictConfig`` (spec 0001).
"""

from typing import cast

from omegaconf import DictConfig, OmegaConf

from anyprec.config.schemas import EvaluateRunConfig, QuantizeRunConfig


def _to_plain_mapping(cfg: DictConfig) -> dict[str, object]:
    """Resolve interpolations and drop Hydra's runtime node.

    :param cfg: A composed Hydra config.
    :return: A plain dictionary ready for pydantic validation.
    :raises TypeError: If the config is not a mapping.
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("the composed Hydra config must be a mapping")
    mapping = cast("dict[object, object]", container)
    plain: dict[str, object] = {str(key): value for key, value in mapping.items()}
    plain.pop("hydra", None)
    return plain


def load_quantize_config(cfg: DictConfig) -> QuantizeRunConfig:
    """Validate a composed ``quantize.yaml`` into a frozen run config.

    :param cfg: A composed Hydra config.
    :return: The validated quantization run config.
    """
    return QuantizeRunConfig.model_validate(_to_plain_mapping(cfg))


def load_evaluate_config(cfg: DictConfig) -> EvaluateRunConfig:
    """Validate a composed ``evaluate.yaml`` into a frozen run config.

    :param cfg: A composed Hydra config.
    :return: The validated evaluation run config.
    """
    return EvaluateRunConfig.model_validate(_to_plain_mapping(cfg))
