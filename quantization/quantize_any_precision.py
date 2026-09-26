"""Quantize the configured model into an any-precision artifact (spec 0009).

Example: ``python quantization/quantize_any_precision.py quantizer.mode=standalone``.
"""

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from anyprec.config.loading import load_quantize_config
from anyprec.quantization.pipeline import run_quantization

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="quantize")
def main(cfg: DictConfig) -> None:
    """Validate the composed config, run the pipeline in Hydra's run directory, and log keys.

    :param cfg: The config composed by Hydra from ``configs/quantize.yaml``.
    """
    run_config = load_quantize_config(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    outcome = run_quantization(run_config, run_dir)
    fisher_state = "reused" if outcome.fisher_reused else "computed"
    quantized_state = "reused" if outcome.quantized_reused else "computed"
    logger.info("fisher %s (%s) at %s", outcome.fisher_key[:16], fisher_state, outcome.fisher_dir)
    logger.info(
        "quantized %s (%s) at %s",
        outcome.quantized_key[:16],
        quantized_state,
        outcome.quantized_dir,
    )


if __name__ == "__main__":
    main()
