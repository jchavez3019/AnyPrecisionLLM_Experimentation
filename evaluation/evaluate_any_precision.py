"""Evaluate cached any-precision artifacts against the unquantized reference (spec 0009).

Example: ``python evaluation/evaluate_any_precision.py eval.max_chunks=2 'modes=[incremental]'``.
"""

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from anyprec.config.loading import load_evaluate_config
from anyprec.evaluation.pipeline import RESULTS_FILE, run_evaluation

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    """Validate the composed config, run the pipeline in Hydra's run directory, and log metrics.

    :param cfg: The config composed by Hydra from ``configs/evaluate.yaml``.
    """
    run_config = load_evaluate_config(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    results = run_evaluation(run_config, run_dir)

    # One line per (mode, bits), so the run log alone shows the quality curve.

    for entry in results.entries:
        perplexity = " ".join(f"ppl_{p.dataset}={p.perplexity:.3f}" for p in entry.perplexity)
        logger.info(
            "%s %d-bit: kl_mean=%.3e top1=%.4f %s",
            entry.mode,
            entry.bits,
            entry.kl_mean,
            entry.top1_agreement,
            perplexity,
        )
    logger.info("results written to %s", run_dir / RESULTS_FILE)


if __name__ == "__main__":
    main()
