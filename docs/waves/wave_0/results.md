# Wave 0: Acceptance Run Results

- Run date: 2026-09-25 (results written 2026-09-26T03:33:05Z)
- Versions: anyprec 0.1.0, torch 2.9.1+cu126, transformers 5.17.0, datasets 5.0.1, CUDA 12.6
- Hardware: RTX 3060 Laptop GPU (6 GiB)
- Results: `outputs/evaluate/2026-09-25/21-38-07/results.json`
- Fisher artifact: `d0f74a509f33d0c0`; quantized artifacts: `f061f6ca51bcb0f5` (incremental), `88c41fe179e8d324` (standalone)
- Acceptance checker: `python evaluation/check_acceptance.py <results.json>` passes 19 of 19 checks

## Wall-clock time and memory

These are the costs of each stage of [spec 0011](0011-integration-and-acceptance.md)'s acceptance run. The GPU throttled thermally for most of the run (82 °C, 502 MHz of 2100 MHz), so the long stages are slower than the hardware allows.

| Stage | Wall clock | Peak GPU memory |
| --- | --- | --- |
| Fisher, 100 sequences (mean loss 3.2242) | 14.3 s | within the 2.63 GiB below |
| k-means, incremental | 16.6 s | 2.63 GiB for Fisher and k-means together |
| k-means, standalone | 2,280 s (38 min) | at most 0.46 GiB above the module's weights and Fisher |
| Evaluation, both modes, 12 precision pairs | 6,898 s (1 h 55 min) | 3.40 GiB |

Incremental k-means is far below the 3 to 5 minutes the spec expected. Standalone mode fits every precision from scratch and is about 140 times slower, so it is the stage to optimise if it stays in later waves.

The quantize rows were measured on a rerun of the incremental pipeline into a throwaway directory, because the logging of peak memory landed after the acceptance quantize runs. The standalone figure is from fitting one module of each large shape, [1024, 2048] and [4096, 1024], not from the whole stage.

## Answer quality

The reference is the float32 model. Its WikiText-2 perplexity, 18.579, is the anchor for every later comparison; its C4 perplexity is 25.993. KL and top-1 agreement are measured on WikiText-2.

| Bits | IU mean KL | SA mean KL | IU top-1 | SA top-1 | IU ppl WikiText-2 | SA ppl WikiText-2 | IU ppl C4 | SA ppl C4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 8.395e-1 | 8.395e-1 | 0.5583 | 0.5583 | 38.912 | 38.912 | 54.328 | 54.328 |
| 4 | 1.921e-1 | 1.748e-1 | 0.7764 | 0.7850 | 21.616 | 21.020 | 29.523 | 29.004 |
| 5 | 4.660e-2 | 3.683e-2 | 0.8839 | 0.8959 | 19.307 | 19.028 | 26.691 | 26.602 |
| 6 | 1.068e-2 | 7.652e-3 | 0.9425 | 0.9515 | 18.772 | 18.724 | 26.145 | 26.129 |
| 7 | 2.273e-3 | 1.384e-3 | 0.9733 | 0.9792 | 18.606 | 18.576 | 26.011 | 26.000 |
| 8 | 4.442e-4 | 2.186e-4 | 0.9883 | 0.9917 | 18.575 | 18.571 | 25.995 | 25.996 |

IU is incremental upscaling and SA is standalone. At 3 bits the two modes are identical, as they must be, because both start from the same seed model.

Nesting costs the most answer quality at 4 and 5 bits, as notebook 02 predicted. The incremental KL is 1.10 times the standalone KL at 4 bits and 1.27 times at 5 bits. In perplexity the gap is 0.60 at 4 bits and 0.28 at 5 bits on WikiText-2. From 6 bits upwards the perplexity gap is below 0.05, although the KL ratio keeps growing to 2.0 at 8 bits.

## Per-module relative error

These observations come from the saved error statistics of all 168 modules in both artifacts.

Nesting's cost per module grows with precision. The incremental-to-standalone error ratio has a median of 1.151 at 4 bits (maximum 1.627) and 1.287 at 5 bits (maximum 2.185). At 8 bits it ranges from 1.36 to 4.65, with a median of 2.62 and a 10th to 90th percentile range of 1.57 to 3.53. That range is wider than the 1.6 to 3.5 that notebook 02 saw on two modules, but the middle 80% matches it.

The largest 3-bit relative errors, about 0.053 to 0.057, are all `self_attn.v_proj` modules, in layers 24, 25, 16, 23, 17, and 7. They do not coincide with the extreme-crest `shared_mlp.output_linear` rows that the first notebook found. `v_proj` has the largest share of heavy-tailed rows (6.5%), which is the more likely cause.
