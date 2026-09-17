# Rehearsal evidence — 17 September 2026

The full local workflow was actually executed. No timings or quality numbers below are estimates.

## Hardware and environment

- Windows 11; Intel Core i9-14900HX (24 cores / 32 logical processors).
- Training: NVIDIA GeForce RTX 4090 Laptop GPU, 16 GiB. Evaluation/benchmark and local app: CPU.
- Isolated Python 3.12.12 environment; torch 2.8.0+cu128, Transformers 4.55.4, PEFT 0.17.1, Accelerate 1.10.1, Datasets 4.0.0, Optimum ONNX 0.0.3, ONNX Runtime 1.30.0, Gradio 5.50.0.
- Full dependency pins are in `requirements.lock` plus the selected CPU/CUDA lock. The CUDA lock dry-run resolves to the installed environment without changes. The CPU lock resolution was checked; the CPU-only wheel was not installed for the final rehearsal. CPU execution was tested using the CUDA-capable torch build with computation on CPU.

## Executed successfully

1. Public downloads pinned to `hub_revisions.json`; no login needed and nothing uploaded.
2. Prepared 3,200 train / 400 validation / 800 test examples with balanced class counts; normalized exact-text deduplication across all subsets; checksum and disjointness preflight passed.
3. Full LoRA training: three epochs, rank 8, alpha 16, dropout 0.1, `q_lin`/`v_lin`, batch 32, learning rate 3e-4, seed 42. **741,124 trainable / 67,697,672 total parameters (1.095%)**. Best validation macro F1 .9074 at epoch 3. Training-loop time 13.55 seconds excludes imports, initial model/data loading and tokenization.
4. Adapter merge, standalone tokenizer/model save, ONNX FP32 export, dynamic INT8 AVX2 quantization. Adapter weight file: 2,968,352 bytes (~2.83 MiB); using the unmerged adapter also requires base weights.
5. Heldout evaluation of all three backends with their own packaged tokenizer; each has the same 800 examples. Majority-class baseline accuracy .25 / macro F1 .10.
6. CPU benchmark with **4 intra-op threads, batch 1, fixed padded sequence length 128, 10 warmups and 100 sequential measured requests**. Includes tokenization + model forward. Excludes model loading, network and queueing. Weight size is disk bytes, not memory use. Raw latency samples are included in `results/benchmark.json`.
7. Real release gate PASS; deliberate Business-recall regression (.10) correctly FAILS with exit 2. All six policy tests passed, including stale-artifact rejection, incomplete class reports and insufficient measurement samples.
8. **All seven executable notebook cells ran successfully**, with saved outputs and no cell errors. The notebook's optional training-smoke switch remained false.
9. Separately ran `train --cpu --smoke` offline: five training steps and validation completed, writing only `smoke_adapter` and `results/smoke_training.json`. Its .5946 validation macro F1 is a smoke-test result, not the released candidate.
10. Local Gradio HTTP page returned 200 and its prediction API correctly returned Sports for the supplied Miami-team example; server closed after this test. See `results/app_smoke.json`.

## Final measured results

| Backend | Accuracy | Macro F1 | CPU p50 (ms) | CPU p95 (ms) | Weight files (MiB) |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | .9025 | .902308 | 33.48 | 59.81 | 255.43 |
| ONNX FP32 | .9025 | .902308 | 30.33 | 41.56 | 255.53 |
| ONNX INT8 | .9050 | .904809 | 25.82 | 33.57 | 64.25 |

For INT8, the 95% Wilson accuracy interval is **[.8827, .9234]**. Per-class recall: World .865, Sports .980, Business .900, Sci/Tech .875. The .25-point accuracy difference between FP32 and INT8 is small and not evidence that quantization generally improves accuracy.

On this run, INT8 p95 was about **1.78× faster** than PyTorch FP32 and its weight files about **74.8% smaller**. These are results for this laptop, inputs and runtime settings; they are not universal gains or multi-user service capacity numbers.

## Provenance and limits

Quality/footprint/latency thresholds were frozen before the heldout test. During code review, report-to-artifact binding and measurement-protocol validation were strengthened; evaluation and benchmark were then rerun once. The table above uses that final run. Evaluation and benchmark now record hashes covering each packaged model, tokenizer, config, label order and truncation contract. The gate checks those against current files and checks the current heldout checksum, preventing mutually stale reports from authorizing a changed runtime.

The final source of truth is `results/evaluation.json`, `results/benchmark.json`, `results/gate.json`, and `results/gate_deliberate_failure.json`. `results/export.json` records exact exported-file sizes and hashes. `results/training.json` records the full GPU training run; the CPU smoke report is separate.

Not executed: multi-GPU training, FSDP/DeepSpeed, ARM64 quantization, edge-device deployment, vLLM/LLM serving, cloud endpoints, concurrent load testing, public Spaces deployment. Related commands are extension recipes. The Gradio UI is a local preview; this teaching release gate does not establish production readiness.

The handout ZIP excludes the virtual environment, downloads and weight files. Recreate them with `setup.ps1` and `rehearse.ps1` before using the notebook on another machine. Existing local artifacts on the presenter's laptop are ready to rehearse offline.
