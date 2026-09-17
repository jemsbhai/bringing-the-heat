# Colab notebook validation

`Bringing_the_Heat_Colab.ipynb` is the complete fresh-runtime exercise. `Bringing_the_Heat.ipynb` is the original local stage driver with saved laptop measurements.

The complete notebook was executed successfully in an actual Google Colab hosted T4 runtime. All 11 code cells completed in order with zero cell errors, including full training, exported-artifact evaluation, inference and bundle creation. Seven meaningful release-gate tests also pass, including the configurable CPU thread contract.

## Actual Google Colab run — completed 17 September 2026

- [Open the hosted notebook](https://colab.research.google.com/drive/1Q9WkLenTKLEd-R0wTzsCgF1mrLS1NCIh).
- Python **3.13.15**, PyTorch **2.11.0+cu128**, **Tesla T4** GPU; Linux VM with **two logical CPUs**. GPU trains; CPU evaluates and benchmarks.
- Pinned teaching packages: Transformers 4.55.4, PEFT 0.17.1, Accelerate 1.10.1, Datasets 4.0.0, Optimum ONNX 0.0.3, ONNX Runtime 1.30.0; compatible optional Diffusers 0.35.1 pinned explicitly.
- Fresh run began **16:02:31 UTC**, completion marker **16:17:52 UTC**: **15 minutes 21 seconds**, followed by about five seconds for ZIP compression. Resource contention and future runtimes can change timings.
- The downloaded executed copy, `Bringing_the_Heat_Colab_executed.ipynb`, has execution counts 1–11 and zero errors. Its code is byte-for-byte identical to `Bringing_the_Heat_Colab.ipynb`; combined code SHA256 is `54c59fde8bb281eb357ae0bf332f3cd3a75bb58dc38eec488b95e31858d61e8c`.
- Exact JSON reports, raw latency samples, source/artifact hashes and the input manifest were recovered from the runtime-generated ZIP into **`results/colab_validation.json`**. The bundle's completion record exactly matches the saved notebook output.

The public downloads, split checksums, full three-epoch LoRA training, adapter merge, FP32 ONNX export, dynamic INT8 quantization, all three heldout evaluations and all three CPU benchmarks completed. The **real gate passed**, the **deliberate Business-recall regression failed with the expected exit 2**, and the four inference examples correctly returned Sports, Business, Sci/Tech and World.

Training command time was **69.4 seconds**. Epoch 3 was selected using validation macro F1 **0.909692**. The slowest stage was CPU evaluation of all 800 examples across three backends: **683.3 seconds**. Export took 27.6 seconds and the benchmark 55.6 seconds. For a 30-minute presentation, **run the notebook before the talk and present the saved outputs**. The full Run all exercise is suitable for attendees to reproduce afterward.

| Backend | Test accuracy | Test macro F1 | CPU p50 (ms) | CPU p95 (ms) | Weight files (MiB) |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | 0.89500 | 0.894733 | 124.61 | 251.38 | 255.43 |
| ONNX FP32 | 0.89500 | 0.894733 | 104.33 | 223.92 | 255.53 |
| ONNX INT8 | 0.89875 | 0.898576 | 103.66 | 204.86 | 64.25 |

Benchmark: two CPU threads, batch 1, 128 padded tokens, ten warmups and 100 sequential measured requests; tokenization and forward included, loading/network/queueing excluded. Table values are rounded; the JSON contains full precision and raw samples. INT8 accuracy's 95% Wilson interval is **[0.875904, 0.917785]**. Per-class recall is World **0.855**, Sports **0.985**, Business **0.890**, Sci/Tech **0.865**. These Colab measurements are separate from the Windows laptop measurements shown in the talk and from the supplementary Linux CPU run below.

## Supplementary CPU fallback — completed 17 September 2026

All 11 executable notebook cells completed under Ubuntu/WSL, Python 3.12.3, PyTorch 2.11.0+cpu, using the presenter's laptop CPU. Full three-epoch training took 652.5 seconds including command startup; export 11.0 seconds, evaluation 72.6 seconds and the benchmark 25.3 seconds. The selected checkpoint was epoch 1, validation macro F1 0.906663.

| Backend | Test accuracy | Test macro F1 | CPU p95 (ms) |
|---|---:|---:|---:|
| PyTorch FP32 | 0.88125 | 0.880663 | 82.34 |
| ONNX FP32 | 0.88125 | 0.880663 | 67.03 |
| ONNX INT8 | 0.88625 | 0.885867 | 44.35 |

The real teaching gate passed; the deliberate Business-recall regression blocked correctly; all four supplied inference examples returned their intended classes; the downloadable bundle was created. Full structured evidence is in `results/linux_cpu_validation.json`, including raw timing samples and artifact identities. This is separate from the earlier Windows GPU rehearsal.

This supplementary run began before the final Diffusers dependency correction. `diffusers==0.35.1` was installed and its Optimum imports checked before export. It verifies CPU training and the downstream pipeline; the hosted Colab run verifies the final fresh-runtime notebook including that dependency pin. The initial hosted Colab attempt exposed an incompatible preinstalled Diffusers version, which is why the final Colab requirements explicitly pin this otherwise optional dependency.

## Design

- Self-contained source, source hashes and pinned public Hub revisions; no repository bootstrap or Drive mount.
- Pinned Hugging Face / ONNX teaching stack in `requirements-colab.txt`; runtime PyTorch reused and constrained to its installed version.
- Full three-epoch training, merge/export/INT8, all three heldout evaluations, 100-sample CPU benchmarks, gate, deliberate failure and inference.
- New run directory for each Run all. Reports and model exports never overwrite the laptop evidence.
- Frozen Colab teaching policy: two benchmark threads, 250 ms CPU p95 illustration, 100 MiB weights, quality thresholds unchanged from the laptop exercise.
- A blocked real gate is handled explicitly; other command failures stop execution.
- Completion marker: `results/run_complete.json` with `ALL_PIPELINE_STAGES_COMPLETED`, environment, gate outcome and source hashes.

## Official runtime references

- [Colab FAQ](https://research.google.com/colaboratory/faq.html): notebooks can be loaded from GitHub; shared notebooks need their own installation/loading cells; runtime files are ephemeral.
- [Colab backend information](https://github.com/googlecolab/backend-info): runtime versions change. At preparation, the 2026.07 runtime is documented as Python 3.12.13 / PyTorch 2.11.0; current package snapshots may be newer and have a rollout delay.

The hosted Colab run, supplementary Linux CPU run, and original Windows rehearsal each capture their actual environments; their metrics and timings must not be combined as one experiment.
