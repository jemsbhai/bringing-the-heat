# Colab notebook validation

`Bringing_the_Heat_Colab.ipynb` is the complete fresh-runtime exercise. `Bringing_the_Heat.ipynb` is the original local stage driver with saved laptop measurements.

The Colab notebook schema validates and every code cell compiles. Seven meaningful release-gate tests pass, including the configurable CPU thread contract. An independent clean Linux execution is being performed; actual Google Colab runtime verification is tracked separately. This file will be updated with completed evidence, not inferred platform compatibility.

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

Do not describe a Linux rehearsal as an executed Colab run. The notebook captures its actual environment so those claims can be distinguished.
