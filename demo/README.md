# Bringing the Heat — executable demo

One coherent path: **Hub + Datasets → Transformers → PEFT LoRA + Accelerate → Optimum ONNX INT8 → measured release gate → local Gradio UI**.

The classifier routes AG News text to World, Sports, Business, or Sci/Tech. It is deliberately small enough to explain to students and concrete enough to discuss with ML engineers. Muntaser's custom MultiSpecQR assets are a separate Hub showcase: this transformer LoRA recipe does not pretend to fit that CNN architecture.

## Run the full exercise in Google Colab

Open **`Bringing_the_Heat_Colab.ipynb`**, choose **Runtime → Change runtime type → T4 GPU** when available, then **Runtime → Run all**. CPU also works but full training takes longer. The notebook performs the complete pipeline: dependency installation, pinned public downloads, three-epoch training, export/quantization, evaluation, CPU benchmark, real and deliberately failing gates, inference, and a downloadable artifact/evidence bundle. There are no replayed metrics or optional training skips.

**Complete visual notebook tested on hosted Colab:** all 11 code cells completed on a T4/Python 3.13 runtime with zero errors and **ten inline figures**. The run reached its completion marker in **15 minutes 48 seconds**, followed by ZIP compression. The real gate passed and the deliberate regression correctly blocked. Full CPU evaluation was the longest stage. Pre-run before presenting; `Bringing_the_Heat_Colab_executed.ipynb` contains the latest verified outputs, and `COLAB_TESTED.md` records the environment and measurements. Its code exactly matches the clean, unexecuted notebook provided for a new run.

The visual notebook includes a workflow diagram, data-split provenance, a LoRA path and measured trainable-parameter budget, epoch loss/validation curves, export footprint, a confusion matrix with class-recall floors, latency distributions and p50/p95, quality-versus-size comparisons, a release-gate matrix, and inference score bars. Numeric plots read the current run's JSON reports. `visuals.py` is embedded in the notebook; Matplotlib is pinned in its isolated environment. The final bundle includes all ten figures as both PNG and SVG, so they can be reused in slides or handouts.

The notebook embeds the exact source and Hub revisions, so it runs without cloning this repository or mounting Drive. No Hugging Face token is required. It creates a notebook-owned virtual environment, installs `requirements-colab.txt`, and reuses Colab's installed PyTorch/CUDA build without changing the notebook kernel or requiring a restart. Source hashes, runtime package versions and hardware are recorded.

Colab uses `colab_policy.json`: the same quality and footprint checks, **two CPU benchmark threads**, and an illustrative **250 ms p95** teaching budget. That policy is frozen before training/evaluation. The laptop's four-thread/75 ms policy and measured results remain separate. A real gate BLOCK is a successful execution outcome; unexpected errors still stop the notebook. Never lower thresholds after looking at the heldout test. Rebenchmark on your deployment hardware before making a production release decision.

Inference is an editable notebook cell using the exact packaged INT8 model. No public Gradio tunnel is created. The final ZIP includes the INT8 model/tokenizer, adapter, source, policy and JSON evidence; use the Colab Files sidebar to download it before the runtime expires. Colab hosts the exercise, not a persistent production service. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html) for resource and sharing behavior.

Run `python make_colab_notebook.py` after editing `demo.py`, `visuals.py`, Hub revisions, Colab dependencies or policy. It regenerates the self-contained notebook without outputs. `COLAB_TESTED.md` records the scope of validation separately from the laptop rehearsal in `TESTED.md`.

## Setup (Windows / PowerShell)

From this directory, with [uv installed](https://docs.astral.sh/uv/getting-started/installation/):

```powershell
./setup.ps1          # CPU path
# OR ./setup.ps1 -Cuda  # NVIDIA CUDA 12.8 wheel, ~3.2 GiB download
.venv/Scripts/python.exe demo.py prepare
.venv/Scripts/python.exe demo.py preflight
```

Python 3.12 is required. The dependency locks record the tested Windows environment. CPU and CUDA torch wheels have separate lock files. On Linux/macOS, use a Python 3.12 environment, install `requirements.lock`, then the appropriate PyTorch wheel for that platform; these platforms are not rehearsed here. There is no requirement for an HF login: the script explicitly downloads public data/model artifacts without a token. No upload, paid endpoint, or cloud resource is created.

`prepare` downloads the exact commits in `hub_revisions.json` and writes `artifacts/manifest.json`. It preserves an existing prepared manifest. The public base weights are approximately 256 MiB. Package installation and first downloads belong in preparation, not on stage. Allow several GiB for dependencies, data and the exported artifacts.

## Rehearse the real run

```powershell
./rehearse.ps1       # Uses the GPU if available; -Cpu forces CPU
```

The underlying commands are intentionally simple:

```powershell
.venv/Scripts/python.exe demo.py train       # 3 epochs; validation chooses checkpoint
.venv/Scripts/python.exe demo.py export      # merges adapter, exports FP32, dynamic INT8 AVX2
.venv/Scripts/python.exe demo.py evaluate    # identical locked test set for all 3 backends
.venv/Scripts/python.exe demo.py benchmark   # CPU 4 threads, batch=1, fixed 128 tokens
.venv/Scripts/python.exe demo.py gate
.venv/Scripts/python.exe demo.py gate --inject-failure # Expected exit 2
.venv/Scripts/python.exe app.py              # http://127.0.0.1:7860
```

`Bringing_the_Heat.ipynb` is the talk driver. Start it with `.venv/Scripts/python.exe -m jupyterlab Bringing_the_Heat.ipynb`. Its ordinary cells replay measurements and run inference against prepared artifacts. A guarded optional cell runs five real training steps with `train --smoke` into a separate `smoke_adapter` so it cannot replace the rehearsed candidate.

On a Linux multi-GPU training host, the same training loop can be launched with:

```bash
accelerate launch --multi_gpu --num_processes 2 demo.py train --batch-size 16
```

This keeps effective batch size 32 across two GPUs. Only single-device training has been rehearsed. DDP replicates this model; it is not model sharding or an inference server. Bigger models may require FSDP/DeepSpeed and a separate deployment architecture.

## What makes the measurements trustworthy

- Fixed seed 42 and pinned Hub revisions; raw split checksums recorded before training.
- Balanced 3,200 train / 400 validation / 800 test examples. Validation is drawn from the original training split; final test is drawn from the original test split. Normalized exact-text duplicates are excluded across these subsets. Near duplicates and source/time leakage need more work in a real dataset.
- LoRA updates DistilBERT's `q_lin` and `v_lin` with rank 8 and trains the classification/pre-classification heads. The exported adapter is small; using it still requires the base model. Merging produces a standalone dense model.
- Highest validation macro F1 chooses the checkpoint; test is for the final report. Do not repeatedly tune on that locked test set.
- Accuracy, macro F1, per-class precision/recall/F1/support, confusion matrix, and a 95% Wilson accuracy interval. The interval describes sampling uncertainty on this small sample, not all possible distribution shift.
- The same heldout examples evaluate PyTorch FP32, ONNX FP32 and ONNX INT8. Quantization can trade quality for footprint/latency; gain is measured, never assumed.
- Each backend uses the tokenizer packaged with its own runtime artifact. Evaluation and benchmark reports contain fingerprints for weights, tokenizer, config, label order, and the 128-token truncation contract. The gate compares them with current files and verifies the current heldout checksum, so stale reports cannot approve a replaced artifact.
- Latency includes tokenization plus model forward after 10 warmups, on the same CPU, four threads, batch 1, fixed padded length 128, 100 sequential requests. Raw samples, p50, p95 and sequential throughput are saved. Loading/network/queueing are excluded. This is not a concurrent server capacity test.
- Frozen teaching gate: accuracy and macro F1 ≥ .80, every-class recall ≥ .65, F1 drop ≤ .02, CPU p95 ≤ 75ms, weight files ≤ 100MiB. These are demonstration budgets, not externally validated product requirements. `gate --inject-failure` simulates Business recall dropping to .10 while leaving real result files unchanged.
- The measurement contract also requires all four class reports and at least 100 raw latency samples with the stated warmup, CPU, thread, sequence-length and batch settings. A one-request timing cannot pass this gate.

`results/*.json` is the source of truth for measured claims. `TESTED.md` states what was actually executed. Files in `artifacts/` and the virtual environment are local preparation outputs and are intentionally excluded from the handout ZIP.

## Deployment discussion

The local Gradio app binds only to localhost, does not share publicly, and is a preview UI. A production rollout still needs a real service/container, input limits, authentication, load tests, observability, canary/rollback, drift checks, and ownership. The gate is one component in that system.

`export --target arm64` is the analogous recipe for a target with ARM64, but this project has only measured AVX2 on the rehearsal laptop. An INT8 model running on a laptop CPU is a concrete constrained-device example, not evidence about every edge device.

## On-stage fallback

Set `$env:HF_HUB_OFFLINE='1'` and `$env:HF_DATASETS_OFFLINE='1'` after preparation. `preflight`, training, exports, inference and results inspection use local inputs. Skip any long-running cell during the talk and open the existing JSON results. If the browser fails, the notebook's `classify` cell uses the exact same local model. If there is no GPU, use the prepared adapter; do not spend talk time retraining on CPU.

## Official API references

- [Accelerate quicktour and distributed evaluation](https://huggingface.co/docs/accelerate/quicktour)
- [PEFT LoRA and merging](https://huggingface.co/docs/peft/developer_guides/lora)
- [Optimum ONNX installation](https://huggingface.co/docs/optimum-onnx/installation)
- [Optimum ONNX dynamic quantization](https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/quantization)
- [AG News dataset card](https://huggingface.co/datasets/fancyzhx/ag_news)
- [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased)

Review source model/data licenses and fitness for your own use. Demo outputs are not a claim that this small news classifier is ready for a high-stakes application.
