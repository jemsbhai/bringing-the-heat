# Hugging Face facts and source notes

Verified against official documentation on 2026-09-17. These are capability notes, not measured results. Recheck package versions, model compatibility, pricing, and hosting eligibility before rehearsal.

## The central story

The Hub connects reusable artifacts. A practical engineering workflow is **version the data and model → establish a baseline → adapt efficiently → evaluate → optimize for the target → deploy and observe**. This is the talk's proposed framework, not a Hugging Face product name.

Keep the live example coherent: a small classifier can demonstrate PEFT, Accelerate, evaluation, ONNX export, and CPU quantization. Explain LLM serving as a second deployment branch. A classifier running on a laptop illustrates constrained CPU deployment; it does not prove operation on a phone, microcontroller, or multi-GPU cluster.

## What the three advertised libraries actually do

| Library | Accurate stage description | Concrete mechanism | Avoid saying |
|---|---|---|---|
| PEFT | Adapt a pretrained model by training a small added parameter set. | `LoraConfig` + `get_peft_model`; LoRA freezes base weights and learns low-rank updates. | PEFT removes the base model, guarantees full-fine-tuning quality, or always makes inference faster. |
| Accelerate | Adapt and launch PyTorch training across hardware/distributed setups. | `Accelerator`, `prepare`, `backward`, `accelerate config`, `accelerate launch`; integrations include FSDP and DeepSpeed. | Accelerate automatically turns a notebook into a production API, or uses all GPUs efficiently without an appropriate strategy. |
| Optimum | Hardware-oriented optimization integrations. The ONNX path is now the separate Optimum ONNX distribution. | Export supported architectures to ONNX; run graph optimization and quantization through ONNX Runtime. | Optimum itself is the LoRA trainer, supports every model export, or guarantees a speedup. |

Sources: [PEFT quicktour](https://huggingface.co/docs/peft/quicktour), [LoRA concepts](https://huggingface.co/docs/peft/main/conceptual_guides/lora), [Accelerate quicktour](https://huggingface.co/docs/accelerate/quicktour), [Optimum](https://huggingface.co/docs/optimum/index), [Optimum ONNX](https://huggingface.co/docs/optimum-onnx/index).

PEFT's saved adapter contains the additional trained weights and configuration, not a standalone copy of the base model. Record the base model revision with the adapter. Supported LoRA adapters can be merged with `merge_and_unload()` for standalone inference; confirm the model/quantization combination supports the intended merge. QLoRA combines low-bit base-model loading with adapter training; it is distinct from post-training ONNX INT8 quantization. Sources: [PEFT saving](https://huggingface.co/docs/peft/quicktour#save-model), [LoRA merging](https://huggingface.co/docs/peft/main/conceptual_guides/lora), [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization).

Accelerate's `gather_for_metrics` gathers predictions/labels across workers and handles duplicate end-of-dataset items. Compute metrics over the gathered examples, not just one worker. Its Big Model Inference dispatch/offload path with `device_map="auto"` is documented for inference, not distributed training; loading across devices is not equivalent to high-throughput tensor-parallel serving. Sources: [Distributed evaluation](https://huggingface.co/docs/accelerate/quicktour#distributed-evaluation), [Big Model Inference](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference).

## Current ONNX packaging and API

Install from the current dedicated distribution:

```text
pip install "optimum-onnx[onnxruntime]"
```

The imports still use `optimum.onnxruntime`. The documented sequence is:

1. Export a supported Transformers model with `ORTModelForSequenceClassification.from_pretrained(path_or_id, export=True)`.
2. Save the ONNX model and tokenizer.
3. Construct `ORTQuantizer.from_pretrained(ort_model)` or use its saved directory.
4. Choose an appropriate `AutoQuantizationConfig` for the target hardware, such as `avx2`, `avx512_vnni`, or `arm64`, with `is_static=False` for dynamic quantization.
5. Call `quantize(save_dir=..., quantization_config=...)`.
6. Load the result using `ORTModelForSequenceClassification.from_pretrained(directory, file_name="model_quantized.onnx")`.

Static quantization additionally needs representative calibration data and computed activation ranges. Use training/calibration data, not the held-out test set. The docs note limitations for multi-file sequence-to-sequence models, so keep the live export to an encoder classifier. Sources: [Installation](https://huggingface.co/docs/optimum-onnx/installation), [Runtime quickstart](https://huggingface.co/docs/optimum-onnx/onnxruntime/quickstart), [Quantization guide](https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/quantization).

Recommended demo practice: merge the classifier's LoRA adapter into its base first, export that merged artifact, and compare base/adapted/exported/quantized predictions on the same fixed held-out examples. Report hardware, threads, batch size, sequence lengths, warm-up, sample count, and latency measurement scope. A smaller file alone is not proof of lower latency. These are engineering recommendations.

## Deployment branches

| Target | Suitable ecosystem path | Practical boundary |
|---|---|---|
| Laptop or constrained CPU | Optimum ONNX + ONNX Runtime; model/quantization config matched to CPU. | Measure on the actual target. The laptop demo is not evidence for every edge device. |
| Browser | Transformers.js using ONNX Runtime; WASM CPU by default, WebGPU when selected and available. | Use a supported exported architecture and available dtype; browser/device support and download size matter. |
| Local LLM runtime | Hub artifacts consumed by llama.cpp or MLX, when the model/format is supported. | GGUF/MLX artifacts are different export/runtime paths from the classifier ONNX demo. |
| Self-managed LLM GPU service | vLLM or SGLang with compatible models. | Tensor parallelism helps fit one model across GPUs; replicas/data parallelism serve more independent requests. |
| Managed dedicated service | Hugging Face Inference Endpoints. | Offers autoscaling, logs/metrics, and supported engines/custom containers; choose hardware and cost limits. |

Sources: [Transformers.js](https://huggingface.co/docs/transformers.js/index), [vLLM parallelism](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/), [Endpoints](https://huggingface.co/docs/inference-endpoints/en/index), [vLLM on Endpoints](https://huggingface.co/docs/inference-endpoints/en/engines/vllm).

**Important current change:** TGI is in maintenance mode. Its official documentation recommends vLLM, SGLang, and local engines such as llama.cpp or MLX going forward. TGI may still appear in Endpoints and older examples, but do not present it as the preferred new deployment route. Source: [TGI status](https://huggingface.co/docs/text-generation-inference/en/index).

For multi-GPU slides, distinguish three questions: Does the model fit? How many requests can run concurrently? What latency is acceptable? Weight memory alone omits KV cache, activations, and runtime overhead. Avoid copying absolute memory/throughput guarantees from example tables.

## Five quick ecosystem extras worth showing

1. **Reproducible artifacts:** `snapshot_download(revision=<full commit SHA>)` pins a repository snapshot; model cards document intended use, training data, limitations, and evaluation. Show an actual model card and commit rather than a wall of logos. Sources: [Download revisions](https://huggingface.co/docs/huggingface_hub/guides/download), [Model cards](https://huggingface.co/docs/hub/model-cards).
2. **Datasets streaming:** `load_dataset(..., streaming=True)` starts consuming an iterable without downloading the full dataset first. Buffer-based shuffling is approximate; the buffer size and shard order affect the sequence. Source: [Streaming guide](https://huggingface.co/docs/datasets/stream).
3. **Spaces:** a versioned repository can become a Gradio, Docker, or static HTML demo, with rebuilds after commits. Current docs say creating Gradio/Docker compute Spaces requires a paid plan; static Spaces remain free, with a stated exception for eligible free accounts to host up to two ZeroGPU Gradio Spaces. Verify eligibility in the user's account before promising free hosting. Source: [Spaces overview](https://huggingface.co/docs/hub/spaces-overview).
4. **Jobs:** run scripts or containers on hosted compute for training, data processing, batch inference, or evaluation. Current docs require a positive credit balance and describe a default 30-minute timeout. Treat as an optional prepared command, not an unapproved paid live launch. Sources: [Jobs](https://huggingface.co/docs/hub/jobs), [Jobs client guide](https://huggingface.co/docs/huggingface_hub/en/guides/jobs).
5. **Inference Providers:** one client can route supported model requests across providers, with unified authentication/billing options. This is a fast way to compare hosted models before choosing a dedicated deployment; it does not host every arbitrary Hub checkpoint automatically. Source: [Inference Providers](https://huggingface.co/docs/inference-providers/index).

## Evaluation: credible claims for the talk

For the classifier demo, show macro-F1, per-class precision/recall, a confusion matrix, and several failures. Establish a held-out test split, choose thresholds before viewing final test results, and compare the exact candidate artifact after export/quantization. Label tiny or synthetic demo sets clearly. These are proposed engineering practices, not guarantees supplied by a library.

For LLMs, task-appropriate metrics and fixed prompts/generation settings matter. LightEval supports configurable tasks, metrics, multiple backends, and saved results. The current Evaluate landing page explicitly points to LightEval as more actively maintained for recent approaches; Evaluate still documents reusable general ML metrics. Do not call Evaluate removed or universally deprecated. Sources: [LightEval](https://huggingface.co/docs/lighteval/main/index), [Metric list](https://github.com/huggingface/lighteval/blob/main/docs/source/metric-list.mdx), [Evaluate status and scope](https://huggingface.co/docs/evaluate/index).

Suggested talk line: **"The metric does not approve the model. Your release criteria do."** A result becomes actionable when paired with the dataset revision, evaluation implementation, acceptance threshold, target hardware, and an accountable release decision.

## Final accuracy checklist

- Distinguish live execution, previously measured results, and illustrative architecture diagrams.
- Do not invent a benchmark, cost saving, accuracy gain, or deployment result.
- Pin the tested environment; current documentation can target a main branch rather than the installed release.
- The speaker's custom models demonstrate real work, but do not imply every custom CNN supports Transformers/PEFT/Optimum wrappers automatically.
- Spend most of the 30 minutes following one artifact through the workflow; use the ecosystem extras as a short tour or appendix.
