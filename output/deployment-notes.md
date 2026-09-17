# Deployment extensions: from one machine to the target

**Status: every recipe in this file is UNEXECUTED.** These are companion exercises, not benchmark results or evidence of a deployed service. The main talk follows a DistilBERT news classifier; the causal-LLM examples below are a separate model and task. Official documentation checked 2026-09-17.

## 1. Multi-GPU classifier training with Accelerate

Use a Linux machine with two compatible GPUs, a matching PyTorch/CUDA environment, and the demo's tested dependency versions. Start from a separate copy of `demo` so a new run cannot replace the rehearsal adapter. Prepare the dataset/model once, then verify the manifest before launching workers.

From that copy's parent directory, the intended commands are:

```bash
python demo/demo.py prepare
python demo/demo.py preflight
accelerate launch --multi_gpu --num_processes 2 --mixed_precision no \
  demo/demo.py train --epochs 3 --batch-size 16
```

This requests distributed data parallel training: each GPU holds a model copy and processes different examples. The effective batch is 2 × 16 = 32 examples per optimizer step, matching the default single-process batch of 32. Data order and numerical results can still differ.

The existing script explicitly selects `mixed_precision="no"`; changing the launcher flag alone does not enable mixed precision. Its prepared validation loader and `gather_for_metrics` combine workers' predictions. Only the main process writes the adapter/report. Check the recorded `world_size`, sample counts, validation result, and saved artifact after the run.

DDP helps distribute work when the model fits per GPU. For a model that does not fit, configure and test FSDP or DeepSpeed separately, including PEFT wrapping and checkpoint save/reload. This recipe has not validated those paths. `device_map="auto"` is an inference dispatch/offload mechanism, not a replacement for distributed training.

Sources: [Accelerate quicktour](https://huggingface.co/docs/accelerate/quicktour), [Big Model Inference](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference).

## 2. Causal-LLM serving with vLLM

Example task: short instruction-following responses with [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). This is a teaching example, not a claim that it is the newest or best model. It does not load the classifier adapter.

Use a separate Linux serving environment and install a vLLM release compatible with the GPU/driver stack. Pin that version or container digest after a smoke test. Record the model's full Hub commit SHA; the placeholder below must be replaced before execution.

```bash
export LLM_REVISION="REPLACE_WITH_FULL_COMMIT_SHA"
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --revision "$LLM_REVISION" \
  --tokenizer-revision "$LLM_REVISION" \
  --served-model-name talk-llm \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --dtype auto --max-model-len 4096 \
  --gpu-memory-utilization 0.80
```

The flags above are present in the current official `vllm serve` reference. The context limit covers **prompt plus generated output**. The GPU-memory fraction is a per-instance planning setting, not a guarantee that the workload fits or isolation from other GPU processes. Localhost binding makes this a local exercise.

After the server reports readiness, a separate terminal can send a bounded request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"talk-llm","messages":[{"role":"user","content":"Explain overfitting in two sentences."}],"max_tokens":96,"temperature":0}'
```

For a **two-GPU tensor-parallel exercise**, change `--tensor-parallel-size 1` to `2` on a host with two supported visible GPUs and compatible model dimensions. This small example may fit one GPU; using two may add overhead. For a genuinely larger model, choose the minimum supported parallelism that fits weights **and** the intended KV cache/workload, then benchmark it.

Tensor parallelism splits layer computation across devices. Separate replicas serve independent requests; batching changes utilization and latency. Multi-node serving additionally needs a configured distributed runtime, identical artifacts/environments, and a suitable interconnect. A two-GPU command does not validate a cluster.

Before sharing a service, test concurrency, p95 latency, time to first token, output tokens/second, memory, cancellation, and long inputs. Keep prompt/output lengths fixed when comparing runs. Add appropriate authentication, routing, request limits, and operational monitoring for the intended service.

Sources: [vLLM CLI](https://docs.vllm.ai/en/latest/cli/serve/), [Parallelism and scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/), [Online serving](https://docs.vllm.ai/en/latest/serving/online_serving/).

## 3. Conceptual QLoRA setup for that causal LLM

**Not a complete training script and not executed.** Dependencies are a compatible PyTorch build plus `transformers`, `peft`, `accelerate`, and `bitsandbytes`. Check the bitsandbytes hardware matrix and test the exact package combination in a separate environment; the classifier environment does not establish support for this extension.

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
revision = os.environ["LLM_REVISION"]  # Full, recorded commit SHA
assert torch.cuda.is_available(), "This conceptual recipe assumes a supported CUDA GPU"
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
base = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, quantization_config=quantization,
    device_map={"": 0},  # One GPU; not distributed training
)
base.config.use_cache = False
base = prepare_model_for_kbit_training(base)
model = get_peft_model(base, LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules="all-linear",
))
model.print_trainable_parameters()
```

The base uses low-bit storage while added adapter parameters train. This differs from the talk's post-training ONNX INT8 export. Remaining work: prepare licensed task data; split before tuning; apply the model's chat template; build correctly masked labels; set sequence/batch limits; train with a compatible loop or trainer; select on validation; evaluate the final candidate.

Freeze prompt formatting and generation settings for evaluation. Budget activations, optimizer state, and sequence length as well as weights. Do not infer that a given parameter count fits from “4-bit” alone. Saving an adapter still requires the matching base revision. Loading adapters, merging, and serving quantized models depend on the selected runtime; test the intended artifact path rather than assuming the vLLM command above serves this adapter.

Sources: [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), [bitsandbytes integration and hardware support](https://huggingface.co/docs/transformers/quantization/bitsandbytes), [LoRA concepts](https://huggingface.co/docs/peft/main/conceptual_guides/lora).

## 4. CPU, edge, and browser boundaries

| Target | Next experiment | What the talk's laptop run does not establish |
|---|---|---|
| x86 CPU | Measure exported classifier with an appropriate AVX configuration. | Performance on another CPU, thread count, or concurrency level. |
| ARM device | Export/quantize for the intended ARM runtime and test on-device. | Thermal behavior, battery use, memory headroom, or device compatibility. |
| Browser | Use Transformers.js and a supported ONNX model/precision. | That the classifier's Python export is automatically a complete JS package. |
| Phone / microcontroller | Choose a suitable architecture, runtime, and device-specific artifact. | That any Hub model fits, or that browser support implies native-device support. |

Transformers.js runs through ONNX Runtime: browser CPU execution uses WASM, while supported environments can use WebGPU. Verify tokenizer/config files, model architecture, dtype availability, download size, and browser support. Measure initial download/startup separately from warm inference. Never embed a private Hub token in public browser code.

Sources: [Transformers.js](https://huggingface.co/docs/transformers.js/index), [Optimum ONNX quantization](https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/quantization).

## 5. Dedicated Endpoint provisioning plan

**Planning only: no Endpoint has been created and no paid resource has been launched.** Complete this worksheet before provisioning:

1. **Artifact:** choose the exact repository and commit, task, tokenizer, and successful evaluation report. A custom CNN needs a compatible handler/container; upload alone does not make it deployable.
2. **Engine:** use an appropriate supported engine; vLLM is one causal-LLM option. Decide separately whether the classifier needs the toolkit or a custom container.
3. **Hardware:** record cloud/region, instance, precision, maximum context/input size, concurrency target, and measured memory requirements.
4. **Cost and scaling:** record current hourly price, minimum/maximum replicas, idle behavior, and an operating budget. Maximum replicas bounds one cost driver; it is not a universal hard spending cap.
5. **Access:** select the intended endpoint visibility and authentication, network requirements, secrets, and data-handling policy.
6. **Verification:** exercise startup, health, representative requests, load, timeouts, failure responses, logs, and quality checks before directing users to it.
7. **Release:** keep the previous immutable revision and a rollback procedure. Establish monitoring and pause/delete ownership for the end of the exercise.

Scale-to-zero can introduce a cold-start delay. Decide whether that tradeoff meets the service's response-time requirement. Confirm that a client handles the platform's documented startup behavior. A local Gradio UI demonstrates inference integration; it does not replace these service checks.

Sources: [Endpoint configuration](https://huggingface.co/docs/inference-endpoints/en/guides/configuration), [Autoscaling](https://huggingface.co/docs/inference-endpoints/en/guides/autoscaling), [vLLM on Endpoints](https://huggingface.co/docs/inference-endpoints/en/engines/vllm).

## 6. What to record after an extension is actually run

Keep the environment/container version, model/tokenizer/data revisions, hardware, launch arguments, effective batch or request shape, raw evaluation outputs, latency protocol, failures, and resulting artifact hashes. Update the status from **UNEXECUTED** only with an attached run record.

Current ecosystem note: [TGI is in maintenance mode](https://huggingface.co/docs/text-generation-inference/en/index); its documentation recommends vLLM/SGLang and local engines such as llama.cpp/MLX for new work. These are alternative runtime paths, not interchangeable file formats.
