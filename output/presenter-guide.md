# Presenter guide

**Bringing the Heat: Supercharging Your ML Pipelines with Hugging Face**

30 minutes plus 5 minutes for Q&A. Slides 1–17 are the talk, slide 18 is Q&A, and slides 19–22 are optional appendix. Audience: students and ML engineers.

## Run of show

| Slide | Time | Topic |
|---|---|---|
| 1 | 0:00–0:45 | Bringing the Heat: Supercharging Your ML Pipelines with HuggingFace |
| 2 | 0:45–2:00 | The production contract |
| 3 | 2:00–4:00 | My Hub: MultiSpecQR |
| 4 | 4:00–5:30 | A workflow through the ecosystem |
| 5 | 5:30–6:30 | The live build: a news classifier |
| 6 | 6:30–8:00 | Data and model provenance |
| 7 | 8:00–10:00 | LoRA changes the training budget |
| 8 | 10:00–12:00 | PEFT in the training script |
| 9 | 12:00–14:00 | Accelerate carries the training loop |
| 10 | 14:00–16:00 | Live checkpoint: the candidate artifact |
| 11 | 16:00–18:30 | Evaluation as a release gate |
| 12 | 18:30–20:30 | Optimum ONNX targets the CPU |
| 13 | 20:30–22:00 | Measured quality and CPU latency |
| 14 | 22:00–24:00 | Inference across deployment targets |
| 15 | 24:00–26:00 | A deployment you can operate |
| 16 | 26:00–28:00 | Five more ways to save work |
| 17 | 28:00–30:00 | The workflow to take home |
| 18 | 30:00–35:00 | Questions |
| 19 | Appendix | Appendix: QLoRA for a causal LLM |
| 20 | Appendix | Appendix: what a second GPU solves |
| 21 | Appendix | Appendix: evaluation follows the task |
| 22 | Appendix | Appendix: useful starting points |

## Speaker notes

### 1. Bringing the Heat: Supercharging Your ML Pipelines with HuggingFace (0:00–0:45)

OPEN (45 seconds). Ask for a quick show of hands: who has downloaded a model from the Hub, and who has had to support one after deployment? Pause briefly. Say: Today we will follow one model through a repeatable workflow. You will see the code, the evidence that could block a release, and the choices that change when the target hardware changes. Introduce yourself with the supplied bio: Muntaser Syed, Lead GenAI Engineer at Insight Global, formerly NVIDIA. Venue: Miami Dade College. Do not add employer performance claims.
Transition: A notebook prediction is the beginning of the story.

- <https://muntasersyed.com>
- <https://drive.google.com/file/d/1cWJ_CbI899vkuPVZTnfYJgcGdz8jv3t7/view>

### 2. The production contract (0:45–2:00)

75 seconds. Point out the repo QR: it opens the code, Colab notebook, slides, and recorded results. Give a concrete failure: the notebook looks good, but a new package version changes preprocessing, or one minority class collapses behind a high average accuracy. The engineering question is what evidence travels with the weights. Introduce the contract: source revisions, preprocessing, test results, runtime configuration, and rollback. This is our proposed engineering framework, not a promise provided by a library. Students should listen for the role of each tool. Engineers should listen for the boundary between training code and an operational service.
Transition: First, a real artifact you can inspect.

- <https://huggingface.co/docs/hub/model-cards>
- <https://huggingface.co/docs/huggingface_hub/guides/download>
- <https://github.com/jemsbhai/bringing-the-heat>

### 3. My Hub: MultiSpecQR (2:00–4:00)

2 minutes including browser switch. Open https://huggingface.co/Jemsbhai, then the RGB model card, Files, and commit history. Explain that color channels can carry different payloads and the custom CNN helps unmix the layers. Show one real dataset sample. Open the RGB dataset viewer and point to train/validation/test splits. The palette6 dataset currently exposes a train split only, a useful reminder that evaluation splits need deliberate design. The model cards describe the custom multispecqr library, not a Transformers architecture. The library’s from_pretrained helper is a familiar interface but does not imply automatic PEFT or Optimum support. Do not promise hosted inference from an arbitrary uploaded checkpoint.
If the venue network stalls, use showcase/README.md, hub_manifest.json, and the embedded dataset sample.
Transition: The same Hub conventions support a small classifier, a vision model, or a large language model.

- <https://huggingface.co/Jemsbhai>
- <https://huggingface.co/Jemsbhai/multispecqr-rgb>
- <https://huggingface.co/datasets/Jemsbhai/multispecqr-rgb-dataset>
- <https://github.com/jemsbhai/multispecqr>

### 4. A workflow through the ecosystem (4:00–5:30)

90 seconds. This is the only broad ecosystem map before the demo. Distinguish the three named libraries precisely: PEFT changes how many parameters we train; Accelerate handles training execution and distributed concerns; Optimum provides hardware-oriented integrations, with ONNX in a dedicated package. None independently supplies all of production. Evaluate supplies reusable ML metrics; LightEval is a current option for LLM evaluation. Our classifier uses scikit-learn for transparent metric calculations rather than pretending one metric package is mandatory.
Transition: Keep the problem small enough to inspect the whole path.

- <https://huggingface.co/docs/peft/quicktour>
- <https://huggingface.co/docs/accelerate/quicktour>
- <https://huggingface.co/docs/optimum-onnx/index>
- <https://huggingface.co/docs/evaluate/index>

### 5. The live build: a news classifier (5:30–6:30)

1 minute. Explain why the main demo uses a compact classifier: students can inspect the errors and the same trained weights can pass through every advertised library. This task is separate from MultiSpecQR. Open the notebook or prepared terminal. All commands live in demo/README.md. Show the prepared data/model revisions and preflight output. Run preparation and the full training before the event; during the talk show a short training invocation only if rehearsal fits the allotted time, otherwise use recorded training results clearly labeled as a prior local run. Do not leave the audience waiting for downloads.
Transition: A reproducible run starts before the optimizer.

- <https://huggingface.co/distilbert/distilbert-base-uncased>
- <https://huggingface.co/datasets/fancyzhx/ag_news>

### 6. Data and model provenance (6:30–8:00)

90 seconds. These are schematic excerpts with configuration variables; the executable script resolves and records the full revisions. A repository name is an address, while a full commit SHA identifies a version. Show the actual manifest. Training fits parameters. Validation selects settings. The held-out test estimates performance after choices are locked. Normalize and check duplicate text across splits; real projects also split by user, source, or time when leakage follows those relationships. This classroom slice is not proof for a future news distribution. Streaming is useful for large sources, but requires explicit sampling and shuffling decisions.
Transition: Before adapting anything, we need an honest baseline.

- <https://huggingface.co/docs/huggingface_hub/guides/download>
- <https://huggingface.co/docs/datasets/stream>

### 7. LoRA changes the training budget (8:00–10:00)

2 minutes. Explain the matrix idea without requiring linear algebra: keep the main model, learn a compact adjustment. For a d_out by d_in matrix, full tuning updates d_out*d_in entries; rank-r LoRA adds r*(d_in+d_out), where r is usually much smaller than either dimension. This is parameter arithmetic, not a claim about equal quality or speed. The task classifier head still trains and must be saved. For this demo the starting classification head is newly initialized, so an untrained base would be a sanity check, not a strong pretrained news baseline. The evaluation report includes a majority-class baseline. A serious study would also compare full fine-tuning and simpler task-specific alternatives.
Transition: Here is the part that attaches the adapter.

- <https://huggingface.co/docs/peft/main/conceptual_guides/lora>
- <https://huggingface.co/docs/peft/quicktour>

### 8. PEFT in the training script (10:00–12:00)

2 minutes. Show the actual LoraConfig in demo.py and printed trainable count. q_lin and v_lin are DistilBERT-specific module names, not universal choices. Explain rank as a tunable capacity parameter, chosen with validation rather than the test set. PEFT SEQ_CLS handles standard classifier head saving; the executable code explicitly keeps the needed modules. The saved adapter is not a standalone full model. The lecture uses ordinary LoRA. The appendix explains QLoRA for a causal LLM without claiming our classifier run demonstrates it. Avoid promising inference gains from LoRA by itself.
Transition: The learning code also needs to run on the hardware we have.

- <https://huggingface.co/docs/peft/quicktour>
- <https://huggingface.co/docs/peft/developer_guides/quantization>

### 9. Accelerate carries the training loop (12:00–14:00)

2 minutes. The slide is a shortened training excerpt, not a complete loop: the executable also clears gradients, moves data through prepared loaders, controls seeds, and saves on the main process. Explain that DDP replicates a model per process; more GPUs do not automatically make a too-large model fit. FSDP/DeepSpeed shard state but add configuration and communication costs. Show accelerate launch in the runbook. Our local rehearsal validates one process; multi-GPU is an explicit extension that needs a cluster test. Inference device_map='auto' is a different path and should not be taught as distributed training.
Transition: The trained adapter now becomes a candidate artifact.

- <https://huggingface.co/docs/accelerate/quicktour>
- <https://huggingface.co/docs/accelerate/concept_guides/big_model_inference>

### 10. Live checkpoint: the candidate artifact (14:00–16:00)

2 minutes including terminal switch. Open the real preparation/training outputs. State whether the run is happening now or was completed before the talk. Show that the adapter folder is small relative to the complete merged model, without claiming a fixed ratio unless the measured files support it. Read a validation metric and a known failure. Keep the audience focused on the saved artifact and its dependency on the base model. If training is still running, immediately switch to the completed rehearsal outputs. Do not rerun a long job on stage.
Transition: A convincing example is not the release decision.

- <https://huggingface.co/docs/peft/quicktour#save-model>

### 11. Evaluation as a release gate (16:00–18:30)

2 minutes 30 seconds. Show the actual evaluation JSON and confusion matrix. Accuracy is useful, but macro-F1 weights classes equally; per-class recall reveals a weak class. A small held-out slice has uncertainty, and news examples do not validate safety-critical decisions or distribution shift. Show gate thresholds set in config before the test run. Run the deliberately failing gate fixture, explicitly label it synthetic test data, and show the nonzero exit status. Then show the real candidate gate outcome without changing thresholds to obtain a pass. A failed gate is a successful engineering demonstration. For generative tasks, swap in task-specific exact match/schema checks, groundedness/factuality review, and calibrated human review; a single judge score is insufficient.
Transition: The runtime artifact gets evaluated too.

- <https://huggingface.co/docs/evaluate/index>
- <https://huggingface.co/docs/lighteval/main/index>

### 12. Optimum ONNX targets the CPU (18:30–20:30)

2 minutes. Show the executable export command or the completed exported artifacts, according to rehearsal timing. The slide omits quantizer construction/save calls to keep it legible; they are in demo.py. Our selected architecture supports this path. Dynamic INT8 changes runtime arithmetic and storage; it differs from QLoRA's 4-bit base loading for adapter training. AVX2 is our x86 target choice, not a universal configuration for ARM or GPUs. Static quantization would need representative calibration data from training/calibration sources, never tuning on the held-out test. Save tokenizer, configuration and label mapping beside the exported model.
Transition: Smaller files are only one part of the result.

- <https://huggingface.co/docs/optimum-onnx/installation>
- <https://huggingface.co/docs/optimum-onnx/onnxruntime/quickstart>
- <https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/quantization>

### 13. Measured quality and CPU latency (20:30–22:00)

90 seconds. These are actual measurements from the local rehearsal on September 17, 2026. The test set is a balanced, fixed 800-example slice with 200 examples per class. Quality uses the same held-out rows for the merged PyTorch model and the two exported packages. The benchmark is one warm sequential CPU run, not a concurrent service capacity test. All models use 4 CPU threads and fixed 128-token padding, 100 requests after 10 warm-ups. Timings include tokenization and forward execution, excluding startup, network, and queueing. Weight size excludes tokenizer/config files. Hardware: Intel Core i9-14900HX; training used an RTX 4090 Laptop GPU, but every latency here is CPU.

PyTorch FP32: accuracy 90.25%, macro-F1 0.902308, p50 33.48 ms, p95 59.81 ms, weights 267838720 bytes.
ONNX FP32: accuracy 90.25%, macro-F1 0.902308, p50 30.33 ms, p95 41.56 ms, weights 267938323 bytes.
ONNX INT8: accuracy 90.50%, macro-F1 0.904809, p50 25.82 ms, p95 33.57 ms, weights 67370935 bytes.

The small INT8 quality difference is not evidence of an accuracy improvement. Quantization can trade quality for footprint and speed, so inspect the actual errors and gate. This measured run does not predict a phone, microcontroller, or other CPU. Rehearse under event conditions. Show the source JSON and current artifact hashes if asked.
Transition: Deployment target determines the next branch.

- <demo/results/evaluation.json>
- <demo/results/benchmark.json>
- <demo/results/gate.json>
- <https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/quantization>

### 14. Inference across deployment targets (22:00–24:00)

2 minutes. Separate the demonstrated path from architecture guidance. The laptop classifier is a constrained CPU example, not an actual phone deployment. Transformers.js is a separate supported-model/export path. A causal LLM served with vLLM is a different model and runtime. Tensor parallelism partitions a model to fit/work across GPUs; replicas improve capacity for independent requests. KV cache and activation memory matter in addition to weights. Endpoints offers managed dedicated deployment but still needs appropriate hardware, limits, tests, and monitoring. TGI is in maintenance mode in current docs; vLLM/SGLang are current new-deployment examples.
Transition: Serving the file still leaves release operations.

- <https://huggingface.co/docs/transformers.js/index>
- <https://docs.vllm.ai/en/latest/serving/parallelism_scaling/>
- <https://huggingface.co/docs/inference-endpoints/en/index>
- <https://huggingface.co/docs/text-generation-inference/en/index>

### 15. A deployment you can operate (24:00–26:00)

2 minutes including the Colab demo. Open notebook section 8, change HEADLINES, and rerun the inference cell. Point to the exported INT8 artifact that the cell loads and show the resulting predictions. The local Gradio app is an optional fallback if Colab is unavailable. This demonstrates the exported model artifact in an interactive demo. Explain the remaining service work: auth for sensitive access, input limits, health/readiness checks, concurrency/load testing, structured logs, monitored quality, and rollback to an immutable artifact. A Space is useful for sharing a demo; a dedicated Endpoint or owned service can meet different operational requirements. No paid resources or public writes are required in this talk package.
Transition: The ecosystem also shortens work outside the training loop.

- <https://huggingface.co/docs/hub/spaces-overview>
- <https://huggingface.co/docs/inference-endpoints/en/index>

### 16. Five more ways to save work (26:00–28:00)

2 minutes. Use a quick tour rather than opening five unpredictable live tabs. Streaming can avoid a full initial download; approximate shuffle buffers still matter. Spaces connects a repository to a demo app. Current compute-space eligibility and pricing vary, so do not promise every audience member a free CPU Space. Jobs moves a script to hosted compute and may incur charges. Providers routes supported models to providers; uploading any custom model does not make it provider-hosted. Collections can organize the talk's models, data, docs, and Spaces when the speaker chooses to publish one. Optional take-home branches include Sentence Transformers for retrieval, Diffusers for generative media, and LeRobot for robotics.
Transition: Everyone can take one version of this workflow home.

- <https://huggingface.co/docs/datasets/stream>
- <https://huggingface.co/docs/hub/spaces-overview>
- <https://huggingface.co/docs/hub/jobs>
- <https://huggingface.co/docs/inference-providers/index>
- <https://huggingface.co/docs/hub/collections>

### 17. The workflow to take home (28:00–30:00)

2 minutes. Return to the promise: Hub conventions make artifacts accessible; PEFT reduces the trainable update; Accelerate handles training execution; Optimum targets the runtime; evaluation and operations turn these into an engineering workflow. Ask one audience prediction: if quantization cuts file size but harms a class recall floor, do we ship? Let someone answer, then point back to the release contract. Direct students to the notebook and handout. Direct engineers to the manifest, gate, benchmark protocol, and deployment notes. End on time at minute 30 and move to questions. No claim that the local demo validates a multi-GPU cluster or a real edge product.



### 18. Questions (30:00–35:00)

5 minutes. Take questions. Good prompts if the room is quiet: When would you use RAG instead of fine-tuning? How would you evaluate rare classes? What changes on a phone? What does a second GPU solve? The appendix covers QLoRA, multi-GPU strategy, evaluation beyond classification, and sources. Use the Q&A guide for short answers. Keep the boundary clear between verified local results and unexecuted hardware recipes.

- <https://huggingface.co/Jemsbhai>

### 19. Appendix: QLoRA for a causal LLM (Appendix)

Use only if asked. QLoRA combines low-bit base loading and LoRA, often using bitsandbytes for supported hardware. It is not the ONNX INT8 export shown in the main talk. A causal model requires task-appropriate prompt formatting, data licenses, evaluation, and memory planning. Do not promise that a particular parameter count will fit solely from weight bits: activations, optimizer states, sequence length, and other overhead matter. Serving the adapter or merging depends on the selected model/runtime/quantization path. This package's executed results apply to the classifier only.

- <https://huggingface.co/docs/peft/developer_guides/quantization>
- <https://huggingface.co/docs/transformers/quantization/bitsandbytes>

### 20. Appendix: what a second GPU solves (Appendix)

Answer the hardware question before naming a strategy. More devices can improve capacity, fit, or throughput, but each has a communication and operational cost. Show deployment-notes.md for a parameterized vLLM launch recipe and a multi-GPU Accelerate recipe. Those paths are not executed on this single-GPU machine. device_map='auto' provides big-model inference dispatch/offload, not a substitute for distributed training or throughput-oriented serving.

- <https://huggingface.co/docs/accelerate/quicktour>
- <https://docs.vllm.ai/en/latest/serving/parallelism_scaling/>
- <https://huggingface.co/docs/accelerate/concept_guides/big_model_inference>

### 21. Appendix: evaluation follows the task (Appendix)

Metrics become useful when tied to a concrete decision. Lock prompts, generation settings, datasets and model revisions. Test subgroup and adversarial cases where relevant. Calibrate automated judges against representative human review and track disagreement. A benchmark average does not establish correctness for every user. Quantify uncertainty for the intended use and keep final test examples out of iterative prompt/model selection. LightEval is an orchestration option for LLM evaluation, not a certification system.

- <https://huggingface.co/docs/lighteval/main/index>
- <https://huggingface.co/docs/evaluate/index>

### 22. Appendix: useful starting points (Appendix)

Point the audience to attendee-handout.md, the executable notebook, and the reference list. All externally researched slide claims have source links in notes. Documentation was checked September 17, 2026; rehearse with the included pinned environment and current hosting terms before the event. This is an independent community talk, not an official Hugging Face presentation.

- <https://huggingface.co/Jemsbhai>
- <https://huggingface.co/docs/peft>
- <https://huggingface.co/docs/accelerate>
- <https://huggingface.co/docs/optimum-onnx>
- <https://huggingface.co/learn>

