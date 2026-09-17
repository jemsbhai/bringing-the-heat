# Five-minute Q&A guide

Use a short answer first, then the example. The appendix is optional; leave the question slide up unless a diagram/table directly helps.

**“Should I fine-tune or use RAG?”**  
Ask what is missing. Retrieval can supply changing or private facts at inference time. Fine-tuning can adapt behavior or a task pattern. They can be combined. Compare both against a simpler baseline using examples that reflect the actual task. Fine-tuning is not a dependable database update mechanism.

**“Why a classifier instead of a large chatbot?”**  
A small, concrete task lets us inspect the complete artifact path in 30 minutes. It supports an understandable metric and a consistent PEFT-to-ONNX demonstration. LLMs add generation, prompt formatting, longer contexts, KV cache, and more complicated evaluation. The companion deployment notes explain that separate branch.

**“Does LoRA always match full fine-tuning?”**  
No universal guarantee. Rank, target layers, data, and task matter. It reduces the number of trained parameters, while the base model still exists. Compare quality and resource use on the same validation protocol. The demo also trains and saves its classification head.

**“How is QLoRA different from INT8 export?”**  
QLoRA combines low-bit base loading with adapter training. The talk’s INT8 path quantizes a merged classifier for ONNX Runtime inference. They apply at different stages and have different compatibility constraints.

**“Can I upload a custom model and immediately call it through an API?”**  
The Hub can host the artifact. Execution also needs a compatible library and runtime. Supported provider models can be called through Inference Providers. Custom models may need a custom service or container. MultiSpecQR is a useful example of a custom library that knows how to load its own Hub weights.

**“Does Accelerate make any model work across all GPUs?”**  
It supplies training abstractions and launcher integrations. You still choose a suitable strategy. DDP copies the model per GPU. FSDP or DeepSpeed can shard training state. Inference offloading and tensor-parallel serving solve different problems.

**“Can I call this production-ready?”**  
The demo proves specific local steps. A real release also needs representative evaluation, measured concurrency, service limits, monitoring, and rollback. Say exactly which checks were run and on which hardware. The local Gradio UI is a demonstration interface.

**“Why did the release gate fail?”**  
Read the failed condition. The deliberate fixture is designed to fail. A real candidate can also fail a predeclared quality, regression, or latency budget. Keep the policy intact, investigate on training/validation data, and preserve test independence. Changing the threshold just to obtain a pass undermines the demonstration.

**“What should I measure for an LLM?”**  
Start with the job: correct structured fields, retrieval coverage, supported answers, or a domain-specific rubric. Freeze the prompt and generation settings. Human review can calibrate automated judges. LightEval can orchestrate a benchmark, but it does not choose acceptable risk for your application.

**“Does a smaller quantized model mean faster inference?”**  
Only a measurement on the target can answer that. Kernel support, CPU/GPU, batch size, input lengths, and overhead all matter. Report latency percentiles and quality together. Do not compare a batch-one CPU number with a batched GPU throughput number as if they were the same workload.

**“Where should a student begin?”**  
Run one compact task end to end. Inspect errors, record your revisions, and share a useful model card. The Hub, courses, and public datasets lower setup friction. Avoid starting with a cluster before you can explain what one model does on one example.

**“What about licenses?”**  
Check the selected model, dataset, and code terms for your intended use. Public availability and permissions are separate questions. Record the versions and terms alongside the artifact; this talk does not determine the rights for an attendee’s project.

For technical sources, use the relevant slide’s notes and `research/hf-facts.md`. These answers are framing advice, not additional benchmark results.
