# Bringing the Heat

**Supercharging Your ML Pipelines with Hugging Face**  
Muntaser Syed, Lead GenAI Engineer at Insight Global, formerly NVIDIA  
Academic affiliation: Florida Institute of Technology

Venue: Miami Dade College

## The workflow

1. **Version the inputs.** Keep the data and model commit SHAs, split membership, tokenizer, labels, dependencies, and training configuration.
2. **Establish a baseline.** Compare with a simple approach appropriate to the task. Inspect errors before adding complexity.
3. **Adapt efficiently.** PEFT can train small adapter updates. Accelerate handles training execution across supported hardware configurations.
4. **Evaluate the exact candidate.** Keep the test split out of model selection. Inspect aggregate scores, important classes or slices, and actual failures.
5. **Optimize for a target.** Export or quantize a supported architecture. Measure the resulting artifact on the intended hardware.
6. **Release with evidence.** Keep a repeatable runtime package, a quality gate, observability, and a rollback revision.

The Hub supplies shared artifact conventions. Your application still defines acceptable quality, performance, and operating behavior.

## Which tool does what?

| Tool | Why you might use it |
|---|---|
| [Hub](https://huggingface.co/docs/hub/index) | Discover, version, document, and share models and datasets |
| [Datasets](https://huggingface.co/docs/datasets/index) | Load, transform, stream, and split data |
| [Transformers](https://huggingface.co/docs/transformers/index) | Load and run supported pretrained architectures |
| [PEFT](https://huggingface.co/docs/peft/quicktour) | Train parameter-efficient adapters such as LoRA |
| [Accelerate](https://huggingface.co/docs/accelerate/quicktour) | Adapt PyTorch training loops to hardware and distributed execution |
| [Optimum ONNX](https://huggingface.co/docs/optimum-onnx/index) | Export supported models and optimize execution with ONNX Runtime |
| [Evaluate](https://huggingface.co/docs/evaluate/index) | Reusable general ML metrics |
| [LightEval](https://huggingface.co/docs/lighteval/main/index) | Configurable LLM evaluation tasks, metrics, and backends |
| [Transformers.js](https://huggingface.co/docs/transformers.js/index) | Run supported models in browsers or JavaScript environments |
| [Spaces](https://huggingface.co/docs/hub/spaces-overview) | Share interactive demos from a versioned repository |
| [Jobs](https://huggingface.co/docs/hub/jobs) | Run scripts on hosted compute |
| [Inference Providers](https://huggingface.co/docs/inference-providers/index) | Call supported hosted models through a common client |
| [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/en/index) | Operate dedicated managed model services |

Hosting availability, hardware, and charges depend on the chosen service and account. Consult current documentation before provisioning.

## The demo

[Open the demo in Colab](https://colab.research.google.com/github/jemsbhai/bringing-the-heat/blob/main/demo/Bringing_the_Heat_Colab.ipynb), select a T4 GPU if available, and choose **Runtime → Run all**. It downloads pinned public inputs, trains DistilBERT on AG News with LoRA, exports ONNX INT8, evaluates the held-out test, benchmarks CPU inference, checks a release gate, and predicts headlines inside the notebook. No Hugging Face token or Drive mount is required. Download the result bundle at the end.

The [public repository](https://github.com/jemsbhai/bringing-the-heat) includes the slides, notes, notebook, scripts, and measured reports. The optional local Gradio interface and offline rehearsal instructions are in `demo/README.md`.

Use the notebook's plots to ask three questions: does validation improve as training loss falls, which classes still fail, and what changes after export and quantization? Compare the same test examples and benchmark protocol across runtimes. The slide charts report the laptop rehearsal; your Colab charts report your own run.

The example makes the mechanics inspectable. Its small classroom data slice and local timing measurements do not establish production quality, mobile-device performance, or multi-GPU scaling.

## A release record you can copy

```text
Task and intended users:
Base model ID + commit:
Dataset ID + commit + split membership:
Tokenizer / preprocessing / label mapping:
Training code commit + configuration:
Adapter / merged model / runtime artifact hashes:
Evaluation implementation + held-out results:
Per-class or important-slice failures:
Acceptance thresholds chosen before final evaluation:
Hardware + runtime + latency measurement scope:
Deployment configuration + previous release:
Owner and rollback procedure:
```

Accuracy measures the fraction of correct predictions. Macro-F1 averages each class’s F1 equally, so large classes do not dominate it. Per-class recall asks how many examples of each true class the model catches. A confusion matrix shows where errors go. All of these depend on representative test data and sample size.

## Experiments after the talk

**First experiment, about 30–60 minutes after setup:** run the prepared classifier, inspect five incorrect predictions, and write down a specific data or modeling hypothesis. Change one factor on validation data. Preserve the final test set for a later locked comparison.

**Engineering extension:** measure the same inputs under different batch sizes and thread counts. Separate model-only latency from tokenization and request overhead. Add a concurrent-load test before making service throughput claims.

**Your own problem:** substitute a dataset with clear usage terms and a task-appropriate model. Design splits around the unit that could leak, such as person, device, source, or time. Keep a simpler baseline.

## Explore the speaker’s work

- [Jemsbhai on the Hub](https://huggingface.co/Jemsbhai)
- [muntasersyed.com](https://muntasersyed.com)
- [MultiSpecQR RGB decoder](https://huggingface.co/Jemsbhai/multispecqr-rgb)
- [RGB dataset](https://huggingface.co/datasets/Jemsbhai/multispecqr-rgb-dataset)
- [MultiSpecQR library](https://github.com/jemsbhai/multispecqr)

These custom CNN decoders illustrate how the Hub supports work beyond language models. Their `from_pretrained` method belongs to the MultiSpecQR library. It does not imply automatic compatibility with Transformers, PEFT, or Optimum wrappers.

Continue learning through [Hugging Face Learn](https://huggingface.co/learn). For other projects, explore [Sentence Transformers](https://www.sbert.net/) for embeddings, [Diffusers](https://huggingface.co/docs/diffusers/index) for generative media, and [LeRobot](https://huggingface.co/docs/lerobot/index) for robotics.

Documentation checked September 17, 2026. The included lock file records the demo’s tested packages.
