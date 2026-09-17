"""Bringing the Heat: reproducible Hub -> LoRA -> ONNX -> release gate.

Run `python demo.py --help`. All outputs stay beside this script. No uploads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
RESULTS = ROOT / "results"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
SEED = 42
MAX_LENGTH = 128


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_hash(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def manifest():
    return read_json(ART / "manifest.json")


def rows(split):
    path = ART / "data" / f"{split}.jsonl"
    expected = manifest()["splits"][split]["sha256"]
    if sha(path) != expected:
        raise RuntimeError(f"{split} checksum changed; refusing to use an altered split")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def environment():
    import torch
    names = ["torch", "transformers", "peft", "accelerate", "datasets", "optimum-onnx", "onnxruntime", "scikit-learn", "gradio"]
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None  # The Colab notebook uses inline inference without Gradio.
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "cpu_logical_count": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": {name: version(name) for name in names}}


def prepare(args):
    """Download public, immutable revisions; make disjoint, balanced splits once."""
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    if (ART / "manifest.json").exists():
        print("Prepared manifest already exists. Run preflight to verify it; preserved unchanged.")
        return
    api = HfApi(token=False)
    pins_path = ROOT / "hub_revisions.json"
    if pins_path.exists():
        pins = read_json(pins_path)
    else:
        pins = {"model_id": "distilbert/distilbert-base-uncased", "dataset_id": "fancyzhx/ag_news"}
        pins["model_revision"] = api.model_info(pins["model_id"]).sha
        pins["dataset_revision"] = api.dataset_info(pins["dataset_id"]).sha
        write_json(pins_path, pins)
    ART.mkdir(exist_ok=True)
    snapshot_download(pins["model_id"], revision=pins["model_revision"], token=False,
                      local_dir=ART / "base_model",
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"])
    ds = load_dataset(pins["dataset_id"], revision=pins["dataset_revision"], token=False)
    # Exact-normalized-text deduplication also blocks exact cross-split duplicates.
    seen = set()
    def select(source, count, source_name, rng_seed):
        indices = list(range(len(source)))
        random.Random(rng_seed).shuffle(indices)
        counts = [0] * 4
        selected = []
        for index in indices:
            row = source[index]
            label, fingerprint = int(row["label"]), text_hash(row["text"])
            if counts[label] >= count // 4 or fingerprint in seen:
                continue
            seen.add(fingerprint)
            counts[label] += 1
            selected.append({"id": f"{source_name}:{index}", "text": row["text"], "label": label,
                             "text_sha256": fingerprint})
            if len(selected) == count:
                break
        if len(selected) != count:
            raise RuntimeError("Not enough unique data for the requested balanced split")
        return selected
    partitions = {"train": select(ds["train"], 3200, "train", SEED),
                  "validation": select(ds["train"], 400, "train", SEED + 1),
                  "test": select(ds["test"], 800, "test", SEED + 2)}
    split_metadata = {}
    (ART / "data").mkdir(exist_ok=True)
    for name, items in partitions.items():
        p = ART / "data" / f"{name}.jsonl"
        p.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n", encoding="utf-8")
        split_metadata[name] = {"count": len(items), "sha256": sha(p), "class_counts": [sum(r["label"] == i for r in items) for i in range(4)]}
    write_json(ART / "manifest.json", {**pins, "seed": SEED, "max_length": MAX_LENGTH,
        "labels": LABELS, "splits": split_metadata,
        "split_policy": "Balanced fixed subsets; validation from official train, test from official test; normalized exact-text deduplication across all subsets. Near duplicates are not detected.",
        "policy_sha256": sha(ROOT / "release_policy.json"), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps(manifest(), indent=2))


def preflight(args):
    import torch
    from transformers import AutoTokenizer
    data = {name: rows(name) for name in ("train", "validation", "test")}
    sets = {name: {r["text_sha256"] for r in values} for name, values in data.items()}
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])
    assert sha(ROOT / "release_policy.json") == manifest()["policy_sha256"], "Release policy changed after preparation"
    AutoTokenizer.from_pretrained(ART / "base_model", local_files_only=True)
    assert (ART / "base_model" / "model.safetensors").exists()
    report = {"status": "PASS", "environment": environment(), "split_integrity": "PASS",
              "cpu_threads_for_benchmark": read_json(ROOT / "release_policy.json").get("benchmark_threads", 4), "cuda_smoke": None}
    if torch.cuda.is_available():
        report["cuda_smoke"] = float((torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda"))[0, 0])
    write_json(RESULTS / "preflight.json", report)
    print(json.dumps(report, indent=2))


def make_loader(items, tokenizer, batch_size=32, shuffle=False):
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    ds = Dataset.from_list(items)
    ds = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH), batched=True,
                remove_columns=["id", "text", "text_sha256"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=DataCollatorWithPadding(tokenizer))


def metrics(labels, predictions):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    p, r, f, n = precision_recall_fscore_support(labels, predictions, labels=list(range(4)), zero_division=0)
    accuracy = float(accuracy_score(labels, predictions))
    # Wilson interval makes the heldout set's sampling uncertainty visible.
    z, count = 1.96, len(labels)
    center = (accuracy + z * z / (2 * count)) / (1 + z * z / count)
    half = z * math.sqrt(accuracy * (1 - accuracy) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return {"n": count, "accuracy": accuracy, "macro_f1": float(f.mean()),
        "accuracy_95pct_wilson": [center - half, center + half],
        "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(n[i])} for i, name in enumerate(LABELS)},
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(4))).tolist()}


def train(args):
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(4)
    accelerator = Accelerator(cpu=args.cpu, mixed_precision="no")
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(ART / "base_model", local_files_only=True)
    base = AutoModelForSequenceClassification.from_pretrained(ART / "base_model", num_labels=4,
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
        local_files_only=True, attn_implementation="eager")
    config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"], modules_to_save=["pre_classifier", "classifier"])
    model = get_peft_model(base, config)
    trainable, total = model.get_nb_trainable_parameters()
    accelerator.print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    train_loader = make_loader(rows("train"), tokenizer, args.batch_size, shuffle=True)
    validation_loader = make_loader(rows("validation"), tokenizer, args.batch_size)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    model, optimizer, train_loader, validation_loader = accelerator.prepare(model, optimizer, train_loader, validation_loader)
    started, history, best = time.perf_counter(), [], -1.0
    num_epochs = 1 if args.smoke else args.epochs
    output_name = "smoke_adapter" if args.smoke else "adapter"
    for epoch in range(num_epochs):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(**batch).loss
            accelerator.backward(loss)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 25 == 0:
                accelerator.print(f"Epoch {epoch + 1}/{num_epochs} step {step}/{len(train_loader)} loss={losses[-1]:.4f}", flush=True)
            if args.smoke and step >= 4:
                break
        model.eval()
        ys, ps = [], []
        with torch.inference_mode():
            for batch in validation_loader:
                pred, target = accelerator.gather_for_metrics((model(**batch).logits.argmax(-1), batch["labels"]))
                ps.extend(pred.cpu().tolist())
                ys.extend(target.cpu().tolist())
        score = metrics(ys, ps)
        history.append({"epoch": epoch + 1, "mean_loss": statistics.mean(losses), "validation": score})
        accelerator.print(f"Validation macro F1: {score['macro_f1']:.4f}")
        if score["macro_f1"] > best:
            best = score["macro_f1"]
            accelerator.wait_for_everyone()
            state = accelerator.get_state_dict(model)
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(ART / output_name, state_dict=state)
                tokenizer.save_pretrained(ART / output_name)
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        write_json(RESULTS / ("smoke_training.json" if args.smoke else "training.json"), {"environment": environment(), "seconds": time.perf_counter() - started,
            "seed": SEED, "epochs": num_epochs, "smoke_only": args.smoke, "learning_rate": args.learning_rate,
            "per_device_batch_size": args.batch_size, "world_size": accelerator.num_processes,
            "effective_batch_size": args.batch_size * accelerator.num_processes,
            "trainable_parameters": trainable, "total_parameters": total,
            "trainable_percent": 100 * trainable / total, "selection": "highest validation macro F1", "history": history})


def export(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    torch.set_num_threads(4)
    base = AutoModelForSequenceClassification.from_pretrained(ART / "base_model", num_labels=4,
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)}, local_files_only=True,
        attn_implementation="eager")
    model = PeftModel.from_pretrained(base, ART / "adapter", local_files_only=True).merge_and_unload()
    model.save_pretrained(ART / "merged")
    tokenizer = AutoTokenizer.from_pretrained(ART / "base_model", local_files_only=True)
    tokenizer.save_pretrained(ART / "merged")
    ort_model = ORTModelForSequenceClassification.from_pretrained(ART / "merged", export=True, provider="CPUExecutionProvider")
    ort_model.save_pretrained(ART / "onnx_fp32")
    tokenizer.save_pretrained(ART / "onnx_fp32")
    quantizer = ORTQuantizer.from_pretrained(ART / "onnx_fp32")
    factory = getattr(AutoQuantizationConfig, args.target)
    quantizer.quantize(save_dir=ART / "onnx_int8", quantization_config=factory(is_static=False, per_channel=False))
    tokenizer.save_pretrained(ART / "onnx_int8")
    files = {}
    for name in ["adapter", "merged", "onnx_fp32", "onnx_int8"]:
        files[name] = {str(p.relative_to(ART / name)): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in (ART / name).rglob("*") if p.is_file() and ".cache" not in p.parts}
    write_json(RESULTS / "export.json", {"target": args.target, "quantization": "dynamic INT8 weights / runtime activations; embeddings may remain FP32", "artifacts": files})
    print("Merged adapter, exported ONNX FP32, and generated ONNX dynamic INT8.")


def load_backend(backend, threads=4):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(threads)
    folder = {"pytorch": "merged", "onnx": "onnx_fp32", "int8": "onnx_int8"}[backend]
    tokenizer = AutoTokenizer.from_pretrained(ART / folder, local_files_only=True)
    if backend == "pytorch":
        model = AutoModelForSequenceClassification.from_pretrained(ART / "merged", local_files_only=True,
            attn_implementation="eager").eval()
    else:
        import onnxruntime as ort
        from optimum.onnxruntime import ORTModelForSequenceClassification
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        folder = "onnx_int8" if backend == "int8" else "onnx_fp32"
        filename = "model_quantized.onnx" if backend == "int8" else "model.onnx"
        model = ORTModelForSequenceClassification.from_pretrained(ART / folder, file_name=filename,
            provider="CPUExecutionProvider", session_options=options, local_files_only=True)
    return model, tokenizer


def artifact_identity(backend):
    folder = ART / {"pytorch": "merged", "onnx": "onnx_fp32", "int8": "onnx_int8"}[backend]
    files = {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*"))
             if p.is_file() and ".cache" not in p.parts}
    if not files:
        raise RuntimeError(f"Missing runtime artifact: {folder}")
    description = {"files": files, "max_length": MAX_LENGTH, "truncation": True, "label_order": LABELS}
    return {"sha256": hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest(), **description}


def evaluate(args):
    import torch
    test = rows("test")
    labels = [r["label"] for r in test]
    report = {"split": "locked test", "split_sha256": manifest()["splits"]["test"]["sha256"],
              "models": {}, "environment": environment(), "artifacts": {name: artifact_identity(name) for name in ["pytorch", "onnx", "int8"]}}
    train_labels = [r["label"] for r in rows("train")]
    majority_label = max(range(4), key=train_labels.count)
    report["majority_class_baseline"] = {"chosen_label": LABELS[majority_label],
        "rule": "Most frequent training label; ties choose lowest label index", **metrics(labels, [majority_label] * len(labels))}
    for backend in ["pytorch", "onnx", "int8"]:
        model, tokenizer = load_backend(backend)
        predictions = []
        with torch.inference_mode():
            for i in range(0, len(test), args.batch_size):
                inputs = tokenizer([r["text"] for r in test[i:i + args.batch_size]], padding=True,
                    truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
                predictions.extend(model(**inputs).logits.argmax(-1).tolist())
        report["models"][backend] = metrics(labels, predictions)
        write_json(RESULTS / f"predictions_{backend}.json", {"ids": [r["id"] for r in test], "labels": labels, "predictions": predictions})
        print(f"{backend}: accuracy={report['models'][backend]['accuracy']:.4f}, macro F1={report['models'][backend]['macro_f1']:.4f}", flush=True)
        del model
    write_json(RESULTS / "evaluation.json", report)


def percentile(values, p):
    ordered = sorted(values)
    i = (len(values) - 1) * p
    lo = math.floor(i)
    return ordered[lo] + (ordered[min(lo + 1, len(values) - 1)] - ordered[lo]) * (i - lo)


def benchmark(args):
    import torch
    test = rows("test")
    # Fixed batch=1, 128-token padded shape for comparable latency; fixed CPU threads.
    texts = [r["text"] for r in test[:args.samples]]
    report = {"environment": environment(), "threads": args.threads, "batch_size": 1,
        "sequence_length": MAX_LENGTH, "samples": len(texts), "warmup": args.warmup,
        "device": "CPU", "timing_scope": "Tokenization + model forward, sequential batch=1 requests; excludes load, networking and queueing",
        "split_sha256": manifest()["splits"]["test"]["sha256"], "models": {},
        "artifacts": {name: artifact_identity(name) for name in ["pytorch", "onnx", "int8"]}}
    for backend in ["pytorch", "onnx", "int8"]:
        model, tokenizer = load_backend(backend, args.threads)
        def infer(text):
            inputs = tokenizer(text, padding="max_length", max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
            return model(**inputs).logits
        with torch.inference_mode():
            for i in range(args.warmup):
                infer(texts[i % len(texts)])
            timings = []
            for text in texts:
                started = time.perf_counter_ns()
                infer(text)
                timings.append((time.perf_counter_ns() - started) / 1e6)
        folder = {"pytorch": "merged", "onnx": "onnx_fp32", "int8": "onnx_int8"}[backend]
        weights_bytes = sum(p.stat().st_size for p in (ART / folder).rglob("*") if p.is_file() and p.suffix in (".safetensors", ".onnx", ".data"))
        report["models"][backend] = {"latencies_ms": timings, "p50_ms": percentile(timings, .5),
            "p95_ms": percentile(timings, .95), "mean_ms": statistics.mean(timings),
            "sequential_requests_per_second": 1000 / statistics.mean(timings), "weights_bytes": weights_bytes}
        print(f"{backend}: p50={report['models'][backend]['p50_ms']:.2f}ms, p95={report['models'][backend]['p95_ms']:.2f}ms, weights={weights_bytes / 1024 ** 2:.1f}MiB", flush=True)
        del model
    write_json(RESULTS / "benchmark.json", report)


def gate_checks(evaluation, benchmark_report, policy, current_artifacts, current_split_sha):
    baseline, candidate = evaluation["models"]["pytorch"], evaluation["models"]["int8"]
    checks = {
        "same_heldout_split": evaluation["split_sha256"] == benchmark_report["split_sha256"] == current_split_sha,
        "exact_runtime_artifacts": all(evaluation.get("artifacts", {}).get(name, {}).get("sha256") ==
            benchmark_report.get("artifacts", {}).get(name, {}).get("sha256") == current_artifacts[name]["sha256"]
            for name in ["pytorch", "onnx", "int8"]),
        "complete_class_report": set(candidate["per_class"]) == set(LABELS),
        "benchmark_protocol": benchmark_report["device"] == "CPU" and benchmark_report["threads"] == policy.get("benchmark_threads", 4)
            and benchmark_report["batch_size"] == 1 and benchmark_report["sequence_length"] == MAX_LENGTH
            and benchmark_report["samples"] >= 100 and benchmark_report["warmup"] >= 10
            and len(benchmark_report["models"]["int8"]["latencies_ms"]) == benchmark_report["samples"],
        "minimum_test_size": candidate["n"] >= policy["min_test_examples"],
        "accuracy": candidate["accuracy"] >= policy["min_accuracy"],
        "macro_f1": candidate["macro_f1"] >= policy["min_macro_f1"],
        "every_class_recall": min((c["recall"] for c in candidate["per_class"].values()), default=0) >= policy["min_class_recall"],
        "quantization_f1_regression": baseline["macro_f1"] - candidate["macro_f1"] <= policy["max_macro_f1_drop"],
        "cpu_p95_latency": benchmark_report["models"]["int8"]["p95_ms"] <= policy["max_cpu_p95_ms"],
        "weight_size": benchmark_report["models"]["int8"]["weights_bytes"] <= policy["max_weights_bytes"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def gate(args):
    evaluation, bench = read_json(RESULTS / "evaluation.json"), read_json(RESULTS / "benchmark.json")
    policy = read_json(ROOT / "release_policy.json")
    if sha(ROOT / "release_policy.json") != manifest()["policy_sha256"]:
        raise RuntimeError("Release policy changed after preparation; freeze it before observing the heldout test")
    rows("test")  # Verify the current heldout file, not just matching old report identifiers.
    identities = {name: artifact_identity(name) for name in ["pytorch", "onnx", "int8"]}
    if args.inject_failure:
        # Obvious simulated regression; authentic evaluation stays untouched.
        evaluation["models"]["int8"]["per_class"]["Business"]["recall"] = 0.10
    result = {**gate_checks(evaluation, bench, policy, identities, manifest()["splits"]["test"]["sha256"]), "policy": policy,
              "injected_failure": args.inject_failure, "scope": "Teaching gate on this locked sample and CPU; not a production certification"}
    name = "gate_deliberate_failure.json" if args.inject_failure else "gate.json"
    write_json(RESULTS / name, result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


def predict(args):
    """Exercise the exported artifact directly, including in hosted notebooks."""
    import torch
    model, tokenizer = load_backend("int8", threads=args.threads)
    predictions = []
    for text in args.text:
        if not text.strip() or len(text) > 10000:
            raise ValueError("News text must contain 1–10,000 characters")
        with torch.inference_mode():
            inputs = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            scores = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
        predictions.append({"text": text, "label": LABELS[max(range(len(scores)), key=scores.__getitem__)],
                            "scores": dict(zip(LABELS, scores))})
    report = {"backend": "onnx_int8", "artifact": artifact_identity("int8"),
              "scores_are_calibrated": False, "predictions": predictions}
    write_json(RESULTS / "inference.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command, fn in [("prepare", prepare), ("preflight", preflight), ("train", train), ("export", export), ("evaluate", evaluate), ("benchmark", benchmark), ("gate", gate), ("predict", predict)]:
        p = subs.add_parser(command)
        p.set_defaults(fn=fn)
        if command == "train":
            p.add_argument("--cpu", action="store_true")
            p.add_argument("--smoke", action="store_true", help="Five training steps; writes separate smoke_adapter, preserves the rehearsed artifacts")
            p.add_argument("--epochs", type=int, default=3)
            p.add_argument("--batch-size", type=int, default=32)
            p.add_argument("--learning-rate", type=float, default=3e-4)
        elif command == "export":
            p.add_argument("--target", choices=["avx2", "avx512", "avx512_vnni", "arm64"], default="avx2")
        elif command == "evaluate":
            p.add_argument("--batch-size", type=int, default=32)
        elif command == "benchmark":
            p.add_argument("--threads", type=int, default=4)
            p.add_argument("--samples", type=int, default=100)
            p.add_argument("--warmup", type=int, default=10)
        elif command == "gate":
            p.add_argument("--inject-failure", action="store_true")
        elif command == "predict":
            p.add_argument("--text", action="append", required=True)
            p.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
