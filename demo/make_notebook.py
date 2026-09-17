"""Regenerate the talk driver. The notebook calls the same CLI as the rehearsal."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True), "execution_count": None, "outputs": []})


md("""# Bringing the Heat 🔥
## Notebook → reproducible artifact → release decision
**Muntaser Syed · Miami Dade College**

Our application is a news router: **World / Sports / Business / Sci/Tech**.
One small classifier connects the whole stack: Hub, Datasets, Transformers, PEFT, Accelerate, Optimum, Gradio.

**Before the talk:** run `setup.ps1`, then `rehearse.ps1` from this folder. This notebook is the live driver over prepared local artifacts. Package downloads and full training happen before the session.

**Live rule:** narrate a running operation for 15 seconds; if it is still busy, move to the saved evidence. Never make the audience wait for a download.
""")
code("""from pathlib import Path
import json, os, subprocess, sys
import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / 'demo.py').exists():
    ROOT = ROOT / 'demo'
assert (ROOT / 'demo.py').exists(), 'Open this notebook from the demo folder.'
os.chdir(ROOT)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

def run(*args, expected=0):
    result = subprocess.run([sys.executable, 'demo.py', *args], capture_output=True, text=True)
    print(result.stdout)
    allowed = (expected,) if isinstance(expected, int) else expected
    if result.returncode not in allowed:
        print(result.stderr)
        raise RuntimeError(f'Expected exit {expected}, got {result.returncode}')
    return result

run('preflight');
""")
md("""## 1 · Hub objects become reproducible inputs
The Hub holds more than model weights: revisions, dataset schemas, cards, licenses, adapters, and runnable demos all matter.

`from_pretrained()` gets us started. A **commit SHA + environment lock + immutable split** makes the experiment explainable later.

This is a *balanced teaching subset*. Validation comes from the official training split; the locked test comes from the official test split. We remove normalized exact duplicates across all three; near duplicates require additional checks.
""")
code("""manifest = read('artifacts/manifest.json')
display(pd.DataFrame(manifest['splits']).T[['count', 'sha256']])
print('Model:', manifest['model_id'], '@', manifest['model_revision'])
print('Dataset:', manifest['dataset_id'], '@', manifest['dataset_revision'])
print('Class order:', manifest['labels'])
""")
md("""## 2 · Change the small part: PEFT + Accelerate
LoRA learns low-rank updates to attention projections while most of the base model stays frozen. Our new classification head also trains.

```python
config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=['q_lin', 'v_lin'],
    modules_to_save=['pre_classifier', 'classifier'],
)
model = get_peft_model(base, config)
model, optimizer, train_loader, validation_loader = accelerator.prepare(
    model, optimizer, train_loader, validation_loader
)
loss = model(**batch).loss
accelerator.backward(loss)
optimizer.step()
```

**Division of labor:** PEFT controls *what learns*; Accelerate controls *where and how the loop runs*. DDP replicates the model across GPUs. `device_map='auto'` is a different inference-placement tool, not this distributed training recipe.
""")
code("""# Optional real, isolated five-step training demo. Keep False for instant replay.
RUN_TRAINING_SMOKE = False
if RUN_TRAINING_SMOKE:
    run('train', '--smoke')  # writes smoke_adapter; preserves rehearsed adapter

training = read('results/training.json')
print(f\"Trainable: {training['trainable_parameters']:,} / {training['total_parameters']:,}\")
print(f\"Trainable fraction: {training['trainable_percent']:.2f}%\")
display(pd.DataFrame([{'epoch': row['epoch'], 'loss': row['mean_loss'],
    'validation_macro_f1': row['validation']['macro_f1']} for row in training['history']]))
print('Checkpoint chosen using validation only.')
""")
md("""## 3 · Evaluate the candidate, then evaluate the exported artifact
Accuracy can hide a weak class. Macro F1 gives each class equal weight; per-class recall answers *which category are we missing?*

We compare three implementations on exactly the same 800 examples. A majority-class baseline makes “better” meaningful. Read the confidence interval as uncertainty on this sample, not a promise about tomorrow's news.
""")
code("""evaluation = read('results/evaluation.json')
summary = {name: {key: scores[key] for key in ['n', 'accuracy', 'macro_f1']}
           for name, scores in evaluation['models'].items()}
summary['majority_baseline'] = {key: evaluation['majority_class_baseline'][key]
                               for key in ['n', 'accuracy', 'macro_f1']}
display(pd.DataFrame(summary).T)
display(pd.DataFrame(evaluation['models']['int8']['per_class']).T)
print('INT8 accuracy 95% Wilson interval:', evaluation['models']['int8']['accuracy_95pct_wilson'])
""")
md("""## 4 · Optimum: export and measure on the target
The adapter is first **merged into the base** to produce a standalone model, then exported to ONNX. Dynamic INT8 quantization targets this laptop's AVX2 CPU.

```python
ort_model = ORTModelForSequenceClassification.from_pretrained('artifacts/merged', export=True)
quantizer = ORTQuantizer.from_pretrained('artifacts/onnx_fp32')
quantizer.quantize(save_dir='artifacts/onnx_int8',
    quantization_config=AutoQuantizationConfig.avx2(is_static=False, per_channel=False))
```

Do not compare GPU batch throughput to CPU single-request latency. Here all three backends use **CPU, four threads, batch 1, fixed 128 tokens, 10 warmups, 100 measured requests**. Time includes tokenization + forward; excludes loading, network and queueing. Weight bytes are file size, not peak RAM.
""")
code("""benchmark = read('results/benchmark.json')
display(pd.DataFrame({name: {'p50_ms': value['p50_ms'], 'p95_ms': value['p95_ms'],
    'sequential_requests_per_second': value['sequential_requests_per_second'],
    'weights_MiB': value['weights_bytes'] / 1024**2}
    for name, value in benchmark['models'].items()}).T.round(2))
print(benchmark['environment']['platform'])
print(benchmark['timing_scope'])
""")
md("""## 5 · Make “ship it” executable
Our *teaching* policy was written before the test: minimum quality, every-class recall, limited quantization regression, a CPU p95 budget, and a footprint budget. Real product owners set their own values from costs, risks and service goals.

First pass the real candidate. Then inject a simulated Business-recall regression. The overall accuracy is unchanged in this deliberate demonstration, but the gate must still block release.
""")
code("""run('gate', expected=(0, 2))  # A genuine block is a valid demonstration outcome.
run('gate', '--inject-failure', expected=2);
""")
md("""## 6 · The exact artifact behind a local app
Try your own headline. Scores are **not calibrated confidence** and this four-way classifier has no “unknown” class. Mixed-topic text can be ambiguous.

For the Gradio UI, open a terminal in this folder and run:

```powershell
.venv/Scripts/python.exe app.py
```

Open <http://127.0.0.1:7860>. This binds to localhost; it is a preview UI, not an authenticated production service.
""")
code("""from app import classify
display(classify('The Miami team won the championship after a dramatic final quarter.'))
""")
md("""## The engineering handoff
- **Edge:** ship the measured artifact and tokenizer; remeasure on the real device. ARM64 has a different export/quantization target.
- **GPU training:** use Accelerate's launcher and config; keep effective batch size and evaluation correct as world size changes.
- **LLM serving:** separately choose a server/runtime that supports your model and hardware. Sharding to fit memory and batching for throughput are different decisions.
- **Release:** version the artifact, gate it, stage it, observe it, and keep a rollback path.

**Take-home challenge:** change LoRA rank using validation, propose a new slice metric, or benchmark a different target. Freeze the test and policy before measuring the final candidate.

[Accelerate quicktour](https://huggingface.co/docs/accelerate/quicktour) · [PEFT LoRA](https://huggingface.co/docs/peft/developer_guides/lora) · [Optimum ONNX](https://huggingface.co/docs/optimum-onnx) · [AG News](https://huggingface.co/datasets/fancyzhx/ag_news)
""")
notebook = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12.12"}}, "cells": cells}
for i, cell in enumerate(cells):
    cell['id'] = f'heat-{i:02}'
(ROOT / 'Bringing_the_Heat.ipynb').write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
