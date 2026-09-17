"""Build a self-contained, fresh-runtime Colab notebook from the demo source."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})


def code(source, *, hidden=False):
    cells.append({"cell_type": "code", "metadata": {"cellView": "form"} if hidden else {},
                  "source": source.splitlines(keepends=True), "execution_count": None, "outputs": []})


md("""# Bringing the Heat 🔥 — run the complete pipeline in Colab
**Muntaser Syed · [muntasersyed.com](https://muntasersyed.com)**

**Hub → reproducible dataset → PEFT LoRA + Accelerate → Optimum ONNX INT8 → evaluation → release gate → inference.**

1. Use **Runtime → Change runtime type → T4 GPU** if available. CPU also works, but full training is substantially slower. This notebook does not use a TPU.
2. Choose **Runtime → Run all**. The first run installs the teaching libraries and downloads public model/data files. Every stage runs for real; there are no precomputed metrics or training bypasses.
3. For a 30-minute presentation, run it beforehand and narrate the saved results. Keep the notebook open while it executes; Colab resources and GPUs are not guaranteed.

No Hugging Face token, Drive mount, paid API, or public web server is required. The notebook carries its own version of the source files. Its outputs live in the current runtime and can be downloaded at the end.

**The task:** classify news as World, Sports, Business or Sci/Tech. This small model makes the engineering workflow visible; the recipe is not an LLM or a claim of production readiness.
""")
md("""## 0 · Set the contract before seeing results
We use the same 3,200 training / 400 validation / 800 locked test examples as the talk.

The Colab **teaching** policy is separate from the presenter's laptop policy: two CPU benchmark threads and a fixed illustrative 250 ms p95 budget on this VM. Hardware and timing are recorded with every run. This is not an edge-device measurement or a product SLA. For a real release, freeze a budget for your actual serving hardware before evaluation.

Keep three epochs for the complete exercise. A gate that returns **BLOCK** is a valid measurement outcome: do not lower thresholds after looking at the test. New experiments should use validation; repeated tuning against this test invalidates it as a heldout decision set.
""")
code("""from pathlib import Path
import datetime, hashlib, importlib.metadata, json, os, platform, site, subprocess, sys, time, zipfile
from IPython.display import display, Markdown, Image

EPOCHS = 3
BATCH_SIZE = 32
BENCHMARK_THREADS = 2
# Change policy only before a fresh experiment, not in response to test results.
CPU_P95_BUDGET_MS = 250.0
RUN_LABEL = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
BASE = Path('/content') if Path('/content').is_dir() else Path.cwd()
WORK = BASE / f'bringing-the-heat-{RUN_LABEL}'
WORK.mkdir(parents=True, exist_ok=False)
print('Run folder:', WORK)
print('Notebook Python:', platform.python_version())

# Child processes inherit only task controls here; no secrets are read or printed.
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['PYTHONUNBUFFERED'] = '1'

def read(relative):
    return json.loads((WORK / relative).read_text(encoding='utf-8'))

def command(argv, *, log_name, allowed=(0,)):
    # Stream progress, preserve a log, and propagate every unexpected failure.
    started = time.perf_counter()
    logs = WORK / 'logs'
    logs.mkdir(exist_ok=True)
    with (logs / f'{log_name}.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([str(arg) for arg in argv], cwd=WORK,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding='utf-8', errors='replace', bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        result = process.wait()
    elapsed = time.perf_counter() - started
    print(f'[{log_name}] exit={result}; elapsed={elapsed:.1f}s')
    if result not in allowed:
        raise RuntimeError(f'{log_name} failed with exit {result}; see output above.')
    return result
""")

payload = {name: (ROOT / name).read_text(encoding="utf-8") for name in
           ["demo.py", "visuals.py", "hub_revisions.json", "requirements-colab.txt", "colab_policy.json"]}
hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in payload.items()}
code("#@title Materialize the exact demo source bundled with this notebook\n"
     + "# Expand this cell to inspect the complete source. No repository clone is required.\n"
     + "BUNDLED_FILES = " + repr(payload) + "\n"
     + "BUNDLED_SHA256 = " + repr(hashes) + "\n"
     + """for name, source in BUNDLED_FILES.items():
    assert hashlib.sha256(source.encode()).hexdigest() == BUNDLED_SHA256[name]
    (WORK / name).write_text(source, encoding='utf-8')
policy = read('colab_policy.json')
policy['benchmark_threads'] = BENCHMARK_THREADS
policy['max_cpu_p95_ms'] = CPU_P95_BUDGET_MS
(WORK / 'release_policy.json').write_text(json.dumps(policy, indent=2) + '\\n', encoding='utf-8')
(WORK / 'source_manifest.json').write_text(json.dumps(BUNDLED_SHA256, indent=2) + '\\n', encoding='utf-8')
display(policy)
print('Source and policy frozen before prepare/train/evaluate.')
""", hidden=True)

md("""## 1 · Install without restarting the notebook
The libraries run in a notebook-owned virtual environment that can reuse Colab's installed PyTorch and CUDA build. We constrain PyTorch to that exact installed version, so this setup does not replace it. Pinned teaching libraries override packages only inside this environment; the Colab kernel stays intact.

The setup checks the actual imports before downloading the model. All model work runs in fresh child processes, avoiding the “installed a package but forgot to restart” problem. The complete environment is recorded in the reports.
""")
code("""torch_version = importlib.metadata.version('torch')
PYENV = BASE / 'bringing-the-heat-colab-env'
command([sys.executable, '-m', 'venv', '--system-site-packages', '--without-pip', PYENV],
        log_name='create_environment')
PYTHON = PYENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
# Also supports a local Jupyter kernel that is itself in a virtual environment.
child_site = subprocess.check_output([str(PYTHON), '-c',
    "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True).strip()
(Path(child_site) / 'heat_runtime_packages.pth').write_text(
    '\\n'.join(site.getsitepackages()) + '\\n', encoding='utf-8')
(WORK / 'runtime-constraints.txt').write_text(f'torch=={torch_version}\\n', encoding='utf-8')
command([sys.executable, '-m', 'pip', '--python', PYTHON, 'install', '--quiet',
         '--disable-pip-version-check', '--no-warn-conflicts',
         '-r', WORK / 'requirements-colab.txt', '-c', WORK / 'runtime-constraints.txt'],
        log_name='install')

def run(*args, allowed=(0,)):
    return command([PYTHON, WORK / 'demo.py', *args],
                   log_name='-'.join(str(arg).replace('--', '') for arg in args[:2]), allowed=allowed)

command([PYTHON, '-c', '''import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig
from accelerate import Accelerator
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
import datasets, onnxruntime, sklearn, matplotlib
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Training device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU — full training will take longer')
print('Teaching stack imports: PASS')'''], log_name='import_check')

def draw(stage, *figure_names):
    # Render from this run's files in the same pinned environment as the pipeline.
    command([PYTHON, WORK / 'visuals.py', stage, '--root', WORK], log_name=f'plot-{stage}')
    for name in figure_names:
        display(Image(filename=str(WORK / 'figures' / f'{name}.png'), width=1100))

draw('workflow', '01_workflow')
""")

md("""## 2 · Turn Hub objects into reproducible inputs
`prepare` downloads **DistilBERT** and **AG News** at explicit commit revisions. It writes balanced splits once, removes normalized exact duplicates across all splits and hashes every split. Validation comes from the original training split; our test comes from the original test split. Near-duplicate and time/source leakage checks are further production work.

The Hub is more than a download page: model cards, dataset cards, licenses, revisions, adapters and runnable examples all become part of the engineering handoff.
""")
code("""run('prepare')
run('preflight')
manifest = read('artifacts/manifest.json')
display({name: {'examples': split['count'], 'sha256': split['sha256']}
         for name, split in manifest['splits'].items()})
print('Model:', manifest['model_id'], '@', manifest['model_revision'])
print('Dataset:', manifest['dataset_id'], '@', manifest['dataset_revision'])
draw('splits', '02_splits')
""")

md("""## 3 · Train the small part: PEFT + Accelerate
**PEFT controls what learns; Accelerate controls how the training loop runs on the available device.**

```python
config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
    lora_dropout=0.1, target_modules=['q_lin', 'v_lin'],
    modules_to_save=['pre_classifier', 'classifier'])
model = get_peft_model(base, config)
model, optimizer, train_loader, validation_loader = accelerator.prepare(
    model, optimizer, train_loader, validation_loader)
accelerator.backward(model(**batch).loss)
optimizer.step()
```

We train the low-rank attention updates and classification heads. The rest is frozen. The **best validation macro F1** selects the checkpoint. No test data is used for checkpoint selection. This cell always performs the complete training run, on CUDA if available and CPU otherwise.
""")
code("""run('train', '--epochs', EPOCHS, '--batch-size', BATCH_SIZE)
training = read('results/training.json')
assert not training['smoke_only'] and training['epochs'] == EPOCHS
print(f\"Trainable: {training['trainable_parameters']:,} / {training['total_parameters']:,} ({training['trainable_percent']:.2f}%)\")
display([{'epoch': item['epoch'], 'loss': item['mean_loss'],
          'validation_macro_f1': item['validation']['macro_f1']}
         for item in training['history']])
draw('training', '03_lora_budget', '04_training_progress')
""")

md("""## 4 · Export the candidate, then quantize with Optimum
Merge the adapter into the base to produce a standalone model, export ONNX FP32, then apply dynamic INT8 quantization for the x86 AVX2 CPU target. The unmerged adapter remains useful for storage/distribution but requires its base model.

```python
ort_model = ORTModelForSequenceClassification.from_pretrained(merged, export=True)
quantizer = ORTQuantizer.from_pretrained(onnx_fp32)
quantizer.quantize(save_dir=onnx_int8,
    quantization_config=AutoQuantizationConfig.avx2(is_static=False, per_channel=False))
```

INT8 does not guarantee a speedup or unchanged quality. We test the exported artifact, including its packaged tokenizer, next.
""")
code("""run('export', '--target', 'avx2')
exported = read('results/export.json')
display({name: round(sum(item['bytes'] for item in files.values()) / 1024**2, 2)
         for name, files in exported['artifacts'].items()})
print('Values above are total packaged MiB, not peak memory.')
draw('export', '05_export_footprint')
""")

md("""## 5 · Evaluate the exact artifact on the locked test
Compare PyTorch FP32, ONNX FP32 and ONNX INT8 on the **same 800 examples**, with a majority-class baseline. Macro F1 gives each class equal weight; per-class recall exposes weak slices. The Wilson interval describes sampling uncertainty on this dataset, not tomorrow's distribution shift.
""")
code("""run('evaluate')
evaluation = read('results/evaluation.json')
display({name: {key: score[key] for key in ['n', 'accuracy', 'macro_f1']}
         for name, score in evaluation['models'].items()})
display({'majority_class_baseline': evaluation['majority_class_baseline']})
display({'int8_per_class': evaluation['models']['int8']['per_class']})
print('INT8 accuracy 95% Wilson interval:', evaluation['models']['int8']['accuracy_95pct_wilson'])
draw('evaluation', '06_evaluation')
""")

md("""## 6 · Measure inference on this VM
All three implementations use CPU, the same thread count, batch 1, a fixed padded 128-token input, 10 warmups and 100 sequential requests. Timing includes tokenization + forward and excludes model loading, network and queueing. Disk weight size is not RAM usage. The GPU accelerates training; these measurements isolate CPU inference.

Colab's shared VM is a learning environment. Results will differ from the presenter's laptop and your target device. Keep these measurements with this run's hardware metadata and remeasure on the deployment target.
""")
code("""run('benchmark', '--threads', BENCHMARK_THREADS, '--samples', 100, '--warmup', 10)
benchmark = read('results/benchmark.json')
display({name: {'p50_ms': round(score['p50_ms'], 2), 'p95_ms': round(score['p95_ms'], 2),
                'weights_MiB': round(score['weights_bytes'] / 1024**2, 2)}
         for name, score in benchmark['models'].items()})
display(benchmark['environment'])
draw('benchmark', '07_latency', '08_quality_size')
""")

md("""## 7 · Make the release decision executable
The gate checks quality, every-class recall, quantization regression, sample size, footprint, the frozen CPU timing budget, the measurement protocol, and hashes of the exact runtime files.

**PASS and BLOCK are both valid results.** Unexpected execution failures still stop the notebook. We then inject a simulated Business-recall failure and require it to block, without modifying the real evaluation report. The illustrative policy is not production certification.
""")
code("""gate_exit = run('gate', allowed=(0, 2))
gate = read('results/gate.json')
assert gate['passed'] == (gate_exit == 0)
display(Markdown('### Teaching gate: ' + ('PASS' if gate['passed'] else 'BLOCK')))
display(gate['checks'])
run('gate', '--inject-failure', allowed=(2,))
injected = read('results/gate_deliberate_failure.json')
assert not injected['checks']['every_class_recall'] and not injected['passed']
print('Deliberate slice regression was blocked correctly.')
draw('gates', '09_release_gate')
""")

md("""## 8 · Inference with the packaged INT8 artifact
This cell exercises the same exported model directly inside the notebook. Edit `HEADLINES` and rerun it to try your own examples. There is no external web server or public Gradio tunnel.

Scores are not calibrated confidence. The model always chooses one of four categories; it has no “unknown” class, and input is truncated to 128 tokens. A blocked candidate can still be inspected here; the teaching gate does not authorize a deployment.
""")
code("""HEADLINES = [
    'The Miami team won the championship after a dramatic final quarter.',
    'Shares climbed after the company reported higher quarterly earnings.',
    'Scientists unveiled a new processor for energy-efficient computing.',
    'Leaders met for talks on a new international peace agreement.',
]
predict_args = ['predict', '--threads', BENCHMARK_THREADS]
for headline in HEADLINES:
    predict_args.extend(['--text', headline])
run(*predict_args)
display(read('results/inference.json')['predictions'])
draw('inference', '10_inference')
""")

md("""## 9 · Keep the evidence and artifact
The bundle contains the measured INT8 runtime, tokenizer, adapter, pinned-input manifest, source, policy, JSON reports, and every visual as **PNG + SVG**. The workflow and LoRA path are conceptual diagrams; numerical charts are generated from this run's actual results. It excludes the base/merged FP32 weights and raw dataset. Download it using Colab's **Files** sidebar before the runtime expires. Set `DOWNLOAD_NOW=True` for an immediate browser download.

For a real rollout, this handoff still needs a deployment target, service limits, authentication, concurrent load tests, observability, staged rollout and rollback. Colab hosts the exercise; it is not your production serving platform.
""")
code("""complete = {
    'status': 'ALL_PIPELINE_STAGES_COMPLETED', 'run_label': RUN_LABEL,
    'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_sha256': BUNDLED_SHA256,
    'training_epochs': training['epochs'], 'smoke_only': training['smoke_only'],
    'environment': benchmark['environment'], 'gate_passed': gate['passed'],
    'deliberate_failure_blocked': not injected['passed'],
    'inference_examples': len(read('results/inference.json')['predictions']),
    'visuals': read('figures/provenance.json'),
}
(WORK / 'results' / 'run_complete.json').write_text(json.dumps(complete, indent=2) + '\\n', encoding='utf-8')
BUNDLE = BASE / f'bringing-the-heat-colab-{RUN_LABEL}.zip'
files = [WORK / name for name in ['demo.py', 'visuals.py', 'hub_revisions.json', 'requirements-colab.txt',
    'release_policy.json', 'source_manifest.json', 'artifacts/manifest.json']]
for folder in ['results', 'figures', 'artifacts/adapter', 'artifacts/onnx_int8']:
    files.extend(path for path in (WORK / folder).rglob('*') if path.is_file() and '.cache' not in path.parts)
with zipfile.ZipFile(BUNDLE, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(WORK))
display(complete)
print(f'Bundle ready: {BUNDLE} ({BUNDLE.stat().st_size / 1024**2:.1f} MiB)')
DOWNLOAD_NOW = False
if DOWNLOAD_NOW:
    from google.colab import files as colab_files
    colab_files.download(str(BUNDLE))
""")

md("""## Take it further
- Change LoRA rank using validation, add a difficult slice and freeze a new policy before evaluating a fresh heldout set.
- Rebenchmark the exported runtime on the real edge device. ARM64 needs a different export/quantization target.
- Use Accelerate's multi-GPU launcher on a suitable host; keep effective batch size constant. DDP replicates this model; it does not provide model sharding or an inference server.
- For LLMs, separately choose quantization, sharding, batching and a serving runtime appropriate to the model and hardware.

**Official references:** [Colab FAQ](https://research.google.com/colaboratory/faq.html) · [Colab runtime versions](https://github.com/googlecolab/backend-info) · [Accelerate](https://huggingface.co/docs/accelerate/quicktour) · [PEFT LoRA](https://huggingface.co/docs/peft/developer_guides/lora) · [Optimum ONNX](https://huggingface.co/docs/optimum-onnx) · [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) · [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased)
""")

for i, cell in enumerate(cells):
    cell['id'] = f'heat-colab-{i:02d}'
    cell['metadata']['id'] = cell['id']

notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "colab": {"name": "Bringing_the_Heat_Colab.ipynb", "provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
    },
    "cells": cells,
}
(ROOT / 'Bringing_the_Heat_Colab.ipynb').write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print('Built self-contained Colab notebook:', len(cells), 'cells')
