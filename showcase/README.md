# Your Hub, on stage

A two-minute personal introduction to Hugging Face using **Muntaser Syed's real public MultiSpecQR work**. This segment connects the talk's engineering theme to a concrete computer-vision project, then hands off to the main fine-tuning and deployment demo.

Public metadata checked **September 17, 2026**. The canonical Hub username is **Jemsbhai**. The local snapshot records exact repository SHAs; counts below describe the dataset viewer at capture time.

## Two-minute stage script

**0:00–0:25 — Display the genuine RGB sample and your profile.**

“This is an actual sample from a dataset I put on Hugging Face. MultiSpecQR packs multiple independent payloads into a color QR image. The RGB version has three layers; the palette variants go up to nine. My neural decoder learns to separate those layers. It's a small computer-vision problem, and it uses the same Hub that hosts foundation models.”

**0:25–0:50 — Open the RGB model card, then Files.**

“Here's my model card: the architecture, the output contract, and the library that knows how to load it. The Hub gives a custom project a shareable address. Someone else can find my weights, read the intended use, and use the source code linked from the card.”

**0:50–1:15 — Open commit history; point to the saved full SHA.**

“The engineering move is to identify exactly what I ran. A model name is a convenient address; a commit is a reproducible version. I keep the model revision, data revision, and code revision together. That makes a result traceable when a teammate asks what changed.”

**1:15–1:40 — Open the RGB dataset viewer; show the split selector and metadata.**

“The data is inspectable too: training, validation, test, and fields such as augmentation and QR version. My RGB dataset exposes all three splits. For any dataset, I inspect that structure before choosing how to evaluate. For this problem, a useful test is whether every payload comes back correctly under the conditions we care about.”

**1:40–2:00 — Return to the workflow slide.**

“That's the pattern I want you to take away: discover, inspect, version, evaluate, deliver. The model can change, but this workflow travels with it. Now let's use the wider Hugging Face stack to make that workflow efficient enough to ship.”

## Tabs to pre-open

| Stop | Exact link | Point to |
|---|---|---|
| Your profile | [Jemsbhai](https://huggingface.co/Jemsbhai) | Four model repositories and two datasets |
| Model contract | [RGB model card](https://huggingface.co/Jemsbhai/multispecqr-rgb) | `RGBMLDecoder`, three outputs, `LayerUnmixingCNN`, library import |
| Artifact | [RGB files at the captured commit](https://huggingface.co/Jemsbhai/multispecqr-rgb/tree/f2053f53af6e137ac3b7a36045bb864f4f7ed00e) | `README.md` and `model.pt` |
| Reproducibility | [RGB commit history](https://huggingface.co/Jemsbhai/multispecqr-rgb/commits/main) | Commit identifier, change history |
| Data inspection | [RGB test viewer, row 0](https://huggingface.co/datasets/Jemsbhai/multispecqr-rgb-dataset/viewer/default/test?row=0) | Split selector, `sample_id`, `augmentation`, `qr_version` |
| Source | [MultiSpecQR source at captured commit](https://github.com/jemsbhai/multispecqr/tree/08063b6b6c2d62a081305f7e784c3b6180074b40) | A real custom library connecting to the Hub |

The viewer stores this dataset's `image` and `labels` as nested `uint8` lists. Use the saved PNG on the slide to show the actual image; use the browser viewer to show structure and metadata. Do not depend on the viewer producing an image thumbnail.

## Verified inventory

| Repository | What it contributes |
|---|---|
| [multispecqr-rgb](https://huggingface.co/Jemsbhai/multispecqr-rgb) | RGB decoder; three output layers |
| [multispecqr-palette6](https://huggingface.co/Jemsbhai/multispecqr-palette6) | Palette decoder; six output layers |
| [multispecqr-palette8](https://huggingface.co/Jemsbhai/multispecqr-palette8) | Palette decoder; eight output layers |
| [multispecqr-palette9](https://huggingface.co/Jemsbhai/multispecqr-palette9) | Palette decoder; nine output layers |
| [multispecqr-rgb-dataset](https://huggingface.co/datasets/Jemsbhai/multispecqr-rgb-dataset) | Viewer: 243,000 rows — train 233,000; validation 5,000; test 5,000 |
| [multispecqr-palette6-dataset](https://huggingface.co/datasets/Jemsbhai/multispecqr-palette6-dataset) | Viewer: 20,000 rows in train |

The model cards identify `LayerUnmixingCNN` and the `multispecqr` library. The GitHub project describes RGB channel encoding and palette modes with 64, 256, and 512 colors. These are project capabilities; this segment does not report a new accuracy or performance benchmark.

The counts come from the [RGB viewer size API](https://datasets-server.huggingface.co/size?dataset=Jemsbhai%2Fmultispecqr-rgb-dataset) and [palette6 viewer size API](https://datasets-server.huggingface.co/size?dataset=Jemsbhai%2Fmultispecqr-palette6-dataset), captured in `hub_manifest.json`.

## Reliable live and offline commands

Run from the talk workspace. Both scripts use only Python's standard library.

```powershell
# Stage-safe: saved metadata, no network or installation.
python showcase/hub_inventory.py --offline

# Rehearsal: refresh local public metadata and pin the observed commits.
python showcase/hub_inventory.py --refresh

# Optional: recapture one real dataset row and its provenance.
python showcase/capture_sample.py
```

`hub_inventory.py` is read-only with respect to the Hub. It sends no authentication token, downloads no checkpoint or full dataset, and writes only the local manifest. That manifest includes all six repository SHAs, model card text and metadata, file inventories, dataset viewer counts, and the source-code commit and file hashes.

If the venue network fails, show the PNG, run `--offline`, and use the same script. Say “this is my saved snapshot from September 17” when presenting the saved inventory.

## Slide asset and provenance

- **Image:** `assets/multispecqr-rgb-test-row0.png` — 512 × 512, exact RGB values from the public dataset's `image` column. No generation, enhancement, or resampling.
- **Caption:** “A real sample from my MultiSpecQR RGB dataset · 3 payload layers · Hugging Face Hub.”
- **Provenance:** `assets/multispecqr-rgb-test-row0.provenance.json` records the dataset, test row 0, sample ID `44d1224185a2e8ba`, observed repository SHA, API URL, capture time, and pixel/file hashes.
- **Exact source response:** `assets/multispecqr-rgb-test-row0.viewer.json.gz` preserves the full row and schema, compressed for an offline backup. Python's `gzip` and `json` can read it.

The Dataset Viewer serves its current cached data. The provenance records the repository SHA observed immediately before and after capture; it does **not** claim the viewer request was pinned to that SHA. This preserves the distinction between a reproducibly saved response and a version-pinned dataset read.

## Technical speaker notes

**Hub integration has levels.** The `from_pretrained` method in the card belongs to `multispecqr.ml_decoder`, not `transformers.AutoModel`. Uploading a custom model gives it versioned storage and discovery; Transformers pipelines, PEFT injection, Optimum export, inference widgets, and hosted providers each need compatible integrations. The [Hub library integration matrix](https://huggingface.co/docs/hub/models-libraries) explains these different capabilities. Use the supported architecture from the main demo for the PEFT/Accelerate/Optimum path.

**Demonstrate pinning at the Hub boundary.** In the [captured custom loader source](https://github.com/jemsbhai/multispecqr/blob/08063b6b6c2d62a081305f7e784c3b6180074b40/src/multispecqr/ml_decoder.py), `from_pretrained` downloads `model.pt` without forwarding a `revision`. Do not add `revision=...` to that custom call and assume it pins the model. A simple reproducible demonstration is a pinned card download through `huggingface_hub`:

```python
from pathlib import Path
from huggingface_hub import hf_hub_download

card_path = hf_hub_download(
    repo_id="Jemsbhai/multispecqr-rgb",
    filename="README.md",
    revision="f2053f53af6e137ac3b7a36045bb864f4f7ed00e",
)
print(Path(card_path).read_text(encoding="utf-8"))
```

This optional example downloads only text and requires `huggingface_hub`; the inventory commands above require no third-party package. See the [official versioned-download guide](https://huggingface.co/docs/huggingface_hub/guides/download).

**Keep the two-minute segment predictable.** The custom loader uses `torch.load(..., weights_only=False)` for its `.pt` checkpoint. The supplied showcase inspects metadata and source, and does not execute that loader. A live model inference demonstration would require a separately rehearsed dependency environment and checkpoint review.

**Use evaluation that matches the outcome.** For a future MultiSpecQR experiment, report per-layer payload exact match and all-payload exact match, broken down by degradation such as blur or JPEG artifacts, plus latency on the intended device. Pixel-level accuracy alone does not establish successful payload recovery. These are proposed evaluation criteria, not results from a test run.

**Check the split contract.** RGB currently exposes train/validation/test. Palette6 currently exposes train only, so create a deliberate held-out evaluation design before reporting generalization from that dataset. A useful discussion prompt is how to group base payloads or generated examples to keep related augmentations together across splits.

The inspected dataset repositories did not contain README cards or license metadata at capture time. The assets here support the owner's talk preparation; add the intended dataset license and documentation before presenting a broader reuse contract. Nothing in the public repositories was changed.
