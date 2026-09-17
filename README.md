# Bringing the Heat: talk materials

**[Muntaser Syed](https://muntasersyed.com) · Miami Dade College · 30 minutes + 5 minutes Q&A**

Designed for students and working ML engineers. The talk opens with Muntaser's actual public MultiSpecQR models and datasets, then follows one compact classifier through the Hugging Face stack: **Hub → reproducible data → PEFT LoRA + Accelerate → Optimum ONNX INT8 → evaluation → release gate → inference**.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jemsbhai/bringing-the-heat/blob/main/demo/Bringing_the_Heat_Colab.ipynb)

## Run the complete demo

1. Open the Colab notebook above and select **Runtime → Change runtime type → T4 GPU** if available. CPU works but full training is substantially slower.
2. Choose **Runtime → Run all**. The notebook installs its teaching libraries, downloads public model/data files, trains for three epochs, exports and quantizes, evaluates the heldout test set, benchmarks inference, and checks the release policy.
3. Inspect the real gate outcome, run the deliberate failure example, and download your results bundle from the final cell.

No Hugging Face token, Drive mount, paid API, or public web server is required. The notebook embeds its source, so it starts from a fresh runtime. Colab availability varies; rehearse before presenting. The [Colab validation record](demo/COLAB_TESTED.md) separates actual Colab evidence from local checks.

**Verified on a fresh Colab T4 runtime:** all 11 code cells completed in about 15½ minutes, including the full training/evaluation workflow, a passing teaching gate, a correctly blocked deliberate regression, and four inference examples. The [executed notebook](demo/Bringing_the_Heat_Colab_executed.ipynb) preserves the outputs. CPU evaluation took most of that time; run the complete notebook before the talk and rerun headline inference live.

## Start here

- [Download the complete materials kit](https://github.com/jemsbhai/bringing-the-heat/releases/download/v1.0.0/Bringing-the-Heat-kit.zip): slides, notebooks, speaker materials, source, and recorded results.
- [Public deck on Google Drive](https://drive.google.com/file/d/1cWJ_CbI899vkuPVZTnfYJgcGdz8jv3t7/view): the opening slide's QR code links here.
- [Hosted Colab notebook](https://colab.research.google.com/drive/1Q9WkLenTKLEd-R0wTzsCgF1mrLS1NCIh): view the presenter's run or save your own copy. The badge above opens the versioned exercise from GitHub.
- [Editable PowerPoint](output/Bringing-the-Heat.pptx): 17 talk slides, Q&A, and 4 appendix slides. Includes timed speaker notes and source links.
- [Offline slides](output/slides-offline.html): self-contained browser fallback. Arrow keys advance; N toggles notes.
- [Presenter guide](output/presenter-guide.md): full slide-by-slide narrative.
- [Stage runbook](output/stage-runbook.md): live commands, rehearsal, timing cuts, and failure recovery.
- [Colab notebook](demo/Bringing_the_Heat_Colab.ipynb): complete fresh-runtime exercise.
- [Local stage notebook](demo/Bringing_the_Heat.ipynb) and [demo README](demo/README.md): the original local workflow, saved laptop outputs, and tested setup.
- [Attendee handout](output/attendee-handout.md): tool map, exercises, and further learning.
- [Measured results](output/measured-results.md): the actual classifier quality, CPU latency, and artifact sizes.
- [Q&A guide](output/qa-guide.md) and [deployment extensions](output/deployment-notes.md).
- [Personal Hub showcase](showcase/README.md): saved public metadata and a genuine MultiSpecQR sample.
- [Slide authoring files](authoring/README.md): editable slide content and build instructions.

## Talk arc

| Minutes | Focus |
|---|---|
| 0–4 | Production expectations and your own Hub work |
| 4–8 | Ecosystem roles, one task, reproducible data |
| 8–16 | LoRA, Accelerate, and the candidate artifact |
| 16–22 | Release criteria, ONNX INT8, measured results |
| 22–28 | Deployment choices, notebook inference, ecosystem extras |
| 28–30 | Takeaways for students and engineers |
| 30–35 | Questions |

## Demonstration boundary

The core is DistilBERT + AG News, PEFT LoRA, Accelerate, Optimum ONNX, evaluation, a CPU benchmark, and inference. The local kit also includes a Gradio app. [Local validation](demo/TESTED.md) records executed checks, and [saved results](demo/results/) contain the presenter's laptop measurements.

The slides' latency numbers come from that laptop. The Colab notebook records its own hardware and uses a separate, frozen teaching policy. A **BLOCK** result is valid evidence: inspect it rather than lowering thresholds after seeing the test results. Neither policy is a product SLA.

Browser, ARM64, multi-GPU, QLoRA, and managed Endpoints are explicitly labeled extension recipes. The personal MultiSpecQR showcase uses its own custom CNN library; it does not claim automatic compatibility with the classifier recipe.

Public source metadata and official documentation were checked September 17, 2026. Rehearse on the event hardware and recheck hosting terms before the talk. The demo reads public Hub assets without publishing models or datasets to an account.

This repository and the lightweight ZIP contain teaching materials, source, and recorded results. Python environments, cached downloads, trained weights, and internal build-validation files are excluded; the notebook regenerates its model artifacts.
