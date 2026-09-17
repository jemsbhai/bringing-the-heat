# Stage runbook

**30-minute talk, followed by 5 minutes of Q&A.** This page controls the live transitions. The presenter guide contains the speaking notes, and `demo/README.md` contains the complete setup.

## Before the event

The primary demo is [the complete Colab notebook](https://colab.research.google.com/github/jemsbhai/bringing-the-heat/blob/main/demo/Bringing_the_Heat_Colab.ipynb). Save a copy to Drive, select a T4 GPU if available, and choose **Runtime → Run all** before the talk. Keep the completed outputs visible for the presentation. The full run installs dependencies, downloads pinned public inputs, trains three epochs, exports and quantizes, evaluates all three runtimes, benchmarks CPU inference, checks the release gate, and runs example predictions. No Hugging Face token or Drive mount is required.

On stage, use sections 3–8 for the live transitions below. Edit `HEADLINES` in section 8 and rerun the inference cell. Colab's teaching policy uses two CPU threads and an illustrative 250 ms p95 budget; the slide's recorded laptop results use the separate laptop policy. Identify which run you are showing. Save the notebook with outputs and download the bundle from section 9 before the runtime expires.

The local setup below is an optional offline fallback:

1. Rehearse on the laptop you will actually present from. Connect power and use the same performance mode you plan to use on stage.
2. From `demo/`, run `setup.ps1` once if needed, then `rehearse.ps1`. Use the correct CPU/CUDA option. Review `TESTED.md`, the gate result, and all saved measurements. Do not change the acceptance policy to manufacture a pass.
3. Open the deck in your presentation application and check the projector crop and code readability. `slides-offline.html` is a self-contained visual fallback: arrows advance, N shows notes, and the button enables full screen. It can also be printed through the browser.
4. Start the notebook with `.venv/Scripts/python.exe -m jupyterlab Bringing_the_Heat.ipynb`. Use the demo environment's kernel. Run prepared replay/inference cells once so imports and runtime loading have finished.
5. Start `.venv/Scripts/python.exe app.py`. Open `http://127.0.0.1:7860`, run an example, and leave the tab ready. The app stays on localhost.
6. Pre-open [your Hub profile](https://huggingface.co/Jemsbhai), [RGB model](https://huggingface.co/Jemsbhai/multispecqr-rgb), and [RGB dataset](https://huggingface.co/datasets/Jemsbhai/multispecqr-rgb-dataset). Keep `showcase/README.md` and `hub_manifest.json` available if the network fails.
7. Disable notifications, increase terminal/browser text size, hide unrelated tabs, and keep an elapsed timer visible to you. Have a second copy of the PPTX and offline HTML.

Do a final offline rehearsal after downloads:

```powershell
# From the demo directory; these environment changes affect this terminal only.
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
.venv/Scripts/python.exe demo.py preflight
```

Preparation can require several GiB of downloads and disk space. It does not belong in the talk. No Hub writes, paid Jobs, Endpoint creation, or external API calls are needed for the live core.

## Live transitions

| Talk time | Slide | Action | What the audience should notice |
|---|---|---|---|
| 2:00 | 3 | Profile, model card, files, dataset viewer | These are your real reusable artifacts |
| 5:30 | 5 | Prepared notebook and manifest | One concrete task, fixed inputs |
| 10:00 | 8 | Highlight LoRA config and trainable count | Only a small fraction of parameters update |
| 14:00 | 10 | Loss and validation-F1 curves; saved adapter | Lower training loss does not guarantee better validation quality |
| 16:00 | 11 | Confusion matrix, class-recall chart, and deliberate-failure gate visual | Metrics can block a release |
| 18:30 | 12 | Show export code and prepared ONNX folders | The same trained candidate changes runtime |
| 20:30 | 13 | Measured latency, weight-size, and quality charts | Speed and size gains must preserve task quality |
| 24:00 | 15 | Edit a headline in Colab section 8 and run inference; local Gradio is the fallback | The exported artifact backs an interface |
| 28:00 | 17 | Return to slides | Concrete takeaways and handoff |
| 30:00 | 18 | Stop the prepared talk | Protect the five-minute Q&A |

The optional five-step smoke training cell saves into a separate directory and does not replace the prepared candidate. A full local training pass can be quick once dependencies are warm, but never assume the first launch or venue laptop will match the rehearsal.

## The release-gate moment

```powershell
.venv/Scripts/python.exe demo.py gate
.venv/Scripts/python.exe demo.py gate --inject-failure
# The deliberately failing command should return exit code 2.
```

Say: “This second report deliberately changes one class recall so we can test the release rule. It is a synthetic failure, not a second measured model.” The command preserves the real measurements. If the real report fails too, read its failed condition and explain why that is useful. Avoid a passing/failing result without explaining its source.

## Recovery without losing the story

| Problem | Immediate response |
|---|---|
| Hub or venue internet stalls | Use the saved manifest, model-card text, and genuine sample on slide 3 |
| Imports or training exceed 20 seconds on stage | Stop narrating the wait and show the completed rehearsal outputs |
| No Colab GPU available | Show the completed Colab outputs or use the local prepared adapter and CPU inference path |
| Export is slow | Open the prepared ONNX folders and recorded export manifest |
| Gradio tab fails | Use the notebook's prepared inference cell |
| Model runtime fails entirely | Walk the saved measured report; label it as the rehearsal run |
| Projector cannot render the PPTX correctly | Open the offline HTML full screen |
| You are 60–90 seconds behind | Cut slide 16 to 30 seconds and slide 17 to 60 seconds |

The shortened ending creates up to 2 minutes 30 seconds of recovery without removing the core demo. At minute 28, stop any optional exploration and return to the takeaways.

## Wording that keeps the demo credible

- “This ran on my rehearsal laptop” for recorded measurements.
- “This is happening now” only for current live execution.
- “This is a deployment recipe we have not run here” for multi-GPU, browser, ARM64, and managed Endpoint extensions.
- “This local app demonstrates the interface; the deployment notes describe the remaining service work.”

After the session, share the deck, attendee handout, and lightweight demo source bundle. Local model artifacts and virtual environments are excluded from that bundle; recipients recreate them with the pinned preparation commands.
