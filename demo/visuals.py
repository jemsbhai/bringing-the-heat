"""Notebook diagrams and plots. Charts read this run's JSON, never canned metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

BG, PANEL, FG, MUTED = "#0B1120", "#152139", "#F1F5FA", "#B8C5D9"
TEAL, HOT, BLUE, GOLD, RED = "#52DECA", "#FF826B", "#83B4FF", "#F4CE72", "#F28193"
BACKENDS = ["pytorch", "onnx", "int8"]
NAMES = ["PyTorch FP32", "ONNX FP32", "ONNX INT8"]
COLORS = [BLUE, GOLD, TEAL]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "figure.facecolor": BG,
    "axes.facecolor": PANEL, "axes.edgecolor": "#41516D", "axes.labelcolor": FG,
    "text.color": FG, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titleweight": "bold", "axes.titlepad": 16,
    "grid.color": "#52627C", "grid.alpha": .25,
    "savefig.facecolor": BG, "svg.fonttype": "none",
})


def read(root, name):
    return json.loads((root / name).read_text(encoding="utf-8"))


def canvas(title, subtitle, height=7.8):
    fig = plt.figure(figsize=(14, height))
    fig.text(.055, .945, title, fontsize=24, weight="bold", va="top")
    fig.text(.055, .88, subtitle, fontsize=12, color=MUTED, va="top")
    return fig


def save(fig, root, name, note):
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    fig.text(.055, .035, note, color=MUTED, fontsize=10, va="bottom")
    for ext in ["png", "svg"]:
        fig.savefig(folder / f"{name}.{ext}", dpi=145)
    plt.close(fig)
    print(f"Saved figures/{name}.png and .svg")


def clean(ax, *, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid)


def box(ax, x, y, w, h, title, detail="", color=TEAL, size=14):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=PANEL, edgecolor=color, linewidth=1.7))
    ax.text(x+w/2, y+h*.66 if detail else y+h/2, title, fontsize=size,
        ha="center", va="center", color=FG, weight="bold")
    if detail:
        ax.text(x+w/2, y+h*.28, detail, fontsize=10.5, ha="center", va="center", color=MUTED)


def arrow(ax, start, end, color=MUTED, **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17,
        linewidth=1.7, color=color, **kwargs))


def workflow(root):
    fig = canvas("One experiment. An engineering handoff.",
        "Conceptual workflow • inputs and policy are frozen before training and heldout evaluation")
    ax = fig.add_axes([.05, .16, .90, .65]); ax.set(xlim=(0, 14), ylim=(0, 5)); ax.axis("off")
    xs, w, h = [.15, 3.65, 7.15, 10.65], 3.15, 1.25
    top = [("01  Hub", "Pin model + dataset revisions", BLUE),
           ("02  Datasets", "Split, deduplicate, checksum", BLUE),
           ("03  PEFT + Accelerate", "Train adapter; select on validation", HOT),
           ("04  Optimum", "Merge → ONNX → INT8", GOLD)]
    bottom = [("08  Inference", "Inspect the packaged artifact", TEAL),
              ("07  Release gate", "PASS or BLOCK from evidence", TEAL),
              ("06  Benchmark", "CPU latency + footprint", GOLD),
              ("05  Evaluate", "Same locked test; three runtimes", HOT)]
    for row, y in [(top, 3.3), (bottom, .6)]:
        for x, (title, detail, color) in zip(xs, row):
            box(ax, x, y, w, h, title, detail, color, 12.5)
    for i in range(3):
        arrow(ax, (xs[i]+w, 3.925), (xs[i+1], 3.925))
        arrow(ax, (xs[i+1], 1.225), (xs[i]+w, 1.225))
    arrow(ax, (xs[-1]+w/2, 3.3), (xs[-1]+w/2, 1.85))
    ax.text(5.4, 2.45, "Every transition produces a versioned artifact or report",
            fontsize=12, ha="center", color=MUTED)
    save(fig, root, "01_workflow", "Conceptual diagram • a passed teaching gate is one input to a real deployment decision, not production certification.")


def splits(root):
    manifest = read(root, "artifacts/manifest.json")
    parts = manifest["splits"]
    fig = canvas("Separate learning from the release decision.",
        "Actual prepared splits • balanced subsets, normalized exact-text deduplication, saved checksums")
    ax = fig.add_axes([.05, .14, .56, .66]); ax.set(xlim=(0, 10), ylim=(0, 7)); ax.axis("off")
    box(ax, .15, 5.4, 3.25, 1.15, "Official train split", "AG News", BLUE)
    box(ax, .15, .6, 3.25, 1.15, "Official test split", "AG News", GOLD)
    for name, y, title, purpose, color in [
        ("train", 5.4, "Training", "Fit LoRA + heads", HOT),
        ("validation", 3.0, "Validation", "Select checkpoint", BLUE),
        ("test", .6, "Locked test", "One final evaluation", TEAL)]:
        box(ax, 5.7, y, 4.0, 1.15, f"{title} · {parts[name]['count']:,}", purpose, color)
    arrow(ax, (3.4, 5.975), (5.7, 5.975))
    arrow(ax, (3.4, 5.6), (5.7, 3.8), connectionstyle="arc3,rad=.12")
    arrow(ax, (3.4, 1.175), (5.7, 1.175))
    ax.text(4.95, 2.45, "No test-to-training arrow", color=TEAL, fontsize=12, ha="center")
    ax2 = fig.add_axes([.70, .24, .24, .48])
    labels = manifest["labels"]
    counts = np.array([parts[name]["class_counts"] for name in ["train", "validation", "test"]])
    bottoms = np.zeros(3)
    for j, color in enumerate([BLUE, TEAL, GOLD, HOT]):
        ax2.bar([0, 1, 2], counts[:, j], bottom=bottoms, color=color, width=.65, label=labels[j])
        bottoms += counts[:, j]
    ax2.set_xticks([0, 1, 2], ["Train", "Validation", "Test"], fontsize=10)
    ax2.set_ylabel("Examples"); ax2.set_ylim(0, max(bottoms)*1.18)
    for x, count in enumerate(bottoms):
        ax2.text(x, count+70, f"{int(count):,}", ha="center", weight="bold")
    ax2.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=2,
               frameon=False, labelcolor=FG, fontsize=10)
    clean(ax2)
    save(fig, root, "02_splits", "Measured counts from artifacts/manifest.json • exact duplicates are excluded; near-duplicate and time/source leakage need additional checks.")


def training(root):
    report = read(root, "results/training.json")
    fig = canvas("Freeze the large weights. Learn a small update.",
        "LoRA concept at an attention projection • measured parameter budget includes the trained classification heads")
    ax = fig.add_axes([.045, .16, .59, .64]); ax.set(xlim=(0, 11), ylim=(0, 6)); ax.axis("off")
    ax.text(.25, 3.1, "x", fontsize=24, ha="center")
    box(ax, 2.1, 3.85, 5.8, 1.2, "Frozen base W", "Original projection weights", BLUE)
    box(ax, 2.1, 1.2, 2.6, 1.25, "A", "Down-project to rank 8", HOT)
    box(ax, 5.3, 1.2, 2.6, 1.25, "B", "Up-project", HOT)
    for start, end in [((.7,3.2),(2.1,4.45)),((.7,3.0),(2.1,1.825)),
                       ((4.7,1.825),(5.3,1.825)),((7.9,4.45),(9.25,3.35)),
                       ((7.9,1.825),(9.25,2.9)),((9.9,3.1),(10.75,3.1))]:
        arrow(ax,start,end)
    ax.text(9.55,3.1,"+",fontsize=28,ha="center",va="center",color=TEAL)
    ax.text(10.8,3.1,"y",fontsize=24,ha="center",va="center")
    ax.text(5,.35,"y = Wx + (α/r) BAx",ha="center",fontsize=16,color=TEAL)
    trainable, total = report["trainable_parameters"], report["total_parameters"]
    fraction = trainable / total
    ax2 = fig.add_axes([.70,.32,.23,.38]); ax2.set_facecolor(BG)
    ax2.pie([fraction,1-fraction],colors=[HOT,"#2C3C58"],startangle=90,counterclock=False,
        wedgeprops={"width":.22,"edgecolor":BG})
    ax2.text(0,.1,f"{100*fraction:.2f}%",ha="center",va="center",fontsize=27,weight="bold",color=HOT)
    ax2.text(0,-.22,"trainable",ha="center",fontsize=12,color=MUTED)
    fig.text(.815,.23,f"{trainable:,} / {total:,} parameters",ha="center",fontsize=12)
    fig.text(.815,.17,"LoRA adapters + classification heads",ha="center",fontsize=10,color=MUTED)
    save(fig,root,"03_lora_budget","Conceptual LoRA path; counts from this run • frozen does not mean free: base weights still require memory and computation.")

    history=report["history"]; epochs=[x["epoch"] for x in history]
    losses=[x["mean_loss"] for x in history]; scores=[x["validation"]["macro_f1"] for x in history]
    best=int(np.argmax(scores))
    fig=canvas("Let validation choose the checkpoint.",
        "Measured epoch history • training loss and validation quality are different signals")
    axes=fig.subplots(1,2); fig.subplots_adjust(left=.08,right=.95,bottom=.18,top=.76,wspace=.24)
    for ax, values, color, title, ylabel in zip(axes,[losses,scores],[HOT,TEAL],
        ["Training loss","Validation macro F1"],["Mean cross-entropy loss","Macro F1"]):
        ax.plot(epochs,values,"o-",color=color,lw=3,markersize=9)
        ax.set(title=title,xlabel="Epoch",ylabel=ylabel,xticks=epochs)
        clean(ax)
        for x,y in zip(epochs,values):
            ax.annotate(f"{y:.3f}",(x,y),xytext=(0,13),textcoords="offset points",ha="center",color=color)
    axes[0].set_ylim(0,max(losses)*1.25)
    axes[1].set_ylim(max(0,min(scores)-.045),min(1,max(scores)+.045))
    axes[1].axvline(epochs[best],color=TEAL,alpha=.35,ls="--")
    axes[1].scatter([epochs[best]],[scores[best]],s=280,facecolors="none",edgecolors=TEAL,lw=2)
    axes[1].text(.04,.08,f"Selected: epoch {epochs[best]}",transform=axes[1].transAxes,color=TEAL,weight="bold")
    save(fig,root,"04_training_progress","Measured from results/training.json • the validation F1 axis is zoomed for comparison; heldout test scores do not select this checkpoint.")


def export(root):
    report=read(root,"results/export.json")
    keys=["adapter","merged","onnx_fp32","onnx_int8"]
    values=[sum(v["bytes"] for v in report["artifacts"][key].values())/1024**2 for key in keys]
    fig=canvas("The adapter is small. The runtime must stand alone.",
        "Measured packaged size • adapter → merged model → ONNX FP32 → ONNX INT8")
    ax=fig.add_axes([.24,.20,.69,.56])
    bars=ax.barh(range(4),values,color=[HOT,BLUE,GOLD,TEAL],height=.55)
    ax.set_yticks(range(4),["LoRA adapter*","Merged model","ONNX FP32","ONNX INT8"])
    ax.invert_yaxis(); ax.set_xlim(0,max(values)*1.2); ax.set_xlabel("Total packaged files (MiB)")
    for bar,value in zip(bars,values):
        ax.text(value+max(values)*.025,bar.get_y()+bar.get_height()/2,f"{value:.2f}",va="center",weight="bold")
    clean(ax,grid="x")
    save(fig,root,"05_export_footprint","*The adapter also requires the base model • totals include tokenizer/config files; the later benchmark measures weight-file bytes only, not RAM.")


def evaluation(root):
    report=read(root,"results/evaluation.json"); policy=read(root,"release_policy.json")
    candidate=report["models"]["int8"]; labels=list(candidate["per_class"])
    cm=np.asarray(candidate["confusion_matrix"])
    fig=canvas("A strong average can hide a weak class.",
        f"Measured INT8 heldout predictions • {candidate['n']:,} examples • every class is checked")
    ax=fig.add_axes([.08,.20,.39,.56])
    ax.imshow(cm,cmap="Blues",vmin=0,vmax=max(cm.sum(axis=1)))
    ax.set_xticks(range(4),labels); ax.set_yticks(range(4),labels)
    ax.set(xlabel="Predicted category",ylabel="True category",title="Confusion matrix · counts")
    for i in range(4):
        for j in range(4):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",weight="bold",fontsize=15,
                color="white" if cm[i,j]>max(cm.sum(axis=1))*.5 else "#132A42")
    ax2=fig.add_axes([.61,.20,.33,.56]); recalls=[candidate["per_class"][label]["recall"] for label in labels]
    bars=ax2.barh(labels,recalls,color=[TEAL if v>=policy["min_class_recall"] else RED for v in recalls],height=.56)
    ax2.invert_yaxis(); ax2.set(xlim=(0,1.05),xlabel="Recall",title="Recall by class")
    ax2.xaxis.set_major_formatter(PercentFormatter(1))
    ax2.axvline(policy["min_class_recall"],color=HOT,ls="--",lw=2)
    ax2.text(policy["min_class_recall"],1.015,f"Floor {policy['min_class_recall']:.0%}",
        transform=ax2.get_xaxis_transform(),ha="center",va="bottom",color=HOT,fontsize=10)
    for bar,value in zip(bars,recalls):
        ax2.text(value-.025,bar.get_y()+bar.get_height()/2,f"{value:.1%}",va="center",ha="right",color=BG,weight="bold")
    clean(ax2,grid="x")
    save(fig,root,"06_evaluation","Measured from results/evaluation.json • diagonal = correct; off-diagonal = mistakes. The recall floor was frozen before test evaluation.")


def benchmark(root):
    bench=read(root,"results/benchmark.json"); ev=read(root,"results/evaluation.json"); policy=read(root,"release_policy.json")
    rows=[bench["models"][key] for key in BACKENDS]
    fig=canvas("Compare the distribution, not one lucky request.",
        f"Measured CPU inference • {bench['threads']} threads • batch {bench['batch_size']} • {bench['sequence_length']} tokens • {bench['samples']} requests")
    axes=fig.subplots(1,2); fig.subplots_adjust(left=.08,right=.96,bottom=.22,top=.76,wspace=.26)
    for row,name,color in zip(rows,NAMES,COLORS):
        ordered=np.sort(row["latencies_ms"])
        axes[0].step(ordered,np.arange(1,len(ordered)+1)/len(ordered),where="post",label=name,color=color,lw=2.6)
    axes[0].set(xlabel="Request latency (ms)",ylabel="Cumulative share of requests",title="All measured requests · empirical CDF")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1)); axes[0].axhline(.95,color=MUTED,ls=":",lw=1)
    axes[0].legend(loc="lower right",frameon=False,labelcolor=FG,fontsize=10); clean(axes[0],grid="both")
    y=np.arange(3)
    axes[1].barh(y-.16,[r["p50_ms"] for r in rows],height=.28,color=COLORS,alpha=.48,label="p50")
    axes[1].barh(y+.16,[r["p95_ms"] for r in rows],height=.28,color=COLORS,label="p95")
    axes[1].set_yticks(y,NAMES); axes[1].invert_yaxis(); axes[1].set(xlabel="Latency (ms)",title="Typical request and tail latency")
    maximum=max(policy["max_cpu_p95_ms"],max(r["p95_ms"] for r in rows))
    axes[1].set_xlim(0,maximum*1.25)
    axes[1].axvline(policy["max_cpu_p95_ms"],color=HOT,ls="--",lw=1.5)
    axes[1].text(policy["max_cpu_p95_ms"],2.82,f"INT8 gate budget: {policy['max_cpu_p95_ms']:g} ms",ha="center",color=HOT,fontsize=10)
    for i,row in enumerate(rows):
        for offset,key in [(-.16,"p50_ms"),(.16,"p95_ms")]:
            axes[1].text(row[key]+maximum*.025,i+offset,f"{row[key]:.1f}",va="center",fontsize=10)
    axes[1].legend(loc="lower right",frameon=False,labelcolor=FG,fontsize=10); clean(axes[1],grid="x")
    save(fig,root,"07_latency","Measured from raw latency samples • tokenization + forward after warmup; excludes loading, network and queueing. CDF = fraction at or below a latency.")

    fig=canvas("Keep the quality. Measure the footprint.",
        "Measured comparison on the same locked test • weight bytes are disk size, not peak memory")
    axes=fig.subplots(1,2); fig.subplots_adjust(left=.08,right=.96,bottom=.24,top=.76,wspace=.28)
    weights=[r["weights_bytes"]/1024**2 for r in rows]
    bars=axes[0].bar(range(3),weights,color=COLORS,width=.62)
    axes[0].set(xticks=range(3),xticklabels=NAMES,ylabel="Weight files (MiB)",title="Standalone weight footprint",ylim=(0,max(weights)*1.22))
    axes[0].tick_params(axis="x",labelsize=10)
    for bar,value in zip(bars,weights):
        axes[0].text(bar.get_x()+bar.get_width()/2,value+max(weights)*.035,f"{value:.1f}",ha="center",weight="bold")
    reduction=1-weights[2]/weights[0]
    axes[0].text(.5,-.24,f"INT8 vs PyTorch: {reduction:.1%} fewer weight-file bytes",transform=axes[0].transAxes,
        ha="center",color=TEAL,weight="bold",fontsize=11); clean(axes[0])
    for offset,key,title,alpha in [(-.17,"accuracy","Accuracy",.45),(.17,"macro_f1","Macro F1",1.0)]:
        values=[ev["models"][k][key] for k in BACKENDS]
        axes[1].bar(np.arange(3)+offset,values,width=.3,color=COLORS,alpha=alpha,label=title)
        for x,value in zip(np.arange(3)+offset,values):
            axes[1].text(x,value+.018,f"{value:.3f}",ha="center",fontsize=9)
    axes[1].axhline(ev["majority_class_baseline"]["accuracy"],color=MUTED,ls=":",label="Majority baseline accuracy")
    axes[1].set(xticks=range(3),xticklabels=NAMES,ylim=(0,1.03),ylabel="Score",title="Heldout quality · full 0–1 scale")
    axes[1].tick_params(axis="x",labelsize=10)
    axes[1].legend(loc="lower center",frameon=False,labelcolor=FG,fontsize=9); clean(axes[1])
    save(fig,root,"08_quality_size","Measured from evaluation + benchmark reports • a small accuracy difference is not evidence that quantization generally improves quality.")


def gates(root):
    real=read(root,"results/gate.json"); bad=read(root,"results/gate_deliberate_failure.json")
    keys=list(real["checks"])
    names={"same_heldout_split":"Same heldout split","exact_runtime_artifacts":"Exact artifact fingerprints",
        "complete_class_report":"Every class reported","benchmark_protocol":"Benchmark protocol",
        "minimum_test_size":"Minimum test size","accuracy":"Accuracy floor","macro_f1":"Macro F1 floor",
        "every_class_recall":"Every-class recall floor","quantization_f1_regression":"Quantization F1 regression",
        "cpu_p95_latency":"CPU p95 budget","weight_size":"Weight-file budget"}
    fig=canvas("Make the release decision visible.",
        "Actual teaching gate vs a deliberate simulated Business-recall failure • every required check must pass",height=9)
    ax=fig.add_axes([.06,.12,.88,.66]); ax.set(xlim=(0,14),ylim=(-.6,len(keys)+1.4)); ax.axis("off")
    for x,report,title in [(8,real,"Measured candidate"),(11.5,bad,"Injected regression")]:
        status="PASS" if report["passed"] else "BLOCK"
        ax.text(x,len(keys)+1.1,title,ha="center",fontsize=13,weight="bold")
        ax.text(x,len(keys)+.45,status,ha="center",color=TEAL if report["passed"] else RED,fontsize=17,weight="bold")
    for i,key in enumerate(keys):
        y=len(keys)-1-i
        ax.text(.2,y,names.get(key,key),fontsize=12,va="center")
        ax.plot([.15,13.3],[y-.43,y-.43],color="#2B3B55",lw=.7)
        for x,report in [(8,real),(11.5,bad)]:
            okay=report["checks"][key]; color=TEAL if okay else RED
            ax.scatter([x-.75],[y],s=80,color=color,marker="o" if okay else "X")
            ax.text(x-.45,y,"PASS" if okay else "FAIL",va="center",fontsize=11,color=color,weight="bold")
    save(fig,root,"09_release_gate","The injected report changes Business recall to 0.10 without changing the authentic report • a real BLOCK is a valid outcome; it is never relabeled as PASS.")


def inference(root):
    report=read(root,"results/inference.json"); predictions=report["predictions"]
    # The notebook supplies four examples; show up to four without changing inference.
    chosen=predictions[:4]
    fig=canvas("Inspect the scores behind each prediction.",
        "Actual INT8 inference examples • these scores are not calibrated confidence",height=9)
    axes=fig.subplots(2,2); fig.subplots_adjust(left=.11,right=.96,bottom=.15,top=.76,wspace=.30,hspace=.63)
    import textwrap
    for ax,item in zip(axes.flat,chosen):
        labels=list(item["scores"]); values=list(item["scores"].values())
        ax.barh(labels,values,color=[TEAL if key==item["label"] else "#4A638B" for key in labels],height=.55)
        ax.invert_yaxis(); ax.set_xlim(0,1.04); ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(textwrap.fill(item["text"],54),loc="left",fontsize=10,pad=12)
        for j,value in enumerate(values):
            ax.text(max(.02,value-.02),j,f"{value:.1%}",va="center",ha="right" if value>.18 else "left",
                color=BG if value>.18 else FG,fontsize=9)
        clean(ax,grid="x")
    for ax in list(axes.flat)[len(chosen):]: ax.axis("off")
    save(fig,root,"10_inference","Measured from results/inference.json • four-way classification has no unknown class; text is truncated to 128 tokens. Inference does not authorize release.")


STAGES={"workflow":workflow,"splits":splits,"training":training,"export":export,
        "evaluation":evaluation,"benchmark":benchmark,"gates":gates,"inference":inference}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=[*STAGES,"all"])
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    stages=STAGES.values() if args.stage=="all" else [STAGES[args.stage]]
    for stage in stages: stage(args.root)
    destination=args.root/"figures"/"provenance.json"
    destination.write_text(json.dumps({"matplotlib":matplotlib.__version__,
        "data_source":"This run's artifacts/manifest.json, results/*.json and frozen release_policy.json",
        "conceptual_diagrams":["01_workflow","03_lora_budget (left panel only)"],
        "png_files":sorted(p.name for p in destination.parent.glob("*.png"))},indent=2)+"\n",encoding="utf-8")


if __name__=="__main__": main()
