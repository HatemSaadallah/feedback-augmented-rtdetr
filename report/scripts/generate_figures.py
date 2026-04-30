"""Generate all matplotlib figures for the LaTeX report.

Reads source data from v1_results/ and v2_results/ (no HPC dependency).
Writes PDF figures to ../figures/.

Run with the rtdetr conda env's python:
    /home/hatem/miniconda3/envs/rtdetr/bin/python generate_figures.py
"""

import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/hatem/Desktop/rt-detr")
V1 = ROOT / "v1_results"
V2 = ROOT / "v2_results"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_V1 = "#888888"
C_V2 = "#1f6feb"
C_BASE = "#d62728"
C_HL = "#2ca02c"


def load_log_aps(path):
    """Parse log.txt with one JSON per line, return list of (epoch, AP, AP_S, AP_M, AP_L)."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ap = d["test_coco_eval_bbox"]
                rows.append({
                    "epoch": i,
                    "AP": ap[0] * 100,
                    "AP_S": ap[3] * 100,
                    "AP_M": ap[4] * 100,
                    "AP_L": ap[5] * 100,
                })
            except Exception:
                pass
    return rows


def load_gate_trajectory(out_glob):
    """Grep 'feedback: ... gate=N.NN' lines from chain .out files; return ordered list."""
    files = sorted(out_glob)
    gates = []
    for fp in files:
        with open(fp, errors="replace") as f:
            for line in f:
                m = re.search(r"feedback:.*gate=([\d.]+)", line)
                if m:
                    gates.append(float(m.group(1)))
    return gates


# ---------- Figure 1: Learning curves AP_S(epoch) v1 vs v2 ----------
def fig_learning_curves():
    v1 = load_log_aps(V1 / "logs" / "log.txt")
    v2 = load_log_aps(V2 / "logs" / "log.txt")

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot([r["epoch"] for r in v1], [r["AP_S"] for r in v1],
            "o-", color=C_V1, linewidth=1.5, markersize=4, label="v1 feedback (no floor, all-levels)")
    ax.plot([r["epoch"] for r in v2], [r["AP_S"] for r in v2],
            "o-", color=C_V2, linewidth=2.0, markersize=5, label="v2 feedback (floor=0.1, P2/P3-only)")
    ax.axhline(34.7, color=C_BASE, linestyle="--", linewidth=1.0,
               label="RT-DETR paper baseline (34.7)")
    ax.axhline(33.91, color=C_V1, linestyle=":", linewidth=1.0,
               label="v1 final (33.91)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\mathrm{AP}_S$ on COCO val2017 @ 640")
    ax.set_title("Per-epoch small-object AP, v1 vs v2")
    ax.set_xticks(range(0, 13))
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.pdf")
    plt.close(fig)
    print(f"  wrote learning_curves.pdf  (v1 final={v1[-1]['AP_S']:.2f}, v2 final={v2[-1]['AP_S']:.2f})")


# ---------- Figure 2: Gate evolution ----------
def fig_gate_evolution():
    # v2 gate trajectory from chain .out files
    v2_outs = list((V2 / "logs").glob("p2_v2_chain_*.out"))
    gates_v2 = load_gate_trajectory(v2_outs)
    # The gate is logged once per epoch's eval pass; v2 has 13 entries
    # (epoch indices 0..12, possibly with the warm-up first epoch held)
    if len(gates_v2) == 0:
        print("  WARN: no gate trajectory found, skipping gate_evolution.pdf")
        return

    # Pad / align to 13 epochs (v2 had warmup hold at first epoch)
    epochs = list(range(len(gates_v2)))

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(epochs, gates_v2, "o-", color=C_V2, linewidth=2.0, markersize=5,
            label="v2 effective gate")
    ax.axhline(0.10, color=C_BASE, linestyle="--", linewidth=1.0,
               label="floor = 0.10 (cannot collapse below)")
    ax.axhline(0.55, color="#888888", linestyle=":", linewidth=1.0,
               label=r"v2 init = 0.55 ($\alpha=0$, floor=0.1)")
    ax.axhline(0.12, color=C_V1, linestyle="-.", linewidth=1.0,
               label="v1 inferred final ≈ 0.12")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Effective gate $g_\mathrm{eff}$")
    ax.set_title("Gate trajectory: v2 stays meaningful, v1 collapses")
    ax.set_ylim(0.0, 0.65)
    ax.set_xticks(range(0, max(13, len(epochs))))
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "gate_evolution.pdf")
    plt.close(fig)
    print(f"  wrote gate_evolution.pdf  ({len(gates_v2)} gate readings, range "
          f"[{min(gates_v2):.3f}, {max(gates_v2):.3f}])")


# ---------- Figure 3: Ablation bar chart ----------
def fig_ablation_bars():
    on = json.load(open(V2 / "ablations" / "ablation_v2_feedback_on_640.json"))
    off = json.load(open(V2 / "ablations" / "ablation_v2_feedback_off_640.json"))

    metrics = ["AP", "AP_S", "AP_M", "AP_L"]
    deltas = [(on[m] - off[m]) * 100 for m in metrics]
    on_vals = [on[m] * 100 for m in metrics]
    off_vals = [off[m] * 100 for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.4),
                                    gridspec_kw={"width_ratios": [1.6, 1.0]})

    x = np.arange(len(metrics))
    w = 0.35
    ax1.bar(x - w / 2, off_vals, w, color="#bbbbbb", label="feedback OFF")
    ax1.bar(x + w / 2, on_vals, w, color=C_V2, label="feedback ON")
    ax1.set_xticks(x)
    ax1.set_xticklabels([r"$\mathrm{AP}$", r"$\mathrm{AP}_S$",
                         r"$\mathrm{AP}_M$", r"$\mathrm{AP}_L$"])
    ax1.set_ylabel("COCO val2017 @ 640 (%)")
    ax1.set_title("v2 same-checkpoint ablation")
    ax1.legend(loc="lower right", framealpha=0.95)
    for i, (o, n) in enumerate(zip(off_vals, on_vals)):
        ax1.text(i - w / 2, o + 0.4, f"{o:.2f}", ha="center", fontsize=8)
        ax1.text(i + w / 2, n + 0.4, f"{n:.2f}", ha="center", fontsize=8)
    ax1.set_ylim(0, max(on_vals) * 1.15)
    ax1.grid(True, alpha=0.3, axis="y")

    colors = [C_HL if d > 0 else C_BASE for d in deltas]
    ax2.bar(x, deltas, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r"$\Delta\mathrm{AP}$", r"$\Delta\mathrm{AP}_S$",
                         r"$\Delta\mathrm{AP}_M$", r"$\Delta\mathrm{AP}_L$"])
    ax2.set_ylabel(r"$\Delta$ (ON $-$ OFF), pp")
    ax2.set_title("Causal contribution")
    for i, d in enumerate(deltas):
        ax2.text(i, d + 0.05, f"{d:+.2f}", ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.set_ylim(0, max(deltas) * 1.4)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "ablation_bars.pdf")
    plt.close(fig)
    print(f"  wrote ablation_bars.pdf  (Δ AP_S = {deltas[1]:+.2f})")


# ---------- Figure 4: Gate floor reparameterization curve ----------
def fig_gate_reparam():
    alpha = np.linspace(-6, 6, 400)
    sig = 1.0 / (1.0 + np.exp(-alpha))
    floors = [0.0, 0.1, 0.3]

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for floor, ls in zip(floors, ["--", "-", "-."]):
        g_eff = floor + (1.0 - floor) * sig
        label = f"floor = {floor:.1f}"
        if floor == 0.0:
            label += " (v1, plain sigmoid)"
        elif floor == 0.1:
            label += " (v2)"
        ax.plot(alpha, g_eff, ls, linewidth=2.0, label=label)

    ax.axvline(-2, color="#888888", linestyle=":", linewidth=1.0)
    ax.axvline(0, color=C_V2, linestyle=":", linewidth=1.0)
    ax.text(-2, 0.02, r"v1 init $\alpha=-2$", color="#444444", fontsize=8, ha="center")
    ax.text(0, 0.02, r"v2 init $\alpha=0$", color=C_V2, fontsize=8, ha="center")
    ax.set_xlabel(r"Raw gate parameter $\alpha$")
    ax.set_ylabel(r"Effective gate $g_\mathrm{eff} = \mathrm{floor} + (1-\mathrm{floor})\,\sigma(\alpha)$")
    ax.set_title("Gate reparameterization: floor prevents collapse to zero")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "gate_reparam.pdf")
    plt.close(fig)
    print("  wrote gate_reparam.pdf")


# ---------- Figure 5: AP/AP_S/AP_M/AP_L progression with both versions ----------
def fig_full_metrics():
    v2 = load_log_aps(V2 / "logs" / "log.txt")
    epochs = [r["epoch"] for r in v2]

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for metric, color, marker in [
        ("AP", "#444444", "o"),
        ("AP_S", C_V2, "s"),
        ("AP_M", "#ff7f0e", "^"),
        ("AP_L", "#2ca02c", "d"),
    ]:
        vals = [r[metric] for r in v2]
        ax.plot(epochs, vals, marker=marker, linestyle="-", linewidth=1.4,
                markersize=4, color=color, label=metric.replace("_", r"$_") + r"$" if "_" in metric else metric)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("COCO val2017 @ 640 (%)")
    ax.set_title("v2 full metric progression")
    ax.set_xticks(range(0, 13))
    ax.legend(loc="lower right", ncol=2, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "full_metrics_v2.pdf")
    plt.close(fig)
    print("  wrote full_metrics_v2.pdf")


def main():
    print("Generating figures...")
    fig_learning_curves()
    fig_gate_evolution()
    fig_ablation_bars()
    fig_gate_reparam()
    fig_full_metrics()
    print(f"\nDone. Figures in {OUT}")
    for f in sorted(OUT.glob("*.pdf")):
        print(f"  {f.name}  {f.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
