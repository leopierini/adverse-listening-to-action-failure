"""Generate publication-quality figures for the thesis.

Produces:
  fig1_dose_response.png       — TSA vs SNR per degradation, one panel per ASR
  fig2_absorption_heatmap.png  — GAIN(critical−function) heatmap, ASR × degradation × SNR
  fig3_per_intent.png          — TSA % heatmap by intent × degradation, Whisper-Turbo
  fig4_h1_forest.png           — bootstrap CI forest plot for GAIN(crit−func) across all cells

Inputs:
  - results/analysis/*_pf.jsonl
  - results/analysis/absorption_ci_final.csv

All figures saved to results/figures/.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# v2 kept only so figures regenerate against legacy pilot JSONLs; current runs use v3.
ASR_ORDER = ["whisper-base-mlx", "parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v3",
             "whisper-large-v3-turbo"]
ASR_LABELS = {
    "whisper-base-mlx": "Whisper Base (74M)",
    "parakeet-tdt-0.6b-v2": "Parakeet TDT v2 (0.6B)",
    "parakeet-tdt-0.6b-v3": "Parakeet TDT v3 (0.6B)",
    "whisper-large-v3-turbo": "Whisper Large-v3 Turbo (809M)",
}
DEG_ORDER = ["noise", "reverb", "farfield", "codec", "clipping"]
DEG_LABELS = {
    "noise": "Additive noise",
    "reverb": "Reverberation",
    "farfield": "Far-field",
    "codec": "Codec (Opus)",
    "clipping": "Clipping",
}
SNR_LEVELS = [None, 20, 10, 0]


def load_cells(jsonl_dir="results/analysis"):
    cells = defaultdict(list)
    for p in Path(jsonl_dir).glob("*_pf.jsonl"):
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                key = (
                    (r.get("asr_model") or "?").split("/")[-1],
                    r.get("degradation") or "clean",
                    r.get("snr_db"),
                )
                cells[key].append(r)
    return cells


def cell_mean(rows, k):
    vals = [r.get(k) for r in rows if r.get(k) is not None]
    if not vals:
        return None
    if k == "wer":
        vals = [min(v, 1.0) for v in vals]
    return float(np.mean(vals))


def fig1_dose_response(cells):
    """Three panels (one per ASR) of TSA vs SNR for each degradation."""
    asrs = [a for a in ASR_ORDER if any(k[0] == a for k in cells)]
    fig, axes = plt.subplots(1, len(asrs), figsize=(5 * len(asrs), 4.2), sharey=True)
    if len(asrs) == 1:
        axes = [axes]

    snr_x = [20, 10, 0]  # numeric only; clean shown as separate marker
    snr_x_full = ["clean"] + [f"{s}" for s in snr_x]
    x_positions = list(range(len(snr_x_full)))

    palette = plt.cm.tab10(np.linspace(0, 1, len(DEG_ORDER)))

    for ax, asr in zip(axes, asrs):
        # Plot clean TSA as a horizontal reference line
        clean_rows = cells.get((asr, "clean", None), [])
        clean_tsa = cell_mean(clean_rows, "tsa") if clean_rows else None
        if clean_tsa is not None:
            ax.axhline(100 * clean_tsa, color="gray", linestyle=":", linewidth=1, alpha=0.5)

        for i, deg in enumerate(DEG_ORDER):
            ys, mask = [], []
            for snr in snr_x:
                rows = cells.get((asr, deg, snr), [])
                t = cell_mean(rows, "tsa")
                ys.append(100 * t if t is not None else np.nan)
                mask.append(t is not None)
            xs_local = [1, 2, 3]  # positions for 20, 10, 0
            ax.plot(xs_local, ys, marker="o", color=palette[i], label=DEG_LABELS[deg], linewidth=1.8)

        # Mark clean point
        if clean_tsa is not None:
            ax.scatter([0], [100 * clean_tsa], color="black", marker="D", s=40, zorder=5, label="clean")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(snr_x_full)
        ax.set_xlabel("SNR (dB)")
        ax.set_title(ASR_LABELS.get(asr, asr))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("TSA (%)")
    axes[-1].legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig.suptitle("Tool Selection Accuracy across degradation dose-response",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / "fig1_dose_response.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


def fig2_absorption_heatmap(ci_csv):
    """Heatmap of GAIN(crit−func) median across ASR × degradation × SNR."""
    rows = []
    with open(ci_csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    asrs = [a for a in ASR_ORDER if any(r["asr"] == a for r in rows)]

    # Build a matrix per ASR: degradations × SNRs (incl clean column)
    snr_labels = ["clean", "20", "10", "0"]
    fig, axes = plt.subplots(1, len(asrs), figsize=(4 * len(asrs), 3.6), sharey=True)
    if len(asrs) == 1:
        axes = [axes]
    for ax, asr in zip(axes, asrs):
        mat = np.full((len(DEG_ORDER), len(snr_labels)), np.nan)
        for r in rows:
            if r["asr"] != asr:
                continue
            deg = r["degradation"]
            snr_v = r["snr_db"]
            snr_label = "clean" if deg == "clean" else snr_v
            if snr_label not in snr_labels:
                continue
            col_i = snr_labels.index(snr_label)
            # For clean rows, put them in a synthetic "row" — we'll show only deg rows for now
            if deg == "clean":
                # broadcast to all rows for context
                for j in range(len(DEG_ORDER)):
                    if np.isnan(mat[j, 0]):
                        mat[j, 0] = 100 * float(r["gain_crit_minus_func_median"])
            elif deg in DEG_ORDER:
                row_i = DEG_ORDER.index(deg)
                mat[row_i, col_i] = 100 * float(r["gain_crit_minus_func_median"])

        im = ax.imshow(mat, cmap="RdYlGn", vmin=-30, vmax=30, aspect="auto")
        ax.set_xticks(range(len(snr_labels)))
        ax.set_xticklabels(snr_labels)
        ax.set_yticks(range(len(DEG_ORDER)))
        ax.set_yticklabels([DEG_LABELS[d] for d in DEG_ORDER])
        ax.set_title(ASR_LABELS.get(asr, asr), fontsize=10)
        # Annotate cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                            color="black" if abs(v) < 18 else "white", fontsize=9)
        ax.set_xlabel("SNR (dB)")

    fig.suptitle("Absorption gain: GAIN(critical − function) [%], 95% CI median",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=axes, label="GAIN(crit − func) (%)", shrink=0.7)
    out = FIG_DIR / "fig2_absorption_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


def fig3_per_intent(cells, target_asr="whisper-large-v3-turbo"):
    """Heatmap of TSA% per intent × condition for Whisper-Turbo."""
    intents = sorted(set(r.get("gold_intent", "?")
                         for k, rows in cells.items() if k[0] == target_asr
                         for r in rows))
    conditions = ["clean"] + DEG_ORDER
    mat = np.full((len(intents), len(conditions)), np.nan)
    for i, intent in enumerate(intents):
        for j, cond in enumerate(conditions):
            ns, tsa = [], []
            for k, rows in cells.items():
                if k[0] != target_asr or k[1] != cond:
                    continue
                # For corrupted conditions: average across SNRs (or use only snr=10 for cleaner view)
                if cond != "clean" and k[2] != 10:
                    continue
                for r in rows:
                    if r.get("gold_intent") == intent and r.get("tsa") is not None:
                        tsa.append(r["tsa"])
            if len(tsa) >= 5:
                mat[i, j] = 100 * np.mean(tsa)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=20, vmax=90, aspect="auto")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(["clean"] + [DEG_LABELS[d] for d in DEG_ORDER], rotation=20, ha="right")
    ax.set_yticks(range(len(intents)))
    ax.set_yticklabels(intents)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="black" if 35 < v < 75 else "white", fontsize=8)
    ax.set_title(f"TSA (%) per intent × condition — {ASR_LABELS.get(target_asr, target_asr)} @ SNR=10 dB",
                 fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="TSA (%)", shrink=0.8)
    fig.tight_layout()
    out = FIG_DIR / "fig3_per_intent.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


def fig4_h1_forest(ci_csv):
    """Forest plot of GAIN(crit−func) median + 95% CI across all cells."""
    rows = []
    with open(ci_csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    # Sort: by ASR, then by degradation, then by SNR (descending — clean first)
    def keyf(r):
        asr_idx = ASR_ORDER.index(r["asr"]) if r["asr"] in ASR_ORDER else 99
        deg_idx = 0 if r["degradation"] == "clean" else (DEG_ORDER.index(r["degradation"]) + 1
                                                          if r["degradation"] in DEG_ORDER else 99)
        snr_v = r["snr_db"]
        snr_idx = 0 if snr_v == "" else -int(snr_v)  # negative so 20→-20 comes first
        return (asr_idx, deg_idx, snr_idx)

    rows = sorted(rows, key=keyf)

    medians = [100 * float(r["gain_crit_minus_func_median"]) for r in rows]
    los = [100 * float(r["gain_crit_minus_func_lo"]) for r in rows]
    his = [100 * float(r["gain_crit_minus_func_hi"]) for r in rows]
    labels = [
        f"{r['asr']} · {r['degradation']}{(' snr=' + r['snr_db']) if r['snr_db'] else ''}"
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(rows))))
    y = np.arange(len(rows))
    colors = ["#1b7837" if l > 0 else ("#762a83" if h < 0 else "#9e9e9e")
              for l, h in zip(los, his)]
    ax.errorbar(medians, y, xerr=[np.array(medians) - np.array(los),
                                    np.array(his) - np.array(medians)],
                fmt="o", color="black", ecolor="gray", capsize=2, markersize=4)
    for i, (m, c) in enumerate(zip(medians, colors)):
        ax.scatter([m], [i], color=c, s=40, zorder=5)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("GAIN(critical − function) (%)")
    ax.set_title("H1 forest plot — bootstrap 95% CI of GAIN(crit − func) per cell\n"
                 "(green = strictly rejects H1, purple = supports H1, gray = borderline)",
                 fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    out = FIG_DIR / "fig4_h1_forest.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


def main():
    cells = load_cells()
    print(f"Loaded {sum(len(v) for v in cells.values())} rows across {len(cells)} cells.")
    fig1_dose_response(cells)
    fig2_absorption_heatmap("results/analysis/absorption_ci_final.csv")
    fig3_per_intent(cells)
    fig4_h1_forest("results/analysis/absorption_ci_final.csv")
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
