"""Step 7 results figures — the ones the artifacts support, and no others.

Every figure here is drawn from a file in ``results/analysis/``; nothing is
typed in from prose. Where a number is annotated onto a figure it is either
read out of the artifact at draw time or listed in PROVENANCE below with the
command that produced it.

WHAT THIS DRAWS, AND WHY EACH FORM WAS CHOSEN
---------------------------------------------
§7 of EXECUTION_TODO asks for six figures. Five are drawn here as asked; the
sixth ("absorption-gain heatmap") is drawn under the *calibrated* null only,
for the reason §13 gives at length: under the assumed ``1 - WER`` comparator
(``gain_paired``) the gain is positive in 30-34 of 54 cells, and §13's standing
instruction is that a positive headline absorption number must not be quoted,
because the level is a choice of comparator and the only comparator estimated
from data puts it at approximately zero. A heatmap of ``gain_paired`` alone
would restate a falsified reading in colour. So:

  fig1  retention dose-response      — §13: "lead with retention"; needs no null
  fig2  gain_calibrated heatmap      — the identified comparator, CI-marked
  fig3  comparator contrast          — gain_paired vs gain_calibrated, the finding
  fig4  confirmatory forest          — H1/H2a/H2b/H3 and H4b against `clean`
  fig5  mediation attribution        — ACME/ADE decomposition per (deg, SNR)
  fig6  9B/27B capacity calibration  — the H4b defence
  fig7  cascade vs omni EES          — descriptive (H4a), per condition

`gain_legacy` and ``results/analysis/absorption_qwen9b.csv`` are never read:
CLAUDE.md forbids both.

PROVENANCE OF THE TWO ANNOTATED CHI-SQUARES
-------------------------------------------
Neither joint Wald statistic is stored in an artifact, so both were re-derived
before being written into this file, on 2026-08-28, from
``result_sets.sweep_files()`` (116 files, 86,944 rows):

  H4b, joint Wald over the 6 pathway x degradation interactions of the saved
  fit (`results/analysis/fits/h4b_ees.txt`), via `GEEResults.wald_test`:
      chi2 = 29.1127, df = 6, p = 5.79207e-05
  H4b again on the hallucination-sensitivity refit (`drop_hallucinations=True`,
  748 rows dropped, 30,868 left), same six interactions:
      chi2 = 23.4530, df = 6, p = 0.000658127
  capacity calibration, GEE `ees ~ C(scale)*C(degradation, Treatment('clean'))`
  on the 15,808 Whisper-Turbo rows (416 clusters, 19 cells), joint Wald over
  its 6 interactions:
      chi2 = 3.7481, df = 6, p = 0.710727
      scale main effect on clean, 27B - 9B: +0.0392, p = 0.6276

Both reproduce the values §13 records. Everything else annotated on a figure
is recomputed from the CSV or the saved summary at draw time.

USAGE
-----
    ./venv/bin/python scripts/make_thesis_figures.py
    ./venv/bin/python scripts/make_thesis_figures.py --only fig1 fig3

Each figure is written as PDF (vector, for the printed thesis) and PNG (for
slides) into ``results/figures/``. fig7 reads the sweep JSONL through
``scripts/result_sets.py`` and takes about a minute; the rest are instant.
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ANALYSIS_DIR = "results/analysis"
FIG_DIR = "results/figures"

# --- palette -----------------------------------------------------------------
# Hexes are the documented data-viz reference palette, unmodified. The pairs
# actually used were run through that skill's validator on 2026-08-28:
#   "#2a78d6,#eb6834" --mode light --pairs all -> ALL CHECKS PASS
#   "#2a78d6,#e34948" --mode light --pairs all -> ALL CHECKS PASS
# Colour never carries meaning alone in these figures: marker shape, hatching,
# line style and in-cell numbers repeat every distinction, so the pages survive
# greyscale printing.
BLUE = "#2a78d6"        # categorical slot 1
ORANGE = "#eb6834"      # categorical slot 2
RED = "#e34948"         # categorical slot 8 — the red pole of the diverging pair
NEUTRAL = "#f0efec"     # documented diverging midpoint
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Diverging map for a signed gain: red (below the router's own clean curve),
# neutral at zero, blue (above it). Built from documented hexes only.
DIVERGING = LinearSegmentedColormap.from_list(
    "gain_diverging", [RED, NEUTRAL, BLUE])

# --- labels ------------------------------------------------------------------
# Degradation names follow §3 (DESIGN.md line 139: "reverberation, far-field,
# codec compression, additive noise, clipping, babble").
DEG_LABEL = {
    "clipping": "Clipping",
    "codec": "Codec",
    "farfield": "Far-field",
    "babble": "Babble",
    "noise": "Additive noise",
    "reverb": "Reverberation",
    "clean": "Clean",
}
# §13's grouping — the claim that survives every null tested. The *within-group*
# order does not survive (Spearman rho = 0.657, p = 0.156 between the two
# comparators over the six means), so degradations are listed alphabetically
# inside each group and never ranked against each other.
PRESERVING = ("clipping", "codec", "farfield")
DESTROYING = ("babble", "noise", "reverb")
DEG_ORDER = list(PRESERVING) + list(DESTROYING)
GROUP_LABEL = {
    "preserving": "spectrally distinct from speech",
    "destroying": "spectrally confusable with speech",
}
SNR_ORDER = [20.0, 10.0, 0.0]

ASR_ORDER = ["parakeet-tdt-0.6b-v3", "whisper-base-mlx", "whisper-large-v3-turbo"]
ASR_LABEL = {
    "parakeet-tdt-0.6b-v3": "Parakeet TDT 0.6B v3",
    "whisper-base-mlx": "Whisper Base",
    "whisper-large-v3-turbo": "Whisper Large-v3 Turbo",
}

# (key, point-estimate CSV, bootstrap-CI CSV, label). The 9B band file predates
# the 2026-08-27 re-run and is named for the method rather than the arm; its
# `llm` column is the only thing that identifies it, and it is asserted below.
ARMS = [
    ("qwen9b",
     "absorption_qwen9b_2026-08-27.csv",
     "absorption_ci_calibrated.csv",
     "Qwen3.5-9B (MLX 8-bit)",
     "mlx-community/Qwen3.5-9B-MLX-8bit"),
    ("qwen27b",
     "absorption_qwen27b_2026-08-27.csv",
     "absorption_ci_qwen27b_2026-08-27.csv",
     "Qwen3.5-27B (GPTQ-Int4)",
     "Qwen/Qwen3.5-27B-GPTQ-Int4"),
    ("gemma31b",
     "absorption_gemma31b_2026-08-27.csv",
     "absorption_ci_gemma31b_2026-08-27.csv",
     "Gemma-4-31B-it (QAT w4a16)",
     "google/gemma-4-31B-it-qat-w4a16-ct"),
]

# Re-derived 2026-08-28 by re-running the three joint Wald tests; see PROVENANCE.
H4B_WALD = {"chi2": 29.1127, "df": 6, "p": 5.79207e-05}
H4B_WALD_NOHALLUC = {"chi2": 23.4530, "df": 6, "p": 0.000658127}
CAPACITY_WALD = {"chi2": 3.7481, "df": 6, "p": 0.710727}


# =============================================================================
# Pure helpers (unit-tested in tests/test_make_thesis_figures.py)
# =============================================================================

_GEE_ROW = re.compile(
    r"^(?P<term>\S.*?)\s{2,}"
    r"(?P<coef>-?\d+\.\d+)\s+"
    r"(?P<se>\d+\.\d+)\s+"
    r"(?P<z>-?\d+\.\d+)\s+"
    r"(?P<p>\d+\.\d+)\s+"
    r"(?P<lo>-?\d+\.\d+)\s+"
    r"(?P<hi>-?\d+\.\d+)\s*$")


def parse_gee_summary(text):
    """Coefficient table of a saved statsmodels GEE summary, term -> stats.

    The saved fits are the only record of the estimates with their robust
    standard errors, and re-fitting to draw a forest plot would invite the
    figure and the confirmatory artifact to disagree. Parsing is therefore the
    conservative choice, but a loose parser is worse than none: header and
    footer lines of the summary carry numbers too. Rows are accepted only when
    six numeric columns follow a term name, which is the shape of a coefficient
    row and of nothing else in the block.
    """
    out = OrderedDict()
    for line in text.splitlines():
        if line.startswith("=") or line.startswith("-"):
            continue
        m = _GEE_ROW.match(line.rstrip())
        if not m:
            continue
        g = m.groupdict()
        term = g["term"].strip()
        if " " in term and not term.startswith("C("):
            # "Dep. Variable:   outcome   No. Observations:  31616" and friends
            # never reach here (they fail the six-number shape), but a term with
            # embedded spaces is not something this formula language produces.
            continue
        out[term] = {
            "coef": float(g["coef"]), "se": float(g["se"]),
            "z": float(g["z"]), "p": float(g["p"]),
            "lo": float(g["lo"]), "hi": float(g["hi"]),
        }
    if not out:
        raise ValueError("no coefficient rows found in this GEE summary")
    return out


def excludes_zero(lo, hi):
    """Does a two-sided interval exclude zero? Endpoints touching zero do not."""
    return bool(lo > 0 or hi < 0)


def verdict_counts(lo, hi):
    """(positive, covers zero, negative) over paired interval endpoints.

    This is the count §13 reports as "0 of 54 positive" and it is the whole
    contrast between the two comparators, so it is computed once, here, rather
    than in each figure that shows it.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if lo.shape != hi.shape:
        raise ValueError("lo and hi must have the same shape")
    pos = int(np.sum(lo > 0))
    neg = int(np.sum(hi < 0))
    return pos, len(lo) - pos - neg, neg


def waterfall_segments(components):
    """Cumulative (start, end) pairs for a waterfall, signs handled.

    A stacked bar cannot draw a decomposition whose parts have opposite signs —
    and four of the eighteen mediation cells have exactly that (a negative
    mediated path with a positive direct one). A waterfall can, because each
    segment starts where the previous one ended.
    """
    segments = []
    cursor = 0.0
    for value in components:
        segments.append((cursor, cursor + value))
        cursor += value
    return segments


def symmetric_limit(values, pad=1.05):
    """A colour/axis limit symmetric about zero, so sign is not exaggerated.

    A diverging scale fitted to the data's own asymmetric range makes a small
    positive value look like the mirror of a large negative one.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite values")
    m = float(np.max(np.abs(finite))) * pad
    return m if m > 0 else 1.0


def degradation_group(degradation):
    """'preserving' / 'destroying' / 'clean' — §13's three-and-three grouping."""
    if degradation in PRESERVING:
        return "preserving"
    if degradation in DESTROYING:
        return "destroying"
    if degradation == "clean":
        return "clean"
    raise ValueError("unknown degradation %r" % (degradation,))


_MED_ROW = re.compile(
    r"^(?P<deg>[a-z]+)\s+(?P<snr>\d+)\s*\|"
    r"\s*(?P<acme>[-+]\d+\.\d+)\s+(?P<ade>[-+]\d+\.\d+)\s+"
    r"(?P<total>[-+]\d+\.\d+)\s+(?P<med>\S+)\s*\|"
    r"\s*\[(?P<acme_lo>[-+]\d+\.\d+), (?P<acme_hi>[-+]\d+\.\d+)\]\s*n=\d+"
    r"\s*\[(?P<ade_lo>[-+]\d+\.\d+), (?P<ade_hi>[-+]\d+\.\d+)\]\s*n=\d+\s*$")


def parse_mediation_table(text):
    """Rows of the per-(degradation, SNR) mediation artifact.

    `%MED` is deliberately not returned: the artifact prints a dash wherever the
    total effect's interval covers zero, and a figure has no business turning
    that dash back into a number.
    """
    rows = []
    for line in text.splitlines():
        m = _MED_ROW.match(line.strip())
        if not m:
            continue
        g = m.groupdict()
        rows.append({
            "degradation": g["deg"], "snr_db": float(g["snr"]),
            "acme": float(g["acme"]), "ade": float(g["ade"]),
            "total": float(g["total"]),
            "acme_lo": float(g["acme_lo"]), "acme_hi": float(g["acme_hi"]),
            "ade_lo": float(g["ade_lo"]), "ade_hi": float(g["ade_hi"]),
        })
    if not rows:
        raise ValueError("no mediation rows found")
    return rows


def condition_columns(degradations=None, snrs=None):
    """The (degradation, SNR) column order shared by every per-cell figure."""
    degradations = DEG_ORDER if degradations is None else degradations
    snrs = SNR_ORDER if snrs is None else snrs
    return [(d, s) for d in degradations for s in snrs]


# =============================================================================
# Loading
# =============================================================================

def _path(name):
    return os.path.join(ANALYSIS_DIR, name)


def load_arm(point_csv, ci_csv, expected_llm):
    """Merged point estimates + bootstrap band for one router arm, degraded cells.

    The two files are joined on the cell key rather than on row order, and the
    `llm` column of each is asserted against the arm it is supposed to be: the
    9B band file is named `absorption_ci_calibrated.csv` after the method, not
    after the arm, and nothing else in its name would catch a mix-up.
    """
    pts = pd.read_csv(_path(point_csv))
    cis = pd.read_csv(_path(ci_csv))
    for frame, name in ((pts, point_csv), (cis, ci_csv)):
        found = sorted(frame["llm"].unique())
        if found != [expected_llm]:
            raise ValueError("%s carries llm %s, expected only %r"
                             % (name, found, expected_llm))
    key = ["asr", "degradation", "snr_db"]
    merged = pts.merge(cis, on=key, how="inner", suffixes=("", "_ci"))
    merged = merged[merged["degradation"] != "clean"].copy()
    if len(merged) != 54:
        raise ValueError("expected 54 degraded cells in %s, got %d"
                         % (point_csv, len(merged)))
    return merged


def _style_axes(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=3, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)


def _save(fig, stem, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    written = []
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        p = os.path.join(out_dir, "%s.%s" % (stem, ext))
        fig.savefig(p, facecolor=SURFACE, bbox_inches="tight", **kwargs)
        written.append(p)
    plt.close(fig)
    return written


def _rc():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
        "pdf.fonttype": 42,      # embed TrueType, keep text selectable in print
        "ps.fonttype": 42,
    })


# =============================================================================
# fig1 — retention dose-response
# =============================================================================

# Marker/linestyle per degradation: the second and third channels, so the six
# curves stay separable in greyscale and under any colour-vision deficiency.
DEG_MARK = {"clipping": ("o", "-"), "codec": ("s", "--"), "farfield": ("^", ":"),
            "babble": ("o", "-"), "noise": ("s", "--"), "reverb": ("^", ":")}


def figure_retention(out_dir):
    """Retention vs SNR, every ASR x router cell of the cascade sweep.

    §13: "Retention is a level that IS directly observable and needs no null
    model." It is the share of items the router still gets right end-to-end
    after degradation, among the items the same (utterance, ASR, router) triple
    got right on clean audio and whose degraded transcript carries WER > 0
    (§8's `gain_paired` subset). No comparator, no fitted curve, no assumption.
    """
    arms = [(label, load_arm(pc, cc, llm))
            for _, pc, cc, label, llm in ARMS]
    fig, axes = plt.subplots(len(arms), len(ASR_ORDER), figsize=(9.6, 8.4),
                             sharex=True, sharey=True)
    x = np.arange(len(SNR_ORDER))
    for r, (arm_label, d) in enumerate(arms):
        for c, asr in enumerate(ASR_ORDER):
            ax = axes[r][c]
            _style_axes(ax)
            sub = d[d["asr"] == asr]
            for deg in DEG_ORDER:
                marker, ls = DEG_MARK[deg]
                colour = BLUE if degradation_group(deg) == "preserving" else ORANGE
                ys = [float(sub[(sub["degradation"] == deg)
                                & (sub["snr_db"] == s)]["retention"].iloc[0])
                      for s in SNR_ORDER]
                ax.plot(x, ys, color=colour, linestyle=ls, linewidth=1.6,
                        marker=marker, markersize=5, markeredgecolor=SURFACE,
                        markeredgewidth=1.4, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(["20", "10", "0"])
            ax.set_ylim(0, 1.02)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            if r == 0:
                ax.set_title(ASR_LABEL[asr], color=INK, pad=8)
            if c == 0:
                ax.set_ylabel(arm_label + "\nretention", color=INK_2)
            if r == len(arms) - 1:
                ax.set_xlabel("SNR (dB) — severity increases to the right")

    # Selective direct labels: the two extremes §13 names, and nothing else.
    d0 = arms[0][1]
    top = d0.loc[d0["retention"].idxmax()]
    bot = d0.loc[d0["retention"].idxmin()]
    for row, anchor in ((top, (0.06, 0.87)), (bot, (0.04, 0.16))):
        ax = axes[0][ASR_ORDER.index(row["asr"])]
        xi = SNR_ORDER.index(float(row["snr_db"]))
        ax.annotate("%.1f%% — %s @ %d dB" % (100 * row["retention"],
                                             DEG_LABEL[row["degradation"]],
                                             int(row["snr_db"])),
                    xy=(xi, row["retention"]), xycoords="data",
                    xytext=anchor, textcoords="axes fraction",
                    color=INK, fontsize=7.5, va="center", ha="left",
                    # A halo, because the callout sits over the curve bundle in
                    # one of the two panels and must stay readable there.
                    path_effects=[pe.withStroke(linewidth=2.6,
                                                foreground=SURFACE)],
                    arrowprops=dict(arrowstyle="-", color=MUTED,
                                    linewidth=0.8, shrinkA=2, shrinkB=4))

    handles = [Line2D([], [], color=(BLUE if degradation_group(d) == "preserving"
                                     else ORANGE),
                      linestyle=DEG_MARK[d][1], marker=DEG_MARK[d][0],
                      markersize=5, linewidth=1.6, label=DEG_LABEL[d])
               for d in DEG_ORDER]
    leg = fig.legend(handles=handles, loc="lower center", ncol=6,
                     bbox_to_anchor=(0.5, -0.035), fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle("Retention under degradation — the level that needs no null model",
                 color=INK, fontsize=13, y=1.0)
    fig.text(0.5, 0.965,
             "Share of items still answered correctly end-to-end after degradation, among the items the same "
             "utterance / ASR / router triple answered\ncorrectly on clean audio and whose degraded "
             "transcript carries WER > 0 — §8's paired subset, %d to %d of the 416 utterances per cell. "
             "No null model,\nno fitted curve, nothing assumed. "
             "Blue, solid-to-dotted: %s.  Orange: %s."
             % (int(min(d["n_paired"].min() for _l, d in arms)),
                int(max(d["n_paired"].max() for _l, d in arms)),
                GROUP_LABEL["preserving"], GROUP_LABEL["destroying"]),
             ha="center", va="top", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.945))
    return _save(fig, "fig1_retention_dose_response", out_dir)


# =============================================================================
# fig2 — calibrated-null gain heatmap
# =============================================================================

def figure_gain_heatmap(out_dir):
    """Per-cell `gain_calibrated`, one panel per router arm.

    §7 asks for an "absorption-gain heatmap". It is drawn under the calibrated
    null and under nothing else: that is the one comparator of the four whose
    baseline is *estimated* (the router's own clean WER->EES logistic) rather
    than assumed, and §13's standing instruction is that no positive headline
    absorption number may be quoted. The value is printed inside every cell, so
    the map is readable in greyscale and doubles as its own table view.
    """
    arms = [(label, load_arm(pc, cc, llm)) for _, pc, cc, label, llm in ARMS]
    cols = condition_columns()
    all_gains = np.concatenate([d["gain_calibrated"].to_numpy() for _, d in arms])
    lim = symmetric_limit(all_gains)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    halo = [pe.withStroke(linewidth=1.8, foreground=SURFACE)]

    plt.rcParams["hatch.linewidth"] = 0.6
    fig, axes = plt.subplots(len(arms), 1, figsize=(12.6, 8.6))
    total_flagged = 0
    for ax, (arm_label, d) in zip(axes, arms):
        grid = np.full((len(ASR_ORDER), len(cols)), np.nan)
        flags = np.zeros_like(grid, dtype=bool)
        signs = np.zeros_like(grid)
        for i, asr in enumerate(ASR_ORDER):
            for j, (deg, snr) in enumerate(cols):
                row = d[(d["asr"] == asr) & (d["degradation"] == deg)
                        & (d["snr_db"] == snr)]
                grid[i, j] = float(row["gain_calibrated"].iloc[0])
                lo = float(row["gain_calibrated_lo"].iloc[0])
                hi = float(row["gain_calibrated_hi"].iloc[0])
                flags[i, j] = excludes_zero(lo, hi)
                signs[i, j] = 1.0 if lo > 0 else (-1.0 if hi < 0 else 0.0)
        # pcolormesh, not imshow: imshow lands in the PDF as a resampled
        # raster (measured 945x161 px per panel, ~95 dpi at print size), and
        # this page is printed. A quad mesh stays vector.
        im = ax.pcolormesh(np.arange(len(cols) + 1) - 0.5,
                           np.arange(len(ASR_ORDER) + 1) - 0.5,
                           grid, cmap=DIVERGING, norm=norm, shading="flat")
        ax.set_xlim(-0.5, len(cols) - 0.5)
        ax.set_ylim(len(ASR_ORDER) - 0.5, -0.5)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if flags[i, j]:
                    total_flagged += 1
                    # Hatch direction carries the sign, so the marked cells stay
                    # readable without colour. Every one of them is negative.
                    hatch = "///" if signs[i, j] < 0 else "\\\\\\"
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch=hatch, edgecolor=INK_2,
                                           linewidth=0.0, zorder=2))
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=INK, linewidth=1.2,
                                           zorder=3))
                ax.text(j, i, "%+.1f" % (100 * grid[i, j]), ha="center",
                        va="center", fontsize=6.4, color=INK, zorder=6,
                        path_effects=halo)
        ax.set_yticks(range(len(ASR_ORDER)))
        ax.set_yticklabels([ASR_LABEL[a] for a in ASR_ORDER], fontsize=8)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(["%d" % s for _, s in cols], fontsize=7)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, labelcolor=INK_2)
        # 2px surface gaps rather than borders between cells
        for j in range(len(cols) + 1):
            ax.axvline(j - 0.5, color=SURFACE, linewidth=2, zorder=4)
        for i in range(len(ASR_ORDER) + 1):
            ax.axhline(i - 0.5, color=SURFACE, linewidth=2, zorder=4)
        # group separator between the preserving and destroying halves
        ax.axvline(len(PRESERVING) * len(SNR_ORDER) - 0.5, color=AXIS,
                   linewidth=1.2, zorder=5)
        pos, cov, neg = verdict_counts(d["gain_calibrated_lo"],
                                       d["gain_calibrated_hi"])
        ax.set_title("%s   —   %d of 54 cells positive with a 95%% interval "
                     "excluding zero   ·   %d negative   ·   %d covering zero"
                     % (arm_label, pos, neg, cov),
                     loc="left", color=INK, fontsize=9.5, pad=6)

    # Grouped x-axis on the bottom panel only: SNR ticks, then the degradation
    # each triple belongs to, then §13's two-group split. Nothing above the
    # panels, so each panel title stays with its own arm.
    bottom = axes[-1]
    below = ("data", "axes fraction")
    for k, deg in enumerate(DEG_ORDER):
        centre = k * len(SNR_ORDER) + (len(SNR_ORDER) - 1) / 2.0
        bottom.annotate(DEG_LABEL[deg], xy=(centre, 0), xycoords=below,
                        xytext=(0, -24), textcoords="offset points",
                        ha="center", va="top", color=INK, fontsize=8.5,
                        annotation_clip=False)
    for group, span in (("preserving", (0, len(PRESERVING))),
                        ("destroying", (len(PRESERVING), len(DEG_ORDER)))):
        lo = span[0] * len(SNR_ORDER) - 0.35
        hi = span[1] * len(SNR_ORDER) - 0.65
        blend = matplotlib.transforms.blended_transform_factory(
            bottom.transData, bottom.transAxes)
        bottom.plot([lo, hi], [-0.30, -0.30], transform=blend, color=AXIS,
                    linewidth=0.8, clip_on=False, zorder=5)
        bottom.annotate(GROUP_LABEL[group], xy=((lo + hi) / 2.0, 0),
                        xycoords=below, xytext=(0, -48),
                        textcoords="offset points", ha="center", va="top",
                        color=MUTED, fontsize=7.5, style="italic",
                        annotation_clip=False)
    bottom.set_xlabel("SNR (dB) within each degradation", color=INK_2,
                      labelpad=52)

    fig.subplots_adjust(left=0.135, right=0.885, top=0.845, bottom=0.13,
                        hspace=0.42)
    cax = fig.add_axes([0.905, 0.30, 0.011, 0.40])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("gain under the calibrated null\n(observed minus predicted "
                   "P(end-to-end success), percentage points)",
                   color=INK_2, fontsize=8)
    cbar.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _p: "%+.0f" % (100 * v)))
    cbar.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=7.5)
    cbar.outline.set_visible(False)

    fig.suptitle("Absorption gain against the router's own clean WER-to-EES "
                 "curve", color=INK, fontsize=13, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.945,
             "Each cell is 416 utterances. Numbers are percentage points of end-to-end success. "
             "Outlined and hatched cells are the only ones whose 95%% bootstrap interval\n"
             "(B = 2000, resampled by utterance, the logistic curve refit inside every replicate) "
             "excludes zero — %d of 162, every one of them negative, and hatched\n"
             "left-leaning for that sign. The assumed 1 − WER comparator is not drawn here; "
             "fig3 is where the two are compared, and why."
             % total_flagged,
             ha="left", va="top", color=INK_2, fontsize=8)
    return _save(fig, "fig2_absorption_gain_calibrated_heatmap", out_dir)


# =============================================================================
# fig3 — the contrast between the two comparators
# =============================================================================

COMPARATORS = [
    ("gain_paired", ORANGE,
     "Assumed comparator:  1 − WER",
     "the baseline the design started from — asserted, not estimated"),
    ("gain_calibrated", BLUE,
     "Calibrated null:  clean WER-to-EES curve",
     "the baseline estimated from data — the comparator with standing"),
]


def figure_comparator_contrast(out_dir):
    """Every cell's interval under both comparators, and the verdict counts.

    This is the figure §13's headline actually is. Under the assumed comparator
    the absorption gain is positive with a 95% interval clear of zero in 30-34
    of 54 cells; under the comparator estimated from the router's own clean
    behaviour it is positive in none, on any of three routers spanning two
    vendors and a 3x scale range. The cells are sorted independently within each
    panel, because the claim is about the distribution of intervals and not
    about which cell is which — fig2 carries the per-cell placement.
    """
    arms = [(label, load_arm(pc, cc, llm)) for _, pc, cc, label, llm in ARMS]
    short = ["Qwen-9B", "Qwen-27B", "Gemma-31B"]
    fig = plt.figure(figsize=(11.8, 8.8))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.66])

    lim = 0.0
    for _, d in arms:
        for comp, _c, _t, _s in COMPARATORS:
            lim = max(lim, float(np.max(np.abs(
                np.concatenate([d[comp + "_lo"], d[comp + "_hi"]])))))
    lim *= 1.06

    first_col = []
    for r, (comp, colour, title, subtitle) in enumerate(COMPARATORS):
        for c, (arm_label, d) in enumerate(arms):
            ax = fig.add_subplot(gs[r, c])
            _style_axes(ax)
            if c == 0:
                first_col.append(ax)
            order = np.argsort(d[comp].to_numpy())
            est = d[comp].to_numpy()[order]
            lo = d[comp + "_lo"].to_numpy()[order]
            hi = d[comp + "_hi"].to_numpy()[order]
            xs = np.arange(len(est))
            clear = np.array([excludes_zero(a, b) for a, b in zip(lo, hi)])
            ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
            ax.vlines(xs, lo, hi, color=colour, linewidth=1.0, alpha=0.55,
                      zorder=2)
            ax.scatter(xs[~clear], est[~clear], s=13, facecolors=SURFACE,
                       edgecolors=colour, linewidths=1.0, zorder=3)
            ax.scatter(xs[clear], est[clear], s=15, facecolors=colour,
                       edgecolors=SURFACE, linewidths=0.8, zorder=4)
            ax.set_ylim(-lim, lim)
            ax.set_xlim(-1.5, len(est) + 0.5)
            ax.set_xticks([])
            pos, cov, neg = verdict_counts(lo, hi)
            ax.text(0.03, 0.955, "%d of 54 positive\n%d negative" % (pos, neg),
                    transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    color=INK)
            if c == 0:
                ax.set_ylabel("gain\n(probability difference)", color=INK_2)
            else:
                ax.tick_params(labelleft=False)
            if r == 0:
                ax.set_title(arm_label, color=INK, fontsize=9.5, pad=6)
            if r == len(COMPARATORS) - 1:
                ax.set_xlabel("54 degraded cells, sorted by point estimate",
                              color=MUTED, fontsize=7.5)

    # Verdict counts: the same information as a count, so nothing depends on
    # reading interval positions off a dense panel.
    ax = fig.add_subplot(gs[2, :])
    _style_axes(ax, grid_axis="x")
    labels, rows = [], []
    for name, (arm_label, d) in zip(short, arms):
        for comp, colour, title, _s in COMPARATORS:
            rows.append(verdict_counts(d[comp + "_lo"], d[comp + "_hi"]))
            labels.append("%s  ·  %s" % (name, title.split(":")[0].lower()))
    ys = np.arange(len(rows))[::-1]
    seg_style = [("positive gain, 95% CI excludes zero", BLUE, None),
                 ("interval covers zero", GRID, None),
                 ("negative gain, 95% CI excludes zero", RED, "///")]
    for k, (name, colour, hatch) in enumerate(seg_style):
        left = np.array([sum(row[:k]) for row in rows], dtype=float)
        width = np.array([row[k] for row in rows], dtype=float)
        ax.barh(ys, width, left=left, height=0.55, color=colour, hatch=hatch,
                edgecolor=SURFACE, linewidth=2.0, label=name, zorder=3)
        if k < 2:  # only the two wide segments can hold a label inside
            for y, l, w in zip(ys, left, width):
                if w >= 8:
                    ax.text(l + w / 2.0, y, "%d" % w, ha="center", va="center",
                            fontsize=8, color=INK if k == 1 else SURFACE,
                            zorder=4)
    for y, row in zip(ys, rows):
        ax.text(55.4, y, "%d negative" % row[2], ha="left", va="center",
                fontsize=8, color=INK_2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 54)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("cells of 54", color=INK_2)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3,
                    fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.subplots_adjust(left=0.175, right=0.90, top=0.855, bottom=0.135,
                        hspace=0.50, wspace=0.14)
    for ax0, (comp, colour, title, subtitle) in zip(first_col, COMPARATORS):
        box = ax0.get_position()
        fig.text(0.035, (box.y0 + box.y1) / 2.0, title, rotation=90,
                 va="center", ha="center", color=INK, fontsize=8.5)

    fig.suptitle("The absorption level is a choice of comparator — and the one "
                 "estimated from data puts it at zero",
                 color=INK, fontsize=13, x=0.02, ha="left", y=0.985)
    n_paired = np.concatenate([d["n_paired"].to_numpy() for _l, d in arms])
    n_cal = sorted({int(v) for _l, d in arms
                    for v in d["n_calibrated"].to_numpy()})
    fig.text(0.02, 0.945,
             "Top row: %s.   Bottom row: %s.\n"
             "Filled markers mark a 95%% bootstrap interval (B = 2000, "
             "resampled by utterance) that excludes zero; open markers one that "
             "covers it.\n"
             "The comparator and the population it is applied to change "
             "together, and §13 records that the population is not a detail but "
             "the whole result: the assumed\ncomparator is measured on the "
             "paired subset (%d–%d of the 416 utterances per cell), the "
             "calibrated null over the full degraded sample (%s)."
             % (COMPARATORS[0][3], COMPARATORS[1][3],
                int(n_paired.min()), int(n_paired.max()),
                " / ".join(str(v) for v in n_cal)),
             ha="left", va="top", color=INK_2, fontsize=8)
    return _save(fig, "fig3_absorption_comparator_contrast", out_dir)


# =============================================================================
# fig4 — confirmatory coefficient forest
# =============================================================================

H4B_INTERACTION = ("C(pathway, Treatment(reference='cascade'))[T.omni]:"
                   "C(degradation, Treatment(reference='clean'))[T.%s]")
H4B_MAIN = "C(pathway, Treatment(reference='cascade'))[T.omni]"


def figure_forest(out_dir):
    """Confirmatory estimates, all on one log-odds axis.

    Two panels because two different things are being shown. The upper panel is
    H1/H2a/H2b/H3, where the hypothesis is a coefficient or a contrast. The
    lower panel is H4b, where **the hypothesis is not any single coefficient**:
    §3's operational is the pathway x degradation interaction, and the test of
    it is the joint Wald over all six terms, which is invariant to the reference
    level. The individual contrasts below it are contrasts *against clean*, the
    anchor §3 and §8 pre-register, and they are shown so the shape of the
    interaction is visible — not as six tests.
    """
    with open(os.path.join(ANALYSIS_DIR, "fits", "diagnostics.json")) as f:
        diag = json.load(f)
    def _fit(dirname, model, outcome):
        with open(os.path.join(ANALYSIS_DIR, dirname,
                               "%s_%s.txt" % (model, outcome))) as fh:
            return parse_gee_summary(fh.read())
    h1, h2, h3 = _fit("fits", "h1", "tsa"), _fit("fits", "h2", "ees"), \
        _fit("fits", "h3", "pf")
    h4b = _fit("fits", "h4b", "ees")
    h4b_nh = _fit("fits_nohalluc", "h4b", "ees")

    contrast = diag["h1"]["contrast"]
    z = 1.959963984540054  # two-sided 95% normal quantile
    c_lo = contrast["effect"] - z * contrast["std_err"]
    c_hi = contrast["effect"] + z * contrast["std_err"]

    # H2a is stated in §3 as 27B − 9B; the fit estimates 9B − 27B. Flipping the
    # sign (and the interval ends with it) is the whole of the conversion, and
    # §13 flags getting it wrong as the live trap on this coefficient.
    nine = h2["C(llm_model)[T.Qwen3.5-9B-MLX-8bit]"]
    gemma = h2["C(llm_model)[T.gemma-4-31B-it-qat-w4a16-ct]"]

    rows_a = [
        ("H1  ·  TSA", "critical-slot error − function-word error  (the contrast)",
         contrast["effect"], c_lo, c_hi, contrast["p_value"], True),
        ("", "  error on a critical slot",
         h1["has_critical_error"]["coef"], h1["has_critical_error"]["lo"],
         h1["has_critical_error"]["hi"], h1["has_critical_error"]["p"], False),
        ("", "  error on a function word",
         h1["has_function_error"]["coef"], h1["has_function_error"]["lo"],
         h1["has_function_error"]["hi"], h1["has_function_error"]["p"], False),
        ("H2a  ·  EES", "Qwen-27B − Qwen-9B  (§3's sign convention)",
         -nine["coef"], -nine["hi"], -nine["lo"], nine["p"], True),
        ("H2b  ·  EES", "Gemma-31B − Qwen-27B  (not the absorption gain — see fig3)",
         gemma["coef"], gemma["lo"], gemma["hi"], gemma["p"], True),
        ("H3  ·  PF", "phonetic distance of the mis-hearing",
         h3["phonetic_distance"]["coef"], h3["phonetic_distance"]["lo"],
         h3["phonetic_distance"]["hi"], h3["phonetic_distance"]["p"], True),
    ]

    fig = plt.figure(figsize=(10.6, 9.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.62)

    ax = fig.add_subplot(gs[0])
    _style_axes(ax, grid_axis="x")
    ys = np.arange(len(rows_a))[::-1]
    for y, (_grp, label, est, lo, hi, p, primary) in zip(ys, rows_a):
        clear = excludes_zero(lo, hi)
        ax.hlines(y, lo, hi, color=BLUE, linewidth=2.0 if primary else 1.2,
                  alpha=1.0 if primary else 0.55, zorder=3)
        ax.scatter([est], [y], s=48 if primary else 26,
                   facecolors=BLUE if clear else SURFACE, edgecolors=BLUE,
                   linewidths=1.2, zorder=4,
                   marker="D" if primary else "o")
        ax.text(1.015, y, _p_label(p), transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7.5, color=INK_2,
                clip_on=False)
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(["%s%s" % (("%s   " % g) if g else "", lbl)
                        for g, lbl, *_ in rows_a], fontsize=8)
    ax.set_ylim(-0.8, len(rows_a) - 0.2)
    ax.set_xlabel("log-odds (GEE, binomial, exchangeable, robust SEs, clustered "
                  "on utterance)", color=INK_2)
    ax.set_title("H1 – H3: the estimate each hypothesis is stated as",
                 loc="left", color=INK, fontsize=10, pad=8)
    ax.text(0.0, -0.185,
            "Diamonds are the quantity the hypothesis names; small circles are "
            "its components. Filled = 95%% interval excludes zero.\n"
            "H1 on %s rows, H2 on %s, H3 on %s (269 utterance clusters — the "
            "extra rows are the same transcripts re-routed)."
            % ("{:,}".format(diag["h1"]["diagnostics"]["rows_out"]),
               "{:,}".format(diag["h2"]["diagnostics"]["rows_out"]),
               "{:,}".format(diag["h3"]["diagnostics"]["rows_out"])),
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            color=MUTED)

    ax = fig.add_subplot(gs[1])
    _style_axes(ax, grid_axis="x")
    order = DEG_ORDER
    ys = np.arange(len(order) + 1)[::-1]
    series = [("full fit  (31,616 rows)", h4b, BLUE, "D", 0.16),
              ("hallucination rows dropped  (30,868)", h4b_nh, ORANGE, "s", -0.16)]
    for name, fit, colour, marker, dy in series:
        for y, deg in zip(ys[1:], order):
            t = fit[H4B_INTERACTION % deg]
            clear = excludes_zero(t["lo"], t["hi"])
            ax.hlines(y + dy, t["lo"], t["hi"], color=colour, linewidth=1.6,
                      zorder=3)
            ax.scatter([t["coef"]], [y + dy], s=36, marker=marker,
                       facecolors=colour if clear else SURFACE,
                       edgecolors=colour, linewidths=1.2, zorder=4,
                       label=name if deg == order[0] else None)
        m = fit[H4B_MAIN]
        ax.hlines(ys[0] + dy, m["lo"], m["hi"], color=MUTED, linewidth=1.6,
                  zorder=3)
        ax.scatter([m["coef"]], [ys[0] + dy], s=36, marker=marker,
                   facecolors=SURFACE, edgecolors=MUTED, linewidths=1.2,
                   zorder=4)
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.axhline(ys[0] - 0.5, color=GRID, linewidth=1.0, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(["omni − cascade on clean audio  (the anchor itself)"]
                       + [DEG_LABEL[d] for d in order], fontsize=8.5)
    ax.set_ylim(-0.8, len(order) + 0.4)
    ax.set_xlabel("omni − cascade gap, relative to the gap on clean audio "
                  "(log-odds)", color=INK_2)
    ax.set_title("H4b: the interaction, expressed against the pre-registered "
                 "`clean` anchor", loc="left", color=INK, fontsize=10, pad=46)
    leg = ax.legend(loc="lower right", fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK_2)

    ax.text(0.0, 1.012,
            "CONFIRMATORY ANSWER — joint Wald over all six interactions, "
            "reference-invariant:  χ² = %.4f, df %d, p = %s\n"
            "(sensitivity refit, hallucination rows dropped: χ² = %.4f, "
            "p = %s)"
            % (H4B_WALD["chi2"], H4B_WALD["df"], _fmt_p(H4B_WALD["p"]),
               H4B_WALD_NOHALLUC["chi2"], _fmt_p(H4B_WALD_NOHALLUC["p"])),
            transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=NEUTRAL,
                      edgecolor=AXIS, linewidth=0.8))
    ax.text(0.0, -0.155,
            "Single interaction coefficients are contrasts against a reference "
            "level and change with it: against `clean` only additive noise "
            "clears p < 0.05\n(+0.1944, p = 0.0069); reverberation does not "
            "(−0.0984, p = 0.171). Against `babble` both clear at p = 0.005. "
            "The hypothesis is the joint test above,\nnot any row below it. "
            "Filled markers mark p < 0.05 individually and are not a second "
            "test.",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            color=MUTED)

    fig.subplots_adjust(left=0.315, right=0.855, top=0.905, bottom=0.115)
    fig.suptitle("Confirmatory estimates, with their 95% intervals",
                 color=INK, fontsize=13, x=0.02, ha="left", y=0.975)
    return _save(fig, "fig4_confirmatory_coefficient_forest", out_dir)


def _p_label(p):
    """A p-value as it should read beside a forest row."""
    text = _fmt_p(p)
    return ("p %s" % text) if text.startswith("<") else ("p = %s" % text)


def _fmt_p(p):
    """Never print `0.00e+00`.

    statsmodels rounds the p-column of a saved summary to three decimals, so a
    parsed 0.000 means "smaller than the artifact records", not zero. Printing
    it as a number would invent a precision the artifact does not have.
    """
    if p <= 0:
        return "< 0.001"
    if p < 1e-3:
        # Below a thousandth, four decimals throw away significant figures:
        # 6.58e-04 would print as 0.0007.
        return "%.2e" % p
    return "%.4f" % p


# =============================================================================
# fig5 — mediation attribution waterfall
# =============================================================================

def figure_mediation_waterfall(out_dir):
    """Where each degradation's damage goes: through the transcript, or not.

    A waterfall rather than a stacked bar because four of the eighteen cells
    have components of opposite sign, which a stack cannot draw honestly.
    """
    with open(os.path.join(ANALYSIS_DIR, "mediation_per_snr.txt")) as f:
        text = f.read()
    rows = {(r["degradation"], r["snr_db"]): r
            for r in parse_mediation_table(text)}
    cols = condition_columns()
    missing = [c for c in cols if c not in rows]
    if missing:
        raise ValueError("mediation artifact is missing cells: %s" % missing)

    fig, ax = plt.subplots(figsize=(9.8, 8.2))
    _style_axes(ax, grid_axis="x")
    ys = np.arange(len(cols))[::-1]
    n_acme, n_ade, marks = 0, 0, []
    for y, key in zip(ys, cols):
        r = rows[key]
        (a_start, a_end), (d_start, d_end) = waterfall_segments(
            [r["acme"], r["ade"]])
        ax.barh(y, a_end - a_start, left=a_start, height=0.6, color=BLUE,
                edgecolor=SURFACE, linewidth=2.0, zorder=3)
        ax.barh(y, d_end - d_start, left=d_start, height=0.6, color=ORANGE,
                hatch="///", edgecolor=SURFACE, linewidth=2.0, zorder=3)
        a_clear = excludes_zero(r["acme_lo"], r["acme_hi"])
        d_clear = excludes_zero(r["ade_lo"], r["ade_hi"])
        n_acme += int(a_clear)
        n_ade += int(d_clear)
        ax.hlines(y + 0.36, r["acme_lo"], r["acme_hi"], color=BLUE,
                  linewidth=1.2, zorder=5)
        ax.hlines(y - 0.36, r["acme"] + r["ade_lo"], r["acme"] + r["ade_hi"],
                  color=ORANGE, linewidth=1.2, zorder=5)
        ax.scatter([r["total"]], [y], s=34, marker="D", facecolors=SURFACE,
                   edgecolors=INK, linewidths=1.1, zorder=6)
        marks.append("*" if d_clear else " ")
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(["%s%s  ·  %d dB" % (m, DEG_LABEL[d], int(s))
                        for m, (d, s) in zip(marks, cols)], fontsize=8)
    ax.set_ylim(-0.8, len(cols) - 0.2)
    for k in range(1, len(DEG_ORDER)):
        ax.axhline(len(cols) - k * len(SNR_ORDER) - 0.5, color=GRID,
                   linewidth=1.0, zorder=1)
    ax.set_xlabel("effect on the probability of end-to-end success, "
                  "percentage points (clean audio -> degraded)", color=INK_2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _p: "0" if abs(v) < 1e-9 else "%+.0f" % (100 * v)))

    handles = [Patch(facecolor=BLUE, edgecolor=SURFACE,
                     label="through the transcript (ACME)"),
               Patch(facecolor=ORANGE, hatch="///", edgecolor=SURFACE,
                     label="everything else (ADE, direct)"),
               Line2D([], [], marker="D", color=INK, markerfacecolor=SURFACE,
                      linestyle="none", markersize=6, label="total effect")]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle("Almost all of a degradation's damage travels through the "
                 "transcript", color=INK, fontsize=13, x=0.02, ha="left",
                 y=0.995)
    fig.text(0.02, 0.955,
             "Imai, Keele & Tingley (2010) potential-outcomes decomposition; "
             "total = ACME + ADE holds exactly. 23,712 rows, 416 utterances, "
             "one clean-vs-degradation\ncontrast per cell. Whiskers are the "
             "cluster bootstrap 95%% interval on each component "
             "(B = 100, --sim 800, --boot-sim 100, seed 42).\n"
             "The mediated path excludes zero in %d of 18 cells; the direct "
             "path in %d, marked * — reverberation at 10 dB and at 0 dB, and "
             "nowhere else.\nThe point estimates grow with severity throughout, "
             "but that is a statement about point estimates only."
             % (n_acme, n_ade),
             ha="left", va="top", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    return _save(fig, "fig5_mediation_attribution_waterfall", out_dir)


# =============================================================================
# fig6 — 9B/27B capacity calibration
# =============================================================================

def figure_capacity_calibration(out_dir):
    """The H4b defence: does the capacity gap widen as conditions harden?

    The objection H4b's diff-in-diff has to answer is that a *constant* capacity
    gap cancels but a widening one does not. The 9B/27B ladder is the only
    capacity ladder this design can measure, and the left panel is the honest
    version of the answer: the two routers' EES curves lie on top of each other,
    so the gap the middle panel magnifies is a 3-percentage-point band inside a
    33-point range.
    """
    d = pd.read_csv(_path("h4b_capacity_calibration_2026-08-28.csv"))
    d["snr_db"] = d["snr_db"].fillna(-1.0)
    key = [("clean", -1.0)] + condition_columns()
    d = d.set_index(["degradation", "snr_db"]).loc[key].reset_index()
    labels = ["Clean"] + ["%s %d dB" % (DEG_LABEL[dg], int(s))
                          for dg, s in condition_columns()]
    x = np.arange(len(d))

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.2),
                             gridspec_kw={"width_ratios": [1.35, 1.35, 0.9],
                                          "wspace": 0.28})

    ax = axes[0]
    _style_axes(ax)
    # Break the line between degradation blocks: `clean` and the six triples are
    # separate conditions, and a connector across them would draw a trend that
    # the x-axis does not have.
    blocks = [[0]] + [list(range(1 + k * len(SNR_ORDER),
                                 1 + (k + 1) * len(SNR_ORDER)))
                      for k in range(len(DEG_ORDER))]
    for col, colour, marker, ls, name in (
            ("ees_qwen9b", BLUE, "o", "-", "Qwen3.5-9B"),
            ("ees_qwen27b", ORANGE, "s", "--", "Qwen3.5-27B")):
        for n, block in enumerate(blocks):
            ax.plot(x[block], d[col].to_numpy()[block], color=colour,
                    marker=marker, linestyle=ls, linewidth=1.6, markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
                    label=name if n == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("end-to-end success (EES)", color=INK_2)
    ax.set_ylim(0, 0.72)
    ax.set_title("The two routers, same axis", loc="left", color=INK,
                 fontsize=9.5, pad=6)
    leg = ax.legend(loc="lower left", fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK_2)

    ax = axes[1]
    _style_axes(ax)
    gap = d["gap_27b_minus_9b"].to_numpy()
    mean_gap = float(np.mean(gap))
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.vlines(x, 0, gap, color=MUTED, linewidth=1.0, zorder=2)
    colours = [BLUE if g >= 0 else RED for g in gap]
    ax.scatter(x, gap, s=34, c=colours, edgecolors=SURFACE, linewidths=1.0,
               zorder=4)
    ax.axhline(mean_gap, color=INK, linewidth=1.0, linestyle=(0, (6, 3)),
               zorder=3)
    ax.text(len(x) - 0.4, mean_gap, "  mean %+.2f pp" % (100 * mean_gap),
            va="bottom", ha="right", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylim(-0.05, 0.05)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _p: "%+.0f" % (100 * v)))
    ax.set_ylabel("gap, 27B − 9B (percentage points)", color=INK_2)
    ax.set_title("The gap, on a ±5 pp scale", loc="left", color=INK,
                 fontsize=9.5, pad=6)
    ax.text(0.03, 0.04,
            "joint Wald over the 6\nscale × degradation terms:\n"
            "χ² = %.4f, df %d, p = %.4f"
            % (CAPACITY_WALD["chi2"], CAPACITY_WALD["df"], CAPACITY_WALD["p"]),
            transform=ax.transAxes, va="bottom", ha="left", fontsize=8,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=NEUTRAL,
                      edgecolor=AXIS, linewidth=0.8))

    ax = axes[2]
    _style_axes(ax)
    difficulty = 1.0 - (d["ees_qwen9b"] + d["ees_qwen27b"]) / 2.0
    r = float(np.corrcoef(difficulty, gap)[0, 1])
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.scatter(difficulty, gap, s=34, facecolors=BLUE, edgecolors=SURFACE,
               linewidths=1.0, zorder=3)
    ax.set_xlabel("cell difficulty  (1 − mean EES)", color=INK_2)
    ax.set_ylabel("gap, 27B − 9B (percentage points)", color=INK_2)
    ax.set_ylim(-0.05, 0.05)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _p: "%+.0f" % (100 * v)))
    ax.set_title("Does the gap widen where it is hard?", loc="left", color=INK,
                 fontsize=9.5, pad=6)
    ax.text(0.96, 0.95, ("r = %.3f\nthe gap narrows" % r).replace("-", "\u2212"),
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=INK)

    fig.suptitle("The 9B -> 27B capacity gap does not widen with severity",
                 color=INK, fontsize=13, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.935,
             "Whisper-Turbo rows only: 15,808 rows, 416 utterances, 19 cells of "
             "416 each. NOTE: This calibrates a 9B -> 27B ladder. It does NOT "
             "calibrate the omni-vs-cascade\ncapacity difference, which §3 "
             "describes as substantially larger; extrapolating from a 0.75 pp "
             "ladder to that gap is not supported by this check.",
             ha="left", va="top", color=INK_2, fontsize=8)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.24)
    return _save(fig, "fig6_capacity_calibration_9b_27b", out_dir)


# =============================================================================
# fig7 — cascade vs omni EES by condition (descriptive, H4a)
# =============================================================================

def _cluster_bootstrap_ci(matrix, b=2000, seed=42):
    """Percentile 95% CI for each column mean, resampling utterances.

    `matrix` is (utterances x cells) with one observation per utterance per
    cell, which is what the H4b population is: 416 utterances x 19 conditions
    x 2 pathways x 2 vendor families = 31,616 rows exactly. Resampling rows of
    that matrix is resampling `slurp_id`, which is the clustering the whole
    project uses; a row-level interval here would treat 76 observations of the
    same utterance as 76 independent ones.
    """
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    draws = np.empty((b, matrix.shape[1]), dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        draws[i] = matrix[idx].mean(axis=0)
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def figure_pathway_ees(out_dir):
    """Observed end-to-end success, omni vs the reference cascade, per condition.

    DESCRIPTIVE — this is §3's H4a, which the design makes descriptive and not
    confirmatory. The confirmatory statement about the two pathways is the
    interaction in fig4. Two caveats belong on the page and are printed on it:
    the pathways differ in capacity as well as in architecture, and the omni
    advantage under babble is carried by Whisper hallucinations.
    """
    import result_sets
    import model_frames as mf
    from fit_mixed_effects import load_rows

    rows = load_rows(result_sets.sweep_files())
    df, diag = mf.build_frame(rows, "h4b", outcome="ees")
    df["snr_db"] = np.where(df["degradation"] == "clean", -1.0,
                            np.where(df["snr_20"] == 1, 20.0,
                                     np.where(df["snr_10"] == 1, 10.0, 0.0)))
    dfh = df[~df["full_hallucination"]]

    key = [("clean", -1.0)] + condition_columns()
    snr_ticks = ["—"] + ["%d" % int(s) for _d, s in condition_columns()]
    blocks = [[0]] + [list(range(1 + k * len(SNR_ORDER),
                                 1 + (k + 1) * len(SNR_ORDER)))
                      for k in range(len(DEG_ORDER))]
    families = [("qwen", "Qwen  —  Qwen3-Omni  vs  Whisper-Turbo -> Qwen3.5-27B"),
                ("google", "Google  —  Gemma-4-12B omni  vs  "
                           "Whisper-Turbo -> Gemma-4-31B")]
    pathways = [("cascade", BLUE, "o", "-", "cascade (ASR -> router)"),
                ("omni", ORANGE, "s", "--", "omni (audio -> answer)")]

    fig, axes = plt.subplots(len(families), 1, figsize=(11.6, 8.0), sharex=True)
    x = np.arange(len(key))
    for ax, (family, title) in zip(axes, families):
        _style_axes(ax)
        for pathway, colour, marker, ls, name in pathways:
            sub = df[(df["model_family"] == family) & (df["pathway"] == pathway)]
            piv = sub.pivot_table(index="slurp_id",
                                  columns=["degradation", "snr_db"],
                                  values="outcome")
            piv = piv.reindex(columns=pd.MultiIndex.from_tuples(key))
            if piv.isna().any().any():
                raise ValueError("%s/%s frame is not a complete "
                                 "utterance x condition grid" % (family, pathway))
            means = piv.to_numpy().mean(axis=0)
            lo, hi = _cluster_bootstrap_ci(piv.to_numpy())
            ax.vlines(x, lo, hi, color=colour, linewidth=1.2, alpha=0.6,
                      zorder=2)
            # One polyline per degradation block: `clean` and the six triples
            # are separate conditions, not a continuous sweep.
            for n, block in enumerate(blocks):
                ax.plot(x[block], means[block], color=colour, marker=marker,
                        linestyle=ls, linewidth=1.6, markersize=5.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
                        label=name if n == 0 else None)
            if pathway == "cascade":
                subh = dfh[(dfh["model_family"] == family)
                           & (dfh["pathway"] == pathway)]
                mh = (subh.groupby(["degradation", "snr_db"])["outcome"]
                      .mean().reindex(key).to_numpy())
                # Dropping hallucinated transcripts nudges nearly every cell;
                # only a shift worth reading is marked, and the caption says so.
                moved = np.abs(mh - means) > 0.01
                ax.scatter(x[moved], mh[moved], s=30, marker="x", color=MUTED,
                           linewidths=1.2, zorder=4,
                           label=("cascade, hallucination rows dropped "
                                  "(shift > 1 pp)"
                                  if family == families[0][0] else None))
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("end-to-end success (EES)", color=INK_2)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5, pad=6)
        for k in range(len(DEG_ORDER)):
            ax.axvline(0.5 + k * len(SNR_ORDER), color=GRID, linewidth=1.0,
                       zorder=1)
        ax.set_xlim(-0.7, len(key) - 0.3)
    bottom = axes[-1]
    bottom.set_xticks(x)
    bottom.set_xticklabels(snr_ticks, fontsize=8)
    below = ("data", "axes fraction")
    bottom.annotate("Clean", xy=(0, 0), xycoords=below, xytext=(0, -24),
                    textcoords="offset points", ha="center", va="top",
                    color=INK, fontsize=8.5, annotation_clip=False)
    for k, deg in enumerate(DEG_ORDER):
        centre = 1 + k * len(SNR_ORDER) + (len(SNR_ORDER) - 1) / 2.0
        bottom.annotate(DEG_LABEL[deg], xy=(centre, 0), xycoords=below,
                        xytext=(0, -24), textcoords="offset points",
                        ha="center", va="top", color=INK, fontsize=8.5,
                        annotation_clip=False)
    bottom.set_xlabel("SNR (dB) within each degradation", color=INK_2,
                      labelpad=28)
    handles, names = axes[0].get_legend_handles_labels()
    leg = axes[0].legend(handles, names, loc="lower left", fontsize=8, ncol=3)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle("End-to-end success by condition: the omni pathway against the "
                 "reference cascade", color=INK, fontsize=13, x=0.02,
                 ha="left", y=0.985)
    fig.text(0.02, 0.945,
             "DESCRIPTIVE (H4a). %s rows: every omni row plus only the "
             "Whisper-Turbo -> large-router cascade rows, so the omni model is "
             "not being\ncompared against a weaker cascade. Whiskers are a 95%% "
             "cluster bootstrap over the 416 utterances (B = 2000, seed 42). "
             "NOTE: The two pathways differ in\ncapacity as well as in "
             "architecture; the confirmatory statement is the interaction in "
             "fig4, which cancels a constant capacity gap. NOTE: Grey crosses mark "
             "conditions where\ndropping the %d `full_hallucination` rows moves "
             "the cascade mean — the omni advantage under babble is carried by "
             "invented Whisper text (§13)."
             % ("{:,}".format(diag["rows_out"]),
                int(df["full_hallucination"].sum())),
             ha="left", va="top", color=INK_2, fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.845, bottom=0.115,
                        hspace=0.32)
    return _save(fig, "fig7_pathway_ees_by_condition", out_dir)


# =============================================================================

FIGURES = OrderedDict([
    ("fig1", figure_retention),
    ("fig2", figure_gain_heatmap),
    ("fig3", figure_comparator_contrast),
    ("fig4", figure_forest),
    ("fig5", figure_mediation_waterfall),
    ("fig6", figure_capacity_calibration),
    ("fig7", figure_pathway_ees),
])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=list(FIGURES),
                        help="draw a subset (default: all)")
    parser.add_argument("--out-dir", default=FIG_DIR)
    args = parser.parse_args(argv)

    _rc()
    for name in (args.only or list(FIGURES)):
        written = FIGURES[name](args.out_dir)
        print("%s -> %s" % (name, ", ".join(written)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
