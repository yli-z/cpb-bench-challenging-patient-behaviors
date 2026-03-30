"""
plot_negative_distribution.py

Plot the truncation-point distribution of the re-sampled negative cases
(actual_pct = turn_index / total_turns) in the same style as the positive
case distribution plot (negative_case_distribution.png).

Output: Negative_cases/negative_sampled_distribution.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SAMPLE_DIR = Path(__file__).resolve().parent          # …/Negative_sampling_segment/
ROOT       = SAMPLE_DIR.parent.parent                  # …/CPB-Bench/
OUT_PATH   = SAMPLE_DIR / "negative_sampled_distribution.png"

DATASET_FILES = {
    "ACI":     "ACI_negative_cases_sampled.json",
    "IMCS21":  "IMCS21_negative_cases_sampled.json",
    "MedDG":   "MedDG_negative_cases_sampled.json",
    "MediTOD": "MediTOD_negative_cases_sampled.json",
}

# Colors matching the positive-case plot
COLORS = {
    "ACI":     "#5B8DB8",   # steel blue
    "IMCS21":  "#E08C5A",   # orange
    "MedDG":   "#6BAE72",   # green
    "MediTOD": "#C96B78",   # rose/red
    "Overall": "#8E85C2",   # purple
}

# ── Load data ─────────────────────────────────────────────────────────────────
pcts_per_ds: dict[str, list] = {}
for ds, fname in DATASET_FILES.items():
    with open(SAMPLE_DIR / fname, encoding="utf-8") as f:
        records = json.load(f)
    pcts_per_ds[ds] = [r["actual_pct"] for r in records]

all_pcts = [p for v in pcts_per_ds.values() for p in v]

# ── Plot helper ───────────────────────────────────────────────────────────────
def plot_panel(ax, pcts, label, color, n_bins=10):
    pcts = np.array(pcts)
    n    = len(pcts)

    # Histogram
    ax.hist(pcts, bins=n_bins, density=True,
            color=color, alpha=0.45, edgecolor="white", linewidth=0.6,
            label="Histogram")

    # KDE
    if n > 1:
        xs = np.linspace(0, 1, 300)
        kde = gaussian_kde(pcts, bw_method="scott")
        ys  = kde(xs)
        ax.plot(xs, ys, color=color, linewidth=2.2, label="KDE")

    # Percentile lines
    p25, p50, p75 = np.percentile(pcts, [25, 50, 75])
    ax.axvline(p25, color="black", linestyle="--", linewidth=1.2,
               label=f"p25={p25:.2f}")
    ax.axvline(p50, color="black", linestyle="-",  linewidth=1.2,
               label=f"p50={p50:.2f}")
    ax.axvline(p75, color="black", linestyle=":",  linewidth=1.5,
               label=f"p75={p75:.2f}")

    # Mean / std text box
    mean_, std_ = pcts.mean(), pcts.std()
    ax.text(0.98, 0.97, f"mean={mean_:.2f}\nstd={std_:.2f}",
            transform=ax.transAxes,
            va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.8))

    ax.set_title(f"{label}  (n={n})", fontsize=12, fontweight="bold")
    ax.set_xlabel("turn_index / total_turns", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=8)

    # Remove outer box (all four spines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(fontsize=7.5, loc="upper left",
              framealpha=0.8, edgecolor="grey")

# ── Build figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "Negative Case Sampled Truncation-Point Distribution\n"
    "(actual_pct = turn_index / total_turns, 1 sample per case, "
    "bootstrapped from positive-case distribution)",
    fontsize=13, fontweight="bold", y=1.01
)

from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Top row: ACI, IMCS21, MedDG
ax_aci     = fig.add_subplot(gs[0, 0])
ax_imcs    = fig.add_subplot(gs[0, 1])
ax_meddg   = fig.add_subplot(gs[0, 2])

# Bottom-left: MediTOD (1 cell)
ax_meditod = fig.add_subplot(gs[1, 0])

# Bottom-center + right: Overall (spans 2 cells → wide rectangle)
ax_overall = fig.add_subplot(gs[1, 1:])

for ax, ds in zip([ax_aci, ax_imcs, ax_meddg], ["ACI", "IMCS21", "MedDG"]):
    plot_panel(ax, pcts_per_ds[ds], ds, COLORS[ds])

plot_panel(ax_meditod, pcts_per_ds["MediTOD"], "MediTOD", COLORS["MediTOD"])
plot_panel(ax_overall, all_pcts, "Overall", COLORS["Overall"], n_bins=14)

plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
plt.show()
