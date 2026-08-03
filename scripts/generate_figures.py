"""
generate_figures.py
Generates all evaluation figures for the thesis.
Run from the repository root:  python code/scripts/generate_figures.py
Output: thesis/figures/*.pdf  (one file per figure)
"""

import json
import os
import statistics
from collections import defaultdict, Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
THESIS = os.path.join(BASE, "..", "..", "thesis")  # code/scripts/ → Thesis/
FIG_DIR = os.path.join(THESIS, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DATA_PATHS = {
    "gemma": {
        "llm":    os.path.join(BASE, "../llm-base/evaluation/results/gemma4-31b/evaluation_results.json"),
        "hybrid": os.path.join(BASE, "../hybrid/evaluation/results/gemma4-31b/evaluation_results.json"),
    },
    "gpt4omini": {
        "llm":    os.path.join(BASE, "../llm-base/evaluation/results/gpt-4o-mini/evaluation_results.json"),
        "hybrid": os.path.join(BASE, "../hybrid/evaluation/results/gpt-4o-mini/evaluation_results.json"),
    },
    "gpt54": {
        "llm":    os.path.join(BASE, "../llm-base/evaluation/results/gpt-5.4-mini/evaluation_results.json"),
        "hybrid": os.path.join(BASE, "../hybrid/evaluation/results/gpt-5.4-mini/evaluation_results.json"),
    },
    "qwen": {
        "llm":    os.path.join(BASE, "../llm-base/evaluation/results/qwen2.5-7b_local/evaluation_results.json"),
        "hybrid": os.path.join(BASE, "../hybrid/evaluation/results/qwen2.5-7b_local/evaluation_results.json"),
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Style constants
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8.5,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Colour palette  (condition → colour)
COND_COLORS = {
    "baseline":                 "#4e79a7",
    "multi_agent_no_context":   "#f28e2b",
    "multi_agent_with_context": "#e15759",
    "hybrid_no_context":        "#76b7b2",
    "hybrid_with_context":      "#59a14f",
}
COND_LABELS = {
    "baseline":                 "Baseline\n(Single Agent)",
    "multi_agent_no_context":   "MAS\nNo Context",
    "multi_agent_with_context": "MAS\nWith Context",
    "hybrid_no_context":        "Hybrid\nNo Context",
    "hybrid_with_context":      "Hybrid\nWith Context",
}
MODEL_LABELS = {
    "gemma":     "Gemma 4 31B",
    "gpt4omini": "GPT-4o-mini",
    "gpt54":     "GPT-5.4-mini",
    "qwen":      "Qwen2.5-7B",
}
CAT_LABELS = {
    "simple":       "Simple",
    "multi_step":   "Multi-Step",
    "cross_system": "Cross-System",
    "error_prone":  "Error-Prone",
}

# ──────────────────────────────────────────────────────────────────────────────
# Data loading & helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_all():
    records = []
    for model, archs in DATA_PATHS.items():
        for arch, path in archs.items():
            with open(path) as f:
                rows = json.load(f)
            for r in rows:
                r["model"] = model
                r["arch"]  = arch
            records.extend(rows)
    return records


def actual_outcome(r):
    if r["escalated"]:    return "escalation"
    if r["success"]:      return "success"
    if r["error_count"] > 0: return "error"
    return "error"


def is_match(r):
    return actual_outcome(r) == r["expected_outcome"]


def agg(records, key_fn, val_fn):
    """Aggregate: returns {key: list_of_values}."""
    d = defaultdict(list)
    for r in records:
        d[key_fn(r)].append(val_fn(r))
    return d


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
ALL = load_all()
print(f"Loaded {len(ALL)} records total.\n")

# Pre-build lookup: (model, condition) → list of records
by_mc = defaultdict(list)
for r in ALL:
    by_mc[(r["model"], r["condition"])].append(r)

# All conditions in display order
ALL_CONDITIONS = [
    "baseline",
    "multi_agent_no_context",
    "multi_agent_with_context",
    "hybrid_no_context",
    "hybrid_with_context",
]
LLM_CONDITIONS  = ALL_CONDITIONS[:3]
HYBRID_CONDITIONS = ALL_CONDITIONS[3:]

ALL_MODELS = ["gemma", "gpt4omini", "gpt54", "qwen"]
HYBRID_MODELS = ["gemma", "gpt4omini", "gpt54", "qwen"]


def match_rate(model, condition):
    rows = by_mc[(model, condition)]
    if not rows:
        return None
    return 100 * sum(is_match(r) for r in rows) / len(rows)


def raw_rate(model, condition):
    rows = by_mc[(model, condition)]
    if not rows:
        return None
    return 100 * sum(r["success"] for r in rows) / len(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Primary results: Outcome Match Rate by model × condition
# ══════════════════════════════════════════════════════════════════════════════
def fig_outcome_match():
    models = ALL_MODELS
    conditions = ALL_CONDITIONS
    n_models = len(models)
    n_conds  = len(conditions)
    bar_w    = 0.14
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(10, 5))

    offsets = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * bar_w

    for i, cond in enumerate(conditions):
        vals = [match_rate(m, cond) for m in models]
        # Replace None with 0 for plotting; track which bars are N/A
        plot_vals = [v if v is not None else 0.0 for v in vals]
        na_flags  = [v is None for v in vals]

        bars = ax.bar(
            x + offsets[i], plot_vals, bar_w,
            color=COND_COLORS[cond],
            label=COND_LABELS[cond].replace("\n", " "),
            zorder=3,
        )
        for bar, v, na in zip(bars, vals, na_flags):
            if na:
                bar.set_hatch("///")
                bar.set_alpha(0.25)
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.6,
                    f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel("Outcome Match Rate (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.axhline(y=50, color="gray", linewidth=0.7, linestyle=":", zorder=0)
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.set_title("Outcome Match Rate by Model and Condition")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)

    save(fig, "fig1_outcome_match_rate.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Metric Divergence: Raw Success vs Outcome Match
# ══════════════════════════════════════════════════════════════════════════════
def fig_metric_divergence():
    """For each model show raw-success and outcome-match side by side,
    with an arrow showing the divergence direction."""

    conditions = ALL_CONDITIONS
    models = ALL_MODELS

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5), sharey=True)

    for ax, model in zip(axes, models):
        raw_vals   = []
        match_vals = []
        labels     = []
        colors     = []

        for cond in conditions:
            rv = raw_rate(model, cond)
            mv = match_rate(model, cond)
            if rv is None:
                continue
            raw_vals.append(rv)
            match_vals.append(mv)
            labels.append(COND_LABELS[cond].replace("\n", " "))
            colors.append(COND_COLORS[cond])

        y = np.arange(len(labels))
        bar_h = 0.35

        ax.barh(y + bar_h / 2, raw_vals,   bar_h, color=colors, alpha=0.45, label="Raw Success")
        ax.barh(y - bar_h / 2, match_vals, bar_h, color=colors, alpha=0.95, label="Outcome Match")

        # Divergence arrow annotation
        for yi, (rv, mv) in enumerate(zip(raw_vals, match_vals)):
            diff = mv - rv
            color = "#1a7d3a" if diff > 0 else "#c0392b"
            ax.annotate(
                f"{diff:+.0f}pp",
                xy=(max(rv, mv) + 0.5, yi),
                va="center", ha="left", fontsize=7, color=color, fontweight="bold",
            )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 105)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.axvline(x=50, color="gray", linewidth=0.6, linestyle=":", zorder=0)
        ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)

    # Shared legend — placed below subplots
    legend_patches = [
        mpatches.Patch(color="gray", alpha=0.45, label="Raw Success Rate"),
        mpatches.Patch(color="gray", alpha=0.95, label="Outcome Match Rate (Primary)"),
    ]
    fig.legend(handles=legend_patches, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8,
               frameon=True)

    fig.suptitle(
        "Raw Success Rate vs Outcome Match Rate per Condition and Model",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, "fig2_metric_divergence.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — LLM Call Counts: distribution per condition (GPT-5.4-mini)
# ══════════════════════════════════════════════════════════════════════════════
def fig_llm_calls():
    model = "gpt54"
    conditions = ALL_CONDITIONS

    fig, axes = plt.subplots(1, len(conditions), figsize=(12, 3.8), sharey=False)

    for ax, cond in zip(axes, conditions):
        rows = by_mc[(model, cond)]
        if not rows:
            ax.set_visible(False)
            continue
        counts = [r["llm_calls_total"] for r in rows]
        freq = Counter(counts)
        x_vals = sorted(freq.keys())
        y_vals = [100 * freq[x] / len(counts) for x in x_vals]

        ax.bar(x_vals, y_vals, color=COND_COLORS[cond], zorder=3, width=0.7)
        ax.axvline(x=statistics.mean(counts), color="black", linewidth=1.2,
                   linestyle="--", label=f"mean={statistics.mean(counts):.1f}")
        ax.set_title(COND_LABELS[cond], fontsize=9)
        ax.set_xlabel("LLM Calls / Run")
        ax.set_ylabel("% of Runs" if cond == conditions[0] else "")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)

    fig.suptitle("Distribution of LLM Calls per Run — GPT-5.4-mini", fontsize=11)
    fig.tight_layout()
    save(fig, "fig3_llm_call_distribution.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Performance by Scenario Category
# ══════════════════════════════════════════════════════════════════════════════
def fig_category_breakdown():
    cats = ["simple", "multi_step", "cross_system", "error_prone"]
    # Show 5 conditions for both available models (gemma & gpt54)
    highlight_conditions = ALL_CONDITIONS
    highlight_models = ["gemma", "gpt54"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, model in zip(axes, highlight_models):
        n_cats  = len(cats)
        n_conds = len(highlight_conditions)
        bar_w   = 0.12
        x = np.arange(n_cats)
        offsets = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * bar_w

        for i, cond in enumerate(highlight_conditions):
            vals = []
            for cat in cats:
                rows = [r for r in by_mc[(model, cond)] if r["scenario_category"] == cat]
                vals.append(100 * sum(is_match(r) for r in rows) / len(rows) if rows else 0)
            ax.bar(
                x + offsets[i], vals, bar_w,
                color=COND_COLORS[cond],
                label=COND_LABELS[cond].replace("\n", " "),
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([CAT_LABELS[c] for c in cats], fontsize=9)
        ax.set_title(MODEL_LABELS[model])
        ax.set_ylim(0, 108)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
        ax.axhline(y=50, color="gray", linewidth=0.6, linestyle=":", zorder=0)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
        if model == highlight_models[0]:
            ax.set_ylabel("Outcome Match Rate (%)")
        ax.legend(ncol=3, fontsize=7.5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18))

    fig.suptitle("Outcome Match Rate by Scenario Category and Condition", fontsize=11)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    save(fig, "fig4_category_breakdown.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Context Effect: delta (with-ctx − no-ctx) per model & architecture
# ══════════════════════════════════════════════════════════════════════════════
def fig_context_effect():
    comparisons = [
        ("gemma",     "multi_agent_no_context",  "multi_agent_with_context",  "LLM-based"),
        ("gpt4omini", "multi_agent_no_context",  "multi_agent_with_context",  "LLM-based"),
        ("gpt54",     "multi_agent_no_context",  "multi_agent_with_context",  "LLM-based"),
        ("qwen",      "multi_agent_no_context",  "multi_agent_with_context",  "LLM-based"),
        ("gemma",     "hybrid_no_context",        "hybrid_with_context",       "Hybrid"),
        ("gpt4omini", "hybrid_no_context",        "hybrid_with_context",       "Hybrid"),
        ("gpt54",     "hybrid_no_context",        "hybrid_with_context",       "Hybrid"),
        ("qwen",      "hybrid_no_context",        "hybrid_with_context",       "Hybrid"),
    ]

    labels  = []
    deltas  = []
    colors  = []
    arches  = []

    for model, cond_no, cond_yes, arch in comparisons:
        mv_no  = match_rate(model, cond_no)
        mv_yes = match_rate(model, cond_yes)
        if mv_no is None or mv_yes is None:
            continue
        delta = mv_yes - mv_no
        labels.append(f"{MODEL_LABELS[model]}\n({arch})")
        deltas.append(delta)
        colors.append("#59a14f" if delta >= 0 else "#e15759")
        arches.append(arch)

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(labels))
    bars = ax.barh(y, deltas, color=colors, zorder=3, height=0.6)

    for bar, delta in zip(bars, deltas):
        sign = "+" if delta >= 0 else ""
        label_text = f"{sign}{delta:.1f} pp"
        bar_width = abs(delta)
        # For large bars, place annotation inside with white text
        if bar_width >= 5.0:
            x_pos = delta / 2
            ha = "center"
            txt_color = "white"
        else:
            x_pos = delta + (0.2 if delta >= 0 else -0.2)
            ha = "left" if delta >= 0 else "right"
            txt_color = "#333333"
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            label_text,
            va="center", ha=ha,
            fontsize=8.5, fontweight="bold", color=txt_color,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_xlabel("Change in Outcome Match Rate (with context − without context, pp)")
    ax.set_title("Effect of Shared Context on Outcome Match Rate")
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)

    legend_patches = [
        mpatches.Patch(color="#59a14f", label="Context improves performance"),
        mpatches.Patch(color="#e15759", label="Context degrades performance"),
    ]
    fig.legend(handles=legend_patches, fontsize=8.5,
               loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, frameon=True)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig5_context_effect.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Scenario Heatmap: per-scenario outcome match (GPT-5.4-mini)
# ══════════════════════════════════════════════════════════════════════════════
def fig_scenario_heatmap():
    model = "gpt54"
    conditions = ALL_CONDITIONS

    # Build 50×5 matrix of match rates (each cell = mean over 5 runs)
    scenarios = sorted(set(r["scenario_id"] for r in ALL))  # S01-S50

    matrix = np.full((len(scenarios), len(conditions)), np.nan)
    cats   = []  # scenario category per row

    cat_lookup = {}
    for r in ALL:
        cat_lookup[r["scenario_id"]] = r["scenario_category"]

    for si, scen in enumerate(scenarios):
        cats.append(cat_lookup.get(scen, ""))
        for ci, cond in enumerate(conditions):
            rows = [r for r in by_mc[(model, cond)] if r["scenario_id"] == scen]
            if rows:
                matrix[si, ci] = 100 * sum(is_match(r) for r in rows) / len(rows)

    fig, ax = plt.subplots(figsize=(8, 13))

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0, vmax=100,
        interpolation="nearest",
    )

    # Grid lines
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(
        [COND_LABELS[c].replace("\n", " ") for c in conditions],
        rotation=25, ha="right", fontsize=8.5,
    )
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=7)

    # Colour the y-tick labels by category
    cat_colors = {
        "simple":       "#4e79a7",
        "multi_step":   "#f28e2b",
        "cross_system": "#e15759",
        "error_prone":  "#76b7b2",
    }
    for yi, (scen, cat) in enumerate(zip(scenarios, cats)):
        ax.get_yticklabels()[yi].set_color(cat_colors.get(cat, "black"))

    # Annotate cells with values
    for si in range(len(scenarios)):
        for ci in range(len(conditions)):
            v = matrix[si, ci]
            if not np.isnan(v):
                ax.text(
                    ci, si, f"{v:.0f}",
                    ha="center", va="center",
                    fontsize=5.5,
                    color="black" if 20 < v < 80 else "white",
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label("Outcome Match Rate (%)", fontsize=8)

    # Category legend for y-axis colours
    legend_patches = [
        mpatches.Patch(color=c, label=CAT_LABELS[k])
        for k, c in cat_colors.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
              bbox_to_anchor=(1.22, 1.0), title="Category")

    ax.set_title(f"Per-Scenario Outcome Match Rate — {MODEL_LABELS[model]}\n"
                 "(Y-axis colour indicates scenario category)")

    fig.tight_layout()
    save(fig, "fig6_scenario_heatmap.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Cost vs Performance scatter (latency × outcome-match trade-off)
# ══════════════════════════════════════════════════════════════════════════════
def fig_cost_performance():
    """Scatter: x=mean latency, y=outcome match rate.
    Color encodes condition; marker shape encodes model.
    No inline text labels — two-part legend instead."""

    # Distinct marker per model
    MODEL_MARKERS = {
        "gemma":     "o",   # circle
        "gpt4omini": "s",   # square
        "gpt54":     "^",   # triangle-up
        "qwen":      "D",   # diamond
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for (model, cond), rows in by_mc.items():
        if not rows:
            continue
        om = match_rate(model, cond)
        if om is None:
            continue
        lat = statistics.mean(r["elapsed_time"] for r in rows)
        ax.scatter(
            lat, om,
            s=90,
            marker=MODEL_MARKERS.get(model, "o"),
            color=COND_COLORS.get(cond, "gray"),
            zorder=3,
            alpha=0.90,
            linewidths=0.6,
            edgecolors="white",
        )

    ax.set_xlabel("Mean Elapsed Time per Run (s)")
    ax.set_ylabel("Outcome Match Rate (%)")
    ax.set_title(
        "Latency vs Outcome Match Rate — All Conditions and Models\n"
        "(Lower-right corner = ideal: fast and accurate)",
        pad=14,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)

    # Legend part 1 — condition colors (colored patches)
    cond_handles = [
        mpatches.Patch(color=COND_COLORS[c],
                       label=COND_LABELS[c].replace("\n", " "))
        for c in ALL_CONDITIONS
    ]
    # Legend part 2 — model marker shapes (gray markers)
    model_handles = [
        mlines.Line2D([], [], color="gray",
                      marker=MODEL_MARKERS[m], linestyle="None",
                      markersize=7, label=MODEL_LABELS[m])
        for m in ALL_MODELS
    ]

    # Blank spacer between the two groups
    spacer = mpatches.Patch(color="none", label=" ")

    # Single combined legend: Condition group, spacer, Model group
    ax.legend(
        handles=cond_handles + [spacer] + model_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        handlelength=1.4,
    )

    fig.tight_layout()
    save(fig, "fig7_latency_vs_outcome.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Outcome distribution stacked bar (what went wrong per condition)
# ══════════════════════════════════════════════════════════════════════════════
def fig_outcome_distribution():
    """Stacked bar showing proportion of success / error / escalation
    per condition for the two primary models."""

    models = ["gpt54", "gemma"]
    outcome_types = ["success", "error", "escalation"]
    outcome_colors = {
        "success":    "#59a14f",
        "error":      "#e15759",
        "escalation": "#b07aa1",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, model in zip(axes, models):
        conditions_avail = [c for c in ALL_CONDITIONS if by_mc[(model, c)]]
        x = np.arange(len(conditions_avail))
        bottoms = np.zeros(len(conditions_avail))

        for ot in outcome_types:
            vals = []
            for cond in conditions_avail:
                rows = by_mc[(model, cond)]
                count = sum(1 for r in rows if actual_outcome(r) == ot)
                vals.append(100 * count / len(rows) if rows else 0)
            ax.bar(x, vals, bottom=bottoms, color=outcome_colors[ot],
                   label=ot.capitalize(), zorder=3)
            bottoms += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [COND_LABELS[c].replace("\n", " ") for c in conditions_avail],
            rotation=18, ha="right", fontsize=8.5,
        )
        ax.set_title(MODEL_LABELS[model])
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
        if model == models[0]:
            ax.set_ylabel("Proportion of Runs (%)")
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4, zorder=0)
        ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("Actual Outcome Distribution per Condition and Model\n"
                 "(Outcome Match requires correct outcome type, not only 'success')",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig8_outcome_distribution.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    fig_outcome_match()
    fig_metric_divergence()
    fig_llm_calls()
    fig_category_breakdown()
    fig_context_effect()
    fig_scenario_heatmap()
    fig_cost_performance()
    fig_outcome_distribution()
    print(f"\nAll figures saved to: {FIG_DIR}")
