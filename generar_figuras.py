"""
Genera las figuras del analisis de ablacion para el paper (formato IEEE).

Salida en figuras/:
    fig_cycles.pdf     ciclos depredador-presa emergentes (hallazgo principal)
    fig_ablation.pdf   efecto pareado de cada condicion LLM vs linea base
    fig_learning.pdf   curvas de aprendizaje por condicion

Uso:
    python generar_figuras.py
"""
import csv
import statistics
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / "experiment_results"
OUT = Path(__file__).resolve().parent / "figuras"
OUT.mkdir(exist_ok=True)

CONDITIONS = {
    "Baseline (PPO)":        "base_2000",
    "Reward shaping":        "reward-shaping_2000",
    "Behavior selector":     "behavior_selector_2000",
    "Shaping + selector":    "reward-shaping_and_behavior-selector_2000",
    "Shaping + reflexion":   "reward-shaping_and_reflexion_2000",
    "Selector + reflexion":  "behavior-selector_and_reflexion_2000",
    "All three modes":       "all_2000",
}

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

COL_W = 3.5      # ancho de una columna IEEE (pulgadas)
DOUBLE_W = 7.16  # ancho de dos columnas


def read_run(monitor: Path) -> dict:
    rows = list(csv.DictReader(open(monitor, encoding="utf-8-sig")))
    out = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[k] = np.array(vals)
    return out


def runs_of(folder: str) -> list:
    return [read_run(p) for p in sorted((ROOT / folder).glob("run_*/monitor.csv"))]


def smooth(v, w=50):
    return np.convolve(v, np.ones(w) / w, mode="valid")


# ── Figura 1: ciclos depredador-presa ────────────────────────────────────
def fig_cycles():
    runs = runs_of("base_2000")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 2.1), sharey=True)

    for ax, run, seed in zip(axes, runs, [1005, 1006, 1007]):
        prey = smooth(run["herbivore_survival_pct"])
        kills = smooth(run["predator_attack_kill_count"])
        x = np.arange(len(prey))

        ax.plot(x, prey, color="#2a78d6", lw=1.2, label="Prey survival (%)")
        ax.set_xlabel("Training iteration")
        ax.set_ylim(0, 70)

        ax2 = ax.twinx()
        ax2.plot(x, kills, color="#e34948", lw=1.2, ls="--", label="Kills / episode")
        ax2.set_ylim(0, 3.2)
        ax2.grid(False)
        if ax is axes[-1]:
            ax2.set_ylabel("Kills per episode", color="#e34948")
        else:
            ax2.set_yticklabels([])
        ax2.tick_params(axis="y", colors="#e34948", labelsize=6)

        ax.set_title(f"Seed {seed}")

    axes[0].set_ylabel("Prey survival (%)", color="#2a78d6")
    axes[0].tick_params(axis="y", colors="#2a78d6")

    h1, l1 = axes[0].get_legend_handles_labels()
    h2 = [plt.Line2D([], [], color="#e34948", ls="--", lw=1.2)]
    fig.legend(h1 + h2, ["Prey survival (%)", "Kills per episode"],
               loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_cycles.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_cycles.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_cycles.pdf")


# ── Figura 2: ablacion pareada ───────────────────────────────────────────
METRICS = [
    ("herbivore_survival_pct",     "Prey\nsurvival",       "up"),
    ("predator_survival_pct",      "Predator\nsurvival",   "up"),
    ("predator_attack_kill_count", "Kills per\nepisode",   "up"),
    ("herbivore_avg_pred_dist",    "Predator\ndistance",   "up"),
    ("herbivore_avg_nn_dist",      "Prey\ngrouping",       "down"),
]
TAIL = 500


def fig_ablation():
    data = {}
    for label, folder in CONDITIONS.items():
        per_seed = []
        for run in runs_of(folder):
            per_seed.append({k: np.nanmean(v[-TAIL:]) for k, v in run.items()})
        data[label] = per_seed

    base = data["Baseline (PPO)"]
    others = [l for l in CONDITIONS if l != "Baseline (PPO)"]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(DOUBLE_W, 2.3))

    for ax, (key, title, better) in zip(axes, METRICS):
        means, errs = [], []
        for label in others:
            deltas = [data[label][i][key] - base[i][key] for i in range(3)]
            signed = [(-d if better == "down" else d) for d in deltas]
            rel = [100 * s / abs(np.mean([b[key] for b in base])) for s in signed]
            means.append(np.mean(rel))
            errs.append(np.std(rel, ddof=1))

        y = np.arange(len(others))
        colors = ["#1baf7a" if m > 0 else "#e34948" for m in means]
        ax.barh(y, means, xerr=errs, color=colors, height=0.6,
                error_kw={"lw": 0.8, "capsize": 2, "ecolor": "#666"})
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(others if ax is axes[0] else [])
        ax.set_title(title)
        ax.set_xlabel("Change vs\nbaseline (%)")
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_ablation.pdf")


# ── Figura 3: curvas de aprendizaje ──────────────────────────────────────
def fig_learning():
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.4))
    palette = ["#333333", "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4", "#eb6834"]

    for (label, folder), color in zip(CONDITIONS.items(), palette):
        runs = runs_of(folder)
        for ax, key, ylabel in [
            (axes[0], "herbivore_survival_pct", "Prey survival (%)"),
            (axes[1], "predator_survival_pct", "Predator survival (%)"),
        ]:
            curves = np.vstack([smooth(r[key], 100) for r in runs])
            mean = curves.mean(axis=0)
            x = np.arange(len(mean))
            lw = 1.6 if label.startswith("Baseline") else 1.0
            ax.plot(x, mean, color=color, lw=lw, label=label)
            ax.set_xlabel("Training iteration")
            ax.set_ylabel(ylabel)

    axes[1].legend(loc="lower right", frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "fig_learning.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_learning.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_learning.pdf")


if __name__ == "__main__":
    print("Generando figuras en figuras/ ...")
    fig_cycles()
    fig_ablation()
    fig_learning()
    print("Listo.")
