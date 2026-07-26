#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CURVE_METRICS = {
    "r_mean": ("Retorno promedio entre ejecuciones", "Return"),
    "l_mean": ("Longitud promedio de episodio", "Timesteps"),
    "herbivore_episode_return": ("Return promedio de herbívoros", "Return"),
    "predator_episode_return": ("Return promedio de depredadores", "Return"),
    "herbivore_survival_pct": ("Supervivencia promedio de herbívoros", "Porcentaje"),
    "predator_survival_pct": ("Supervivencia promedio de depredadores", "Porcentaje"),
    "herbivore_critical_ratio": ("Critical ratio promedio de herbívoros", "Ratio"),
    "predator_critical_ratio": ("Critical ratio promedio de depredadores", "Ratio"),
    "attacks_attempted": ("Intentos de ataque promedio", "Conteo"),
    "attacks_kill": ("Muertes por ataque promedio", "Conteo"),
    "attack_hit_rate": ("Tasa promedio de aciertos", "Rate"),
    "attack_kill_rate": ("Tasa promedio de ataques letales", "Rate"),
}

FINAL_METRICS = [
    "r_mean", "l_mean",
    "herbivore_episode_return", "predator_episode_return",
    "herbivore_survival_pct", "predator_survival_pct",
    "herbivore_reason_timeout_pct",
    "herbivore_reason_starvation_pct",
    "herbivore_reason_dehydration_pct",
    "herbivore_reason_predation_pct",
    "predator_reason_timeout_pct",
    "predator_reason_starvation_pct",
    "predator_reason_dehydration_pct",
    "predator_reason_predation_pct",
    "attacks_attempted", "attacks_hit", "attacks_kill",
    "attack_hit_rate", "attack_kill_rate",
    "herbivore_avg_food", "herbivore_avg_water",
    "predator_avg_food", "predator_avg_water",
    "herbivore_critical_ratio", "predator_critical_ratio",
]


def load_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró: {path}")
    df = pd.read_csv(path)
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} no contiene las columnas requeridas: "
            + ", ".join(sorted(missing))
        )
    return df


def plot_learning_curve(
    curves: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_dir: Path,
    smooth_window: int,
    show: bool,
) -> None:
    data = curves[curves["metric"] == metric].copy()
    if data.empty:
        print(f"[Aviso] No se encontró la métrica: {metric}")
        return

    data["iter"] = pd.to_numeric(data["iter"], errors="coerce")
    data["mean"] = pd.to_numeric(data["mean"], errors="coerce")
    data["standard_deviation"] = pd.to_numeric(
        data["standard_deviation"], errors="coerce"
    )
    data = data.dropna(subset=["iter", "mean"]).sort_values("iter")
    if data.empty:
        return

    window = max(1, min(smooth_window, len(data)))
    mean = data["mean"].rolling(window, min_periods=1).mean()
    std = (
        data["standard_deviation"]
        .fillna(0.0)
        .rolling(window, min_periods=1)
        .mean()
    )

    plt.figure(figsize=(10, 5))
    plt.plot(data["iter"], mean, label="Promedio entre ejecuciones")
    plt.fill_between(
        data["iter"],
        mean - std,
        mean + std,
        alpha=0.2,
        label="± desviación estándar",
    )
    plt.title(title)
    plt.xlabel("Iteración")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"{metric}.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[+] Guardada: {output_path}")


def plot_final_reasons(
    final_summary: pd.DataFrame,
    role: str,
    output_dir: Path,
    show: bool,
) -> None:
    metrics = [
        f"{role}_reason_timeout_pct",
        f"{role}_reason_starvation_pct",
        f"{role}_reason_dehydration_pct",
        f"{role}_reason_predation_pct",
    ]
    labels = {
        f"{role}_reason_timeout_pct": "Timeout",
        f"{role}_reason_starvation_pct": "Hambre",
        f"{role}_reason_dehydration_pct": "Deshidratación",
        f"{role}_reason_predation_pct": "Predación",
    }

    data = final_summary[final_summary["metric"].isin(metrics)].copy()
    if data.empty:
        return

    data["mean"] = pd.to_numeric(data["mean"], errors="coerce")
    data["standard_deviation"] = pd.to_numeric(
        data["standard_deviation"], errors="coerce"
    ).fillna(0.0)
    data = data.dropna(subset=["mean"])

    ordered = []
    for metric in metrics:
        row = data[data["metric"] == metric]
        if not row.empty:
            ordered.append(row.iloc[0])
    if not ordered:
        return

    ordered_df = pd.DataFrame(ordered)
    x_labels = [labels[value] for value in ordered_df["metric"]]

    plt.figure(figsize=(8, 5))
    plt.bar(
        x_labels,
        ordered_df["mean"],
        yerr=ordered_df["standard_deviation"],
        capsize=5,
    )
    role_title = "Herbívoros" if role == "herbivore" else "Depredadores"
    plt.title(f"Razones finales de terminación - {role_title}")
    plt.xlabel("Razón")
    plt.ylabel("Porcentaje promedio")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f"{role}_termination_reasons.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[+] Guardada: {output_path}")


def print_final_summary(final_summary: pd.DataFrame) -> None:
    selected = final_summary[final_summary["metric"].isin(FINAL_METRICS)].copy()
    if selected.empty:
        print("[Aviso] No se encontraron métricas finales seleccionadas.")
        return

    for column in [
        "mean", "standard_deviation", "minimum", "maximum", "number_of_runs"
    ]:
        if column in selected.columns:
            selected[column] = pd.to_numeric(selected[column], errors="coerce")

    selected["order"] = selected["metric"].map(
        {metric: index for index, metric in enumerate(FINAL_METRICS)}
    )
    selected = selected.sort_values("order")
    columns = [
        column
        for column in [
            "metric", "mean", "standard_deviation", "minimum", "maximum", "number_of_runs"
        ]
        if column in selected.columns
    ]

    print("\nResumen final del experimento")
    print("=" * 90)
    print(
        selected[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analiza el promedio de múltiples entrenamientos."
    )
    parser.add_argument(
        "experiment_dir",
        help=(
            "Carpeta del experimento, por ejemplo "
            "experiment_results/experiment_20260726_011826"
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=20,
        help="Ventana de suavizado de las curvas (default: 20).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Además de guardar las gráficas, las muestra en pantalla.",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    output_dir = experiment_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = load_csv(
        experiment_dir / "learning_curve_summary.csv",
        {"iter", "metric", "mean", "standard_deviation"},
    )
    final_summary = load_csv(
        experiment_dir / "final_summary.csv",
        {"metric", "mean", "standard_deviation"},
    )

    print_final_summary(final_summary)

    for metric, (title, ylabel) in CURVE_METRICS.items():
        plot_learning_curve(
            curves=curves,
            metric=metric,
            title=title,
            ylabel=ylabel,
            output_dir=output_dir,
            smooth_window=args.smooth_window,
            show=args.show,
        )

    plot_final_reasons(final_summary, "herbivore", output_dir, args.show)
    plot_final_reasons(final_summary, "predator", output_dir, args.show)

    print(f"\n[+] Análisis completado. Gráficas en: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())