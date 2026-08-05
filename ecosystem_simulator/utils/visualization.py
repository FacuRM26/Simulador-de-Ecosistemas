#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from ecosystem_simulator.config import ENV_CFG, TAIL_ITERS

    DEFAULT_N_PREDATORS = int(ENV_CFG.get("n_predators", 2))
    DEFAULT_TAIL_ITERS = int(TAIL_ITERS)
except Exception:
    DEFAULT_N_PREDATORS = 2
    DEFAULT_TAIL_ITERS = 1

RUN_PATTERN = re.compile(r"^run_(\d+)_seed_(\d+)$")
AGENT_PATTERN = re.compile(r"^(r|l)_agent_(\d+)$")

HERB_REASON_METRICS = [
    "herbivore_reason_timeout_pct",
    "herbivore_reason_starvation_pct",
    "herbivore_reason_dehydration_pct",
    "herbivore_reason_predation_pct",
]
PRED_REASON_METRICS = [
    "predator_reason_timeout_pct",
    "predator_reason_starvation_pct",
    "predator_reason_dehydration_pct",
    "predator_reason_predation_pct",
]


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_monitor(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def discover_runs(experiment_dir: Path) -> list[dict]:
    runs = []

    for run_dir in experiment_dir.iterdir():
        if not run_dir.is_dir():
            continue

        match = RUN_PATTERN.fullmatch(run_dir.name)
        if not match:
            continue

        monitor_path = run_dir / "monitor.csv"
        if not monitor_path.exists():
            print(f"[Aviso] Se omite {run_dir.name}: no contiene monitor.csv")
            continue

        runs.append(
            {
                "folder": run_dir,
                "run": int(match.group(1)),
                "seed": int(match.group(2)),
                "monitor_path": monitor_path,
            }
        )

    runs.sort(key=lambda item: (item["run"], item["seed"]))
    return runs


def summarize_monitor(monitor_path: Path, tail_iters: int) -> dict[str, float]:
    rows = read_monitor(monitor_path)
    if not rows:
        raise ValueError(f"El archivo está vacío: {monitor_path}")

    selected = rows[-max(1, tail_iters):]
    summary: dict[str, float] = {}

    for column in rows[0]:
        if column == "iter":
            continue

        values = [
            number
            for row in selected
            if (number := to_float(row.get(column))) is not None
        ]

        if values:
            summary[column] = statistics.mean(values)

    return summary


def write_runs_summary(experiment_dir: Path, runs: list[dict]) -> None:
    metrics = sorted(
        {metric for run in runs for metric in run["metrics"]}
    )
    path = experiment_dir / "runs_summary.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        fields = ["run", "seed", *metrics]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for run in runs:
            row = {"run": run["run"], "seed": run["seed"]}
            row.update(run["metrics"])
            writer.writerow(row)

    print(f"[+] Reconstruido: {path}")


def write_final_summary(experiment_dir: Path, runs: list[dict]) -> None:
    metrics = sorted(
        {metric for run in runs for metric in run["metrics"]}
    )
    path = experiment_dir / "final_summary.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "metric",
            "mean",
            "standard_deviation",
            "minimum",
            "maximum",
            "number_of_runs",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for metric in metrics:
            values = [
                run["metrics"][metric]
                for run in runs
                if metric in run["metrics"]
            ]
            if not values:
                continue

            writer.writerow(
                {
                    "metric": metric,
                    "mean": statistics.mean(values),
                    "standard_deviation": (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    ),
                    "minimum": min(values),
                    "maximum": max(values),
                    "number_of_runs": len(values),
                }
            )

    print(f"[+] Reconstruido: {path}")


def write_learning_curve_summary(experiment_dir: Path, runs: list[dict]) -> None:
    grouped = defaultdict(list)

    for run in runs:
        for row in read_monitor(run["monitor_path"]):
            iteration_value = to_float(row.get("iter"))
            if iteration_value is None:
                continue

            iteration = int(iteration_value)

            for metric, raw_value in row.items():
                if metric == "iter":
                    continue

                value = to_float(raw_value)
                if value is not None:
                    grouped[(iteration, metric)].append(value)

    path = experiment_dir / "learning_curve_summary.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "iter",
            "metric",
            "mean",
            "standard_deviation",
            "minimum",
            "maximum",
            "number_of_runs",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for iteration, metric in sorted(grouped):
            values = grouped[(iteration, metric)]
            writer.writerow(
                {
                    "iter": iteration,
                    "metric": metric,
                    "mean": statistics.mean(values),
                    "standard_deviation": (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    ),
                    "minimum": min(values),
                    "maximum": max(values),
                    "number_of_runs": len(values),
                }
            )

    print(f"[+] Reconstruido: {path}")


def rebuild_summaries(experiment_dir: Path, tail_iters: int) -> int:
    discovered = discover_runs(experiment_dir)
    if not discovered:
        raise ValueError(
            "No se encontraron carpetas run_XX_seed_XXXX con monitor.csv."
        )

    completed = []

    for position, run in enumerate(discovered, start=1):
        print(
            f"[+] Procesando {run['folder'].name} "
            f"({position}/{len(discovered)})"
        )
        completed.append(
            {
                **run,
                "metrics": summarize_monitor(
                    run["monitor_path"], tail_iters
                ),
            }
        )

    write_runs_summary(experiment_dir, completed)
    write_final_summary(experiment_dir, completed)
    write_learning_curve_summary(experiment_dir, completed)

    print(
        f"[+] Se consolidaron {len(completed)} corridas "
        f"con tail_iters={tail_iters}."
    )
    return len(completed)


def load_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró: {path}")

    df = pd.read_csv(path)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} no contiene las columnas: "
            + ", ".join(sorted(missing))
        )
    return df


def curve_data(curves: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = curves[curves["metric"] == metric].copy()
    if data.empty:
        return data

    data["iter"] = pd.to_numeric(data["iter"], errors="coerce")
    data["mean"] = pd.to_numeric(data["mean"], errors="coerce")
    data["standard_deviation"] = pd.to_numeric(
        data["standard_deviation"], errors="coerce"
    ).fillna(0.0)

    return data.dropna(subset=["iter", "mean"]).sort_values("iter")


def smooth(series: pd.Series, window: int) -> pd.Series:
    effective = max(1, min(window, len(series)))
    return series.rolling(effective, min_periods=1).mean()


def finish_plot(path: Path, show: bool) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[+] Guardada: {path}")


def plot_global_dashboard(
    curves: pd.DataFrame,
    output_dir: Path,
    window: int,
    show: bool,
) -> None:
    specs = [
        ("r_mean", "Recompensa media", "Return"),
        ("l_mean", "Longitud media", "Timesteps"),
        ("herbivore_survival_pct", "Supervivencia herbívoros", "%"),
        ("predator_survival_pct", "Supervivencia depredadores", "%"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for axis, (metric, title, ylabel) in zip(axes.ravel(), specs):
        data = curve_data(curves, metric)
        if data.empty:
            axis.set_visible(False)
            continue

        mean = smooth(data["mean"], window)
        std = smooth(data["standard_deviation"], window)

        axis.plot(data["iter"], mean, label="Promedio")
        axis.fill_between(
            data["iter"],
            mean - std,
            mean + std,
            alpha=0.18,
            label="± desviación estándar",
        )
        axis.set_title(title)
        axis.set_xlabel("Iteración")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend()

    finish_plot(output_dir / "01_global_dashboard.png", show)


def discover_agent_indices(curves: pd.DataFrame) -> list[int]:
    indices = set()
    for metric in curves["metric"].dropna().astype(str).unique():
        match = AGENT_PATTERN.fullmatch(metric)
        if match:
            indices.add(int(match.group(2)))
    return sorted(indices)


def agent_label(index: int, n_predators: int) -> str:
    if index < n_predators:
        return f"P{index}"
    return f"H{index - n_predators}"


def plot_agents_all_lines(
    curves: pd.DataFrame,
    agent_indices: list[int],
    prefix: str,
    title: str,
    ylabel: str,
    output_path: Path,
    window: int,
    n_predators: int,
    show: bool,
    show_std: bool,
) -> None:
    plt.figure(figsize=(12, 6))
    plotted = False

    for index in agent_indices:
        metric = f"{prefix}_agent_{index}"
        data = curve_data(curves, metric)
        if data.empty:
            continue

        mean = smooth(data["mean"], window)
        std = smooth(data["standard_deviation"], window)
        is_predator = index < n_predators

        plt.plot(
            data["iter"],
            mean,
            label=agent_label(index, n_predators),
            linestyle="-" if is_predator else "--",
        )

        if show_std:
            plt.fill_between(
                data["iter"],
                mean - std,
                mean + std,
                alpha=0.06,
            )

        plotted = True

    if not plotted:
        plt.close()
        print(f"[Aviso] No hay métricas {prefix}_agent_X")
        return

    plt.title(title)
    plt.xlabel("Iteración")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    finish_plot(output_path, show)


def plot_agents_by_role(
    curves: pd.DataFrame,
    agent_indices: list[int],
    prefix: str,
    ylabel: str,
    output_dir: Path,
    window: int,
    n_predators: int,
    show: bool,
) -> None:
    groups = [
        (
            "Depredadores",
            [i for i in agent_indices if i < n_predators],
            "predators",
        ),
        (
            "Herbívoros",
            [i for i in agent_indices if i >= n_predators],
            "herbivores",
        ),
    ]

    for role_title, indices, file_key in groups:
        if not indices:
            continue

        plt.figure(figsize=(12, 6))
        plotted = False

        for index in indices:
            metric = f"{prefix}_agent_{index}"
            data = curve_data(curves, metric)
            if data.empty:
                continue

            mean = smooth(data["mean"], window)
            plt.plot(
                data["iter"],
                mean,
                label=agent_label(index, n_predators),
            )
            plotted = True

        if not plotted:
            plt.close()
            continue

        metric_title = "Return" if prefix == "r" else "Longitud"
        plt.title(f"{metric_title} por agente - {role_title}")
        plt.xlabel("Iteración")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()

        finish_plot(
            output_dir / f"{prefix}_agents_{file_key}.png",
            show,
        )


def plot_role_dashboard(
    curves: pd.DataFrame,
    output_dir: Path,
    window: int,
    show: bool,
) -> None:
    specs = [
        (
            "Return por rol",
            "Return",
            [
                ("herbivore_episode_return", "Herbívoros"),
                ("predator_episode_return", "Depredadores"),
            ],
        ),
        (
            "Longitud por rol",
            "Timesteps",
            [
                ("herbivore_episode_len", "Herbívoros"),
                ("predator_episode_len", "Depredadores"),
            ],
        ),
        (
            "Supervivencia por rol",
            "%",
            [
                ("herbivore_survival_pct", "Herbívoros"),
                ("predator_survival_pct", "Depredadores"),
            ],
        ),
        (
            "Critical ratio por rol",
            "Ratio",
            [
                ("herbivore_critical_ratio", "Herbívoros"),
                ("predator_critical_ratio", "Depredadores"),
            ],
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for axis, (title, ylabel, metrics) in zip(axes.ravel(), specs):
        plotted = False
        for metric, label in metrics:
            data = curve_data(curves, metric)
            if data.empty:
                continue
            axis.plot(
                data["iter"],
                smooth(data["mean"], window),
                label=label,
            )
            plotted = True

        if not plotted:
            axis.set_visible(False)
            continue

        axis.set_title(title)
        axis.set_xlabel("Iteración")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend()

    finish_plot(output_dir / "04_metrics_by_role.png", show)


def plot_termination_reasons(
    curves: pd.DataFrame,
    output_dir: Path,
    window: int,
    show: bool,
) -> None:
    labels = {
        "timeout": "timeout %",
        "starvation": "starvation %",
        "dehydration": "dehydration %",
        "predation": "predation %",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    role_specs = [
        (axes[0], "Herbívoros", "herbivore", HERB_REASON_METRICS),
        (axes[1], "Depredadores", "predator", PRED_REASON_METRICS),
    ]

    for axis, role_title, role_key, metrics in role_specs:
        plotted = False

        for metric in metrics:
            data = curve_data(curves, metric)
            if data.empty:
                continue

            reason = (
                metric.replace(f"{role_key}_reason_", "")
                .replace("_pct", "")
            )
            axis.plot(
                data["iter"],
                smooth(data["mean"], window),
                label=labels.get(reason, reason),
            )
            plotted = True

        if not plotted:
            axis.set_visible(False)
            continue

        axis.set_title(f"Razones de terminación - {role_title}")
        axis.set_xlabel("Iteración")
        axis.set_ylabel("Porcentaje")
        axis.grid(alpha=0.3)
        axis.legend()

    finish_plot(output_dir / "05_termination_reasons_by_role.png", show)


def plot_resources_by_role(
    curves: pd.DataFrame,
    output_dir: Path,
    window: int,
    show: bool,
) -> None:
    metrics = [
        ("herbivore_avg_food", "Herbívoros - comida"),
        ("herbivore_avg_water", "Herbívoros - agua"),
        ("predator_avg_food", "Depredadores - comida"),
        ("predator_avg_water", "Depredadores - agua"),
    ]

    plt.figure(figsize=(12, 5))
    plotted = False

    for metric, label in metrics:
        data = curve_data(curves, metric)
        if data.empty:
            continue

        plt.plot(
            data["iter"],
            smooth(data["mean"], window),
            label=label,
        )
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.title("Promedio de recursos por rol")
    plt.xlabel("Iteración")
    plt.ylabel("Promedio normalizado")
    plt.grid(alpha=0.3)
    plt.legend()
    finish_plot(output_dir / "06_resources_by_role.png", show)


def plot_attacks(
    curves: pd.DataFrame,
    output_dir: Path,
    window: int,
    show: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    count_metrics = [
        ("attacks_attempted", "Intentos"),
        ("attacks_hit", "Aciertos"),
        ("attacks_kill", "Muertes"),
    ]
    rate_metrics = [
        ("attack_hit_rate", "Hit rate"),
        ("attack_kill_rate", "Kill rate"),
    ]

    for metric, label in count_metrics:
        data = curve_data(curves, metric)
        if not data.empty:
            axes[0].plot(
                data["iter"],
                smooth(data["mean"], window),
                label=label,
            )

    for metric, label in rate_metrics:
        data = curve_data(curves, metric)
        if not data.empty:
            axes[1].plot(
                data["iter"],
                smooth(data["mean"], window),
                label=label,
            )

    axes[0].set_title("Ataques por iteración")
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel("Conteo")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Tasas de ataque")
    axes[1].set_xlabel("Iteración")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    finish_plot(output_dir / "07_attack_metrics.png", show)


def plot_final_agents(
    final_summary: pd.DataFrame,
    agent_indices: list[int],
    prefix: str,
    title: str,
    ylabel: str,
    output_path: Path,
    n_predators: int,
    show: bool,
) -> None:
    rows = []

    for index in agent_indices:
        metric = f"{prefix}_agent_{index}"
        row = final_summary[final_summary["metric"] == metric]
        if row.empty:
            continue

        record = row.iloc[0]
        rows.append(
            {
                "label": agent_label(index, n_predators),
                "mean": pd.to_numeric(record["mean"], errors="coerce"),
                "std": pd.to_numeric(
                    record["standard_deviation"], errors="coerce"
                ),
            }
        )

    if not rows:
        return

    data = pd.DataFrame(rows).dropna(subset=["mean"])
    data["std"] = data["std"].fillna(0.0)

    plt.figure(figsize=(11, 6))
    plt.bar(
        data["label"],
        data["mean"],
        yerr=data["std"],
        capsize=4,
    )
    plt.title(title)
    plt.xlabel("Agente")
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    finish_plot(output_path, show)


def print_final_summary(final_summary: pd.DataFrame) -> None:
    selected = final_summary.copy()

    for column in [
        "mean",
        "standard_deviation",
        "minimum",
        "maximum",
        "number_of_runs",
    ]:
        if column in selected.columns:
            selected[column] = pd.to_numeric(
                selected[column], errors="coerce"
            )

    columns = [
        column
        for column in [
            "metric",
            "mean",
            "standard_deviation",
            "minimum",
            "maximum",
            "number_of_runs",
        ]
        if column in selected.columns
    ]

    print("\nResumen final del experimento")
    print("=" * 100)
    print(
        selected[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruye y analiza un experimento completo con "
            "gráficas globales, por rol y por agente."
        )
    )
    parser.add_argument(
        "experiment_dir",
        help="Carpeta que contiene las corridas run_XX_seed_XXXX.",
    )
    parser.add_argument(
        "--tail-iters",
        type=int,
        default=DEFAULT_TAIL_ITERS,
        help=(
            "Iteraciones finales usadas para resumir cada corrida "
            f"(default: {DEFAULT_TAIL_ITERS})."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=20,
        help="Ventana de media móvil (default: 20).",
    )
    parser.add_argument(
        "--n-predators",
        type=int,
        default=DEFAULT_N_PREDATORS,
        help=f"Cantidad de depredadores (default: {DEFAULT_N_PREDATORS}).",
    )
    parser.add_argument(
        "--agent-std",
        action="store_true",
        help=(
            "Dibuja también las bandas de desviación estándar en la "
            "gráfica que contiene los ocho agentes."
        ),
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Usa los CSV consolidados existentes sin reconstruirlos.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Muestra las figuras además de guardarlas.",
    )
    args = parser.parse_args()

    if args.tail_iters < 1 or args.smooth_window < 1:
        print("[!] tail-iters y smooth-window deben ser mayores o iguales a 1.")
        return 1

    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.exists():
        print(f"[!] No existe la carpeta: {experiment_dir}")
        return 1

    try:
        if not args.no_rebuild:
            rebuild_summaries(experiment_dir, args.tail_iters)

        curves = load_csv(
            experiment_dir / "learning_curve_summary.csv",
            {"iter", "metric", "mean", "standard_deviation"},
        )
        final_summary = load_csv(
            experiment_dir / "final_summary.csv",
            {"metric", "mean", "standard_deviation"},
        )

        output_dir = experiment_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        agent_dir = output_dir / "agents"
        agent_dir.mkdir(parents=True, exist_ok=True)

        print_final_summary(final_summary)

        plot_global_dashboard(
            curves, output_dir, args.smooth_window, args.show
        )

        agent_indices = discover_agent_indices(curves)
        print(
            "[+] Agentes detectados:",
            ", ".join(str(index) for index in agent_indices) or "ninguno",
        )

        plot_agents_all_lines(
            curves,
            agent_indices,
            "r",
            "Return por agente (promedio entre corridas)",
            "Return",
            agent_dir / "02_return_by_agent.png",
            args.smooth_window,
            args.n_predators,
            args.show,
            args.agent_std,
        )
        plot_agents_all_lines(
            curves,
            agent_indices,
            "l",
            "Longitud por agente (promedio entre corridas)",
            "Timesteps",
            agent_dir / "03_length_by_agent.png",
            args.smooth_window,
            args.n_predators,
            args.show,
            args.agent_std,
        )

        plot_agents_by_role(
            curves,
            agent_indices,
            "r",
            "Return",
            agent_dir,
            args.smooth_window,
            args.n_predators,
            args.show,
        )
        plot_agents_by_role(
            curves,
            agent_indices,
            "l",
            "Timesteps",
            agent_dir,
            args.smooth_window,
            args.n_predators,
            args.show,
        )

        plot_role_dashboard(
            curves, output_dir, args.smooth_window, args.show
        )
        plot_termination_reasons(
            curves, output_dir, args.smooth_window, args.show
        )
        plot_resources_by_role(
            curves, output_dir, args.smooth_window, args.show
        )
        plot_attacks(
            curves, output_dir, args.smooth_window, args.show
        )

        plot_final_agents(
            final_summary,
            agent_indices,
            "r",
            "Return final promedio por agente",
            "Return",
            agent_dir / "08_final_return_by_agent.png",
            args.n_predators,
            args.show,
        )
        plot_final_agents(
            final_summary,
            agent_indices,
            "l",
            "Longitud final promedio por agente",
            "Timesteps",
            agent_dir / "09_final_length_by_agent.png",
            args.n_predators,
            args.show,
        )

        print(
            "\n[+] Análisis completado."
            f"\n[+] Gráficas: {output_dir}"
            f"\n[+] Corridas encontradas: {len(discover_runs(experiment_dir))}"
        )
        return 0

    except Exception as error:
        print(f"[!] Error durante el análisis: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())