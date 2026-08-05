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

RUN_RE = re.compile(r"^run_(\d+)_seed_(\d+)$")
AGENT_RE = re.compile(r"^(r|l)_agent_(\d+)$")

CURVE_METRICS = {
    "r_mean": ("Retorno promedio entre ejecuciones", "Return"),
    "l_mean": ("Longitud promedio de episodio", "Timesteps"),
    "herbivore_episode_return": ("Return promedio de herbívoros", "Return"),
    "predator_episode_return": ("Return promedio de depredadores", "Return"),
    "herbivore_episode_len": ("Longitud promedio de herbívoros", "Timesteps"),
    "predator_episode_len": ("Longitud promedio de depredadores", "Timesteps"),
    "herbivore_survival_pct": ("Supervivencia promedio de herbívoros", "Porcentaje"),
    "predator_survival_pct": ("Supervivencia promedio de depredadores", "Porcentaje"),
    "herbivore_avg_food": ("Comida promedio de herbívoros", "Promedio normalizado"),
    "herbivore_avg_water": ("Agua promedio de herbívoros", "Promedio normalizado"),
    "predator_avg_food": ("Comida promedio de depredadores", "Promedio normalizado"),
    "predator_avg_water": ("Agua promedio de depredadores", "Promedio normalizado"),
    "herbivore_critical_ratio": ("Critical ratio de herbívoros", "Ratio"),
    "predator_critical_ratio": ("Critical ratio de depredadores", "Ratio"),
    "attacks_attempted": ("Intentos de ataque promedio", "Conteo"),
    "attacks_hit": ("Ataques acertados promedio", "Conteo"),
    "attacks_kill": ("Muertes por ataque promedio", "Conteo"),
    "attack_hit_rate": ("Tasa promedio de aciertos", "Rate"),
    "attack_kill_rate": ("Tasa promedio de ataques letales", "Rate"),
}

FINAL_METRICS = list(CURVE_METRICS) + [
    "herbivore_reason_timeout_pct",
    "herbivore_reason_starvation_pct",
    "herbivore_reason_dehydration_pct",
    "herbivore_reason_predation_pct",
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
        match = RUN_RE.fullmatch(run_dir.name)
        if not match:
            continue
        monitor = run_dir / "monitor.csv"
        if not monitor.exists():
            print(f"[Aviso] Se omite {run_dir.name}: no tiene monitor.csv")
            continue
        runs.append({
            "run": int(match.group(1)),
            "seed": int(match.group(2)),
            "folder": run_dir,
            "monitor": monitor,
        })
    return sorted(runs, key=lambda item: (item["run"], item["seed"]))


def summarize_monitor(path: Path, tail_iters: int) -> dict[str, float]:
    rows = read_monitor(path)
    if not rows:
        raise ValueError(f"El archivo está vacío: {path}")
    selected = rows[-max(1, tail_iters):]
    summary = {}
    for column in rows[0]:
        if column == "iter":
            continue
        values = [to_float(row.get(column)) for row in selected]
        values = [value for value in values if value is not None]
        if values:
            summary[column] = statistics.mean(values)
    return summary


def rebuild_summaries(experiment_dir: Path, tail_iters: int) -> int:
    discovered = discover_runs(experiment_dir)
    if not discovered:
        raise ValueError(
            "No se encontraron carpetas run_XX_seed_XXXX con monitor.csv."
        )

    completed = []
    for position, run in enumerate(discovered, start=1):
        print(f"[+] Procesando {run['folder'].name} ({position}/{len(discovered)})")
        completed.append({
            **run,
            "metrics": summarize_monitor(run["monitor"], tail_iters),
        })

    metric_names = sorted({
        metric for run in completed for metric in run["metrics"]
    })

    runs_path = experiment_dir / "runs_summary.csv"
    with runs_path.open("w", newline="", encoding="utf-8") as file:
        fields = ["run", "seed", *metric_names]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for run in completed:
            row = {"run": run["run"], "seed": run["seed"]}
            row.update(run["metrics"])
            writer.writerow(row)

    final_path = experiment_dir / "final_summary.csv"
    with final_path.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "metric", "mean", "standard_deviation",
            "minimum", "maximum", "number_of_runs",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for metric in metric_names:
            values = [
                run["metrics"][metric]
                for run in completed
                if metric in run["metrics"]
            ]
            writer.writerow({
                "metric": metric,
                "mean": statistics.mean(values),
                "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
                "minimum": min(values),
                "maximum": max(values),
                "number_of_runs": len(values),
            })

    grouped = defaultdict(list)
    for run in completed:
        for row in read_monitor(run["monitor"]):
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

    curves_path = experiment_dir / "learning_curve_summary.csv"
    with curves_path.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "iter", "metric", "mean", "standard_deviation",
            "minimum", "maximum", "number_of_runs",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for iteration, metric in sorted(grouped):
            values = grouped[(iteration, metric)]
            writer.writerow({
                "iter": iteration,
                "metric": metric,
                "mean": statistics.mean(values),
                "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
                "minimum": min(values),
                "maximum": max(values),
                "number_of_runs": len(values),
            })

    print(f"[+] Se consolidaron {len(completed)} corridas.")
    print(f"[+] Resumen por corrida: {runs_path}")
    print(f"[+] Resumen final: {final_path}")
    print(f"[+] Curvas consolidadas: {curves_path}")
    return len(completed)


def load_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró: {path}")
    df = pd.read_csv(path)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} no contiene: {', '.join(sorted(missing))}")
    return df


def prepare_curve(curves: pd.DataFrame, metric: str) -> pd.DataFrame:
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
    window = max(1, min(window, len(series)))
    return series.rolling(window, min_periods=1).mean()


def save_curve(
    curves: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    window: int,
    show: bool,
) -> bool:
    data = prepare_curve(curves, metric)
    if data.empty:
        print(f"[Aviso] No se encontró la métrica: {metric}")
        return False

    mean = smooth(data["mean"], window)
    std = smooth(data["standard_deviation"], window)

    plt.figure(figsize=(10, 5))
    plt.plot(data["iter"], mean, label="Promedio entre ejecuciones")
    plt.fill_between(
        data["iter"], mean - std, mean + std,
        alpha=0.2, label="± desviación estándar",
    )
    plt.title(title)
    plt.xlabel("Iteración")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[+] Guardada: {output_path}")
    return True


def discover_agents(curves: pd.DataFrame) -> list[int]:
    indices = set()
    for metric in curves["metric"].dropna().astype(str).unique():
        match = AGENT_RE.fullmatch(metric)
        if match:
            indices.add(int(match.group(2)))
    return sorted(indices)


def agent_label(index: int, n_predators: int) -> str:
    if index < n_predators:
        return f"Depredador {index + 1}"
    return f"Herbívoro {index - n_predators + 1}"

def plot_termination_reasons_over_time(
    curves: pd.DataFrame,
    role: str,
    output_dir: Path,
    smooth_window: int,
    show: bool,
) -> None:
    metrics = {
        f"{role}_reason_timeout_pct": "Timeout",
        f"{role}_reason_starvation_pct": "Hambre",
        f"{role}_reason_dehydration_pct": "Deshidratación",
        f"{role}_reason_predation_pct": "Predación",
    }

    plt.figure(figsize=(11, 6))
    plotted = False

    for metric, label in metrics.items():
        data = prepare_curve(curves, metric)

        if data.empty:
            continue

        mean = smooth(
            data["mean"],
            smooth_window,
        )

        plt.plot(
            data["iter"],
            mean,
            label=label,
        )

        plotted = True

    if not plotted:
        plt.close()
        print(
            f"[Aviso] No se encontraron razones "
            f"de terminación para {role}."
        )
        return

    role_title = (
        "Herbívoros"
        if role == "herbivore"
        else "Depredadores"
    )

    plt.title(
        f"Razones de terminación durante el entrenamiento - "
        f"{role_title}"
    )
    plt.xlabel("Iteración")
    plt.ylabel("Porcentaje")
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = (
        output_dir
        / f"{role}_termination_reasons_over_time.png"
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close()
    print(f"[+] Guardada: {output_path}")

def plot_individual_agents(
    curves: pd.DataFrame,
    agents: list[int],
    prefix: str,
    output_dir: Path,
    window: int,
    n_predators: int,
    show: bool,
) -> None:
    title_metric = "Return" if prefix == "r" else "Longitud de episodio"
    ylabel = "Return" if prefix == "r" else "Timesteps"

    for index in agents:
        metric = f"{prefix}_agent_{index}"
        data = prepare_curve(curves, metric)
        if data.empty:
            continue

        mean = smooth(data["mean"], window)
        std = smooth(data["standard_deviation"], window)
        label = agent_label(index, n_predators)

        plt.figure(figsize=(10, 5))
        plt.plot(data["iter"], mean, label=f"{label}: promedio")
        plt.fill_between(
            data["iter"], mean - std, mean + std,
            alpha=0.2, label="± desviación estándar entre corridas",
        )
        plt.title(f"{title_metric} - {label}")
        plt.xlabel("Iteración")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        path = output_dir / "agents" / "individual" / prefix / f"{metric}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=220, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        print(f"[+] Guardada: {path}")


def plot_agents_by_role(
    curves: pd.DataFrame,
    agents: list[int],
    prefix: str,
    output_dir: Path,
    window: int,
    n_predators: int,
    show: bool,
) -> None:
    groups = {
        "Depredadores": [index for index in agents if index < n_predators],
        "Herbívoros": [index for index in agents if index >= n_predators],
    }
    title_metric = "Return" if prefix == "r" else "Longitud de episodio"
    ylabel = "Return" if prefix == "r" else "Timesteps"

    for role, indices in groups.items():
        if not indices:
            continue

        plotted = False
        plt.figure(figsize=(11, 6))
        for index in indices:
            data = prepare_curve(curves, f"{prefix}_agent_{index}")
            if data.empty:
                continue
            plt.plot(
                data["iter"],
                smooth(data["mean"], window),
                label=agent_label(index, n_predators),
            )
            plotted = True

        if not plotted:
            plt.close()
            continue

        plt.title(f"{title_metric} por agente - {role}")
        plt.xlabel("Iteración")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        role_key = "predators" if role == "Depredadores" else "herbivores"
        path = output_dir / "agents" / f"{prefix}_agents_{role_key}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=220, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        print(f"[+] Guardada: {path}")


def plot_final_agents(
    summary: pd.DataFrame,
    agents: list[int],
    prefix: str,
    output_dir: Path,
    n_predators: int,
    show: bool,
) -> None:
    rows = []
    for index in agents:
        metric = f"{prefix}_agent_{index}"
        row = summary[summary["metric"] == metric]
        if row.empty:
            continue
        value = row.iloc[0]
        rows.append({
            "label": agent_label(index, n_predators),
            "mean": pd.to_numeric(value["mean"], errors="coerce"),
            "std": pd.to_numeric(value["standard_deviation"], errors="coerce"),
        })

    if not rows:
        return

    data = pd.DataFrame(rows).dropna(subset=["mean"])
    data["std"] = data["std"].fillna(0.0)
    title_metric = "Return" if prefix == "r" else "Longitud"
    ylabel = "Return" if prefix == "r" else "Timesteps"

    plt.figure(figsize=(11, 6))
    plt.bar(data["label"], data["mean"], yerr=data["std"], capsize=4)
    plt.title(f"{title_metric} final promedio por agente")
    plt.xlabel("Agente")
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "agents" / f"final_{prefix}_by_agent.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[+] Guardada: {path}")


def plot_final_reasons(
    summary: pd.DataFrame,
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
        metrics[0]: "Timeout",
        metrics[1]: "Hambre",
        metrics[2]: "Deshidratación",
        metrics[3]: "Predación",
    }

    rows = []
    for metric in metrics:
        row = summary[summary["metric"] == metric]
        if not row.empty:
            value = row.iloc[0]
            rows.append({
                "label": labels[metric],
                "mean": pd.to_numeric(value["mean"], errors="coerce"),
                "std": pd.to_numeric(value["standard_deviation"], errors="coerce"),
            })

    if not rows:
        return

    data = pd.DataFrame(rows).dropna(subset=["mean"])
    data["std"] = data["std"].fillna(0.0)
    role_title = "Herbívoros" if role == "herbivore" else "Depredadores"

    plt.figure(figsize=(8, 5))
    plt.bar(data["label"], data["mean"], yerr=data["std"], capsize=5)
    plt.title(f"Razones finales de terminación - {role_title}")
    plt.xlabel("Razón")
    plt.ylabel("Porcentaje promedio")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / f"{role}_termination_reasons.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[+] Guardada: {path}")


def print_final_summary(summary: pd.DataFrame) -> None:
    agent_metrics = sorted(
        metric for metric in summary["metric"].astype(str)
        if AGENT_RE.fullmatch(metric)
    )
    selected_metrics = FINAL_METRICS + agent_metrics
    selected = summary[summary["metric"].isin(selected_metrics)].copy()

    for column in [
        "mean", "standard_deviation", "minimum", "maximum", "number_of_runs"
    ]:
        if column in selected.columns:
            selected[column] = pd.to_numeric(selected[column], errors="coerce")

    order = {metric: index for index, metric in enumerate(selected_metrics)}
    selected["order"] = selected["metric"].map(order)
    selected = selected.sort_values("order")
    columns = [
        column for column in [
            "metric", "mean", "standard_deviation",
            "minimum", "maximum", "number_of_runs",
        ] if column in selected.columns
    ]

    print("\nResumen final del experimento")
    print("=" * 100)
    print(selected[columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruye y analiza todas las métricas de un experimento: "
            "globales, por rol y por agente."
        )
    )
    parser.add_argument("experiment_dir", help="Carpeta consolidada del experimento.")
    parser.add_argument(
        "--tail-iters", type=int, default=DEFAULT_TAIL_ITERS,
        help=f"Iteraciones finales usadas por corrida (default: {DEFAULT_TAIL_ITERS}).",
    )
    parser.add_argument("--smooth-window", type=int, default=20)
    parser.add_argument(
        "--n-predators", type=int, default=DEFAULT_N_PREDATORS,
        help=f"Cantidad de depredadores (default: {DEFAULT_N_PREDATORS}).",
    )
    parser.add_argument(
        "--no-rebuild", action="store_true",
        help="Usa los CSV existentes sin reconstruirlos desde los monitor.csv.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.tail_iters < 1:
        raise ValueError("--tail-iters debe ser mayor o igual que 1")
    if args.smooth_window < 1:
        raise ValueError("--smooth-window debe ser mayor o igual que 1")
    if args.n_predators < 0:
        raise ValueError("--n-predators no puede ser negativo")

    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"No existe: {experiment_dir}")

    if not args.no_rebuild:
        rebuild_summaries(experiment_dir, args.tail_iters)

    curves = load_csv(
        experiment_dir / "learning_curve_summary.csv",
        {"iter", "metric", "mean", "standard_deviation"},
    )
    summary = load_csv(
        experiment_dir / "final_summary.csv",
        {"metric", "mean", "standard_deviation"},
    )

    output_dir = experiment_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print_final_summary(summary)

    print("\n[+] Generando métricas globales y por rol...")
    for metric, (title, ylabel) in CURVE_METRICS.items():
        save_curve(
            curves, metric, title, ylabel,
            output_dir / f"{metric}.png",
            args.smooth_window, args.show,
        )

    print("\n[+] Generando razones de terminación...")
    plot_final_reasons(summary, "herbivore", output_dir, args.show)
    plot_final_reasons(summary, "predator", output_dir, args.show)
    plot_termination_reasons_over_time(
    curves=curves,
    role="herbivore",
    output_dir=output_dir,
    smooth_window=args.smooth_window,
    show=args.show,
)

    plot_termination_reasons_over_time(
        curves=curves,
        role="predator",
        output_dir=output_dir,
        smooth_window=args.smooth_window,
        show=args.show,
    )
    print("\n[+] Generando métricas por agente...")
    agents = discover_agents(curves)
    if agents:
        print("[+] Agentes detectados:", agents)
        plot_individual_agents(
            curves, agents, "r", output_dir,
            args.smooth_window, args.n_predators, args.show,
        )
        plot_individual_agents(
            curves, agents, "l", output_dir,
            args.smooth_window, args.n_predators, args.show,
        )
        plot_agents_by_role(
            curves, agents, "r", output_dir,
            args.smooth_window, args.n_predators, args.show,
        )
        plot_agents_by_role(
            curves, agents, "l", output_dir,
            args.smooth_window, args.n_predators, args.show,
        )
        plot_final_agents(
            summary, agents, "r", output_dir,
            args.n_predators, args.show,
        )
        plot_final_agents(
            summary, agents, "l", output_dir,
            args.n_predators, args.show,
        )
    else:
        print("[Aviso] No se encontraron métricas r_agent_X o l_agent_X.")

    print(f"\n[+] Análisis completado. Gráficas en: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())