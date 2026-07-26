#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


RUN_PATTERN = re.compile(r"run_(\d+)_seed_(\d+)$")


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_monitor(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize_monitor(
    monitor_path: Path,
    tail_iters: int,
) -> dict[str, float]:
    rows = read_monitor(monitor_path)

    if not rows:
        raise ValueError(f"El archivo está vacío: {monitor_path}")

    selected = rows[-max(1, tail_iters):]
    summary = {}

    for column in rows[0]:
        if column == "iter":
            continue

        values = []

        for row in selected:
            number = to_float(row.get(column))

            if number is not None:
                values.append(number)

        if values:
            summary[column] = statistics.mean(values)

    return summary


def discover_runs(experiment_dir: Path) -> list[dict]:
    discovered = []

    for run_dir in experiment_dir.iterdir():
        if not run_dir.is_dir():
            continue

        match = RUN_PATTERN.fullmatch(run_dir.name)

        if not match:
            continue

        monitor_path = run_dir / "monitor.csv"

        if not monitor_path.exists():
            print(f"[Aviso] Se omite {run_dir.name}: no tiene monitor.csv")
            continue

        discovered.append({
            "folder": run_dir,
            "run": int(match.group(1)),
            "seed": int(match.group(2)),
            "monitor_path": monitor_path,
        })

    discovered.sort(key=lambda item: (item["run"], item["seed"]))

    return discovered


def write_runs_summary(
    experiment_dir: Path,
    runs: list[dict],
) -> None:
    metrics = sorted({
        metric
        for run in runs
        for metric in run["metrics"]
    })

    path = experiment_dir / "runs_summary.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        fields = ["run", "seed", *metrics]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for run in runs:
            row = {
                "run": run["run"],
                "seed": run["seed"],
            }
            row.update(run["metrics"])
            writer.writerow(row)

    print(f"[+] Creado: {path}")


def write_final_summary(
    experiment_dir: Path,
    runs: list[dict],
) -> None:
    metrics = sorted({
        metric
        for run in runs
        for metric in run["metrics"]
    })

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

            writer.writerow({
                "metric": metric,
                "mean": statistics.mean(values),
                "standard_deviation": (
                    statistics.stdev(values)
                    if len(values) > 1
                    else 0.0
                ),
                "minimum": min(values),
                "maximum": max(values),
                "number_of_runs": len(values),
            })

    print(f"[+] Creado: {path}")


def write_learning_curve_summary(
    experiment_dir: Path,
    runs: list[dict],
) -> None:
    grouped = defaultdict(list)

    for run in runs:
        rows = read_monitor(run["monitor_path"])

        for row in rows:
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

            writer.writerow({
                "iter": iteration,
                "metric": metric,
                "mean": statistics.mean(values),
                "standard_deviation": (
                    statistics.stdev(values)
                    if len(values) > 1
                    else 0.0
                ),
                "minimum": min(values),
                "maximum": max(values),
                "number_of_runs": len(values),
            })

    print(f"[+] Creado: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruye los resúmenes de una carpeta que contiene "
            "varias corridas run_XX_seed_XXXX."
        )
    )

    parser.add_argument(
        "experiment_dir",
        help="Carpeta consolidada del experimento.",
    )

    parser.add_argument(
        "--tail-iters",
        type=int,
        default=1,
        help=(
            "Cantidad de iteraciones finales que se promedian "
            "dentro de cada corrida (default: 1)."
        ),
    )

    args = parser.parse_args()

    if args.tail_iters < 1:
        print("[!] --tail-iters debe ser mayor o igual que 1.")
        return 1

    experiment_dir = Path(args.experiment_dir).resolve()

    if not experiment_dir.exists():
        print(f"[!] No existe la carpeta: {experiment_dir}")
        return 1

    discovered = discover_runs(experiment_dir)

    if not discovered:
        print(
            "[!] No se encontraron carpetas con formato "
            "run_XX_seed_XXXX que contengan monitor.csv."
        )
        return 1

    completed = []

    for position, run in enumerate(discovered, start=1):
        print(
            f"[+] Procesando {run['folder'].name} "
            f"({position}/{len(discovered)})"
        )

        metrics = summarize_monitor(
            monitor_path=run["monitor_path"],
            tail_iters=args.tail_iters,
        )

        completed.append({
            **run,
            "metrics": metrics,
        })

    write_runs_summary(experiment_dir, completed)
    write_final_summary(experiment_dir, completed)
    write_learning_curve_summary(experiment_dir, completed)

    print(
        f"\n[+] Se consolidaron {len(completed)} corridas."
        f"\n[+] Carpeta: {experiment_dir}"
        f"\n[+] Tail iterations: {args.tail_iters}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())